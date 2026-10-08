"""
2-D EKF-SLAM (range-bearing landmarks) with gated nearest-neighbour data association + log-odds occupancy grid.

Landmarks come from the perception stack (detected objects / fiducials) -- identity is NOT assumed known: association is
done with a Mahalanobis (chi-square) gate, new landmarks are initialised when no existing one is inside the gate.
Odometry model: rot1 / trans / rot2 (what leg-odometry from the state estimator provides).
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class EKFSLAM:
    def __init__(self, pose0=(0, 0, 0), a=(0.02, 0.02, 0.04, 0.02), sig_r=0.05, sig_b=np.radians(1.5),
                 gate=9.21, new_gate=25.0):
        self.mu = np.array(pose0, float)
        self.S = np.diag([1e-6, 1e-6, 1e-8])
        self.a = a          # alpha1..4 (Thrun odometry noise)
        self.Q = np.diag([sig_r ** 2, sig_b ** 2])
        self.gate, self.new_gate = gate, new_gate
        self.n_lm = 0

    def predict(self, d_rot1, d_trans, d_rot2):
        a1, a2, a3, a4 = self.a
        th = self.mu[2]
        self.mu[0] += d_trans * np.cos(th + d_rot1)
        self.mu[1] += d_trans * np.sin(th + d_rot1)
        self.mu[2] = wrap(th + d_rot1 + d_rot2)
        n = len(self.mu)
        G = np.eye(n)
        G[0, 2] = -d_trans * np.sin(th + d_rot1); G[1, 2] = d_trans * np.cos(th + d_rot1)
        V = np.array([[-d_trans * np.sin(th + d_rot1), np.cos(th + d_rot1), 0],
                      [d_trans * np.cos(th + d_rot1), np.sin(th + d_rot1), 0], [1, 0, 1]])
        M = np.diag([a1 * d_rot1 ** 2 + a2 * d_trans ** 2, a3 * d_trans ** 2 + a4 * (d_rot1 ** 2 + d_rot2 ** 2),
                     a1 * d_rot2 ** 2 + a2 * d_trans ** 2]) + 1e-8 * np.eye(3)
        R = np.zeros((n, n)); R[:3, :3] = V @ M @ V.T
        self.S = G @ self.S @ G.T + R

    def _h(self, j):
        lx, ly = self.mu[3 + 2 * j], self.mu[4 + 2 * j]
        dx, dy = lx - self.mu[0], ly - self.mu[1]
        q = dx * dx + dy * dy
        z = np.array([np.sqrt(q), wrap(np.arctan2(dy, dx) - self.mu[2])])
        n = len(self.mu)
        H = np.zeros((2, n))
        sq = np.sqrt(q)
        H[:, :3] = [[-dx / sq, -dy / sq, 0], [dy / q, -dx / q, -1]]
        H[:, 3 + 2 * j:5 + 2 * j] = [[dx / sq, dy / sq], [-dy / q, dx / q]]
        return z, H

    def _add_landmark(self, r, b):
        th = self.mu[2]
        lx = self.mu[0] + r * np.cos(th + b); ly = self.mu[1] + r * np.sin(th + b)
        n = len(self.mu)
        self.mu = np.concatenate([self.mu, [lx, ly]])
        Gx = np.array([[1, 0, -r * np.sin(th + b)], [0, 1, r * np.cos(th + b)]])
        Gz = np.array([[np.cos(th + b), -r * np.sin(th + b)], [np.sin(th + b), r * np.cos(th + b)]])
        S = np.zeros((n + 2, n + 2)); S[:n, :n] = self.S
        S[n:, n:] = Gx @ self.S[:3, :3] @ Gx.T + Gz @ self.Q @ Gz.T
        S[n:, :n] = Gx @ self.S[:3, :n]; S[:n, n:] = S[n:, :n].T
        self.S = S; self.n_lm += 1

    def update(self, observations):
        """observations: list of (range, bearing) without identity. Returns number matched."""
        matched = 0
        for (r, b) in observations:
            z = np.array([r, b])
            best, bj, bH, bnu, bSi = np.inf, None, None, None, None
            for j in range(self.n_lm):
                zh, H = self._h(j)
                nu = z - zh; nu[1] = wrap(nu[1])
                Sj = H @ self.S @ H.T + self.Q
                m2 = float(nu @ np.linalg.solve(Sj, nu))
                if m2 < best:
                    best, bj, bH, bnu, bSi = m2, j, H, nu, Sj
            if bj is not None and best < self.gate:
                K = self.S @ bH.T @ np.linalg.inv(bSi)
                self.mu = self.mu + K @ bnu
                self.mu[2] = wrap(self.mu[2])
                I = np.eye(len(self.mu))
                self.S = (I - K @ bH) @ self.S @ (I - K @ bH).T + K @ self.Q @ K.T
                matched += 1
            elif best > self.new_gate or bj is None:
                self._add_landmark(r, b)        # clearly a new landmark; ambiguous cases are skipped
        return matched


class OccupancyGrid:
    def __init__(self, size=(12.0, 12.0), res=0.05, origin=(-6.0, -6.0), l_occ=0.85, l_free=-0.4, l_clip=5.0):
        self.res, self.origin = res, np.array(origin)
        self.shape = (int(size[1] / res), int(size[0] / res))
        self.L = np.zeros(self.shape)
        self.l_occ, self.l_free, self.l_clip = l_occ, l_free, l_clip

    def _ij(self, x, y):
        return int((y - self.origin[1]) / self.res), int((x - self.origin[0]) / self.res)

    def integrate_scan(self, pose, angles, ranges, max_range):
        x, y, th = pose
        i0, j0 = self._ij(x, y)
        for a, r in zip(angles, ranges):
            hit = r < max_range - 1e-6
            rr = min(r, max_range)
            ex, ey = x + rr * np.cos(th + a), y + rr * np.sin(th + a)
            i1, j1 = self._ij(ex, ey)
            n = max(abs(i1 - i0), abs(j1 - j0), 1)
            ii = np.linspace(i0, i1, n + 1).astype(int); jj = np.linspace(j0, j1, n + 1).astype(int)
            ok = (ii >= 0) & (ii < self.shape[0]) & (jj >= 0) & (jj < self.shape[1])
            ii, jj = ii[ok], jj[ok]
            if len(ii) == 0:
                continue
            free = slice(0, len(ii) - 1) if hit else slice(0, len(ii))
            self.L[ii[free], jj[free]] = np.clip(self.L[ii[free], jj[free]] + self.l_free, -self.l_clip, self.l_clip)
            if hit and ok[-1]:
                self.L[ii[-1], jj[-1]] = np.clip(self.L[ii[-1], jj[-1]] + self.l_occ, -self.l_clip, self.l_clip)

    def occupied(self, thr=0.5):
        return self.L > thr
