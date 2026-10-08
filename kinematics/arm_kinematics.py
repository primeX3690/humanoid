"""
7-DOF humanoid arm kinematics in pure NumPy (independent of MuJoCo; verified against it in tests).

Chain (torso frame -> TCP):  shoulder pitch(y) -> roll(x) -> yaw(z) -> upper arm -> elbow(y) -> forearm
                              -> wrist yaw(z) -> pitch(y) -> roll(x) -> palm -> TCP (between the jaw tips)
Provides: FK, geometric Jacobian, damped-least-squares IK with null-space joint-limit avoidance,
manipulability, and a reach-workspace sampler.
"""
from dataclasses import dataclass
import numpy as np


def _rot(axis, q):
    c, s = np.cos(q), np.sin(q)
    if axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return R


def _T(R=None, p=None):
    T = np.eye(4)
    if R is not None:
        T[:3, :3] = R
    if p is not None:
        T[:3, 3] = p
    return T


@dataclass
class ArmParams:
    side: str = "R"                       # "L" or "R"
    torso_h: float = 0.42
    shoulder_half_width: float = 0.17
    upper_arm: float = 0.28
    forearm: float = 0.26
    tcp_offset: float = 0.035 + 0.09 * 0.55   # palm origin -> TCP along hand -z
    q_lo: tuple = (-3.0, None, -1.6, -2.5, -2.0, -1.5, -1.5)
    q_hi: tuple = (1.5, None, 1.6, 0.0, 2.0, 1.5, 1.5)

    def limits(self):
        sgn = 1.0 if self.side == "L" else -1.0
        lo = list(self.q_lo); hi = list(self.q_hi)
        lo[1], hi[1] = (-0.3, 2.0) if sgn > 0 else (-2.0, 0.3)
        return np.array(lo, float), np.array(hi, float)


AXES = ["y", "x", "z", "y", "z", "y", "x"]


class Arm:
    def __init__(self, params: ArmParams = ArmParams()):
        self.p = params
        self.sgn = 1.0 if params.side == "L" else -1.0
        self.q_lo, self.q_hi = params.limits()

    def _chain(self, q):
        """Returns list of (T_i_before_joint, axis_world) for each joint and final TCP transform (torso frame)."""
        p = self.p
        T = _T(p=[0, self.sgn * p.shoulder_half_width, p.torso_h - 0.07])
        frames = []
        offsets = [None, None, None, [0, 0, -p.upper_arm], [0, 0, -p.forearm], None, None]
        for i in range(7):
            if offsets[i] is not None:
                T = T @ _T(p=offsets[i])
            axis_w = T[:3, :3] @ np.eye(3)[{"x": 0, "y": 1, "z": 2}[AXES[i]]]
            frames.append((T[:3, 3].copy(), axis_w))
            T = T @ _T(R=_rot(AXES[i], q[i]))
        T = T @ _T(p=[0, 0, -p.tcp_offset])
        return frames, T

    def fk(self, q):
        """4x4 pose of the TCP in the torso frame."""
        return self._chain(np.asarray(q, float))[1]

    def jacobian(self, q):
        """6x7 geometric Jacobian [Jv; Jw] in the torso frame, at the TCP."""
        frames, T = self._chain(np.asarray(q, float))
        pe = T[:3, 3]
        J = np.zeros((6, 7))
        for i, (o, a) in enumerate(frames):
            J[:3, i] = np.cross(a, pe - o)
            J[3:, i] = a
        return J

    def manipulability(self, q, pos_only=True):
        J = self.jacobian(q)
        if pos_only:
            J = J[:3]
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def ik(self, T_des, q0, pos_only=False, iters=600, tol=1e-5, lam=0.05, k_null=0.2, q_pref=None, step=0.6):
        """Damped-least-squares IK with null-space joint-centering. Returns (q, converged, err)."""
        q = np.array(q0, float)
        q_mid = 0.5 * (self.q_lo + self.q_hi) if q_pref is None else q_pref
        for _ in range(iters):
            T = self.fk(q)
            ep = T_des[:3, 3] - T[:3, 3]
            if pos_only:
                e = ep; J = self.jacobian(q)[:3]
            else:
                R, Rd = T[:3, :3], T_des[:3, :3]
                eo = 0.5 * (np.cross(R[:, 0], Rd[:, 0]) + np.cross(R[:, 1], Rd[:, 1]) + np.cross(R[:, 2], Rd[:, 2]))
                e = np.concatenate([ep, eo]); J = self.jacobian(q)
            if np.linalg.norm(e) < tol:
                return q, True, float(np.linalg.norm(e))
            lam_e = lam * min(1.0, np.linalg.norm(e) / 0.1) + 1e-5      # error-scaled damping: robust far away, exact at the end
            JJt = J @ J.T + lam_e ** 2 * np.eye(J.shape[0])
            Jp = J.T @ np.linalg.inv(JJt)
            k_n = k_null * min(1.0, np.linalg.norm(e) / 0.01)           # null-space centring fades out as the target is reached
            dq = Jp @ e + (np.eye(7) - Jp @ J) @ (k_n * (q_mid - q))
            q = np.clip(q + step * dq, self.q_lo, self.q_hi)
        T = self.fk(q)
        err = np.linalg.norm(T_des[:3, 3] - T[:3, 3])
        return q, bool(err < 1e-3), float(err)

    def reach_samples(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        qs = rng.uniform(self.q_lo, self.q_hi, size=(n, 7))
        return np.array([self.fk(q)[:3, 3] for q in qs])
