"""Joint-space RRT-Connect + shortcut smoothing + velocity-limited time parameterisation, with arm collision checking.

plan(checker, side, q_start, q_goal) -> (path (N,7), ok).  `trajectory()` turns the path into min-jerk-timed (t, q, qd) samples that
respect per-joint speed limits (use model.humanoid_model.joint_speed_limits for the real limits).
"""
from __future__ import annotations
import numpy as np


class JointSpace:
    def __init__(self, checker, side, other_q=None, edge_res=0.05):
        self.c, self.side, self.other = checker, side, other_q
        self.arm = checker.arms[side]; self.lo, self.hi = self.arm.q_lo, self.arm.q_hi; self.res = edge_res

    def free(self, q):
        qq = {self.side: q}
        if self.other is not None: qq["R" if self.side == "L" else "L"] = self.other
        return (not self.c.in_collision(qq)) and bool(np.all(q >= self.lo) and np.all(q <= self.hi))

    def edge_free(self, a, b):
        n = max(int(np.ceil(np.linalg.norm(b - a) / self.res)), 1)
        return all(self.free(a + (b - a) * k / n) for k in range(1, n + 1))


def rrt_connect(space: JointSpace, q0, q1, iters=3000, step=0.25, seed=0, goal_bias=0.1):
    rng = np.random.default_rng(seed)
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    if not (space.free(q0) and space.free(q1)): return None
    if space.edge_free(q0, q1): return np.array([q0, q1])
    Ta = [q0]; pa = [-1]; Tb = [q1]; pb = [-1]; swapped = False

    def nearest(T, q): return int(np.argmin(np.linalg.norm(np.asarray(T) - q, axis=1)))

    def extend(T, P, q):
        i = nearest(T, q); d = q - T[i]; L = np.linalg.norm(d)
        qn = q if L <= step else T[i] + d / L * step
        if space.edge_free(T[i], qn): T.append(qn); P.append(i); return len(T) - 1, np.allclose(qn, q)
        return None, False

    for _ in range(iters):
        qr = q1 if rng.random() < goal_bias and not swapped else rng.uniform(space.lo, space.hi)
        ia, _ = extend(Ta, pa, qr)
        if ia is not None:
            while True:
                ib, reached = extend(Tb, pb, Ta[ia])
                if ib is None: break
                if reached:
                    path_a = []; i = ia
                    while i != -1: path_a.append(Ta[i]); i = pa[i]
                    path_b = []; i = ib
                    while i != -1: path_b.append(Tb[i]); i = pb[i]
                    path = path_a[::-1] + path_b
                    if swapped: path = path[::-1]
                    return np.array(path)
        Ta, Tb = Tb, Ta; pa, pb = pb, pa; swapped = not swapped
    return None


def shortcut(space: JointSpace, path, iters=200, seed=1):
    rng = np.random.default_rng(seed); p = [q for q in path]
    for _ in range(iters):
        if len(p) < 3: break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2: continue
        if space.edge_free(p[i], p[j]): p = p[:i + 1] + p[j:]
    return np.array(p)


def plan(checker, side, q_start, q_goal, other_q=None, seed=0, **kw):
    sp = JointSpace(checker, side, other_q)
    raw = rrt_connect(sp, q_start, q_goal, seed=seed, **kw)
    if raw is None: return None, False
    return shortcut(sp, raw), True


def path_length(p): return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))


def trajectory(path, qd_max, dt=0.01, accel_time=0.3):
    """Piecewise min-jerk through the waypoints; each segment's duration = max(|dq|/qd_max)*1.5 (min-jerk peak velocity is 1.875 avg)."""
    qd_max = np.broadcast_to(np.asarray(qd_max, float), path.shape[1])
    T, Q, V = [0.0], [path[0]], [np.zeros(path.shape[1])]
    t0 = 0.0
    for a, b in zip(path[:-1], path[1:]):
        dur = max(float(np.max(np.abs(b - a) / qd_max)) * 1.875, accel_time)
        n = max(int(dur / dt), 2); u = np.linspace(0, 1, n + 1)[1:]
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5; sd = (30 * u ** 2 - 60 * u ** 3 + 30 * u ** 4) / dur
        for k in range(n):
            T.append(t0 + u[k] * dur); Q.append(a + (b - a) * s[k]); V.append((b - a) * sd[k])
        t0 += dur
    return np.array(T), np.array(Q), np.array(V)
