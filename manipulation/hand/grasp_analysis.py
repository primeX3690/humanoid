"""Multi-contact grasp analysis: grasp matrix, force closure (LP), Ferrari-Canny epsilon quality, minimum-norm contact force distribution."""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull


def _tangents(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1); return t1, np.cross(n, t1)


def friction_cone_edges(n, mu, k=4):
    """k unit-normal-force edges of the linearised friction cone (force on the OBJECT; n points INTO the object)."""
    t1, t2 = _tangents(n); out = []
    for i in range(k):
        a = 2 * np.pi * i / k; e = n + mu * (np.cos(a) * t1 + np.sin(a) * t2); out.append(e / np.linalg.norm(e))
    return np.array(out)


def generator_wrenches(contacts, mu, k=4, torque_scale=1.0):
    """6 x (k*m): each column = wrench [f; p x f] of one cone edge (p about the object centre)."""
    cols = []
    for p, n in contacts:
        for f in friction_cone_edges(np.asarray(n, float) / np.linalg.norm(n), mu, k): cols.append(np.concatenate([f, torque_scale * np.cross(p, f)]))
    return np.array(cols).T


def is_force_closure(contacts, mu, k=4, min_quality=1e-3):
    """Rank-6 + a strictly positive internal-force combination (LP) + epsilon quality >= min_quality. The quality floor matters: with a
    vanishingly small friction coefficient the LP is still technically feasible but the grasp resists nothing in practice."""
    G = generator_wrenches(contacts, mu, k)
    if np.linalg.matrix_rank(G, tol=1e-9) < 6: return False
    m = G.shape[1]
    r = linprog(np.zeros(m), A_eq=G, b_eq=np.zeros(6), bounds=[(1.0, None)] * m, method="highs")      # strictly positive combination = 0
    return bool(r.status == 0) and epsilon_quality(contacts, mu) >= min_quality


def epsilon_quality(contacts, mu, k=6, torque_scale=1.0):
    """Radius of the largest origin-centred ball inside the convex hull of the (unit-force) generator wrenches (0 if not closure)."""
    G = generator_wrenches(contacts, mu, k, torque_scale).T
    if np.linalg.matrix_rank(G) < 6: return 0.0
    try: h = ConvexHull(G)
    except Exception: return 0.0
    off = h.equations[:, -1]                              # n.x + off <= 0 inside
    return float(max(0.0, -off.max())) if np.all(off < 1e-12) else 0.0


def distribute_forces(contacts, mu, w_ext, f_max=20.0, k=4):
    """Min sum(lambda^2) with G lambda = -w_ext, lambda >= 0, normal force per contact <= f_max. Returns (forces (m,3), ok)."""
    G = generator_wrenches(contacts, mu, k); m = G.shape[1]
    from control.fast_qp import solve_qp
    nc = len(contacts)
    A = np.vstack([G, np.eye(m), np.kron(np.eye(nc), np.ones((1, k)))])
    l = np.concatenate([-w_ext, np.zeros(m), np.zeros(nc)]); u = np.concatenate([-w_ext, np.full(m, np.inf), np.full(nc, f_max)])
    r = solve_qp(np.eye(m) * 2.0, np.zeros(m), A, l, u)
    if r.info.status_val <= 0: return None, False
    lam = np.maximum(r.x, 0); F = []
    for i, (p, n) in enumerate(contacts):
        E = friction_cone_edges(np.asarray(n, float) / np.linalg.norm(n), mu, k); F.append(lam[i * k:(i + 1) * k] @ E)
    F = np.array(F); res = np.linalg.norm(G @ lam + w_ext)
    return F, bool(res < 1e-5)
