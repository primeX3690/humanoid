"""
Lateral (and general-direction) push-recovery step planner with cross-over steps, step-time optimisation and CoP shifting.

Closes the v2 gap "lateral stepping is only solved up to ~90 N and leg-to-leg collision is not modelled".

Model: 2-D LIPM / divergent component of motion  xi = c + v/omega.  During the single-support remainder T the stance
CoP can be moved inside the stance foot (ankle strategy), xi(T) = p_cop + (xi0 - p_cop) e^{omega T}.  Capturable iff after touchdown xi_T lies
inside the convex hull of both feet (then a double-support DCM controller can bring the robot to rest).  The planner searches
step time T and landing point p inside the kinematically reachable set (including the CROSS-OVER region on the far side of the
stance foot, subject to ankle/foot collision, reach and swing-foot speed) and returns the placement with the largest hull margin.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial import ConvexHull
from planning.collision_aware_swing import SwingGeometry, ankle_gap, foot_gap, plan_collision_free_swing


@dataclass
class RecoveryParams:
    com_height: float = 0.85
    T_min: float = 0.18
    T_max: float = 0.70
    n_T: int = 27
    cop_inset_x: float = 0.02
    cop_inset_y: float = 0.01
    use_cop_shift: bool = True
    allow_crossover: bool = True
    x_rel: tuple = (-0.50, 0.65)
    y_out_max: float = 0.55           # natural-side lateral reach beyond stance ankle
    y_cross_max: float = 0.30         # cross-over reach
    r_max: float = 0.65
    v_swing_max: float = 2.2          # m/s average swing-foot speed available
    t_lift: float = 0.08              # time to unload / lift before the swing starts
    grid: float = 0.02
    margin_req: float = 0.0
    geom: SwingGeometry = field(default_factory=SwingGeometry)

    @property
    def omega(self): return float(np.sqrt(9.81 / self.com_height))


_CAND_CACHE = {}


def _candidates(p: RecoveryParams):
    """Reachable landing points relative to the stance foot, cached per parameter set (translation invariant)."""
    g = p.geom
    key = (p.x_rel, p.y_out_max, p.y_cross_max, p.r_max, p.grid, p.allow_crossover, g.r_leg, g.margin, g.foot_len, g.foot_wid)
    if key in _CAND_CACHE: return _CAND_CACHE[key]
    xs = np.arange(p.x_rel[0], p.x_rel[1] + 1e-9, p.grid); ys = np.arange(-p.y_cross_max, p.y_out_max + 1e-9, p.grid)
    DX, DY = np.meshgrid(xs, ys, indexing="ij"); DX, DY = DX.ravel(), DY.ravel()
    h = 0.5 * (g.foot_len - g.foot_wid)
    ok = np.hypot(DX, DY) <= p.r_max
    ok &= np.hypot(DX, DY) >= 2 * g.r_leg + g.margin                          # ankle-ankle clearance
    ok &= np.hypot(np.maximum(np.abs(DX) - 2 * h, 0.0), DY) - g.foot_wid >= g.margin   # foot-foot clearance (parallel capsules)
    if not p.allow_crossover: ok &= DY >= 0
    out = (np.column_stack([DX[ok], DY[ok]]), DY[ok] < 0)
    _CAND_CACHE[key] = out
    return out


def rect(c, p: RecoveryParams, inset=(0.0, 0.0)):
    hx, hy = p.geom.foot_len / 2 - inset[0], p.geom.foot_wid / 2 - inset[1]
    c = np.asarray(c, float)[:2]
    return np.array([c + [sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)])


def hull_signed_margin(pt, verts):
    """Signed distance of pt to the convex hull of verts (inside > 0)."""
    h = ConvexHull(verts)
    # scipy equations: n.x + b <= 0 inside, n unit
    d = -(h.equations[:, :2] @ np.asarray(pt, float)[:2] + h.equations[:, 2])
    return float(d.min())


def project_to_polygon(pt, verts):
    h = ConvexHull(verts); V = verts[h.vertices]
    if hull_signed_margin(pt, verts) >= 0: return np.asarray(pt, float)[:2]
    best, bd = None, np.inf
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]; ab = b - a
        t = np.clip(np.dot(pt[:2] - a, ab) / np.dot(ab, ab), 0, 1); q = a + t * ab
        dd = np.linalg.norm(pt[:2] - q)
        if dd < bd: best, bd = q, dd
    return best


def plan_recovery(com_xy, vel_xy, stance_xy, swing_xy, p: RecoveryParams | None = None, verify_path=True):
    p = p or RecoveryParams()
    w = p.omega
    com = np.asarray(com_xy, float); vel = np.asarray(vel_xy, float)
    st = np.asarray(stance_xy, float)[:2]; sw = np.asarray(swing_xy, float)[:2]
    side = 1.0 if sw[1] >= st[1] else -1.0                        # +1: swing foot is on the +y side of the stance foot
    xi0 = com + vel / w
    cop = xi0.copy()
    if p.use_cop_shift:
        r = rect(st, p, (p.cop_inset_x, p.cop_inset_y)); lo, hi = r.min(0), r.max(0)
        cop = np.clip(xi0, lo, hi)
    else:
        cop = st.copy()
    rel, X = _candidates(p)                                       # (N,2) in (dx, d_natural) coordinates, cross flags
    Q = st + np.column_stack([rel[:, 0], side * rel[:, 1]])
    if len(Q) == 0:
        return dict(feasible=False, reason="no reachable placement")
    best = None
    for T in np.linspace(p.T_min, p.T_max, p.n_T):
        xiT = cop + (xi0 - cop) * np.exp(w * T)
        reach_ok = np.linalg.norm(Q - sw, axis=1) <= p.v_swing_max * max(T - p.t_lift, 1e-3)
        if not reach_ok.any(): continue
        idx = np.where(reach_ok)[0]
        # best candidate = largest hull margin; evaluate the 10 placements closest to xi_T
        order = idx[np.argsort(np.linalg.norm(Q[idx] - xiT, axis=1))[:10]]
        for j in order:
            verts = np.vstack([rect(st, p), rect(Q[j], p)])
            m = hull_signed_margin(xiT, verts)
            if best is None or m > best["margin"] + 1e-9:
                best = dict(margin=m, T=float(T), target=Q[j].copy(), crossover=bool(X[j]), xi_T=xiT.copy())
    if best is None:
        return dict(feasible=False, reason="swing foot too slow for any placement")
    out = dict(best, feasible=best["margin"] >= p.margin_req, xi0=xi0, cop=cop, side=side)
    if verify_path and out["feasible"]:
        sp = plan_collision_free_swing(np.r_[sw, 0.0], np.r_[best["target"], 0.0], np.r_[st, 0.0], T=max(best["T"] - p.t_lift, 0.1), g=p.geom)
        out["swing"] = sp
        if not sp.feasible: out["feasible"] = False; out["reason"] = "swing path not collision-free"
    return out


def simulate_recovery(com_xy, vel_xy, stance_xy, plan, p: RecoveryParams | None = None, T_total=4.0, dt=0.002, k_dcm=3.0):
    """Forward-simulate: single support with the planned CoP for T, then double support with a clipped DCM controller."""
    p = p or RecoveryParams(); w = p.omega
    c = np.asarray(com_xy, float).copy(); v = np.asarray(vel_xy, float).copy()
    st = np.asarray(stance_xy, float)[:2]; tgt = plan["target"]
    verts2 = np.vstack([rect(st, p), rect(tgt, p)]); xref = 0.5 * (st + tgt)
    inside_max = 0.0
    for i in range(int(T_total / dt)):
        t = i * dt; xi = c + v / w
        if t < plan["T"]:
            zmp = plan["cop"]
        else:
            zmp = project_to_polygon(xref + (1 + k_dcm / w) * (xi - xref), verts2)
        a = w * w * (c - zmp)
        v += a * dt; c += v * dt
        inside_max = max(inside_max, np.linalg.norm(c - xref))
    xi = c + v / w
    return dict(final_dcm_err=float(np.linalg.norm(xi - xref)), final_com=c, max_excursion=float(inside_max),
                recovered=bool(np.linalg.norm(xi - xref) < 0.02 and hull_signed_margin(c, verts2) > 0))


def max_capturable_speed(direction, stance_xy, swing_xy, p: RecoveryParams, com_xy=None, hi=3.0):
    """Largest CoM speed along `direction` (unit 2-vector) the planner can capture (bisection)."""
    com = np.asarray(stance_xy, float) if com_xy is None else np.asarray(com_xy, float)
    d = np.asarray(direction, float) / np.linalg.norm(direction)
    lo_v, hi_v = 0.0, hi
    for _ in range(18):
        mid = 0.5 * (lo_v + hi_v)
        r = plan_recovery(com, mid * d, stance_xy, swing_xy, p, verify_path=False)
        if r["feasible"]: lo_v = mid
        else: hi_v = mid
    return lo_v


def choose_recovery_step(com_xy, vel_xy, pL_xy, pR_xy, params: RecoveryParams):
    """Try each foot as the stance foot; return the plan with the largest capturability margin (feasible ones first)."""
    best = None
    for stance_name, st, sw in (("L", pL_xy, pR_xy), ("R", pR_xy, pL_xy)):
        r = plan_recovery(com_xy, vel_xy, st, sw, params)
        if "margin" not in r:
            continue
        key = (bool(r["feasible"]), r["margin"])
        if best is None or key > best[0]:
            best = (key, stance_name, r)
    if best is None:
        return None
    _, stance, r = best
    r = dict(r); r["stance"] = stance; r["side"] = "R" if stance == "L" else "L"
    return r
