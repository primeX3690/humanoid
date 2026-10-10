"""
Collision-aware swing-foot planning (leg-to-leg and foot-to-foot), which the v2 stepping code explicitly did NOT model
("leg-to-leg collision is NOT modelled, so the cross-over path needs a collision-aware swing trajectory").

Geometry (plan view, metres; all ASSUMED from model/humanoid_model.py defaults - change in `SwingGeometry`):
  * each shank/ankle = vertical cylinder of radius r_leg  -> two ankles must stay >= 2*r_leg + margin apart (any height,
    a foot can never pass THROUGH the other leg, only in front of / behind it)
  * each foot = capsule (segment of length foot_len - foot_wid, radius foot_wid/2) of height foot_h; the swing foot may
    overlap the stance foot in plan view only if its sole is >= foot_h + z_margin above the ground
Path candidates: straight line, pass-in-front, pass-behind; the shortest collision-free one is time-parameterised with a
minimum-jerk profile (zero velocity/acceleration at lift-off and touch-down) and a sin^2 lift profile (raised over the stance
foot when needed).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SwingGeometry:
    r_leg: float = 0.06
    margin: float = 0.02
    foot_len: float = 0.20
    foot_wid: float = 0.08
    foot_h: float = 0.05
    z_margin: float = 0.02
    lift: float = 0.06            # nominal apex height of the sole
    lift_max: float = 0.16
    reach: float = 0.60           # max ankle-to-ankle distance the leg can span


def seg_dist(p, a, b):
    ab = b - a; t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def seg_seg_dist(a0, a1, b0, b1):
    # planar segment-segment distance (segments here are parallel to x, so sampling the endpoints + projections is exact)
    return min(seg_dist(a0, b0, b1), seg_dist(a1, b0, b1), seg_dist(b0, a0, a1), seg_dist(b1, a0, a1))


def foot_segment(c, g):
    h = 0.5 * (g.foot_len - g.foot_wid)
    c = np.asarray(c, float)[:2]
    return c + np.array([-h, 0.0]), c + np.array([h, 0.0])


def foot_gap(c1, c2, g):
    """Edge-to-edge distance between two foot capsules in plan view (negative = overlap)."""
    a0, a1 = foot_segment(c1, g); b0, b1 = foot_segment(c2, g)
    return seg_seg_dist(a0, a1, b0, b1) - g.foot_wid


def ankle_gap(c1, c2, g):
    return float(np.linalg.norm(np.asarray(c1)[:2] - np.asarray(c2)[:2])) - (2 * g.r_leg + g.margin)


@dataclass
class SwingPlan:
    t: np.ndarray; pos: np.ndarray; vel: np.ndarray; acc: np.ndarray
    kind: str; feasible: bool; goal: np.ndarray; goal_moved: bool
    min_ankle_gap: float; min_overlap_height_gap: float


def clearance(pos, stance, g: SwingGeometry):
    """Worst-case (min) ankle gap and (min) height margin whenever feet overlap in plan view. pos: (N,3)."""
    ag = min(ankle_gap(p, stance, g) for p in pos)
    hm = np.inf
    for p in pos:
        if foot_gap(p, stance, g) < g.margin:
            hm = min(hm, p[2] - (g.foot_h + g.z_margin))
    return float(ag), float(hm)


def _min_jerk(u):
    return 10 * u**3 - 15 * u**4 + 6 * u**5


def _polyline_points(wps, n):
    seg = np.linalg.norm(np.diff(wps, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)]); L = s[-1]
    if L < 1e-9: return np.tile(wps[0], (n, 1)), 0.0
    ss = np.linspace(0, L, n)
    return np.stack([np.interp(ss, s, wps[:, k]) for k in range(wps.shape[1])], 1), L


def project_goal(goal, stance, g: SwingGeometry, side_hint=0.0):
    """Push the landing point out of collision with the stance leg/foot (outward in y first, then forward)."""
    goal = np.asarray(goal, float).copy(); moved = False
    for _ in range(60):
        if ankle_gap(goal, stance, g) >= 0 and foot_gap(goal, stance, g) >= g.margin: break
        d = goal[:2] - stance[:2]
        sgn = np.sign(d[1]) if abs(d[1]) > 1e-6 else (np.sign(side_hint) or 1.0)
        goal[1] += sgn * 0.01; moved = True
    return goal, moved


def plan_collision_free_swing(start, goal, stance, T=0.5, dt=0.005, g: SwingGeometry | None = None, ground_z=0.0):
    g = g or SwingGeometry()
    start = np.asarray(start, float); goal0 = np.asarray(goal, float); stance = np.asarray(stance, float)
    goal, moved = project_goal(goal0, stance, g, side_hint=start[1] - stance[1])
    n = max(int(round(T / dt)) + 1, 5); u = _min_jerk(np.linspace(0, 1, n)); tt = np.linspace(0, T, n)
    dys, dyg = start[1] - stance[1], goal[1] - stance[1]
    min_dy = 0.0 if dys * dyg < 0 else min(abs(dys), abs(dyg))      # a lateral cross-over passes straight through y = stance.y
    d_pass = np.sqrt(max((2 * g.r_leg + g.margin) ** 2 - min_dy ** 2, 0.0)) + 0.02
    ymid = 0.5 * (start[1] + goal[1])
    cands = {"direct": np.array([start[:2], goal[:2]]),
             "front": np.array([start[:2], [max(stance[0] + d_pass, start[0]), start[1]], [stance[0] + d_pass, ymid], [stance[0] + d_pass, goal[1]], goal[:2]]),
             "back": np.array([start[:2], [min(stance[0] - d_pass, start[0]), start[1]], [stance[0] - d_pass, ymid], [stance[0] - d_pass, goal[1]], goal[:2]])}
    best = None
    for kind, wps in cands.items():
        xy, L = _polyline_points(wps, 400)
        ok = all(ankle_gap(p, stance, g) >= -1e-9 for p in xy) and max(np.linalg.norm(p - stance[:2]) for p in xy) <= g.reach
        if ok and (best is None or L < best[2]): best = (kind, wps, L)
    feasible = best is not None
    if best is None:  # nothing clears: report the least-bad (direct) so callers can see WHY via min_ankle_gap
        best = ("direct", cands["direct"], 0.0)
    kind, wps, _ = best
    xy_u, _ = _polyline_points(wps, 4001)
    idx = np.clip((u * 4000).round().astype(int), 0, 4000)
    xy = xy_u[idx]
    # lift: base sin^2 profile, raised wherever the foot passes over the stance foot in plan view (guaranteed >= required height)
    gap = np.array([foot_gap(np.r_[p, 0], stance, g) for p in xy])
    need_h = g.foot_h + g.z_margin + 0.01
    req = np.where(gap < g.margin, need_h, 0.0)
    k = 12                                                           # dilate in time, then smooth, then re-impose the floor
    dil = np.array([req[max(0, i - k):i + k + 1].max() for i in range(n)])
    w = np.ones(2 * k + 1) / (2 * k + 1)
    sm = np.convolve(np.pad(dil, k, mode="edge"), w, mode="valid")
    floor = np.maximum(sm, dil) if dil.max() > 0 else np.zeros(n)
    floor = np.minimum(floor, g.lift_max)
    base = ground_z + (start[2] - ground_z) * (1 - u) + (goal[2] - ground_z) * u + g.lift * np.sin(np.pi * u) ** 2
    z = np.maximum(base, ground_z + floor)
    pos = np.column_stack([xy, z])
    vel = np.gradient(pos, tt, axis=0); acc = np.gradient(vel, tt, axis=0)
    ag, hm = clearance(pos, stance, g)
    return SwingPlan(tt, pos, vel, acc, kind, feasible and ag >= -1e-6 and hm >= -1e-6, goal, moved, ag, hm)
