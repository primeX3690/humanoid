"""Plan a multi-finger grasp of a sphere / upright cylinder given in the PALM frame: close each digit until its fingertip pad touches the
surface, then verify force closure and quality. Synergy style: one scalar closure per digit (mcp:pip:dip = 1 : 1.1 : 0.7)."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from manipulation.hand.finger_kinematics import tip, surface_points, FLEX_JOINTS, JOINT_RANGE
from manipulation.hand.grasp_analysis import is_force_closure, epsilon_quality, distribute_forces

PATTERN = {"mcp": 1.0, "pip": 1.1, "dip": 0.7, "ip": 0.9}


@dataclass
class Obj:
    kind: str                # "sphere" | "cylinder" (axis along palm x... world-up in the palm frame is -z; here: axis parallel to x)
    center: np.ndarray
    radius: float
    length: float = 0.1


def sdf(o: Obj, p):
    if o.kind == "sphere": return float(np.linalg.norm(p - o.center) - o.radius)
    d = p - o.center; d[0] = 0.0                              # infinite-ish cylinder with axis along palm x (fingers wrap around it)
    return float(np.linalg.norm(d) - o.radius) if abs(p[0] - o.center[0]) <= o.length / 2 else 1e3


def inward_normal(o: Obj, p):
    d = p - o.center
    if o.kind == "cylinder": d[0] = 0.0
    n = -d / max(np.linalg.norm(d), 1e-9); return n


def q_of(d, s): return np.array([s * PATTERN[j] for j in FLEX_JOINTS[d]]).clip(*JOINT_RANGE)


def _touch(d, o, s, pad):
    pts = surface_points(d, q_of(d, s)); v = np.array([sdf(o, p) for p in pts]) - pad
    i = int(np.argmin(v)); return float(v[i]), pts[i]


def close_digit(d, o: Obj, pad=0.008):
    """smallest closure s at which ANY point of the digit (power grasp: palm-side phalanges too) is within `pad` (finger radius) of the
    surface; returns (s, contact point) or None. Digits that start in contact return s = 0."""
    ss = np.linspace(0, 1, 61); prev = _touch(d, o, 0.0, pad)
    if prev[0] <= 0: return 0.0, prev[1]
    for a, b in zip(ss[:-1], ss[1:]):
        vb, pb = _touch(d, o, b, pad)
        if vb <= 0:
            lo, hi = a, b
            for _ in range(30):
                m = 0.5 * (lo + hi)
                if _touch(d, o, m, pad)[0] > 0: lo = m
                else: hi = m
            return hi, _touch(d, o, hi, pad)[1]
    return None


def plan_grasp(o: Obj, mu=0.6, digits=("ix", "md", "rg", "th")):
    from model.hand_model import DIGITS
    if any(sdf(o, np.array(DIGITS[d][0])) < 0.0 for d in DIGITS) or sdf(o, np.array([0.0, 0.0, -0.02])) < 0.0:
        return dict(success=False, joint_targets={}, contacts=[], missing=list(digits), quality=0.0, mu=mu, reason="object overlaps the palm / finger bases (too big or too close)")
    contacts, q, missing = [], {}, []
    for d in digits:
        r = close_digit(d, o)
        if r is None: missing.append(d); continue
        s, t = r; qq = q_of(d, s); n = inward_normal(o, t)
        contacts.append((t - o.center, n)); q[d] = qq
    ok = len(contacts) >= 3 and is_force_closure(contacts, mu)
    return dict(success=bool(ok), joint_targets=q, contacts=contacts, missing=missing,
                quality=epsilon_quality(contacts, mu) if ok else 0.0, mu=mu)


def hold_forces(plan, mass_kg, mu=None):
    w = np.array([0, 0, -9.81 * mass_kg, 0, 0, 0]) * np.array([1, 1, 1, 0, 0, 0])
    w[:3] = [0, 0, -mass_kg * 9.81]                       # in the palm frame gravity direction depends on the wrist pose: caller rotates if needed
    return distribute_forces(plan["contacts"], mu or plan["mu"], w)
