"""
Arm collision model (torso-frame, pure NumPy): arm links as capsules, torso as a box, both arms against each other, plus
user obstacles (spheres / boxes / capsules). v2 manipulation had NO collision checking: the arm went straight to the target.

Link geometry (from model/humanoid_model.py defaults): upper arm 0.28 m (r 0.035-0.04), forearm 0.26 m (r 0.03-0.035), hand sphere at the TCP.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from kinematics.arm_kinematics import Arm


def seg_seg_distance(p1, q1, p2, q2):
    """Closest distance between segments p1q1 and p2q2 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12: return float(np.linalg.norm(r))
    if a < 1e-12: s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e < 1e-12: t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2; den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0, 1) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0: t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1: t, s = 1.0, np.clip((b - c) / a, 0, 1)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def point_box_sdf(p, center, half):
    q = np.abs(np.asarray(p) - center) - half
    return float(np.linalg.norm(np.maximum(q, 0)) + min(max(q.max(), 0) * 0 + q.max(), 0.0))


@dataclass
class Capsule:
    a: np.ndarray; b: np.ndarray; r: float


@dataclass
class Sphere:
    c: np.ndarray; r: float


@dataclass
class Box:
    c: np.ndarray; half: np.ndarray


def arm_capsules(arm: Arm, q, r_upper=0.045, r_fore=0.04, r_hand=0.07):
    frames, T = arm._chain(np.asarray(q, float))
    sh, el, wr = frames[0][0], frames[3][0], frames[4][0]
    tcp = T[:3, 3]
    return [Capsule(sh, el, r_upper), Capsule(el, wr, r_fore), Capsule(wr, tcp, r_hand)]


def capsule_obstacle_distance(cap: Capsule, obs, n=12):
    if isinstance(obs, Capsule): return seg_seg_distance(cap.a, cap.b, obs.a, obs.b) - cap.r - obs.r
    if isinstance(obs, Sphere):
        ab = cap.b - cap.a; t = np.clip((obs.c - cap.a) @ ab / max(ab @ ab, 1e-12), 0, 1)
        return float(np.linalg.norm(cap.a + t * ab - obs.c)) - cap.r - obs.r
    pts = cap.a + np.linspace(0, 1, n)[:, None] * (cap.b - cap.a)               # Box: sampled signed distance (conservative to ~1 mm)
    return min(point_box_sdf(p, obs.c, obs.half) for p in pts) - cap.r


@dataclass
class ArmCollisionChecker:
    arms: dict                                   # {"L": Arm, "R": Arm}
    obstacles: list = field(default_factory=list)
    torso: Box = field(default_factory=lambda: Box(np.array([0.0, 0.0, 0.20]), np.array([0.09, 0.105, 0.22])))   # chest slab; shoulders at y=+-0.17
    margin: float = 0.01

    def clearance(self, q: dict) -> float:
        """Smallest clearance (m) over arm-obstacle, arm-torso and arm-arm pairs. q: {"L": q7, "R": q7} (missing arm = not checked)."""
        caps = {s: arm_capsules(self.arms[s], qq) for s, qq in q.items()}
        best = np.inf
        for s, cs in caps.items():
            for i, c in enumerate(cs):
                if i >= 1:                                    # the upper arm is attached to the shoulder inside the torso region
                    best = min(best, capsule_obstacle_distance(c, self.torso))
                for o in self.obstacles:
                    best = min(best, capsule_obstacle_distance(c, o))
        if len(caps) == 2:
            for c1 in caps["L"][1:]:
                for c2 in caps["R"][1:]:
                    best = min(best, seg_seg_distance(c1.a, c1.b, c2.a, c2.b) - c1.r - c2.r)
        return float(best)

    def in_collision(self, q: dict) -> bool:
        return self.clearance(q) < self.margin
