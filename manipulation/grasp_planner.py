"""
Antipodal parallel-jaw grasp planner for boxes and upright cylinders, scored for width margin, approach direction, centring and
arm reachability (IK + manipulability).  v2 grasped one red block with a fixed top-down yaw; this plans over the object's geometry.

TCP frame convention used here (check it against YOUR gripper model before running on hardware):
    z-axis = -approach direction (the TCP sits along hand -z, see kinematics/arm_kinematics.tcp_offset)
    y-axis = jaw closing axis,   x-axis = y x z
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class GripperSpec:
    max_open: float = 0.08
    min_open: float = 0.0
    margin: float = 0.008            # jaw clearance each side when opening
    finger_len: float = 0.03
    mu: float = 0.5                  # friction used for the force-closure check
    tip_half_thickness: float = 0.01


@dataclass
class Grasp:
    T: np.ndarray                    # 4x4 TCP pose (world/torso frame, same as the object pose)
    width: float                     # object width across the jaws
    approach: np.ndarray
    score: float
    kind: str
    parts: dict


def frame_from(approach, closing, center):
    z = -np.asarray(approach, float) / np.linalg.norm(approach)
    y = np.asarray(closing, float); y = y - (y @ z) * z; y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4); T[:3, :3] = np.column_stack([x, y, z]); T[:3, 3] = center
    return T


def force_closure_2contact(n1, n2, p1, p2, mu):
    """Antipodal test: the line p1->p2 must lie inside both friction cones (n inward-pointing from each contact)."""
    d = (p2 - p1) / np.linalg.norm(p2 - p1)
    ang = np.arctan(mu)
    return bool(np.arccos(np.clip(n1 @ d, -1, 1)) <= ang + 1e-9 and np.arccos(np.clip(n2 @ -d, -1, 1)) <= ang + 1e-9)


def box_grasps(center, R, half, g: GripperSpec, table_z=None, approach_dirs=None):
    """R: object orientation (columns = box axes in world). half: half extents along those axes."""
    out = []
    center = np.asarray(center, float)
    cands = approach_dirs or [np.array([0, 0, -1.0]), np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0])]
    for ci in range(3):                                       # closing axis = box axis ci
        w = 2 * half[ci]
        if w > g.max_open - 2 * g.margin or w < g.min_open: continue
        closing = R[:, ci]
        for a in cands:
            a = np.asarray(a, float); a = a / np.linalg.norm(a)
            if abs(a @ closing) > 0.2: continue               # approach must be (nearly) perpendicular to the jaw axis
            if table_z is not None:                              # fingertips must stay above the support surface
                low = center[2] - g.tip_half_thickness - (half[ci] + g.tip_half_thickness if abs(closing[2]) > 0.5 else 0.0)
                if low < table_z + 0.002: continue
            p1 = center - closing * half[ci]; p2 = center + closing * half[ci]
            fc = force_closure_2contact(closing, -closing, p1, p2, g.mu)
            if not fc: continue
            T = frame_from(a, closing, center)
            verticality = max(-a[2], 0.0)
            score = 1.0 * (g.max_open - w) / g.max_open + 0.6 * verticality + 0.2
            out.append(Grasp(T, w, a, float(score), "box", dict(axis=ci)))
    return out


def cylinder_grasps(center, radius, height, g: GripperSpec, n_yaw=8, table_z=None):
    """Upright cylinder: side grasps (any yaw, horizontal approach) and top-down grasps across the diameter."""
    out = []; w = 2 * radius; center = np.asarray(center, float)
    if w > g.max_open - 2 * g.margin: return out
    for k in range(n_yaw):
        th = np.pi * k / n_yaw; closing = np.array([np.cos(th), np.sin(th), 0.0])
        a = np.array([0, 0, -1.0])
        out.append(Grasp(frame_from(a, closing, center + np.array([0, 0, height / 2 - g.finger_len / 2])), w, a, 0.9 + (g.max_open - w) / g.max_open, "cyl_top", dict(yaw=th)))
        ah = np.array([-np.sin(th), np.cos(th), 0.0]) * 0 + np.array([np.cos(th + np.pi / 2), np.sin(th + np.pi / 2), 0.0])
        out.append(Grasp(frame_from(ah, closing, center), w, ah, 0.5 + (g.max_open - w) / g.max_open, "cyl_side", dict(yaw=th)))
    return out


def rank_by_reachability(grasps, arm, q0, to_torso=None, min_manip=0.01, pregrasp=0.08):
    """Keep grasps the arm can reach (grasp pose AND a pre-grasp offset along the approach), re-score with manipulability."""
    ranked = []
    for gr in grasps:
        T = gr.T if to_torso is None else to_torso @ gr.T
        q, ok, err = arm.ik(T, q0)
        if not ok: continue
        Tp = T.copy(); Tp[:3, 3] -= gr.approach * pregrasp
        q2, ok2, _ = arm.ik(Tp, q)
        if not ok2: continue
        m = arm.manipulability(q)
        if m < min_manip: continue
        gr2 = Grasp(gr.T, gr.width, gr.approach, gr.score + 2.0 * m, gr.kind, dict(gr.parts, q_grasp=q, q_pre=q2, manip=m))
        ranked.append(gr2)
    return sorted(ranked, key=lambda x: -x.score)
