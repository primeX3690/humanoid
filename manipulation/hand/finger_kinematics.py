"""Planar finger kinematics in the palm frame, consistent with model/hand_model.py (same DIGITS table)."""
from __future__ import annotations
import numpy as np
from model.hand_model import DIGITS, FLEX_JOINTS, JOINT_RANGE


def link_dirs(d, q):
    """unit direction of every phalanx; angle 0 = pointing -z, positive q curls toward the opposing side (-y for fingers, +y for the thumb)."""
    sgn = DIGITS[d][2]; phi = np.cumsum(np.asarray(q, float))
    return np.stack([np.zeros_like(phi), -sgn * np.sin(phi), -np.cos(phi)], 1)


def joint_positions(d, q):
    base, L = np.array(DIGITS[d][0]), DIGITS[d][1]; dirs = link_dirs(d, q)
    pts = [base]
    for i in range(len(q)): pts.append(pts[-1] + L[i] * dirs[i])
    return np.array(pts)                     # (n+1, 3): base, each joint, tip


def tip(d, q): return joint_positions(d, q)[-1]


def tip_jacobian(d, q):
    """d tip / d q (3 x n) for the planar chain."""
    sgn = DIGITS[d][2]; L = DIGITS[d][1]; phi = np.cumsum(q); n = len(q); J = np.zeros((3, n))
    for j in range(n):
        for i in range(j, n):
            J[1, j] += -sgn * L[i] * np.cos(phi[i]); J[2, j] += L[i] * np.sin(phi[i])
    return J


def clip_q(d, q): return np.clip(q, JOINT_RANGE[0], JOINT_RANGE[1])


def surface_points(d, q, n_per=6):
    """points along every phalanx axis (for power-grasp contact search); the last one is the fingertip."""
    jp = joint_positions(d, q); pts = []
    for a, b in zip(jp[:-1], jp[1:]):
        for t in np.linspace(0, 1, n_per, endpoint=False): pts.append(a + t * (b - a))
    pts.append(jp[-1]); return np.array(pts)
