"""
kinematics/leg_fk.py

Forward kinematics for a standard 6-DOF anthropomorphic humanoid leg:
hip (yaw, roll, pitch) -> thigh -> knee (pitch) -> shank -> ankle (pitch, roll).
This exact joint arrangement (3 hip + 1 knee + 2 ankle = 6 DOF) is the
standard configuration used on real humanoid legs (e.g. NAO, and the
general class analyzed in "Complete Analytical Forward and Inverse
Kinematics for the NAO Humanoid Robot" and similar humanoid-leg-IK
literature) - not an invented arrangement.

Frame convention (consistent with dynamics/lipm.py's sagittal=x,
lateral=y axes): x = forward, y = left, z = up. Joint angle sign
convention: positive hip/knee/ankle PITCH flexes the leg forward
(swings the lower segment forward); positive ROLL is a right-hand
rotation about the forward (x) axis.

kinematics/leg_ik.py solves the inverse problem numerically (see that
module's docstring for why numerical rather than a hand-derived
closed-form solution was chosen here) and verifies against THIS forward
kinematics function - so correctness of the whole leg model rests on
this function being right, which is why it's kept as simple, explicit
4x4 homogeneous transforms rather than anything cleverer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LegParams:
    thigh_length_m: float = 0.45   # hip to knee
    shank_length_m: float = 0.45   # knee to ankle - together, 0.9m max reach vs the
                                    # LIPM's 0.8m CoM height (see docs/BUGS_FOUND.md:
                                    # a leg whose max reach exactly equals standing CoM
                                    # height leaves NO margin for the extra reach a
                                    # swinging leg needs while the hip trails behind the
                                    # swing foot mid-stride - a real, physical constraint,
                                    # not just a numeric edge case)
    foot_height_m: float = 0.05    # ankle joint to sole (for ground-contact bookkeeping elsewhere)


def _rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def forward_kinematics(joint_angles: np.ndarray, hip_position: np.ndarray, leg: LegParams
                        ) -> tuple[np.ndarray, np.ndarray]:
    """
    joint_angles: [hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll] (radians)
    hip_position: (3,) hip joint center in world/body frame
    Returns: (ankle_position (3,), ankle_orientation (3x3 rotation matrix))

    Chain: R_hip = Rz(yaw) @ Rx(roll) @ Ry(pitch) orients the thigh segment,
    which points along -z in the hip's own frame when all angles are zero
    (leg hanging straight down) - a standard, simple "leg at rest" pose.
    """
    hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll = joint_angles

    R_hip = _rot_z(hip_yaw) @ _rot_x(hip_roll) @ _rot_y(hip_pitch)
    knee_pos = hip_position + R_hip @ np.array([0.0, 0.0, -leg.thigh_length_m])

    R_knee = R_hip @ _rot_y(knee_pitch)
    ankle_pos = knee_pos + R_knee @ np.array([0.0, 0.0, -leg.shank_length_m])

    R_ankle = R_knee @ _rot_y(ankle_pitch) @ _rot_x(ankle_roll)

    return ankle_pos, R_ankle
