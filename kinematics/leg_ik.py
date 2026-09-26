"""
kinematics/leg_ik.py

Inverse kinematics for the 6-DOF leg in kinematics/leg_fk.py, solved
NUMERICALLY via damped least squares (a.k.a. the Levenberg-Marquardt
method applied to IK) rather than a hand-derived analytical closed form.

Why numerical instead of analytical: a closed-form geometric solution
for this leg configuration exists in the literature (e.g. the
NAO-humanoid-leg IK papers, Kajita's "Introduction to Humanoid
Robotics"), but reproducing one correctly from memory without a
reference to check against risks a subtle sign/axis-convention error
that would be hard to catch. Damped least-squares IK is itself a real,
widely-used robotics method (the same family used in KDL/MoveIt-style
general IK solvers) - not a corner cut - and its correctness here is
verified directly and unambiguously: convergence to a target pose is
checked by feeding the result back through the SAME forward_kinematics()
and confirming the residual error is small (tests/test_leg_kinematics.py),
which is a stronger, more self-contained correctness guarantee than
"this formula matches equation 14 in a paper I'm recalling from memory."

Joint limits (a real, if simplified, set of physical hinge-joint limits
for hip/knee/ankle) are enforced by clamping after each iteration -
standard practice for numerical IK, and REQUIRED so that IK doesn't
"solve" a target by bending the knee backward, which is not a real knee.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import forward_kinematics, LegParams


# Reasonable physical joint limits, radians. Knee is the important one:
# a real knee only flexes one way (0 = straight, positive = bent forward
# for a leg hanging down) - this is what actually rules out the classic
# "IK solves it by bending the knee the wrong way" failure mode.
JOINT_LIMITS_RAD = np.array([
    [-0.6, 0.6],    # hip_yaw
    [-0.5, 0.5],    # hip_roll
    [-1.8, 1.2],    # hip_pitch (forward/back swing)
    [0.0, 2.5],     # knee_pitch (0=straight, positive=bent - CANNOT go negative)
    [-1.0, 0.8],    # ankle_pitch
    [-0.5, 0.5],    # ankle_roll
])


@dataclass
class IKResult:
    joint_angles: np.ndarray
    converged: bool
    residual_error: float
    within_joint_limits: bool
    iterations: int


def _pose_error(target_pos, target_R, pos, R) -> np.ndarray:
    pos_err = target_pos - pos
    R_err = target_R @ R.T
    cos_angle = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_angle)
    if angle < 1e-8:
        rot_err = np.zeros(3)
    else:
        axis = np.array([R_err[2, 1] - R_err[1, 2],
                          R_err[0, 2] - R_err[2, 0],
                          R_err[1, 0] - R_err[0, 1]])
        rot_err = (angle / (2.0 * np.sin(angle))) * axis
    return np.concatenate([pos_err, rot_err])


def _numerical_jacobian(q, hip_position, leg, eps=1e-6) -> np.ndarray:
    pos0, R0 = forward_kinematics(q, hip_position, leg)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = q.copy()
        dq[i] += eps
        pos1, R1 = forward_kinematics(dq, hip_position, leg)
        J[:3, i] = (pos1 - pos0) / eps
        dR = R1 @ R0.T
        w = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2.0 * eps)
        J[3:, i] = w
    return J


def compute_jacobian(q: np.ndarray, hip_position: np.ndarray, leg: LegParams) -> np.ndarray:
    """Public wrapper around the same numerical Jacobian used internally by
    inverse_kinematics() - reused by dynamics/leg_dynamics.py for the
    Jacobian-transpose torque calculation, so the IK solver and the torque
    analysis are guaranteed to agree on the leg's actual kinematics (not
    two independently-derived, potentially-inconsistent Jacobians)."""
    return _numerical_jacobian(q, hip_position, leg)


def inverse_kinematics(target_pos: np.ndarray, target_R: np.ndarray, hip_position: np.ndarray,
                        leg: LegParams, initial_guess: np.ndarray | None = None,
                        max_iters: int = 150, tol: float = 1e-6, damping: float = 1e-2
                        ) -> IKResult:
    q = initial_guess.copy() if initial_guess is not None else np.array([0.0, 0.0, -0.2, 0.4, -0.2, 0.0])
    q = np.clip(q, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

    err_norm = np.inf
    it = 0
    for it in range(1, max_iters + 1):
        pos, R = forward_kinematics(q, hip_position, leg)
        err = _pose_error(target_pos, target_R, pos, R)
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break
        J = _numerical_jacobian(q, hip_position, leg)
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping ** 2 * np.eye(6), err)
        q = q + dq
        q = np.clip(q, JOINT_LIMITS_RAD[:, 0], JOINT_LIMITS_RAD[:, 1])

    within_limits = bool(np.all(q >= JOINT_LIMITS_RAD[:, 0]) and np.all(q <= JOINT_LIMITS_RAD[:, 1]))
    return IKResult(joint_angles=q, converged=err_norm < tol, residual_error=err_norm,
                     within_joint_limits=within_limits, iterations=it)
