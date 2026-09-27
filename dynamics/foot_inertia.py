"""
dynamics/foot_inertia.py

Closes the honest, explicitly-stated scope limit left open by
dynamics/rigid_body_leg_6dof.py: "the ankle joints (pitch, roll)
contribute ZERO columns/rows to this mass matrix - not because of an
approximation, but because this module only models the THIGH and SHANK
as rigid bodies... A true 6-body-inertia treatment needs a foot link".
This module adds that third rigid body.

METHOD: identical technique to rigid_body_leg_6dof.py (geometric
Jacobian from joint axes) - the foot's COM Jacobian uses ALL 6 joints
as active columns (unlike thigh/shank, the foot is distal to every
joint in the chain, so every joint moves it), added to the ALREADY-
VERIFIED thigh+shank mass matrix, gravity vector, and Coriolis matrix
from rigid_body_leg_6dof.py rather than re-deriving the whole 6x6
system from scratch. This is valid because mass matrices, gravity
vectors, and (since the Christoffel-symbol formula is LINEAR in the
mass matrix entries) Coriolis matrices from independent rigid bodies
in the same kinematic chain simply ADD:

    M_total = M_thigh_shank + M_foot
    G_total = G_thigh_shank + G_foot
    C_total = C_thigh_shank + C_foot

REAL BUG FOUND WHILE BUILDING THIS (see docs/BUGS_FOUND.md): getting
the foot's Jacobian right required CORRECT world-frame origins for the
ankle_pitch/ankle_roll joints. rigid_body_leg_6dof.py's internal
`_joint_frames()` had been setting those origins to `knee_pos` (and its
own comment incorrectly claimed the ankle joints were "co-located with
the hip") - completely harmless for the thigh/shank-only model, since
neither _THIGH_JOINTS nor _SHANK_JOINTS ever uses index 4 or 5's
origin, but exactly wrong for a foot Jacobian, which needs all 6.
Fixed directly in rigid_body_leg_6dof.py (ankle origins now computed as
the actual ankle position), verified to cause zero change to any of
that module's own (already-passing) tests - a true dead-code bug, not
a behavior change.

FOOT SHAPE/INERTIA MODEL (stated assumptions, not measured data): the
foot is treated as a uniform rectangular box of dimensions
`leg.foot_length_m` x `leg.foot_width_m` x `leg.foot_height_m`
(typical adult scale defaults - see kinematics/leg_fk.py), giving the
standard box-inertia formulas about its own COM. Its COM is placed at a
stated fraction forward of the ankle and half the foot's height below
it - a reasonable, explicitly-approximate placement (real foot COM
location varies with footwear/structure), not measured anthropometric
data the way the thigh/shank LENGTHS and MASS FRACTIONS are (de Leva,
1996 - dynamics/leg_dynamics.py). Foot mass fraction (1.37% of total
body mass) IS the real de Leva (1996) male regression value.

HONEST, STATED SCOPE LIMIT (not fixed here): this still does not model
the ANKLE's own rotational compliance/actuator-side effects (see
docs/SCOPE.md's separate "no actuator dynamics" gap), and it does not
model ground-contact/sole pressure distribution at all - it only adds
the foot's own rigid-body inertia to the swing-leg dynamics equation,
same scope as rigid_body_leg_6dof.py had for thigh/shank.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams, forward_kinematics
from dynamics.leg_dynamics import RobotMassParams, FOOT_MASS_FRACTION
from dynamics.rigid_body_leg_6dof import (
    N_DOF, GRAVITY, _joint_frames, mass_matrix as mass_matrix_thigh_shank,
    gravity_torque as gravity_torque_thigh_shank, coriolis_matrix as coriolis_matrix_thigh_shank,
    link_inertias,
)

# Stated placement assumptions (see module docstring) - NOT measured data
_FOOT_COM_FORWARD_FRACTION = 0.35   # of foot_length_m, forward of the ankle
_FOOT_COM_VERTICAL_FRACTION = 0.5   # of foot_height_m, below the ankle


@dataclass
class FootInertia:
    mass_kg: float
    com_forward_m: float    # forward offset from ankle, in the ankle's own oriented frame
    com_down_m: float       # downward offset from ankle, in the ankle's own oriented frame
    i_roll_kgm2: float      # about the foot's own x (roll) axis, through its COM
    i_pitch_kgm2: float     # about the foot's own y (pitch) axis, through its COM
    i_yaw_kgm2: float       # about the foot's own z (yaw) axis, through its COM


def foot_inertia(leg: LegParams, robot_mass: RobotMassParams) -> FootInertia:
    m = FOOT_MASS_FRACTION * robot_mass.total_mass_kg
    L, W, H = leg.foot_length_m, leg.foot_width_m, leg.foot_height_m
    return FootInertia(
        mass_kg=m,
        com_forward_m=_FOOT_COM_FORWARD_FRACTION * L,
        com_down_m=_FOOT_COM_VERTICAL_FRACTION * H,
        i_roll_kgm2=m * (W ** 2 + H ** 2) / 12.0,
        i_pitch_kgm2=m * (L ** 2 + H ** 2) / 12.0,
        i_yaw_kgm2=m * (L ** 2 + W ** 2) / 12.0,
    )


def _foot_jacobians(q: np.ndarray, hip_position: np.ndarray, leg: LegParams, foot: FootInertia
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Jv_foot, Jw_foot, R_ankle), each Jacobian shaped (3, 6)
    with ALL 6 columns active (the foot is distal to every joint)."""
    origins, axes, _thigh_com, _shank_com, _R_hip, _R_knee = _joint_frames(q, hip_position, leg)
    ankle_pos, R_ankle = forward_kinematics(q, hip_position, leg)
    foot_com = ankle_pos + R_ankle @ np.array([foot.com_forward_m, 0.0, -foot.com_down_m])

    Jv_foot = np.zeros((3, N_DOF))
    Jw_foot = np.zeros((3, N_DOF))
    for i in range(N_DOF):
        Jv_foot[:, i] = np.cross(axes[i], foot_com - origins[i])
        Jw_foot[:, i] = axes[i]

    return Jv_foot, Jw_foot, R_ankle


def _body_inertia_tensor(foot: FootInertia) -> np.ndarray:
    return np.diag([foot.i_roll_kgm2, foot.i_pitch_kgm2, foot.i_yaw_kgm2])


def foot_mass_matrix_term(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                           foot: FootInertia) -> np.ndarray:
    Jv, Jw, R_ankle = _foot_jacobians(q, hip_position, leg, foot)
    I_world = R_ankle @ _body_inertia_tensor(foot) @ R_ankle.T
    return foot.mass_kg * (Jv.T @ Jv) + Jw.T @ I_world @ Jw


def foot_gravity_term(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                       foot: FootInertia) -> np.ndarray:
    Jv, _, _ = _foot_jacobians(q, hip_position, leg, foot)
    return foot.mass_kg * GRAVITY * Jv[2, :]


def mass_matrix_with_foot(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                           robot_mass: RobotMassParams) -> np.ndarray:
    thigh, shank = link_inertias(leg, robot_mass)
    foot = foot_inertia(leg, robot_mass)
    return (mass_matrix_thigh_shank(q, hip_position, leg, thigh, shank)
            + foot_mass_matrix_term(q, hip_position, leg, foot))


def gravity_torque_with_foot(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                              robot_mass: RobotMassParams) -> np.ndarray:
    thigh, shank = link_inertias(leg, robot_mass)
    foot = foot_inertia(leg, robot_mass)
    return (gravity_torque_thigh_shank(q, hip_position, leg, thigh, shank)
            + foot_gravity_term(q, hip_position, leg, foot))


def coriolis_matrix_with_foot(q: np.ndarray, qdot: np.ndarray, hip_position: np.ndarray,
                               leg: LegParams, robot_mass: RobotMassParams, eps: float = 1e-6
                               ) -> np.ndarray:
    """C_total = C_thigh_shank + C_foot - valid because the Christoffel-
    symbol formula is linear in the mass matrix (see module docstring)."""
    thigh, shank = link_inertias(leg, robot_mass)
    foot = foot_inertia(leg, robot_mass)
    C_thigh_shank = coriolis_matrix_thigh_shank(q, qdot, hip_position, leg, thigh, shank, eps=eps)

    M0 = foot_mass_matrix_term(q, hip_position, leg, foot)
    dM = []
    for k in range(N_DOF):
        qk = q.copy()
        qk[k] += eps
        dM.append((foot_mass_matrix_term(qk, hip_position, leg, foot) - M0) / eps)

    C_foot = np.zeros((N_DOF, N_DOF))
    for i in range(N_DOF):
        for j in range(N_DOF):
            s = 0.0
            for k in range(N_DOF):
                s += (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qdot[k]
            C_foot[i, j] = 0.5 * s

    return C_thigh_shank + C_foot


def swing_leg_rigid_body_torque_with_foot(q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray,
                                           hip_position: np.ndarray, leg: LegParams,
                                           robot_mass: RobotMassParams) -> np.ndarray:
    """Drop-in replacement for
    rigid_body_leg_6dof.swing_leg_rigid_body_torque_6dof with the foot's
    own inertia now included - ankle rows are no longer structurally
    zero (see tests/test_foot_inertia.py)."""
    M = mass_matrix_with_foot(q, hip_position, leg, robot_mass)
    C = coriolis_matrix_with_foot(q, qdot, hip_position, leg, robot_mass)
    G = gravity_torque_with_foot(q, hip_position, leg, robot_mass)
    return M @ qddot + C @ qdot + G
