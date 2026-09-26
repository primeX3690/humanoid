"""
dynamics/rigid_body_leg.py

Full rigid-body dynamics (the real "manipulator equation"
tau = M(q) qddot + C(q, qdot) qdot + G(q)) for the leg's 2 dominant
sagittal-plane joints (hip_pitch, knee_pitch), replacing
leg_dynamics.py's earlier crude lumped-point-mass proxy for the SWING
leg's own inertial torque. Each link (thigh, shank) is modeled as a
uniform rod: mass at its real anthropometric fraction (de Leva 1996),
center of mass at the geometric midpoint, moment of inertia about its
own center I = (1/12) m L^2 (the standard uniform-rod formula) - a real,
if simplified, rigid-body model (a real leg's mass isn't uniformly
distributed along its length, but this is the standard first-order
model used before higher-fidelity CAD-based inertia is available, and
is the same simplification implicitly used by the parallel-axis-theorem
check below).

METHOD (why derived this way rather than hand-written recursive
Newton-Euler cross-products): the mass matrix M(q) is built from each
link's center-of-mass position Jacobian, which is derived ANALYTICALLY
here using one verified identity (see the module-level derivation note
below) - not recursive 3D cross-product bookkeeping, which is easy to
get a sign wrong in without a reference to check against (the same
reasoning kinematics/leg_ik.py's docstring gives for using numerical IK
instead of a memorized closed-form solution). The Coriolis/centrifugal
matrix C(q, qdot) is then obtained from M(q)'s standard Christoffel-
symbol formula, using ONE well-conditioned single (not doubly-nested)
numerical differentiation of the already-analytic M(q).

Rotation-derivative identity used throughout (verified numerically
against finite differences before use, see docs/BUGS_FOUND.md):
for a 2D rotation R2(theta) = [[cos,sin],[-sin,cos]] (matching this
project's leg_fk.py Ry-rotation convention restricted to the x-z
plane), d/dtheta [R2(theta) @ v] = R2(theta) @ perp(v), where
perp((vx, vz)) = (vz, -vx).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import RobotMassParams, THIGH_MASS_FRACTION, SHANK_MASS_FRACTION

GRAVITY = 9.81


def _R2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [-s, c]])


def _perp(v: np.ndarray) -> np.ndarray:
    return np.array([v[1], -v[0]])


@dataclass
class LinkInertia:
    mass_kg: float
    length_m: float
    com_distance_m: float  # from proximal joint, = length/2 for a uniform rod
    inertia_kgm2: float    # about own COM, = (1/12)*m*L^2 for a uniform rod


def link_inertias(leg: LegParams, robot_mass: RobotMassParams) -> tuple[LinkInertia, LinkInertia]:
    m1 = THIGH_MASS_FRACTION * robot_mass.total_mass_kg
    m2 = SHANK_MASS_FRACTION * robot_mass.total_mass_kg
    thigh = LinkInertia(mass_kg=m1, length_m=leg.thigh_length_m,
                         com_distance_m=leg.thigh_length_m / 2.0,
                         inertia_kgm2=m1 * leg.thigh_length_m ** 2 / 12.0)
    shank = LinkInertia(mass_kg=m2, length_m=leg.shank_length_m,
                         com_distance_m=leg.shank_length_m / 2.0,
                         inertia_kgm2=m2 * leg.shank_length_m ** 2 / 12.0)
    return thigh, shank


def _com_positions(q1: float, q2: float, thigh: LinkInertia, shank: LinkInertia
                    ) -> tuple[np.ndarray, np.ndarray]:
    thigh_com = _R2(q1) @ np.array([0.0, -thigh.com_distance_m])
    shank_com = _R2(q1) @ np.array([0.0, -thigh.length_m]) + _R2(q1 + q2) @ np.array([0.0, -shank.com_distance_m])
    return thigh_com, shank_com


def _com_jacobians(q1: float, q2: float, thigh: LinkInertia, shank: LinkInertia
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Analytic d(COM)/dq for each link, using the verified rotation-
    derivative identity - see module docstring."""
    d_thigh_dq1 = _R2(q1) @ _perp(np.array([0.0, -thigh.com_distance_m]))
    d_thigh_dq2 = np.zeros(2)
    J1 = np.column_stack([d_thigh_dq1, d_thigh_dq2])

    d_shank_dq1 = (_R2(q1) @ _perp(np.array([0.0, -thigh.length_m])) +
                   _R2(q1 + q2) @ _perp(np.array([0.0, -shank.com_distance_m])))
    d_shank_dq2 = _R2(q1 + q2) @ _perp(np.array([0.0, -shank.com_distance_m]))
    J2 = np.column_stack([d_shank_dq1, d_shank_dq2])
    return J1, J2


def mass_matrix(q1: float, q2: float, thigh: LinkInertia, shank: LinkInertia) -> np.ndarray:
    J1, J2 = _com_jacobians(q1, q2, thigh, shank)
    M = thigh.mass_kg * (J1.T @ J1) + shank.mass_kg * (J2.T @ J2)
    M += thigh.inertia_kgm2 * np.array([[1.0, 0.0], [0.0, 0.0]])  # omega1 = qdot1
    M += shank.inertia_kgm2 * np.array([[1.0, 1.0], [1.0, 1.0]])  # omega2 = qdot1+qdot2
    return M


def gravity_torque(q1: float, q2: float, thigh: LinkInertia, shank: LinkInertia) -> np.ndarray:
    J1, J2 = _com_jacobians(q1, q2, thigh, shank)
    # G_i = d(PE)/dq_i = mass * g * d(z_com)/dq_i  (z is the 2nd row of the Jacobian)
    return thigh.mass_kg * GRAVITY * J1[1, :] + shank.mass_kg * GRAVITY * J2[1, :]


def coriolis_matrix(q1: float, q2: float, qdot: np.ndarray, thigh: LinkInertia, shank: LinkInertia,
                     eps: float = 1e-6) -> np.ndarray:
    """Standard Christoffel-symbol formula: C_ij = 0.5 * sum_k
    (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qdot_k, using a single
    (not doubly-nested) numerical differentiation of the already-analytic
    mass_matrix()."""
    M0 = mass_matrix(q1, q2, thigh, shank)
    dM_dq1 = (mass_matrix(q1 + eps, q2, thigh, shank) - M0) / eps
    dM_dq2 = (mass_matrix(q1, q2 + eps, thigh, shank) - M0) / eps
    dM = [dM_dq1, dM_dq2]

    C = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            s = 0.0
            for k in range(2):
                s += (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qdot[k]
            C[i, j] = 0.5 * s
    return C


def swing_leg_rigid_body_torque(q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray,
                                 leg: LegParams, robot_mass: RobotMassParams) -> np.ndarray:
    """Full rigid-body torque for hip_pitch (index 2) and knee_pitch
    (index 3) of the 6-DOF joint vector - replaces
    leg_dynamics.swing_leg_inertial_torque's crude lumped-mass proxy for
    these two dominant joints. Other 4 joints (hip yaw/roll, ankle
    pitch/roll) are not modeled by this 2-link planar treatment - their
    inertial loads are much smaller for a mostly-sagittal gait and are
    left at the crude-proxy/zero level (see docs/SCOPE.md)."""
    q1, q2 = q[2], q[3]
    qdot12 = np.array([qdot[2], qdot[3]])
    qddot12 = np.array([qddot[2], qddot[3]])
    thigh, shank = link_inertias(leg, robot_mass)

    M = mass_matrix(q1, q2, thigh, shank)
    C = coriolis_matrix(q1, q2, qdot12, thigh, shank)
    G = gravity_torque(q1, q2, thigh, shank)

    tau12 = M @ qddot12 + C @ qdot12 + G

    tau = np.zeros(6)
    tau[2], tau[3] = tau12
    return tau
