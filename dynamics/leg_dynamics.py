"""
dynamics/leg_dynamics.py

Answers the question SCOPE.md flagged as open: "whether a real robot's
actual motors can produce the torques this plan implies... is a
completely separate, real question this model doesn't answer yet."

Method: quasi-static Jacobian-transpose torque estimation. During single
support, the stance leg's foot experiences a ground reaction force R
that must support the robot's weight PLUS react its CoM acceleration
(Newton's second law applied to the whole robot as one point mass at the
CoM - the same lumped-mass assumption the LIPM itself already makes, so
this is consistent with, not an extra assumption beyond, the rest of the
model):

    R = M * (g_vector + a_com)

The joint torques needed to statically react this force (i.e. what the
actuators must hold against, ignoring the leg LINKS' OWN inertia/weight
- a real, stated simplification, not hidden) follow from the principle
of virtual work: tau = J^T @ R, where J is the SAME leg Jacobian
kinematics/leg_ik.py's IK solver uses (see compute_jacobian() - reusing
it, rather than deriving a second independent Jacobian, guarantees the
two are consistent).

Segment masses use real anthropometric regression data (de Leva, 1996,
male parameters): thigh = 14.16% of total body mass, shank = 4.33% -
used only to report the SWING leg's own inertial torque contribution
(much smaller than the stance leg's weight-bearing torque, but included
rather than assumed away at zero).

Honest limitation (stated, not hidden): this is quasi-static +
point-mass CoM dynamics, NOT full rigid-body inverse dynamics (no link
angular momentum, no Coriolis/centrifugal terms, no distributed link
inertia tensors). It answers the DOMINANT question (can the motor
survive holding the robot's own weight through the gait) but not the
complete one - see docs/SCOPE.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams
from kinematics.leg_ik import compute_jacobian

GRAVITY_VEC = np.array([0.0, 0.0, -9.81])

# de Leva (1996) male segment mass fraction regression parameters
THIGH_MASS_FRACTION = 0.1416
SHANK_MASS_FRACTION = 0.0433


@dataclass
class RobotMassParams:
    total_mass_kg: float = 25.0  # stated assumption: a compact DIY-scale humanoid platform


def stance_leg_required_torque(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                                robot_mass: RobotMassParams, com_accel: np.ndarray) -> np.ndarray:
    """Required joint torque (6,) for the STANCE leg (the one currently
    bearing the robot's full weight during single support) to hold the
    quasi-static ground-reaction force implied by the robot's total mass
    and current CoM acceleration.

    Derivation: Newton's second law for the whole robot, modeled (like
    the LIPM itself) as a single point mass at the CoM:
        M * a_com = F_gravity + R  =>  R = M * (a_com - g_vector)
    With g_vector = (0, 0, -9.81) pointing down, standing still
    (a_com = 0) gives R = M * (0, 0, +9.81) - a purely UPWARD reaction
    force of magnitude M*g, which is the correct, physically sane result
    (verified in tests/test_leg_dynamics.py, since an earlier version of
    this function had R's sign backwards and would have reported the
    ground PULLING the robot down)."""
    R = robot_mass.total_mass_kg * (com_accel - GRAVITY_VEC)
    J = compute_jacobian(q, hip_position, leg)
    wrench = np.concatenate([R, np.zeros(3)])  # pure force at the foot, no applied moment
    tau = J.T @ wrench
    return tau


def swing_leg_inertial_torque(q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray,
                               leg: LegParams, robot_mass: RobotMassParams) -> np.ndarray:
    """Rough (lumped end-point mass, not distributed-inertia) estimate of
    the swing leg's OWN torque requirement to accelerate itself - much
    smaller than the stance leg's weight-bearing torque for a typical
    walking speed, but included rather than silently treated as zero.
    Uses a simple lumped-mass approximation: thigh mass at the knee,
    shank+foot mass at the ankle, each requiring F=ma to accelerate along
    with the leg's motion. This is intentionally cruder than the stance-leg
    calculation above (a full treatment needs each link's own inertia
    tensor and the Coriolis terms this quasi-static model doesn't include -
    see the module's honest-limitation note)."""
    thigh_mass = THIGH_MASS_FRACTION * robot_mass.total_mass_kg
    shank_mass = SHANK_MASS_FRACTION * robot_mass.total_mass_kg
    # crude proxy: treat swing-leg acceleration requirement as proportional
    # to joint angular acceleration times an effective lever arm (thigh/shank
    # length), converted to an equivalent joint torque via F=ma at the
    # segment's approximate center of mass distance from its own joint.
    lever_thigh = leg.thigh_length_m / 2.0
    lever_shank = leg.shank_length_m / 2.0
    tau = np.zeros(6)
    tau[2] = thigh_mass * lever_thigh ** 2 * qddot[2]   # hip_pitch reacting thigh's own inertia
    tau[3] = shank_mass * lever_shank ** 2 * qddot[3]   # knee_pitch reacting shank's own inertia
    return tau
