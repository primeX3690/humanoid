"""
simulation/torque_analysis.py

Ties together the CoM trajectory (simulation/walk_simulator.py), the
joint-angle trajectory (simulation/joint_trajectory_generator.py), and
the quasi-static torque model (dynamics/leg_dynamics.py) to answer, for
an ENTIRE walk, the question SCOPE.md flagged as open: does a real,
specific, buyable actuator (actuators/motor_specs.py's Dynamixel MX-106)
have enough torque and speed margin for this gait?

Stance/swing assignment per timestep: during single support, the
STANCE foot bears the robot's FULL weight (stance_leg_required_torque);
the SWING leg only needs to overcome its own limb inertia
(swing_leg_inertial_torque) - much smaller. During double support, load
is split 50/50 between both feet (a standard, stated simplifying
assumption for a symmetric stance - a real robot's actual split depends
on the ZMP's position within the support polygon, which this quasi-
static double-support model does not resolve further).

CoM acceleration and joint velocities/accelerations are obtained via
central finite differences of the already-computed position/angle
trajectories (consistent with how a real system would estimate velocity/
acceleration from position sensors, not an idealized analytic value).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planning.footstep_planner import GaitParams
from simulation.walk_simulator import WalkResult
from simulation.joint_trajectory_generator import JointTrajectoryResult
from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import stance_leg_required_torque, RobotMassParams
from dynamics.rigid_body_leg import swing_leg_rigid_body_torque
from actuators.motor_specs import MotorSpec, DYNAMIXEL_MX106, check_feasibility, FeasibilityResult


def _central_diff(x: np.ndarray, dt: float) -> np.ndarray:
    """Central finite difference, with one-sided differences at the
    endpoints - standard numerical-differentiation practice, not just
    np.diff (which would misalign array length)."""
    d = np.zeros_like(x)
    d[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    d[0] = (x[1] - x[0]) / dt
    d[-1] = (x[-1] - x[-2]) / dt
    return d


@dataclass
class TorqueAnalysisResult:
    t: np.ndarray
    left_hip_pitch_torque: np.ndarray
    left_knee_torque: np.ndarray
    right_hip_pitch_torque: np.ndarray
    right_knee_torque: np.ndarray
    left_hip_pitch_speed: np.ndarray
    left_knee_speed: np.ndarray
    right_hip_pitch_speed: np.ndarray
    right_knee_speed: np.ndarray
    hip_feasibility: FeasibilityResult
    knee_feasibility: FeasibilityResult


def analyze_torque_feasibility(walk: WalkResult, jt: JointTrajectoryResult, gait: GaitParams,
                                leg: LegParams | None = None,
                                robot_mass: RobotMassParams | None = None,
                                motor: MotorSpec = DYNAMIXEL_MX106,
                                hip_lateral_offset_m: float | None = None) -> TorqueAnalysisResult:
    leg = leg or LegParams()
    robot_mass = robot_mass or RobotMassParams()
    hip_offset = hip_lateral_offset_m if hip_lateral_offset_m is not None else gait.step_width_m / 2.0
    dt = gait.dt

    com_ax = _central_diff(_central_diff(walk.com_x, dt), dt)
    com_ay = _central_diff(_central_diff(walk.com_y, dt), dt)

    left_qdot = np.gradient(jt.left_joint_angles, dt, axis=0)
    left_qddot = np.gradient(left_qdot, dt, axis=0)
    right_qdot = np.gradient(jt.right_joint_angles, dt, axis=0)
    right_qddot = np.gradient(right_qdot, dt, axis=0)

    n = len(jt.t)
    left_tau = np.zeros((n, 6))
    right_tau = np.zeros((n, 6))

    # which foot is in single support, and for which side, at each timestep
    stance_side = np.full(n, "double")  # default: double support (load split)
    for step in walk.footsteps:
        mask = (jt.t >= step.start_time) & (jt.t < step.end_time)
        # during footstep i's swing, the OPPOSITE side is the stance leg
        opposite = "left" if step.side == "right" else "right"
        stance_side[mask] = opposite

    for k in range(n):
        com_accel = np.array([com_ax[k], com_ay[k], 0.0])
        hip_pos_left = np.array([walk.com_x[k], walk.com_y[k] + hip_offset, leg.thigh_length_m + leg.shank_length_m])
        hip_pos_right = np.array([walk.com_x[k], walk.com_y[k] - hip_offset, leg.thigh_length_m + leg.shank_length_m])

        if stance_side[k] == "left":
            left_tau[k] = stance_leg_required_torque(jt.left_joint_angles[k], hip_pos_left, leg,
                                                       robot_mass, com_accel)
            right_tau[k] = swing_leg_rigid_body_torque(jt.right_joint_angles[k], right_qdot[k],
                                                        right_qddot[k], leg, robot_mass)
        elif stance_side[k] == "right":
            right_tau[k] = stance_leg_required_torque(jt.right_joint_angles[k], hip_pos_right, leg,
                                                        robot_mass, com_accel)
            left_tau[k] = swing_leg_rigid_body_torque(jt.left_joint_angles[k], left_qdot[k],
                                                       left_qddot[k], leg, robot_mass)
        else:  # double support: split load 50/50
            half_mass = RobotMassParams(total_mass_kg=robot_mass.total_mass_kg / 2.0)
            left_tau[k] = stance_leg_required_torque(jt.left_joint_angles[k], hip_pos_left, leg,
                                                       half_mass, com_accel)
            right_tau[k] = stance_leg_required_torque(jt.right_joint_angles[k], hip_pos_right, leg,
                                                        half_mass, com_accel)

    hip_torques = np.concatenate([left_tau[:, 2], right_tau[:, 2]])
    hip_speeds = np.concatenate([left_qdot[:, 2], right_qdot[:, 2]])
    knee_torques = np.concatenate([left_tau[:, 3], right_tau[:, 3]])
    knee_speeds = np.concatenate([left_qdot[:, 3], right_qdot[:, 3]])

    hip_feas = check_feasibility(hip_torques, hip_speeds, motor)
    knee_feas = check_feasibility(knee_torques, knee_speeds, motor)

    return TorqueAnalysisResult(
        t=jt.t,
        left_hip_pitch_torque=left_tau[:, 2], left_knee_torque=left_tau[:, 3],
        right_hip_pitch_torque=right_tau[:, 2], right_knee_torque=right_tau[:, 3],
        left_hip_pitch_speed=left_qdot[:, 2], left_knee_speed=left_qdot[:, 3],
        right_hip_pitch_speed=right_qdot[:, 2], right_knee_speed=right_qdot[:, 3],
        hip_feasibility=hip_feas, knee_feasibility=knee_feas,
    )
