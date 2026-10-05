"""
simulation/joint_trajectory_generator.py

The final integration step: takes the CoM trajectory from
simulation/walk_simulator.py and the per-foot target trajectories from
planning/footstep_planner.py's foot_target_trajectories(), and solves
leg IK (kinematics/leg_ik.py) at every timestep for both legs, producing
the actual joint-angle trajectories (12 DOF total: 6 per leg) a real
robot's joint controllers would track.

Hip positions are derived from the CoM by offsetting laterally by half
the stance width (a simplifying assumption - real robots' hip-to-hip
width and foot stance width aren't necessarily identical - documented
in docs/SCOPE.md) at the LIPM's fixed CoM height.

Performance/correctness note: IK is WARM-STARTED from the previous
timestep's solution rather than re-solved from a fixed initial guess
every time. This is not just a speed optimization - it's the standard
real-world approach (a real robot's IK runs at its previous joint state
every control cycle, not from scratch), and it matters for correctness
here too: warm-starting keeps the solver in the same branch of the
otherwise-multi-valued IK (avoiding a spurious "knee suddenly flips to
a different valid solution" discontinuity between adjacent timesteps).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams, forward_kinematics
from kinematics.leg_ik import inverse_kinematics, IKResult
from planning.footstep_planner import GaitParams, foot_target_trajectories, heading_profile
from simulation.walk_simulator import WalkResult

IDENTITY_R = np.eye(3)  # target foot orientation: always flat/level (see docs/SCOPE.md)


@dataclass
class JointTrajectoryResult:
    t: np.ndarray
    left_joint_angles: np.ndarray   # (n, 6)
    right_joint_angles: np.ndarray  # (n, 6)
    left_ik_converged: np.ndarray   # (n,) bool
    right_ik_converged: np.ndarray  # (n,) bool
    left_foot_xyz: np.ndarray       # (n, 3) actual FK output, for verifying against the target
    right_foot_xyz: np.ndarray


def generate_joint_trajectory(walk: WalkResult, gait: GaitParams,
                               leg: LegParams | None = None,
                               hip_lateral_offset_m: float | None = None,
                               hip_height_profile_m: np.ndarray | None = None) -> JointTrajectoryResult:
    """hip_height_profile_m: optional per-timestep world-frame hip height
    (shape (n,), matching walk.t) - e.g. from
    dynamics/terrain_adaptive_com_height.hip_height_profile(). Purely
    additive: omitted (the default), this reproduces the exact original
    behavior (one constant height for the whole walk) unchanged - see
    dynamics/terrain_adaptive_com_height.py for why that constant is
    actually the wrong thing to use once terrain has nonzero height."""
    leg = leg or LegParams()
    hip_offset = hip_lateral_offset_m if hip_lateral_offset_m is not None else gait.step_width_m / 2.0

    t, left_target, right_target = foot_target_trajectories(gait, walk.footsteps)
    n = len(t)
    if hip_height_profile_m is not None:
        assert len(hip_height_profile_m) == n, (
            f"hip_height_profile_m length {len(hip_height_profile_m)} must match "
            f"the walk's own timebase length {n}"
        )
        com_z = hip_height_profile_m
    else:
        com_z = np.full(n, _lipm_height_from_walk(walk))

    left_q = np.zeros((n, 6))
    right_q = np.zeros((n, 6))
    left_ok = np.zeros(n, dtype=bool)
    right_ok = np.zeros(n, dtype=bool)
    left_fk_xyz = np.zeros((n, 3))
    right_fk_xyz = np.zeros((n, 3))

    left_guess = None
    right_guess = None
    t_head, heading = heading_profile(gait, walk.footsteps)

    for k in range(n):
        c, s = np.cos(heading[k]), np.sin(heading[k])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        # hip lateral offset rotates WITH the body heading (feet stay side-by-side relative to
        # the direction of travel, not fixed to world +y) - at heading=0 this is exactly the
        # original hip_pos_left/right below, unchanged
        offset_l = Rz @ np.array([0.0, hip_offset, 0.0])
        offset_r = Rz @ np.array([0.0, -hip_offset, 0.0])
        hip_pos_left = np.array([walk.com_x[k], walk.com_y[k], 0.0]) + offset_l + np.array([0, 0, com_z[k]])
        hip_pos_right = np.array([walk.com_x[k], walk.com_y[k], 0.0]) + offset_r + np.array([0, 0, com_z[k]])

        res_l: IKResult = inverse_kinematics(left_target[k], Rz, hip_pos_left, leg,
                                              initial_guess=left_guess)
        res_r: IKResult = inverse_kinematics(right_target[k], Rz, hip_pos_right, leg,
                                              initial_guess=right_guess)
        left_q[k] = res_l.joint_angles
        right_q[k] = res_r.joint_angles
        left_ok[k] = res_l.converged
        right_ok[k] = res_r.converged
        left_guess = res_l.joint_angles
        right_guess = res_r.joint_angles

        left_fk_xyz[k], _ = forward_kinematics(res_l.joint_angles, hip_pos_left, leg)
        right_fk_xyz[k], _ = forward_kinematics(res_r.joint_angles, hip_pos_right, leg)

    return JointTrajectoryResult(t=t, left_joint_angles=left_q, right_joint_angles=right_q,
                                  left_ik_converged=left_ok, right_ik_converged=right_ok,
                                  left_foot_xyz=left_fk_xyz, right_foot_xyz=right_fk_xyz)


def _lipm_height_from_walk(walk: WalkResult) -> float:
    """The CoM height isn't stored in WalkResult (the LIPM treats it as a
    fixed parameter, not a state variable) - recovered here from the
    default used by simulate_walk() unless the caller used a custom one.
    Kept as an explicit, named function rather than a silent magic number
    so a mismatch is easy to spot and fix if simulate_walk's default ever
    changes."""
    from dynamics.lipm import LIPMParams
    return LIPMParams().com_height_m
