"""
Loco-manipulation: carrying things while walking.

 * payload_com_shift   - how a payload held by the hands moves the whole-body CoM (feed it to the LIPM reference)
 * max_static_payload  - heaviest payload a given arm pose can hold, from the shoulder/elbow torque limits (arm Jacobian transpose)
 * bimanual_box_ik     - both arms grasp a box at fixed hand poses relative to it (closed-chain constraint), IK for both
Pure NumPy (uses kinematics.arm_kinematics.Arm).
"""
from __future__ import annotations
import numpy as np
from kinematics.arm_kinematics import Arm

G = 9.81


def payload_com_shift(robot_mass, com_body, payload_mass, payload_pos):
    """New whole-body CoM and the shift relative to the body CoM (all in the same frame)."""
    com_body = np.asarray(com_body, float); payload_pos = np.asarray(payload_pos, float)
    new = (robot_mass * com_body + payload_mass * payload_pos) / (robot_mass + payload_mass)
    return new, new - com_body


def static_joint_torque(arm: Arm, q, force_world_on_tcp, link_masses=(1.0, 0.85, 0.8)):
    """Motor torques (7,) that hold an external force F ON the hand (e.g. a payload weight (0,0,-m g)) plus the arm's own link
    weights as point masses at the link mid-points (default masses ~ model.humanoid_model m_uarm/m_farm/m_hand incl. joint housings -
    ASSUMED; replace with CAD values). A link's weight only loads the joints PROXIMAL to it."""
    from manipulation.arm_collision import arm_capsules
    q = np.asarray(q, float)
    J = arm.jacobian(q)
    tau = J[:3].T @ (-np.asarray(force_world_on_tcp, float))
    frames, _ = arm._chain(q)
    for cap, m, n_loaded in zip(arm_capsules(arm, q), link_masses, (3, 4, 7)):
        mid = 0.5 * (cap.a + cap.b)
        for i in range(n_loaded):
            o, a_ = frames[i]
            tau[i] += np.cross(a_, mid - o) @ np.array([0.0, 0.0, m * G])      # (a x r).W with W=(0,0,+m g): torque the motor must supply
    return tau


def max_static_payload(arm: Arm, q, tau_limits, direction=np.array([0, 0, -1.0]), lo=0.0, hi=20.0):
    """Largest payload mass (kg) whose weight, applied at the TCP, keeps every joint within tau_limits (bisection)."""
    tau_limits = np.broadcast_to(np.asarray(tau_limits, float), (7,))
    for _ in range(30):
        m = 0.5 * (lo + hi)
        tau = static_joint_torque(arm, q, direction * m * G)          # external load ON the hand = weight, pointing along `direction`
        if np.all(np.abs(tau) <= tau_limits): lo = m
        else: hi = m
    return lo


def bimanual_box_ik(arms: dict, T_box_world, hand_offset_y, qL0, qR0, grasp_x=0.0, grasp_z=0.0):
    """Left/right TCP at box_y = +/- hand_offset_y (hands squeezing the box sides), z axis horizontal pointing at the box centre.
    Returns (qL, qR, converged, relative-pose error). T_box_world is expressed in the arms' common (torso) frame."""
    out = {}; poses = {}
    for side, sgn in (("L", +1.0), ("R", -1.0)):
        p_local = np.array([grasp_x, sgn * hand_offset_y, grasp_z])
        # TCP sits at -z of the palm, so the palm/wrist must be OUTSIDE the box: z points away from the box centre (approach = toward it)
        z_loc = np.array([0, sgn, 0.0]); x_loc = np.array([1.0, 0, 0]); y_loc = np.cross(z_loc, x_loc)
        R_loc = np.column_stack([x_loc, y_loc, z_loc])
        T = np.eye(4); T[:3, :3] = T_box_world[:3, :3] @ R_loc; T[:3, 3] = T_box_world[:3, 3] + T_box_world[:3, :3] @ p_local
        poses[side] = T
        out[side] = arms[side].ik(T, qL0 if side == "L" else qR0)
    ok = out["L"][1] and out["R"][1]
    err = max(out["L"][2], out["R"][2])
    return out["L"][0], out["R"][0], bool(ok), float(err), poses
