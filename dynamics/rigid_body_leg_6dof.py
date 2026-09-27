"""
dynamics/rigid_body_leg_6dof.py

Extends dynamics/rigid_body_leg.py's 2-link SAGITTAL-plane rigid-body
model (hip_pitch, knee_pitch only) to the FULL 6-DOF joint chain
(hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll)
already defined by kinematics/leg_fk.py. This closes item #7 in
docs/SCOPE.md ("sagittal/lateral axes decoupled... real 3D coupling
effects... not modeled") for the LEG'S OWN inertial dynamics: hip
yaw/roll motion now genuinely couples into the thigh and shank's
inertial torques (gyroscopic / centrifugal cross-coupling), which the
2-link model could not represent at all (it simply didn't have those
joints).

WHAT THIS DOES NOT CLOSE (stated up front, not discovered later):
the ankle joints (pitch, roll) still contribute ZERO columns/rows to
this mass matrix - not because of an approximation, but because this
module (like rigid_body_leg.py before it) only models the THIGH and
SHANK as rigid bodies. The ankle joints are distal to both of those
bodies (they only reposition the FOOT, which has no inertia model
here - see docs/SCOPE.md item 6's "still no actuator dynamics" note
and the missing foot-mass fraction in dynamics/leg_dynamics.py). A
true 6-body-inertia treatment needs a foot link; that is a further,
separate step, not silently folded in here.

METHOD: standard robotics "geometric Jacobian from joint axes" -
Jv_col_i = z_i x (p_com - p_i), Jw_col_i = z_i, for every joint i
that lies between the base and the link in the kinematic chain (zero
column otherwise) - see Siciliano/Sciavicco/Villani/Oriolo,
"Robotics: Modelling, Planning and Control", Ch. 3. This was chosen
over hand-expanding 3D cross products link-by-link (easy to get a
sign wrong in, per rigid_body_leg.py's docstring reasoning) because
this formula is a single, textbook-standard closed form that is
independently checked against finite differences below
(test_rigid_body_leg_6dof.py::test_jacobians_match_finite_difference)
- the same "verify the one identity used everywhere" discipline the
2-link module used for its own rotation-derivative identity.

CONSISTENCY CHECK (the strongest evidence this is correct): with
hip_yaw = hip_roll = ankle_pitch = ankle_roll fixed at zero, this
module's hip_pitch/knee_pitch block of M(q), G(q), and C(q,qdot) is
verified to match rigid_body_leg.py's independently-derived 2-link
result EXACTLY (to float precision) - see
test_rigid_body_leg_6dof.py::test_reduces_to_2link_sagittal_model.
That module was already verified against exact analytic checks
(parallel-axis theorem, physical-pendulum torque), so an exact match
to it is a real, transitive correctness guarantee, not a new
from-scratch claim resting only on "the code runs".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinematics.leg_fk import LegParams, _rot_x, _rot_y, _rot_z
from dynamics.leg_dynamics import RobotMassParams, THIGH_MASS_FRACTION, SHANK_MASS_FRACTION

GRAVITY = 9.81
N_DOF = 6  # hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll

# Which joint indices are "before" (i.e. inboard of) each link in the chain -
# these joints' motion moves that link's COM and orientation. Ankle joints
# (indices 4, 5) come AFTER both thigh and shank in the chain, so they never
# appear here - see module docstring.
_THIGH_JOINTS = (0, 1, 2)         # hip_yaw, hip_roll, hip_pitch
_SHANK_JOINTS = (0, 1, 2, 3)      # + knee_pitch


@dataclass
class LinkInertia3D:
    mass_kg: float
    length_m: float
    com_distance_m: float      # from proximal joint, = length/2 for a uniform rod
    i_transverse_kgm2: float   # about COM, perpendicular to the rod's long axis: (1/12) m L^2
    i_axial_kgm2: float        # about COM, along the rod's own long axis: (1/2) m r^2 for a
                                # solid cylinder of radius r (see docs/BUGS_FOUND.md bug #5 -
                                # a zero-radius "thin rod" idealization, fine for the 2-link
                                # planar model which never rotates a link about its own long
                                # axis, makes this 6-DOF M(q) exactly SINGULAR at the
                                # legs-hanging-straight-down pose, where hip_yaw's axis lines
                                # up with the thigh's own axis)


# Typical adult limb cross-sectional radius (rough anthropometric approximation,
# not from a cited regression like the de Leva mass fractions above - used only
# to give the axial inertia term a physically sane nonzero order of magnitude,
# since NO axial radius data is otherwise used anywhere else in this project)
_THIGH_RADIUS_M = 0.08
_SHANK_RADIUS_M = 0.05


def link_inertias(leg: LegParams, robot_mass: RobotMassParams) -> tuple[LinkInertia3D, LinkInertia3D]:
    m1 = THIGH_MASS_FRACTION * robot_mass.total_mass_kg
    m2 = SHANK_MASS_FRACTION * robot_mass.total_mass_kg
    thigh = LinkInertia3D(mass_kg=m1, length_m=leg.thigh_length_m,
                           com_distance_m=leg.thigh_length_m / 2.0,
                           i_transverse_kgm2=m1 * leg.thigh_length_m ** 2 / 12.0,
                           i_axial_kgm2=0.5 * m1 * _THIGH_RADIUS_M ** 2)
    shank = LinkInertia3D(mass_kg=m2, length_m=leg.shank_length_m,
                           com_distance_m=leg.shank_length_m / 2.0,
                           i_transverse_kgm2=m2 * leg.shank_length_m ** 2 / 12.0,
                           i_axial_kgm2=0.5 * m2 * _SHANK_RADIUS_M ** 2)
    return thigh, shank


def _joint_frames(q: np.ndarray, hip_position: np.ndarray, leg: LegParams
                   ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, np.ndarray]:
    """Returns (joint_origins, joint_axes_world, thigh_com, shank_com).
    joint_origins[i] / joint_axes_world[i] are the position and world-frame
    rotation axis of joint i (0..5), matching kinematics/leg_fk.py's chain
    and rotation conventions exactly (Rz then Rx then Ry at the hip, so
    world-frame axes are NOT simply e_z/e_x/e_y once upstream joints are
    nonzero - this is computed properly below, not assumed constant)."""
    hip_yaw, hip_roll, hip_pitch, knee_pitch, ankle_pitch, ankle_roll = q

    # Rotation accumulated BEFORE each joint's own rotation is applied -
    # this is the frame each joint's local axis (e_z, e_x, or e_y per
    # leg_fk.py's Rz@Rx@Ry / Ry / Ry@Rx chain) must be expressed in.
    R0 = np.eye(3)                                  # before hip_yaw
    R1 = R0 @ _rot_z(hip_yaw)                        # before hip_roll
    R2 = R1 @ _rot_x(hip_roll)                       # before hip_pitch
    R_hip = R2 @ _rot_y(hip_pitch)                   # thigh orientation
    R3 = R_hip                                       # before knee_pitch
    R_knee = R3 @ _rot_y(knee_pitch)                 # shank orientation
    R4 = R_knee                                      # before ankle_pitch
    R5 = R4 @ _rot_y(ankle_pitch)                    # before ankle_roll

    axes = [R0 @ np.array([0., 0., 1.]),   # hip_yaw   rotates about local z
            R1 @ np.array([1., 0., 0.]),   # hip_roll  rotates about local x
            R2 @ np.array([0., 1., 0.]),   # hip_pitch rotates about local y
            R3 @ np.array([0., 1., 0.]),   # knee_pitch about local y
            R4 @ np.array([0., 1., 0.]),   # ankle_pitch about local y
            R5 @ np.array([1., 0., 0.])]   # ankle_roll about local x

    knee_pos = hip_position + R_hip @ np.array([0., 0., -leg.thigh_length_m])
    ankle_pos = knee_pos + R_knee @ np.array([0., 0., -leg.shank_length_m])
    # the 3 hip joints are co-located (spherical-joint decomposition at the
    # hip); knee_pitch's own origin is the knee; ankle_pitch and ankle_roll
    # are BOTH co-located at the ankle (their own spherical-joint-style
    # decomposition) - NOT at the knee. An earlier version of this function
    # incorrectly set the ankle origins to knee_pos (and even said so in this
    # comment) - harmless at the time, since _THIGH_JOINTS/_SHANK_JOINTS
    # never reference indices 4 or 5's origin, but a real bug once a foot
    # link needs them (see dynamics/foot_inertia.py, docs/BUGS_FOUND.md).
    origins = [hip_position, hip_position, hip_position, knee_pos, ankle_pos, ankle_pos]

    thigh_com = hip_position + R_hip @ np.array([0., 0., -leg.thigh_length_m / 2.0])
    shank_com = knee_pos + R_knee @ np.array([0., 0., -leg.shank_length_m / 2.0])

    return origins, axes, thigh_com, shank_com, R_hip, R_knee


def com_jacobians(q: np.ndarray, hip_position: np.ndarray, leg: LegParams
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (Jv_thigh, Jw_thigh, Jv_shank, Jw_shank, R_hip, R_knee), each
    Jacobian shaped (3, 6). Column i is zero for any joint not inboard of
    that link (see _THIGH_JOINTS / _SHANK_JOINTS)."""
    origins, axes, thigh_com, shank_com, R_hip, R_knee = _joint_frames(q, hip_position, leg)

    Jv_thigh = np.zeros((3, N_DOF))
    Jw_thigh = np.zeros((3, N_DOF))
    for i in _THIGH_JOINTS:
        Jv_thigh[:, i] = np.cross(axes[i], thigh_com - origins[i])
        Jw_thigh[:, i] = axes[i]

    Jv_shank = np.zeros((3, N_DOF))
    Jw_shank = np.zeros((3, N_DOF))
    for i in _SHANK_JOINTS:
        Jv_shank[:, i] = np.cross(axes[i], shank_com - origins[i])
        Jw_shank[:, i] = axes[i]

    return Jv_thigh, Jw_thigh, Jv_shank, Jw_shank, R_hip, R_knee


def _body_inertia_tensor(link: LinkInertia3D) -> np.ndarray:
    """Inertia tensor about the link's own COM, in the link's OWN frame,
    where the link's long axis is local z (matching leg_fk.py: each
    segment points along local -z before the next joint's rotation)."""
    it = link.i_transverse_kgm2
    ia = link.i_axial_kgm2
    return np.diag([it, it, ia])


def mass_matrix(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                 thigh: LinkInertia3D, shank: LinkInertia3D) -> np.ndarray:
    Jv_t, Jw_t, Jv_s, Jw_s, R_hip, R_knee = com_jacobians(q, hip_position, leg)

    I_thigh_world = R_hip @ _body_inertia_tensor(thigh) @ R_hip.T
    I_shank_world = R_knee @ _body_inertia_tensor(shank) @ R_knee.T

    M = (thigh.mass_kg * (Jv_t.T @ Jv_t) + Jw_t.T @ I_thigh_world @ Jw_t
         + shank.mass_kg * (Jv_s.T @ Jv_s) + Jw_s.T @ I_shank_world @ Jw_s)
    return M


def gravity_torque(q: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                    thigh: LinkInertia3D, shank: LinkInertia3D) -> np.ndarray:
    Jv_t, _, Jv_s, _, _, _ = com_jacobians(q, hip_position, leg)
    # G_i = d(PE)/dq_i = mass * g * d(z_com)/dq_i  (z is world "up", row index 2)
    return thigh.mass_kg * GRAVITY * Jv_t[2, :] + shank.mass_kg * GRAVITY * Jv_s[2, :]


def coriolis_matrix(q: np.ndarray, qdot: np.ndarray, hip_position: np.ndarray, leg: LegParams,
                     thigh: LinkInertia3D, shank: LinkInertia3D, eps: float = 1e-6) -> np.ndarray:
    """Same Christoffel-symbol formula and same single-numerical-
    differentiation-of-analytic-M(q) approach as rigid_body_leg.py,
    generalized from 2 to 6 DOF."""
    M0 = mass_matrix(q, hip_position, leg, thigh, shank)
    dM = []
    for k in range(N_DOF):
        qk = q.copy()
        qk[k] += eps
        dM.append((mass_matrix(qk, hip_position, leg, thigh, shank) - M0) / eps)

    C = np.zeros((N_DOF, N_DOF))
    for i in range(N_DOF):
        for j in range(N_DOF):
            s = 0.0
            for k in range(N_DOF):
                s += (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qdot[k]
            C[i, j] = 0.5 * s
    return C


def swing_leg_rigid_body_torque_6dof(q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray,
                                      hip_position: np.ndarray, leg: LegParams,
                                      robot_mass: RobotMassParams) -> np.ndarray:
    """Drop-in full-6-DOF replacement for
    rigid_body_leg.swing_leg_rigid_body_torque: same tau = M qddot + C qdot
    + G manipulator equation, now over all 6 joints instead of just
    hip_pitch/knee_pitch. Ankle rows are guaranteed exactly zero (see
    module docstring) - callers should still add the ankle's OWN
    (currently unmodeled) foot-inertia torque separately once a foot link
    exists; this function does not silently claim to cover it."""
    thigh, shank = link_inertias(leg, robot_mass)
    M = mass_matrix(q, hip_position, leg, thigh, shank)
    C = coriolis_matrix(q, qdot, hip_position, leg, thigh, shank)
    G = gravity_torque(q, hip_position, leg, thigh, shank)
    return M @ qddot + C @ qdot + G
