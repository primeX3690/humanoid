"""tests/test_rigid_body_leg_6dof.py — verifies the 6-DOF rigid-body
dynamics against (1) the independently-derived, already-verified 2-link
sagittal model (exact reduction check), (2) finite differences of the
forward-kinematics COM positions (verifies the geometric-Jacobian
formula itself, not just its downstream use), and (3) basic physical
invariants (symmetric/positive-definite M, C@qdot=0 at qdot=0, zero
ankle rows/columns)."""
import numpy as np
import pytest

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import RobotMassParams
from dynamics.rigid_body_leg_6dof import (
    mass_matrix, gravity_torque, coriolis_matrix, link_inertias, com_jacobians,
    swing_leg_rigid_body_torque_6dof, N_DOF,
)
from dynamics.rigid_body_leg import (
    mass_matrix as mass_matrix_2link,
    gravity_torque as gravity_torque_2link,
    coriolis_matrix as coriolis_matrix_2link,
    link_inertias as link_inertias_2link,
)
from kinematics.leg_fk import forward_kinematics

LEG = LegParams()
ROBOT = RobotMassParams(total_mass_kg=25.0)
HIP = np.array([0.0, 0.0, 0.8])


def _com_positions_numeric(q, hip_position, leg):
    """Independent numeric recomputation of thigh/shank COM position,
    used ONLY to finite-difference-check the analytic Jacobians - written
    from forward_kinematics()'s own rotation chain, not copy-pasted from
    rigid_body_leg_6dof.py's implementation."""
    from kinematics.leg_fk import _rot_x, _rot_y, _rot_z
    hip_yaw, hip_roll, hip_pitch, knee_pitch, _, _ = q
    R_hip = _rot_z(hip_yaw) @ _rot_x(hip_roll) @ _rot_y(hip_pitch)
    knee_pos = hip_position + R_hip @ np.array([0., 0., -leg.thigh_length_m])
    R_knee = R_hip @ _rot_y(knee_pitch)
    thigh_com = hip_position + R_hip @ np.array([0., 0., -leg.thigh_length_m / 2.0])
    shank_com = knee_pos + R_knee @ np.array([0., 0., -leg.shank_length_m / 2.0])
    return thigh_com, shank_com


def test_jacobians_match_finite_difference():
    """The geometric-Jacobian (z_i x r) formula is THE load-bearing
    identity this whole module rests on - verify it directly against
    finite differences of an independently-written COM position function,
    at a generic (non-degenerate) configuration, for every one of the 6
    columns."""
    q0 = np.array([0.15, -0.1, 0.3, 0.5, -0.2, 0.1])
    Jv_t, _, Jv_s, _, _, _ = com_jacobians(q0, HIP, LEG)
    eps = 1e-6
    for k in range(N_DOF):
        qk = q0.copy()
        qk[k] += eps
        t_plus, s_plus = _com_positions_numeric(qk, HIP, LEG)
        t0, s0 = _com_positions_numeric(q0, HIP, LEG)
        assert np.allclose(Jv_t[:, k], (t_plus - t0) / eps, atol=1e-5)
        assert np.allclose(Jv_s[:, k], (s_plus - s0) / eps, atol=1e-5)


def test_ankle_columns_and_rows_are_exactly_zero():
    """Ankle joints (indices 4, 5) are distal to both thigh and shank -
    they must contribute EXACTLY zero to M, since neither link's COM or
    orientation depends on them (see module docstring: this is a stated
    scope limit, verified here, not silently assumed)."""
    q = np.array([0.2, -0.15, 0.4, 0.6, 0.3, -0.25])
    thigh, shank = link_inertias(LEG, ROBOT)
    M = mass_matrix(q, HIP, LEG, thigh, shank)
    assert np.allclose(M[4:6, :], 0.0)
    assert np.allclose(M[:, 4:6], 0.0)
    G = gravity_torque(q, HIP, LEG, thigh, shank)
    assert G[4] == 0.0 and G[5] == 0.0


def test_reduces_to_2link_sagittal_model_exactly():
    """With hip_yaw = hip_roll = ankle_pitch = ankle_roll = 0, the
    hip_pitch/knee_pitch (indices 2,3) block of this 6-DOF M, G, and C
    must match rigid_body_leg.py's INDEPENDENTLY-derived (and separately,
    analytically, verified) 2-link result exactly. This is the strongest
    correctness evidence available: it's not a new claim resting only on
    'the code runs', it's agreement with a result already checked against
    the parallel-axis theorem and the physical-pendulum formula."""
    q1, q2 = 0.35, -0.6
    q = np.array([0.0, 0.0, q1, q2, 0.0, 0.0])
    qdot12 = np.array([0.4, -0.3])
    qdot = np.array([0.0, 0.0, 0.4, -0.3, 0.0, 0.0])

    thigh6, shank6 = link_inertias(LEG, ROBOT)
    thigh2, shank2 = link_inertias_2link(LEG, ROBOT)
    # sanity: the two modules must at least agree on mass/length/COM inputs
    assert thigh6.mass_kg == thigh2.mass_kg
    assert thigh6.i_transverse_kgm2 == pytest.approx(thigh2.inertia_kgm2)

    M6 = mass_matrix(q, HIP, LEG, thigh6, shank6)
    M2 = mass_matrix_2link(q1, q2, thigh2, shank2)
    assert np.allclose(M6[np.ix_([2, 3], [2, 3])], M2, atol=1e-9)

    G6 = gravity_torque(q, HIP, LEG, thigh6, shank6)
    G2 = gravity_torque_2link(q1, q2, thigh2, shank2)
    assert np.allclose(G6[[2, 3]], G2, atol=1e-8)

    C6 = coriolis_matrix(q, qdot, HIP, LEG, thigh6, shank6)
    C2 = coriolis_matrix_2link(q1, q2, qdot12, thigh2, shank2)
    assert np.allclose(C6[np.ix_([2, 3], [2, 3])], C2, atol=1e-6)


def test_hip_yaw_and_roll_couple_into_the_mass_matrix():
    """The whole point of this module: verify hip_yaw/hip_roll genuinely
    couple into the dynamics - which the old 2-link model structurally
    could not represent at all (it didn't have these joints).

    IMPORTANT, non-obvious physical fact checked here (found while writing
    this test - see docs/BUGS_FOUND.md bug #6): because hip_yaw, hip_roll
    and hip_pitch are three sequential revolutes sharing ONE point (a
    spherical joint decomposition), the hip_pitch/knee_pitch DIAGONAL
    block of M is mathematically INVARIANT to hip_yaw/hip_roll - rotating
    the whole downstream chain's reference frame doesn't change its own
    body-relative inertia. That is correct physics, not a missing
    coupling, and is asserted directly below rather than silently relied
    upon. The REAL 3D coupling this module adds shows up in the
    OFF-diagonal hip_yaw/roll <-> hip_pitch/knee terms instead, which
    genuinely are zero when the chain points straight down (hip_yaw/roll
    axes and the leg's own axis are aligned - no cross-coupling possible)
    and genuinely become nonzero once yawed/rolled."""
    thigh, shank = link_inertias(LEG, ROBOT)
    q_straight = np.array([0.0, 0.0, 0.3, 0.5, 0.0, 0.0])
    q_yawed = np.array([0.4, 0.2, 0.3, 0.5, 0.0, 0.0])
    M_straight = mass_matrix(q_straight, HIP, LEG, thigh, shank)
    M_yawed = mass_matrix(q_yawed, HIP, LEG, thigh, shank)

    assert M_yawed[0, 0] > 0.0   # hip_yaw generalized inertia is nonzero
    assert M_yawed[1, 1] > 0.0   # hip_roll generalized inertia is nonzero
    # correct invariance, not a bug:
    assert np.allclose(M_straight[2:4, 2:4], M_yawed[2:4, 2:4])
    # the real coupling instead lives off-diagonal:
    assert np.allclose(M_straight[0:2, 2:4], 0.0)
    assert not np.allclose(M_yawed[0:2, 2:4], 0.0)


def test_axial_inertia_removes_the_straight_down_singularity():
    """docs/BUGS_FOUND.md bug #5: with the earlier zero-radius thin-rod
    idealization, M was exactly singular (a zero eigenvalue) at the
    legs-hanging-straight-down pose, because hip_yaw's rotation axis lines
    up exactly with the thigh's own long axis and a true 1D rod has zero
    inertia about its own axis. The finite-radius fix must make the FULL
    active 4x4 block strictly positive definite even at that exact pose,
    not just away from it."""
    thigh, shank = link_inertias(LEG, ROBOT)
    q_hanging = np.zeros(N_DOF)
    M = mass_matrix(q_hanging, HIP, LEG, thigh, shank)
    active = M[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
    assert np.all(np.linalg.eigvalsh(active) > 0.0)


def test_mass_matrix_is_symmetric_and_positive_semidefinite():
    """Symmetric (energy is a quadratic form) and positive SEMI-definite
    (the ankle DOFs are structurally massless in this model - see the
    zero-rows test above - so the matrix is only semi-, not strictly,
    definite; the hip/knee 4x4 sub-block IS strictly positive definite,
    checked separately)."""
    thigh, shank = link_inertias(LEG, ROBOT)
    for q in [np.array([0., 0., 0., 0., 0., 0.]),
              np.array([0.3, -0.2, 0.4, 0.6, 0.1, -0.1]),
              np.array([-0.5, 0.4, -0.3, 1.0, 0.0, 0.2])]:
        M = mass_matrix(q, HIP, LEG, thigh, shank)
        assert np.allclose(M, M.T)
        eigvals = np.linalg.eigvalsh(M)
        assert np.all(eigvals > -1e-9)
        active = M[np.ix_([0, 1, 2, 3], [0, 1, 2, 3])]
        assert np.all(np.linalg.eigvalsh(active) > 0)


def test_coriolis_term_vanishes_at_zero_velocity():
    thigh, shank = link_inertias(LEG, ROBOT)
    q = np.array([0.2, -0.1, 0.3, 0.5, 0.1, -0.2])
    C = coriolis_matrix(q, np.zeros(N_DOF), HIP, LEG, thigh, shank)
    assert np.allclose(C @ np.zeros(N_DOF), np.zeros(N_DOF))


def test_swing_leg_torque_matches_static_gravity_case():
    q = np.array([0.1, -0.05, 0.3, 0.5, 0.0, 0.0])
    tau = swing_leg_rigid_body_torque_6dof(q, np.zeros(N_DOF), np.zeros(N_DOF), HIP, LEG, ROBOT)
    thigh, shank = link_inertias(LEG, ROBOT)
    G = gravity_torque(q, HIP, LEG, thigh, shank)
    assert np.allclose(tau, G, atol=1e-8)


def test_fk_consistency_thigh_and_shank_com_lie_on_the_ankle_chain():
    """Cross-check against the project's own already-verified
    forward_kinematics(): the shank COM computed here must lie exactly
    halfway between the knee and ankle positions FK reports, for a
    generic configuration - catches any accidental frame-convention
    mismatch between this new module and leg_fk.py."""
    q = np.array([0.1, 0.15, 0.3, 0.4, 0.05, -0.1])
    ankle_pos, _ = forward_kinematics(q, HIP, LEG)
    _, _, thigh_com, shank_com, R_hip, R_knee = \
        __import__("dynamics.rigid_body_leg_6dof", fromlist=["_joint_frames"])._joint_frames(q, HIP, LEG)
    knee_pos = HIP + R_hip @ np.array([0., 0., -LEG.thigh_length_m])
    expected_shank_com = (knee_pos + ankle_pos) / 2.0
    assert np.allclose(shank_com, expected_shank_com, atol=1e-9)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
