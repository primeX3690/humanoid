"""tests/test_foot_inertia.py — verifies dynamics/foot_inertia.py: the
ankle-origin bug fix in rigid_body_leg_6dof.py causes no regression
there, the foot's own inertia is genuinely nonzero on the ankle
rows/columns (closing the stated scope gap), the additive
M/G/C = thigh_shank + foot construction is independently verified
against a direct from-scratch Christoffel-symbol computation on the
FULL 3-body mass matrix (not just internally self-consistent), and
basic physical sanity (symmetric, positive definite, matches gravity
sign convention)."""
import numpy as np
import pytest

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import RobotMassParams, FOOT_MASS_FRACTION
from dynamics.rigid_body_leg_6dof import (
    mass_matrix as mass_matrix_thigh_shank, N_DOF, link_inertias,
)
from dynamics.foot_inertia import (
    foot_inertia, mass_matrix_with_foot, gravity_torque_with_foot,
    coriolis_matrix_with_foot, swing_leg_rigid_body_torque_with_foot,
)

LEG = LegParams()
ROBOT = RobotMassParams(total_mass_kg=25.0)
HIP = np.array([0.0, 0.0, 0.8])


def _direct_christoffel(mass_matrix_fn, q, qdot, eps=1e-6):
    """Independent, from-scratch Christoffel-symbol computation on
    whatever mass_matrix_fn(q) returns - used to verify
    coriolis_matrix_with_foot's additive shortcut against a totally
    separate derivation, not just checking it agrees with itself."""
    n = len(q)
    M0 = mass_matrix_fn(q)
    dM = []
    for k in range(n):
        qk = q.copy()
        qk[k] += eps
        dM.append((mass_matrix_fn(qk) - M0) / eps)
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += (dM[k][i, j] + dM[j][i, k] - dM[i][j, k]) * qdot[k]
            C[i, j] = 0.5 * s
    return C


def test_ankle_rows_are_no_longer_structurally_zero():
    """The whole point: with the foot's own inertia included, M's ankle
    rows/columns must be genuinely nonzero (rigid_body_leg_6dof.py's own
    tests lock in that they WERE exactly zero without a foot link)."""
    q = np.array([0.2, -0.15, 0.4, 0.6, 0.3, -0.25])
    M = mass_matrix_with_foot(q, HIP, LEG, ROBOT)
    assert not np.allclose(M[4:6, :], 0.0)
    assert not np.allclose(M[:, 4:6], 0.0)
    assert M[4, 4] > 0.0
    assert M[5, 5] > 0.0


def test_thigh_shank_only_ankle_rows_are_still_exactly_zero():
    """Regression guard: the ORIGINAL (foot-less) module's own documented
    behavior must be completely unaffected by the ankle-origin bugfix -
    those origins were dead values for thigh/shank, and must stay dead."""
    q = np.array([0.2, -0.15, 0.4, 0.6, 0.3, -0.25])
    thigh, shank = link_inertias(LEG, ROBOT)
    M = mass_matrix_thigh_shank(q, HIP, LEG, thigh, shank)
    assert np.allclose(M[4:6, :], 0.0)
    assert np.allclose(M[:, 4:6], 0.0)


def test_mass_matrix_with_foot_is_symmetric_and_positive_definite_everywhere():
    """Unlike the thigh/shank-only model (positive SEMI-definite, exactly
    zero on ankle rows), adding the foot's own inertia should make the
    FULL 6x6 matrix strictly positive definite, including at the
    legs-hanging-straight pose that was the source of
    docs/BUGS_FOUND.md bug #5."""
    for q in [np.zeros(6),
              np.array([0.3, -0.2, 0.4, 0.6, 0.1, -0.1]),
              np.array([-0.5, 0.4, -0.3, 1.0, 0.0, 0.2])]:
        M = mass_matrix_with_foot(q, HIP, LEG, ROBOT)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0.0)


def test_coriolis_additive_shortcut_matches_independent_full_christoffel_derivation():
    """The key structural claim in dynamics/foot_inertia.py's docstring -
    C_total = C_thigh_shank + C_foot is valid because the Christoffel
    formula is linear in M - verified here against a TOTALLY SEPARATE,
    from-scratch Christoffel computation on the full 3-body mass matrix,
    not just checked for internal self-consistency."""
    q = np.array([0.2, -0.15, 0.4, 0.6, 0.3, -0.25])
    qdot = np.array([0.3, -0.2, 0.5, -0.4, 0.2, 0.1])
    C_shortcut = coriolis_matrix_with_foot(q, qdot, HIP, LEG, ROBOT)
    C_direct = _direct_christoffel(lambda qq: mass_matrix_with_foot(qq, HIP, LEG, ROBOT), q, qdot)
    assert np.allclose(C_shortcut, C_direct, atol=1e-6)


def test_foot_mass_fraction_is_the_real_de_leva_value():
    assert FOOT_MASS_FRACTION == pytest.approx(0.0137)


def test_gravity_with_foot_exceeds_gravity_without_foot():
    """Adding a foot with positive mass must strictly increase the
    magnitude of at least the joints the foot actually depends on
    (sanity check on sign/magnitude, not just 'code runs')."""
    from dynamics.rigid_body_leg_6dof import gravity_torque as gravity_thigh_shank
    q = np.array([0.1, 0.05, 0.3, 0.5, 0.1, -0.1])
    thigh, shank = link_inertias(LEG, ROBOT)
    g_without = gravity_thigh_shank(q, HIP, LEG, thigh, shank)
    g_with = gravity_torque_with_foot(q, HIP, LEG, ROBOT)
    assert not np.allclose(g_without, g_with)
    assert np.linalg.norm(g_with) > np.linalg.norm(g_without)


def test_foot_com_offset_is_forward_and_below_the_ankle_at_rest():
    """Sanity check on the stated placement assumption: with the leg
    hanging straight down and the foot pointing straight down too (zero
    ankle angles), the foot's COM must sit forward of and below the
    ankle - not behind it or above it."""
    foot = foot_inertia(LEG, ROBOT)
    assert foot.com_forward_m > 0.0
    assert foot.com_down_m > 0.0
    assert foot.mass_kg > 0.0


def test_swing_leg_torque_with_foot_matches_static_gravity_case():
    q = np.array([0.1, -0.05, 0.3, 0.5, 0.1, -0.1])
    tau = swing_leg_rigid_body_torque_with_foot(q, np.zeros(N_DOF), np.zeros(N_DOF), HIP, LEG, ROBOT)
    g = gravity_torque_with_foot(q, HIP, LEG, ROBOT)
    assert np.allclose(tau, g, atol=1e-8)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
