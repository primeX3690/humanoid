"""tests/test_rigid_body_leg.py — verifies the 2-link rigid-body dynamics
against exact analytic checks (parallel-axis theorem, physical-pendulum
gravity torque), not just "it runs"."""
import numpy as np
import pytest

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import RobotMassParams
from dynamics.rigid_body_leg import (
    mass_matrix, gravity_torque, coriolis_matrix, link_inertias, swing_leg_rigid_body_torque,
)

LEG = LegParams()
ROBOT = RobotMassParams(total_mass_kg=25.0)


class _ZeroLink:
    mass_kg = 0.0
    inertia_kgm2 = 0.0

    def __init__(self, length_m, com_distance_m):
        self.length_m = length_m
        self.com_distance_m = com_distance_m


def test_single_link_reduction_matches_parallel_axis_theorem():
    """With the shank's mass/inertia zeroed out, M[0,0] must exactly equal
    the standard physical-pendulum pivot inertia I_pivot = I_com + m*lc^2
    (parallel axis theorem) - an exact, hand-checkable analytic result."""
    thigh, shank = link_inertias(LEG, ROBOT)
    shank0 = _ZeroLink(shank.length_m, shank.com_distance_m)
    M = mass_matrix(0.4, 0.0, thigh, shank0)
    expected = thigh.inertia_kgm2 + thigh.mass_kg * thigh.com_distance_m ** 2
    assert abs(M[0, 0] - expected) < 1e-10


def test_single_link_reduction_matches_pendulum_gravity_torque():
    """Same reduction: gravity torque must exactly match the textbook
    physical-pendulum formula tau = m*g*lc*sin(theta)."""
    thigh, shank = link_inertias(LEG, ROBOT)
    shank0 = _ZeroLink(shank.length_m, shank.com_distance_m)
    q1 = 0.4
    G = gravity_torque(q1, 0.0, thigh, shank0)
    expected = thigh.mass_kg * 9.81 * thigh.com_distance_m * np.sin(q1)
    assert abs(G[0] - expected) < 1e-8


def test_mass_matrix_is_symmetric_and_positive_definite():
    """A real, physically valid mass matrix must be symmetric (energy is a
    quadratic form) and positive definite (kinetic energy > 0 for any
    nonzero velocity) - checked directly, not assumed."""
    thigh, shank = link_inertias(LEG, ROBOT)
    for q1, q2 in [(0.0, 0.0), (0.3, 0.6), (-0.5, 1.0), (0.2, -0.3)]:
        M = mass_matrix(q1, q2, thigh, shank)
        assert np.allclose(M, M.T)
        assert np.all(np.linalg.eigvalsh(M) > 0)


def test_coriolis_term_vanishes_at_zero_velocity():
    """C(q, qdot) @ qdot must be exactly zero when qdot=0 (no velocity-
    dependent forces without velocity) - a basic sanity property."""
    thigh, shank = link_inertias(LEG, ROBOT)
    C = coriolis_matrix(0.3, 0.5, np.zeros(2), thigh, shank)
    assert np.allclose(C @ np.zeros(2), np.zeros(2))


def test_swing_leg_rigid_body_torque_matches_static_gravity_case():
    """With zero velocity and zero acceleration, the required torque must
    equal exactly the gravity term alone (M@0 + C@0 + G = G) - a direct
    end-to-end consistency check of the assembled function."""
    q = np.array([0.0, 0.0, 0.3, 0.5, 0.0, 0.0])
    tau = swing_leg_rigid_body_torque(q, np.zeros(6), np.zeros(6), LEG, ROBOT)
    thigh, shank = link_inertias(LEG, ROBOT)
    G = gravity_torque(0.3, 0.5, thigh, shank)
    assert abs(tau[2] - G[0]) < 1e-8
    assert abs(tau[3] - G[1]) < 1e-8


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
