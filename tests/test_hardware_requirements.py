"""tests/test_hardware_requirements.py — verifies actuators/hardware_requirements.py:
every number traces to a real, sourced datasheet value or simple stated
arithmetic on it, and locks in the real, honest finding that the
recommended actuators alone consume a large fraction of the assumed
total robot mass."""
import pytest

from actuators.hardware_requirements import derive_requirements, UNMODELED_HARDWARE_REQUIREMENTS
from actuators.motor_specs import DYNAMIXEL_H54P_200
from dynamics.leg_dynamics import RobotMassParams


def test_actuator_count_matches_the_6dof_leg_model():
    r = derive_requirements()
    assert r.n_actuators == 12  # 6 DOF x 2 legs - dynamics/rigid_body_leg_6dof.py's N_DOF


def test_torque_margin_matches_the_documented_finding():
    """docs/BUGS_FOUND.md: H54P-200's 44.7 Nm continuous rating gives a
    1.39x margin over the 32.2 Nm peak stance-leg requirement."""
    r = derive_requirements()
    assert r.torque_margin == pytest.approx(44.7 / 32.2, rel=1e-6)
    assert r.torque_margin > 1.0  # feasible, matching the documented finding


def test_actuator_mass_consumes_a_large_fraction_of_assumed_total_mass():
    """The real, honest finding this module surfaces: 12 real H54P-200
    units (0.855 kg each, real datasheet weight) already consume a large
    chunk of the 25 kg total-mass assumption used throughout this
    project's dynamics - locked in with the actual measured fraction so
    it doesn't silently drift if either number changes without the
    other being reconsidered."""
    r = derive_requirements()
    assert r.total_actuator_mass_kg == pytest.approx(12 * 0.855, rel=1e-9)
    assert r.actuator_mass_fraction_of_total == pytest.approx(10.26 / 25.0, rel=1e-6)
    assert r.actuator_mass_fraction_of_total > 0.35  # a genuinely large fraction, not negligible


def test_worst_case_power_uses_real_datasheet_current_and_voltage():
    r = derive_requirements()
    assert r.worst_case_simultaneous_power_w == pytest.approx(12 * 24.0 * 9.3, rel=1e-9)


def test_requirements_respond_to_a_different_robot_mass_assumption():
    """If a future revision changes the assumed total mass, the derived
    fraction must update accordingly, not stay pinned to today's number -
    confirms this is a live derivation, not a hardcoded report."""
    lighter = derive_requirements(robot_mass=RobotMassParams(total_mass_kg=50.0))
    assert lighter.actuator_mass_fraction_of_total == pytest.approx(10.26 / 50.0, rel=1e-6)
    assert lighter.actuator_mass_fraction_of_total < derive_requirements().actuator_mass_fraction_of_total


def test_unmodeled_requirements_list_is_non_empty_and_stated_plainly():
    """This module must not imply hardware is 'done' - the unmodeled list
    (structure, sensing, power system, thermal, feet, other body
    segments) must exist and be non-trivial."""
    assert len(UNMODELED_HARDWARE_REQUIREMENTS) >= 5
    joined = " ".join(UNMODELED_HARDWARE_REQUIREMENTS).lower()
    for topic in ["structural", "sensing", "power system", "thermal", "arms"]:
        assert topic in joined


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
