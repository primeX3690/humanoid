"""tests/test_actuator_dynamics.py — verifies actuators/actuator_dynamics.py:
the RK4 integrator against the exact analytic first-order step response,
the sourced electrical-parameter derivation (including the honest
no-load-speed over-prediction finding), and the real, measured torque-
tracking behavior when driving a real gait's required knee torque
through the model - both the "normal" small tracking error and the
sharp spikes at velocity-reversal instants (see docs/BUGS_FOUND.md)."""
import numpy as np
import pytest

from planning.footstep_planner import GaitParams
from simulation.walk_simulator import simulate_walk
from simulation.joint_trajectory_generator import generate_joint_trajectory
from dynamics.foot_inertia import swing_leg_rigid_body_torque_with_foot
from dynamics.leg_dynamics import RobotMassParams
from kinematics.leg_fk import LegParams
from actuators.actuator_dynamics import (
    mx106_electrical_params, current_step_response_analytic, simulate_current_response,
    output_torque, track_required_torque,
)

LEG = LegParams()
ROBOT = RobotMassParams(total_mass_kg=25.0)
HIP = np.array([0.0, 0.0, 0.8])


def test_sourced_electrical_parameters_match_hand_derivation():
    """Every number here traces to the real, cited MX-106 datasheet
    figures (V=12, I_stall=5.2A, tau_stall=8.40Nm, I_noload=0.17A) via
    the stated (Bodine-convention) formulas - not invented."""
    p = mx106_electrical_params()
    assert p.resistance_ohm == pytest.approx(12.0 / 5.2, rel=1e-9)
    assert p.kt_nm_per_a == pytest.approx(8.40 / (5.2 - 0.17), rel=1e-9)
    assert p.kb_v_per_rads == pytest.approx(p.kt_nm_per_a)
    assert p.coulomb_friction_nm == pytest.approx(p.kt_nm_per_a * 0.17, rel=1e-9)
    assert p.inductance_henry == pytest.approx(0.0115 * p.resistance_ohm, rel=1e-9)


def test_honest_no_load_speed_overprediction_is_real_and_documented():
    """docs/BUGS_FOUND.md's stated finding: calibrating Kt/Kb from stall
    data over-predicts the datasheet's real 45 RPM no-load speed by
    ~47%. Locked in here with the actual measured ratio so it doesn't
    silently drift or get accidentally "fixed" without the finding being
    updated to match."""
    p = mx106_electrical_params()
    predicted_rpm = p.predicted_no_load_speed_rad_s * 60.0 / (2 * np.pi)
    real_rpm = 45.0
    assert predicted_rpm > real_rpm  # over-prediction, not under
    assert predicted_rpm / real_rpm == pytest.approx(1.475, abs=0.02)


def test_rk4_integrator_matches_exact_analytic_step_response():
    """The RK4 numerical integrator must reproduce the CLOSED-FORM
    solution of the same linear ODE (constant voltage, constant speed)
    to near machine precision - this is the core correctness check for
    the general-purpose integrator used everywhere else in this module."""
    p = mx106_electrical_params()
    dt = 1e-5
    t = np.arange(0, 0.05, dt)
    V, omega = 8.0, 10.0
    analytic = current_step_response_analytic(p, V, omega, t)
    numeric = simulate_current_response(p, np.full(len(t), V), np.full(len(t), omega), dt)
    assert np.max(np.abs(analytic - numeric)) < 1e-10


def test_step_response_reaches_63_percent_at_one_time_constant():
    """Direct, independent sanity check on the analytic formula itself:
    by definition of an exponential step response, I(tau_e) should sit
    63.2% of the way from I(0) to the steady-state value."""
    p = mx106_electrical_params()
    tau_e = p.inductance_henry / p.resistance_ohm
    t = np.array([0.0, tau_e])
    i = current_step_response_analytic(p, voltage_v=6.0, omega_rad_s=0.0, t=t, i0=0.0)
    i_ss = 6.0 / p.resistance_ohm
    assert i[1] / i_ss == pytest.approx(1 - np.exp(-1), rel=1e-9)


def _knee_torque_and_omega_for_first_left_swing():
    gait = GaitParams(n_steps=6)
    walk = simulate_walk(gait)
    jt = generate_joint_trajectory(walk, gait)
    dt = gait.dt
    q = jt.left_joint_angles
    qdot = np.gradient(q, dt, axis=0)
    qddot = np.gradient(qdot, dt, axis=0)
    left_steps = [s for s in walk.footsteps if s.side == "left"]
    step = left_steps[0]
    mask = (walk.t >= step.start_time) & (walk.t < step.end_time)
    idx = np.where(mask)[0]
    knee_tau = np.array([swing_leg_rigid_body_torque_with_foot(q[k], qdot[k], qddot[k], HIP, LEG, ROBOT)[3]
                          for k in idx])
    knee_omega = qdot[idx, 3]
    return knee_tau, knee_omega, dt


def test_torque_tracking_on_real_gait_data_stays_within_actuator_limits():
    """This specific knee-swing torque profile (unlike hip_pitch's, which
    already exceeds the MX-106's STALL rating on pure magnitude alone -
    a separate, torque-MAGNITUDE finding, not an electrical-dynamics one)
    stays within the motor's real current/voltage limits throughout - no
    saturation - which is what makes it a fair, uncofounded test of
    electrical DYNAMICS specifically."""
    knee_tau, knee_omega, dt = _knee_torque_and_omega_for_first_left_swing()
    p = mx106_electrical_params()
    achieved, current, vcmd = track_required_torque(knee_tau, knee_omega, p, dt)
    assert np.max(np.abs(current)) < p.v_max / p.resistance_ohm  # well under stall current
    assert np.max(np.abs(vcmd)) < p.v_max - 1e-9  # never actually saturates


def test_torque_tracking_error_is_small_away_from_velocity_reversals_but_spikes_at_them():
    """The real, measured finding (docs/BUGS_FOUND.md): started from a
    realistic (already-at-target, not cold-zero) initial current, this
    knee's electrical tracking error stays small (under ~0.2 Nm) through
    most of the swing, but spikes to ~0.4-0.5 Nm for a single sample at
    each velocity-reversal instant, where the required Coulomb-friction
    compensation direction flips discontinuously - a real, physical
    consequence of finite electrical bandwidth trying to track an
    actually-discontinuous required-current signal, not a modeling
    artifact. Locked in with the measured magnitudes so this doesn't
    silently drift."""
    from actuators.actuator_dynamics import simulate_current_response
    knee_tau, knee_omega, dt = _knee_torque_and_omega_for_first_left_swing()
    p = mx106_electrical_params()

    i_required = (knee_tau + p.coulomb_friction_nm * np.sign(knee_omega)) / p.kt_nm_per_a
    v_command = np.clip(i_required * p.resistance_ohm + p.kb_v_per_rads * knee_omega,
                         -p.v_max, p.v_max)
    current = simulate_current_response(p, v_command, knee_omega, dt, i0=i_required[0])
    achieved = output_torque(p, current, knee_omega)
    err = achieved - knee_tau

    # velocity-reversal samples: where sign(omega) differs from the previous sample
    reversal = np.zeros(len(knee_omega), dtype=bool)
    reversal[1:] = np.sign(knee_omega[1:]) != np.sign(knee_omega[:-1])

    away_from_reversal = np.abs(err[~reversal])
    at_reversal = np.abs(err[reversal])

    assert np.sum(reversal) >= 2  # this swing phase genuinely crosses zero velocity more than once
    assert np.max(away_from_reversal) < 0.2
    assert np.max(at_reversal) > 0.35   # the spike is real and substantial...
    assert np.max(at_reversal) < 0.6    # ...but bounded, not runaway


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
