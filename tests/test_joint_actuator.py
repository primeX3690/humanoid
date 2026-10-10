import numpy as np
import pytest
from actuators.joint_actuator import ActuatorBank, PH54_200, MX106, QDD_REFERENCE, ideal, spec_for_joint, humanoid_bank
from sim_realism.latency import DelayLine


def run(bank, tau, qd, T):
    out = []
    for _ in range(int(T / bank.dt)):
        out.append(bank.step(np.array([tau]), np.array([qd]))[0])
    return np.array(out)


def test_current_loop_lag_matches_analytic_first_order():
    s = ideal(PH54_200); s = type(s)(**{**s.__dict__, "tau_electrical_s": 0.02})
    b = ActuatorBank([s], dt=1e-4, thermal=False)
    out = run(b, 10.0, 0.0, 0.1)
    t = np.arange(1, len(out) + 1) * 1e-4
    assert np.allclose(out, 10.0 * (1 - np.exp(-t / 0.02)), atol=0.02)


def test_ideal_actuator_is_plain_clipped_torque():
    b = ActuatorBank([ideal(PH54_200)], dt=1e-3, thermal=False, speed_limit=False)
    assert run(b, 10.0, 0.0, 0.01)[-1] == pytest.approx(10.0, abs=1e-6)
    assert run(b, 500.0, 0.0, 0.01)[-1] == pytest.approx(44.7, abs=1e-6)


def test_torque_speed_envelope_collapses_at_no_load_speed():
    b = ActuatorBank([ideal(PH54_200)], dt=1e-3, thermal=False)
    w = PH54_200.w_noload_rad_s
    corner = run(b, 100.0, 0.5 * w, 0.01)[-1]
    mid = run(b, 100.0, 0.75 * w, 0.01)[-1]
    full = run(b, 100.0, 1.0 * w, 0.01)[-1]
    assert corner == pytest.approx(44.7, rel=0.02)           # full torque up to the corner speed
    assert mid == pytest.approx(0.5 * 44.7, rel=0.02)
    assert abs(full) < 0.5
    # braking (torque opposing speed) is NOT speed-limited
    assert run(b, -30.0, w, 0.01)[-1] == pytest.approx(-30.0, rel=0.02)


def test_command_delay_is_exact_in_ticks():
    b = ActuatorBank([ideal(PH54_200)], dt=1e-3, thermal=False, speed_limit=False, cmd_delay_s=0.005)
    out = run(b, 10.0, 0.0, 0.02)
    first = int(np.argmax(out > 5.0))
    assert first == 5            # 5 ms == 5 ticks


def test_thermal_steady_state_at_continuous_torque_is_derate_onset():
    s = type(PH54_200)(**{**PH54_200.__dict__, "thermal_tau_s": 20.0, "tau_electrical_s": 0.0})
    b = ActuatorBank([s], dt=1e-2)
    run(b, s.tau_cont_nm, 0.0, 200.0)
    assert b.temp[0] == pytest.approx(s.t_derate_c, abs=0.5)
    assert b.derate()[0] == pytest.approx(1.0, abs=0.02)


def test_overload_heats_derates_then_cools_back():
    s = type(QDD_REFERENCE)(**{**QDD_REFERENCE.__dict__, "thermal_tau_s": 10.0, "tau_electrical_s": 0.0})
    b = ActuatorBank([s], dt=1e-2)
    run(b, s.tau_peak_nm, 0.0, 30.0)                     # sustained peak torque: 4x continuous -> 16x loss
    assert b.temp[0] > s.t_derate_c and b.derate()[0] < 1.0
    hot = run(b, s.tau_peak_nm, 0.0, 1.0)[-1]
    assert hot < s.tau_peak_nm                          # torque folded back
    run(b, 0.0, 0.0, 120.0)
    assert b.temp[0] < s.t_derate_c


def test_encoder_offset_bounded_by_half_backlash_and_quantised():
    b = ActuatorBank([PH54_200], dt=1e-3, thermal=False, seed=3)
    for _ in range(500):
        b.step(np.array([40.0]), np.array([0.5]))
    q = b.read_encoder(np.array([0.3]))[0]
    assert abs(q - 0.3) <= 0.5 * PH54_200.backlash_rad + 5 * PH54_200.enc_noise_rad + PH54_200.enc_resolution_rad
    for _ in range(500):
        b.step(np.array([-40.0]), np.array([-0.5]))
    assert b.off[0] < 0


def test_fractional_delay_with_jitter_stays_bounded_and_interpolates():
    d = DelayLine(1, 1e-3, delay_s=0.0025)
    outs = [d.step([float(i)])[0] for i in range(20)]
    assert outs[10] == pytest.approx(10 - 2.5)
    dj = DelayLine(1, 1e-3, delay_s=0.003, jitter_s=0.001, seed=1)
    o = np.array([dj.step([float(i)])[0] for i in range(300)])
    i = np.arange(10, 300)
    assert np.all(o[10:] <= i + 1e-9) and np.all(o[10:] >= i - 9.0 - 1e-9)   # delay clipped to [0, delay+6 sigma]


def test_humanoid_assignment_and_electrical_power_nonnegative():
    names = [f"{s}_{j}" for s in "LR" for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll")] + ["waist_yaw", "L_sh_pitch", "L_wr_roll"]
    b = humanoid_bank(names)
    assert b.tau_peak[3] == 44.7 and b.tau_peak[-2] == 25.3 and b.tau_peak[-1] == 8.4
    b.step(np.full(len(names), 5.0), np.full(len(names), 0.3))
    assert np.all(b.power_w >= 0)


def test_tradeoff_reproduces_documented_knee_finding():
    import json
    from tools.actuator_tradeoff import evaluate
    res = evaluate(json.load(open("results/fullbody/motor_requirements.json")))
    ph = res[0]["rows"]; by = {round(r["com_height_m"], 2): r for r in ph}
    assert not by[0.85]["feasible"] and not by[0.87]["feasible"]           # knee needs 52 / 47 Nm > 44.7
    assert by[0.89]["torque_margin_cont"] == pytest.approx(44.7 / 42.416 - 1, abs=0.01)
    assert not by[0.92]["feasible"]                                       # speed 3.70 > 3.47
    assert any(r["feasible"] for r in res[2]["rows"])                      # a faster/stronger actuator removes the constraint
