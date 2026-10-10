import numpy as np
import pytest
from sim_realism.domain_randomization import RandomizationConfig, sample, push_schedule, apply_to_bank
from sim_realism.realism_layer import RealismLayer
from actuators.joint_actuator import PH54_200, MX106

BODIES = ["pelvis", "torso", "L_thigh", "R_thigh"]
JOINTS = ["L_knee", "R_knee", "L_wr_roll"]


def test_sample_is_reproducible_and_within_ranges():
    cfg = RandomizationConfig()
    a, b = sample(cfg, BODIES, JOINTS, seed=5), sample(cfg, BODIES, JOINTS, seed=5)
    assert np.allclose(a["mass_scale"], b["mass_scale"]) and a["cmd_delay_s"] == b["cmd_delay_s"]
    for s in range(50):
        d = sample(cfg, BODIES, JOINTS, seed=s)
        assert cfg.friction_range[0] <= d["friction"] <= cfg.friction_range[1]
        assert np.all((d["torque_scale"] >= 0.85) & (d["torque_scale"] <= 1.0))
        assert np.all(np.abs(d["com_offset"]) <= cfg.com_offset_m)
        assert cfg.cmd_delay_range_s[0] <= d["cmd_delay_s"] <= cfg.cmd_delay_range_s[1]


def test_mass_scale_is_unbiased_lognormal():
    cfg = RandomizationConfig(mass_sigma=0.1)
    m = np.concatenate([sample(cfg, BODIES, JOINTS, seed=s)["mass_scale"] for s in range(400)])
    assert abs(np.mean(np.log(m))) < 0.02 and 0.08 < np.std(np.log(m)) < 0.12


def test_bank_realises_sample_weaker_motors_and_delay():
    d = sample(RandomizationConfig(), BODIES, JOINTS, seed=1)
    bank = apply_to_bank([PH54_200, PH54_200, MX106], d)
    assert np.all(bank.torque_scale == d["torque_scale"])
    out = bank.step(np.array([1e3, 1e3, 1e3]), np.zeros(3))
    assert np.allclose(out, 0.0, atol=1e-6)      # command delay: nothing arrives on the first tick
    for _ in range(40): out = bank.step(np.array([1e3, 1e3, 1e3]), np.zeros(3))
    assert np.all(out <= bank.tau_peak * 1.0 + 1e-9) and out[0] <= 44.7 * d["torque_scale"][0] + 1e-6


def test_push_schedule_poisson_rate_and_force_bounds():
    cfg = RandomizationConfig(push_prob_per_s=0.5)
    n = [len(push_schedule(cfg, 100.0, seed=s)) for s in range(40)]
    assert 40 < np.mean(n) < 60
    for t, dur, f in push_schedule(cfg, 100.0, seed=0):
        assert cfg.push_force_range_n[0] <= np.linalg.norm(f) <= cfg.push_force_range_n[1] + 1e-9


def test_realism_layer_velocity_estimate_tracks_true_velocity():
    from actuators.joint_actuator import ActuatorBank, ideal
    bank = ActuatorBank([ideal(PH54_200)], dt=1e-3, thermal=False)
    L = RealismLayer(bank, vel_cutoff_hz=80.0)
    for i in range(500):
        L.tick(np.array([0.5 * i * 1e-3]))
    assert L.qd_meas[0] == pytest.approx(0.5, rel=0.02)
