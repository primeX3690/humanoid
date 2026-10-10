import pytest
import numpy as np
from estimation.so3 import exp_so3, log_so3, R_to_quat, quat_to_R
from estimation.lipm_disturbance_kf import LIPMDisturbanceKF


def test_so3_roundtrips():
    rng = np.random.default_rng(0)
    for _ in range(20):
        w = rng.normal(size=3)
        assert np.allclose(log_so3(exp_so3(w)), w, atol=1e-9) if np.linalg.norm(w) < 3 else True
        R = exp_so3(w)
        assert np.allclose(quat_to_R(R_to_quat(R)), R, atol=1e-9)


@pytest.mark.mujoco
@pytest.mark.slow
def test_eskf_beats_dead_reckoning_by_orders_of_magnitude():
    from simulation.estimation_sim import run_estimation_experiment
    rec, est, sens = run_estimation_experiment(T=5.0, seed=1)
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    assert rms(rec["ez_est"]) < 0.005                       # height within 5 mm RMS
    assert rms(rec["ev_est"]) < 0.05                        # velocity within 5 cm/s RMS
    assert np.degrees(rms(rec["etilt_est"])) < 1.0
    assert rms(rec["ez_dr"]) > 20 * rms(rec["ez_est"])      # dead reckoning is drastically worse
    assert np.mean(np.abs(rec["ev_vec"]) < 3 * rec["sig_v"]) > 0.9   # filter covariance is honest


def test_lipm_kf_estimates_external_force():
    """Closed-loop (capture-point stabilised) LIPM + unknown push; KF must recover the push acceleration from c and ZMP only."""
    rng = np.random.default_rng(0)
    dt, zc, g = 0.005, 0.92, 9.81; w = np.sqrt(g / zc); w2 = w * w
    kf = LIPMDisturbanceKF(zc=zc)
    c, v = 0.0, 0.0; est_d = []
    for i in range(1600):
        t = i * dt
        d = 0.5 if 1.0 <= t < 5.0 else 0.0              # external push, expressed as acceleration (F / m)
        zmp = 3.0 * (c + v / w)                          # capture-point feedback: poles at -w, -2w
        a = w2 * (c - zmp) + d
        c += v * dt + 0.5 * a * dt * dt; v += a * dt
        kf.step(c + rng.normal(0, 0.003), zmp, dt)
        est_d.append(kf.x[2])
    est_d = np.array(est_d)
    assert abs(np.mean(est_d[int(3.0 / dt):int(4.5 / dt)]) - 0.5) < 0.15
    assert abs(np.mean(est_d[int(6.5 / dt):])) < 0.15       # and it returns to zero after the push ends
