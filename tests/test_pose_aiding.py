import numpy as np
import pytest
from estimation.base_eskf import BaseESKF
from estimation.pose_aiding import update_position, update_yaw, update_pose, widen_for_aiding, yaw_of, wrap, ContactDetector
from estimation.so3 import exp_so3


def run(aid, T=120.0, bg_true=0.004, seed=0, outlier=False, single=True):
    rng = np.random.default_rng(seed)
    f = BaseESKF(np.eye(3), np.array([0, 0, 0.9]), {"L": np.array([0, 0.1, 0]), "R": np.array([0, -0.1, 0])})
    if aid: widen_for_aiding(f)
    dt = 0.005
    for k in range(int(T / dt)):
        a_m = np.array([0, 0, 9.81]) + 0.04 * rng.standard_normal(3)
        w_m = np.array([0, 0, bg_true]) + 0.006 * rng.standard_normal(3)          # the robot is not rotating; gyro-z has a bias
        phase = (k * dt) % 1.0 < 0.5                       # alternating single support (like walking): feet re-anchor every half second
        c = {"L": True, "R": True} if not single else ({"L": True, "R": False} if phase else {"L": False, "R": True})
        f.predict(a_m, w_m, dt, c)
        f.update_foot_kinematics({"L": np.array([0, 0.1, -0.9]), "R": np.array([0, -0.1, -0.9])}, c)
        if aid and k % 200 == 0:                                                   # 1 Hz absolute fix
            ok = update_pose(f, np.array([0, 0, 0.9]) + 0.02 * rng.standard_normal(3), 0.0 + 0.02 * rng.standard_normal(), 0.02, 0.02)
        if aid and outlier and k == 2000:
            update_position(f, np.array([5.0, 5.0, 0.9]), 0.02)                     # a wild SLAM jump
    return f


def test_yaw_drifts_without_aiding_and_is_bounded_with_it():
    free = run(False); aided = run(True)
    assert abs(wrap(yaw_of(free.R))) > 0.05            # global yaw is unobservable once feet re-anchor: drifts (~0.14 rad / 2 min here)
    assert abs(wrap(yaw_of(aided.R))) < 0.02
    assert abs(aided.bg[2] - 0.004) < 0.0015            # and the gyro-z bias becomes observable


def test_position_fix_pulls_state_and_shrinks_covariance():
    f = BaseESKF(np.eye(3), np.zeros(3), {"L": np.array([0, .1, -.9]), "R": np.array([0, -.1, -.9])}); widen_for_aiding(f)
    p0 = f.P[0, 0]
    ok, _ = update_position(f, np.array([0.3, -0.2, 0.0]), 0.02)
    assert ok and abs(f.p[0] - 0.3) < 0.01 and f.P[0, 0] < p0 * 0.01


def test_outlier_fix_is_rejected_by_the_gate():
    f = run(True, T=15.0, outlier=True, single=False)
    assert np.linalg.norm(f.p[:2]) < 0.1


def test_yaw_update_handles_wraparound():
    f = BaseESKF(exp_so3(np.array([0, 0, np.pi - 0.01])), np.zeros(3), {"L": np.zeros(3), "R": np.zeros(3)}); widen_for_aiding(f)
    ok, _ = update_yaw(f, -np.pi + 0.01, 0.01)                          # +0.02 rad across the +-pi seam
    assert ok and abs(wrap(yaw_of(f.R) - (-np.pi + 0.01))) < 0.01


def test_contact_detector_hysteresis_cop_and_slip():
    d = ContactDetector(f_on=60, f_off=25, debounce=2)
    seq = [0, 0, 80, 80, 40, 40, 20, 20, 20]
    states = [d.update([0, 0, fz, 0, 0, 0])["contact"] for fz in seq]
    assert states == [False, False, False, True, True, True, True, False, False]   # on after 2 samples >60, stays on at 40, off after 2 < 25
    out = d.update([0, 0, 200, 4.0, -6.0, 0])                                      # tx=4 Nm, ty=-6 Nm at Fz=200 N
    assert np.allclose(out["cop"], [6.0 / 200, 4.0 / 200])
    d2 = ContactDetector(); [d2.update([0, 0, 300, 0, 0, 0]) for _ in range(3)]
    assert d2.update([250, 0, 300, 0, 0, 0])["slip"] and not d2.update([50, 0, 300, 0, 0, 0])["slip"]
