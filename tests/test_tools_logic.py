import numpy as np
import pytest
from tools.tune_rl_gains import apply_scales, sweep, best
from tools.measure_design_torque import peaks_to_row
from hardware.mass_budget import ACTUATED_30
from rl.fullbody_env import CLASS_GAINS


def test_apply_scales_scales_kp_and_sqrt_kd_and_only_ankle_when_asked():
    t = apply_scales(CLASS_GAINS, kp_all=4.0)
    assert t["knee"][1] == pytest.approx(4 * CLASS_GAINS["knee"][1]) and t["knee"][2] == pytest.approx(2 * CLASS_GAINS["knee"][2]) and t["knee"][0] == CLASS_GAINS["knee"][0]
    a = apply_scales(CLASS_GAINS, kp_ankle_pitch=2.0)
    assert a["ankle_pitch"][1] == pytest.approx(2 * CLASS_GAINS["ankle_pitch"][1]) and a["knee"] == CLASS_GAINS["knee"]


def test_sweep_orchestration_restores_the_table_and_picks_the_best_standing_setting():
    state = {"t": dict(CLASS_GAINS)}
    def set_table(t): state["t"] = t
    def selfcheck():                                   # fake physics: stands iff the ankle kp >= 400
        ok = state["t"]["ankle_pitch"][1] >= 400
        return dict(fell=not ok, seconds_stood=5.0 if ok else 3.0, pelvis_z=0.85 if ok else 0.6, tilt=0.05 if ok else 0.9)
    def push(): return dict(fell=state["t"]["ankle_pitch"][1] < 800, seconds_stood=5.0 if state["t"]["ankle_pitch"][1] >= 800 else 2.0)
    rows = sweep(selfcheck, push, dict(CLASS_GAINS), [0.5, 1.0, 2.0], [1.0, 2.0], set_table)
    assert len(rows) == 5 and state["t"] == dict(CLASS_GAINS)
    b = best(rows); assert b is not None and not b["stand_fell"] and not b["push_fell"] and b["scale"] in (2.0, 2.0)
    assert best([dict(sweep="all", scale=1, stand_fell=True, stand_s=1, pelvis_z=0, tilt=1, push_fell=True, push_s=1)]) is None


def test_peaks_to_row_groups_joints_by_name():
    tau = np.zeros(30); qd = np.zeros(30)
    for i, n in enumerate(ACTUATED_30):
        if n == "R_knee": tau[i] = 51.0; qd[i] = 2.7
        if n == "L_hip_roll": tau[i] = 14.0
        if n == "waist_pitch": tau[i] = 9.0
        if n == "L_sh_pitch": tau[i] = 99.0                       # arm joints must NOT leak into leg/waist groups
    r = peaks_to_row(0.89, ACTUATED_30, tau, qd)
    assert r["knee_Nm"] == 51.0 and r["hip_roll_Nm"] == 14.0 and r["waist_Nm"] == 9.0 and r["leg_speed_rad_s"] == 2.7 and r["hip_pitch_Nm"] == 0.0
