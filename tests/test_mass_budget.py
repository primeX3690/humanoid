import numpy as np, pytest
from hardware.mass_budget import mass_budget, avg_power_w, battery_for, time_to_derate_s, ACTUATED_30, actuator_mass
from actuators.joint_actuator import PH54_200, ActuatorBank, ideal


def test_actuator_inventory_and_mass_gap():
    assert len(ACTUATED_30) == 30
    m, _ = actuator_mass(); assert 17 < m < 22
    b = mass_budget(); assert b["left_for_structure_kg"] < 5.0 and not b["feasible"]
    from actuators.joint_actuator import MX106, spec_for_joint
    light = lambda n: MX106 if any(k in n for k in ("sh_", "elbow", "wr_", "neck")) else spec_for_joint(n)
    assert mass_budget(spec_fn=light)["feasible"]


def test_battery_scales_linearly_with_runtime_and_power():
    a, b = battery_for(300, 30), battery_for(300, 60); assert b["energy_wh"] == pytest.approx(2 * a["energy_wh"])
    assert avg_power_w(leg_torque_frac=0.8) > avg_power_w(leg_torque_frac=0.3)


def test_closed_form_time_to_derate_matches_simulation():
    s = type(PH54_200)(**{**PH54_200.__dict__, "thermal_tau_s": 60.0, "tau_electrical_s": 0.0, "tau_peak_nm": 100.0})
    tq = 1.3 * s.tau_cont_nm; t_cf = time_to_derate_s(s, tq)
    bank = ActuatorBank([s], dt=0.01, speed_limit=False); t = 0.0
    while bank.temp[0] < s.t_derate_c and t < 2000:
        bank.step(np.array([tq]), np.array([0.0])); t += 0.01
    assert t == pytest.approx(t_cf, rel=0.03)
    assert time_to_derate_s(s, 0.9 * s.tau_cont_nm) == float("inf")
