import numpy as np
from control.lateral_recovery import RecoveryParams, choose_recovery_step

P = RecoveryParams()
PL, PR = np.array([0.0, 0.12]), np.array([0.0, -0.12])


def test_forward_push_steps_forward_and_is_feasible():
    r = choose_recovery_step(np.zeros(2), np.array([0.5, 0.0]), PL, PR, P)
    assert r["feasible"] and r["target"][0] > 0.1 and not r["crossover"]


def test_outward_push_over_right_foot_uses_right_stance_and_left_crossover():
    r = choose_recovery_step(np.array([0.0, -0.12]), np.array([0.0, -0.3]), PL, PR, P)
    assert r["feasible"] and r["stance"] == "R" and r["side"] == "L" and r["crossover"]
    assert r["target"][1] < PR[1]                                     # lands beyond the stance foot
    assert r["swing"].min_ankle_gap >= -1e-6


def test_mirror_symmetry_left_right():
    a = choose_recovery_step(np.array([0.0, -0.12]), np.array([0.1, -0.3]), PL, PR, P)
    b = choose_recovery_step(np.array([0.0, 0.12]), np.array([0.1, 0.3]), PL, PR, P)
    assert a["stance"] == "R" and b["stance"] == "L"
    assert abs(a["margin"] - b["margin"]) < 1e-6 and abs(a["T"] - b["T"]) < 1e-9


def test_hopeless_push_returns_infeasible_not_exception():
    r = choose_recovery_step(np.zeros(2), np.array([3.0, 3.0]), PL, PR, P)
    assert r is not None and not r["feasible"]
