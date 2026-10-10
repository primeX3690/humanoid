import numpy as np
import pytest
from dataclasses import replace
from control.lateral_recovery import (RecoveryParams, plan_recovery, simulate_recovery, max_capturable_speed, hull_signed_margin, rect)

ST = np.array([0.0, -0.12]); SW = np.array([0.0, 0.12])       # right foot stance, left foot swinging; +y is the swing side
P = RecoveryParams()


def test_small_push_toward_swing_side_needs_no_crossover_and_recovers_in_simulation():
    com = np.array([0.0, 0.0]); vel = np.array([0.0, 0.25])
    r = plan_recovery(com, vel, ST, SW, P)
    assert r["feasible"] and not r["crossover"]
    s = simulate_recovery(com, vel, ST, r, P)
    assert s["recovered"]


def test_push_away_from_swing_side_requires_crossover_then_recovers():
    com = np.array([0.0, -0.12]); vel = np.array([0.0, -0.3])       # CoM falling outward past the stance foot
    nocross = plan_recovery(com, vel, ST, SW, replace(P, allow_crossover=False))
    cross = plan_recovery(com, vel, ST, SW, P)
    assert not nocross["feasible"] and cross["feasible"] and cross["crossover"]
    assert cross["swing"].feasible and cross["swing"].min_ankle_gap >= -1e-6
    assert simulate_recovery(com, vel, ST, cross, P)["recovered"]


def test_crossover_and_cop_shift_each_extend_the_capturable_speed():
    d = np.array([0.0, -1.0]); com = ST.copy()
    base = max_capturable_speed(d, ST, SW, replace(P, allow_crossover=False, use_cop_shift=False), com)
    cop = max_capturable_speed(d, ST, SW, replace(P, allow_crossover=False, use_cop_shift=True), com)
    full = max_capturable_speed(d, ST, SW, P, com)
    assert cop >= base and full > base * 1.2 and full >= cop


def test_planner_feasible_means_simulation_recovers():
    rng = np.random.default_rng(0); ok = 0
    for _ in range(40):
        ang = rng.uniform(0, 2 * np.pi); sp = rng.uniform(0.05, 0.8)
        com = ST + rng.uniform(-0.02, 0.02, 2); vel = sp * np.array([np.cos(ang), np.sin(ang)])
        r = plan_recovery(com, vel, ST, SW, P)
        if r["feasible"]:
            assert simulate_recovery(com, vel, ST, r, P)["recovered"], (vel, r["margin"])
            ok += 1
    assert ok >= 10


def test_hopeless_push_is_reported_infeasible():
    r = plan_recovery(ST.copy(), np.array([0.0, -4.0]), ST, SW, P)
    assert not r["feasible"]


def test_hull_margin_sign():
    v = np.vstack([rect(ST, P), rect(SW, P)])
    assert hull_signed_margin([0.0, 0.0], v) > 0 and hull_signed_margin([0.0, 0.5], v) < 0
