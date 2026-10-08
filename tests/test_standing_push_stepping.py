"""Capture-point stepping recovers from a push that the standing (ankle/hip) strategy cannot survive."""
from simulation.push_recovery_exp import run


def test_stepping_survives_push_that_topples_standing_controller():
    standing = run(110, (1, 0), stepping=False, T=3.0)
    stepping = run(110, (1, 0), stepping=True, T=3.5)
    assert standing["fell"]
    assert not stepping["fell"]
    assert 1 <= stepping["steps"] <= 4
    assert stepping["final_com_vel"] < 0.05                      # came to rest
    for s in stepping["step_log"]:                               # feet land where they were commanded
        assert abs(s["landed"][0] - s["target"][0]) < 0.03 and abs(s["landed"][1] - s["target"][1]) < 0.03


def test_lateral_crossover_step_survives_90N():
    r = run(90, (0, 1), stepping=True, T=3.5)
    assert not r["fell"] and r["steps"] >= 1 and r["final_com_vel"] < 0.05
