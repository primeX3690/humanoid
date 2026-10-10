import numpy as np
import pytest
from planning.footstep_planner import GaitParams, plan_footsteps, apply_terrain
from control.preview_nmpc import simulate_preview, PreviewNMPCConfig
from terrain.terrain_profile import StepTerrain


def test_lateral_tracking_matches_the_reference_closely_fixes_the_old_xfail():
    g = GaitParams(n_steps=2); r = simulate_preview(g, plan_footsteps(g))
    assert np.max(np.abs(r["com_y"] - r["zmp_ref_y"])) < 0.15            # the old controller: +-1 m. This one:
    assert np.max(np.abs(r["com_y"] - r["zmp_ref_y"])) < 0.06
    assert r["zmp_in_support"].mean() > 0.99


def test_long_walk_stays_bounded_and_inside_support():
    g = GaitParams(n_steps=6); r = simulate_preview(g, plan_footsteps(g))
    assert np.all(np.isfinite(r["com_x"])) and np.all(np.abs(r["com_y"]) < 0.12)
    assert np.all((r["com_z"] > 0.6) & (r["com_z"] < 0.95)) and r["zmp_in_support"].mean() > 0.98


def test_com_moves_forward_with_the_footsteps():
    g = GaitParams(n_steps=6); fs = plan_footsteps(g); r = simulate_preview(g, fs)
    assert r["com_x"][-1] > 0.5 * fs[-1].x


def test_height_follows_a_step_up_terrain():
    g = GaitParams(n_steps=6); fs = apply_terrain(plan_footsteps(g), StepTerrain(0.5, 0.06)); r = simulate_preview(g, fs)
    assert r["com_z"][-1] > r["com_z"][0] + 0.03 and np.all(np.isfinite(r["com_z"]))
