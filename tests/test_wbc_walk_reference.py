import numpy as np
import pytest
from planning.wbc_walk_reference import build_walk_reference
from planning.footstep_planner import GaitParams, plan_footsteps, plan_turning_footsteps, apply_terrain
from terrain.terrain_profile import StepTerrain, RampTerrain, FlatTerrain
from terrain.terrain_xml import terrain_geoms, terrain_to_xml, top_surface_z


def test_apply_terrain_preserves_heading_regression():
    g = GaitParams(n_steps=6)
    fs = plan_turning_footsteps(g, 0.1)
    out = apply_terrain(fs, StepTerrain(0.5, 0.05))
    assert [s.heading for s in out] == pytest.approx([s.heading for s in fs]) and abs(out[-1].heading) > 0.3
    assert out[-1].z == pytest.approx(0.05)


def test_straight_flat_reference_matches_v2_walk():
    from simulation.walk_simulator import simulate_walk
    from dynamics.lipm import LIPMParams
    ref = build_walk_reference(8, 0.85, 0.3)
    g = GaitParams(n_steps=8, step_length_m=0.3)
    w = simulate_walk(g, LIPMParams(com_height_m=0.85, dt=g.dt))
    n = min(len(ref["t"]), len(w.com_x))
    assert np.allclose(ref["com"][:n, 0], w.com_x[:n], atol=1e-9) and np.allclose(ref["com"][:n, 1], w.com_y[:n], atol=1e-9)
    assert np.all(ref["heading"] == 0) and np.allclose(ref["zc_ref"], 0.85)


def test_turning_reference_curves_and_heading_advances():
    ref = build_walk_reference(8, 0.85, 0.3, turn_rate=0.08)
    assert ref["heading"][-1] == pytest.approx(8 * 0.08, abs=1e-9)
    assert abs(ref["com"][-1, 1]) > 0.2                       # path bends sideways
    L, R = ref["foot"]["L"], ref["foot"]["R"]
    assert np.all(np.isfinite(L)) and np.all(np.isfinite(R))


def test_terrain_reference_raises_feet_and_com_height_reference():
    ref = build_walk_reference(8, 0.85, 0.3, terrain=StepTerrain(0.9, 0.06))
    assert ref["foot"]["L"][:, 2].max() == pytest.approx(0.06, abs=1e-6) or ref["foot"]["R"][:, 2].max() >= 0.06 - 1e-6
    assert ref["zc_ref"].max() == pytest.approx(0.85 + 0.06, abs=1e-6)


def test_step_xml_top_surface_matches_profile():
    t = StepTerrain(1.0, 0.08); g = terrain_geoms(t)[0]
    assert top_surface_z(g, 1.5) == pytest.approx(0.08) and "terrain_step" in terrain_to_xml(t)


def test_ramp_xml_top_surface_matches_profile():
    t = RampTerrain(1.0, 2.0, 0.1); gs = {q["name"]: q for q in terrain_geoms(t)}
    for x in (1.0, 1.25, 1.5, 2.0):
        assert top_surface_z(gs["terrain_ramp"], x) == pytest.approx(t.height_at(x, 0.0), abs=1e-6)
    assert top_surface_z(gs["terrain_plateau"], 3.0) == pytest.approx(0.1)
    assert terrain_geoms(FlatTerrain()) == []
