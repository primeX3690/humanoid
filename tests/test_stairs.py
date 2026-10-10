import numpy as np
import pytest
from terrain.terrain_profile import StairTerrain
from terrain.terrain_xml import terrain_geoms, terrain_to_xml, top_surface_z
from trajectory.stair_swing import StairSwingParams, stair_swing_position, toe_heel_clearance, mj
from trajectory.swing_foot import swing_foot_position, SwingFootParams
from planning.stairs_reference import plan_stair_footsteps, build_stairs_reference
from planning.footstep_planner import GaitParams

ST = StairTerrain(first_x=0.5, tread_m=0.28, riser_m=0.12, n=4)


def traj(fn, a, b, n=200): return np.array([fn(s, a, b) for s in np.linspace(0, 1, n)])


def test_stair_profile_heights():
    assert ST.height_at(0.4, 0) == 0 and ST.height_at(0.6, 0) == pytest.approx(0.12) and ST.height_at(0.9, 0) == pytest.approx(0.24) and ST.height_at(9.0, 0) == pytest.approx(0.48)


def test_legacy_swing_clips_the_stair_edge_but_stair_swing_clears_it():
    a, b = np.array([0.30, 0.1, 0.0]), np.array([0.64, 0.1, 0.12])           # toe region crosses the first riser at x=0.5
    old = traj(lambda s, p, q: swing_foot_position(s, p, q, SwingFootParams()), a, b)
    new = traj(lambda s, p, q: stair_swing_position(s, p, q), a, b)
    assert toe_heel_clearance(old, ST) < -0.02          # legacy profile drives the foot through the nosing
    assert toe_heel_clearance(new, ST) >= -0.003


def test_stair_swing_is_at_rest_at_the_ends_and_continuous():
    a, b = np.array([0.0, 0, 0.0]), np.array([0.3, 0, 0.12]); T = traj(lambda s, p, q: stair_swing_position(s, p, q), a, b, 2001)
    v = np.diff(T, axis=0) * 2000
    assert np.allclose(T[0], a) and np.allclose(T[-1], b) and np.linalg.norm(v[0]) < 0.05 and np.linalg.norm(v[-1]) < 0.05
    assert np.abs(np.diff(v, axis=0)).max() < 0.5          # no velocity jumps


def test_xml_top_surfaces_match_profile_for_all_stairs():
    gs = terrain_geoms(ST); assert len(gs) == 4
    for k, g in enumerate(gs): assert top_surface_z(g, ST.first_x + (k + 0.5) * ST.tread_m) == pytest.approx((k + 1) * ST.riser_m)
    assert terrain_to_xml(ST).count("<geom") == 4


def test_footsteps_land_on_tread_centres_at_riser_heights():
    g = GaitParams(n_steps=2 + 8, step_width_m=0.2, step_duration_s=1.0, double_support_s=0.5)
    fs = plan_stair_footsteps(ST, g, approach=2)
    for i, f in enumerate(fs[2:]):
        k = i // 2
        assert f.x == pytest.approx(ST.first_x + (k + 0.5) * ST.tread_m) and f.z == pytest.approx((k + 1) * ST.riser_m)
        assert ST.height_at(f.x, f.y) == pytest.approx(f.z)
    assert fs[0].z == 0 and fs[0].x < ST.first_x


def test_reference_feet_clear_every_stair_and_com_climbs():
    ref = build_stairs_reference(ST, n_stairs=3)
    for side in "LR":
        T = ref["foot"][side]
        assert toe_heel_clearance(T, ST) >= -0.004
        assert T[-1, 2] == pytest.approx(3 * ST.riser_m, abs=1e-6)
    assert ref["zc_ref"][-1] > ref["zc_ref"][0] + 0.25 and ref["com"][-1, 0] > ST.first_x
    assert np.all(np.isfinite(ref["com"])) and np.abs(ref["com"][:, 1]).max() < 0.2
