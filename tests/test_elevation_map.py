import numpy as np
import pytest
from perception.elevation_map import ElevationMap, ElevationMapTerrain
from perception.nav_planner import astar, inflate, plan_on_map
from terrain.terrain_profile import StepTerrain, RampTerrain
from planning.footstep_planner import GaitParams, plan_footsteps, apply_terrain


def cloud(terrain, n=60000, seed=0, box=((0.0, 3.0), (-1.0, 1.0)), outliers=0.02, sensor=(-0.5, 0.0, 1.0)):
    r = np.random.default_rng(seed)
    x = r.uniform(*box[0], n); y = r.uniform(*box[1], n)
    z = np.array([terrain.height_at(a, b) for a, b in zip(x, y)])
    d = np.linalg.norm(np.c_[x, y, z] - np.asarray(sensor), axis=1)
    z = z + (0.004 + 0.004 * d) * r.standard_normal(n)
    bad = r.random(n) < outliers; z[bad] += r.uniform(0.2, 1.5, bad.sum())      # spikes (flying pixels)
    return np.c_[x, y, z], sensor


def build(terrain, **kw):
    m = ElevationMap(size=(3.5, 2.5), res=0.04, origin=(-0.25, -1.25))
    for s in range(4):
        pts, sensor = cloud(terrain, seed=s, **kw); m.integrate(pts, sensor)
    return m


def test_step_height_recovered_within_5mm_and_edge_located():
    t = StepTerrain(1.2, 0.08); m = build(t)
    assert abs(m.height_at(0.4, 0.0) - 0.0) < 0.006 and abs(m.height_at(2.2, 0.3) - 0.08) < 0.006
    edges = m.step_edges(0.04); ij = np.argwhere(edges); xs = m.cell_xy(ij)[:, 0]
    assert abs(np.median(xs) - 1.2) < 0.06


def test_outlier_spikes_do_not_corrupt_the_map():
    m = build(RampTerrain(0.8, 1.8, 0.10), outliers=0.05)
    err = [abs(m.height_at(x, y) - RampTerrain(0.8, 1.8, 0.10).height_at(x, y)) for x in np.linspace(0.1, 2.5, 25) for y in (-0.5, 0.0, 0.5)]
    assert np.median(err) < 0.005 and np.max(err) < 0.03


def test_slope_and_traversability_masks():
    m = build(RampTerrain(0.8, 1.8, 0.20))                    # atan(0.2/1.0) = 11.3 deg
    sl = m.slope_deg(); mid = sl[int((1.3 + 0.25) / 0.04), 30]
    assert abs(mid - 11.3) < 2.5
    assert m.traversable(max_slope_deg=15)[int(1.3 / 0.04), 30] and not m.traversable(max_slope_deg=6)[int(1.3 / 0.04), 30]
    s = build(StepTerrain(1.2, 0.18))
    tr = s.traversable(max_step_m=0.10)
    assert not tr[int((1.2 + 0.25) / 0.04), 30] and tr[int(0.5 / 0.04), 30]


def test_planner_places_feet_on_perceived_terrain():
    m = build(StepTerrain(0.9, 0.06)); fs = apply_terrain(plan_footsteps(GaitParams(n_steps=8)), ElevationMapTerrain(m))
    assert max(f.z for f in fs) == pytest.approx(0.06, abs=0.01) and fs[0].z == pytest.approx(0.0, abs=0.01)


def test_astar_goes_around_inflated_obstacle_and_reports_blocked():
    free = np.ones((40, 40), bool); blocked = np.zeros((40, 40), bool); blocked[10:30, 20] = True
    fr = ~inflate(blocked, 0.1, 0.2)
    p = astar(fr, (20, 5), (20, 35)); assert p is not None
    assert all(fr[c] for c in p) and max(abs(c[0] - 20) for c in p) > 8        # had to detour around the wall
    wall = np.zeros((40, 40), bool); wall[:, 20] = True
    assert astar(~inflate(wall, 0.1, 0.2), (20, 5), (20, 35)) is None
    assert len(astar(np.ones((40, 40), bool), (0, 0), (0, 39))) == 40          # open grid: straight line


def test_plan_on_map_avoids_high_step():
    t = StepTerrain(1.2, 0.25); m = build(t)       # 25 cm curb across the whole width -> no traversable route
    assert plan_on_map(m, (0.2, 0.0), (2.5, 0.0), robot_radius=0.1, max_step_m=0.10) is None
    flat = build(StepTerrain(9.0, 0.0))
    p = plan_on_map(flat, (0.2, 0.0), (2.5, 0.0), robot_radius=0.1)
    assert p is not None and abs(p[-1][0] - 2.5) < 0.1
