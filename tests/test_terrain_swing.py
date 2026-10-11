import numpy as np
import pytest
from terrain.terrain_profile import StepTerrain, RampTerrain, StairTerrain
from trajectory.terrain_swing import make_terrain_swing
from trajectory.stair_swing import stair_swing_position, toe_heel_clearance
from trajectory.swing_foot import swing_foot_position, SwingFootParams
from planning.wbc_walk_reference import build_walk_reference

SS = np.linspace(0, 1, 1001); TSW = 0.4


def traj(fn, a, b): return np.array([fn(s, a, b) for s in SS])


def peak_speed(P): return float(np.abs(np.gradient(P[:, 0], SS * TSW)).max())


@pytest.mark.parametrize("h", [0.04, 0.08, 0.12])
def test_clears_the_step_edge_by_construction_with_unchanged_foot_speed(h):
    T = StepTerrain(0.9, h); a, b = np.array([0.75, -0.06, 0.0]), np.array([1.35, -0.06, h])      # the 0.6 m alternate-foot swing that failed on MuJoCo
    P = traj(make_terrain_swing(T), a, b)
    legacy = traj(lambda s, p, q: swing_foot_position(s, p, q, SwingFootParams()), a, b); stair = traj(lambda s, p, q: stair_swing_position(s, p, q), a, b)
    assert toe_heel_clearance(P, T) >= -0.002 and toe_heel_clearance(legacy, T) < -0.003 if h == 0.04 else toe_heel_clearance(P, T) >= -0.002
    assert peak_speed(P) <= 1.1 * peak_speed(legacy) and peak_speed(stair) > 1.5 * peak_speed(legacy)      # the stair swing is what made the foot too fast here


def test_end_points_exact_flat_swing_is_gentle_and_z_is_continuous():
    T = StepTerrain(5.0, 0.1); a, b = np.array([0.0, 0.1, 0.0]), np.array([0.3, 0.1, 0.0]); P = traj(make_terrain_swing(T), a, b)
    assert np.allclose(P[0], a) and np.allclose(P[-1], b) and 0.025 < P[:, 2].max() < 0.06
    assert np.abs(np.diff(P[:, 2])).max() < 0.003 and np.abs(np.diff(P[:, 0])).max() < 0.01


def test_ramp_swing_follows_the_slope_without_clipping():
    T = RampTerrain(0.6, 1.8, 0.1); a, b = np.array([0.3, 0.06, T.height_at(0.3, 0)]), np.array([0.9, 0.06, T.height_at(0.9, 0)])
    P = traj(make_terrain_swing(T), a, b)
    assert toe_heel_clearance(P[50:-50], T) >= -0.002        # interior of the swing; at the end points a FLAT sole on a slope is inherently ~8 mm below the toe-side terrain (foot pitch is the WBC's job)


def test_walk_reference_with_step_uses_terrain_swing_and_every_swing_clears():
    for h in (0.04, 0.08):
        T = StepTerrain(0.9, h); ref = build_walk_reference(6, 0.85, 0.3, terrain=T)
        for side in "LR": assert toe_heel_clearance(ref["foot"][side], T) >= -0.003, (h, side)


def test_step_terrain_is_run_through_the_stairs_pipeline_in_walk_wbc():
    import types, sys
    if "mujoco" not in sys.modules:
        try: import mujoco  # noqa: F401
        except ImportError: sys.modules["mujoco"] = types.ModuleType("mujoco")
    from simulation.walk_wbc import build_plan
    plan = build_plan(6, 0.85, 0.3, terrain=StepTerrain(0.9, 0.04))
    assert "stairs" in plan and plan["stairs"].n == 1 and plan["stairs"].riser_m == 0.04
    for side in "LR": assert plan["foot"][side][-1, 2] == pytest.approx(0.04, abs=1e-6) and toe_heel_clearance(plan["foot"][side], StepTerrain(0.9, 0.04)) >= -0.004
