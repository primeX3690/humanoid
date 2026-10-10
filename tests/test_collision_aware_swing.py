import numpy as np
import pytest
from planning.collision_aware_swing import (SwingGeometry, plan_collision_free_swing, clearance, ankle_gap, foot_gap, project_goal)

G = SwingGeometry()
STANCE = np.array([0.0, 0.0, 0.0])


def naive(start, goal, T=0.5, n=101):
    u = np.linspace(0, 1, n)[:, None]; return np.asarray(start) * (1 - u) + np.asarray(goal) * u


def test_normal_outward_step_goes_direct_and_is_clear():
    p = plan_collision_free_swing([0.0, 0.24, 0], [0.3, 0.24, 0], STANCE)
    assert p.kind == "direct" and p.feasible and p.min_ankle_gap >= 0


def test_crossover_naive_line_collides_planned_path_does_not():
    start, goal = np.array([0.0, 0.14, 0]), np.array([0.18, -0.14, 0])
    ag, _ = clearance(naive(start, goal), STANCE, G)
    assert ag < 0                                               # straight line drives the ankle through the stance leg
    p = plan_collision_free_swing(start, goal, STANCE)
    assert p.feasible and p.kind in ("front", "back") and p.min_ankle_gap >= -1e-6
    assert np.allclose(p.pos[0], start) and np.allclose(p.pos[-1], goal)


def test_endpoint_velocity_and_acceleration_are_zero():
    p = plan_collision_free_swing([0.0, 0.14, 0], [0.18, -0.14, 0], STANCE)
    assert np.linalg.norm(p.vel[0]) < 1e-6 and np.linalg.norm(p.vel[-1]) < 1e-6
    assert np.linalg.norm(p.acc[0]) < 0.05 * np.abs(p.acc).max() and np.linalg.norm(p.acc[-1]) < 0.05 * np.abs(p.acc).max()


def test_goal_inside_stance_foot_is_projected_out():
    goal, moved = project_goal(np.array([0.02, -0.02, 0]), STANCE, G, side_hint=-1.0)
    assert moved and ankle_gap(goal, STANCE, G) >= 0 and foot_gap(goal, STANCE, G) >= G.margin


def test_planner_never_reports_feasible_when_clearance_violated():
    for sy in (0.12, 0.2, -0.15):
        for gy in (-0.12, 0.0, 0.1, -0.2):
            for gx in (-0.2, 0.0, 0.25):
                p = plan_collision_free_swing([0.0, sy, 0], [gx, gy, 0], STANCE)
                if p.feasible:
                    assert p.min_ankle_gap >= -1e-6 and p.min_overlap_height_gap >= -1e-6


def test_reach_limit_is_respected():
    p = plan_collision_free_swing([0.0, 0.2, 0], [1.2, 0.2, 0], STANCE)
    assert not p.feasible
