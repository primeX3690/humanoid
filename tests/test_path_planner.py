import math
import pytest
from planning.path_planner import plan_path, simplify_path, Obstacle


def test_no_obstacles_gives_direct_path():
    assert plan_path((0, 0), (4, 0), []) == [(0, 0), (4, 0)]


def test_path_avoids_a_blocking_obstacle():
    obs = [Obstacle(x=2.0, y=0.0, radius_m=0.6)]
    path = plan_path((0, 0), (4, 0), obs, margin_m=0.25)
    for x, y in path:
        assert math.hypot(x - 2.0, y - 0.0) >= 0.6 + 0.25 - 1e-6
    assert path[0] == (0, 0) and path[-1] == (4, 0)


def test_unreachable_goal_raises_not_silently_returns_a_bad_path():
    ring = [Obstacle(x=2.0, y=0.0, radius_m=r) for r in [0.05]] * 0  # placeholder, real ring below
    obs = [Obstacle(x=2.0 + 0.3 * math.cos(a), y=0.3 * math.sin(a), radius_m=0.2)
           for a in [i * math.pi / 4 for i in range(8)]]
    with pytest.raises(ValueError):
        plan_path((2.0, 0.0), (10.0, 0.0), obs, margin_m=0.1, cell_m=0.1)


def test_simplify_path_reduces_collinear_points():
    straight = [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert simplify_path(straight) == [(0, 0), (3, 0)]
