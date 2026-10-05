"""
planning/path_planner.py

HONEST SCOPE (read this before anything else): this is NOT perception or
SLAM. There is no sensor simulation and no mapping from sensor data - the
obstacle map is given directly (a KNOWN map), exactly like classical
"path planning" in the mobile-robotics literature (as opposed to
"SLAM", which is about BUILDING that map from sensor data in an
UNKNOWN environment). A real unmanned deployment would still need a
camera/LiDAR + a SLAM/localization stack to produce the map this module
consumes - that remains entirely unaddressed.

What this DOES do, for real: A* search (Hart, Nilsson, Raphael, 1968) on
a 2D occupancy grid with circular obstacles, returning a waypoint path
that `planning/autonomous_mission.py` then walks using the turning gait
in `planning/footstep_planner.plan_turning_footsteps`.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass


@dataclass
class Obstacle:
    x: float
    y: float
    radius_m: float


def _blocked(cx: float, cy: float, obstacles: list[Obstacle], margin_m: float) -> bool:
    return any(math.hypot(cx - o.x, cy - o.y) <= o.radius_m + margin_m for o in obstacles)


def plan_path(start_xy: tuple[float, float], goal_xy: tuple[float, float],
              obstacles: list[Obstacle] | None = None, cell_m: float = 0.15,
              margin_m: float = 0.25, bounds_margin_m: float = 1.0) -> list[tuple[float, float]]:
    """8-connected grid A*. Returns a list of (x, y) waypoints from start
    to goal, or [start_xy, goal_xy] directly when there are no obstacles
    (the common case, and the exact behavior tests/test_path_planner.py's
    regression check locks in). Raises ValueError if no path exists."""
    obstacles = obstacles or []
    if not obstacles:
        return [start_xy, goal_xy]

    xmin = min(start_xy[0], goal_xy[0]) - bounds_margin_m
    xmax = max(start_xy[0], goal_xy[0]) + bounds_margin_m
    ymin = min(start_xy[1], goal_xy[1]) - bounds_margin_m
    ymax = max(start_xy[1], goal_xy[1]) + bounds_margin_m

    def to_cell(p):
        return (round((p[0] - xmin) / cell_m), round((p[1] - ymin) / cell_m))

    def to_xy(c):
        return (xmin + c[0] * cell_m, ymin + c[1] * cell_m)

    start, goal = to_cell(start_xy), to_cell(goal_xy)
    nx, ny = int((xmax - xmin) / cell_m) + 2, int((ymax - ymin) / cell_m) + 2

    def h(c):
        return math.hypot(c[0] - goal[0], c[1] - goal[1])

    open_set = [(h(start), 0.0, start)]
    came_from = {}
    g_score = {start: 0.0}
    visited = set()
    neighbors = [(dx, dy, math.hypot(dx, dy)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 if not (dx == 0 and dy == 0)]

    while open_set:
        _, g, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            path_cells = [current]
            while current in came_from:
                current = came_from[current]
                path_cells.append(current)
            path_cells.reverse()
            return [start_xy] + [to_xy(c) for c in path_cells[1:-1]] + [goal_xy]

        for dx, dy, cost in neighbors:
            nb = (current[0] + dx, current[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if _blocked(*to_xy(nb), obstacles, margin_m):
                continue
            ng = g + cost * cell_m
            if ng < g_score.get(nb, math.inf):
                g_score[nb] = ng
                came_from[nb] = current
                heapq.heappush(open_set, (ng + h(nb) * cell_m, ng, nb))

    raise ValueError("no path found - goal may be unreachable given the obstacles/margin")


def simplify_path(waypoints: list[tuple[float, float]], angle_tol_deg: float = 8.0
                   ) -> list[tuple[float, float]]:
    """Collapse near-collinear consecutive waypoints (A* grid paths are
    jagged) into a small number of straight segments - what the turning
    gait actually needs (a heading per segment), not a dense grid trace."""
    if len(waypoints) <= 2:
        return waypoints
    out = [waypoints[0]]
    for i in range(1, len(waypoints) - 1):
        a, b, c = out[-1], waypoints[i], waypoints[i + 1]
        h1 = math.atan2(b[1] - a[1], b[0] - a[0])
        h2 = math.atan2(c[1] - b[1], c[0] - b[0])
        if abs(math.degrees(h1 - h2)) > angle_tol_deg:
            out.append(b)
    out.append(waypoints[-1])
    return out
