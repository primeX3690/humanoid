"""Grid navigation: inflated-obstacle A* over an ElevationMap traversability mask (8-connected, octile cost, slope penalty)."""
from __future__ import annotations
import heapq
import numpy as np
from scipy import ndimage as ndi


def inflate(blocked, res, radius_m):
    d = ndi.distance_transform_edt(~blocked) * res
    return d < radius_m


def astar(free, start, goal, cost_extra=None):
    """free: bool grid; start/goal: (i,j). Returns list of (i,j) or None."""
    n0, n1 = free.shape
    if not (free[start] and free[goal]): return None
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, 2 ** .5), (-1, 1, 2 ** .5), (1, -1, 2 ** .5), (1, 1, 2 ** .5)]
    h = lambda p: max(abs(p[0] - goal[0]), abs(p[1] - goal[1])) + (2 ** .5 - 1) * min(abs(p[0] - goal[0]), abs(p[1] - goal[1]))
    g = {start: 0.0}; came = {}; pq = [(h(start), start)]; closed = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed: continue
        if cur == goal:
            path = [cur]
            while cur in came: cur = came[cur]; path.append(cur)
            return path[::-1]
        closed.add(cur)
        for di, dj, c in moves:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < n0 and 0 <= nb[1] < n1) or not free[nb] or nb in closed: continue
            if di and dj and not (free[cur[0] + di, cur[1]] and free[cur[0], cur[1] + dj]): continue    # no corner cutting
            ng = g[cur] + c + (cost_extra[nb] if cost_extra is not None else 0.0)
            if ng < g.get(nb, np.inf):
                g[nb] = ng; came[nb] = cur; heapq.heappush(pq, (ng + h(nb), nb))
    return None


def plan_on_map(emap, start_xy, goal_xy, robot_radius=0.30, **trav):
    blocked = ~emap.traversable(**trav)
    free = ~inflate(blocked, emap.res, robot_radius)
    s = tuple(np.floor((np.asarray(start_xy) - emap.origin) / emap.res).astype(int)); gl = tuple(np.floor((np.asarray(goal_xy) - emap.origin) / emap.res).astype(int))
    slope_cost = np.clip(emap.slope_deg() / 20.0, 0, 1) * 2.0
    p = astar(free, s, gl, slope_cost)
    return None if p is None else np.array([emap.cell_xy(c) for c in p])
