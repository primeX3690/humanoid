"""
terrain/terrain_profile.py

Terrain height profiles: given a footstep's (x, y) landing position,
what's the ground height there? Used by
planning/footstep_planner.apply_terrain() to place footsteps at the
correct real height, and by trajectory/swing_foot.py (already supports
different start/end z - see its module docstring - since it was built
to interpolate z_base between the liftoff and landing positions, not
just add a fixed clearance hump on top of a flat z=0).

Honest scope limitation (see docs/SCOPE.md): the LIPM itself
(dynamics/lipm.py) still assumes a CONSTANT CoM height above the CURRENT
stance foot - it does not re-derive its own dynamics for a
continuously-varying ground height. What this module adds is correct
FOOTSTEP PLACEMENT and SWING TRAJECTORY height on real, uneven terrain;
a fully terrain-adaptive CoM-height LIPM (or switching to nonlinear
centroidal-dynamics MPC) is a further, harder step, not silently
included here.
"""
from __future__ import annotations

from dataclasses import dataclass


class TerrainProfile:
    def height_at(self, x: float, y: float) -> float:
        raise NotImplementedError


class FlatTerrain(TerrainProfile):
    """The default, original assumption everywhere else in this project -
    ground height is always exactly 0."""
    def height_at(self, x: float, y: float) -> float:
        return 0.0


@dataclass
class StepTerrain(TerrainProfile):
    """A single up-step (like a curb or stair) at x = step_x: ground is at
    z=0 before it, z=step_height_m after it."""
    step_x: float
    step_height_m: float

    def height_at(self, x: float, y: float) -> float:
        return self.step_height_m if x >= self.step_x else 0.0


@dataclass
class RampTerrain(TerrainProfile):
    """A linear ramp/slope between x=ramp_start and x=ramp_end, rising by
    a total of rise_m over that horizontal distance - a real, simple
    incline profile (e.g. a wheelchair ramp gradient)."""
    ramp_start_x: float
    ramp_end_x: float
    rise_m: float

    def height_at(self, x: float, y: float) -> float:
        if x <= self.ramp_start_x:
            return 0.0
        if x >= self.ramp_end_x:
            return self.rise_m
        frac = (x - self.ramp_start_x) / (self.ramp_end_x - self.ramp_start_x)
        return frac * self.rise_m


@dataclass
class StairTerrain(TerrainProfile):
    """n identical stairs: tread `tread_m` deep, riser `riser_m` high, first riser at x = first_x. Flat (z=0) before, plateau after the last."""
    first_x: float
    tread_m: float = 0.28
    riser_m: float = 0.12
    n: int = 4

    def height_at(self, x: float, y: float) -> float:
        if x < self.first_x:
            return 0.0
        k = int((x - self.first_x) // self.tread_m) + 1
        return min(k, self.n) * self.riser_m
