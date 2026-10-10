"""Turn a TerrainProfile (terrain/terrain_profile.py) into static MuJoCo geoms, so a WBC walk can actually step on what the planner assumed.

    from terrain.terrain_xml import terrain_to_xml
    make_model(params, table=False, terrain_xml=terrain_to_xml(StepTerrain(1.0, 0.08)))
Step -> one box; ramp -> one tilted box + a top plateau. Pure Python (testable without MuJoCo).
"""
from __future__ import annotations
import numpy as np
from terrain.terrain_profile import StepTerrain, RampTerrain, FlatTerrain, TerrainProfile

THICK = 0.2


def terrain_geoms(terrain: TerrainProfile, x_max=8.0, half_y=1.5):
    """List of dicts(name, kind, pos, half, euler_deg). A box's TOP face defines the walkable surface."""
    g = []
    if isinstance(terrain, FlatTerrain):
        return g
    if isinstance(terrain, StepTerrain):
        L = x_max - terrain.step_x
        g.append(dict(name="terrain_step", pos=(terrain.step_x + L / 2, 0.0, terrain.step_height_m - THICK / 2),
                      half=(L / 2, half_y, THICK / 2), euler_deg=(0, 0, 0)))
    elif isinstance(terrain, RampTerrain):
        run = terrain.ramp_end_x - terrain.ramp_start_x; rise = terrain.rise_m
        th = np.arctan2(rise, run); n = np.array([-np.sin(th), 0.0, np.cos(th)])
        mid = np.array([(terrain.ramp_start_x + terrain.ramp_end_x) / 2, 0.0, rise / 2])
        c = mid - (THICK / 2) * n
        g.append(dict(name="terrain_ramp", pos=tuple(c), half=(np.hypot(run, rise) / 2, half_y, THICK / 2),
                      euler_deg=(0, -float(np.degrees(th)), 0)))
        L = x_max - terrain.ramp_end_x
        g.append(dict(name="terrain_plateau", pos=(terrain.ramp_end_x + L / 2, 0.0, rise - THICK / 2),
                      half=(L / 2, half_y, THICK / 2), euler_deg=(0, 0, 0)))
    else:
        raise ValueError(f"no MuJoCo geometry for {type(terrain).__name__}")
    return g


def terrain_to_xml(terrain: TerrainProfile, friction=0.8, **kw) -> str:
    out = []
    for q in terrain_geoms(terrain, **kw):
        out.append(f'<geom name="{q["name"]}" type="box" pos="{q["pos"][0]:.5f} {q["pos"][1]:.5f} {q["pos"][2]:.5f}" '
                   f'size="{q["half"][0]:.5f} {q["half"][1]:.5f} {q["half"][2]:.5f}" euler="{q["euler_deg"][0]:.4f} {q["euler_deg"][1]:.4f} {q["euler_deg"][2]:.4f}" '
                   f'friction="{friction} 0.005 0.0001" rgba="0.6 0.6 0.7 1"/>')
    return "\n    ".join(out)


def top_surface_z(geom: dict, x: float) -> float:
    """Height of the box's top face at world x (used by tests to check the XML against the profile)."""
    th = -np.radians(geom["euler_deg"][1])
    n = np.array([-np.sin(th), np.cos(th)])                          # (x,z) normal of the top face
    c = np.array([geom["pos"][0], geom["pos"][2]]); top_c = c + geom["half"][2] * n
    return float(top_c[1] + (x - top_c[0]) * np.tan(th))
