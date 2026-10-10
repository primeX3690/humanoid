"""Stair-climbing reference for the WBC (step-to gait: both feet on each stair): footsteps, stair-safe swing trajectories, CoM plan.

build_stairs_reference(...) returns the same dict as planning.wbc_walk_reference.build_walk_reference, so simulation/walk_wbc.py can run it
with `--terrain stairs`. CoM xy/z come from the height-coupled preview NMPC (control/preview_nmpc.py) because the CoM height changes by a
riser per stair - exactly the case a fixed-height LIPM cannot represent.
"""
from __future__ import annotations
import numpy as np
from planning.footstep_planner import GaitParams, plan_footsteps, Footstep, zmp_reference_trajectory, foot_target_trajectories
from terrain.terrain_profile import StairTerrain
from trajectory.stair_swing import StairSwingParams, stair_swing_position, toe_heel_clearance


def plan_stair_footsteps(stairs: StairTerrain, g: GaitParams, approach=2, approach_len=None):
    base = plan_footsteps(g)
    sw = g.step_width_m / 2.0; out = []
    approach_len = approach_len if approach_len is not None else max(stairs.first_x - 0.12, 0.2)
    for i, s in enumerate(base):
        if i < approach:
            x, z = approach_len * (i + 1) / approach, 0.0
        else:
            k = min((i - approach) // 2, stairs.n - 1)
            x = stairs.first_x + (k + 0.5) * stairs.tread_m; z = (k + 1) * stairs.riser_m
        y = sw if s.side == "left" else -sw
        out.append(Footstep(x=x, y=y, side=s.side, start_time=s.start_time, end_time=s.end_time, z=z))
    return out


def build_stairs_reference(stairs: StairTerrain, zc=0.85, approach=2, n_stairs=None, swing: StairSwingParams = StairSwingParams(), step_width=0.20, use_nmpc=True):
    n_st = min(n_stairs or stairs.n, stairs.n)
    g = GaitParams(n_steps=approach + 2 * n_st, step_width_m=step_width, step_duration_s=1.0, double_support_s=0.5)
    fs = plan_stair_footsteps(stairs, g, approach)
    t, L, R = foot_target_trajectories(g, fs, swing_fn=lambda s, a, b: stair_swing_position(s, a, b, swing))
    if use_nmpc:
        from control.preview_nmpc import simulate_preview, PreviewNMPCConfig
        r = simulate_preview(g, fs, PreviewNMPCConfig(z_ref=zc, z_max=zc + 0.08 + stairs.riser_m * n_st, z_min=zc - 0.1))
        n = min(len(t), len(r["t"])); com = np.column_stack([r["com_x"][:n], r["com_y"][:n]]); zc_ref = r["com_z"][:n]
    else:
        from planning.wbc_walk_reference import lipm_walk_for
        t2, com, _, _ = lipm_walk_for(g, fs, zc); n = min(len(t), len(t2)); com = com[:n]; zc_ref = np.minimum(L[:n, 2], R[:n, 2]) + zc
    t = t[:n]; vel = np.gradient(com, t, axis=0); acc = np.gradient(vel, t, axis=0)
    fv = {k: np.gradient(v[:n], t, axis=0) for k, v in (("L", L), ("R", R))}; fa = {k: np.gradient(fv[k], t, axis=0) for k in "LR"}
    return dict(g=g, t=t, com=com, com_v=vel, com_a=acc, foot={"L": L[:n], "R": R[:n]}, foot_v=fv, foot_a=fa, steps=fs,
                heading=np.zeros(n), zc_ref=zc_ref, w=None, stairs=stairs)
