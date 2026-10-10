"""
Walking reference for the whole-body controller with TURNING and TERRAIN (v2's WBC walks were straight, flat-ground only).

build_walk_reference(...) returns the same dict layout as simulation/walk_wbc.build_plan plus
  heading(t)   commanded body yaw per sample (torso / foot orientation targets for the WBC)
  zc_ref(t)    terrain-adaptive CoM height reference (min(foot z) + zc)  [dynamics/terrain_adaptive_com_height.py logic]
Pure NumPy/SciPy: runs and is tested without MuJoCo. The LIPM/ZMP preview loop is the repo's own (fixed zc, as documented in SCOPE.md).
"""
from __future__ import annotations
import numpy as np
from dynamics.lipm import LIPM, LIPMParams
from control.zmp_preview_controller import ZMPPreviewController, PreviewControllerConfig
from planning.footstep_planner import (GaitParams, plan_footsteps, plan_turning_footsteps, apply_terrain,
                                       zmp_reference_trajectory, foot_target_trajectories, heading_profile)


def lipm_walk_for(gait: GaitParams, footsteps, zc: float):
    p = LIPMParams(com_height_m=zc, dt=gait.dt)
    t, zx, zy = zmp_reference_trajectory(gait, footsteps)
    lx, ly = LIPM(p), LIPM(p); cx, cy = ZMPPreviewController(lx, PreviewControllerConfig()), ZMPPreviewController(ly, PreviewControllerConfig())
    N = cx.N; zxp = np.concatenate([zx, np.full(N + 1, zx[-1])]); zyp = np.concatenate([zy, np.full(N + 1, zy[-1])])
    xs, ys = np.zeros(3), np.zeros(3); n = len(t)
    com = np.zeros((n, 2)); zmp = np.zeros((n, 2))
    for k in range(n):
        com[k] = xs[0], ys[0]; zmp[k] = lx.zmp(xs), ly.zmp(ys)
        xs = lx.step(xs, cx.compute_jerk(xs, zxp[k:k + N + 1])); ys = ly.step(ys, cy.compute_jerk(ys, zyp[k:k + N + 1]))
    return t, com, zmp, np.column_stack([zx, zy])


def build_walk_reference(n_steps=8, zc=0.85, step_length=0.3, turn_rate=0.0, terrain=None, step_width=None):
    g = GaitParams(n_steps=n_steps, step_length_m=step_length)
    if step_width is not None: g.step_width_m = step_width
    fs = plan_turning_footsteps(g, turn_rate) if turn_rate != 0.0 else plan_footsteps(g)
    if terrain is not None: fs = apply_terrain(fs, terrain)
    t, com, zmp, zmp_ref = lipm_walk_for(g, fs, zc)
    tf, L, R = foot_target_trajectories(g, fs)
    n = min(len(t), len(tf)); t = t[:n]; com = com[:n]
    vel = np.gradient(com, t, axis=0); acc = np.gradient(vel, t, axis=0)
    fv = {k: np.gradient(v[:n], t, axis=0) for k, v in (("L", L), ("R", R))}
    fa = {k: np.gradient(fv[k], t, axis=0) for k in "LR"}
    th, hd = heading_profile(g, fs); hd = hd[:n]
    zc_ref = np.minimum(L[:n, 2], R[:n, 2]) + zc
    return dict(g=g, t=t, com=com, com_v=vel, com_a=acc, foot={"L": L[:n], "R": R[:n]}, foot_v=fv, foot_a=fa,
                steps=fs, heading=hd, zc_ref=zc_ref, zmp=zmp[:n], zmp_ref=zmp_ref[:n], w=None)
