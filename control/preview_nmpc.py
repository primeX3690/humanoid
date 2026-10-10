"""
Preview nonlinear MPC with CoM height as a decision variable - fixes the open item in docs/SCOPE.md 12d.

WHY the old control/nonlinear_mpc.py oscillated laterally (diagnosed in v3): it re-solved a generic SLSQP problem over only 0.2 s with the
ZMP reference held CONSTANT over the horizon. The CoM dynamics of a walking LIPM have time constant sqrt(zc/g) ~ 0.3 s and are
non-minimum-phase w.r.t. the ZMP: to start moving toward the next support foot the CoM must start moving ~1 s BEFORE the ZMP reference
switches. A 0.2 s horizon with a constant reference cannot do that, so it always reacts late and rings.

This module:
  * horizon 1.6 s (N=32 @ 50 ms) WITH the real future ZMP reference and support polygons (preview),
  * the ZMP equation  zmp = x - (z - z_foot)/(zdd + g) * xdd  is nonlinear in (x, z); given a height plan z(t) it is LINEAR in the
    horizontal jerks, so x and y are solved as condensed QPs (control/fast_qp.py) with ZMP-in-support constraints,
  * the height plan is optimised (L-BFGS-B, bounded) given the horizontal plan, and the two steps alternate (2 sweeps) - an
    alternating-minimisation form of NMPC that keeps every sub-problem convex or tiny,
  * terminal DCM cost for stability, jerk applied by exact triple-integrator update at the raw rate.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from control.fast_qp import FastQP

G = 9.81


@dataclass
class PreviewNMPCConfig:
    N: int = 32
    dt_m: float = 0.05
    z_ref: float = 0.85
    z_min: float = 0.60
    z_max: float = 0.95
    w_zmp: float = 1.0
    w_jerk: float = 1e-6
    w_term: float = 50.0
    w_height: float = 5.0
    w_jerk_z: float = 1e-3
    margin: float = 0.01
    sweeps: int = 2


def _prediction(dt, N):
    A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]]); B = np.array([dt ** 3 / 6, dt * dt / 2, dt])
    Phi = np.zeros((N, 3, 3)); Gam = np.zeros((N, 3, N)); Ak = np.eye(3)
    powers = [np.eye(3)]
    for k in range(N): powers.append(A @ powers[-1])
    for k in range(N):
        Phi[k] = powers[k + 1]
        for i in range(k + 1): Gam[k, :, i] = powers[k - i] @ B
    return Phi, Gam, A, B


def _solve_axis(s0, c, ref, lo, hi, cfg, Phi, Gam, w_dcm, solver, key):
    """Condensed QP for one horizontal axis. zmp_k = pos_k - c_k acc_k."""
    N = cfg.N
    Mz = Gam[:, 0, :] - c[:, None] * Gam[:, 2, :]
    m0 = Phi[:, 0, :] @ s0 - c * (Phi[:, 2, :] @ s0)
    # terminal DCM: pos_N + vel_N / w  -> ref_N
    dcm_row = Gam[N - 1, 0, :] + Gam[N - 1, 1, :] / w_dcm
    dcm0 = Phi[N - 1, 0, :] @ s0 + Phi[N - 1, 1, :] @ s0 / w_dcm
    P = 2 * (cfg.w_zmp * Mz.T @ Mz + cfg.w_jerk * np.eye(N) + cfg.w_term * np.outer(dcm_row, dcm_row))
    q = 2 * (cfg.w_zmp * Mz.T @ (m0 - ref) + cfg.w_term * dcm_row * (dcm0 - ref[-1]))
    l = lo + cfg.margin - m0; u = hi - cfg.margin - m0
    bad = l > u                                           # polygon narrower than twice the margin: collapse to its centre
    mid = 0.5 * (lo + hi) - m0
    l = np.where(bad, mid, l); u = np.where(bad, mid, u)
    r = solver.solve(P, q, Mz, l, u, key=key)
    if r.info.status_val > 0: return r.x
    r2 = FastQP(eps=1e-4, max_iter=20000, polish=False).solve(P, q, Mz, np.where(bad, mid, lo - m0 - 0.05), np.where(bad, mid, hi - m0 + 0.05))
    return r2.x if r2.info.status_val > 0 else np.zeros(N)


def _z_plan(z0, xs_acc, ys_acc, xs_pos, ys_pos, ref_x, ref_y, zf, cfg, Phi, Gam, jz0):
    z_target = zf + cfg.z_ref
    N = cfg.N

    def roll(jz):
        z = Phi[:, :, :] @ z0 + np.einsum("kij,j->ki", Gam.transpose(0, 2, 1)[:, :, :].transpose(0, 2, 1), jz) if False else None
        zpos = Phi[:, 0, :] @ z0 + Gam[:, 0, :] @ jz; zacc = Phi[:, 2, :] @ z0 + Gam[:, 2, :] @ jz
        return zpos, zacc

    def cost(jz):
        zp, za = roll(jz); c = (zp - zf) / (za + G)
        ex = xs_pos - c * xs_acc - ref_x; ey = ys_pos - c * ys_acc - ref_y
        pen = np.sum(np.clip(zf + cfg.z_min - zp, 0, None) ** 2 + np.clip(zp - (zf + cfg.z_max), 0, None) ** 2) * 1e4
        return cfg.w_zmp * np.sum(ex ** 2 + ey ** 2) + cfg.w_height * np.sum((zp - z_target) ** 2) + cfg.w_jerk_z * np.sum(jz ** 2) + pen
    res = minimize(cost, jz0, method="L-BFGS-B", bounds=[(-60, 60)] * N, options=dict(maxiter=25))
    jz = res.x if np.all(np.isfinite(res.x)) else jz0
    zp, za = roll(jz)
    return jz, (zp - zf) / (za + G)


def simulate_preview(gait, footsteps, cfg: PreviewNMPCConfig | None = None) -> dict:
    from planning.footstep_planner import zmp_reference_trajectory, support_polygon_at
    cfg = cfg or PreviewNMPCConfig()
    t, zx, zy = zmp_reference_trajectory(gait, footsteps)
    n = len(t); dt = gait.dt; sub = max(int(round(cfg.dt_m / dt)), 1); dtm = sub * dt
    cfg = PreviewNMPCConfig(**{**cfg.__dict__, "dt_m": dtm})
    Phi, Gam, _, _ = _prediction(dtm, cfg.N)
    w_dcm = np.sqrt(G / cfg.z_ref)
    solver = FastQP(eps=1e-7, max_iter=8000); solver_y = FastQP(eps=1e-7, max_iter=8000)
    x = np.zeros(3); y = np.zeros(3); z = np.array([cfg.z_ref + stance_z(0.0) if False else cfg.z_ref, 0.0, 0.0])
    out = {k: np.zeros(n) for k in ("com_x", "com_y", "com_z", "zmp_x", "zmp_y")}
    def stance_z(tk):                                   # height of the foot that is (last) in stance at time tk
        z_now = 0.0
        for s_ in footsteps:
            if s_.start_time <= tk: z_now = s_.z
        return z_now
    k = 0; jz_prev = np.zeros(cfg.N); iters = 0
    A1 = lambda d: (np.array([[1, d, d * d / 2], [0, 1, d], [0, 0, 1]]), np.array([d ** 3 / 6, d * d / 2, d]))
    A_raw, B_raw = A1(dt)
    while k < n:
        idx = np.minimum(k + sub * (np.arange(cfg.N) + 1), n - 1)
        ref_x, ref_y = zx[idx], zy[idx]
        bounds = np.array([support_polygon_at(t[i], gait, footsteps) for i in idx])
        zf = np.array([stance_z(t[i]) for i in idx])
        c = (cfg.z_ref) / (G + 0.0 * zf) * np.ones(cfg.N)
        jz = jz_prev
        for sweep in range(cfg.sweeps):
            jx = _solve_axis(x, c, ref_x, bounds[:, 0], bounds[:, 1], cfg, Phi, Gam, w_dcm, solver, "x")
            jy = _solve_axis(y, c, ref_y, bounds[:, 2], bounds[:, 3], cfg, Phi, Gam, w_dcm, solver_y, "y")
            xp = Phi[:, 0, :] @ x + Gam[:, 0, :] @ jx; xa = Phi[:, 2, :] @ x + Gam[:, 2, :] @ jx
            yp = Phi[:, 0, :] @ y + Gam[:, 0, :] @ jy; ya = Phi[:, 2, :] @ y + Gam[:, 2, :] @ jy
            jz, c = _z_plan(z, xa, ya, xp, yp, ref_x, ref_y, zf, cfg, Phi, Gam, jz)
        jz_prev = np.concatenate([jz[1:], jz[-1:]])
        for _ in range(sub):
            if k >= n: break
            out["com_x"][k], out["com_y"][k], out["com_z"][k] = x[0], y[0], z[0]
            den = z[2] + G; zfk = stance_z(t[k])
            out["zmp_x"][k] = x[0] - (z[0] - zfk) / den * x[2]; out["zmp_y"][k] = y[0] - (z[0] - zfk) / den * y[2]
            x = A_raw @ x + B_raw * jx[0]; y = A_raw @ y + B_raw * jy[0]; z = A_raw @ z + B_raw * jz[0]
            k += 1
    inside = np.array([(lambda b: b[0] <= out["zmp_x"][i] <= b[1] and b[2] <= out["zmp_y"][i] <= b[3])(support_polygon_at(t[i], gait, footsteps)) for i in range(n)])
    return dict(t=t, zmp_ref_x=zx, zmp_ref_y=zy, zmp_in_support=inside, **out)
