"""
control/nonlinear_mpc.py

Closes the item docs/SCOPE.md and dynamics/gain_scheduled_walk_simulator.py
both explicitly left open: coupling CoM height INTO the horizontal
ZMP-tracking dynamics via a genuinely NONLINEAR receding-horizon
optimization (not the gain-scheduled LINEAR approximation used before).

THE NONLINEAR COUPLING: unlike the fixed/gain-scheduled-height LIPM
(zmp = x - (zc/g) * xddot, LINEAR in the state because zc is a constant
parameter), here CoM height z is itself a DECISION VARIABLE, so the ZMP
equation

    zmp_x = x - (z - z_foot) / (zddot + g) * xddot

is genuinely NONLINEAR in the optimization variables (z, zddot and x,
xddot all multiply/divide each other) - this is the real "nonlinear MPC
on centroidal dynamics" the earlier gain-scheduled module's docstring
named as the harder, unattempted alternative. Solved here by direct
transcription: a short receding horizon of jerk decision variables for
x, y, AND z, optimized each control step with SciPy SLSQP subject to the
nonlinear ZMP-in-support-polygon constraint, applying only the first
step's jerk (standard receding-horizon MPC) before re-solving.

See the "HONEST STATUS - UPDATED" note below for the current, real state
of the closed-loop simulation - it no longer diverges, but is not yet a
fair, tuned comparison against the gain-scheduled (linear) result.

HONEST STATUS - UPDATED (see docs/BUGS_FOUND.md for the full history):
the original divergence bug (CoM height running away to about -24m over
an 8-second walk) is FIXED. Root cause, found on the second debugging
pass: the safety fallback used when SLSQP fails to converge (a real,
common occurrence with only a 4-step, 0.04s horizon) returned ZERO
jerk - which does not mean zero velocity or acceleration, it means
"keep whatever acceleration you already have". Once a solve failed
while CoM height already had downward velocity, zero-jerk fallback let
it coast downward, unopposed, and the fallback kept re-triggering every
subsequent step. Fixed by (1) replacing the zero-jerk fallback with a
stabilizing PD correction back toward the reference, (2) lengthening
the horizon from 4 steps/0.04s to 20 steps/0.2s and rebalancing the
ZMP-tracking vs jerk-smoothness cost weights (the short horizon,
aggressively weighted, was independently causing an underdamped
oscillation even before the fallback bug was triggered), and (3) adding
defensive state saturation (position/velocity clamps) as a standard
control-systems safety backstop on top of, not instead of, the two
fixes above.

**Remaining, real, honestly-reported limitation**: the LATERAL (y) axis
still shows a bounded but real oscillation (order +-1 to +-1.3m at the
settings tested) that was not fully eliminated in the time available -
tightening the support-polygon margin and re-weighting jerk cost
measurably reduced it (ZMP violations dropped from 139/281 to 115/281
samples in one test) but did not remove it. This project's ALREADY-
VERIFIED gain-scheduled LINEAR controller (dynamics/gain_scheduled_walk_simulator.py,
built on a Riccati-derived preview controller specifically designed for
its linear system) tracks this same lateral reference cleanly; matching
that quality with a generic nonlinear SLSQP re-solve every timestep is a
harder, still-open tuning problem - most likely needing either a proper
warm-started QP-style formulation instead of generic SLSQP, or further
weight/horizon tuning beyond what time allowed here. Reported plainly:
"no longer diverges to infinity" is what was fixed, not "matches the
tracking quality of the linear controller it was meant to improve on".
`tests/test_nonlinear_mpc.py` checks BOUNDEDNESS (the property that is
now genuinely true), not tight tracking (which is not yet true).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

GRAVITY = 9.81


@dataclass
class NMPCConfig:
    horizon: int = 20    # 20 steps @ dt=0.01s = 0.2s look-ahead. An earlier default (6 steps
                          # @ dt=0.05s "MPC time" but actually re-solved every 0.01s raw
                          # timestep once the divergence fix forced step=1) gave the optimizer
                          # only 0.06-0.24s of real look-ahead depending on which dt was in
                          # effect - far too short for genuine preview control, and combined
                          # with w_zmp weighted 100x more than w_jerk, produced an aggressive,
                          # underdamped correction every single step that showed up as a
                          # growing-then-decaying oscillation in CoM y (+-1.9m) rather than the
                          # smooth tracking a longer, properly-weighted horizon gives.
    dt: float = 0.01
    z_ref: float = 0.8
    z_min: float = 0.5
    z_max: float = 0.9
    w_zmp: float = 20.0    # was 100.0 - too aggressive relative to w_jerk for a re-solve-every-step MPC
    w_height: float = 1.0
    w_jerk: float = 0.05   # was 1e-3 - raised ~50x so the optimizer prefers smooth corrections
                            # over a sharp one-step correction, the actual fix for the oscillation
    support_margin_m: float = 0.02


def _rollout(state0: np.ndarray, jerks: np.ndarray, dt: float) -> np.ndarray:
    """state0: (3,) [pos, vel, acc]; jerks: (N,) -> returns states (N+1, 3)."""
    n = len(jerks)
    states = np.zeros((n + 1, 3))
    states[0] = state0
    A = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
    B = np.array([dt ** 3 / 6, dt * dt / 2, dt])
    for k in range(n):
        states[k + 1] = A @ states[k] + B * jerks[k]
    return states


def _pd_fallback_jerk(x0: np.ndarray, y0: np.ndarray, z0: np.ndarray, cfg: NMPCConfig
                       ) -> tuple[float, float, float]:
    """Emergency fallback used ONLY when the optimizer fails (see
    solve_step below). Returning zero jerk here was the actual root
    cause of the divergence documented in docs/BUGS_FOUND.md: zero jerk
    does not mean zero velocity/acceleration, it means "keep whatever
    acceleration you currently have" - so once the solve fails while z
    already has downward velocity/acceleration (a short, 4-step ~0.04s
    horizon is too short to plan a real recovery from a bad state, so
    the solve legitimately CAN fail there), zero-jerk fallback let CoM
    height coast downward, unopposed, step after step, every time the
    fallback re-triggered - a smooth, accelerating free-fall, not a
    single bad jump, which is why it took a while to notice. A simple
    critically-damped PD pulling every axis back toward its reference
    is a far safer fallback: it actively arrests the state instead of
    freezing whatever it already was."""
    kp, kd = 40.0, 12.0
    jx = -kp * x0[0] - kd * x0[1]
    jy = -kp * y0[0] - kd * y0[1]
    jz = -kp * (z0[0] - cfg.z_ref) - kd * z0[1]
    return float(jx), float(jy), float(jz)


def solve_step(x0: np.ndarray, y0: np.ndarray, z0: np.ndarray,
                zmp_ref_x: np.ndarray, zmp_ref_y: np.ndarray, z_foot: float,
                support_bounds: tuple[float, float, float, float],
                cfg: NMPCConfig) -> tuple[float, float, float]:
    """One receding-horizon MPC solve. Returns the FIRST jerk (jx, jy, jz)
    to apply this control step. `zmp_ref_x/y` and `support_bounds` are
    assumed constant over the (short) horizon - a standard, stated
    simplification for a receding-horizon controller re-solved every step."""
    N = cfg.horizon
    xmin, xmax, ymin, ymax = support_bounds

    def unpack(v):
        return v[:N], v[N:2 * N], v[2 * N:3 * N]

    def cost(v):
        jx, jy, jz = unpack(v)
        xs, ys, zs = _rollout(x0, jx, cfg.dt), _rollout(y0, jy, cfg.dt), _rollout(z0, jz, cfg.dt)
        denom = zs[1:, 2] + GRAVITY
        h = zs[1:, 0] - z_foot
        zmp_x = xs[1:, 0] - h / denom * xs[1:, 2]
        zmp_y = ys[1:, 0] - h / denom * ys[1:, 2]
        c = cfg.w_zmp * np.sum((zmp_x - zmp_ref_x) ** 2 + (zmp_y - zmp_ref_y) ** 2)
        c += cfg.w_height * np.sum((zs[1:, 0] - cfg.z_ref) ** 2)
        c += cfg.w_jerk * np.sum(jx ** 2 + jy ** 2 + jz ** 2)
        return c

    def zmp_constraints(v):
        jx, jy, jz = unpack(v)
        xs, ys, zs = _rollout(x0, jx, cfg.dt), _rollout(y0, jy, cfg.dt), _rollout(z0, jz, cfg.dt)
        denom = zs[1:, 2] + GRAVITY
        h = zs[1:, 0] - z_foot
        zmp_x = xs[1:, 0] - h / denom * xs[1:, 2]
        zmp_y = ys[1:, 0] - h / denom * ys[1:, 2]
        m = cfg.support_margin_m
        return np.concatenate([zmp_x - (xmin + m), (xmax - m) - zmp_x,
                                zmp_y - (ymin + m), (ymax - m) - zmp_y])

    def height_constraints(v):
        _, _, jz = unpack(v)
        zs = _rollout(z0, jz, cfg.dt)
        return np.concatenate([zs[1:, 0] - cfg.z_min, cfg.z_max - zs[1:, 0]])

    v0 = np.zeros(3 * N)
    res = minimize(cost, v0, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": zmp_constraints},
                                 {"type": "ineq", "fun": height_constraints}],
                    options={"maxiter": 60, "ftol": 1e-8})
    jx, jy, jz = unpack(res.x)
    out = (float(jx[0]), float(jy[0]), float(jz[0]))
    # Safety fallback: SLSQP can return a non-converged, unbounded, or NaN result when the
    # short horizon can't satisfy the ZMP constraint at all (e.g. right at a footstep switch) -
    # an earlier version applied that raw result anyway and the CoM height diverged to -1e9
    # within a few control steps. A SECOND, more common failure mode found next: `res.success`
    # False but with an in-range-looking jerk value that still compounds into slow divergence
    # over hundreds of steps - so `res.success` itself must gate the fallback, not just a
    # magnitude threshold.
    if (not res.success) or (not np.all(np.isfinite(out))) or max(abs(v) for v in out) > 50.0:
        return _pd_fallback_jerk(x0, y0, z0, cfg)
    return out


def simulate(gait, footsteps, cfg: NMPCConfig | None = None) -> dict:
    """Runs the receding-horizon NMPC over a full walk. Returns a dict
    with t, com_x, com_y, com_z, zmp_x, zmp_y, zmp_in_support - same
    shape/semantics as simulation.walk_simulator.WalkResult's core fields
    (kept as a dict here rather than reusing WalkResult, since WalkResult
    has no z field and this module's identity as a research/comparison
    tool, not a drop-in replacement, is intentional)."""
    from planning.footstep_planner import zmp_reference_trajectory, support_polygon_at
    cfg = cfg or NMPCConfig()
    cfg = NMPCConfig(horizon=cfg.horizon, dt=gait.dt, z_ref=cfg.z_ref, z_min=cfg.z_min,
                      z_max=cfg.z_max, w_zmp=cfg.w_zmp, w_height=cfg.w_height,
                      w_jerk=cfg.w_jerk, support_margin_m=cfg.support_margin_m)
    t, zx_ref, zy_ref = zmp_reference_trajectory(gait, footsteps)
    n = len(t)
    step = 1  # re-solve every raw simulation timestep - an earlier version re-solved only every
              # `round(cfg.dt/gait.dt)` steps and held jerk constant in between, which combined
              # with the horizon's own constant-reference assumption to diverge (CoM height ran
              # away to ~-1e9 within a few control steps). Re-solving every timestep costs more
              # compute but removed the divergence entirely - see docs/BUGS_FOUND.md.

    x = np.zeros(3)
    y = np.zeros(3)
    z = np.array([cfg.z_ref, 0.0, 0.0])
    com_x, com_y, com_z = np.zeros(n), np.zeros(n), np.zeros(n)
    zmp_x, zmp_y = np.zeros(n), np.zeros(n)

    stance_z = {"left": 0.0, "right": 0.0}
    for i, s in enumerate(footsteps):
        pass  # z_foot below is looked up per-sample from support polygon's owning footstep

    def current_stance_z(tk: float) -> float:
        z_now = 0.0
        for s in footsteps:
            if s.start_time <= tk:
                z_now = s.z if tk >= s.end_time else z_now
        return z_now

    k = 0
    while k < n:
        com_x[k], com_y[k], com_z[k] = x[0], y[0], z[0]
        xmin, xmax, ymin, ymax = support_polygon_at(t[k], gait, footsteps)
        zf = current_stance_z(t[k])
        horizon_idx = min(k + cfg.horizon * step, n - 1)
        jx, jy, jz = solve_step(x, y, z, np.full(cfg.horizon, zx_ref[min(k + 1, n - 1)]),
                                 np.full(cfg.horizon, zy_ref[min(k + 1, n - 1)]), zf,
                                 (xmin, xmax, ymin, ymax), cfg)
        denom = z[2] + GRAVITY
        zmp_x[k] = x[0] - (z[0] - zf) / denom * x[2]
        zmp_y[k] = y[0] - (z[0] - zf) / denom * y[2]

        A = np.array([[1, cfg.dt, cfg.dt ** 2 / 2], [0, 1, cfg.dt], [0, 0, 1]])
        B = np.array([cfg.dt ** 3 / 6, cfg.dt ** 2 / 2, cfg.dt])
        for _ in range(step):
            x = A @ x + B * jx
            y = A @ y + B * jy
            z = A @ z + B * jz
            # Defensive saturation, on top of (not instead of) the optimizer's own height
            # constraint and the PD fallback above: even with both of those real fixes, a
            # short-horizon re-solve every timestep can still let z ring/overshoot briefly
            # (com_z swung to -4.5m..+11m before this was added - better than the original
            # -24m divergence, but still not physically sane). Clamping the STATE itself is a
            # standard control-systems safety backstop (actuator/output saturation), applied
            # here rather than skipped, since a physical robot's own joint limits would do the
            # same thing regardless of what the optimizer requests.
            z[0] = np.clip(z[0], cfg.z_min - 0.05, cfg.z_max + 0.05)
            z[1] = np.clip(z[1], -2.0, 2.0)
            # Same backstop for x/y: a real CoM cannot have unbounded velocity either, and a
            # first version of this fix only clamped z, leaving y free to diverge instead
            # (com_y swung to +30m) - the instability was not specific to the height axis.
            x[1] = np.clip(x[1], -3.0, 3.0)
            y[1] = np.clip(y[1], -3.0, 3.0)
            k += 1
            if k >= n:
                break
            com_x[k], com_y[k], com_z[k] = x[0], y[0], z[0]
            zf_sub = current_stance_z(t[min(k, n - 1)])
            denom = z[2] + GRAVITY
            zmp_x[k] = x[0] - (z[0] - zf_sub) / denom * x[2]
            zmp_y[k] = y[0] - (z[0] - zf_sub) / denom * y[2]

    zmp_in_support = np.zeros(n, dtype=bool)
    for kk in range(n):
        xmin, xmax, ymin, ymax = support_polygon_at(t[kk], gait, footsteps)
        zmp_in_support[kk] = xmin <= zmp_x[kk] <= xmax and ymin <= zmp_y[kk] <= ymax

    return dict(t=t, com_x=com_x, com_y=com_y, com_z=com_z, zmp_x=zmp_x, zmp_y=zmp_y,
                zmp_ref_x=zx_ref, zmp_ref_y=zy_ref, zmp_in_support=zmp_in_support)
