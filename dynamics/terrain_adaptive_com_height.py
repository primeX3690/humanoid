"""
dynamics/terrain_adaptive_com_height.py

Closes the concrete, ALREADY-TESTED manifestation of docs/SCOPE.md
item 4 ("Constant CoM height... breaks the linear model...") -
specifically the finding in docs/BUGS_FOUND.md that stepping DOWN a
10cm curb exceeds the leg's reach margin "because the CoM-height model
doesn't adapt its assumed constant height to terrain at all". This
module makes the assumed CoM height ABOVE THE GROUND adapt continuously
to terrain, so the hip lowers as a foot descends toward a lower
landing (or hasn't yet risen onto a higher one) - instead of keeping a
rigid, terrain-blind world-frame height.

DERIVATION (why this exact formula, not a hand-tuned one): at any
instant, the vertical reach demand on EACH leg is (hip_height -
that_leg's_own_foot_height). This project's hip-position model uses
ONE SHARED hip height for both legs (see
simulation/joint_trajectory_generator.py), so at any instant where the
two feet sit at DIFFERENT heights (mid-transition over a step, or one
foot mid-swing while the other is planted), that ONE hip height must
satisfy BOTH legs' reach at once. To keep BOTH reaches within the same
budget (`default_zc`, the value the leg-length margin in
docs/BUGS_FOUND.md was already sized against on flat ground) at every
instant:

    hip - left_foot_z(t)  <= default_zc
    hip - right_foot_z(t) <= default_zc
    =>  hip(t) = min(left_foot_z(t), right_foot_z(t)) + default_zc

On flat ground this reduces to exactly `default_zc`, matching the
original constant-height behavior bit-for-bit (verified in
tests/test_terrain_adaptive_com_height.py). Directly reuses
planning.footstep_planner.foot_target_trajectories' ALREADY-VERIFIED
per-foot z profile (the same one the swing trajectory itself uses) as
the two `foot_z(t)` inputs, rather than re-deriving a second,
possibly-inconsistent interpolation.

FIRST ATTEMPT, KEPT AS A DOCUMENTED FAILURE (see docs/BUGS_FOUND.md
bug #7): an earlier version computed one target CoM height PER
FOOTSTEP PHASE and linearly ramped between phase targets across each
double-support window (mirroring the ZMP reference's own ramp). That
version fixed the documented downward-step failure but broke a
previously-working case (stepping UP), because ramping the shared hip
height UP - in preparation for a higher upcoming stance - during a
double-support window where the OTHER, not-yet-moved foot was still at
the OLD (lower) height, pushed that other foot's instantaneous reach
demand to the leg's exact maximum, for a case that used to have slack.
The min-formula above is not just simpler than that per-phase/ramp
scheme, it structurally cannot make that mistake: it can never demand
more than `default_zc` of reach from whichever foot is CURRENTLY
higher, at any instant, by construction.

MEASURED RESULT (docs/BUGS_FOUND.md bug #7): with this fix, IK now
converges for the entire walk for a 10cm step DOWN (previously failed
outright - see tests/test_terrain.py's
`test_stepping_down_exceeds_current_leg_reach_margin`, which is left
UNCHANGED and still passing, since it deliberately exercises the
ORIGINAL, non-adaptive path) and up to at least 12cm; it starts failing
again somewhere between 12cm and 14cm - a real, new, smaller-but-still-
present boundary, not claimed to be solved for arbitrary drop depth.
10cm steps UP continue to converge exactly as before (unaffected,
verified identical joint angles to the original constant-height model
on flat ground).

HONEST, STATED SCOPE LIMIT (not fixed here - see docs/SCOPE.md item 4):
this adapts the hip-HEIGHT REFERENCE consumed by leg IK. It does NOT
re-derive the LIPM/ZMP-preview-control horizontal (x, y) tracking loop
(dynamics/lipm.py, control/zmp_preview_controller.py) for a
time-varying zc - that loop still uses ONE fixed nominal zc for its own
internal dynamics and gain design, exactly as before. Properly coupling
a varying zc INTO the horizontal ZMP-tracking dynamics is the
"nonlinear MPC on centroidal dynamics" research area SCOPE.md already
names as a separate, harder step; this module closes the leg-reach
consequence of the flat-CoM-height assumption, not the full coupled
problem.
"""
from __future__ import annotations

import numpy as np

from planning.footstep_planner import GaitParams, Footstep, foot_target_trajectories


def _support_foot_z(gait: GaitParams, footsteps: list[Footstep]) -> list[float]:
    """z-height counterpart to footstep_planner._support_foot_positions:
    support_z[i] = terrain height of the STATIONARY foot during
    footsteps[i]'s swing (wherever the opposite foot last landed, or 0.0
    - flat ground - for its initial stance position). Used only for the
    diagnostic per-phase target below, not by hip_height_profile itself
    (which works directly off the continuous per-foot z trajectories -
    see module docstring for why)."""
    last_z = {"left": 0.0, "right": 0.0}
    support_z = []
    for step in footsteps:
        opposite = "left" if step.side == "right" else "right"
        support_z.append(last_z[opposite])
        last_z[step.side] = step.z
    return support_z


def compute_zc_profile(gait: GaitParams, footsteps: list[Footstep], default_zc: float,
                        min_zc: float | None = None) -> np.ndarray:
    """DIAGNOSTIC helper (used in tests/docs, not by hip_height_profile
    itself): the target CoM height AT THE MOMENT footstep i lands,
    `default_zc - max(drop_i, 0)` where `drop_i` is how much lower the
    landing is than the stance foot during that swing. This is exactly
    the value hip_height_profile's continuous min-formula converges to
    at each footstep's landing instant (verified in
    tests/test_terrain_adaptive_com_height.py) - kept as a separate,
    simple closed-form function because it's a much easier number to
    reason about / assert on in tests than reading it off a full
    per-timestep trajectory. `min_zc` (if given) is an explicit, stated
    safety clip on how low this diagnostic target is allowed to read."""
    stance_z = _support_foot_z(gait, footsteps)
    zc = np.empty(len(footsteps))
    for i, step in enumerate(footsteps):
        drop = stance_z[i] - step.z
        target = default_zc - max(drop, 0.0)
        if min_zc is not None:
            target = max(target, min_zc)
        zc[i] = target
    return zc


def hip_height_profile(gait: GaitParams, footsteps: list[Footstep], default_zc: float
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t, hip_z), the world-frame hip height at every sampled
    time (same timebase as foot_target_trajectories /
    zmp_reference_trajectory - same dt, same total duration):

        hip_z(t) = min(left_foot_z(t), right_foot_z(t)) + default_zc

    See module docstring for the derivation and why this is robust to
    both upward and downward terrain changes without the earlier
    ramp-based attempt's failure mode."""
    t, left_xyz, right_xyz = foot_target_trajectories(gait, footsteps)
    hip_z = np.minimum(left_xyz[:, 2], right_xyz[:, 2]) + default_zc
    return t, hip_z
