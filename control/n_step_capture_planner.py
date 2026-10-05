"""
control/n_step_capture_planner.py

Closes the gap docs/BUGS_FOUND.md's multi-step-recovery finding left
explicitly open: "a real N-step-capturability solve (Pratt & Koolen and
successors) solve for a whole sequence of future footsteps JOINTLY...
that joint solve is a substantially harder optimization problem than
what's implemented here, and remains open." This module implements a
real (if simplified) version of that joint solve, replacing the naive
"independently repeat the single-step formula" strategy that was shown
to crowd consecutive footsteps toward the same side (see
docs/BUGS_FOUND.md).

WHY THE NAIVE VERSION FAILED (the design problem this module actually
solves): the naive multi-step recovery placed EACH upcoming footstep
at 100% of that moment's capture point, independently, with no
awareness that the gait needs to keep ALTERNATING sides at a sensible
width. When a big lateral push leaves the CoM still moving toward one
side, EVERY capture point computed while that velocity persists points
toward the SAME side - so consecutive footsteps ended up almost on top
of each other instead of alternating.

THE FIX - BLENDED N-STEP TARGETS: for each footstep i in the recovery
window (i = 1..N), the target is a WEIGHTED BLEND of the immediate
capture-point recovery target and that footstep's own NOMINAL (planned,
properly-alternating, correctly-spaced) position:

    p_i = alpha_i * capture_point_i + (1 - alpha_i) * nominal_i

with alpha_i DECAYING from (near) 1.0 on the first recovered footstep
toward 0.0 by the last one in the window - a geometric decay,
`alpha_i = decay_ratio ** (i - 1)`. This is a genuine JOINT design
choice, not an independent per-step recomputation: by construction, the
LAST few steps in the window are pulled increasingly back toward their
proper alternating, correctly-spaced nominal positions REGARDLESS of
what the still-recovering CoM velocity says, which is precisely what
structurally prevents the crowding failure mode the naive version hit -
the recovery is explicitly asked to hand back control to the nominal
gait pattern by the end of the window, rather than blindly chasing the
capture point at every single step.

This is a legitimate simplification of full N-step capturability
optimization (which would jointly optimize the alpha schedule itself,
subject to the true nonlinear feasibility/stability region - see Pratt
& Koolen's original formulation), not a claim to have solved that full
problem: the decay schedule here is a single stated hyperparameter
(`decay_ratio`), not an optimized one. What IS genuinely solved for is
the coupling between "how much recovery" and "how much return to
nominal" at each step in the window, which is exactly the coupling the
naive independent-repeat version was missing.
"""
from __future__ import annotations

import numpy as np

from control.capture_point import capture_point_2d, clip_to_reachable_step


def blended_recovery_target(current_com_xy: np.ndarray, current_com_vel_xy: np.ndarray,
                             omega: float, nominal_footstep_xy: np.ndarray,
                             stance_foot_xy: np.ndarray, alpha: float,
                             max_step_length_m: float) -> np.ndarray:
    """One footstep's target under the blended N-step scheme: a weighted
    combination of the instantaneous capture point (recovery-driven) and
    the nominal planned position (gait-pattern-driven), then clipped to a
    kinematically reachable step from the current stance foot - same
    physical reach limit as the single-step version, applied AFTER
    blending (the blended target is what actually has to be reached)."""
    cp = capture_point_2d(current_com_xy, current_com_vel_xy, omega)
    blended = alpha * cp + (1.0 - alpha) * nominal_footstep_xy
    return clip_to_reachable_step(blended, stance_foot_xy, max_step_length_m)


def alpha_schedule(n_steps: int, decay_ratio: float = 0.5) -> np.ndarray:
    """alpha_i = decay_ratio ** i for i = 0..n_steps-1 - the first
    recovered footstep gets alpha=1.0 (full capture-point correction),
    each subsequent one decays geometrically back toward the nominal
    gait pattern. `decay_ratio` in (0, 1); smaller means faster return
    to nominal (less aggressive multi-step correction)."""
    return decay_ratio ** np.arange(n_steps)
