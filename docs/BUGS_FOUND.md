# Real bugs and findings caught while building this

## Bug 1: swing-foot target used as the ZMP reference instead of the stance foot

First version of `zmp_reference_trajectory()` set the single-support ZMP
reference to `step.x`/`step.y` - the position the SWINGING foot was
heading TO. That's backwards: during single support, the ZMP must sit
under the STATIONARY (stance) foot, which is wherever the *opposite*
foot last landed (or its initial position, for the first step on each
side). Caught immediately by
`tests/test_walk_simulation.py::test_zmp_stays_in_support_polygon_for_entire_walk`
- with the bug, the ZMP was up to 2.9m outside the support polygon.
Fixed by tracking stance-foot position explicitly
(`_support_foot_positions()`), separate from the footstep-landing list.

## Bug 2: instantaneous ZMP reference jump at double-support boundaries

Even after fixing bug 1, the double-support ZMP reference jumped
INSTANTLY from the old stance foot to the new one at the start of the
double-support window, rather than ramping across it. A real gait
generator ramps this transition because an instantaneous lateral
weight-shift demand (e.g. 0.1m to -0.1m in one 10ms sample) is not
something the LIPM's own bandwidth can track without leaving the
support polygon - no controller gain fixes an physically-infeasible
reference. Fixed by linearly interpolating the ZMP reference across
each full double-support window (including the initial stand-to-walk
transition), in `zmp_reference_trajectory()`'s `ramp()` helper.

## Finding: more aggressive tracking gain (Q) made stability WORSE, not better

The instinctive choice for a "the ZMP must track tightly" cost weight is
a very large Q (tried 1e6, relative to R=1). This turned out to be
actively harmful: the LIPM->ZMP transfer has a well-known non-minimum-
phase zero (to move the ZMP one way, the CoM briefly accelerates the
OTHER way first - this is real biped-dynamics behavior, not a modeling
artifact, and is exactly why preview control exists in the first place).
With Q=1e6 and a lateral weight-shift only ~0.2-0.4s away (near the start
of a walk, or right after a footstep lands), the controller's own
optimal response to that non-minimum-phase behavior was a large
counter-lean - large enough, at that gain, to itself leave the support
polygon.

Swept Q from 1 to 1e6 (`Q in {1,10,30,100,300,1000,...,1e6}`) against
several `(step_width_m, double_support_s)` combinations and confirmed
this is a real trade-off, not a one-off artifact: moderate Q (~30) with
a slightly narrower stance (0.12m vs the first-guess 0.2m) and slightly
longer double support (0.4s vs 0.2s) gives **zero support-polygon
violations with a genuine ~8.6mm minimum margin** over a 16-step walk;
Q=1e6 with the same gait reproduces the violation reliably (kept as
`tests/test_walk_simulation.py::test_overly_aggressive_tracking_gain_reduces_stability_not_improves_it`,
a regression guard, not just a one-time observation).

**Honest limitation this leaves**: 8.6mm is a real, non-zero margin for
the *nominal, disturbance-free* case this model represents - but it is
thin. A real robot needs margin against model error, sensor noise, and
external pushes, none of which this LIPM-only model includes yet. That
margin, not zero, is why this is flagged as a limitation rather than a
solved problem - see docs/SCOPE.md for what a push-recovery /
disturbance-rejection layer on top of this would need to add.

## Bug 3: per-foot landing-position bookkeeping stuck on stale data (leg IK layer)

`foot_target_trajectories()` tracked each foot's "current planted
position" (`last_pos[side]`) and was supposed to update it the instant
that foot's swing phase ended. The update condition compared a sample's
time against `step.end_time - dt/2` - which silently failed for MOST
footsteps, because footstep end-times accumulate floating-point error
(e.g. `1.6 + 0.8 = 2.4000000000000004`, not exactly `2.4`), so whether
`t < step.end_time` happens to include or exclude the sample nominally
AT end_time varies step to step depending on rounding, and the
`>= end_time - dt/2` threshold ended up never true for 4 of the 8 steps
in the default gait. Effect: those feet kept "swinging toward" an
already-stale earlier landing spot for the rest of the walk - by the
last step, one foot's target was off by **732mm** (caught by
`tests/test_joint_trajectory_generator.py`'s full-walk tracking-error
check, not by any single-timestep test, which is exactly why that
integration test exists). Fixed by triggering the landing update on a
PHASE-INDEX TRANSITION (comparing `step_idx_at_time[k]` to its own
previous value) instead of a raw floating-point time comparison - immune
to accumulated rounding by construction.

## Bug 4: swing-foot height profile had nonzero touchdown velocity

The ground-clearance height profile used `step_height * sin(pi*s)`. This
peaks at the right height and returns to zero at both ends of the
swing - but its DERIVATIVE at `s=0` and `s=1` is `step_height*pi`, not
zero, meaning the foot is still moving vertically at the instant of
touchdown: a real impact, not a soft landing, and inconsistent with the
minimum-jerk (zero velocity at both ends) profile already used for the
horizontal motion. Caught directly by
`tests/test_swing_trajectory.py::test_zero_velocity_at_liftoff_and_touchdown`.
Fixed by switching to `step_height * sin(pi*s)^2`, whose derivative
(`step_height*pi*sin(2*pi*s)`) is exactly zero at `s=0, 0.5, 1`.

## Finding: neither a small hobby servo nor a research-grade actuator has enough knee torque margin - the knee, not the hip, is the binding actuator constraint

`dynamics/leg_dynamics.py` computes the joint torque this gait ACTUALLY
requires (Jacobian-transpose method: required ground-reaction force,
from the robot's mass and CoM acceleration, projected through the leg's
own Jacobian - the same Jacobian `kinematics/leg_ik.py`'s IK solver
uses, so the two are guaranteed geometrically consistent) for an assumed
25kg humanoid, and checks it against two REAL actuators' published
datasheets (`actuators/motor_specs.py`):

| Actuator (real, buyable) | Continuous torque rating | Peak hip demand | Peak knee demand |
|---|---|---|---|
| Dynamixel MX-106 (12V, common hobby servo) | 1.68 Nm | 32.2 Nm - **19x over** | 54.2 Nm - **32x over** |
| Dynamixel PRO Plus H54P-200 (24V, research-grade) | 44.7 Nm | 32.2 Nm - **within margin (1.39x)** | 54.2 Nm - **~20% short (0.82x)** |

**This is a specific, actionable hardware-selection finding, not a vague
"motors might not be strong enough"**: a popular hobbyist servo is off
by more than an order of magnitude and was never a realistic choice for
this mass class; a real research-grade actuator gets the HIP within
budget but the KNEE - not the hip - is the actual binding constraint,
by a modest (~20%) margin. Real next steps this points to, honestly
listed rather than solved here: (a) a knee-specific mechanical advantage
change (different lever arm/linkage geometry - many real biped designs
DO use non-1:1 knee linkages for exactly this reason), (b) accepting
brief above-continuous-rating operation during peak stance (real
actuators tolerate this for short durations - this model only checked
the CONSERVATIVE continuous rating), (c) reducing assumed robot mass,
or (d) a dual-motor/higher-reduction knee joint. Which of these is right
is a real hardware-design decision this software model cannot make by
itself - it can only tell you, correctly, that one of them is needed.

## Finding: leg segment length must exceed standing CoM height with real margin

First-pass leg lengths (thigh=shank=0.40m, 0.80m max reach) exactly
equaled the LIPM's 0.8m CoM height - meaning a perfectly straight,
fully-extended leg was the ONLY way to reach standing height, leaving
zero slack for the extra reach a SWINGING leg needs while the hip trails
behind the moving swing foot mid-stride (a real, physical constraint on
any leg design, not a simulation artifact). This showed up as IK
non-convergence with real position errors up to several centimeters
during mid-swing, on specific steps where the geometry was tightest -
not a random/flaky failure. Fixed by increasing leg segments to
thigh=shank=0.45m (0.9m max reach, 0.1m margin over the 0.8m CoM
height) - all IK calls converge to sub-millimeter accuracy across an
8-step walk after this change.

## Finding: capture-point push recovery helps, but only for steps AFTER the recovered one

`simulation/push_recovery_simulator.py` injects a real velocity-impulse
push into the CoM mid-walk and compares two outcomes: ignoring it
entirely vs. replacing the next upcoming footstep's landing position
with the Instantaneous Capture Point (`control/capture_point.py`,
Pratt et al. 2006 - the real theoretical basis used on robots like
Atlas), clipped to a kinematically reachable single-step distance.

**Real result**: recovery CANNOT undo the instantaneous ZMP transient
at the moment of the push itself - the support polygon is defined by
feet already planted before the push happened, so a brief deviation is
physically unavoidable regardless of what any future footstep does.
What recovery DOES measurably achieve is reducing violations in the
steps AFTER the recovered footstep (e.g. for a 1.0 m/s lateral push:
8 "later" support-polygon violations without recovery vs. 4 with it -
tests/test_push_recovery.py locks this specific comparison in). This is
a real, modest, honestly-modest benefit, not a magic fix - single-step
capture-point recovery doesn't replan the REST of the gait, so for very
large disturbances (tested up to 1.5 m/s) the benefit can even
disappear, because only one footstep is corrected while the rest of the
pre-planned sequence stays fixed and can become newly inconsistent with
the recovered position.

## Finding: stepping DOWN a curb exceeds the leg-reach margin that stepping UP doesn't

`terrain/terrain_profile.py` + `planning/footstep_planner.apply_terrain()`
place footsteps at real, non-flat heights. Testing a modest 10cm
step: stepping UP converges fine (IK succeeds at every timestep) since
it only REDUCES the hip-to-foot distance the leg needs to reach - but
stepping DOWN the same 10cm, which INCREASES that distance, exceeds the
0.1m reach margin established in the leg-length finding above (the
margin was sized for mid-swing dynamics on FLAT ground, not for terrain
elevation changes on top of that). This is because the CoM-height model
(the LIPM) doesn't adapt its assumed constant height to terrain at all -
a real, compounding limitation between two parts of the model
(terrain awareness in the footstep/kinematics layer vs. the LIPM's own
flat-ground assumption) that only shows up when both are exercised
together, exactly the kind of gap a full-pipeline integration test
catches and a single-layer test cannot
(`tests/test_terrain.py::test_stepping_down_exceeds_current_leg_reach_margin`
locks this in as a known, not-yet-solved limitation).

## Bug 5: zero-radius "thin rod" idealization made the 6-DOF mass matrix exactly singular at one specific, common pose

Extending `dynamics/rigid_body_leg.py`'s verified 2-link SAGITTAL model to
the full 6-DOF chain (`dynamics/rigid_body_leg_6dof.py`) reused the same
uniform-rod inertia idealization the 2-link model used - but that model
only ever needed TRANSVERSE inertia (rotation perpendicular to the rod's
own axis, e.g. hip/knee pitch), since it had no way to rotate a link
about its own long axis. The 6-DOF model DOES have that motion
(hip_yaw), and a true zero-radius rod has EXACTLY ZERO moment of inertia
about its own axis. At the legs-hanging-straight-down pose
(`q = [0,0,0,0,0,0]`), hip_yaw's rotation axis lines up exactly with the
thigh's own long axis, so `M[hip_yaw, hip_yaw]` came out to exactly
`0.0` - a genuinely singular mass matrix at a completely ordinary,
frequently-visited pose (standing still), not an exotic edge case.
Caught by `tests/test_rigid_body_leg_6dof.py::test_mass_matrix_is_symmetric_and_positive_semidefinite`
(`eigvalsh` returned a `0.0` eigenvalue for the "active" 4-DOF block at
that exact pose). Fixed by giving each link a stated, non-cited
anthropometric cross-sectional radius (thigh 8cm, shank 5cm - rough
approximations, unlike the cited de Leva mass fractions) and computing a
real solid-cylinder axial inertia `I_axial = 0.5 * m * r^2` instead of
`0`; `test_axial_inertia_removes_the_straight_down_singularity` locks in
that the fix makes the matrix strictly positive definite at that pose.

## Finding: hip_yaw/hip_roll do NOT change the hip_pitch/knee_pitch diagonal block - the real 3D coupling is off-diagonal

First draft of `test_rigid_body_leg_6dof.py` assumed hip_yaw/hip_roll
motion should change the hip_pitch/knee_pitch DIAGONAL block of `M(q)` -
it doesn't, and the test's assumption (not the dynamics code) was wrong.
Because hip_yaw, hip_roll and hip_pitch are three sequential revolutes
sharing one point (a spherical-joint decomposition), rotating the
upstream two just re-orients the whole downstream chain's reference
frame - and body-relative inertia doesn't depend on the orientation of
its own reference frame. Verified this is exact equality (not
"approximately similar"), not just a hunch: `M[2:4,2:4]` at
`q=[0,0,0.3,0.5,0,0]` matches `q=[0.4,0.2,0.3,0.5,0,0]` to float
precision. The real, genuine 3D coupling this 6-DOF model adds over the
2-link one lives in the OFF-diagonal hip_yaw/roll <-> hip_pitch/knee
terms instead: exactly zero when the leg hangs straight down (no
possible cross-coupling when all axes are aligned) and measurably
nonzero once yawed/rolled (`test_hip_yaw_and_roll_couple_into_the_mass_matrix`).

## Bug 7: terrain-adaptive CoM height, first attempt broke a previously-working case; the working fix, and its real remaining boundary

Closing docs/SCOPE.md item 4's stepping-down-a-curb finding
(`dynamics/terrain_adaptive_com_height.py`) took two attempts.

**First attempt (kept here as a documented failure, not silently
discarded):** compute one target CoM height per footstep phase
(`default_zc - max(drop, 0)`, `drop` = how much lower the landing is
than the current stance) and linearly RAMP between phase targets across
each double-support window - mirroring the ZMP reference's own ramp in
`planning/footstep_planner.py`. This fixed the 10cm-step-DOWN case, but
broke the previously-passing 10cm-step-UP case
(`tests/test_terrain.py::test_stepping_up_within_leg_reach_margin_still_converges`
would have started failing). Cause: this project's hip-position model
uses ONE SHARED world-frame hip height for BOTH legs at once
(`simulation/joint_trajectory_generator.py`). Ramping that shared height
UP in preparation for a higher upcoming stance, DURING the double-support
window where the OTHER foot (about to swing next) hadn't moved yet and
was still at the OLD, lower height, pushed that other foot's
INSTANTANEOUS reach demand to the leg's exact maximum (0.9m) - a case
that had comfortable slack before this "fix" and lost it entirely.

**Working fix:** stop planning per-phase targets and ramping between
them; instead compute the hip height CONTINUOUSLY from both feet's
own already-computed z-trajectories: `hip(t) = min(left_foot_z(t),
right_foot_z(t)) + default_zc`. This is provably safe by construction -
it can never demand more than `default_zc` of reach from whichever foot
is currently HIGHER, so it cannot reproduce the first attempt's failure
mode (`tests/test_terrain_adaptive_com_height.py::test_stepping_up_10cm_still_converges_with_adaptive_height`
locks this in). On flat ground it reduces to exactly `default_zc`,
reproducing the original model's output bit-for-bit
(`test_flat_ground_reproduces_the_original_constant_height_exactly`).

**Honest remaining boundary, measured not assumed:** this genuinely
extends the safe step-down depth (10cm now converges, where it failed
completely before), but not to arbitrary depth - a 12cm step down still
converges, a 14cm one does not
(`test_new_failure_boundary_is_real_and_measured_not_claimed_solved`).
The new failure isn't purely a vertical-reach problem either (a quick
check at the failing configuration showed a comfortable vertical margin
but an IK non-convergence on the SWINGING foot specifically, suggesting
the binding constraint past ~12-14cm is the COMBINED horizontal+vertical
reach, or a solver/warm-start sensitivity near the workspace boundary,
not pinned down further here) - reported honestly as a real, smaller,
still-present limit rather than claimed away.

## Finding: naive multi-step push recovery does NOT beat single-step, for a real, measured mechanistic reason

Extending `simulation/push_recovery_simulator.py` to keep replanning
footsteps after the first one (`n_recovery_steps` > 1 - closing
docs/SCOPE.md's "multi-step (not just single-step) push-recovery
replanning" gap) was implemented and works mechanically (each
subsequent recovered footstep's capture point is computed from the
ACTUALLY-SIMULATED CoM state at that moment, not a hand-rolled
prediction; the un-touched remainder of the gait is rigidly shifted to
stay consistent with the recovered position afterward - see the module
docstring). But the measured result for THIS system is a genuine,
useful negative one, not the hoped-for improvement:

For a 1.0 m/s lateral push, `n_recovery_steps=1` and `=3` give
IDENTICAL results (4 later violations either way) - the second and
third "corrections" end up choosing targets indistinguishable from
nominal, because the first correction already arrested most of the
disturbance. For LARGER pushes (1.5 m/s, 2.0 m/s), where the earlier
single-step finding already noted the benefit "can even disappear",
multi-step makes it measurably WORSE (14 -> 16 later violations at
1.5 m/s; 19 -> 21 at 2.0 m/s -
`tests/test_multi_step_recovery.py::test_naive_multistep_recovery_does_not_beat_single_step_for_large_pushes`
locks in these exact counts).

**Root cause, verified directly, not guessed:** consecutive
independently-computed capture points land close TO EACH OTHER
laterally instead of alternating properly to opposite sides. For the
1.5 m/s case, footsteps 3, 4, 5's recovered y-positions come out to
0.380, 0.350, 0.343 - only 2.9cm and 0.7cm apart
(`test_naive_multistep_recovery_crowds_consecutive_footsteps_laterally`),
versus the gait's own nominal 12cm stance width. This happens because
each footstep's capture point is computed from whatever CoM LATERAL
VELOCITY exists at that moment - and a single step's worth of tracking
isn't enough to reverse the push-induced velocity, so the "correction"
for the NEXT footstep, computed independently, gets pulled toward the
same side again instead of returning to the gait's normal alternating
pattern - a real geometric near-crossover a physical robot's legs
couldn't actually execute, which this simplified 2D-footprint model
doesn't itself flag as infeasible.

This is not a bug in the implementation (the mechanism does exactly
what it was built to do); it's a real limitation of the NAIVE strategy
of repeating a single-step formula independently at each future
footstep. It's also not evidence multi-step recovery is a bad idea in
general - real N-step-capturability methods (Pratt & Koolen and
successors) solve for a whole sequence of future footsteps JOINTLY,
which would let a later step's placement account for what an earlier
one already committed to (including, presumably, alternating sides
properly) - that joint solve is a substantially harder optimization
problem than what's implemented here, and remains open.

## Bug 8: ankle joint origins were dead placeholder values (and mislabeled in a comment) - harmless until the foot link needed them

`dynamics/rigid_body_leg_6dof.py`'s internal `_joint_frames()` set the
ankle_pitch/ankle_roll joint origins to `knee_pos`, with a comment
claiming they were "co-located with the hip" - neither the value nor
the comment was correct (ankle joints are physically located at the
ankle, the bottom of the shank). This was completely harmless for that
module's own thigh/shank-only mass matrix: `_THIGH_JOINTS = (0,1,2)`
and `_SHANK_JOINTS = (0,1,2,3)` never reference index 4 or 5's origin
at all, so the wrong value was simply never read - a genuine dead-code
bug, not a live one, and none of `test_rigid_body_leg_6dof.py`'s 9
tests could have caught it (correctly - it had no observable effect on
anything that module computed).

It became a REAL, live bug the moment `dynamics/foot_inertia.py`
needed a correct foot-COM Jacobian: the foot is distal to ALL 6 joints,
so its Jacobian's ankle_pitch/ankle_roll columns actually use those
origins. Fixed directly in `rigid_body_leg_6dof.py` (ankle origins now
computed as the real ankle position via the same rotation chain the
knee position already uses) - verified this fix changes NOTHING in
that module's own test suite (all 9 still pass unchanged), confirming
it really was dead code before and is now correct where it matters.

## Foot inertia model: closes the last stated gap in the leg's own rigid-body dynamics

`dynamics/foot_inertia.py` adds the foot as a third rigid body to
`rigid_body_leg_6dof.py`'s mass matrix/gravity/Coriolis, using the SAME
geometric-Jacobian technique, added onto (not replacing) the
already-verified thigh+shank result: `M_total = M_thigh_shank + M_foot`,
same for `G` and (since the Christoffel-symbol formula is linear in the
mass matrix) `C`. That additivity claim for `C` isn't just asserted -
`test_coriolis_additive_shortcut_matches_independent_full_christoffel_derivation`
checks it against a totally separate, from-scratch Christoffel
computation run on the FULL 3-body mass matrix, not merely internal
self-consistency.

**Result**: the ankle rows/columns of `M`, which
`rigid_body_leg_6dof.py` correctly and honestly left at exactly zero
(no foot link existed), are now genuinely nonzero
(`test_ankle_rows_are_no_longer_structurally_zero`), and the full
6-joint mass matrix is strictly positive definite everywhere tested,
including the legs-hanging-straight pose that was bug #5's original
singularity (adding the foot's off-axis mass removes the last
degenerate direction). The foot's mass fraction (1.37% of total body
mass) is the real de Leva (1996) male regression value, same source as
the thigh/shank fractions; its COM location and box dimensions are
STATED, explicitly-approximate placement assumptions (see the module
docstring), not measured anthropometric data - honestly labeled as
such rather than presented with false precision.

## Actuator electrical dynamics: closes the last item on SCOPE.md's original open-questions list

`actuators/actuator_dynamics.py` builds a real, sourced electrical
model of the Dynamixel MX-106 (the same actuator `actuators/motor_specs.py`
already uses): resistance and torque/back-EMF constants derived from
real, published datasheet numbers (V=12V, stall current 5.2A, stall
torque 8.40 N*m, no-load current 0.17A, no-load speed 45 RPM - Robotis/
CrustCrawler MX-106 spec sheet), using the standard manufacturer formula
`Kt = stall_torque / (stall_current - no_load_current)` (the same
convention Bodine Electric's own "Motor Constants for Gearmotors"
technical note uses). Inductance is NOT published for this integrated
smart-servo, so a cited, stated approximation is used instead (11.5 ms
electrical time constant, measured for a comparable small PMDC motor -
University of Utah ECE 3510 lab data) rather than an invented number.

**Real, honest tension found while deriving this**: calibrating Kt/Kb
from the stall-torque data point this way and using it to PREDICT the
no-load speed (`omega = (V - I_noload*R) / Kb`) gives 66.4 RPM - a 47%
OVER-prediction of the datasheet's real 45 RPM. A single linear
Kt=Kb-with-one-Coulomb-friction-term model cannot fit both the real
stall-torque and no-load-speed operating points of this actual,
low-cost integrated servo simultaneously (real efficiency and friction
are speed-dependent in ways this simple model doesn't capture). This
module deliberately calibrates to the stall/high-torque regime, since
that's what `dynamics/leg_dynamics.py`'s feasibility checks care about
most, and states the resulting speed-prediction inaccuracy plainly
rather than hiding it (`predicted_no_load_speed_rad_s` is exposed on
the params object specifically so this can't be silently forgotten).

**Applying the model to a REAL gait's required torque - two separate,
honest findings, not conflated:**

1. The hip_pitch swing torque for this gait (already computed by
   `dynamics/foot_inertia.py`) peaks at 9.30 N*m - which ALREADY EXCEEDS
   the MX-106's 8.40 N*m STALL rating on pure magnitude alone, before
   electrical dynamics even enter the picture. This is a torque-
   MAGNITUDE finding (consistent with the much larger, already-
   documented stance-leg finding earlier in this file - 32.2 Nm needed
   vs 1.68 Nm continuous / 8.4 Nm stall), not an electrical-dynamics
   one - testing electrical bandwidth on a torque profile the motor
   can't physically produce anyway would conflate two different failure
   modes. The knee joint's swing torque for the SAME gait (0.49-2.06
   N*m) stays comfortably within the MX-106's real limits, making it the
   fair test case for what follows.

2. Tracking that real knee-torque profile through the full electrical
   model (current never exceeds ~1.1A of the 5.2A stall rating, voltage
   command never saturates the 12V supply) still shows a real, physical
   tracking error - but a precisely characterized one, not a vague "some
   lag exists": away from velocity-reversal instants, tracking error
   stays under 0.2 N*m (a modest fraction of the 0.5-2.1 N*m torque
   range) - confirming the ~11.5ms electrical time constant is indeed
   fast relative to this ~0.4s swing phase's smooth torque changes, as
   expected. But AT each velocity-reversal instant (where the required
   Coulomb-friction compensation direction flips discontinuously - this
   swing phase crosses zero velocity multiple times), tracking error
   spikes to 0.40-0.50 N*m for a single sample before decaying back
   within a few electrical time constants - a real, physical consequence
   of trying to drive an ACTUALLY-discontinuous required-current signal
   through a finite-inductance winding, not a modeling artifact (an
   earlier version of this analysis conflated this real spike with an
   unrelated, spurious cold-start transient from initializing simulated
   current at 0A at the start of an already-in-progress swing phase -
   fixed by starting from the steady-state current the required torque
   already implies, which is what a real, already-operating controller
   would be doing).

**Honest, stated scope limit**: gearbox BACKLASH (the other half of
SCOPE.md's phrase "motor inductance/friction") is explicitly NOT
modeled - it's a position-dependent, hysteretic dead-zone effect,
fundamentally different from the continuous current/torque dynamics
modeled here, and would need its own separate treatment in the position
control loop, not this one.


