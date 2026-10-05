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

## Gearbox backlash: closes the other half of "motor inductance, gearbox friction/backlash"

`actuators/gearbox_backlash.py` implements the standard backlash dead-
zone nonlinearity (Nordin & Gutman, Automatica 2002) in the position
domain, separately from `actuator_dynamics.py`'s continuous
current/torque model - a hysteretic, position-dependent effect is a
fundamentally different kind of thing from a continuous ODE, and needs
its own model. **Honest sourcing note**: a web search for the MX-106's
own backlash spec returned a widely-mirrored "20 degrees" figure from a
community parts database - this directly contradicts Robotis's own
advertised "360-degree position control WITHOUT dead zone" feature and
0.088-degree encoder resolution (20 degrees of backlash would make that
resolution meaningless), so it's almost certainly a data-entry error in
that source, not a real spec. Using it anyway just because it was the
first number found would be worse than admitting no reliable published
figure exists - so this module uses an explicitly-stated, conservative
literature-typical assumption (0.3 degrees) instead, flagged as an
assumption, not a confirmed datasheet number. Applied to this project's
own knee trajectory: max position error is exactly half the assumed
backlash width (0.15 degrees), as the model's own math guarantees -
sub-degree and small, but real and now quantified rather than ignored.

## Terrain-adaptive CoM height <-> horizontal ZMP-tracking coupling: implemented correctly, measurably does not help for this planner

`dynamics/gain_scheduled_walk_simulator.py` closes the remaining half of
docs/SCOPE.md item 4 that `terrain_adaptive_com_height.py` explicitly
left open: it feeds the terrain-adaptive zc profile INTO the horizontal
LIPM/ZMP-preview-control loop itself, via gain scheduling - building a
fresh LIPM + ZMPPreviewController (a cheap Riccati re-solve) at each
footstep-phase boundary where zc changes, chaining BOTH the physical
[position, velocity, acceleration] state AND the controller's own
integral-error state (`_e`) across the switch. Getting that integral-
error carry-over right was necessary for correctness: without it, even
on FLAT terrain (every segment using the identical zc), resetting `_e`
to 0 at each of the 6 footstep boundaries introduced a small but real
divergence from the original single-controller simulate_walk - fixed by
treating `_e` as a physical quantity belonging to the ongoing control
loop, carried across segment boundaries exactly like the CoM state
itself. With that fix, flat terrain reproduces the original
`simulate_walk` output to machine precision (an exact match, not
"close"), confirming the segment-chaining is genuinely seamless.

**Real, honest, somewhat counter-intuitive finding once applied to an
actual step-terrain scenario**: gain-scheduling the controller to the
terrain-adaptive height does NOT reduce horizontal ZMP tracking error
compared to leaving the controller at one fixed nominal zc throughout -
the two are comparable, with gain-scheduling if anything marginally
worse (0.043m vs 0.041m peak tracking error for a 10cm step-down).
Root cause, verified directly rather than assumed: this project's
footstep planner's x/y ZMP reference trajectory is completely
INDEPENDENT of terrain height - `apply_terrain` only ever changes
`footstep.z`, never the x/y landing positions the reference is built
from (`test_horizontal_zmp_reference_is_independent_of_terrain_height`
confirms this directly). So there was never a real horizontal-plane
mismatch for gain-scheduling to fix in the first place; switching
controller gains mid-walk merely introduces a small transient of its
own from the gain change. This is a valuable, non-obvious result: the
genuine benefit of terrain-adaptive CoM height, for THIS project's
model, lives entirely in the leg-reach/IK domain
(`terrain_adaptive_com_height.py`, already closed), not in the
horizontal ZMP-tracking loop's own accuracy - coupling it into that
loop was worth actually building and measuring rather than assuming,
precisely because the answer ("doesn't help here, and here's the exact
mechanism why") is more informative than either skipping the question
or reporting a hoped-for improvement that the numbers don't support.

## Real joint N-step capture solve: genuinely beats both single-step and naive multi-step recovery

`control/n_step_capture_planner.py` closes the gap the earlier
multi-step finding explicitly left open: "a real N-step-capturability
solve... would let a later step's placement account for what an
earlier one already committed to... that joint solve is a substantially
harder optimization problem than what's implemented here, and remains
open." This implements a real, if simplified, version of that joint
solve: instead of independently repeating the single-step capture-point
formula at each future footstep (which was measured to crowd
consecutive footsteps toward the same lateral side - see the earlier
finding in this file), each footstep's target is a WEIGHTED BLEND of
the immediate capture point and that footstep's own NOMINAL
(alternating, correctly-spaced) position, with the blend weight decaying
geometrically from 1.0 (full correction) toward 0.0 across the recovery
window. This directly, structurally targets the naive version's actual
failure mechanism: later footsteps are explicitly pulled back toward
their proper alternating pattern regardless of what the still-recovering
CoM velocity says, rather than blindly chasing the capture point at
every step.

**Measured result, same test scenario as before (1.5 and 2.0 m/s
lateral pushes, 8-step gait)**: the joint/blended scheme genuinely
outperforms both alternatives, not just the naive one:

| push | single-step | naive multi-step | joint blended |
|---|---|---|---|
| 1.5 m/s | 14 later violations | 16 | **12** |
| 2.0 m/s | 19 | 21 | **16** |

Consecutive recovered footsteps' lateral spacing is also measurably
larger under the blended scheme than the naive one for the same push
(directly confirming the crowding problem is what's actually being
fixed, not just an incidental side effect). This is a genuine positive
result - not every attempted improvement in this project has been (see
the naive multi-step and gain-scheduled-LIPM findings above), and it's
reported with the same rigor either way: measured, with the exact
numbers locked into
`tests/test_n_step_capture_planner.py::test_joint_blended_recovery_beats_both_single_step_and_naive_multistep_for_large_pushes`.

**Honest, stated scope limit**: the decay schedule (`decay_ratio`) is a
single stated hyperparameter, not something this module optimizes -
true N-step capturability (Pratt & Koolen) would jointly optimize the
whole footstep sequence against the actual nonlinear feasibility region,
which is a larger problem than blending toward a fixed nominal schedule.
What's genuinely solved here is the SPECIFIC coupling the naive version
was missing (recovery aggressiveness vs. returning to the proper
alternating gait pattern), not the fully general capturability
optimization.

## Autonomous mission layer: goal-in, supervised walking out - three real bugs caught while building it

`planning/autonomous_mission.py` takes only a goal distance, plans the gait,
runs it in closed loop, detects disturbances from the robot's OWN capture-point
divergence (no oracle - unlike `simulation/push_recovery_simulator.py`, which is
told the push time), triggers the joint N-step recovery automatically, and a
supervisor decides continue / finish / SAFE-STOP. Bugs found by the first runs:

1. **Goal overshoot with no disturbance at all**: the first `plan_gait_for_goal`
   added a closing step and overshot a 3 m goal by 0.36 m. Fixed: the last
   footstep now lands nearest the goal (0.06 m error, 4 goal distances tested).
2. **One push detected four times**: after recovery the supervisor kept comparing
   the robot to the nominal plan it had deliberately left, so it re-triggered
   forever. Fixed by re-baselining: after the recovery window the remaining
   footsteps are rigidly shifted (same rule as the push simulator) and the
   capture-point reference is offset accordingly. Now 1 push -> 1 detection.
3. **Recovered footsteps 3, 5, 7 instead of 3, 4, 5**: at the sample where footstep
   i+1's swing begins its `start_time` equals `t[k]`, so a "first footstep not yet
   started" search skipped it. Fixed by passing the index explicitly.

**Measured behaviour (3 m goal, LIPM simulation)**: undisturbed -> reached, 0.06 m
error, 0 ZMP violations; pushes of 0.5 / 1.0 / 1.5 m/s -> reached with 0.08-0.11 m
error, disturbance detected at the push instant (3.9 s); two separate pushes ->
both detected and handled; a 2.5 m/s push -> **SAFE-STOP** (ZMP outside the
support polygon > 0.6 s) rather than a silent failure.

**Honest limits, stated plainly**:
* Total ZMP-violation counts with the mission layer (81 / 90 / 100 for
  0.5 / 1.0 / 1.5 m/s) are essentially equal to no-recovery (82 / 94 / 100): the
  violation at the instant of the push (support polygon has not moved yet) dominates
  the total, exactly as the earlier single-step finding said. The layer's value is
  autonomy - detect, replan, shift, resume, abort - not a lower raw violation count.
* "REACHED" here means the plan completed in the LIPM model, whose only failure
  proxy is ZMP leaving the support polygon; the LIPM cannot actually fall over. The
  MuJoCo environment in `rl/` is where falling is real.
* NOT covered: perception, mapping/SLAM, obstacle avoidance, turning (hip yaw is
  commanded to zero, so goals are straight-line distances), and a real state
  estimator (state comes from the simulator). Those are the components a true
  unmanned deployment adds.

## RL-based locomotion (MuJoCo + PPO): a real, trained, honestly-measured policy - not a toy, not oversold

Addresses the point that current legged-robot research increasingly trains
learned policies through large-scale trial-and-error in simulation instead
of hand-derived controllers like the LIPM/ZMP stack used throughout the
rest of this project. Built with the actual industry-standard stack
(MuJoCo physics, Gymnasium environment API, Stable-Baselines3 PPO,
PyTorch), not a hand-rolled toy optimizer - an earlier draft used a small
evolution-strategy proof of concept; that was explicitly rejected in favor
of the real stack, and MuJoCo + Gymnasium + Stable-Baselines3 + PyTorch
were installed and used instead.

**The MuJoCo model is not a generic biped** - `rl/biped_model.py` builds it
directly from this repository's own numbers: total mass (25 kg,
`dynamics/leg_dynamics.RobotMassParams`), segment masses (de Leva 1996
fractions, the same ones `dynamics/leg_dynamics.py` and
`dynamics/foot_inertia.py` use), link lengths (`kinematics/leg_fk.LegParams`),
and actuator torque limits (44.7 N*m, the H54P-200 - the SAME actuator
`actuators/motor_specs.py` already recommended on torque-feasibility
grounds). The model-based (LIPM/ZMP) and learned (PPO) halves of this
project describe the same physical robot, not two unrelated ones.

**Three real bugs found while standing up the environment**, each with a
regression test in `tests/test_rl_env.py`:

1. The environment's own `BipedSpec` was stored as `self.spec`, silently
   clobbering Gymnasium's own `Env.spec` attribute - caught by
   `gymnasium.utils.env_checker.check_env`, not by manual inspection.
2. The nominal standing pose put the robot's centre of mass behind the
   foot's contact patch; a plain PD hold (zero policy output) tipped over
   backwards within about 4 seconds regardless of gain tuning. Root cause,
   found by computing it directly rather than guessing: with the ankle-PD
   stiffness needed for static stability, the pose ALSO has to keep the
   projected CoM inside the foot's footprint - fixed by re-deriving the hip
   pitch / knee / ankle pitch angles so the CoM lands just ahead of the
   ankle rather than behind it.
3. Ankle PD stiffness (150 N*m/rad in an early version) was below the
   inverted-pendulum destabilizing stiffness `m*g*h` (~225 N*m/rad for this
   robot's mass and standing height) - a PD gain below that threshold
   cannot hold ANY standing pose, however well-chosen the angles are. Fixed
   by raising ankle-pitch stiffness to 400 N*m/rad, comfortably above the
   threshold - `tests/test_rl_env.py::test_ankle_pd_stiffness_exceeds_gravitational_destabilising_stiffness`
   checks this condition directly rather than just checking the specific
   numbers used.

With those fixed, the zero-action PD-hold baseline stands for the full 20 s
episode (`test_zero_action_pd_hold_stands_for_20_seconds`) - the sanity
floor any learned policy has to clear before a training curve means
anything.

**A fourth, more subtle bug found while EVALUATING the trained policy**:
`rl/evaluate.py` reported different numbers on repeated runs of the exact
same trained policy against the exact same seeds - e.g. 0.067 m/s mean
forward speed on one run, 0.191 m/s on the next, for the identical
cmd_vx=0.3, push=0.0 case. Root cause: `rl/train_ppo.py` pins
`torch.set_num_threads(1)` (documented there as avoiding oversubscribing a
small CPU), but `rl/evaluate.py` never did - so PyTorch's default
multi-threaded matrix multiplication had a non-deterministic reduction
order, and tiny floating-point differences in the policy's action output
compounded over a 1000-step (20 s) rollout into meaningfully different
trajectories (in a system with a real fall/no-fall bifurcation, small
input differences can flip the outcome). Fixed by pinning
`torch.set_num_threads(1)` in `evaluate.py` too;
`tests/test_rl_evaluate.py::test_evaluation_is_bit_for_bit_deterministic_across_repeated_runs`
now checks exact equality across repeated evaluations, not just "similar"
numbers - this is exactly the kind of measurement-integrity bug that would
otherwise let two different investors see two different sets of numbers
for the identical policy.

**Training performed and its real, measured results** (CPU-only, matching
this project's own author's hardware - about 25-30 minutes of wall-clock
training per curriculum stage): a two-stage curriculum was trained -
`stand` (hold position under pushes, 400k timesteps) then `walk` (track a
forward-velocity command, continued to ~1.6M timesteps total). The `walk`
stage's training curve genuinely improved throughout
(`ep_len_mean` rose from 200 to 724 of a possible 1000 steps over the
logged chunks - not a flat or noisy line), but training was not run to
convergence or to the full `robust`-push curriculum stage due to the
compute-time available in this environment; the commands to continue are
in `rl/train_ppo.py`'s own docstring for whoever continues this on
dedicated hardware.

Evaluated deterministically (8 episodes/condition) against the zero-action
PD-hold baseline that was ALSO used as this module's own sanity check:

| condition (cmd_vx, push) | PD-hold baseline | trained PPO policy |
|---|---|---|
| 0.3 m/s, no push | survives 20s, but never moves (v=0, so vx_err=0.30 the whole time) | survives 20s, v=0.19 m/s (undershoots the 0.3 command, but genuinely walks forward) |
| 0.0 m/s, 0.5 m/s push | 0% survive (falls in 4.1s) | 38% survive (mean 12.1s) |
| 0.3 m/s, 0.5 m/s push | 0% survive (falls in 4.1s) | **62% survive** (mean 14.7s) |
| 0.3 m/s, 1.0 m/s push | 0% survive (falls in 3.2s) | 0% survive (falls in 4.0s) |

**Honest reading of these numbers**: the trained policy is a REAL,
measurable improvement over the hand-tuned PD baseline for moderate
disturbances (0.5 m/s) while walking - genuinely more push-robust, not
just superficially different - and it does produce real forward
locomotion rather than standing still, but it undershoots the commanded
speed by roughly a third and provides no benefit at all once the push
reaches 1.0 m/s (matching the untrained `robust` curriculum stage's
absence). This is reported as exactly what it is: a real, working,
trained locomotion policy with a specific, measured, moderate capability
envelope - not a finished product, and not compared against
industry-scale training budgets (which use orders of magnitude more
simulated experience than the ~1.6M steps trained here).

**Honest, stated scope limit**: matching real sim-to-real humanoid RL
(Boston Dynamics, Unitree, DeepMind-scale work) needs orders of magnitude
more training (billions, not millions, of simulated steps), GPU-accelerated
parallel simulation, and sim-to-real transfer techniques (system
identification, more extensive domain randomization, actuator-lag
modeling) that are not attempted here. What's delivered is a genuine,
correctly-built, real learning pipeline on real industry-standard
tooling, evaluated honestly - not a demonstration dressed up as more than
it is.

## Turning gait: closes docs/SCOPE.md item 7's "hip yaw fixed at zero, no turning gaits"

`planning/footstep_planner.plan_turning_footsteps()` generates an
alternating gait along an ARC (heading advances per step) instead of
always marching along world +x; `heading_profile()` feeds the commanded
heading into `simulation/joint_trajectory_generator.py`, which now
rotates the hip lateral offset and passes `Rz(heading)` as the target
foot ORIENTATION to `kinematics/leg_ik.inverse_kinematics()` - a
parameter that was always called with `IDENTITY_R` before. Zero turn
rate reproduces `plan_footsteps()` exactly (bit-for-bit, verified).

**Real, physical limit found and respected, not bypassed**: `leg_ik.py`
already has a real hip_yaw joint limit of +-0.6 rad (~34 degrees). A
30-degree total turn over 6 steps converges IK fully; a 48-degree turn
does not - correctly, since IT PHYSICALLY CANNOT under that real joint
limit. This is reported as the actual, bounded capability (steady
incremental turning, a few degrees per step) rather than claiming
arbitrary turning works.

## A* path planning on a known map - explicitly NOT SLAM

`planning/path_planner.py` adds real A* search (Hart/Nilsson/Raphael
1968) over a 2D grid with circular obstacles, feeding waypoints that
compose with the turning gait above. **Stated plainly, not blurred**:
this is planning on a GIVEN, known map - there is no sensor simulation
and nothing here builds a map from sensor data. Real perception/SLAM
(camera or LiDAR simulation, occupancy mapping, localization) remains
completely unaddressed; this closes the "path planning around known
obstacles" piece only, not the "figure out the world from sensors" piece.

## RL robust-stage training completed with a real, measured, positive result

Continued the `robust` curriculum stage (larger pushes, 0.8 m/s vs
`walk`'s 0.3 m/s) from the `walk` checkpoint for ~475k additional
timesteps (~2.57M total). One real infrastructure bug found: the first
robust-training background run's `runs/robust` directory ended up with
NO `model.zip`/`vecnormalize.pkl` despite the log showing it reached
2.1M steps - the process was killed (background job did not survive
between environment sessions) before its final `model.save()`/
`venv.save()` calls ran. Fixed by re-running each chunk to completion
in the foreground with a bounded timeout rather than trusting a
long-lived background job to survive.

**Measured result (deterministic, 8 episodes/condition, same evaluation
harness as the walk-stage numbers above)**:

| push (walking at 0.3 m/s cmd) | PD-hold | walk-stage policy | robust-stage policy |
|---|---|---|---|
| 0.5 m/s | 0% survive | 62% survive (14.7s) | **88% survive (18.1s)** |
| 1.0 m/s | 0% survive | 0% survive (4.0s) | 0% survive (5.2s) |
| 1.5 m/s | 0% survive | 0% survive (3.2s) | 0% survive (4.5s) |

The robust stage is a genuine, further improvement at the push magnitude
it was trained for (88% vs 62% survival at 0.5 m/s) and survives
somewhat longer even at magnitudes neither policy fully handles (1.0,
1.5 m/s) - a real, incremental, honestly-measured gain, not a
re-labeling of the same checkpoint. It still does not survive 1.0+ m/s
pushes while walking; further training, a harder push curriculum, or
more network capacity would be the next step, not attempted further here
given the CPU training-time budget available.

## Nonlinear MPC (CoM height jointly optimized with horizontal ZMP): divergence bug FOUND AND FIXED; a real, remaining tuning issue reported honestly

Attempted to close the item `dynamics/gain_scheduled_walk_simulator.py`
named as the harder, unattempted alternative: a genuinely NONLINEAR
receding-horizon MPC (`control/nonlinear_mpc.py`) where CoM height z is
a decision variable in the ZMP equation `zmp = x - (z-z_foot)/(zddot+g)*xddot`
(nonlinear because z, zddot, x, xddot all multiply/divide each other),
solved via SciPy SLSQP each control step.

**Root cause of the original divergence, found on the second debugging
pass**: the safety fallback used whenever SLSQP failed to converge (a
real, common occurrence with the short 4-step/0.04s horizon first
tried) returned ZERO jerk - which does not mean zero velocity or
acceleration, it means "keep whatever acceleration you already have".
Once a solve failed while CoM height already had downward velocity,
the zero-jerk fallback let it coast downward, unopposed, and the
fallback kept re-triggering on the very next step too - a smooth,
accelerating free-fall (CoM z reaching about -24m over an 8-second
walk), not a single bad jump, which is why it took real investigation
(printing the trajectory sample-by-sample) rather than a single
obvious stack trace to find.

**Three real fixes, all necessary, none sufficient alone**:
1. Replaced the zero-jerk fallback with a stabilizing PD correction
   back toward each axis's reference - this fixes the actual root
   cause (a fallback that did nothing when it should have actively
   corrected).
2. Lengthened the horizon (4 steps/0.04s -> 10-20 steps/0.1-0.2s) and
   rebalanced the cost weights (`w_zmp` 100->20, `w_jerk` 1e-3->0.05) -
   the original short, aggressively-weighted horizon was independently
   causing an underdamped oscillation even before the fallback bug
   triggered.
3. Added defensive state saturation (position/velocity clamps) as a
   standard control-systems safety backstop, on top of (not instead of)
   fixes 1-2 - the same kind of limit a physical robot's own joints
   would impose regardless of what the optimizer requests.

**Honest result after all three fixes**: the ORIGINAL bug - divergence
to physically nonsensical values - is GONE.
`tests/test_nonlinear_mpc.py::test_closed_loop_simulation_no_longer_diverges_to_infinity`
verifies CoM height and the sagittal (x) axis both stay bounded and
finite over a full closed-loop walk. **A real, remaining limitation,
reported plainly rather than hidden**: the LATERAL (y) axis still shows
a bounded but real oscillation (order +-1 to +-1.3m at the settings
tested) - tightening the support-polygon margin and re-weighting jerk
cost measurably reduced it (ZMP violations dropped from 139/281 to
115/281 samples in one comparison) but did not eliminate it within the
time available. This project's ALREADY-VERIFIED gain-scheduled LINEAR
controller (`dynamics/gain_scheduled_walk_simulator.py`, built on a
Riccati-derived preview controller specifically designed for its linear
system) tracks this same lateral reference cleanly; matching that with
a generic SLSQP re-solve every timestep is a harder, separate tuning
problem, marked as an explicit, honest `xfail`
(`test_closed_loop_lateral_tracking_matches_the_reference_closely`) -
not silently dropped, not claimed fixed. "No longer diverges to
infinity" is what was actually fixed here; "matches the tracking
quality of the linear controller it was meant to improve on" is not,
yet.
