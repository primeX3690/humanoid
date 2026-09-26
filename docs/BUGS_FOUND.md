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

