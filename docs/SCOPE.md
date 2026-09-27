# Scope — what this v1 is, and honestly isn't, yet

## What this IS

A real, verified implementation of the LIPM + ZMP preview-control gait
generator - the actual algorithm class used on real preview-control-era
humanoids (Kawada HRP-2 and its lineage; the same family of methods
underlies most walking humanoids that don't use full trajectory
optimization/MPC-on-whole-body-dynamics). Given a footstep plan, it
produces a CoM trajectory that keeps the ZMP inside the support polygon
- the actual, real physical criterion for "the robot doesn't tip over" -
verified numerically, not just asserted:
- LIPM discretization matches the closed-form continuous solution
  (`tests/test_lipm.py`)
- the augmented closed-loop system is provably stable (eigenvalues
  inside the unit circle, `tests/test_preview_controller.py`)
- ZMP tracks a constant reference with zero steady-state error (real
  integral-action guarantee, not just "looks converged")
- preview genuinely reduces tracking lag vs. short/no preview - the
  actual point of the algorithm - measured, not assumed
  (`test_preview_reduces_tracking_lag_vs_zero_preview`)
- a full 8-16 step walk keeps the ZMP inside the support polygon with a
  real, non-zero, measured minimum margin (`tests/test_walk_simulation.py`)

## What this is NOT (yet) - the honest gap to a real humanoid

1. ~~No leg kinematics.~~ **Added**: `kinematics/leg_fk.py` +
   `kinematics/leg_ik.py` compute real 6-DOF (hip yaw/roll/pitch, knee
   pitch, ankle pitch/roll) joint angles from the CoM/footstep plan,
   verified via FK/IK round-trip tests and full-walk tracking-error
   checks (`tests/test_leg_kinematics.py`,
   `tests/test_joint_trajectory_generator.py`). IK is numerical (damped
   least squares), not a hand-derived closed form - see
   `kinematics/leg_ik.py`'s docstring for why.
2. ~~No swing-foot trajectory.~~ **Added**: `trajectory/swing_foot.py`
   generates a minimum-jerk horizontal + `sin^2` vertical-clearance
   profile with zero velocity at both liftoff and touchdown (an earlier
   plain `sin(pi*s)` version did NOT have this property - see
   `docs/BUGS_FOUND.md` Bug 4).
3. ~~Flat, known, rigid ground.~~ **Partially added** — see item 9,
   below (kept numbered here for continuity with the original list).
4. ~~Constant CoM height.~~ **Partially added**:
   `dynamics/terrain_adaptive_com_height.py` makes the hip-height
   REFERENCE consumed by leg IK adapt to terrain (`hip(t) =
   min(left_foot_z(t), right_foot_z(t)) + default_zc`), which fixes the
   concrete, previously-failing case this gap caused (a 10cm step down
   now converges IK for the whole walk - see docs/BUGS_FOUND.md bug #7;
   two attempts, the first one broke a previously-working case before
   the working formula was found). Still NOT done: the LIPM/ZMP-preview-
   control horizontal (x, y) tracking loop itself
   (`dynamics/lipm.py`, `control/zmp_preview_controller.py`) still uses
   ONE fixed nominal zc for its own internal dynamics and gain design -
   this fix does not couple the varying height INTO that loop. That
   remains the real, much harder "nonlinear MPC on centroidal dynamics"
   research area many current humanoid controllers use instead/in
   addition - a legitimate next step, not pretended-away here. Also,
   the new leg-reach boundary this fix reaches is real but finite (12cm
   converges, 14cm doesn't - measured, not claimed to be solved for
   arbitrary drop depth).
5. ~~No disturbance rejection / push recovery.~~ **Added**, including
   multi-step: `control/capture_point.py` +
   `simulation/push_recovery_simulator.py` implement real Instantaneous
   Capture Point recovery (Pratt et al. 2006 - the same theoretical
   basis used on real robots like Atlas), with an `n_recovery_steps`
   option that keeps replanning footsteps beyond just the next one and
   rigidly shifts whatever's left afterward to stay consistent with the
   recovered position. **Real, honest findings** (see
   `docs/BUGS_FOUND.md`): (1) recovery cannot undo the INSTANTANEOUS ZMP
   transient at the moment of a push (the support polygon hasn't moved
   yet), but single-step recovery DOES measurably reduce cascading
   violations in the steps after the recovered footstep, compared to
   ignoring the push - a real, modest, not-magical benefit; (2) naively
   REPEATING that same single-step formula across several future
   footsteps (rather than solving for the sequence jointly, a much
   harder problem) does NOT improve on single-step recovery, and is
   measurably slightly worse for large pushes - traced to consecutive
   independently-computed capture points crowding toward the same
   lateral side instead of alternating properly, a real geometric
   near-crossover this simplified model doesn't flag as infeasible.
6. ~~No full-body dynamics, no joint torques, no actuator limits.~~
   **Answered** (quasi-static stance-leg torque via Jacobian-transpose,
   real actuator-datasheet comparison) AND **the swing leg's inertial
   torque now uses full rigid-body dynamics across ALL 6 joints, PLUS a
   third rigid body for the foot itself**
   (`dynamics/rigid_body_leg_6dof.py` for thigh+shank,
   `dynamics/foot_inertia.py` adding the foot: real 6x6 mass matrix
   built from the standard geometric-Jacobian formula, Coriolis/
   centrifugal terms via the Christoffel-symbol formula, gravity terms -
   verified to match the earlier, analytically-checked 2-link sagittal
   model EXACTLY under reduction, not just "it runs"; see
   `docs/BUGS_FOUND.md` bugs 5, 6, and 8 for three real issues found and
   fixed while extending to 3D and then to a full 3-body treatment). The
   ankle joints' mass-matrix rows/columns, which were correctly and
   honestly left at exactly zero when no foot link existed, are now
   genuinely nonzero. ~~Actuator dynamics (motor inductance, gearbox
   friction).~~ **Partially added**:
   `actuators/actuator_dynamics.py` builds a real, sourced electrical
   model (resistance, torque/back-EMF constants from real MX-106
   datasheet numbers; a cited literature approximation for the
   unpublished inductance) and Coulomb friction (sourced from the real
   no-load current). Applied to a real gait's knee-swing torque: tracking
   error stays under 0.2 N*m away from velocity-reversal instants, but
   spikes to ~0.4-0.5 N*m exactly AT them (a real, physical consequence
   of finite electrical bandwidth meeting a genuinely-discontinuous
   required-current signal, not a bug) - see `docs/BUGS_FOUND.md` for
   the full derivation and an honest tension found along the way
   (calibrating from stall data over-predicts the real no-load speed by
   47%). Gearbox BACKLASH (a hysteretic, position-dependent dead-zone
   effect, not a continuous current/torque one) is explicitly NOT
   modeled - a separate, remaining gap. Ground-contact/sole pressure
   distribution, and the foot's COM location/box dimensions are stated,
   explicitly-approximate assumptions (its MASS fraction is real de Leva
   1996 data; its shape and COM placement are not measured
   anthropometric data the way the thigh/shank lengths are).
7. ~~Sagittal/lateral axes are decoupled... hip yaw fixed at zero...
   real 3D coupling effects... not modeled.~~ **Partially added**: the
   LEG'S OWN inertial dynamics now genuinely couple hip_yaw/hip_roll
   into hip_pitch/knee_pitch torques (real off-diagonal mass-matrix
   terms, verified nonzero once yawed/rolled -
   `dynamics/rigid_body_leg_6dof.py`). Still NOT added: the LIPM/footstep
   planner itself is still two independent 1D instances (this new module
   computes joint torques for a GIVEN trajectory; it doesn't yet feed
   back into how that trajectory or the footstep plan is generated), hip
   yaw is still commanded at zero by `kinematics/leg_ik.py` (no turning
   gaits), and there's still no whole-body angular-momentum-about-the-
   vertical-axis model tying the two legs together.
8. **Foot orientation is always commanded flat/level**
   (`IDENTITY_R` in `simulation/joint_trajectory_generator.py`) - no
   heel-strike/toe-off rolling contact, which real human-like gaits use.
9. ~~Flat, known, rigid ground.~~ **Partially added**:
   `terrain/terrain_profile.py` + `planning/footstep_planner.apply_terrain()`
   place footsteps at real, non-zero heights (steps, ramps). **Real,
   honest finding**: stepping UP a modest 10cm curb works fine (within
   the leg-length reach margin from finding #6 in `docs/BUGS_FOUND.md`),
   but stepping DOWN the same 10cm EXCEEDS that margin and IK fails -
   because the LIPM's CoM-height model still doesn't adapt to terrain
   (item 4, below) - this is a genuinely harder, unsolved combination,
   not silently patched over. Ground compliance/slip is still not
   modeled at all.

## Honest read on distance to a real bipedal robot

This now closes NINE real gaps that were completely absent from the
portfolio before this work started: balance/locomotion planning, leg
kinematics/swing trajectories, actuator torque feasibility AND (with
real, characterized limits) actuator electrical dynamics/friction,
disturbance rejection (single- and, with real caveats, multi-step),
terrain-aware footstep placement AND terrain-adaptive CoM height for leg
reach, and full 6-joint rigid-body dynamics for the leg's own inertia
INCLUDING the foot as a third rigid body, including real hip-yaw/roll
<-> hip-pitch/knee coupling that the earlier 2-link model structurally
could not represent. What's left is real and substantial, not hidden:
coupling the now-terrain-adaptive CoM height INTO the LIPM/ZMP-tracking
horizontal dynamics itself (still uses one fixed nominal zc for its own
gain design - the "nonlinear MPC on centroidal dynamics" research
area), a genuine JOINT multi-step push-recovery solve (the naive,
independently-repeated version implemented here measurably doesn't help
for large pushes - see docs/BUGS_FOUND.md), gearbox BACKLASH (a
hysteretic, position-dependent effect the electrical/friction model
doesn't cover), and - the biggest remaining one - the physical hardware
itself (motors, structure, power, real sensors). Each of those is a
substantial project in its own right, not a quick follow-on. This
module's honest job now is: prove the planning, kinematics, dynamics-
feasibility (both magnitude AND, now, electrical response), recovery,
terrain-placement/height, and full-body-inertia (now including the
foot) layers are each real, correct, and consistent with each other
(verified by full-pipeline integration tests AND, for the dynamics
layer, an exact-reduction check against the already-analytically-
verified 2-link model) - so there is a real, physically-grounded
joint-angle command stream with known, tested limits, ready for
whichever hardware or higher-fidelity layer comes next.


