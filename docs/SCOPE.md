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
4. **Constant CoM height.** The LIPM's core simplifying assumption. Real
   walking (especially over rough terrain, or fast/dynamic gaits) needs
   variable CoM height, which breaks the linear model this preview
   controller relies on (that's the whole "nonlinear MPC on centroidal
   dynamics" research area many current humanoid controllers use
   instead/in addition - a legitimate, much harder next step, not
   pretended-away here).
5. ~~No disturbance rejection / push recovery.~~ **Added**:
   `control/capture_point.py` + `simulation/push_recovery_simulator.py`
   implement real Instantaneous Capture Point recovery (Pratt et al.
   2006 - the same theoretical basis used on real robots like Atlas).
   **Real, honest finding**: recovery cannot undo the INSTANTANEOUS ZMP
   transient at the moment of a push (the support polygon hasn't moved
   yet), but DOES measurably reduce cascading violations in the steps
   AFTER the recovered footstep, compared to ignoring the push - a real,
   modest, not-magical benefit. See `docs/BUGS_FOUND.md`.
6. ~~No full-body dynamics, no joint torques, no actuator limits.~~
   **Answered** (quasi-static stance-leg torque via Jacobian-transpose,
   real actuator-datasheet comparison) AND **the swing leg's inertial
   torque now uses full 2-link rigid-body dynamics**
   (`dynamics/rigid_body_leg.py`: real mass matrix, Coriolis/centrifugal
   terms via the Christoffel-symbol formula, gravity terms - verified
   against the exact analytic parallel-axis-theorem and physical-
   pendulum results, not just "it runs"), not the earlier crude
   lumped-point-mass proxy. Still NOT full 3D/6-DOF rigid-body dynamics
   (only the 2 dominant sagittal joints, hip_pitch/knee_pitch) and still
   no actuator dynamics (motor inductance, gearbox backlash/friction).
7. **Sagittal/lateral axes are decoupled** (two independent 1D LIPM
   instances) and **hip yaw is fixed at zero** (straight-line walking
   only - see `kinematics/leg_ik.py`). Real 3D coupling effects (e.g.
   angular momentum about the vertical axis) and turning gaits are not
   modeled.
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

This now closes FIVE real gaps that were completely absent from the
portfolio before this work started: balance/locomotion planning, leg
kinematics/swing trajectories, actuator torque feasibility, disturbance
rejection, and terrain-aware footstep placement. What's left is
real and substantial, not hidden: full 3D/6-DOF rigid-body dynamics
(only the 2 dominant sagittal joints have real dynamics; hip yaw/roll
and ankle roll/pitch still don't), a terrain-adaptive CoM-height model
(the LIPM still assumes flat ground under it, which is EXACTLY why
stepping down a curb currently fails - see docs/BUGS_FOUND.md),
multi-step (not just single-step) push-recovery replanning, actuator
dynamics (motor inductance, gearbox friction/backlash), and - the
biggest remaining one - the physical hardware itself (motors,
structure, power, real sensors). Each of those is a substantial project
in its own right, not a quick follow-on. This module's honest job now
is: prove the planning, kinematics, dynamics-feasibility, recovery, and
terrain-placement layers are each real, correct, and consistent with
each other (verified by full-pipeline integration tests, not just each
layer in isolation) - so there is a real, physically-grounded joint-angle
command stream with known, tested limits, ready for whichever hardware
or higher-fidelity dynamics layer comes next.

