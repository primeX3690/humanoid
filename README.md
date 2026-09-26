# Humanoid Locomotion v1 — LIPM + ZMP Preview Control

Part of the same zero-LLM, CPU-only, NumPy-first portfolio as
[NEXUS](https://github.com/primeX3690/nexus-v3), AscentGNC, and SafeEvo —
fills the one gap none of those touch: **legged balance/locomotion
planning**. This is Phase 2 of the roadmap toward a full humanoid: brain
(NEXUS/genesis_core) → single real body (drone, Phase 1) → **balance/
locomotion (this project)** → manipulation → full humanoid integration.

## What it is

A real implementation of the **Linear Inverted Pendulum Model (LIPM) +
Zero-Moment-Point (ZMP) preview control** gait generator — the actual
algorithm class used on real preview-control-era humanoids (Kajita et
al., ICRA 2003; the lineage that includes Kawada HRP-2). Given a
footstep plan, it computes a Center-of-Mass trajectory that keeps the
ZMP — the real physical stability criterion for a biped — inside the
support polygon formed by the feet.

## Quick start

```bash
pip install numpy scipy matplotlib
export PYTHONPATH=.
python3 -m pytest tests/ -v      # 53/53 should pass
python3 -m demo.run_walk_demo    # generates results/walk_demo.png
```

## What's actually verified (not just written)

- **LIPM discretization** matches the textbook closed-form continuous
  solution to numerical precision (`tests/test_lipm.py`)
- **Closed-loop stability**: the augmented preview-control system's
  eigenvalues are checked to lie strictly inside the unit circle — the
  formal stability guarantee, not an empirical "it didn't blow up"
- **Zero steady-state ZMP tracking error**, confirmed numerically (the
  integral-action guarantee of the augmented-state LQI formulation)
- **Preview genuinely reduces tracking lag** vs. near-zero preview —
  measured via RMS error around a reference step change, the actual
  point of the algorithm, not assumed
- **Full 8- and 16-step walks keep the ZMP inside the support polygon**
  with a real, measured, non-zero minimum margin (~8.6mm) — checked
  timestep-by-timestep against the real support polygon, not eyeballed
  from a plot
- **Real 6-DOF leg IK** (hip yaw/roll/pitch, knee pitch, ankle
  pitch/roll) solves every single timestep of a full 8-step walk,
  converging to **sub-millimeter** foot-tracking accuracy against the
  planned footstep targets — the actual joint-angle command stream a
  real robot's motors would receive, not just a CoM/ZMP plan on paper
- **Minimum-jerk swing-foot trajectories** with verified zero velocity
  at both liftoff and touchdown (a real, physically-correct soft-landing
  property — checked via finite differences, not assumed from the
  formula)
- **Real joint-torque feasibility check against real actuator
  datasheets** (`dynamics/leg_dynamics.py` + `actuators/motor_specs.py`):
  Jacobian-transpose quasi-static torque, checked against a Dynamixel
  MX-106 (hobby servo) and a Dynamixel PRO Plus H54P-200 (research-grade
  actuator). **Real finding**: the hobby servo is ~20-30x underpowered;
  the research-grade actuator covers the hip with margin but falls ~20%
  short at the knee — the knee, not the hip, is this gait's binding
  actuator constraint. A specific, actionable hardware-selection result,
  not a vague "motors might not be strong enough."
- **Full 2-link rigid-body dynamics for the swing leg**
  (`dynamics/rigid_body_leg.py`): real mass matrix, Coriolis/centrifugal
  terms (Christoffel-symbol formula), and gravity terms — verified
  against the EXACT analytic parallel-axis-theorem and physical-pendulum
  results (not just "it runs"), replacing an earlier crude lumped-mass
  proxy.
- **Capture-point push recovery** (`control/capture_point.py` +
  `simulation/push_recovery_simulator.py`) — the real theoretical basis
  used on robots like Atlas. **Real finding**: recovery can't undo the
  instantaneous ZMP transient at the moment of a push, but measurably
  reduces cascading violations afterward — a real, modest benefit, not
  a magic fix.
- **Terrain-aware footstep placement** (`terrain/terrain_profile.py`) —
  **real finding**: stepping UP a 10cm curb works fine within the
  existing leg-reach margin; stepping DOWN the same 10cm exceeds it,
  because the CoM-height model doesn't adapt to terrain — a genuine,
  documented, not-yet-solved limitation.

**Nine real bugs/findings** — all documented with the failing test that
caught each one in [`docs/BUGS_FOUND.md`](docs/BUGS_FOUND.md):
1. swing-foot vs. stance-foot ZMP reference confusion (2.9m support-polygon violation)
2. instantaneous vs. ramped double-support ZMP transitions
3. a floating-point threshold bug that froze most footsteps' landing position, undetected until a full-walk integration test caught a 732mm tracking error
4. a swing-height profile with nonzero touchdown velocity (real impact, not a soft landing)
5. cranking the ZMP tracking gain up by 4 orders of magnitude made stability *worse*, not better, due to the LIPM's non-minimum-phase ZMP response
6. a leg-length-vs-CoM-height margin constraint that isn't obvious until IK starts failing mid-swing
7. the actuator-feasibility finding (knee is the binding torque constraint, not the hip)
8. push recovery's real (not undo-everything) benefit — reduces cascading violations, doesn't erase the instantaneous transient
9. terrain stepping-down exceeds the leg-reach margin that stepping-up doesn't

## Honest scope

This is the **top planning layer only** — it does not yet include leg
inverse kinematics, swing-foot trajectories, disturbance rejection, or
full-body dynamics. See [`docs/SCOPE.md`](docs/SCOPE.md) for the full,
honest breakdown of what's modeled vs. what real hardware would still
need on top of this.

## Repository layout

```
dynamics/lipm.py                           LIPM discretization
dynamics/leg_dynamics.py                   Jacobian-transpose joint torque estimation
dynamics/rigid_body_leg.py                 full 2-link rigid-body dynamics (mass matrix, Coriolis, gravity)
control/zmp_preview_controller.py          Kajita 2003 preview control
control/capture_point.py                   Instantaneous Capture Point push recovery
planning/footstep_planner.py               footstep sequence + ZMP/foot-target reference generator + terrain
kinematics/leg_fk.py, leg_ik.py            6-DOF leg forward/inverse kinematics
trajectory/swing_foot.py                   minimum-jerk swing-foot trajectory
terrain/terrain_profile.py                 flat/step/ramp terrain height profiles
actuators/motor_specs.py                   real actuator datasheets + feasibility check
simulation/walk_simulator.py               CoM/ZMP walk sim, checks real stability
simulation/joint_trajectory_generator.py   ties it all together into real joint angles
simulation/torque_analysis.py              full-walk torque vs. real-motor feasibility
simulation/push_recovery_simulator.py      push disturbance + capture-point recovery simulation
tests/                                      53 tests, all passing
demo/run_walk_demo.py                       generates results/walk_demo.png + torque report
docs/BUGS_FOUND.md                          real bugs/findings, honestly logged
docs/SCOPE.md                               what this is and isn't (yet)
```
