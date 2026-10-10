# v3 changes (Phase 0-5)

**Verification status, read this first.** v3 was written in a sandbox WITHOUT MuJoCo, OSQP, CasADi, torch or internet.
All pure NumPy/SciPy modules below have unit tests that PASS there (`python tools/minipytest.py tests`: 240 tests pass, 0 fail, 10 skipped for missing MuJoCo/torch/CasADi, 1 documented xfail).
Code that touches MuJoCo (marked **[sim-glue, untested]**) was only syntax-checked. Run `make test` on your machine and
`python -m simulation.push_recovery_exp --planned`, `python -m simulation.walk_wbc --turn 0.05 --terrain step --qp fast`,
then send me any failures. The v2 numbers in docs/FULL_BODY.md were NOT re-run by me.

## Phase 0 - hygiene
- real `__init__.py` files (junk `__init__ (n).py` removed); newer BUGS_FOUND.md kept, duplicate removed
- FULL_BODY.md push-recovery contradiction resolved from `results/fullbody/push_recovery_stepping.json` (lateral stepping works to ~90 N, cross-over)
- SCOPE 12d corrected (nonlinear-MPC divergence fixed; lateral oscillation still xfail)
- `pytest.ini`, markers (mujoco/osqp/casadi/torch/slow), auto-skip, `requirements.txt`, `Makefile`, `tools/check_env.py`, `tools/minipytest.py` (offline runner)
- bug fixed: `apply_terrain` dropped `heading` (silently cancelled turns)

## Phase 1 - sim-to-real realism
- `actuators/joint_actuator.py`: command latency, current-loop lag, torque-speed envelope (corner speed), thermal fold-back, friction, backlash-offset encoder, quantisation/noise; presets PH54-200/PH54-100/MX-106 + HYPOTHETICAL QDD
- `sim_realism/`: fractional delay line with jitter, `RealismLayer`, domain randomization (masses, COM, friction, damping, motor strength, latency, noise, pushes)
- `WBCSim(realism=...)` hook **[sim-glue, untested]**; `tools/actuator_tradeoff.py`
- Finding: PH54-200 knee needs 52/47/42/36 Nm at CoM 0.85/0.87/0.89/0.92 m vs 44.7 Nm continuous

## Phase 2 - locomotion
- `control/fast_qp.py`: dependency-free warm-started dense ADMM QP (matches SLSQP; ~3.5 ms warm on a 90x170 problem); WBC `qp_backend="fast"` **[sim-glue, untested]**
- `planning/collision_aware_swing.py`: ankle/foot collision-free swing (front/back pass, lift over foot)
- `control/lateral_recovery.py`: DCM step-time + landing + cross-over + CoP-shift optimiser; forward-simulation agrees with the planner on 40 random pushes
- `control/push_recovery_v3.py` **[sim-glue, untested]**; `planning/wbc_walk_reference.py` (turning + terrain), `terrain/terrain_xml.py`, `walk_wbc --turn/--terrain/--qp` **[sim-glue, untested]**

## Phase 3 - estimation / perception
- `estimation/pose_aiding.py`: absolute position/yaw fixes with gating (yaw drift 0.14 rad/2 min -> 0.007), foot F/T `ContactDetector`
- `perception/elevation_map.py` (Kalman height map, slope/step/roughness, traversability, TerrainProfile adapter), `perception/nav_planner.py` (A*)

## Phase 4 - manipulation
- `manipulation/`: arm collision model, RRT-Connect planner, antipodal grasp planner, admittance + force regulator, payload/loco-manipulation (PH54-100 limits: arm outstretched holds ~2.6 kg), bimanual box IK
- `tasks/mission_manager.py`: failure effects + re-planning (extended ops: walk, carry, release)

## Phase 5 - hardware handoff
- `hardware/hal.py` (one interface for sim/real, control loop), `hardware/safety.py` (limits, watchdog, thermal, tilt/DCM fall detection, latched e-stop)
- `hardware/dynamixel_protocol2.py` (packets/CRC/stuffing; register addresses deliberately NOT guessed)
- `model/urdf_export.py` + `export/humanoid.urdf` (30 revolute + 2 prismatic, 25.0 kg)
- `hardware/mass_budget.py`, `tools/hardware_report.py` -> `docs/HARDWARE_REPORT.md`

## Still open (be honest with yourself before building hardware)
1. 30 datasheet actuators = 20 kg of a 25 kg robot: not buildable as specified (see HARDWARE_REPORT.md). Decide: lighter arms/neck, higher mass target, or fewer DoF.
2. Thermal/backlash/latency numbers are ASSUMED; calibrate on one real joint first (docs/HARDWARE_HANDOFF.md).
3. Lateral/MPC xfail, hands are 1-DoF grippers, no stairs gait in WBC, no learned policy on the full body.
