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

---
# v3.1 (second pass: the six open items)
Same verification rule: pure NumPy modules are tested here (272 tests pass, 0 fail, 10 skipped for missing MuJoCo/torch/CasADi, 1 documented xfail); MuJoCo-side code is checked by `tools/static_check.py` (0 findings
project-wide), by mock-robot tests (`tests/test_glue_mock.py`), and must be confirmed with ONE command on your machine: `python -m simulation.sim_smoke`.

- **Glue verification:** `tools/static_check.py` (undefined names + wrong imports, proven on a deliberately broken file), `tests/test_glue_mock.py`
  (PlannedPushRecoveryStepper end-to-end on a mock robot - found and fixed a real gap: uncapturable pushes fell back to the old straight-line swing),
  `simulation/sim_smoke.py` (model builds, stand, realism, QP backends, v2-vs-planned push table, straight/turn/step/stairs walks, RL self-check, hand).
- **C - nonlinear MPC fixed:** `control/preview_nmpc.py`. Root cause of the old +-1 m lateral oscillation: 0.2 s horizon with a CONSTANT ZMP reference
  (no preview). New: 1.6 s preview, condensed QPs per axis with height-coupled coefficients, alternating height optimisation. Lateral error 0.037 m, ZMP in support 100 %.
- **D - mass blocker:** `hardware/design_search.py`, `tools/design_search.py`, `model/design_v3.py`, `docs/DESIGN_DECISION.md` (answer: ~28 kg robot + stronger knee).
- **B - stairs:** `terrain.StairTerrain`, `trajectory/stair_swing.py` (lift-first swing; legacy swing clips the nosing, new one clears it), `planning/stairs_reference.py`
  (step-to gait, NMPC CoM with height), `walk_wbc --terrain stairs`, stair XML.
- **B - dexterous hand:** `model/hand_model.py` (+ `build_xml(hand="dexterous")`, 11 DoF/hand added after the jaws, default model unchanged), `manipulation/hand/`
  (finger kinematics, force closure, epsilon quality, force distribution, power-grasp planner).
- **A - full-body RL:** `rl/fullbody_env.py` (30-DoF real model, actuator realism, domain randomization, pushes, biped reward-hacking guards, NumPy fake backend for tests),
  `rl/train_fullbody.py` (curriculum stand -> push -> walk -> robust). Run `python -m rl.fullbody_env --selfcheck` BEFORE training: PD gains are ASSUMED.

---
# MEASURED on real MuJoCo (sim_smoke --only push, user's machine) - the planned stepper does NOT yet beat v2
| case | v2 (re-run) | v2 (stored JSON) | planned |
|---|---|---|---|
| lateral 60 N | ok, 1 step, 126 mm | ok, 1 step, 127 mm | ok, 2 steps, **417 mm** |
| lateral 90 N | ok, 2 steps, 322 mm | ok, 2 steps, 339 mm | **FELL**, 3 steps, 836 mm |
| lateral 120 N | FELL | FELL | FELL (2 steps) |
| forward 90 N | FELL, 4 steps | ok, 3 steps (borderline: 295 mm) | FELL, 1596 mm |
| backward 120 N | ok, 2 steps, 447 mm | ok, 2 steps, 444 mm | **FELL**, 3 steps |
v2 reproduces its stored results in 4/5 cases (forward 90 N is a borderline case that flipped); the v3 edits to the v2 path are behaviour-neutral (diffed).
The planner's "40 random pushes agree" test only proved consistency with its OWN LIPM model, not with the WBC robot. Hypotheses (UNTESTED): long swing times
(planner allows up to 0.7 s; v2 uses 0.25 s), CoP-shift assumed by the plan but not realised by the controller (the STEP-state CoM target is still the v2 midpoint rule),
cross-over geometry, swing-foot speed. `python tools/diag_push.py` runs the ablation (planned / no CoP shift / T<=0.3 s / no cross-over) and logs intended vs actual per step.
Until that is read, treat `PlannedPushRecoveryStepper` as EXPERIMENTAL; keep using `PushRecoveryStepper` (v2).


---
# v3.3 - corrections after the first real run on Windows / Python 3.14 (pytest 286 pass, 6 fail; sim_smoke)
- **RL zero-action stand fell at 3.34 s** (`rl.fullbody_env --selfcheck`). Cause is physics, not a bug: standing on a planted foot needs ankle stiffness
  > m*g*h ~ 214 Nm/rad; my assumed gain was 120 (3-link model: min eigenvalue of Kp+Hess(V) = -137 -> topples). New gains: +221. kd is now implicit MuJoCo joint
  damping. `tests/test_stand_gains.py` guards this. NOT yet confirmed in MuJoCo - rerun `python -m rl.fullbody_env --selfcheck`.
- sim_smoke's `rl_selfcheck` PASSED falsely (it stopped at 3.0 s, before the fall). Now 5 s and judged exactly like the env termination.
- **Fast QP backend is ~4.5x SLOWER than OSQP on the real WBC** (walk_fast_qp 1044 s vs walk_straight 230 s). It was only benchmarked on random QPs. Default stays OSQP;
  do NOT use `--qp fast` until `tools/dump_wbc_qps.py` data lets me fix it. Skipped from `sim_smoke --quick`.
- test fixes: glue mock tests failed with real MuJoCo (strict `mj_name2id`) -> per-test stub; URDF mass tolerance 1e-3 (6 significant digits).
- `test_stairs` showed an OLD `terrain/terrain_xml.py`: an update overwrite was skipped while unzipping. New `tools/verify_install.py` + `MANIFEST.json` list stale files.
- `static_check` no longer scans `venv/`, reads files as UTF-8; sim_smoke writes one timestamped JSON per run (the earlier JSON was overwritten by the next run).


---
# v3.4 - automation of the remaining sim-side loops
Status from the user's real run: dexterous hand XML compiles in MuJoCo (sim_smoke `model` + `hand` PASS), straight WBC walk and walk with the actuator realism layer PASS.
Still open and DATA-DEPENDENT (cannot be fixed blind): planned-stepper push table, fast-QP slowness, turn/step/stairs walks, RL gains/training, torque demand of the 28 kg design.
New: `tools/collect_results.py` (runs everything, writes results/send_to_claude.zip), `tools/tune_rl_gains.py` (gain sweep), `tools/measure_design_torque.py`
(replaces the M/25 scaling guess with a MuJoCo measurement; `design_search.py --req`), `walk_wbc.run(params=...)`.
