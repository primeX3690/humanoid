# Full-body layer (v2.0) — arms, torso, head, whole-body control, walking-with-WBC, perception, estimation, planning

Everything here is **additive**: no existing file of the leg-only project was changed except `requirements.txt`
(2 lines appended) and `README.md` (a section appended). Run from the repo root with `export PYTHONPATH=.`

## What was added (and which gap it closes)

| Gap | Module |
|---|---|
| Arms + hands | `model/humanoid_model.py` (32-dof MuJoCo humanoid: 12 leg, 2 waist, 2 neck, 2x7 arm, 2 grippers), `kinematics/arm_kinematics.py` |
| Torso, head/neck | waist yaw/pitch, neck yaw/pitch in the model; `kinematics/head_neck_kinematics.py` |
| Whole-body control | `control/whole_body_qp.py` — strict-priority hierarchical QP (contact dynamics, friction cones, torque limits; balance > swing foot/hands > gaze > posture) |
| Push recovery by stepping | `control/push_recovery.py` (capture-point step trigger, swing-foot planner, contact switching), `simulation/push_recovery_exp.py` |
| Walking on the full body | `simulation/walk_wbc.py` — your LIPM/ZMP-preview plan + footsteps + swing trajectories drive the WBC, contact set switches single/double support |
| Push recovery by stepping | `control/push_recovery.py` — capture-point stepper (STAND/STEP state machine) on top of the WBC |
| Dynamic / athletic motion | `planning/centroidal_jump_to.py` (CasADi/IPOPT) incl. **joint-speed limits** of the chosen actuator |
| Perception | `perception/object_detector.py`, `render.py` (head RGB-D camera), `perception/ekf_slam.py` (2-D EKF-SLAM + occupancy grid) |
| State estimation | `estimation/base_eskf.py` contact-aided ESKF, `estimation/lipm_disturbance_kf.py`, `simulation/estimated_state.py` (feeds the controller instead of ground truth) |
| Task planning / behaviour | `tasks/symbolic_planner.py`, `tasks/behavior_tree.py`, `tasks/pick_skills.py`, `simulation/pick_demo.py` |
| LLM | **not included on purpose** — plug a local model in at `tasks/symbolic_planner.py::CommandParser` |

## Actuator limits are YOUR hardware choice
`model.humanoid_model.repo_actuator_params()`: legs + waist = Dynamixel PH54-200 (44.7 Nm continuous, 33.1 rpm = 3.47 rad/s no-load,
from `actuators/motor_specs.py`). Shoulders/elbows (PH54-100, 25.3 Nm) and wrists/neck (MX-106-class, 8.4 Nm) are **assumptions** — arms were
outside the original scope. `ModelParams()` (default) is a stronger 130 Nm-leg robot used as an "unconstrained actuators" reference.

## Results (all produced by the code in this repo; JSON in `results/fullbody/`)

### Walking: LIPM/ZMP plan -> whole-body QP -> MuJoCo (8 steps, 0.3 m, your default GaitParams)
Rows 1-2 were measured on an earlier 27.4 kg variant of the model, rows 3-4 on the final 25.0 kg model.
| Run | Fell | Distance (plan 2.16 m) | CoM tracking RMS | Landing error (mean / max) | Peak knee torque | Peak leg speed |
|---|---|---|---|---|---|---|
| 130 Nm actuators, CoM 0.85 m | no | 2.175 m | 0.3 mm | 0.24 / 0.27 mm | 58 Nm (45 % of 130) | 2.08 rad/s |
| **Repo actuators (44.7 Nm), CoM 0.85 m** | **yes, t = 3.5 s** | 0.49 m | – | – | knee saturated at 44.7 | 25 rad/s (during the fall) |
| **Repo actuators, CoM 0.89 m, 25 kg** | no | 2.175 m | 0.3 mm | 0.24 / 0.27 mm | 42.6 Nm = **95 % of the knee limit** | 2.40 rad/s (limit 3.47) |
| Repo actuators, CoM 0.89 m, 25 kg, **noisy IMU + encoders through the ESKF** | no | 2.185 m | 9.6 mm | 14 / 31 mm | knee **at the limit (100 %)** | 2.38 rad/s |

Reading: the knee of your chosen actuator is the bottleneck, exactly as `docs/HARDWARE_REQUIREMENTS.md` predicted. The real-actuator robot falls with the
legs bent for a 0.85 m CoM and only walks when the legs are straightened to 0.89 m, with 0-5 % torque margin. Joint *speed* is not a problem for walking.

### Push recovery (`python -m simulation.push_recovery_exp`, 150 ms push at torso height, 25 kg model, 130 Nm-class actuators)
| Direction | Standing only (ankle/hip strategy) | With capture-point stepping |
|---|---|---|
| forward | 60 N ok (peak 67 mm), 90 N falls | 90 N ok (3 steps), 120 N ok (2 steps), 150 N and 180 N fall |
| backward | 60 N ok (10 mm), 90 N falls | 90 / 120 / 150 N ok (2-3 steps), 180 N falls |
| lateral | 45 N ok (10 mm), 60 N falls | 60 N ok (1 step), 90 N ok (2 steps, cross-over), 120 N and 150 N fall |
Stepping roughly doubles the survivable push in the sagittal plane. Swing-phase detail: with a strict CoM-first hierarchy the swing leg froze (CoM target is
unreachable while the capture point is outside the stance foot); CoM, torso and swing foot are therefore solved together (weights 1:1:25) during the swing, with
dynamics/friction/CoP constraints still hard. Lateral recovery needs a CROSS-OVER step (the unloaded foot steps to the far side of the loaded one); leg-to-leg collision is NOT modelled, so on hardware that path needs a collision-aware swing trajectory. Beyond ~90-100 N the leg reach runs out.

### Actuator requirement measured in simulation (`python -m simulation.motor_requirements`)
Peak joint demand of your 8-step gait with non-saturating joints (worst of left/right leg, Nm; speed in rad/s):
| CoM height | Knee | Hip pitch | Hip roll | Ankle pitch | Waist | Peak speed |
|---|---|---|---|---|---|---|
| 0.85 m | 52.2 | 15.5 | 22.6 | 22.7 | 4.8 | 2.09 |
| 0.87 m | 47.4 | 15.3 | 25.5 | 21.8 | 4.7 | 2.10 |
| 0.89 m | 42.4 | 15.3 | 26.4 | 20.8 | 4.7 | 2.40 |
| 0.92 m | 36.0 | 32.1 | 23.7 | 23.1 | 12.5 | **3.70** |
The knee exceeds 44.7 Nm below ~0.88 m; straightening the legs lowers knee torque but at 0.92 m the speed (3.7 rad/s) exceeds the PH54-200 no-load speed (3.47). Usable window ~0.89-0.91 m with ~0-5 % margin. A knee actuator with ~60-70 Nm continuous and >= 4-5 rad/s removes the constraint.

### Videos (`python -m simulation.make_videos`, written to `results/videos/`)
walk (8 steps, repo actuators, estimated state), pick-up of the red block, push forward 120 N with stepping, push forward 90 N standing (falls), push lateral 90 N cross-over step.

### Dynamic motion (centroidal trajectory optimisation, 27.4 kg, 2 legs, PH54-200 limits)
* Torque alone would allow a jump with 0.4 s flight (0.23 m apex gain) — **but** that needs 10–17 rad/s joint speed vs 3.47 rad/s available.
* With the speed limit added, no jump with flight >= 0.2 s is feasible; the feasible 0.15 s cases have ~0 apex gain. **PH54-200 cannot do athletic motion**; it needs ~3-5x faster joints (QDD/low-ratio actuators).
* Joint-speed check covers the stance phases only; point-mass model; not tracked by the WBC (flight-phase tracking was not built).

### Push recovery - data of record: `results/fullbody/push_recovery_stepping.json` (130 Nm-class actuators, 25 kg model)
Earlier revisions of this file carried two contradicting push tables; the second one (claiming lateral stepping fails at every force) was stale.
The JSON below is what the code actually produced and is the only source this document now follows.
| Direction | Standing only (WBC ankle/hip strategy) | With capture-point stepping |
|---|---|---|
| Forward | 60 N ok, 90 N falls | 90 N ok (3 steps), 120 N ok (2 steps), 150 / 180 N fall |
| Backward | 60 N ok, 90 N falls | 90 / 120 N ok (2 steps), 150 N ok (3 steps, ~1.0 m excursion), 180 N falls |
| Lateral | 60 N and 90 N fall | 60 N ok (1 step), 90 N ok (2 steps, cross-over), 120 / 150 N fall |
Lateral stepping therefore works up to ~90 N, not beyond; leg-to-leg collision is not modelled in the swing (see `planning/collision_aware_swing.py`, added in v3).
Standing-only PD vs WBC (final model, forward, `wbc_push_sweep_final_model.json`): 50 N both survive (WBC peak 26 mm, PD 57 mm); 60 N WBC recovers, PD ends 620 mm off; 70/80 N both fall.
Stepping needed a fix worth knowing: with a strict-priority hierarchy a physically unreachable CoM task freezes the swing leg, so during the swing CoM, torso and swing foot are solved together (weights 1:1:25).

### Other verified results
* WBC reach while balancing; unreachable target is sacrificed, balance is not (tests/test_wbc.py).
* Pick-and-place from a spoken-style command (`python -m simulation.pick_demo`): detection (3 mm), reach, grasp, lift 10 cm — with ground-truth state and with the ESKF state in the loop. Needs 0.45 rad torso lean (arm alone is 5 cm short).
* ESKF (squat + waist twist + arm swing, noisy IMU/encoders): height 0.2 mm RMS vs 443 mm for IMU dead-reckoning; velocity 0.007 vs 0.71 m/s; tilt 0.6 deg. Initial attitude = static alignment (truth + 0.2 deg), accelerometer tilt aid deliberately weak.
* 2-D EKF-SLAM: drift 2.3-2.7 m -> 0.06-0.11 m over 5 seeds; map F1 0.14-0.23 -> 0.74-0.91.

## Things worth knowing / fixing in the existing repo
* `actuators/motor_specs.py` derives a PH54-200 "stall torque" of 223 Nm by dividing the 44.7 Nm *continuous* figure by 0.20. The official datasheet leaves the stall column blank
  (44.7 Nm continuous @24 V); one distributor lists 73 Nm. The `torque_ok_stall` margins therefore look far better than the hardware is known to be. This package only uses the 44.7 Nm figure.
* The accelerometer tilt aid in an ESKF is biased by CoM-sway accelerations while walking (3 deg tilt error -> fall). `simulation/estimated_state.py` uses a nearly disabled aid for gait (`grav_sig=8`).

## Honest limits
* Simulation only. Masses of arms, torso, head are my estimates, scaled so the total is 25.0 kg (your budget); geometry of legs/feet matches `LegParams`.
* Stepping recovery is LIPM-based on flat ground, no turning: forward/backward up to ~120-150 N, lateral up to ~90 N (cross-over step). Walking uses the open-loop LIPM plan (no terrain, no turning in the WBC runs).
* Not done on purpose: tracking a jump plan with the WBC — the chosen actuators' joint speed (3.47 rad/s) rules jumping out, so it would prove nothing about the hardware.
* The default model uses `ModelParams()` (130 Nm legs); `repo_actuator_params()` gives your 44.7 Nm actuators. Windows: EGL is now set only on Linux, no edits needed (not tested on Windows).
* Walking needs ~3 min of CPU per 10 s of simulated gait (1 core, ~15-25 ms per 5-level QP).
* Perception is classical colour segmentation on simulated frames; SLAM is 2-D with simulated landmark/lidar sensing.
* Yaw and global x,y are unobservable in the estimator and drift slowly.
