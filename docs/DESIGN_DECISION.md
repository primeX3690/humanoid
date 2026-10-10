# Design decision: the 25 kg / 30-actuator robot does not close - what to build instead

Generated with `python tools/design_search.py --consistent [--unverified]` (hardware/design_search.py). Torque demand comes from YOUR measured gait
(results/fullbody/motor_requirements.json), arm demand from static holding torque (arm own weight + payload) with the repo's arm kinematics.

## Finding 1 - with datasheet-verified actuators (PH54-200, PH54-100, MX-106) NO consistent design exists
Mass and torque chase each other: a heavier robot needs a stronger knee, which weighs more. The 20 kg of actuators plus 20 % structure needs ~29 kg
of robot, whose knee then needs ~51 Nm (> PH54-200's 44.7 Nm). It never closes at CoM 0.85, 0.89 or 0.92 m.

## Finding 2 - it closes at ~28 kg with ONE change: a stronger knee actuator
| group (count) | choice | note |
|---|---|---|
| knee (2) | QDD-class, >= ~50 Nm peak, >= ~25 Nm cont., >= ~2.6 rad/s | **HYPOTHETICAL entry - this is a specification, not a part number** |
| hip (6) | PH54-200 | |
| ankle (4) | PH54-100 | demand 22-23 Nm |
| shoulder (6), elbow (2) | PH54-100 | the ARM'S OWN WEIGHT outstretched needs ~13-15 Nm at the shoulder even with zero payload |
| waist (2) | XM540-class (UNVERIFIED numbers) | demand ~5-12 Nm |
| wrist (6), neck (2) | XM430-class (UNVERIFIED numbers) | |
Result: robot ~28.2 kg = 19.2 kg actuators + 5.7 kg structure (20 %) + 1.5 kg battery (30 min) + 1.5 kg electronics + 0.4 kg grippers.
Payload 0-1 kg changes nothing; 2 kg pushes the wrist to a stronger class (29 kg).

## What to do
1. Set the target mass to **~28-30 kg**, CoM height **0.89 m** (knee 42 Nm at 25 kg; scales with mass).
2. Get a knee actuator meeting the spec above (or two PH54-200 in a parallel/linkage arrangement - costs mass; run the tool with your numbers).
3. In simulation: `from model.design_v3 import design_v3_params; p, d = design_v3_params(28.0)` and re-run the push/walk checks - torque demands were
   scaled by M/25 (first-order); the real values must be re-measured with the heavier model (`python -m simulation.sim_smoke`).
4. Replace every `verified=False` catalogue entry by the real datasheet numbers before trusting the assignment.
