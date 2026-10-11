# Design decision (final, v3.6): robot mass, CoM height, payload, DoF, knee spec

Inputs: your gait demand (results/fullbody/motor_requirements.json, measured at 25 kg), arm static torques from the repo's arm model, and the allowed mass range
**25-40 kg**. Reproduce everything: `python tools/design_matrix.py` (-> results/design_matrix.json), `python tools/design_search.py --consistent --unverified`.
**v3.7: the demand is now MEASURED in MuJoCo at 32 kg** (results/design_requirements_measured32.json, walking, torque limits lifted). The earlier M/25 scaling guess was within 1-3 % for knee/hip-pitch/ankle/waist
and within 8 % for hip-roll: knee 67.4 / 54.8 / 46.7 Nm at CoM 0.85 / 0.89 / 0.92 m (guess 66.8 / 54.3 / 46.0). The decision below did not change. NOT measured: pushes, stairs, turning - they add torque on top
of walking, which is why the knee ask keeps a margin (>= 65 Nm vs 57 Nm walking-only).

## DECIDED (computed, not guessed)
| item | decision | why |
|---|---|---|
| **Robot mass** | **32 kg** | 25 kg does not close. The mass/torque loop closes from 30 kg (20 % structure) or 32 kg (25 % structure). 32 kg leaves ~9.4 kg (29 %) for structure after 19.2 kg actuators, 1.5 kg battery (30 min), 1.5 kg electronics, 0.4 kg grippers |
| **CoM height** | **0.89 m** | knee demand 42 Nm at 25 kg (36 Nm at 0.92 m but leg speed 3.7 rad/s exceeds every PH54; 0.92 m only closes at 40 kg). 0.85 m needs 52 Nm (+23 %) |
| **Arm payload** | **1 kg per hand (2 kg bimanual)** | the arm's OWN weight outstretched needs ~13-15 Nm at the shoulder; payload 0-1 kg changes nothing. Achieved with PH54-100 shoulders/elbows: ~1.2-1.3 kg per arm at continuous torque. 2 kg per hand needs a stronger wrist class (+0.5 kg) |
| **DoF** | **keep all 30 (+ dexterous hands later)** | dropping wrists saves only <1.5 kg, dropping wrists+neck <2 kg: lowers the closing mass from 32 to 30 kg - not worth losing manipulation |
| **Knee** | **needs a stronger actuator than PH54-200** | spec below |

## Actuator requirements at 32 kg, CoM 0.89 m, 1 kg payload (margin 1.04 on torque, speed x1.1, legs duty 50 %)
| group (count) | peak torque >= | continuous >= | speed >= (rad/s) | chosen in search |
|---|---|---|---|---|
| **knee (2)** | **57 Nm** (ask for **>= 65** to cover the M/25 guess) | **27 Nm** (ask >= 30) | 2.6 | QDD-class (hypothetical) - PH54-200's 44.7 Nm is NOT enough |
| hip (6) | 35 Nm | 17 Nm | 2.6 | PH54-200 |
| ankle (4) | 28 Nm | 13 Nm | 2.6 | PH54-200 (PH54-100 would be marginal) |
| waist (2) | 6.2 Nm | 3.0 Nm | 0.6 | XM540-class (UNVERIFIED numbers) |
| shoulder (6) | 16 Nm hold | 16 Nm | - | PH54-100 |
| elbow (2) | 7 Nm hold | 7 Nm | - | PH54-100 (a lighter 10 Nm-class would do) |
| wrist (6) | 1.2 Nm | 1.2 Nm | - | XM430-class (UNVERIFIED) |
| neck (2) | 1 Nm | 1 Nm | - | XM430-class (UNVERIFIED) |
Robustness: closes with up to +15 % demand over walking-only (knee peak 65 Nm); at +30 % it does not (needs the heavier-duty catalogue entry / a larger robot). Measured peak torques also show hip-roll 35 Nm and a waist demand that jumps to 15 Nm at CoM 0.92 m (another reason to stay at 0.89 m).

## TO VERIFY by you (I have no internet and will not present memory as datasheet fact)
1. **Knee actuator**: find a joint module with >= 65 Nm peak, >= 30 Nm continuous, >= 2.9 rad/s no-load-equivalent, mass <= ~2.2 kg (the search assumed 1.4 kg; heavier eats structure budget:
   every +0.8 kg per knee costs 1.6 kg). Starting points to CHECK (I did not verify any model's numbers): quasi-direct-drive / planetary joint modules in the 60-120 Nm-peak class
   from vendors such as CubeMars (AK series), MyActuator (RMD-X series), T-Motor, Unitree. If nothing fits, a PH54-200 pair per knee in a linkage doubles torque but adds 1.7 kg per knee.
2. **XM540 / XM430 / PH54-100**: stall + continuous torque, no-load speed, MASS from the Robotis e-manual (my catalogue numbers are from memory). PH54-100 mass was assumed = PH54-200 (0.855 kg).
   Edit the entries in `hardware/design_search.py` (verified=True) and rerun `tools/design_matrix.py`.
3. **Battery**: pack Wh/kg of the pack you will buy INCLUDING BMS/housing (search uses 180 Wh/kg; typical Li-ion packs 150-200). 30 min at the ~300-400 W walking average = ~170-210 Wh.
4. **Structure fraction**: after the first CAD pass compute (links + brackets + covers) / total. The 25 % used here is a planning value, not a measurement.

## What would change this decision
- measured demand (measure_design_torque) > 15 % above the M/25 guess -> raise mass toward 35 kg or pick a stronger knee/hip
- a knee actuator heavier than 2.2 kg -> mass 34-35 kg
- payload requirement > 2 kg per hand -> stronger wrist class, re-run matrix
