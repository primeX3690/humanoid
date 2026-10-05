# Hardware Requirements Summary

Generated from this project's own simulation and analysis - every
number below is either a direct, sourced datasheet value or simple,
stated arithmetic on one (see actuators/hardware_requirements.py).
This is a software-side requirements derivation, not a mechanical or
electrical engineering design - see "What This Does NOT Cover" below.

## Actuation

| Item | Value |
|---|---|
| Actuator model | Dynamixel PRO Plus H54P-200-S500-R (24V) |
| Actuators needed (legs only) | 12 (6 DOF x 2 legs) |
| Unit mass | 0.855 kg |
| **Total actuator mass** | **10.26 kg** |
| Peak required torque (stance leg) | 32.2 N*m |
| Actuator continuous torque rating | 44.7 N*m |
| Torque margin | 1.39x |
| Per-joint backlash (assumed, see gearbox_backlash.py) | 0.3 deg |
| Electrical time constant (assumed, see actuator_dynamics.py) | 11.5 ms |

## Mass budget - a real, honest tension

| Item | Value |
|---|---|
| Assumed total robot mass (dynamics/leg_dynamics.RobotMassParams) | 25.0 kg |
| Total actuator mass | 10.26 kg |
| **Actuator mass fraction of total** | **41.0%** |

**This is worth flagging plainly, not glossing over**: the actuators
alone consume over 40% of the total-mass figure this project's entire
dynamics stack (LIPM, rigid-body leg dynamics, torque feasibility) has
assumed throughout. That leaves under 60% of the mass budget for
structure, battery, electronics, feet, and everything else a real robot
needs - a real constraint any physical build would have to confront,
not previously surfaced because no earlier module needed to add up
actuator mass against the total.

## Power

| Item | Value |
|---|---|
| Worst-case simultaneous power draw (all 12 actuators at rated continuous current) | 2678 W |

This is a conservative UPPER BOUND (all actuators drawing rated current
simultaneously is unlikely in normal gait, where only a subset are
near peak load at any instant - see dynamics/leg_dynamics.py's stance/
swing torque split), not a duty-cycle-adjusted estimate; a real battery
sizing exercise would need a proper duty-cycle analysis this project
does not attempt.

## What this DOES NOT cover

- Structural design: link geometry, material selection, stress/fatigue analysis for the loads dynamics/rigid_body_leg_6dof.py and dynamics/foot_inertia.py compute - this project models the LOADS, not the structure that must carry them.
- Sensing: an IMU for real orientation/balance feedback (this project's ZMP and CoM state are computed from an idealized model, not measured sensor data), joint position feedback beyond what the actuators' own internal encoders provide, and foot force/torque sensors for a REAL ZMP measurement (as opposed to the LIPM-model-computed one this project uses throughout).
- Power system: battery sizing (capacity, chemistry, discharge rate) against the worst-case power draw derived below, voltage regulation, wiring gauge and connector selection for the real currents involved.
- Thermal management: sustained near-continuous-torque operation (docs/BUGS_FOUND.md's stance-leg finding) generates real heat in the actuator windings - no thermal analysis or cooling provision exists in this project.
- Foot/sole design: compliance, sensing, and ground-contact mechanics - dynamics/foot_inertia.py models the foot's INERTIA, not its contact surface or structure.
- Arms, torso, head, and any other body segment beyond the two legs - entirely outside this project's scope; N_ACTUATORS above counts LEG joints only.
