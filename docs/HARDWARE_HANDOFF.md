# Hardware hand-off checklist

## A. Before buying anything
1. Resolve the mass budget (docs/HARDWARE_REPORT.md): 20 kg of actuators in a 25 kg robot leaves ~1.6 kg for structure+battery.
2. Re-run `tools/actuator_tradeoff.py` with the actuator you will really use; knee demand at your chosen CoM height must have margin.

## B. Calibrate the simulator on ONE real joint (then update `ActuatorSpec`)
- torque step response -> `tau_electrical_s`; position sweep with load -> friction (`coulomb_nm`, `viscous_nm_s`)
- hold a torque ramp -> thermal `thermal_tau_s`, derate/shutdown temperatures
- reverse-torque test with an encoder on the output -> `backlash_rad`
- timestamp a command->motion round trip -> `cmd_delay_s` and jitter; feed them to `domain_randomization.RandomizationConfig`

## C. Bring-up order (each step behind `hardware/safety.py`, e-stop in hand)
1. single joint on a bench, torque mode, `PlantInterface` vs real data
2. one leg on a gantry (weight supported) -> gravity compensation -> joint PD
3. both legs, harness: standing with WBC at low gains
4. harness walking, then pushes
5. untethered only after fall-detection has fired correctly in tests

## D. Software still to write on the robot
- `DynamixelInterface.read/write` with YOUR control-table addresses (e-manual) and a serial transport
- real IMU and foot F/T drivers (the estimator/contact code already expects `RobotState`)
- 1 kHz joint loop in C/C++ or a real-time thread; Python stays for planning
