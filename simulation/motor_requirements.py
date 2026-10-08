"""
Actuator requirement study for the 25 kg humanoid walking your LIPM/ZMP gait: run the full-body WBC with UNCONSTRAINED-strength
joints (130 Nm) at several CoM heights and record the peak torque / speed each joint actually needs.
  python -m simulation.motor_requirements   ->  results/fullbody/motor_requirements.json (+ markdown table)
"""
import json, sys
import numpy as np
from simulation.walk_wbc import run

if __name__ == "__main__":
    rows = []
    for zc in (0.85, 0.87, 0.89, 0.92):
        r, S = run("strong", 8, zc=zc, verbose=False)
        tq, sp = r["peak_torque_Nm"], r["peak_speed_rad_s"]
        row = dict(com_height_m=zc, fell=r["fell"], knee_Nm=max(tq["L_knee"], tq["R_knee"]), hip_pitch_Nm=max(tq["L_hip_pitch"], tq["R_hip_pitch"]),
                   hip_roll_Nm=max(tq["L_hip_roll"], tq["R_hip_roll"]), hip_yaw_Nm=max(tq["L_hip_yaw"], tq["R_hip_yaw"]),
                   ankle_pitch_Nm=max(tq["L_ankle_pitch"], tq["R_ankle_pitch"]), ankle_roll_Nm=max(tq["L_ankle_roll"], tq["R_ankle_roll"]),
                   waist_Nm=max(tq["waist_yaw"], tq["waist_pitch"]), leg_speed_rad_s=r["peak_leg_speed_rad_s"])
        rows.append(row); print(row, flush=True)
        json.dump(rows, open("results/fullbody/motor_requirements.json", "w"), indent=1)
