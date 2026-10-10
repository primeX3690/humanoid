"""Actuator trade study from YOUR measured gait demand (results/fullbody/motor_requirements.json).

NOTE: "at-speed" margin is a PESSIMISTIC bound - it assumes the gait's peak torque and peak joint speed occur together
(they come from different joints/instants), so an actuator can fail it and still walk (PH54-200 at 0.89 m does).
For each candidate actuator and CoM height: torque margin (continuous AND at the gait's peak joint speed on the
torque-speed envelope), speed margin, and total actuator mass for the 14 leg+waist joints.
  python tools/actuator_tradeoff.py [--json results/actuator_tradeoff.json]
"""
import json, argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace
from actuators.joint_actuator import PH54_200, QDD_REFERENCE, ActuatorSpec, RPM, envelope_torque

CANDIDATES = [PH54_200, replace(PH54_200, name="PH54-200 if 73 Nm peak (distributor figure, UNVERIFIED)", tau_peak_nm=73.0),
              QDD_REFERENCE,
              ActuatorSpec("HYPOTHETICAL 70 Nm / 5 rad/s geared", 70.0, 100.0, 5.0, 1.8, hypothetical=True)]
JOINT_KEYS = {"knee_Nm": 2, "hip_pitch_Nm": 2, "hip_roll_Nm": 2, "hip_yaw_Nm": 2, "ankle_pitch_Nm": 2, "ankle_roll_Nm": 2, "waist_Nm": 1}


def evaluate(req_rows, cands=CANDIDATES, n_act=14):
    out = []
    for c in cands:
        rows = []
        for r in req_rows:
            w = r["leg_speed_rad_s"]
            avail_pk = envelope_torque(c, w)
            worst = max(r[k] for k in JOINT_KEYS)
            rows.append(dict(com_height_m=r["com_height_m"], worst_joint_nm=worst, peak_speed_rad_s=w,
                             torque_margin_cont=c.tau_cont_nm / worst - 1.0,
                             torque_margin_at_speed=avail_pk / worst - 1.0,
                             speed_margin=c.w_noload_rad_s / w - 1.0,
                             feasible=bool(c.tau_cont_nm >= worst and avail_pk >= worst and c.w_noload_rad_s > w)))
        out.append(dict(actuator=c.name, hypothetical=c.hypothetical, mass_14_actuators_kg=n_act * c.mass_kg, rows=rows))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--req", default="results/fullbody/motor_requirements.json")
    ap.add_argument("--json", default=None); a = ap.parse_args()
    res = evaluate(json.load(open(a.req)))
    for e in res:
        print(f"\n{e['actuator']}{'  [hypothetical]' if e['hypothetical'] else ''}   14 actuators = {e['mass_14_actuators_kg']:.1f} kg")
        for r in e["rows"]:
            print(f"  zc={r['com_height_m']:.2f}  need {r['worst_joint_nm']:5.1f} Nm @ {r['peak_speed_rad_s']:.2f} rad/s | "
                  f"cont margin {100*r['torque_margin_cont']:+6.0f}%  at-speed {100*r['torque_margin_at_speed']:+6.0f}%  "
                  f"speed {100*r['speed_margin']:+5.0f}%  -> {'OK' if r['feasible'] else 'NO'}")
    if a.json: json.dump(res, open(a.json, "w"), indent=1)
