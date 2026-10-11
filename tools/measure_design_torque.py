"""
Measures the REAL torque demand of the heavier design (model/design_v3.py) in MuJoCo, replacing the M/25 scaling guess used by hardware/design_search.py.

    python tools/measure_design_torque.py                       # 28 kg, CoM 0.85/0.89/0.92 m, 6-step walk each (~15-20 min)
    python tools/measure_design_torque.py --mass 30 --zc 0.89 --steps 6
Torque limits are lifted (300 Nm) so nothing is clipped: the numbers are DEMAND. Writes results/design_requirements.json in the SAME schema as
results/fullbody/motor_requirements.json, so:   python tools/design_search.py --consistent --unverified --req results/design_requirements.json
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform.startswith("linux"): os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

GROUPS = {"hip_pitch_Nm": "hip_pitch", "hip_roll_Nm": "hip_roll", "hip_yaw_Nm": "hip_yaw", "knee_Nm": "knee", "ankle_pitch_Nm": "ankle_pitch", "ankle_roll_Nm": "ankle_roll", "waist_Nm": "waist"}


class PeakRecorder:
    """Plugs into walk_wbc.run(recorder=...): called every control tick with MuJoCo data."""
    def __init__(self, names, dadr):
        self.names, self.dadr = names, dadr; self.tau = np.zeros(len(names)); self.qd = np.zeros(len(names))
    def maybe(self, d, t):
        self.tau = np.maximum(self.tau, np.abs(d.ctrl[:len(self.names)])); self.qd = np.maximum(self.qd, np.abs(d.qvel[self.dadr]))


def peaks_to_row(zc, names, tau, qd):
    row = {"com_height_m": zc}
    for key, frag in GROUPS.items():
        idx = [i for i, n in enumerate(names) if n.split("_", 1)[-1].startswith(frag) or n.startswith(frag)]
        row[key] = float(max(tau[i] for i in idx)) if idx else 0.0
    leg = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
    row["leg_speed_rad_s"] = float(max(qd[i] for i in leg)); return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mass", type=float, default=28.0); ap.add_argument("--zc", default="0.85,0.89,0.92"); ap.add_argument("--steps", type=int, default=6); a = ap.parse_args()
    from model.design_v3 import design_v3_params
    from model.humanoid_model import make_model, joint_dadr, ACTUATED
    from simulation.walk_wbc import run
    big = dict(tau_hip=300.0, tau_knee=300.0, tau_ankle=300.0, tau_waist=300.0)
    rows = []
    for zc in [float(x) for x in a.zc.split(",")]:
        p, _ = design_v3_params(a.mass, zc=min(max(zc, 0.85), 0.92), **big)
        m, _d = make_model(p, table=False); rec = PeakRecorder(list(ACTUATED), joint_dadr(m, ACTUATED)); t0 = time.time()
        r, S = run("repo", n_steps=a.steps, zc=zc, verbose=False, recorder=rec, params=p)
        row = peaks_to_row(zc, list(ACTUATED), rec.tau, rec.qd); row.update(fell=bool(r["fell"]), mass_kg=a.mass, wall_s=round(time.time() - t0))
        rows.append(row); print(json.dumps(row), flush=True)
    os.makedirs("results", exist_ok=True); json.dump(rows, open("results/design_requirements.json", "w"), indent=1)
    base = {r["com_height_m"]: r for r in json.load(open("results/fullbody/motor_requirements.json"))}
    print("\nknee demand: measured vs the M/25 scaling guess")
    for r in rows:
        b = base.get(round(r["com_height_m"], 2))
        if b: print(f"  CoM {r['com_height_m']:.2f} m: measured {r['knee_Nm']:.1f} Nm | guess {b['knee_Nm'] * a.mass / 25:.1f} Nm | 25 kg baseline {b['knee_Nm']:.1f} Nm")
    print("saved results/design_requirements.json  <- send this")


if __name__ == "__main__":
    main()
