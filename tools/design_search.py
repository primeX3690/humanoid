"""python tools/design_search.py [--zc 0.89] [--payload 1.0] [--unverified]  -> prints the lightest feasible actuator assignment + mass budget."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware.design_search import search, to_model_kwargs, GROUPS, CATALOG
from hardware.mass_budget import mass_budget
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--zc", type=float, default=0.89); ap.add_argument("--payload", type=float, default=1.0)
    ap.add_argument("--unverified", action="store_true"); ap.add_argument("--consistent", action="store_true", help="solve mass<->torque self-consistently"); ap.add_argument("--margin", type=float, default=1.04); ap.add_argument("--model-mass", type=float, default=25.0); a = ap.parse_args()
    req = json.load(open("results/fullbody/motor_requirements.json"))
    if a.consistent:
        from hardware.design_search import self_consistent_mass
        best, trace = self_consistent_mass(req, a.zc, a.payload, a.unverified, a.margin)
        if best is None: print("NO self-consistent design with this catalogue (structure/battery never fit). Add stronger/lighter actuators or lower the structure fraction."); sys.exit(1)
        print(f"self-consistent robot mass {best['robot_mass_kg']:.1f} kg | actuators {best['actuator_mass_kg']:.1f} kg | structure {best['structure_kg']:.1f} kg | choice {best['choice']}")
        k = best['robot_mass_kg'] / 25.0
        print(f"knee spec at this mass: peak >= {1.04*k*[r for r in req if abs(r['com_height_m']-a.zc)<1e-6][0]['knee_Nm']:.0f} Nm, speed >= {1.1*[r for r in req if abs(r['com_height_m']-a.zc)<1e-6][0]['leg_speed_rad_s']:.1f} rad/s")
        sys.exit(0)
    d = search(req, a.zc, a.payload, a.unverified, a.margin)
    print(json.dumps({k: v for k, v in d.items() if k != "demands"}, indent=1) if d["feasible"] else d["reason"])
    if d["feasible"]:
        b = mass_budget(a.model_mass, spec_fn=None) if False else None
        left = a.model_mass - d["actuator_mass_kg"] - 1.5 - 0.4 - 1.5
        print(f"left for structure after battery(30 min)/electronics/grippers: {left:.1f} kg = {100*left/a.model_mass:.0f} % of the robot  ->", "OK" if left >= 0.2 * a.model_mass else "TOO LITTLE")
        print("ModelParams kwargs:", to_model_kwargs(d))
