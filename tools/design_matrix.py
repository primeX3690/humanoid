"""python tools/design_matrix.py  - design decision matrix over robot mass (25-40 kg), CoM height, payload, structure fraction and DoF variants."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hardware.design_search import evaluate_design, GROUPS

DOF = {"full 30 DoF": dict(GROUPS), "no wrists (24)": dict(GROUPS, wrist=0), "no wrists, no neck (22)": dict(GROUPS, wrist=0, neck=0)}


def main():
    req = json.load(open("results/fullbody/motor_requirements.json")); out = []
    print(f"{'M kg':>5} {'zc':>5} {'struct%':>7} {'payload':>7} {'DoF':<24} {'closes':<7} {'act kg':>7} {'struct kg':>9} {'knee Nm':>8}  knee actuator")
    for M in (25, 28, 30, 32, 35, 40):
        for zc in (0.85, 0.89, 0.92):
            for sf in (0.20, 0.25):
                for name, g in DOF.items():
                    e = evaluate_design(req, M, zc, 1.0, True, 1.04, sf, groups=g); e.update(zc=zc, struct_frac=sf, dof=name); out.append(e)
    json.dump(out, open("results/design_matrix.json", "w"), indent=1, default=str)
    for e in out:
        if e["zc"] == 0.89 and e["struct_frac"] == 0.25 and e["dof"].startswith("full"):
            print(f"{e['M']:>5} {e['zc']:>5} {100*e['struct_frac']:>6.0f}% {1.0:>7} {e['dof']:<24} " + (f"{'YES' if e['closes'] else 'no':<7}{e['actuator_kg']:>7.1f}{e['structure_kg']:>9.1f}{e['knee_peak_needed']:>8.0f}  {e['choice']['knee']}" if 'actuator_kg' in e else e['reason']))
    return out


if __name__ == "__main__":
    main()
