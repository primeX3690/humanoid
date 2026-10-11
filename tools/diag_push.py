"""
Push-recovery diagnostic / ablation: runs the SAME pushes with several stepper variants and prints one comparison table plus, for every
step, what the planner INTENDED (step time, target, DCM margin, cop) versus what HAPPENED (landing position, landing time).

    python tools/diag_push.py                      # default cases: lateral 90 N, backward 120 N  (~30 min)
    python tools/diag_push.py --cases lateral:60,lateral:90,forward:90,backward:120
    python tools/diag_push.py --variants v2,planned,planned_nocop
Writes results/diag_push.json. Send me that file (or the printed tables).
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataclasses import replace

DIRS = {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "lateral": (0.0, 1.0)}


def variants():
    from control.lateral_recovery import RecoveryParams
    from control.push_recovery import PushRecoveryStepper
    from control.push_recovery_v3 import PlannedPushRecoveryStepper
    P = RecoveryParams
    mk = lambda **kw: (lambda S: PlannedPushRecoveryStepper(S, params=P(**kw)))
    return {
        "v2": lambda S: PushRecoveryStepper(S),
        "planned": mk(),
        "planned_nocop": mk(use_cop_shift=False),                       # the controller does NOT shift the CoP the way the plan assumes
        "planned_T25": mk(T_min=0.20, T_max=0.30),                       # v2 uses a fixed 0.25 s swing; long swings may be the problem
        "planned_T25_nocop": mk(T_min=0.20, T_max=0.30, use_cop_shift=False),
        "planned_nocross": mk(allow_crossover=False),                    # is the cross-over step itself the problem?
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cases", default="lateral:90,backward:120"); ap.add_argument("--variants", default=",".join(variants()))
    ap.add_argument("--T", type=float, default=3.5); a = ap.parse_args()
    from simulation.push_recovery_exp import run
    V = variants(); rows = []
    for case in a.cases.split(","):
        name, F = case.split(":"); F = float(F)
        for vn in a.variants.split(","):
            t0 = time.time()
            try:
                r = run(F, DIRS[name], stepping=True, actuators="strong", T=a.T, stepper=V[vn])
                row = dict(case=f"{name} {F:.0f} N", variant=vn, fell=r["fell"], steps=r["steps"], peak_mm=round(r["peak_com_dev_mm"]), plan_log=r.get("plan_log", []), step_log=r["step_log"])
            except Exception as e:
                import traceback; row = dict(case=f"{name} {F:.0f} N", variant=vn, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=5))
            row["wall_s"] = round(time.time() - t0, 1); rows.append(row)
            print(f"{row['case']:<16}{vn:<20}" + (row.get("error") or f"{'FELL' if row['fell'] else 'ok  '} steps={row['steps']} peak={row['peak_mm']} mm"), flush=True)
            for p in row.get("plan_log", []):
                if p.get("fallback"): print("      plan: FALLBACK to v2 heuristic"); continue
                print(f"      plan t={p['t']:.2f} swing={p['swing']} T={p['T']:.2f} target=({p['target'][0]:+.2f},{p['target'][1]:+.2f}) margin={p['margin']:+.3f} cross={p['crossover']} feas={p['feasible']} xi=({p['xi'][0]:+.2f},{p['xi'][1]:+.2f}) cop=({p['cop'][0]:+.2f},{p['cop'][1]:+.2f})")
            for s in row.get("step_log", []):
                print(f"      done t={s['t']:.2f} side={s['side']} target=({s['target'][0]:+.2f},{s['target'][1]:+.2f}) landed=({s['landed'][0]:+.2f},{s['landed'][1]:+.2f})")
    os.makedirs("results", exist_ok=True); json.dump(rows, open("results/diag_push.json", "w"), indent=1, default=str)
    print("\nSUMMARY (fell=F / ok):")
    for case in dict.fromkeys(r["case"] for r in rows):
        print(f"  {case}: " + "  ".join(f"{r['variant']}={'F' if r.get('fell') else ('ERR' if 'error' in r else 'ok')}({r.get('steps','-')}st,{r.get('peak_mm','-')}mm)" for r in rows if r["case"] == case))


if __name__ == "__main__":
    main()
