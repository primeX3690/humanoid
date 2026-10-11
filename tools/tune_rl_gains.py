"""
Automates the RL PD-gain tuning loop (so it needs ONE run instead of many round trips): zero-action stand (5 s) and stand-under-pushes for several gain scalings.

    python tools/tune_rl_gains.py                         # ~5-10 min
    python tools/tune_rl_gains.py --scales 0.7,1,1.5,2 --ankle-scales 1,1.5,2
Sweep 1 multiplies ALL kp (kd by sqrt); sweep 2 multiplies only the ankle-pitch kp. Prints a table, writes results/rl_gain_sweep.json and names the best setting.
"""
import argparse, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform.startswith("linux"): os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np


def apply_scales(table, kp_all=1.0, kp_ankle_pitch=1.0):
    out = {}
    for k, (sc, kp, kd) in table.items():
        f = kp_all * (kp_ankle_pitch if k == "ankle_pitch" else 1.0)
        out[k] = (sc, kp * f, kd * np.sqrt(f))
    return out


def sweep(selfcheck_fn, push_fn, table, scales, ankle_scales, set_table):
    """selfcheck_fn()/push_fn() -> dict(fell, seconds_stood, ...). set_table(t) installs a gain table. Pure orchestration (unit-tested with fakes)."""
    rows = []
    for kind, values in (("all", scales), ("ankle", ankle_scales)):
        for v in values:
            set_table(apply_scales(table, kp_all=v) if kind == "all" else apply_scales(table, kp_ankle_pitch=v))
            try:
                s = selfcheck_fn(); q = push_fn(); row = dict(sweep=kind, scale=v, stand_fell=s["fell"], stand_s=s["seconds_stood"], pelvis_z=s["pelvis_z"], tilt=s["tilt"], push_fell=q["fell"], push_s=q["seconds_stood"])
            except Exception as e:
                row = dict(sweep=kind, scale=v, error=f"{type(e).__name__}: {e}")
            rows.append(row)
    set_table(table)
    return rows


def best(rows):
    ok = [r for r in rows if "error" not in r and not r["stand_fell"]]
    if not ok: return None
    return max(ok, key=lambda r: (not r["push_fell"], r["push_s"], -abs(r["tilt"])))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scales", default="0.7,1,1.5,2"); ap.add_argument("--ankle-scales", default="0.8,1,1.5,2"); a = ap.parse_args()
    import rl.fullbody_env as E
    orig = copy.deepcopy(E.CLASS_GAINS)

    def set_table(t): E.CLASS_GAINS.clear(); E.CLASS_GAINS.update(t)

    def push():
        env = E.FullBodyEnv(randomize=False, push_force=40.0, push_interval=(1.0, 1.5), max_steps=251); env.reset(seed=1); n = 0; term = False
        for _ in range(250):
            _, _, term, _, _ = env.step(np.zeros(E.N_J)); n += 1
            if term: break
        return dict(fell=bool(term), seconds_stood=n / 50.0)
    rows = sweep(lambda: E.selfcheck(5.0), push, orig, [float(x) for x in a.scales.split(",")], [float(x) for x in a.ankle_scales.split(",")], set_table)
    print(f"{'sweep':<6}{'scale':>6}  {'stand':<14}{'tilt':>6}{'z':>6}   push(40 N)")
    for r in rows:
        print(f"{r['sweep']:<6}{r['scale']:>6}  " + (r["error"] if "error" in r else f"{'FELL' if r['stand_fell'] else 'ok'} {r['stand_s']:.1f}s".ljust(14) + f"{r['tilt']:>6.2f}{r['pelvis_z']:>6.2f}   {'FELL' if r['push_fell'] else 'ok'} {r['push_s']:.1f}s"))
    b = best(rows); print("\nBEST:", b if b else "no setting stood for 5 s - send me this table")
    os.makedirs("results", exist_ok=True); json.dump(dict(rows=rows, best=b), open("results/rl_gain_sweep.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
