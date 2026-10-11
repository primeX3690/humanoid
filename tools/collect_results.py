"""
ONE command that runs every measurement I need and packs the outputs into results/send_to_claude.zip.

    python tools/collect_results.py                      # everything, cheap steps first (~1.5-2 h total; leave it running)
    python tools/collect_results.py --skip diag_push,measure_torque
    python tools/collect_results.py --only install,static,unit,rl_gains
A failing step never stops the next ones; its traceback goes into that step's log, which is included in the zip.
"""
import argparse, glob, os, subprocess, sys, time, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
UNIT = ["tests/test_stand_gains.py", "tests/test_glue_mock.py", "tests/test_stairs.py", "tests/test_hand.py", "tests/test_fullbody_env.py", "tests/test_design_search.py",
        "tests/test_preview_nmpc.py", "tests/test_tools_logic.py", "tests/test_terrain_swing.py", "tests/test_wbc_walk_reference.py"]
STEPS = [  # name, argv, timeout seconds
    ("install", [PY, "tools/verify_install.py"], 120),
    ("static", [PY, "tools/static_check.py"], 300),
    ("unit", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"] + UNIT, 1800),
    ("rl_selfcheck", [PY, "-m", "rl.fullbody_env", "--selfcheck"], 600),
    ("rl_gains", [PY, "tools/tune_rl_gains.py"], 3600),
    ("smoke", [PY, "-m", "simulation.sim_smoke", "--only", "model,stand,stand_realism,qp_backends,walk_straight,walk_turn,walk_step,walk_stairs,rl_selfcheck,hand"], 7200),
    ("qp_dump", [PY, "tools/dump_wbc_qps.py"], 3600),
    ("diag_push", [PY, "tools/diag_push.py"], 10800),
    ("measure_torque", [PY, "tools/measure_design_torque.py"], 7200),
]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--skip", default=""); ap.add_argument("--only", default=""); a = ap.parse_args()
    skip = set(filter(None, a.skip.split(","))); only = set(filter(None, a.only.split(",")))
    os.makedirs(os.path.join(ROOT, "results", "collect"), exist_ok=True); t_start = time.time(); summary = []
    for name, argv, tmo in STEPS:
        if name in skip or (only and name not in only): continue
        log = os.path.join(ROOT, "results", "collect", f"{name}.log"); t0 = time.time(); print(f"[run ] {name} ...", flush=True)
        env = dict(os.environ, PYTHONPATH=ROOT, PYTHONIOENCODING="utf-8")
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as f:
                p = subprocess.run(argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=tmo, env=env); status = "ok" if p.returncode == 0 else f"exit {p.returncode}"
        except subprocess.TimeoutExpired: status = f"TIMEOUT after {tmo}s"
        except Exception as e: status = f"could not start: {e}"
        summary.append((name, status, round(time.time() - t0))); print(f"[{'done' if status == 'ok' else 'FAIL'}] {name}: {status} ({summary[-1][2]} s)", flush=True)
    out = os.path.join(ROOT, "results", "send_to_claude.zip")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("SUMMARY.txt", "\n".join(f"{n:<16}{s:<28}{t} s" for n, s, t in summary))
        for f in glob.glob(os.path.join(ROOT, "results", "collect", "*.log")): z.write(f, os.path.join("collect", os.path.basename(f)))
        for pat in ("sim_smoke_*.json", "diag_push.json", "wbc_qps.npz", "rl_gain_sweep.json", "design_requirements.json"):
            for f in glob.glob(os.path.join(ROOT, "results", pat)):
                if os.path.getmtime(f) >= t_start - 5: z.write(f, os.path.basename(f))
    print(f"\npacked -> {out}\n" + "\n".join(f"  {n:<16}{s}" for n, s, t in summary) + "\nSend me results/send_to_claude.zip")


if __name__ == "__main__":
    main()
