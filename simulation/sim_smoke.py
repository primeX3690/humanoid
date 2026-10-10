"""
One-command verification of everything v3 added on the MuJoCo side. Run on a machine with MuJoCo + OSQP:

    python -m simulation.sim_smoke            # all checks (~20-40 min, push sweep is the slow part)
    python -m simulation.sim_smoke --quick    # skips the push comparison and the long walks
    python -m simulation.sim_smoke --only model,stand,walk_straight

Each check runs in its own try/except, so one failure never hides the others. Writes results/sim_smoke.json and prints a table.
Send me that JSON (and any traceback) and I will fix what is red.
"""
from __future__ import annotations
import os, sys, json, time, traceback
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np

RESULTS = {}


def check(name, quick_skip=False):
    def deco(fn):
        fn._name, fn._quick_skip = name, quick_skip; CHECKS.append(fn); return fn
    return deco


CHECKS = []


def _stand(S, seconds=2.0):
    S.wbc.set_state(*S.robot_state()); c0 = S.wbc.com(); sc = S.wbc.support_center(("L", "R")); com_des = np.array([sc[0], sc[1], c0[2]])
    for _ in range(int(seconds * 1000)):
        S.step(lambda w: [w.task_com(com_des), w.task_torso_orient(), w.task_posture(level=3)])
        if S.fell(): return False
    return True


@check("model")
def c_model():
    import mujoco
    from model.humanoid_model import make_model, ModelParams
    from model.design_v3 import design_v3_params
    out = {}
    for name, kw, p in (("default", {}, ModelParams()), ("dexterous", dict(hand="dexterous"), ModelParams()), ("design_v3", {}, design_v3_params(28.0)[0])):
        m, d = make_model(p, table=False, **kw); mujoco.mj_forward(m, d)
        out[name] = dict(nu=int(m.nu), nbody=int(m.nbody), mass_kg=float(m.body_mass.sum()))
    assert out["dexterous"]["nu"] == out["default"]["nu"] + 22 and abs(out["default"]["mass_kg"] - 25.0) < 1e-6 and abs(out["design_v3"]["mass_kg"] - 28.0) < 1e-3
    return out


@check("stand")
def c_stand():
    from simulation.wbc_sim import WBCSim
    S = WBCSim(table=False); ok = _stand(S, 2.0); z = float(S.d.qpos[2]); assert ok and z > 0.6, f"fell or collapsed (pelvis z={z:.2f})"
    return dict(pelvis_z=z)


@check("stand_realism")
def c_stand_realism():
    from simulation.wbc_sim import WBCSim
    from model.humanoid_model import ACTUATED
    from actuators.joint_actuator import humanoid_bank
    from sim_realism.realism_layer import RealismLayer
    L = RealismLayer(humanoid_bank(ACTUATED, dt=0.001, cmd_delay_s=0.004, enc_delay_s=0.002)); S = WBCSim(table=False, realism=L)
    ok = _stand(S, 2.0); rep = L.report(); assert ok, "fell with actuator lag/latency/limits on"
    return dict(pelvis_z=float(S.d.qpos[2]), max_temp_c=float(rep["temp_c"].max()), electrical_w=rep["electrical_power_w"])


@check("qp_backends")
def c_qp():
    import importlib.util
    from simulation.wbc_sim import WBCSim
    res = {}
    for be in (["osqp"] if importlib.util.find_spec("osqp") else []) + ["fast"]:
        S = WBCSim(table=False); S.wbc.qp_backend = be; t0 = time.time(); ok = _stand(S, 1.0); res[be] = dict(ok=ok, wall_s=time.time() - t0, pelvis_z=float(S.d.qpos[2]))
        assert ok, f"{be} backend fell"
    if "osqp" in res: assert abs(res["osqp"]["pelvis_z"] - res["fast"]["pelvis_z"]) < 0.02, "backends disagree"
    return res


@check("push", quick_skip=True)
def c_push():
    from simulation.push_recovery_exp import run
    cases = [("lateral", (0.0, 1.0), 60), ("lateral", (0.0, 1.0), 90), ("lateral", (0.0, 1.0), 120), ("forward", (1.0, 0.0), 90), ("backward", (-1.0, 0.0), 120)]
    out = []
    for name, dr, F in cases:
        row = dict(case=f"{name} {F} N")
        for tag, planned in (("v2", False), ("planned", True)):
            r = run(F, dr, stepping=True, actuators="strong", T=3.5, planned=planned); row[tag] = dict(fell=r["fell"], steps=r["steps"], peak_mm=r.get("peak_com_dev_mm"))
        out.append(row)
    return out


def _walk(name, **kw):
    from simulation.walk_wbc import run
    r, S = run("repo", verbose=False, **kw)
    assert not r["fell"], f"{name}: fell after {r['sim_time_s']:.1f} s"
    return dict(distance_m=r["distance_m"], plan_m=r["plan_distance_m"], com_rms_mm=r["com_track_rms_mm"], touchdown_mm=r["touchdown_err_mm_mean"], wall_s=r["wall_s"])


@check("walk_straight")
def c_walk(): return _walk("straight", n_steps=4)


@check("walk_fast_qp")
def c_walk_fast(): return _walk("fast qp", n_steps=4, qp_backend="fast")


@check("walk_turn", quick_skip=True)
def c_walk_turn(): return _walk("turn", n_steps=6, turn_rate=0.05)


@check("walk_step", quick_skip=True)
def c_walk_step():
    from terrain.terrain_profile import StepTerrain
    return _walk("step", n_steps=6, terrain=StepTerrain(0.9, 0.04))


@check("walk_stairs", quick_skip=True)
def c_walk_stairs():
    from terrain.terrain_profile import StairTerrain
    return _walk("stairs", n_steps=0, terrain=StairTerrain(0.55, 0.28, 0.08, 2), zc=0.85)


@check("rl_selfcheck")
def c_rl():
    from rl.fullbody_env import selfcheck
    r = selfcheck(3.0); assert not r["fell"], f"zero-action PD stance falls after {r['seconds_stood']:.1f} s: tune CLASS_GAINS in rl/fullbody_env.py"
    return r


@check("hand")
def c_hand():
    import mujoco
    from model.humanoid_model import make_model, ModelParams
    from model.hand_model import joint_names
    m, d = make_model(ModelParams(), table=False, hand="dexterous"); mujoco.mj_forward(m, d)
    for n in joint_names("R"):
        i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n); d.ctrl[i] = 1.0
    for _ in range(500): mujoco.mj_step(m, d)
    assert np.all(np.isfinite(d.qpos)), "NaN after closing the dexterous hand"
    return dict(max_abs_qvel=float(np.abs(d.qvel).max()))


def main(argv):
    quick = "--quick" in argv; only = None
    if "--only" in argv: only = set(argv[argv.index("--only") + 1].split(","))
    rows = []
    for fn in CHECKS:
        if only and fn._name not in only: continue
        if quick and fn._quick_skip: continue
        t0 = time.time(); row = dict(check=fn._name)
        try: row.update(status="PASS", data=fn())
        except Exception as e: row.update(status="FAIL", error=f"{type(e).__name__}: {e}", trace=traceback.format_exc(limit=6))
        row["wall_s"] = round(time.time() - t0, 1); rows.append(row)
        print(f"[{row['status']}] {row['check']:<14} {row['wall_s']:>7.1f}s  " + (row.get("error", "") if row["status"] == "FAIL" else ""), flush=True)
    os.makedirs("results", exist_ok=True); json.dump(rows, open("results/sim_smoke.json", "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    n_fail = sum(r["status"] == "FAIL" for r in rows); print(f"\n{len(rows) - n_fail}/{len(rows)} passed -> results/sim_smoke.json"); return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
