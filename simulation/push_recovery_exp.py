"""Standing push recovery: WBC ankle/hip strategy only vs capture-point stepping.  python -m simulation.push_recovery_exp"""
import os, sys, json
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
from simulation.wbc_sim import WBCSim
from model.humanoid_model import repo_actuator_params, ModelParams
from control.push_recovery import PushRecoveryStepper, StepperConfig


def run(F, direction=(1.0, 0.0), dur=0.15, stepping=True, actuators="strong", T=5.0, t_push=0.8, verbose=False, recorder=None, planned=False):
    params = repo_actuator_params() if actuators == "repo" else ModelParams()
    S = WBCSim(table=False, params=params)
    if planned:                                   # v3: DCM-optimised step time/target, cross-over, collision-free swing
        from control.push_recovery_v3 import PlannedPushRecoveryStepper
        st = PlannedPushRecoveryStepper(S)
    else:
        st = PushRecoveryStepper(S)
    tid = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_BODY, "torso")
    d = np.array(direction) / np.linalg.norm(direction)
    S.wbc.set_state(*S.robot_state()); home = S.wbc.com().copy()
    peak = 0.0; fell = False
    n = int(T / 0.001)
    for i in range(n):
        t = i * 0.001
        S.ext_force = (tid, np.array([F * d[0], F * d[1], 0.0]) if t_push <= t < t_push + dur else np.zeros(3))
        if S.ctrl_sub % S.sub == 0:
            if stepping:
                contacts = st.update(t)
            else:
                contacts = ("L", "R")
                sc = S.wbc.support_center(("L", "R")); st.com_des = np.array([sc[0], sc[1], st.home[2]])
            S.cur_contacts = contacts
        S.step(lambda w: (st.tasks(w, t) if stepping else [w.task_com(st.com_des), w.task_torso_orient(), w.task_posture(level=4)]),
               contacts=S.cur_contacts)
        if recorder:
            recorder.maybe(S.d, t)
        if i % 20 == 0:
            S.wbc.set_state(*S.robot_state())
            peak = max(peak, float(np.linalg.norm(S.wbc.com()[:2] - home[:2])))
        if S.fell():
            fell = True; break
    S.wbc.set_state(*S.robot_state())
    com = S.wbc.com(); v = S.wbc.com_vel()
    return dict(F=F, dir=list(direction), stepping=stepping, actuators=actuators, fell=bool(fell), steps=st.n_steps,
                peak_com_dev_mm=1000 * peak, final_com_vel=float(np.linalg.norm(v[:2])),
                final_support_margin_mm=1000 * S.foot_polygon_margin(com[:2]) if not fell else None, step_log=st.steps_log)


if __name__ == "__main__":
    planned = "--planned" in sys.argv
    out = "results/fullbody/push_recovery_stepping_planned.json" if planned else "results/fullbody/push_recovery_stepping.json"
    rows = json.load(open(out)) if os.path.exists(out) else []          # resume after an interruption
    done = {(r["name"], r["F"], r["stepping"]) for r in rows}
    plan = []
    for direction, name in (((1, 0), "forward"), ((-1, 0), "backward"), ((0, 1), "lateral")):
        plan += [(direction, name, F, False) for F in (60, 90)]            # standing (ankle/hip strategy only)
        plan += [(direction, name, F, True) for F in (90, 120, 150, 180)]  # with capture-point stepping
    for direction, name, F, stepping in plan:
        if (name, F, stepping) in done:
            continue
        r = run(F, direction, stepping=stepping, T=3.5, planned=planned); r["name"] = name; rows.append(r)
        print(name, F, "step " if stepping else "stand", "FELL" if r["fell"] else "ok  ", "steps", r["steps"], "peak %.0f mm" % r["peak_com_dev_mm"], flush=True)
        json.dump(rows, open(out, "w"), indent=1)
