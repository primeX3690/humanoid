import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless on Linux; must precede the first mujoco import
"""
End-to-end demo: "pick up the red block" -> goal parse -> symbolic plan -> behaviour tree -> head-camera perception ->
whole-body-controlled reach/grasp/lift in MuJoCo (balance is priority 1 throughout).
  python -m simulation.pick_demo [--estimated] [--command "pick the red block"]
"""
import argparse, json, time, sys
import numpy as np, mujoco
from simulation.wbc_sim import WBCSim
from perception.render import HeadCameraRenderer
from tasks.symbolic_planner import KeywordCommandParser, plan
from tasks.pick_skills import Ctx, build_tree
from tasks.behavior_tree import Status


def run(command="pick the red block", estimated=False, max_t=40.0, verbose=True, seed=0, recorder=None):
    S = WBCSim()
    R = HeadCameraRenderer(S.m)
    goal = KeywordCommandParser().parse(command)
    names = plan(goal.predicates)
    if verbose:
        print("COMMAND :", command, "\nGOAL    :", set(goal.predicates), "| target:", goal.target_label, "\nPLAN    :", names, flush=True)
    est = None
    if estimated:
        from simulation.estimated_state import EstimatedState
        S.wbc.kd_contact = 0.0           # estimated velocities must not enter the stance-foot constraint
        est = EstimatedState(S, seed)
    state_fn = (lambda: est.state()) if est else None
    ctx = Ctx(S, R, goal, state_fn, verbose)
    S.wbc.set_state(*(state_fn() if state_fn else S.robot_state()))
    c0 = S.wbc.com(); sc = S.wbc.support_center(("L", "R"))
    ctx.com_des = np.array([sc[0], sc[1], c0[2]])
    obj_id = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_BODY, "target_obj")
    ctx.obj_rest_z = S.d.xpos[obj_id][2]
    tree = build_tree(names)
    S.grip_cmd[:] = 0.004
    dt = S.m.opt.timestep
    status = Status.RUNNING; n = 0; t0 = time.time(); com_dev = []
    while ctx.t < max_t:
        if est:
            est.step()
        if n % S.sub == 0:
            S.wbc.set_state(*(state_fn() if state_fn else S.robot_state()))
            status = tree.tick(ctx)
            if status != Status.RUNNING:
                break
        S.step(ctx.wbc_tasks, state_fn=state_fn)
        n += 1; ctx.t = n * dt
        if recorder:
            recorder.maybe(S.d, ctx.t)
        if n % 50 == 0:
            S.record(ctx.t)
            true_com = S.wbc.d.subtree_com[S.wbc.root_body]  # controller model; use truth below
        if S.fell():
            ctx.log("[FATAL] robot fell"); status = Status.FAILURE; break
    # final truth metrics
    tcp = S.tcp_pos(ctx.hand or "R"); op = S.d.xpos[obj_id]
    mj_state = (S.d.qpos[:S.nq_r].copy(), S.d.qvel[:S.nv_r].copy()); S.wbc.set_state(*mj_state)
    com_xy = S.wbc.com()[:2]
    res = dict(command=command, estimated_state_in_loop=estimated, plan=names, success=bool(status == Status.SUCCESS),
               sim_time_s=ctx.t, wall_s=time.time() - t0, fell=bool(S.fell()),
               object_rise_cm=float((op[2] - ctx.obj_rest_z) * 100), object_to_tcp_cm=float(np.linalg.norm(op - tcp) * 100),
               min_support_margin_mm=1000 * min(S.log["support_margin"]) if S.log["support_margin"] else None,
               max_tau_ratio=max(S.log["tau_max_ratio"]) if S.log["tau_max_ratio"] else None,
               com_final_xy_dev_mm=float(np.linalg.norm(com_xy - ctx.com_des[:2]) * 1000),
               detected_object=None if ctx.obj is None else ctx.obj.centroid.tolist(),
               true_object_initial=None)
    if ctx.obj is not None:
        res["detection_err_mm"] = None
    return res, ctx, S


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimated", action="store_true")
    ap.add_argument("--command", default="pick the red block")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res, ctx, S = run(a.command, a.estimated)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
