import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless on Linux; must precede the first mujoco import
"""Closed-loop MuJoCo runner for the whole-body controller (torque control, WBC @ 200 Hz, sim @ 1 kHz)."""
import numpy as np
import mujoco
from model.humanoid_model import make_model, stand_pose, joint_qadr, joint_dadr, ACTUATED, ModelParams
from control.whole_body_qp import WholeBodyController


def min_jerk(t, T):
    s = np.clip(t / T, 0, 1)
    pos = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    vel = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / T
    return pos, vel


class WBCSim:
    def __init__(self, table=True, wbc_hz=200, seed=0, params=None, leg_bend=None, **model_kw):
        params = params or ModelParams()
        self.m, self.d = make_model(params, table=table, **model_kw)
        stand_pose(self.m, self.d, leg_bend)
        self.wbc = WholeBodyController(params, leg_bend=leg_bend)
        self.nq_r, self.nv_r = self.wbc.m.nq, self.wbc.m.nv
        self.sub = int(round(1.0 / (wbc_hz * self.m.opt.timestep)))
        self.ctrl_sub = 0
        self.tau = np.zeros(30)
        self.grip_cmd = np.array([0.004, 0.004])
        self.log = {k: [] for k in ["t", "com", "support_margin", "tau_max_ratio", "fric_ratio", "resid1", "status"]}
        self.ext_force = None   # (body_id, force3) applied each step
        mujoco.mj_forward(self.m, self.d)
        self.qadr = joint_qadr(self.m, ACTUATED)
        self.dadr = joint_dadr(self.m, ACTUATED)

    def robot_state(self):
        return self.d.qpos[:self.nq_r].copy(), self.d.qvel[:self.nv_r].copy()

    def foot_polygon_margin(self, com_xy):
        """Distance (m) from CoM projection to the nearest edge of the support-polygon bounding box of both soles."""
        pts = np.array([self.d.site_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_c{k}")]
                        for s in "LR" for k in range(4)])
        lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        return float(min(com_xy[0] - lo[0], hi[0] - com_xy[0], com_xy[1] - lo[1], hi[1] - com_xy[1]))

    def step(self, tasks_fn, contacts=("L", "R"), state_fn=None):
        """One 1 kHz sim step; re-solves WBC every `sub` steps. state_fn() -> (qpos, qvel) estimate (default: truth)."""
        if self.ctrl_sub % self.sub == 0:
            qpos, qvel = state_fn() if state_fn else self.robot_state()
            self.wbc.set_state(qpos, qvel)
            tasks = tasks_fn(self.wbc)
            out = self.wbc.solve(qpos, qvel, tasks, contacts)
            if out["status"] == "ok":
                self.tau = out["tau"]
            self.last_out = out
        self.ctrl_sub += 1
        self.d.ctrl[:30] = self.tau
        self.d.ctrl[30:32] = self.grip_cmd
        if self.ext_force is not None:
            self.d.xfrc_applied[self.ext_force[0], :3] = self.ext_force[1]
        mujoco.mj_step(self.m, self.d)

    def record(self, t):
        wbc = self.wbc
        qpos, qvel = self.robot_state()
        wbc.set_state(qpos, qvel)
        com = wbc.com()
        self.log["t"].append(t); self.log["com"].append(com)
        self.log["support_margin"].append(self.foot_polygon_margin(com[:2]))
        self.log["tau_max_ratio"].append(float(np.max(np.abs(self.tau) / wbc.tau_max)))
        out = getattr(self, "last_out", None)
        if out and out.get("f") is not None and len(out["f"]) >= 3:
            f = out["f"].reshape(-1, 3)
            fz = np.maximum(f[:, 2], 1e-6)
            self.log["fric_ratio"].append(float(np.max(np.hypot(f[:, 0], f[:, 1]) / fz * (f[:, 2] > 1.0))))
            self.log["resid1"].append(out["resid"].get(1, np.nan))
        self.log["status"].append(getattr(out, "get", lambda k: None)("status") if out else None)

    def tcp_pos(self, side):
        return self.d.site_xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")].copy()

    def fell(self):
        return self.d.qpos[2] < 0.6
