"""
Capture-point push recovery with stepping, on top of the whole-body QP.

State machine
  STAND : double support, CoM over the support centre (ankle/hip strategy happens inside the WBC automatically).
  STEP  : triggered when the capture point  xi = c + v/omega  leaves the (shrunk) support polygon, i.e. when standing
          balance is no longer possible without moving a foot.  A swing foot is chosen on the side of xi, its landing
          point is xi pushed slightly further along the push, clipped to the leg-reach / anti-collision box of the stance
          foot; min-jerk xy + sine-hump z swing; the stance foot is the only contact during the swing.
  After touchdown the new support polygon contains xi again (or another step is taken, up to `max_steps`).
Limits (stated honestly): LIPM-based, sagittal+lateral, flat ground, no turning, fixed step duration.
"""
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class StepperConfig:
    t_swing: float = 0.25
    clearance: float = 0.06
    margin: float = 0.04            # shrink of the support polygon before a step is triggered (m)
    overshoot: float = 0.08         # land this far beyond xi along the push (m)
    reach_x: float = 0.50           # max |dx| of the landing foot from the stance foot
    sep_y_min: float = 0.10         # min lateral foot separation (leg-collision guard)
    sep_y_max: float = 0.40
    cooldown: float = 0.10
    max_steps: int = 4
    kp_com_swing: float = 60.0


def min_jerk01(s):
    s = np.clip(s, 0, 1)
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5, (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4), (60 * s - 180 * s ** 2 + 120 * s ** 3)


class PushRecoveryStepper:
    def __init__(self, S, cfg: StepperConfig = StepperConfig()):
        self.S, self.cfg = S, cfg
        self.w = S.wbc
        self.state = "STAND"
        self.contacts = ("L", "R")
        self.swing = None
        self.n_steps = 0
        self.t_land = -1.0
        self.t_swing_cur = cfg.t_swing      # duration of the CURRENT swing (planners may choose it per step)
        self.t_lift_cur = 0.0
        self.swing_traj = None              # optional planning.collision_aware_swing.SwingPlan overriding the min-jerk line
        self.com_des = None
        self.steps_log = []
        S.wbc.set_state(*S.robot_state())
        c0 = S.wbc.com(); sc = S.wbc.support_center(("L", "R"))
        self.home = np.array([sc[0], sc[1], c0[2]])
        self.omega = np.sqrt(9.81 / c0[2])

    # support polygon (axis-aligned box over the corners of the feet in contact)
    def _poly(self, contacts):
        pts = np.array([self.S.d.site_xpos[mujoco.mj_name2id(self.S.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_c{k}")]
                        for s in contacts for k in range(4)])
        return pts[:, :2].min(0), pts[:, :2].max(0)

    def capture_point(self):
        w = self.w
        return w.com()[:2] + w.com_vel()[:2] / self.omega

    def update(self, t):
        S, w, c = self.S, self.w, self.cfg
        w.set_state(*S.robot_state())
        xi = self.capture_point()
        if self.state == "STAND":
            lo, hi = self._poly(("L", "R"))
            outside = np.any(xi < lo + c.margin) or np.any(xi > hi - c.margin)
            if outside and self.n_steps < c.max_steps and t - self.t_land > c.cooldown:
                self._start_step(t, xi)
            else:
                sc = w.support_center(("L", "R"))
                self.com_des = np.array([sc[0], sc[1], self.home[2]])
        elif self.state == "STEP":
            if t >= self.t_start + self.t_swing_cur:
                self.state = "STAND"; self.contacts = ("L", "R"); self.swing = None; self.t_land = t
                self.steps_log.append(dict(t=float(t), side=self.side, target=self.target.tolist(),
                                           landed=S.d.site_xpos[w.sole[self.side]][:2].tolist()))
            else:
                stance = self.stance
                sp = S.d.site_xpos[w.sole[stance]]
                mid = 0.5 * (sp[:2] + self.target)
                self.com_des = np.array([mid[0], mid[1], self.home[2]])
        return self.contacts

    def _start_step(self, t, xi):
        S, w, c = self.S, self.w, self.cfg
        com = w.com()[:2]; v = w.com_vel()[:2]
        push_dir = v / (np.linalg.norm(v) + 1e-9)
        pL = S.d.site_xpos[w.sole["L"]][:2].copy(); pR = S.d.site_xpos[w.sole["R"]][:2].copy()
        mid_y = 0.5 * (pL[1] + pR[1])
        lateral = abs(xi[1] - mid_y) > 0.04 and abs(xi[1] - mid_y) > 0.5 * abs(xi[0] - 0.5 * (pL[0] + pR[0]))
        if lateral:
            # the CoM is falling onto the "loaded" foot (side of xi); that foot cannot leave the ground.
            # The UNLOADED foot steps to the far (outer) side of the loaded foot: a cross-over step, landing slightly behind.
            # loaded foot = the one nearest to xi on the push side (feet may already be crossed over after a first step)
            loaded = "L" if ((pL[1] > pR[1]) == (xi[1] > mid_y)) else "R"
            side, stance = ("R", "L") if loaded == "L" else ("L", "R")
            sp = S.d.site_xpos[w.sole[stance]][:2]
            push_sgn = 1.0 if xi[1] > mid_y else -1.0
            xi_T = sp + (xi - sp) * np.exp(self.omega * c.t_swing)
            ty = max(push_sgn * (xi_T[1] + push_sgn * c.overshoot - sp[1]), c.sep_y_min)
            ty = min(ty, c.sep_y_max)
            target = np.array([np.clip(sp[0] + (xi_T[0] - sp[0]) * 0.5 - 0.06, sp[0] - c.reach_x, sp[0] + c.reach_x),
                               sp[1] + push_sgn * ty])
        else:
            if abs(xi[1] - mid_y) > 0.04:
                side = "L" if xi[1] > mid_y else "R"
            else:
                side = "L" if (pL - pR) @ push_dir < 0 else "R"      # trailing foot steps forward
            stance = "R" if side == "L" else "L"
            sp = S.d.site_xpos[w.sole[stance]][:2]
            # capture point at touchdown if the CoP stays at the stance-foot centre:  xi_T = p + (xi - p) e^{omega T}
            xi_T = sp + (xi - sp) * np.exp(self.omega * c.t_swing)
            target = xi_T + c.overshoot * push_dir
            target = np.array([np.clip(target[0], sp[0] - c.reach_x, sp[0] + c.reach_x), target[1]])
            sgn = 1.0 if side == "L" else -1.0
            dy = np.clip(sgn * (target[1] - sp[1]), c.sep_y_min, c.sep_y_max)
            target[1] = sp[1] + sgn * dy
        self.side, self.stance, self.target = side, stance, target
        self.p0 = S.d.site_xpos[w.sole[side]].copy()
        self.t_start = t; self.state = "STEP"
        self.contacts = (stance,); self.swing = side; self.n_steps += 1

    def tasks(self, wc, t):
        """STAND: strict hierarchy (balance first).  STEP: the CoM target is physically unreachable while the capture point
        is outside the stance foot, so a strict level-1 CoM task freezes the swing leg; during the swing CoM, torso and
        swing foot are therefore solved together (weights 1 : 1 : 25) -- the dynamics/friction/CoP constraints still hold."""
        c = self.cfg
        if self.state != "STEP":
            return [wc.task_com(self.com_des), wc.task_torso_orient(), wc.task_posture(level=3)]
        ts = [wc.task_com(self.com_des, kp=c.kp_com_swing, weight=1.0), wc.task_torso_orient(weight=1.0)]
        if self.swing_traj is not None:                                  # collision-aware planned swing (planning/collision_aware_swing.py)
            tr = self.swing_traj; tl = min(max(t - self.t_start - self.t_lift_cur, 0.0), tr.t[-1])
            col = lambda A: np.array([np.interp(tl, tr.t, A[:, k]) for k in range(3)])
            ts.append(wc.task_foot(self.swing, col(tr.pos), vel_des=col(tr.vel), acc_ff=col(tr.acc), level=1, weight=25.0))
            ts.append(wc.task_posture(level=2))
            return ts
        s = (t - self.t_start) / self.t_swing_cur
        p, v, a = min_jerk01(s)
        sc_ = min(max(s, 0), 1)
        xy = self.p0[:2] + (self.target - self.p0[:2]) * p
        z = c.clearance * np.sin(np.pi * sc_)
        vxy = (self.target - self.p0[:2]) * v / self.t_swing_cur
        vz = c.clearance * np.pi * np.cos(np.pi * sc_) / self.t_swing_cur
        axy = (self.target - self.p0[:2]) * a / self.t_swing_cur ** 2
        az = -c.clearance * (np.pi / self.t_swing_cur) ** 2 * np.sin(np.pi * sc_)
        ts.append(wc.task_foot(self.swing, np.array([xy[0], xy[1], z]), vel_des=np.array([vxy[0], vxy[1], vz]),
                               acc_ff=np.array([axy[0], axy[1], az]), level=1, weight=25.0))
        ts.append(wc.task_posture(level=2))
        return ts
