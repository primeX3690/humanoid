"""
Planned push-recovery stepper (v3): replaces the heuristic step target + straight-line swing of `push_recovery.py` with
  * `control.lateral_recovery.plan_recovery`   - step time, landing point (incl. cross-over) and stance-CoP shift optimised for DCM margin
  * `planning.collision_aware_swing`            - leg/foot collision-free swing trajectory

`choose_recovery_step` (control/lateral_recovery.py) is pure NumPy and unit-tested without a simulator. `PlannedPushRecoveryStepper` is the MuJoCo glue:
a drop-in replacement for PushRecoveryStepper (same `update(t)` / `tasks(wc, t)` interface):
    st = PlannedPushRecoveryStepper(S)      # instead of PushRecoveryStepper(S)
NOTE: the glue needs MuJoCo + the WBC and is NOT covered by tests in the offline sandbox it was written in - run
`python -m simulation.push_recovery_exp --planned` on your machine (see simulation/push_recovery_exp.py) and compare with the v2 JSON.
"""
from __future__ import annotations
import numpy as np
from control.lateral_recovery import RecoveryParams, plan_recovery, choose_recovery_step   # pure-NumPy planning lives there
from control.push_recovery import PushRecoveryStepper, StepperConfig


class PlannedPushRecoveryStepper(PushRecoveryStepper):
    def __init__(self, S, cfg: StepperConfig = StepperConfig(), params: RecoveryParams | None = None):
        super().__init__(S, cfg)
        self.rp = params or RecoveryParams(com_height=float(self.home[2]))
        self.rp.com_height = float(self.home[2])
        self.plan_log = []                                 # one entry per started step: what the planner intended (compare with steps_log = what happened)

    def _start_step(self, t, xi):
        S, w = self.S, self.w
        com = w.com()[:2]; v = w.com_vel()[:2]
        pL = S.d.site_xpos[w.sole["L"]][:2].copy(); pR = S.d.site_xpos[w.sole["R"]][:2].copy()
        ch = choose_recovery_step(com, v, pL, pR, self.rp, prev_swing=getattr(self, "_last_swing", None))
        if ch is None or "swing" not in ch:               # no collision-free plan: fall back to the v2 heuristic
            self.swing_traj = None; self.t_swing_cur = self.cfg.t_swing; self.t_lift_cur = 0.0
            self.plan_log.append(dict(t=float(t), fallback=True, com=[float(x) for x in com], vel=[float(x) for x in v]))
            return super()._start_step(t, xi)
        self.side, self.stance, self.target = ch["side"], ch["stance"], ch["swing"].goal[:2].copy()
        self.p0 = S.d.site_xpos[w.sole[self.side]].copy()
        self.swing_traj = ch["swing"]; self.t_lift_cur = self.rp.t_lift
        self.t_swing_cur = float(ch["T"])
        self.t_start = t; self.state = "STEP"; self._last_swing = self.side
        self.contacts = (self.stance,); self.swing = self.side; self.n_steps += 1
        self.last_plan = dict(margin=ch["margin"], T=ch["T"], crossover=ch["crossover"], feasible=ch["feasible"])
        self.plan_log.append(dict(t=float(t), swing=self.side, stance=self.stance, T=float(ch["T"]), target=[float(x) for x in self.target], margin=float(ch["margin"]),
                                  crossover=bool(ch["crossover"]), feasible=bool(ch["feasible"]), com=[float(x) for x in com], vel=[float(x) for x in v],
                                  xi=[float(x) for x in ch["xi0"]], xi_T=[float(x) for x in ch["xi_T"]], cop=[float(x) for x in ch["cop"]]))


class HybridPushRecoveryStepper(PushRecoveryStepper):
    """v2 step selection, targets and timing (they work: diag_push shows v2 beats every planner variant) + the collision-aware swing path ONLY when the straight
    swing would collide (cross-over steps). For all other steps the behaviour is IDENTICAL to v2."""

    def _start_step(self, t, xi):
        from planning.collision_aware_swing import plan_collision_free_swing
        super()._start_step(t, xi)
        self.swing_traj = None; self.t_lift_cur = 0.0; self.t_swing_cur = self.cfg.t_swing
        S, w = self.S, self.w
        st = S.d.site_xpos[w.sole[self.stance]].copy()
        sp = plan_collision_free_swing(self.p0, np.r_[self.target, self.p0[2]], st, T=self.cfg.t_swing)
        used = sp.feasible and sp.kind != "direct"
        if used: self.swing_traj = sp; self.target = sp.goal[:2].copy()
        if not hasattr(self, "plan_log"): self.plan_log = []
        self.plan_log.append(dict(t=float(t), hybrid=True, swing=self.side, stance=self.stance, kind=sp.kind, collision_free=bool(sp.feasible), used_planned_swing=bool(used),
                                  target=[float(x) for x in self.target], min_ankle_gap=float(sp.min_ankle_gap)))
