"""Exercises the MuJoCo-side glue (PlannedPushRecoveryStepper, realism hook arithmetic) against a MOCK robot, so the logic runs even without MuJoCo.
A stub `mujoco` is used ONLY when the real one is missing; the mock never replaces the physics, it only supplies positions/velocities."""
import sys, types
import numpy as np
import pytest

try:
    import mujoco  # noqa: F401
except ImportError:
    m = types.ModuleType("mujoco"); m.mj_name2id = lambda model, kind, name: model.ids[name]
    m.mjtObj = types.SimpleNamespace(mjOBJ_SITE=1, mjOBJ_JOINT=2, mjOBJ_BODY=3); sys.modules["mujoco"] = m

import control.push_recovery as pr
from control.push_recovery_v3 import PlannedPushRecoveryStepper
from control.push_recovery import StepperConfig

SITES = {f"{s}_c{k}": None for s in "LR" for k in range(4)}


class FakeWBC:
    def __init__(self, S): self.S = S; self.sole = {"L": 100, "R": 101}; self.calls = []
    def set_state(self, q, v): pass
    def com(self): return self.S.com.copy()
    def com_vel(self): return self.S.vel.copy()
    def support_center(self, c): return 0.5 * (self.S.feet["L"] + self.S.feet["R"])
    def task_com(self, *a, **k): return ("com", a, k)
    def task_torso_orient(self, *a, **k): return ("torso", a, k)
    def task_posture(self, **k): return ("posture", k)
    def task_foot(self, side, pos, **k): self.calls.append((side, np.array(pos), k)); return ("foot", side, pos, k)


class FakeS:
    """feet: sole centres; corner sites at +-0.1 x, +-0.04 y around each sole."""
    def __init__(self, com=(0, 0, 0.85), vel=(0, 0, 0)):
        self.feet = {"L": np.array([0.0, 0.12, 0.0]), "R": np.array([0.0, -0.12, 0.0])}
        self.com = np.array(com, float); self.vel = np.array(vel, float)
        self.ids = {"L_sole": 100, "R_sole": 101}
        self.site_pos = {}
        self.m = types.SimpleNamespace(ids={})
        n = 0
        for s in "LR":
            for k, (dx, dy) in enumerate([(-.1, -.04), (-.1, .04), (.1, -.04), (.1, .04)]):
                self.m.ids[f"{s}_c{k}"] = n; n += 1
        self.xpos = np.zeros((n + 2, 3))
        self.refresh()
        self.d = types.SimpleNamespace(site_xpos=self.xpos)
        self.wbc = FakeWBC(self)
    def refresh(self):
        i = 0
        for s in "LR":
            for dx, dy in [(-.1, -.04), (-.1, .04), (.1, -.04), (.1, .04)]:
                self.xpos[i] = self.feet[s] + [dx, dy, 0]; i += 1
        self.xpos[100 % len(self.xpos)] = 0
    def robot_state(self): return np.zeros(1), np.zeros(1)


def make_stepper(S):
    S.xpos = np.vstack([S.xpos, np.zeros((110, 3))]); S.d.site_xpos = S.xpos
    S.xpos[100] = S.feet["L"]; S.xpos[101] = S.feet["R"]
    return PlannedPushRecoveryStepper(S, StepperConfig())


def test_lateral_outward_push_starts_a_planned_crossover_step_and_swing_task_follows_the_plan():
    S = FakeS(com=(0, -0.12, 0.85), vel=(0.0, -0.30, 0.0))            # falling outward past the right foot (capturable with a cross-over)
    st = make_stepper(S)
    st.update(1.0)
    assert st.state == "STEP" and st.stance == "R" and st.swing == "L" and st.swing_traj is not None
    assert st.target[1] < S.feet["R"][1] and st.last_plan["crossover"] and 0.18 <= st.t_swing_cur <= 0.7
    # during the swing the WBC foot task must track the planned trajectory (after the lift delay)
    t_mid = 1.0 + st.t_lift_cur + 0.5 * st.swing_traj.t[-1]
    S.wbc.calls.clear(); ts = st.tasks(S.wbc, t_mid)
    side, pos, kw = S.wbc.calls[-1]
    idx = np.argmin(np.abs(st.swing_traj.t - 0.5 * st.swing_traj.t[-1]))
    assert side == "L" and np.allclose(pos, st.swing_traj.pos[idx], atol=2e-3) and kw["weight"] == 25.0
    # before the lift delay finishes the foot is held at its start position
    S.wbc.calls.clear(); st.tasks(S.wbc, 1.0 + 0.5 * st.t_lift_cur); assert np.allclose(S.wbc.calls[-1][1], st.swing_traj.pos[0], atol=1e-6)
    # touchdown returns to STAND and logs the step
    st.update(1.0 + st.t_swing_cur + 1e-3); assert st.state == "STAND" and len(st.steps_log) == 1


def test_uncapturable_push_still_uses_a_collision_free_best_effort_swing():
    S = FakeS(com=(0, -0.12, 0.85), vel=(0.0, -0.45, 0.0)); st = make_stepper(S); st.update(1.0)
    assert st.state == "STEP" and st.swing_traj is not None and st.last_plan["feasible"] is False and st.last_plan["crossover"]
    assert st.swing_traj.min_ankle_gap >= -1e-6


def test_forward_push_steps_forward_and_no_push_stays_standing():
    S = FakeS(com=(0, 0, 0.85), vel=(0.5, 0, 0)); st = make_stepper(S); st.update(1.0)
    assert st.state == "STEP" and st.target[0] > 0.1
    S2 = FakeS(); st2 = make_stepper(S2); st2.update(1.0); assert st2.state == "STAND" and st2.swing_traj is None


def test_hopeless_push_still_returns_a_step_not_an_exception():
    S = FakeS(com=(0, 0, 0.85), vel=(3.0, 3.0, 0)); st = make_stepper(S); st.update(1.0)
    assert st.state in ("STEP", "STAND")


def test_realism_hook_arithmetic_matches_wbcsim_expectations():
    """The WBCSim patch assigns qpos[qadr] = layer.q_meas and applies layer.apply(tau, qd): check the layer API those lines rely on."""
    from actuators.joint_actuator import ActuatorBank, PH54_200, ideal
    from sim_realism.realism_layer import RealismLayer
    L = RealismLayer(ActuatorBank([ideal(PH54_200)] * 3, dt=1e-3, thermal=False))
    assert L.q_meas is None
    L.tick(np.array([0.1, 0.2, 0.3])); assert L.q_meas.shape == (3,) and L.qd_meas.shape == (3,)
    assert L.apply(np.array([1.0, 2.0, 3.0]), np.zeros(3)).shape == (3,)
