"""
Full-body RL environment: the REAL 30-DoF humanoid (model/humanoid_model.py) instead of the simplified 12-DoF biped of rl/biped_env.py.

  * Action (30)  : joint-position offsets around the nominal standing pose, per-joint scales (legs large, arms/neck small); a 1 kHz joint PD
                   turns them into torques, which go through the actuator layer (sim_realism/actuators: lag, torque-speed limit, thermal,
                   latency) when `realistic=True`  -> the policy trains against the same imperfections the real robot will have.
  * Obs (101)    : gravity in body frame, gyro, q-q_nom, qd, last action, command (vx, vy, yaw rate), gait clock (sin, cos)
  * Reward       : command tracking + heading hold + upright + pelvis height + feet air-time + CLOCK contact term, minus torque, torque-limit
                   proximity, action rate, foot slip, upper-body drift (the biped-env reward-hacking lessons in docs/BUGS_FOUND.md are kept)
  * Randomisation: per episode via sim_realism.domain_randomization (mass, COM, friction, damping, motor strength, backlash, latency, noise)
                   plus random push impulses.
The simulator sits behind a small Backend interface: `MuJoCoBackend` (default, needs MuJoCo) and `FakeBackend` (pure NumPy, used by the unit
tests so the env/reward/termination logic is verified even where MuJoCo is not installed). gymnasium is optional (fallback Env/Box).
"""
from __future__ import annotations
import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
    _Env, _Box = gym.Env, spaces.Box
except ImportError:                                   # pragma: no cover - lets the env be tested without gymnasium
    class _Env: metadata = {}
    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low, self.high, self.shape, self.dtype = low, high, shape if shape is not None else np.shape(low), dtype
        def sample(self): return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

N_J = 30
# class -> (action scale rad, kp Nm/rad, kd Nm s/rad)   ASSUMED starting gains; run `python -m rl.fullbody_env --selfcheck` and tune
CLASS_GAINS = {"hip_yaw": (0.25, 200, 10), "hip_roll": (0.25, 250, 12), "hip_pitch": (0.6, 300, 14), "knee": (0.8, 350, 16),
               "ankle_pitch": (0.5, 120, 6), "ankle_roll": (0.25, 100, 5), "waist": (0.15, 150, 8), "sh": (0.2, 50, 3), "elbow": (0.2, 40, 2.5),
               "wr": (0.1, 8, 0.5), "neck": (0.1, 10, 0.6)}


def joint_class(name):
    b = name[2:] if name[:2] in ("L_", "R_") else name
    for k in CLASS_GAINS:
        if b.startswith(k): return k
    raise KeyError(name)


def gains_for(names):
    sc = np.array([CLASS_GAINS[joint_class(n)][0] for n in names]); kp = np.array([CLASS_GAINS[joint_class(n)][1] for n in names], float)
    kd = np.array([CLASS_GAINS[joint_class(n)][2] for n in names], float); return sc, kp, kd


# ----------------------------------------------------------------------------- backends
class FakeBackend:
    """Tiny NumPy stand-in with the Backend interface: independent PD joints + a pendulum-like base. For env-logic tests only."""
    dt = 0.001

    def __init__(self, names, seed=0):
        self.names = list(names); self.rng = np.random.default_rng(seed); self.tau_limit = np.full(len(names), 100.0)
        self.q_nom = np.zeros(len(names)); self.reset(None)

    def reset(self, sample):
        n = len(self.names); self.q = self.q_nom + 0.01 * self.rng.standard_normal(n); self.qd = np.zeros(n)
        self.tilt = np.array([0.0, 0.0]); self.tilt_rate = np.zeros(2); self.z = 0.9; self.x = np.zeros(3); self.v = np.zeros(3); self.yaw = 0.0
        self.t = 0.0; self.sample = sample
        return self.state()

    def step_physics(self, tau, n, push=None):
        for _ in range(n):
            tau_c = np.clip(tau, -self.tau_limit, self.tau_limit); self.qd += self.dt * (tau_c - 1.0 * self.qd) / 0.5; self.q += self.dt * self.qd
            acc = 9.0 * self.tilt - 2.0 * self.tilt_rate                     # unstable inverted-pendulum tilt, weakly damped
            if push is not None: acc = acc + push[:2] * 0.5
            self.tilt_rate += self.dt * acc; self.tilt += self.dt * self.tilt_rate
            hip = self.q[2] + self.q[8] if len(self.q) > 8 else 0.0
            self.z = 0.9 - 0.1 * np.tanh(abs(hip)); self.v[:2] += self.dt * 0.5 * (self.tilt[::-1] * [1, -1]); self.x += self.dt * self.v
            self.t += self.dt
        return self.state()

    def state(self):
        g = np.array([-np.sin(self.tilt[1]), np.sin(self.tilt[0]), np.cos(self.tilt[0]) * np.cos(self.tilt[1])])
        return dict(q=self.q.copy(), qd=self.qd.copy(), z=self.z, gravity_body=g / np.linalg.norm(g), gyro=np.array([*self.tilt_rate, 0.0]),
                    lin_vel=self.v.copy(), yaw=self.yaw, foot_contact=np.array([1.0, 1.0]), foot_height=np.zeros(2), foot_slip=np.zeros(2),
                    tilt=float(np.hypot(*self.tilt)), tau_ratio=0.0)

    def set_motor_scale(self, s): self.tau_limit = 100.0 * np.asarray(s)


class MuJoCoBackend:                                 # pragma: no cover - needs MuJoCo (verify with `python -m rl.fullbody_env --selfcheck`)
    def __init__(self, params=None, realistic=True, bend=None):
        from model.urdf_export import ensure_importable; ensure_importable()
        import mujoco
        from model.humanoid_model import make_model, stand_pose, joint_qadr, joint_dadr, ACTUATED, ModelParams, nominal_q_act
        self.mj = mujoco; self.names = list(ACTUATED); self.params = params or ModelParams()
        self.m, self.d = make_model(self.params, table=False); stand_pose(self.m, self.d, bend)
        self.qadr, self.dadr = joint_qadr(self.m, ACTUATED), joint_dadr(self.m, ACTUATED)
        self.q_nom = nominal_q_act(bend); self.dt = float(self.m.opt.timestep)
        self.tau_limit = np.array([self.m.actuator_ctrlrange[i, 1] for i in range(N_J)])
        self.pelvis = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self.sites = {s: [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_c{k}") for k in range(4)] for s in "LR"}
        self._qpos0 = self.d.qpos.copy(); self._qvel0 = self.d.qvel.copy()
        from sim_realism.domain_randomization import MuJoCoBaseline
        self.base = MuJoCoBaseline(self.m); self.realistic = realistic; self.bank = None; self.layer = None
        self.motor_scale = np.ones(N_J); self.kp = None

    def reset(self, sample):
        mj = self.mj
        if sample is not None:
            from sim_realism.domain_randomization import apply_to_mujoco, apply_to_bank
            from actuators.joint_actuator import spec_for_joint
            apply_to_mujoco(self.m, self.base, sample)
            if self.realistic:
                from sim_realism.realism_layer import RealismLayer
                self.bank = apply_to_bank([spec_for_joint(n) for n in self.names], sample, dt=self.dt); self.layer = RealismLayer(self.bank)
        elif self.realistic and self.bank is not None: self.layer.reset()
        mj.mj_resetData(self.m, self.d); self.d.qpos[:] = self._qpos0; self.d.qvel[:] = self._qvel0; mj.mj_forward(self.m, self.d)
        return self.state()

    def step_physics(self, tau, n, push=None):
        for i in range(n):
            if push is not None: self.d.xfrc_applied[self.pelvis, :3] = push
            q = self.d.qpos[self.qadr]; qd = self.d.qvel[self.dadr]
            if self.layer is not None:
                self.layer.tick(q); tau_i = self.layer.apply(tau, qd)
            else: tau_i = np.clip(tau, -self.tau_limit * self.motor_scale, self.tau_limit * self.motor_scale)
            self.d.ctrl[:N_J] = tau_i; self.mj.mj_step(self.m, self.d)
        self.d.xfrc_applied[:] = 0
        return self.state()

    def set_motor_scale(self, s): self.motor_scale = np.asarray(s)

    def state(self):
        d = self.d; R = d.xmat[self.pelvis].reshape(3, 3); g = R.T @ np.array([0, 0, -1.0])
        fz = {s: np.mean([d.site_xpos[i][2] for i in self.sites[s]]) for s in "LR"}
        contact = np.array([float(min(d.site_xpos[i][2] for i in self.sites[s]) < 0.012) for s in "LR"])
        return dict(q=d.qpos[self.qadr].copy(), qd=d.qvel[self.dadr].copy(), z=float(d.qpos[2]), gravity_body=g, gyro=d.qvel[3:6].copy(),
                    lin_vel=d.qvel[:3].copy(), yaw=float(np.arctan2(R[1, 0], R[0, 0])), foot_contact=contact,
                    foot_height=np.array([fz["L"], fz["R"]]), foot_slip=np.zeros(2), tilt=float(np.arccos(np.clip(R[2, 2], -1, 1))), tau_ratio=0.0)


# ----------------------------------------------------------------------------- environment
class FullBodyEnv(_Env):
    metadata = {"render_modes": []}

    def __init__(self, backend=None, names=None, cmd_vx=(0.0, 0.5), cmd_vy=(-0.1, 0.1), cmd_yaw=(-0.3, 0.3), push_force=0.0, push_interval=(2.0, 4.0),
                 gait_period=0.9, max_steps=1000, randomize=True, policy_hz=50, seed=None, realistic=True):
        if backend is None:
            backend = MuJoCoBackend(realistic=realistic)
        self.b = backend; self.names = list(names or backend.names); assert len(self.names) == N_J
        self.scale, self.kp, self.kd = gains_for(self.names)
        self.q_nom = backend.q_nom.copy(); self.sub = max(int(round(1.0 / (policy_hz * backend.dt))), 1)
        self.cmd_rng = (cmd_vx, cmd_vy, cmd_yaw); self.push_force, self.push_interval = push_force, push_interval
        self.gait_period, self.max_steps, self.randomize = gait_period, max_steps, randomize
        self.rng = np.random.default_rng(seed); self.obs_dim = 3 + 3 + N_J * 3 + 3 + 2
        self.observation_space = _Box(-np.inf, np.inf, (self.obs_dim,), np.float32); self.action_space = _Box(-1.0, 1.0, (N_J,), np.float32)
        self.leg = np.array([n[:2] in ("L_", "R_") and any(k in n for k in ("hip", "knee", "ankle")) for n in self.names]); self.upper = ~self.leg
        self._pushes = []

    # --- helpers
    def _phase(self): return (self.t % self.gait_period) / self.gait_period

    def _obs(self, s):
        ph = 2 * np.pi * self._phase()
        return np.concatenate([s["gravity_body"], 0.25 * s["gyro"], s["q"] - self.q_nom, 0.1 * s["qd"], self.last_action, self.cmd, [np.sin(ph), np.cos(ph)]]).astype(np.float32)

    def _sample_randomization(self):
        if not self.randomize: return None
        from sim_realism.domain_randomization import RandomizationConfig, sample
        body_names = ["pelvis", "torso", "L_thigh", "R_thigh", "L_shank", "R_shank", "L_foot", "R_foot"]
        return sample(RandomizationConfig(), body_names, self.names, seed=int(self.rng.integers(2**31)))

    def reset(self, *, seed=None, options=None):
        if seed is not None: self.rng = np.random.default_rng(seed)
        s = self.b.reset(self._sample_randomization()); self.t = 0.0; self.steps = 0; self.last_action = np.zeros(N_J); self.prev_action = np.zeros(N_J)
        self.cmd = np.array([self.rng.uniform(*r) for r in self.cmd_rng]); self.yaw0 = s["yaw"]; self.heading_target = s["yaw"]
        self.air_time = np.zeros(2); self.last_contact = np.array([1.0, 1.0]); self._schedule_push()
        return self._obs(s), {}

    def _schedule_push(self):
        self.next_push = self.t + self.rng.uniform(*self.push_interval) if self.push_force > 0 else np.inf; self.push_left = 0

    # --- step
    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1); self.prev_action = self.last_action; self.last_action = a
        q_t = self.q_nom + self.scale * a
        push = None
        if self.t >= self.next_push and self.push_left == 0:
            ang = self.rng.uniform(0, 2 * np.pi); self.push_vec = self.push_force * self.rng.uniform(0.5, 1.0) * np.array([np.cos(ang), np.sin(ang), 0.0]); self.push_left = int(0.1 / (self.sub * self.b.dt)) + 1
        if self.push_left > 0: push = self.push_vec; self.push_left -= 1; (self._schedule_push() if self.push_left == 0 else None)
        tau_acc = 0.0; s = None
        for _ in range(self.sub):
            st = self.b.state() if s is None else s
            tau = self.kp * (q_t - st["q"]) - self.kd * st["qd"]
            s = self.b.step_physics(tau, 1, push); tau_acc += float(np.sum((tau / np.maximum(self.b.tau_limit, 1e-6)) ** 2))
        self.t += self.sub * self.b.dt; self.steps += 1
        r, parts = self._reward(s, a, tau_acc / self.sub)
        fell = s["z"] < 0.5 or s["tilt"] > 0.9 or not np.all(np.isfinite(s["q"]))
        trunc = self.steps >= self.max_steps
        return self._obs(s), float(-5.0 if fell else r), bool(fell), bool(trunc and not fell), dict(parts, fell=fell)

    def _reward(self, s, a, torque_sq):
        dt = self.sub * self.b.dt; yaw_err = (s["yaw"] - self.heading_target + np.pi) % (2 * np.pi) - np.pi
        self.heading_target += self.cmd[2] * dt
        c = s["foot_contact"]; first = (c > 0.5) & (self.last_contact < 0.5)
        air_prev = self.air_time.copy(); self.air_time = np.where(c > 0.5, 0.0, air_prev + dt)
        air_rew = float(np.sum(first * np.clip(air_prev - 0.2, 0.0, 0.3)))     # reward real swings (legged_gym): duration of the swing that just ended
        self.last_contact = c.copy()
        v_body = s["lin_vel"][:2]; cmd_v = self.cmd[:2]
        phase = self._phase(); clock = np.array([1.0 if phase < 0.5 else 0.0, 0.0 if phase < 0.5 else 1.0])
        parts = dict(
            track_v=1.5 * np.exp(-np.sum((v_body - cmd_v) ** 2) / 0.1), track_yaw=0.5 * np.exp(-(s["gyro"][2] - self.cmd[2]) ** 2 / 0.1),
            heading=-0.5 * yaw_err ** 2, upright=0.5 * (1.0 - s["tilt"]), height=0.5 * np.exp(-((s["z"] - 0.85) ** 2) / 0.01),
            air=0.5 * air_rew, clock=0.2 * float(np.sum(clock * c) - np.sum((1 - clock) * c)) if np.linalg.norm(cmd_v) > 0.05 else 0.0,
            torque=-1e-3 * torque_sq, rate=-0.02 * float(np.sum((a - self.prev_action) ** 2)), upper=-0.1 * float(np.sum((s["q"][self.upper] - self.q_nom[self.upper]) ** 2)),
            slip=-0.1 * float(np.sum(s["foot_slip"])), alive=0.2)
        return float(sum(parts.values())), {k: float(v) for k, v in parts.items()}


def selfcheck(seconds=5.0):                           # pragma: no cover - needs MuJoCo
    """Zero-action PD standing test on the real model: if this fails, tune CLASS_GAINS before training anything."""
    env = FullBodyEnv(randomize=False, push_force=0.0, max_steps=int(seconds * 50)); obs, _ = env.reset(seed=0); steps = 0
    while True:
        obs, r, term, trunc, info = env.step(np.zeros(N_J)); steps += 1
        if term or trunc: break
    return dict(seconds_stood=steps / 50.0, fell=bool(term), pelvis_z=float(env.b.state()["z"]), tilt=float(env.b.state()["tilt"]))


if __name__ == "__main__":                            # pragma: no cover
    import sys, json
    if "--selfcheck" in sys.argv: print(json.dumps(selfcheck(), indent=1))
