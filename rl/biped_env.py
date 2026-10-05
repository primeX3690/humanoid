"""
rl/biped_env.py

Gymnasium environment for training a walking / push-robust standing
policy for the biped defined in rl/biped_model.py (MuJoCo).

Design follows the conventions used in current legged-robot RL work
(ETH legged_gym / Rudin et al. 2021; Cassie clock-based rewards, Siekmann
et al. 2021; domain randomisation, Tobin et al. 2017):

  * Action  : 12 joint-position offsets around the nominal standing pose,
              tracked by MuJoCo position actuators (joint PD with implicit
              damping, physics step 2 ms), policy at 50 Hz (frame_skip = 10).
              Force is saturated at the real actuator limit (H54P-200,
              44.7 N*m); gains live in rl/biped_model.py.
  * Obs     : gravity vector and angular velocity in the pelvis frame,
              joint positions (relative to nominal) and velocities, the
              previous action, the velocity command, and a gait clock
              (sin/cos of the gait phase).
  * Reward  : velocity-command tracking + heading hold + upright/height
              terms + a clock-based foot-contact term + a feet-air-time term
              (legged_gym, Rudin et al. 2021: rewards real swing phases, not
              contact chatter) minus torque, action-rate, foot-slip and
              hip-yaw/roll drift penalties. The heading and air-time terms
              were added after a first policy was found (tests/test_rl_env.py,
              docs/BUGS_FOUND.md) to reach the commanded speed by circling
              (-100 deg heading drift in 20 s) with 5-7 Hz foot chatter and
              2.5-4 cm foot clearance - i.e. exploiting the earlier reward.
  * Domain randomisation (per episode): ground friction, total mass scale,
              motor strength scale.
  * Disturbances: random horizontal velocity impulses applied to the
              pelvis during the episode (magnitude set by `push_velocity`).
  * Termination: pelvis too low or torso tilted beyond a limit (a fall).
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

from rl.biped_model import (
    build_mjcf, default_spec, BipedSpec, NOMINAL_LEG_POSE, N_JOINTS,
    nominal_standing_pelvis_height,
)

# per-joint action scale (rad): how far a unit action moves the PD target from nominal
_ACTION_SCALE_LEG = np.array([0.25, 0.25, 0.6, 0.8, 0.5, 0.25])
ACTION_SCALE = np.tile(_ACTION_SCALE_LEG, 2)
NOMINAL_POSE = np.tile(NOMINAL_LEG_POSE, 2)


class BipedEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, spec: BipedSpec | None = None, cmd_vx_range=(0.0, 0.5),
                 push_velocity: float = 0.0, push_interval_s=(2.0, 4.0),
                 gait_period_s: float = 0.8, max_episode_steps: int = 1000,
                 randomize: bool = True, seed: int | None = None):
        super().__init__()
        self.biped = spec or default_spec()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.biped))
        self.data = mujoco.MjData(self.model)
        self.frame_skip = 10
        self.dt = self.model.opt.timestep * self.frame_skip

        self.cmd_vx_range = cmd_vx_range
        self.push_velocity = push_velocity
        self.push_interval_s = push_interval_s
        self.gait_period_s = gait_period_s
        self.max_episode_steps = max_episode_steps
        self.randomize = randomize
        self.torque_limit = self.biped.torque_limit_nm
        self._np_random_seed = seed

        # cached ids
        self._pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._floor_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_geoms = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{s}_foot_geom")
                            for s in ("left", "right")]
        self._foot_bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_foot")
                             for s in ("left", "right")]

        # nominal model params for domain randomisation
        self._nom_mass = self.model.body_mass.copy()
        self._nom_inertia = self.model.body_inertia.copy()
        self._nom_friction = self.model.geom_friction.copy()
        self._nom_forcerange = self.model.actuator_forcerange.copy()

        self.h_nominal = nominal_standing_pelvis_height(self.biped)
        n_obs = 3 + 3 + N_JOINTS + N_JOINTS + N_JOINTS + 1 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(n_obs,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(N_JOINTS,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._last_action = np.zeros(N_JOINTS)
        self._step = 0
        self._cmd_vx = 0.0
        self._motor_scale = 1.0
        self._next_push_step = 10**9
        self._air_time = np.zeros(2)
        self._prev_contacts = np.ones(2, dtype=bool)

    # ---------------------------------------------------------------- helpers
    def _rot(self) -> np.ndarray:
        return self.data.xmat[self._pelvis].reshape(3, 3)

    def _yaw(self) -> float:
        R = self._rot()
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _gravity_body(self) -> np.ndarray:
        return self._rot().T @ np.array([0.0, 0.0, -1.0])

    def _foot_contacts(self) -> np.ndarray:
        contact = np.zeros(2, dtype=bool)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            geoms = (c.geom1, c.geom2)
            if self._floor_geom in geoms:
                other = geoms[0] if geoms[1] == self._floor_geom else geoms[1]
                for f, g in enumerate(self._foot_geoms):
                    if other == g:
                        contact[f] = True
        return contact

    def _phase(self) -> float:
        return (self._step * self.dt / self.gait_period_s) % 1.0

    def _obs(self) -> np.ndarray:
        q = self.data.qpos[7:7 + N_JOINTS]
        qd = self.data.qvel[6:6 + N_JOINTS]
        ang_vel = self.data.qvel[3:6]  # free-joint angular velocity is in the body frame
        ph = 2.0 * np.pi * self._phase()
        obs = np.concatenate([
            self._gravity_body(), ang_vel * 0.25, q - NOMINAL_POSE, qd * 0.05,
            self._last_action, [self._cmd_vx], [np.sin(ph), np.cos(ph)],
        ])
        return obs.astype(np.float32)

    def _randomize_model(self):
        m = self.model
        m.body_mass[:] = self._nom_mass
        m.body_inertia[:] = self._nom_inertia
        m.geom_friction[:] = self._nom_friction
        m.actuator_forcerange[:] = self._nom_forcerange
        self._motor_scale = 1.0
        if not self.randomize:
            return
        scale = self._rng.uniform(0.9, 1.1)
        m.body_mass[:] = self._nom_mass * scale
        m.body_inertia[:] = self._nom_inertia * scale
        fric = self._rng.uniform(0.6, 1.2)
        m.geom_friction[:, 0] = self._nom_friction[:, 0] * fric
        self._motor_scale = self._rng.uniform(0.9, 1.0)
        m.actuator_forcerange[:] = self._nom_forcerange * self._motor_scale

    # ------------------------------------------------------------- gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self._randomize_model()

        self.data.qpos[0:3] = [0.0, 0.0, self.h_nominal + 0.005]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        noise = self._rng.uniform(-0.03, 0.03, N_JOINTS) if self.randomize else 0.0
        self.data.qpos[7:7 + N_JOINTS] = NOMINAL_POSE + noise
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self._last_action = np.zeros(N_JOINTS)
        self._air_time = np.zeros(2)
        self._prev_contacts = np.ones(2, dtype=bool)
        self._step = 0
        lo, hi = self.cmd_vx_range
        self._cmd_vx = float(self._rng.uniform(lo, hi))
        self._schedule_push()
        return self._obs(), {}

    def _schedule_push(self):
        if self.push_velocity > 0:
            lo, hi = self.push_interval_s
            self._next_push_step = self._step + int(self._rng.uniform(lo, hi) / self.dt)
        else:
            self._next_push_step = 10**9

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = NOMINAL_POSE + ACTION_SCALE * action

        torque_sq = 0.0
        self.data.ctrl[:] = target            # position actuators: PD torque computed inside MuJoCo
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
            torque_sq += float(np.sum(self.data.actuator_force ** 2))
        torque_sq /= self.frame_skip
        self._step += 1

        if self._step >= self._next_push_step:
            ang = self._rng.uniform(0, 2 * np.pi)
            mag = self._rng.uniform(0.3, 1.0) * self.push_velocity
            self.data.qvel[0] += mag * np.cos(ang)
            self.data.qvel[1] += mag * np.sin(ang)
            self._schedule_push()

        reward = self._reward(action, torque_sq)
        g = self._gravity_body()
        fallen = (self.data.qpos[2] < 0.6) or (g[2] > -0.62)
        truncated = self._step >= self.max_episode_steps
        self._last_action = action
        info = {"fallen": bool(fallen), "cmd_vx": self._cmd_vx, "yaw": self._yaw(),
                "vx": float((self._rot().T @ self.data.qvel[:3])[0])}
        return self._obs(), float(reward), bool(fallen), bool(truncated), info

    def _reward(self, action, torque_sq) -> float:
        R = self._rot()
        v_body = R.T @ self.data.qvel[:3]
        wz = self.data.qvel[5]
        g = self._gravity_body()
        q = self.data.qpos[7:7 + N_JOINTS]
        contacts = self._foot_contacts()

        r_vel = 2.0 * np.exp(-4.0 * (v_body[0] - self._cmd_vx) ** 2)
        r_lat = 0.5 * np.exp(-4.0 * v_body[1] ** 2)
        r_yaw = 0.5 * np.exp(-20.0 * wz ** 2)
        r_head = 0.5 * np.exp(-3.0 * self._yaw() ** 2)
        r_up = 0.5 * float(np.clip(-g[2], 0.0, 1.0))
        r_height = 0.3 * np.exp(-50.0 * (self.data.qpos[2] - self.h_nominal) ** 2)

        # clock-based contact: alternate stance feet when commanded to walk, both down when standing
        ph = self._phase()
        if self._cmd_vx > 0.1:
            want = np.array([ph < 0.55, ph >= 0.5 or ph < 0.05])
        else:
            want = np.array([True, True])
        r_contact = 0.5 * float(np.mean(contacts == want))

        # feet air time: on each touchdown reward the preceding swing duration (target ~0.3 s)
        first_contact = contacts & ~self._prev_contacts
        r_air = 0.0
        if self._cmd_vx > 0.1:
            r_air = float(np.sum(np.clip(self._air_time - 0.25, -0.25, 0.25) * first_contact)) * 4.0
        self._air_time = np.where(contacts, 0.0, self._air_time + self.dt)
        self._prev_contacts = contacts.copy()

        drift = 0.0
        for leg in range(2):
            base = leg * 6
            drift += q[base] ** 2 + q[base + 1] ** 2  # hip yaw / roll
        slip = 0.0
        for f, body in enumerate(self._foot_bodies):
            if contacts[f]:
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
                slip += float(np.linalg.norm(vel[3:5]))

        penalty = (2e-5 * torque_sq + 0.05 * float(np.sum((action - self._last_action) ** 2))
                   + 0.5 * drift + 0.1 * slip)
        return 0.5 + r_vel + r_lat + r_yaw + r_head + r_up + r_height + r_contact + r_air - penalty
