"""Closed-loop state estimate for the controller: contact-aided ESKF (base) + noisy encoders (joints)."""
import numpy as np, mujoco
from model.humanoid_model import joint_qadr, joint_dadr, ACTUATED
from estimation.base_eskf import BaseESKF
from estimation.so3 import R_to_quat, quat_to_R, skew
from simulation.estimation_sim import SensorModel, FootKinematics

IMU_OFFSET = np.array([0.0, 0.0, 0.02])      # imu site in pelvis frame


class EstimatedState:
    def __init__(self, S, seed=0, grav_sig=0.5, **eskf_kw):
        # grav_sig: std (m/s^2) of the accelerometer tilt aid. 0.5 is fine when standing/reaching; while WALKING the CoM sway
        # accelerations (~0.5 m/s^2) bias tilt by ~3 deg and the robot falls -> use ~8 (almost no aid) for gait.
        self.grav_sig = grav_sig
        self.S, self.sens, self.fk = S, SensorModel(seed), FootKinematics()
        m, d = S.m, S.d
        self.qa, self.da = joint_qadr(m, ACTUATED), joint_dadr(m, ACTUATED)
        self.nq_r, self.nv_r = S.nq_r, S.nv_r
        self.dt = m.opt.timestep
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
        p = d.site_xpos[sid].copy(); R = d.site_xmat[sid].reshape(3, 3).copy()
        qe, _ = self.sens.enc(d.qpos[self.qa], d.qvel[self.da])
        feet = self.fk.foot_in_base(qe)
        # Initial attitude = static alignment while standing supported (what a real robot does at power-up):
        # truth attitude + 0.2 deg roll/pitch error. (Accelerometer init is invalid here: at t=0 the sim has not settled.)
        from estimation.so3 import exp_so3
        rng = np.random.default_rng(seed + 100)
        R0 = R @ exp_so3(np.array([rng.normal(0, 0.0035), rng.normal(0, 0.0035), 0.0]))
        self.qe = qe
        self.est = BaseESKF(R0, p, {k: p + R0 @ feet[k] for k in "LR"}, **eskf_kw)
        self.i = 0
        self.qd_lp = np.zeros(len(ACTUATED))
        self.qe = qe                     # latest encoder reading (valid before the first step())

    def step(self, contact=None):
        """contact: {'L': bool, 'R': bool} stance flags (from the gait schedule / foot force sensors). Default: both feet down."""
        d = self.S.d
        acc, gyr = self.sens.imu(d.sensordata[0:3], d.sensordata[3:6])
        c = dict(contact) if contact is not None else {"L": True, "R": True}
        self.est.predict(acc, gyr, self.dt, c)
        qe, qde = self.sens.enc(d.qpos[self.qa], d.qvel[self.da])
        self.qd_lp += 0.3 * (qde - self.qd_lp)
        if self.i % 5 == 0:
            self.est.update_foot_kinematics(self.fk.foot_in_base(qe), c)
            self.est.update_gravity(acc, self.grav_sig)
            hd = self.fk.foot_vel_in_base(qe, self.qd_lp)
            hb = self.fk.foot_in_base(qe)
            for k in [k for k in 'LR' if c[k]]:
                zv = -(self.est.R @ (np.cross(self.est.last_w, hb[k]) + hd[k]))
                self.est.update_velocity(zv, 0.08)
        self.i += 1
        self.qe = qe

    def state(self):
        e, d = self.est, self.S.d
        R = e.R
        p_pelvis = e.p - R @ IMU_OFFSET
        w_b = e.last_w
        v_pelvis = e.v - R @ np.cross(w_b, IMU_OFFSET)
        q = d.qpos[:self.nq_r].copy(); qd = d.qvel[:self.nv_r].copy()
        q[0:3] = p_pelvis; q[3:7] = R_to_quat(R); q[self.qa] = self.qe
        qd[0:3] = v_pelvis; qd[3:6] = w_b; qd[self.da] = self.qd_lp
        return q, qd
