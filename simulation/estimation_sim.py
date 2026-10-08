"""Ground-truth MuJoCo excitation (squat + waist twist + arm swing) with noisy IMU/encoders -> ESKF vs IMU dead-reckoning."""
import numpy as np, mujoco
from model.humanoid_model import make_model, stand_pose, joint_qadr, joint_dadr, ACTUATED, nominal_q_act
from estimation.base_eskf import BaseESKF
from estimation.so3 import quat_to_R, log_so3, skew, exp_so3


class SensorModel:
    def __init__(self, seed=0, sig_acc=0.038, sig_gyro=0.0055, bias_acc=0.05, bias_gyro=0.01, sig_enc=0.001, sig_encvel=0.03):
        r = np.random.default_rng(seed)
        self.r = r
        self.sig_acc, self.sig_gyro, self.sig_enc, self.sig_encvel = sig_acc, sig_gyro, sig_enc, sig_encvel
        self.ba = r.normal(0, bias_acc, 3); self.bg = r.normal(0, bias_gyro, 3)

    def imu(self, acc, gyro):
        return (acc + self.ba + self.r.normal(0, self.sig_acc, 3), gyro + self.bg + self.r.normal(0, self.sig_gyro, 3))

    def enc(self, q, qd):
        return q + self.r.normal(0, self.sig_enc, q.shape), qd + self.r.normal(0, self.sig_encvel, qd.shape)


class FootKinematics:
    """Encoder-based forward kinematics: foot-sole position in the IMU/pelvis frame."""
    def __init__(self):
        self.m, self.d = make_model(table=False)
        self.qa = joint_qadr(self.m, ACTUATED)
        self.da = joint_dadr(self.m, ACTUATED)
        self.sid = {s: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_sole_site") for s in "LR"}
        self.imu = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "imu")

    def foot_vel_in_base(self, q_act, qd_act):
        """d/dt of foot position relative to the IMU, expressed in the base frame (encoder kinematics only)."""
        self.d.qpos[:] = 0; self.d.qpos[3] = 1.0
        self.d.qpos[self.qa] = q_act
        mujoco.mj_kinematics(self.m, self.d); mujoco.mj_comPos(self.m, self.d)
        out = {}
        for s in "LR":
            jp = np.zeros((3, self.m.nv)); mujoco.mj_jacSite(self.m, self.d, jp, None, self.sid[s])
            jc = np.zeros((3, self.m.nv)); mujoco.mj_jacSite(self.m, self.d, jc, None, self.imu)
            out[s] = (jp - jc)[:, self.da] @ qd_act
        return out

    def foot_in_base(self, q_act):
        self.d.qpos[:] = 0; self.d.qpos[3] = 1.0
        self.d.qpos[self.qa] = q_act
        mujoco.mj_kinematics(self.m, self.d)
        return {s: self.d.site_xpos[self.sid[s]] - self.d.site_xpos[self.imu] for s in "LR"}


def truth(m, d):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "imu")
    p = d.site_xpos[sid].copy(); R = d.site_xmat[sid].reshape(3, 3).copy()
    vel6 = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, vel6, 0)  # world-aligned
    return p, R, vel6[3:].copy(), vel6[:3].copy()


def run_estimation_experiment(T=8.0, seed=0, use_contact=True, verbose=False):
    m, d = make_model(table=False)
    q_nom = stand_pose(m, d)
    qa, da = joint_qadr(m, ACTUATED), joint_dadr(m, ACTUATED)
    tau_max = np.array([m.actuator_forcerange[i, 1] for i in range(30)])
    sens = SensorModel(seed)
    fk = FootKinematics()
    names = ACTUATED
    ix = {n: names.index(n) for n in names}
    # settle
    for _ in range(1500):                      # settle 1.5 s so the accelerometer-based initial tilt is taken at rest
        q = d.qpos[qa]; qd = d.qvel[da]
        d.ctrl[:30] = np.clip(300 * (q_nom - q) - 30 * qd + d.qfrc_bias[da], -tau_max, tau_max); d.ctrl[30:] = 0.004
        mujoco.mj_step(m, d)
    p_t, R_t, v_t, w_t = truth(m, d)
    # init: static alignment (as at robot power-up while standing supported): truth attitude + ~0.2 deg roll/pitch error.
    # (An accelerometer-only tilt init was used before; it is only valid if the robot is perfectly at rest.)
    R0 = R_t @ exp_so3(np.array([sens.r.normal(0, 0.0035), sens.r.normal(0, 0.0035), 0.0]))
    q0e, qd0e = sens.enc(d.qpos[qa], d.qvel[da])
    feet0 = fk.foot_in_base(q0e)
    pf0 = {k: p_t + R0 @ feet0[k] for k in "LR"}
    est = BaseESKF(R0, p_t, pf0)
    # IMU dead-reckoning baseline (identical init, no corrections)
    dr_p, dr_v, dr_R, dr_ba, dr_bg = p_t.copy(), np.zeros(3), R0.copy(), np.zeros(3), np.zeros(3)
    dt = m.opt.timestep
    n = int(T / dt)
    rec = {k: [] for k in ["t", "ez_est", "ep_est", "ev_est", "etilt_est", "eyaw_est", "ez_dr", "ep_dr", "ev_dr", "etilt_dr",
                           "sig_v", "sig_tilt", "ev_vec", "etilt_vec"]}
    for i in range(n):
        t = i * dt
        s = 0.12 * np.sin(2 * np.pi * t / 2.5)
        q_des = q_nom.copy()
        q_des[ix["L_hip_pitch"]] += -s; q_des[ix["R_hip_pitch"]] += -s
        q_des[ix["L_knee"]] += 2 * s; q_des[ix["R_knee"]] += 2 * s
        q_des[ix["L_ankle_pitch"]] += -s; q_des[ix["R_ankle_pitch"]] += -s
        q_des[ix["waist_yaw"]] = 0.35 * np.sin(2 * np.pi * t / 1.7)
        q_des[ix["R_sh_pitch"]] = -0.5 * np.sin(2 * np.pi * t / 2.1)
        q_des[ix["L_sh_pitch"]] = 0.5 * np.sin(2 * np.pi * t / 2.1)
        q = d.qpos[qa]; qd = d.qvel[da]
        d.ctrl[:30] = np.clip(300 * (q_des - q) - 30 * qd + d.qfrc_bias[da], -tau_max, tau_max); d.ctrl[30:] = 0.004
        mujoco.mj_step(m, d)
        acc_m, gyr_m = sens.imu(d.sensordata[0:3], d.sensordata[3:6])
        contact = {"L": True, "R": True}
        est.predict(acc_m, gyr_m, dt, contact)
        # dead reckoning
        a = acc_m
        Racc = dr_R @ a + BaseESKF.G
        dr_p = dr_p + dr_v * dt + 0.5 * Racc * dt * dt; dr_v = dr_v + Racc * dt; dr_R = dr_R @ exp_so3(gyr_m * dt)
        if i % 5 == 0:         # encoders @ 200 Hz
            qe, _ = sens.enc(d.qpos[qa], d.qvel[da])
            if use_contact:
                est.update_foot_kinematics(fk.foot_in_base(qe), contact)
                est.update_gravity(acc_m, 8.0)   # weak: squat accelerations (~0.5 m/s^2) would bias a strong tilt aid
        if i % 20 == 0:
            p_t, R_t, v_t, _ = truth(m, d)
            rec["t"].append(t)
            rec["ez_est"].append(est.p[2] - p_t[2]); rec["ep_est"].append(np.linalg.norm(est.p - p_t))
            rec["ev_est"].append(np.linalg.norm(est.v - v_t))
            et = log_so3(R_t.T @ est.R); rec["etilt_est"].append(np.linalg.norm(et[:2])); rec["eyaw_est"].append(et[2])
            rec["ez_dr"].append(dr_p[2] - p_t[2]); rec["ep_dr"].append(np.linalg.norm(dr_p - p_t))
            rec["ev_dr"].append(np.linalg.norm(dr_v - v_t))
            rec["etilt_dr"].append(np.linalg.norm(log_so3(R_t.T @ dr_R)[:2]))
            rec["sig_v"].append(np.sqrt(np.diag(est.P)[3:6])); rec["sig_tilt"].append(np.sqrt(np.diag(est.P)[6:8]))
            rec["ev_vec"].append(est.v - v_t); rec["etilt_vec"].append(et[:2])
        if d.qpos[2] < 0.5:
            break
    return {k: np.array(v) for k, v in rec.items()}, est, sens


if __name__ == "__main__":
    rec, est, sens = run_estimation_experiment()
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    print("ESKF   : height err RMS %.2f mm | vel err RMS %.3f m/s | tilt err RMS %.3f deg | pos err (last) %.1f mm" % (
        1e3 * rms(rec["ez_est"]), rms(rec["ev_est"]), np.degrees(rms(rec["etilt_est"])), 1e3 * rec["ep_est"][-1]))
    print("IMU DR : height err RMS %.2f mm | vel err RMS %.3f m/s | tilt err RMS %.3f deg | pos err (last) %.1f mm" % (
        1e3 * rms(rec["ez_dr"]), rms(rec["ev_dr"]), np.degrees(rms(rec["etilt_dr"])), 1e3 * rec["ep_dr"][-1]))
    sv = rec["sig_v"]; ev = rec["ev_vec"]
    print("vel 3-sigma consistency: %.1f%% of samples inside 3 sigma" % (100 * np.mean(np.abs(ev) < 3 * sv)))
    st = rec["sig_tilt"]; et = rec["etilt_vec"]
    print("tilt 3-sigma consistency: %.1f%%" % (100 * np.mean(np.abs(et) < 3 * st)))
    print("true gyro bias", np.round(sens.bg, 4), "est", np.round(est.bg, 4), "| true acc bias", np.round(sens.ba, 3), "est", np.round(est.ba, 3))
    print("yaw drift at end (deg): %.3f" % np.degrees(rec["eyaw_est"][-1]))
