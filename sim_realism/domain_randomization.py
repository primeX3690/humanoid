"""
Domain randomization for sim-to-real: one `sample()` = one plausible "real robot" drawn around the nominal model.

Parameters (ranges are ASSUMED starting points - tighten them as you identify the real robot):
  mass_scale[body]  lognormal   COM offset   uniform +-cm   foot friction   uniform
  joint damping/armature scale  motor torque scale (per joint, weakened motors)   backlash width scale
  command latency  jitter  encoder noise scale   IMU bias   push schedule (random pushes while walking)
`apply_to_bank` is NumPy-only and tested; `apply_to_mujoco` edits a MuJoCo model in place (needs `mujoco`).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class RandomizationConfig:
    mass_sigma: float = 0.08               # lognormal sigma per body (~8 %)
    com_offset_m: float = 0.015            # +-1.5 cm per body
    friction_range: tuple = (0.5, 1.1)
    damping_scale_range: tuple = (0.5, 2.0)
    armature_scale_range: tuple = (0.8, 1.5)
    torque_scale_range: tuple = (0.85, 1.0)    # weaker, never stronger than datasheet
    backlash_scale_range: tuple = (0.5, 3.0)
    cmd_delay_range_s: tuple = (0.001, 0.008)
    cmd_jitter_range_s: tuple = (0.0, 0.002)
    enc_noise_scale_range: tuple = (0.5, 3.0)
    imu_gyro_bias_rad_s: float = 0.01
    imu_acc_bias_m_s2: float = 0.05
    push_prob_per_s: float = 0.15
    push_force_range_n: tuple = (10.0, 60.0)
    push_dur_range_s: tuple = (0.05, 0.2)


def sample(cfg: RandomizationConfig, body_names, joint_names, seed=None) -> dict:
    r = np.random.default_rng(seed)
    u = lambda lo_hi: float(r.uniform(*lo_hi))
    nb, nj = len(body_names), len(joint_names)
    return dict(
        body_names=list(body_names), joint_names=list(joint_names),
        mass_scale=np.exp(cfg.mass_sigma * r.standard_normal(nb)),
        com_offset=r.uniform(-cfg.com_offset_m, cfg.com_offset_m, (nb, 3)),
        friction=u(cfg.friction_range), damping_scale=r.uniform(*cfg.damping_scale_range, nj),
        armature_scale=r.uniform(*cfg.armature_scale_range, nj), torque_scale=r.uniform(*cfg.torque_scale_range, nj),
        backlash_scale=u(cfg.backlash_scale_range), cmd_delay_s=u(cfg.cmd_delay_range_s),
        cmd_jitter_s=u(cfg.cmd_jitter_range_s), enc_noise_scale=u(cfg.enc_noise_scale_range),
        gyro_bias=r.uniform(-cfg.imu_gyro_bias_rad_s, cfg.imu_gyro_bias_rad_s, 3),
        acc_bias=r.uniform(-cfg.imu_acc_bias_m_s2, cfg.imu_acc_bias_m_s2, 3),
        seed=int(r.integers(2**31)),
    )


def push_schedule(cfg: RandomizationConfig, duration_s, seed=0):
    """List of (t_start, duration, force_xy[2]) random pushes (Poisson arrivals)."""
    r = np.random.default_rng(seed)
    t, out = 0.0, []
    while True:
        t += r.exponential(1.0 / max(cfg.push_prob_per_s, 1e-9))
        if t >= duration_s: return out
        ang = r.uniform(0, 2 * np.pi); F = r.uniform(*cfg.push_force_range_n)
        out.append((float(t), float(r.uniform(*cfg.push_dur_range_s)), np.array([F * np.cos(ang), F * np.sin(ang)])))


def apply_to_bank(specs, sample_dict, dt=0.001):
    """Build an ActuatorBank (see actuators/joint_actuator.py) that realises this sample."""
    from dataclasses import replace
    from actuators.joint_actuator import ActuatorBank
    sp = [replace(s, backlash_rad=s.backlash_rad * sample_dict["backlash_scale"],
                  enc_noise_rad=s.enc_noise_rad * sample_dict["enc_noise_scale"]) for s in specs]
    return ActuatorBank(sp, dt=dt, cmd_delay_s=sample_dict["cmd_delay_s"], cmd_jitter_s=sample_dict["cmd_jitter_s"],
                        seed=sample_dict["seed"], torque_scale=sample_dict["torque_scale"])


class MuJoCoBaseline:
    """Snapshot of the nominal model so repeated randomization never compounds."""
    def __init__(self, m):
        self.mass = m.body_mass.copy(); self.inertia = m.body_inertia.copy(); self.ipos = m.body_ipos.copy()
        self.fric = m.geom_friction.copy(); self.damp = m.dof_damping.copy(); self.arm = m.dof_armature.copy()


def apply_to_mujoco(m, base: MuJoCoBaseline, s: dict, foot_geom_prefixes=("L_foot", "R_foot")):
    """Edit MuJoCo model arrays in place from a sample (bodies/joints matched by NAME; unknown names are ignored)."""
    import mujoco
    m.body_mass[:] = base.mass; m.body_inertia[:] = base.inertia; m.body_ipos[:] = base.ipos
    m.geom_friction[:] = base.fric; m.dof_damping[:] = base.damp; m.dof_armature[:] = base.arm
    for k, name in enumerate(s["body_names"]):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if b < 0: continue
        m.body_mass[b] *= s["mass_scale"][k]; m.body_inertia[b] *= s["mass_scale"][k]
        m.body_ipos[b] += s["com_offset"][k]
    for k, name in enumerate(s["joint_names"]):
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0: continue
        a = m.jnt_dofadr[j]
        m.dof_damping[a] *= s["damping_scale"][k]; m.dof_armature[a] *= s["armature_scale"][k]
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if nm.startswith(foot_geom_prefixes):
            m.geom_friction[g, 0] = s["friction"]
    mujoco.mj_setConst(m, mujoco.MjData(m))
