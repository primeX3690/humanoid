"""
Actuator-in-the-loop model: what a REAL joint does between "the controller asked for tau" and "the link felt tau".

Pipeline per joint, per sim tick (vectorised over all joints):
  command -> bus/compute latency (DelayLine) -> current-loop lag (1st order, tau_e)
          -> torque-speed envelope: tau_peak up to corner speed (ASSUMED 50 % of no-load), linear droop to 0 at w_noload (motoring side)
          -> thermal derating (winding temperature from I^2R loss; linear fold-back T_derate..T_max)
          -> friction (Coulomb + viscous) subtracted at the output
  + encoder model: motor-side backlash offset, quantisation, noise.

EVERY NUMBER marked ASSUMED below is a placeholder that must be replaced by a measurement on your real actuator
(thermal R/C especially). Datasheet-sourced values: PH54-200 continuous torque 44.7 Nm and 33.1 rpm no-load (Robotis,
24 V; the stall column is blank, so peak defaults to the continuous figure - pass peak_torque_nm to override),
PH54-100 25.3 Nm / 33.3 rpm, MX-106 8.4 Nm stall (continuous = 20 % per Robotis) / 45 rpm.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import numpy as np
from sim_realism.latency import DelayLine

RPM = 2 * np.pi / 60.0


@dataclass(frozen=True)
class ActuatorSpec:
    name: str
    tau_cont_nm: float
    tau_peak_nm: float
    w_noload_rad_s: float
    mass_kg: float
    tau_electrical_s: float = 0.0115        # current-loop time constant (ASSUMED: value used by actuator_dynamics.py)
    coulomb_nm: float = 0.0                 # ASSUMED fraction below
    viscous_nm_s: float = 0.0
    backlash_rad: float = np.deg2rad(0.3)   # ASSUMED (gearbox_backlash.py default)
    enc_resolution_rad: float = 2 * np.pi / 501923.0   # PRO-series 501,923 pulses/rev (output side)
    enc_noise_rad: float = 1e-5
    # thermal (ASSUMED - measure on hardware)
    t_amb_c: float = 25.0
    t_derate_c: float = 70.0
    t_max_c: float = 90.0
    thermal_tau_s: float = 240.0            # R_th*C_th
    corner_speed_frac: float = 0.5          # ASSUMED: full torque up to this fraction of no-load speed, linear droop to 0 beyond
    hypothetical: bool = False

    def __post_init__(self):
        object.__setattr__(self, "coulomb_nm", self.coulomb_nm or 0.01 * self.tau_cont_nm)
        object.__setattr__(self, "viscous_nm_s", self.viscous_nm_s or 0.005 * self.tau_cont_nm / max(self.w_noload_rad_s, 1e-6) * 10)

    @property
    def loss_cont_w(self):
        """Winding loss at continuous torque (ASSUMED 15 % of continuous mechanical power at half no-load speed)."""
        return max(0.15 * self.tau_cont_nm * 0.5 * self.w_noload_rad_s, 1.0)


PH54_200 = ActuatorSpec("Dynamixel PH54-200-S500-R", 44.7, 44.7, 33.1 * RPM, 0.855)
PH54_100 = ActuatorSpec("Dynamixel PH54-100-S500-R (frame mass ASSUMED = PH54-200)", 25.3, 25.3, 33.3 * RPM, 0.855)
MX106 = ActuatorSpec("Dynamixel MX-106", 8.4 * 0.2, 8.4, 45 * RPM, 0.153, thermal_tau_s=120.0,
                     enc_resolution_rad=2 * np.pi / 4096.0)
QDD_REFERENCE = ActuatorSpec("HYPOTHETICAL quasi-direct-drive (60 Nm cont / 120 peak / 12 rad/s)", 60.0, 120.0, 12.0, 1.4,
                             tau_electrical_s=0.002, backlash_rad=np.deg2rad(0.05), hypothetical=True)


def envelope_torque(spec: ActuatorSpec, speed_rad_s: float) -> float:
    """Peak torque available at a given joint speed (motoring side)."""
    f = np.clip((1.0 - abs(speed_rad_s) / spec.w_noload_rad_s) / (1.0 - spec.corner_speed_frac), 0.0, 1.0)
    return float(spec.tau_peak_nm * f)


def ideal(spec: ActuatorSpec) -> ActuatorSpec:
    """Same limits, but no lag/friction/backlash/thermal - used to prove the layer reduces to the old clipped-torque sim."""
    return replace(spec, tau_electrical_s=0.0, coulomb_nm=1e-12, viscous_nm_s=1e-12, backlash_rad=0.0,
                   enc_resolution_rad=0.0, enc_noise_rad=0.0, thermal_tau_s=1e12, tau_peak_nm=spec.tau_peak_nm)


class ActuatorBank:
    """Vectorised actuators for a whole robot. step(tau_cmd, qd) -> torque actually applied to the joints."""

    def __init__(self, specs, dt=0.001, cmd_delay_s=0.0, cmd_jitter_s=0.0, enc_delay_s=0.0, seed=0,
                 torque_scale=None, thermal=True, speed_limit=True):
        self.specs = list(specs); n = self.n = len(self.specs); self.dt = dt
        f = lambda a: np.array([getattr(s, a) for s in self.specs], float)
        self.tau_cont, self.tau_peak, self.w_nl = f("tau_cont_nm"), f("tau_peak_nm"), f("w_noload_rad_s")
        self.tau_e, self.fc, self.fv = f("tau_electrical_s"), f("coulomb_nm"), f("viscous_nm_s")
        self.backlash, self.enc_res, self.enc_noise = f("backlash_rad"), f("enc_resolution_rad"), f("enc_noise_rad")
        self.t_amb, self.t_der, self.t_max, self.th_tau = f("t_amb_c"), f("t_derate_c"), f("t_max_c"), f("thermal_tau_s")
        self.loss_cont = f("loss_cont_w")
        self.corner = f("corner_speed_frac")
        self.mass = f("mass_kg")
        self.torque_scale = np.ones(n) if torque_scale is None else np.asarray(torque_scale, float)
        self.thermal_on, self.speed_on = thermal, speed_limit
        self.rng = np.random.default_rng(seed)
        self.cmd_line = DelayLine(n, dt, cmd_delay_s, cmd_jitter_s, seed + 1)
        self.enc_line = DelayLine(n, dt, enc_delay_s, 0.0, seed + 2)
        self.reset()

    def reset(self):
        self.tau_i = np.zeros(self.n)          # torque after current-loop lag
        self.temp = self.t_amb.copy()
        self.off = np.zeros(self.n)            # backlash offset state (motor side vs load side)
        self.tau_out = np.zeros(self.n)
        self.power_w = np.zeros(self.n)
        self.cmd_line.reset(); self.enc_line.reset()
        self.saturated = np.zeros(self.n, bool)

    def derate(self):
        d = np.clip((self.t_max - self.temp) / np.maximum(self.t_max - self.t_der, 1e-9), 0.0, 1.0)
        return d if self.thermal_on else np.ones(self.n)

    def limit(self, qd):
        """Instantaneous torque limit (positive number) when torque and speed have the SAME sign (motoring)."""
        speed_fac = np.clip((1.0 - np.abs(qd) / self.w_nl) / (1.0 - self.corner), 0.0, 1.0) if self.speed_on else np.ones(self.n)
        return self.tau_peak * self.torque_scale * speed_fac * self.derate()

    def step(self, tau_cmd, qd):
        tau_cmd = self.cmd_line.step(tau_cmd)
        a = np.where(self.tau_e > 0, 1.0 - np.exp(-self.dt / np.maximum(self.tau_e, 1e-12)), 1.0)
        self.tau_i += a * (tau_cmd - self.tau_i)
        lim_motor = self.limit(qd)
        lim_regen = self.tau_peak * self.torque_scale * self.derate()
        motoring = self.tau_i * qd > 0
        lim = np.where(motoring, lim_motor, lim_regen)
        tau_m = np.clip(self.tau_i, -lim, lim)
        self.saturated = np.abs(self.tau_i) > lim + 1e-9
        fric = self.fc * np.tanh(qd / 0.02) + self.fv * qd
        self.tau_out = tau_m - fric
        # thermal: I^2 R loss ~ (tau/tau_cont)^2 * loss_cont ; steady state at tau_cont sits exactly at T_derate
        p_loss = self.loss_cont * (tau_m / self.tau_cont) ** 2
        r_th = (self.t_der - self.t_amb) / self.loss_cont
        dT = (p_loss * r_th - (self.temp - self.t_amb)) / self.th_tau
        self.temp += self.dt * dT
        self.power_w = p_loss + np.maximum(tau_m * qd, 0.0)         # no regeneration credit (conservative)
        # backlash: tooth-gap side follows the sign of the transmitted torque (first-order approximation)
        tgt = 0.5 * self.backlash * np.tanh(tau_m / (0.02 * self.tau_cont + 1e-9))
        self.off += np.clip(tgt - self.off, -np.abs(qd) * self.dt - 1e-6 * self.backlash, np.abs(qd) * self.dt + 1e-6 * self.backlash) \
            if self.backlash.any() else 0.0
        return self.tau_out.copy()

    def read_encoder(self, q_load):
        """What the controller sees: motor-side position (backlash offset), quantised, noisy, delayed."""
        q = np.asarray(q_load, float) + self.off + self.enc_noise * self.rng.standard_normal(self.n)
        res = np.where(self.enc_res > 0, self.enc_res, 1.0)
        q = np.where(self.enc_res > 0, np.round(q / res) * res, q)
        return self.enc_line.step(q)

    def report(self):
        return dict(temp_c=self.temp.copy(), derate=self.derate(), tau_applied=self.tau_out.copy(),
                    saturated=self.saturated.copy(), electrical_power_w=float(self.power_w.sum()))


def spec_for_joint(name: str) -> ActuatorSpec:
    """Actuator choice per joint of the 30-DoF humanoid (same assignment as model.humanoid_model.repo_actuator_params)."""
    b = name[2:] if name[:2] in ("L_", "R_") else name
    if b in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll") or b.startswith("waist"):
        return PH54_200
    if b in ("sh_pitch", "sh_roll", "sh_yaw", "elbow"):
        return PH54_100
    return MX106


def humanoid_bank(names, dt=0.001, **kw) -> ActuatorBank:
    return ActuatorBank([spec_for_joint(n) for n in names], dt=dt, **kw)
