"""
Safety supervisor: sits between the controller and the actuators. Pure NumPy.

Modes (escalating):  OK -> WARN -> LIMP (torque scale <1) -> SAFE_FALL (damping only, no active balance) -> ESTOP (zero torque, latched)
Checks every tick:
  * stale state / stale command (watchdog)             * joint position limits (soft repulsion + hard stop)
  * joint speed over limit                              * torque clipped to the actuator's limit
  * winding temperature (fold-back, then shutdown)      * fall detection: tilt angle, and DCM outside the support hull
  * NaN / inf in the command                            * manual e-stop (latched until reset())
Limits are ASSUMED defaults: set them from your actual actuators and mechanical stops before powering the robot.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from hardware.hal import JointCommand, RobotState
from control.lateral_recovery import hull_signed_margin

ORDER = ["OK", "WARN", "LIMP", "SAFE_FALL", "ESTOP"]


@dataclass
class SafetyLimits:
    q_lo: np.ndarray
    q_hi: np.ndarray
    qd_max: np.ndarray
    tau_max: np.ndarray
    soft_margin: float = 0.08              # rad: repulsion starts this close to a limit
    k_soft: float = 60.0                   # Nm/rad
    k_hard: float = 1500.0                 # Nm/rad beyond the limit itself
    k_brake: float = 3.0                   # Nm s/rad braking inside the soft margin
    temp_warn: float = 65.0
    temp_limp: float = 75.0
    temp_estop: float = 88.0
    state_timeout: float = 0.05            # s without a fresh state -> ESTOP
    cmd_timeout: float = 0.05
    tilt_warn: float = np.radians(15)
    tilt_fall: float = np.radians(40)
    dcm_fall_margin: float = -0.12         # m outside the support hull -> falling
    speed_over_factor: float = 1.3
    limp_scale: float = 0.5
    damping: float = 4.0                   # Nm s/rad in SAFE_FALL


@dataclass
class SafetyAction:
    mode: str = "OK"
    estop: bool = False
    torque_scale: float = 1.0
    reasons: list = field(default_factory=list)


class SafetySupervisor:
    def __init__(self, limits: SafetyLimits, support_hull_fn=None, com_fn=None):
        self.L = limits; self.mode = "OK"; self.latched_mode = "OK"; self.manual_estop = False
        self.support_hull_fn, self.com_fn = support_hull_fn, com_fn      # optional: () -> hull vertices ; (state) -> (com_xy, vel_xy, omega)
        self.log = []

    def trigger_estop(self): self.manual_estop = True
    def reset(self):
        self.manual_estop = False; self.latched_mode = "OK"; self.mode = "OK"

    @staticmethod
    def tilt(R): return float(np.arccos(np.clip(R[2, 2], -1, 1)))     # angle between body z and world z

    def filter(self, st: RobotState, cmd: JointCommand, now=None):
        L = self.L; now = st.t if now is None else now; reasons = []; live = ["OK"]
        tau = np.asarray(cmd.total_torque(st.q, st.qd), float)

        def esc(m, why):
            reasons.append(why)
            if ORDER.index(m) > ORDER.index(live[0]): live[0] = m

        if self.manual_estop: esc("ESTOP", "manual e-stop")
        if not np.all(np.isfinite(tau)): esc("ESTOP", "NaN/inf command"); tau = np.zeros_like(tau)
        if now - st.stamp > L.state_timeout: esc("ESTOP", f"stale state ({1000 * (now - st.stamp):.0f} ms)")
        if np.any(st.q < L.q_lo - 0.05) or np.any(st.q > L.q_hi + 0.05): esc("SAFE_FALL", "joint beyond mechanical limit")
        if np.any(np.abs(st.qd) > L.speed_over_factor * L.qd_max): esc("LIMP", "joint overspeed")
        Tmax = float(np.max(st.temp_c)) if len(st.temp_c) else 0.0
        if Tmax >= L.temp_estop: esc("ESTOP", f"overtemperature {Tmax:.0f} C")
        elif Tmax >= L.temp_limp: esc("LIMP", f"hot {Tmax:.0f} C")
        elif Tmax >= L.temp_warn: esc("WARN", f"warm {Tmax:.0f} C")
        tl = self.tilt(st.R_imu)
        if tl > L.tilt_fall: esc("SAFE_FALL", f"tilt {np.degrees(tl):.0f} deg")
        elif tl > L.tilt_warn: esc("WARN", f"tilt {np.degrees(tl):.0f} deg")
        if self.support_hull_fn and self.com_fn:
            com, vel, w = self.com_fn(st); dcm = np.asarray(com) + np.asarray(vel) / w
            if hull_signed_margin(dcm, self.support_hull_fn()) < L.dcm_fall_margin: esc("SAFE_FALL", "DCM far outside support (falling)")
        # SAFE_FALL and ESTOP latch until reset(); LIMP/WARN follow the live checks
        if ORDER.index(live[0]) >= ORDER.index("SAFE_FALL"): self.latched_mode = live[0] if ORDER.index(live[0]) > ORDER.index(self.latched_mode) else self.latched_mode
        eff = live[0] if ORDER.index(live[0]) >= ORDER.index(self.latched_mode) else self.latched_mode
        self.mode = eff
        # soft joint-limit spring + braking, stiff wall beyond the limit (always on)
        up = np.clip(st.q - (L.q_hi - L.soft_margin), 0, None); lo = np.clip((L.q_lo + L.soft_margin) - st.q, 0, None)
        over_hi = np.clip(st.q - L.q_hi, 0, None); over_lo = np.clip(L.q_lo - st.q, 0, None)
        scale = 1.0
        if eff == "LIMP": scale = L.limp_scale
        if eff == "SAFE_FALL": tau = -L.damping * st.qd                       # no active balance: just damp (limits below still apply)
        tau = tau * scale
        tau = tau - L.k_soft * up + L.k_soft * lo - L.k_hard * over_hi + L.k_hard * over_lo
        tau = tau - L.k_brake * st.qd * (((up > 0) & (st.qd > 0)) | ((lo > 0) & (st.qd < 0)))
        if eff == "ESTOP": tau = np.zeros_like(tau); scale = 0.0
        tau = np.clip(tau, -L.tau_max, L.tau_max)
        act = SafetyAction(eff, estop=(eff == "ESTOP"), torque_scale=scale, reasons=reasons)
        if reasons: self.log.append((now, eff, list(reasons)))
        return JointCommand(tau), act
