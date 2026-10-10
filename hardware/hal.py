"""
Hardware abstraction layer: ONE interface for the simulator and the real robot, so controllers are written once.

    RobotInterface.read() -> RobotState      RobotInterface.write(JointCommand)      RobotInterface.estop()
Implementations:
    PlantInterface     - NumPy per-joint plant behind an ActuatorBank (tests, tuning latency/thermal without MuJoCo)
    MuJoCoInterface    - wraps simulation.wbc_sim.WBCSim (needs MuJoCo; NOT exercised in the offline sandbox this was written in)
    DynamixelInterface - skeleton over hardware.dynamixel_protocol2 (needs YOUR control-table addresses + a serial transport)
ControlLoop runs read -> controller -> safety -> write at a fixed rate and records deadline misses.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import numpy as np


@dataclass
class RobotState:
    t: float
    q: np.ndarray
    qd: np.ndarray
    tau: np.ndarray                      # measured / estimated joint torque
    temp_c: np.ndarray
    R_imu: np.ndarray = field(default_factory=lambda: np.eye(3))
    gyro: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.array([0, 0, 9.81]))
    wrench: dict = field(default_factory=dict)          # {"L": 6-vector, "R": 6-vector} foot F/T in the sole frame
    stamp: float = 0.0                                   # when the sample was taken (for the watchdog)


@dataclass
class JointCommand:
    tau: np.ndarray                                      # feed-forward torque
    q_des: np.ndarray | None = None                      # optional joint PD on top (position-mode servos)
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None

    def total_torque(self, q, qd):
        t = np.asarray(self.tau, float).copy()
        if self.q_des is not None and self.kp is not None:
            t += self.kp * (self.q_des - q)
        if self.kd is not None:
            t -= self.kd * qd
        return t


class RobotInterface(ABC):
    names: list
    dt: float

    @abstractmethod
    def read(self) -> RobotState: ...
    @abstractmethod
    def write(self, cmd: JointCommand) -> None: ...
    def estop(self) -> None: self.write(JointCommand(np.zeros(len(self.names))))
    def close(self) -> None: pass


class PlantInterface(RobotInterface):
    """Independent joints: J qdd = tau_actuator - b qd - gravity_like(q). Enough to test loops, latency, thermal and safety."""

    def __init__(self, names, bank, inertia=1.0, damping=0.5, grav=None, dt=0.001, q0=None):
        self.names, self.bank, self.dt = list(names), bank, dt
        n = len(self.names); self.J = np.broadcast_to(np.asarray(inertia, float), (n,)).copy(); self.b = np.broadcast_to(np.asarray(damping, float), (n,)).copy()
        self.grav = np.zeros(n) if grav is None else np.asarray(grav, float)
        self.q = np.zeros(n) if q0 is None else np.asarray(q0, float).copy(); self.qd = np.zeros(n); self.t = 0.0; self.tau_applied = np.zeros(n)
        self._cmd = JointCommand(np.zeros(n))

    def write(self, cmd): self._cmd = cmd

    def step(self):
        enc = self.bank.read_encoder(self.q)
        tau = self._cmd.total_torque(enc, self.qd)
        self.tau_applied = self.bank.step(tau, self.qd)
        qdd = (self.tau_applied - self.b * self.qd - self.grav * np.sin(self.q)) / self.J
        self.qd += qdd * self.dt; self.q += self.qd * self.dt; self.t += self.dt

    def read(self):
        return RobotState(self.t, self.bank.read_encoder(self.q), self.qd.copy(), self.tau_applied.copy(), self.bank.temp.copy(), stamp=self.t)


class MuJoCoInterface(RobotInterface):
    """Adapter over WBCSim (MuJoCo). Torque commands go through S.realism if one is attached."""

    def __init__(self, S):
        from model.humanoid_model import ACTUATED
        self.S, self.names, self.dt = S, list(ACTUATED), float(S.m.opt.timestep)

    def read(self):
        S = self.S; q, qd = S.d.qpos[S.qadr].copy(), S.d.qvel[S.dadr].copy()
        return RobotState(float(S.d.time), q, qd, S.tau.copy(), np.full(len(self.names), 25.0), stamp=float(S.d.time))

    def write(self, cmd):
        S = self.S; S.tau = cmd.total_torque(S.d.qpos[S.qadr], S.d.qvel[S.dadr])


class DynamixelInterface(RobotInterface):
    """Skeleton: fill `control_table` from the e-manual of YOUR actuators and pass a transport with .xfer(bytes)->bytes.
    Torque-mode writes use sync-write; state uses sync-read. See hardware/dynamixel_protocol2.py for the packet layer."""

    def __init__(self, bus, ids, names, control_table, dt=0.004):
        self.bus, self.ids, self.names, self.table, self.dt = bus, list(ids), list(names), control_table, dt
        missing = [k for k, v in control_table.items() if v is None]
        if missing: raise ValueError(f"control table entries not set (look them up in the e-manual): {missing}")

    def read(self):                                   # pragma: no cover - needs hardware
        raise NotImplementedError("implement with bus.sync_read(self.ids, addr, length) using your control table")

    def write(self, cmd):                             # pragma: no cover
        raise NotImplementedError("implement with bus.sync_write(self.ids, addr, data) using your control table")


@dataclass
class LoopStats:
    ticks: int = 0
    deadline_misses: int = 0
    worst_ms: float = 0.0
    safety_interventions: int = 0


class ControlLoop:
    def __init__(self, iface: RobotInterface, controller, supervisor=None, rate_hz=250, realtime=False):
        self.iface, self.ctrl, self.sup, self.period, self.realtime = iface, controller, supervisor, 1.0 / rate_hz, realtime
        self.stats = LoopStats()

    def tick(self, step_plant=None):
        t0 = time.perf_counter()
        st = self.iface.read()
        cmd = self.ctrl(st)
        if self.sup is not None:
            cmd, act = self.sup.filter(st, cmd)
            if act.mode != "OK": self.stats.safety_interventions += 1
            if act.estop: self.iface.estop(); self.stats.ticks += 1; return st, cmd
        self.iface.write(cmd)
        if step_plant: step_plant()
        dt = time.perf_counter() - t0
        self.stats.ticks += 1; self.stats.worst_ms = max(self.stats.worst_ms, 1000 * dt)
        if dt > self.period: self.stats.deadline_misses += 1
        if self.realtime and dt < self.period: time.sleep(self.period - dt)
        return st, cmd
