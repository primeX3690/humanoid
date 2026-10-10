"""RealismLayer: plugs the actuator bank + encoder/latency models between a simulator and a controller.

    sim.realism = RealismLayer(humanoid_bank(ACTUATED, cmd_delay_s=0.004, enc_delay_s=0.002))
Every sim tick (1 kHz) the simulator calls  tau_applied = layer.apply(tau_cmd, qd)  and  layer.tick(q_joint);
at control time it reads  q_meas, qd_meas = layer.q_meas, layer.qd_meas  instead of ground truth.
"""
from __future__ import annotations
import numpy as np


class RealismLayer:
    def __init__(self, bank, vel_cutoff_hz=60.0):
        self.bank = bank
        self.dt = bank.dt
        self.a = 1.0 - np.exp(-2 * np.pi * vel_cutoff_hz * self.dt)
        self.q_meas = None
        self.qd_meas = np.zeros(bank.n)
        self._prev = None

    def apply(self, tau_cmd, qd_true):
        return self.bank.step(np.asarray(tau_cmd, float), np.asarray(qd_true, float))

    def tick(self, q_joint_true):
        q = self.bank.read_encoder(q_joint_true)
        if self._prev is None:
            self._prev = q.copy()
            self.q_meas = q
            return
        raw = (q - self._prev) / self.dt
        self.qd_meas += self.a * (raw - self.qd_meas)
        self._prev = q.copy()
        self.q_meas = q

    def reset(self):
        self.bank.reset(); self._prev = None; self.q_meas = None; self.qd_meas[:] = 0

    def report(self):
        return self.bank.report()
