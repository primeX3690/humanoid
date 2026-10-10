"""Fractional-delay line with jitter - models command / sensor / bus latency for vector signals.

Real control stacks never see a measurement the instant it is taken nor apply a command the instant it is computed:
IMU->bus->PC->controller->bus->motor adds a few ms, and the delay jitters. Ignoring this is one of the main reasons
a controller that is perfect in simulation oscillates on hardware.
"""
from __future__ import annotations
import numpy as np


class DelayLine:
    """push(x) every sim tick, read() returns x delayed by `delay_s` (+ jitter), linearly interpolated between ticks."""

    def __init__(self, dim: int, dt: float, delay_s: float = 0.0, jitter_s: float = 0.0, seed: int = 0, init=None):
        self.dim, self.dt = int(dim), float(dt)
        self.delay_s, self.jitter_s = float(delay_s), float(jitter_s)
        self.rng = np.random.default_rng(seed)
        self.n = int(np.ceil((delay_s + 6 * jitter_s) / dt)) + 3     # ring-buffer length (covers +6 sigma)
        self.buf = np.zeros((self.n, self.dim)) if init is None else np.tile(np.asarray(init, float), (self.n, 1))
        self.head = 0              # index of newest sample
        self._d = delay_s

    def push(self, x):
        self.head = (self.head + 1) % self.n
        self.buf[self.head] = np.asarray(x, float)
        if self.jitter_s > 0:
            self._d = float(np.clip(self.delay_s + self.jitter_s * self.rng.standard_normal(), 0.0, (self.n - 3) * self.dt))
        else:
            self._d = self.delay_s

    def read(self):
        k = self._d / self.dt
        i0 = int(np.floor(k)); frac = k - i0
        a = self.buf[(self.head - i0) % self.n]
        b = self.buf[(self.head - i0 - 1) % self.n]
        return (1.0 - frac) * a + frac * b

    def step(self, x):
        self.push(x)
        return self.read()

    def reset(self, value=None):
        self.buf[:] = 0.0 if value is None else np.asarray(value, float)
        self._d = self.delay_s
