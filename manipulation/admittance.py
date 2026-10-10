"""
Contact-aware hand control: Cartesian admittance (force -> motion) and a hybrid position/force regulator.

v2 had no force feedback at all (open-loop gripper close + lift check). With a wrist force/torque sensor (or the WBC's estimated
contact wrench) the hand can: comply with unexpected contact, press with a commanded force, slide along a surface.
Output is a *reference pose offset* meant to be fed to `WholeBodyController.task_hand` (the WBC already does the torque-level work).
"""
from __future__ import annotations
import numpy as np


class Admittance:
    """M x'' + D x' + K x = F_ext  (per Cartesian axis, diagonal), integrated semi-implicitly. x = offset of the reference from nominal."""

    def __init__(self, M=2.0, D=60.0, K=200.0, dt=0.005, x_max=0.08):
        f = lambda v: np.broadcast_to(np.asarray(v, float), (3,)).copy()
        self.M, self.D, self.K, self.dt, self.x_max = f(M), f(D), f(K), dt, x_max
        self.x = np.zeros(3); self.v = np.zeros(3)

    def step(self, F_ext):
        a = (np.asarray(F_ext, float) - self.D * self.v - self.K * self.x) / self.M
        self.v += a * self.dt; self.x += self.v * self.dt
        n = np.linalg.norm(self.x)
        if n > self.x_max: self.x *= self.x_max / n; self.v *= 0.0
        return self.x.copy(), self.v.copy()

    def reset(self): self.x[:] = 0; self.v[:] = 0


class ForceRegulator:
    """Along `normal` (pointing OUT of the surface, toward the hand) regulate the contact force to f_des with a PI law on the reference
    position (hybrid position/force control); tangential axes stay position-controlled by the caller.
    F_meas = force the environment exerts ON the hand (world frame). Returns a displacement of the reference (along -normal when
    the force is too low).  Stability needs  kp * k_env < ~1  (k_env = contact stiffness, m/N * N/m): gain-schedule kp on the estimated stiffness."""

    def __init__(self, normal, kp=0.0002, ki=0.003, dt=0.005, x_max=0.05, f_filter_hz=15.0):
        self.n = np.asarray(normal, float) / np.linalg.norm(normal)
        self.kp, self.ki, self.dt, self.x_max = kp, ki, dt, x_max
        self.I = 0.0; self.f_lp = 0.0; self.a = 1 - np.exp(-2 * np.pi * f_filter_hz * dt)

    def step(self, F_meas, f_des):
        fn = float(np.asarray(F_meas, float) @ self.n)               # pressing into the surface = positive
        self.f_lp += self.a * (fn - self.f_lp)
        e = f_des - self.f_lp
        self.I = float(np.clip(self.I + self.ki * e * self.dt, -self.x_max, self.x_max))
        d = float(np.clip(self.kp * e + self.I, -self.x_max, self.x_max))
        return -self.n * d
