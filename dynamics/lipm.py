"""
dynamics/lipm.py

Linear Inverted Pendulum Model (LIPM) - the "cart-table" model that real
biped robots (Honda ASIMO, Kawada HRP-2, and most preview-control-based
humanoids since) use to plan Center-of-Mass (CoM) motion from a desired
Zero-Moment-Point (ZMP) trajectory.

Physics (Kajita et al., "Biped Walking Pattern Generation by using
Preview Control of Zero-Moment Point", ICRA 2003):

    x_zmp = x_com - (zc/g) * x_com_ddot

i.e. the ZMP (the point on the ground where the net ground-reaction
moment is zero - the real physical stability criterion for a biped: as
long as the ZMP stays inside the support polygon formed by the feet,
the robot won't tip over) equals the CoM position minus a term
proportional to the CoM's horizontal acceleration, scaled by the
(assumed-constant) CoM height zc over gravity g.

This module treats jerk (rate of change of CoM acceleration) as the
control input - the standard trick that turns ZMP tracking into a
linear control problem, discretized into the state-space form the
preview controller in control/zmp_preview_controller.py consumes:

    x(k+1) = A x(k) + B u(k)
    p(k)   = C x(k)

with state x = [position, velocity, acceleration] and control u = jerk.

Scope note: this is the sagittal/lateral-plane simplification used by
essentially every preview-control humanoid gait generator - it assumes
constant CoM height and treats the two horizontal axes as decoupled
(run one instance of this model per axis). It does NOT model full 3D
rigid-body dynamics, leg inertia, or joint torques - that is a
deliberately separate, harder layer (see docs/SCOPE.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY = 9.81  # m/s^2


@dataclass
class LIPMParams:
    com_height_m: float = 0.8   # constant CoM height assumption (typical adult-scale biped)
    dt: float = 0.01            # control/sim timestep, seconds (100Hz, a common preview-control rate)
    gravity: float = GRAVITY


class LIPM:
    """One axis (sagittal OR lateral) of the Linear Inverted Pendulum Model,
    discretized with jerk as the control input."""

    def __init__(self, params: LIPMParams):
        self.p = params
        T = params.dt
        self.A = np.array([
            [1.0, T, T * T / 2.0],
            [0.0, 1.0, T],
            [0.0, 0.0, 1.0],
        ])
        self.B = np.array([[T ** 3 / 6.0], [T * T / 2.0], [T]])
        self.C = np.array([[1.0, 0.0, -params.com_height_m / params.gravity]])

    def natural_frequency(self) -> float:
        """omega = sqrt(g / zc) - the LIPM's characteristic frequency; this is
        the textbook closed-form result every LIPM implementation must match
        (used as a correctness check in tests/test_lipm.py, not just eyeballed)."""
        return float(np.sqrt(self.p.gravity / self.p.com_height_m))

    def step(self, x: np.ndarray, jerk: float) -> np.ndarray:
        """Advance the discretized LIPM state by one timestep."""
        x = x.reshape(3, 1)
        u = np.array([[jerk]])
        return (self.A @ x + self.B @ u).flatten()

    def zmp(self, x: np.ndarray) -> float:
        """Compute the ZMP for a given [pos, vel, acc] state."""
        return float((self.C @ x.reshape(3, 1)).item())

    def free_fall_check(self, x0: np.ndarray, n_steps: int) -> np.ndarray:
        """Simulate with zero control input (jerk=0, i.e. constant CoM
        acceleration) - used only to sanity-check the discretization against
        the continuous-time closed-form solution in tests, not part of any
        real gait."""
        traj = np.zeros((n_steps, 3))
        x = x0.copy()
        for k in range(n_steps):
            traj[k] = x
            x = self.step(x, jerk=0.0)
        return traj
