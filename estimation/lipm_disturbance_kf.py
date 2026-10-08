"""
Kalman filter for the LIPM CoM state + unknown external horizontal force (disturbance observer).

State  x = [c, c_dot, d]   (one horizontal axis; d = external force / total mass, random walk)
Model  c_ddot = w^2 (c - z) + d,   w = sqrt(g / z_c),  z = measured ZMP (input from foot force sensors)
Meas   y = c (CoM position from kinematics), noise sigma_c
Use    d_hat feeds the capture-point controller (feed-forward / push detection without oracle knowledge).
"""
import numpy as np


class LIPMDisturbanceKF:
    def __init__(self, zc=0.92, g=9.81, sig_c=0.003, q_d=400.0, q_c=1e-4, q_v=1e-2, c0=0.0):
        self.w2 = g / zc
        self.x = np.array([c0, 0.0, 0.0])
        self.P = np.diag([1e-4, 1e-2, 1.0])
        self.sig_c, self.q_d, self.q_c, self.q_v = sig_c, q_d, q_c, q_v

    def step(self, c_meas, zmp, dt):
        A = np.array([[0, 1, 0], [self.w2, 0, 1], [0, 0, 0]])
        B = np.array([0, -self.w2, 0])
        F = np.eye(3) + A * dt + 0.5 * A @ A * dt * dt
        x = F @ self.x + B * zmp * dt
        Q = np.diag([self.q_c * dt, self.q_v * dt, self.q_d * dt])
        P = F @ self.P @ F.T + Q
        H = np.array([[1.0, 0, 0]])
        S = H @ P @ H.T + self.sig_c ** 2
        K = (P @ H.T / S).ravel()
        x = x + K * (c_meas - x[0])
        P = (np.eye(3) - np.outer(K, H)) @ P
        self.x, self.P = x, P
        return self.x.copy()
