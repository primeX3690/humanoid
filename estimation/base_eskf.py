"""
Contact-aided error-state Kalman filter for floating-base state estimation (IMU + joint encoders).

Nominal state : p (3), v (3), R (body->world), b_a (3), b_g (3), foot positions p_fL, p_fR (world, 3 each)
Error state   : [dp, dv, dtheta(body-frame), dba, dbg, dpfL, dpfR]  (21)
Propagation   : IMU (specific force + gyro), 
Measurement   : encoder forward-kinematics foot position in the base frame,  z_i = R^T (p_fi - p) + n
                for every foot flagged "in contact" (feet are modelled as stationary while in contact; their
                world position is itself a state -- Bloesch et al., IROS 2012 / Rotella et al. 2014).
Observability (stated honestly): roll, pitch, velocity, height and all bias terms are observable;
global x, y position and yaw are NOT (no GPS / no heading reference) -- they are held by a tight prior and
will drift slowly; everything the controller needs (tilt, velocity, height above feet) is observable.
"""
import numpy as np
from estimation.so3 import skew, exp_so3, quat_to_R


class BaseESKF:
    G = np.array([0.0, 0.0, -9.81])

    def __init__(self, R0, p0, foot0, sig_a=0.04, sig_g=0.006, sig_ba=2e-4, sig_bg=2e-5, sig_foot=2e-3,
                 sig_meas=0.004, p_prior=1e-3, yaw_prior=1e-3):
        self.p = np.array(p0, float); self.v = np.zeros(3); self.R = np.array(R0, float)
        self.ba = np.zeros(3); self.bg = np.zeros(3)
        self.pf = {k: np.array(v, float) for k, v in foot0.items()}      # {"L":..,"R":..}
        self.keys = ["L", "R"]
        n = 21
        self.P = np.zeros((n, n))
        self.P[0:3, 0:3] = np.eye(3) * p_prior ** 2
        self.P[3:6, 3:6] = np.eye(3) * 0.05 ** 2
        self.P[6:9, 6:9] = np.diag([0.02, 0.02, yaw_prior]) ** 2
        self.P[9:12, 9:12] = np.eye(3) * 0.1 ** 2
        self.P[12:15, 12:15] = np.eye(3) * 0.02 ** 2
        self.P[15:18, 15:18] = np.eye(3) * 0.01 ** 2
        self.P[18:21, 18:21] = np.eye(3) * 0.01 ** 2
        self.sig_a, self.sig_g, self.sig_ba, self.sig_bg, self.sig_foot = sig_a, sig_g, sig_ba, sig_bg, sig_foot
        self.sig_meas = sig_meas
        self.last_w = np.zeros(3)

    def predict(self, a_m, w_m, dt, contact):
        a = a_m - self.ba
        w = w_m - self.bg
        self.last_w = w
        Racc = self.R @ a + self.G
        self.p = self.p + self.v * dt + 0.5 * Racc * dt * dt
        self.v = self.v + Racc * dt
        Rold = self.R
        self.R = self.R @ exp_so3(w * dt)
        n = 21
        F = np.eye(n)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -Rold @ skew(a) * dt
        F[3:6, 9:12] = -Rold * dt
        F[6:9, 6:9] = np.eye(3) - skew(w) * dt
        F[6:9, 12:15] = -np.eye(3) * dt
        Q = np.zeros((n, n))
        Q[3:6, 3:6] = np.eye(3) * (self.sig_a ** 2) * dt
        Q[6:9, 6:9] = np.eye(3) * (self.sig_g ** 2) * dt
        Q[9:12, 9:12] = np.eye(3) * (self.sig_ba ** 2) * dt
        Q[12:15, 12:15] = np.eye(3) * (self.sig_bg ** 2) * dt
        for i, k in enumerate(self.keys):
            sl = slice(15 + 3 * i, 18 + 3 * i)
            # a foot that is NOT in contact is free: inflate its uncertainty so it re-anchors at touchdown
            Q[sl, sl] = np.eye(3) * ((self.sig_foot ** 2) * dt if contact.get(k, True) else 1.0 * dt)
        self.P = F @ self.P @ F.T + Q

    def update_foot_kinematics(self, z_meas, contact):
        """z_meas: {"L": foot pos in base frame (3,), "R": ...} from encoder FK."""
        for i, k in enumerate(self.keys):
            if not contact.get(k, True) or k not in z_meas:
                continue
            hhat = self.R.T @ (self.pf[k] - self.p)
            H = np.zeros((3, 21))
            H[:, 0:3] = -self.R.T
            H[:, 6:9] = skew(hhat)
            H[:, 15 + 3 * i:18 + 3 * i] = self.R.T
            Rn = np.eye(3) * self.sig_meas ** 2
            S = H @ self.P @ H.T + Rn
            K = self.P @ H.T @ np.linalg.inv(S)
            dx = K @ (z_meas[k] - hhat)
            self._inject(dx)
            I_KH = np.eye(21) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ Rn @ K.T

    def update_gravity(self, a_m, sig=0.5):
        """Weak accelerometer tilt aid, used only when the body is (nearly) unaccelerated."""
        g_b = self.R.T @ (-self.G)                  # expected specific force direction at rest
        if abs(np.linalg.norm(a_m - self.ba) - 9.81) > 0.6:
            return
        H = np.zeros((3, 21)); H[:, 6:9] = skew(g_b)
        Rn = np.eye(3) * sig ** 2
        S = H @ self.P @ H.T + Rn
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ ((a_m - self.ba) - g_b))
        I_KH = np.eye(21) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rn @ K.T

    def _inject(self, dx):
        self.p += dx[0:3]; self.v += dx[3:6]
        self.R = self.R @ exp_so3(dx[6:9])
        self.ba += dx[9:12]; self.bg += dx[12:15]
        for i, k in enumerate(self.keys):
            self.pf[k] = self.pf[k] + dx[15 + 3 * i:18 + 3 * i]

    def update_velocity(self, z_world, sig):
        """Direct world-frame velocity measurement of the IMU point (from stance-leg odometry: v = -R(w x h + dh/dt))."""
        H = np.zeros((3, 21)); H[:, 3:6] = np.eye(3)
        Rn = np.eye(3) * sig ** 2
        S = H @ self.P @ H.T + Rn
        K = self.P @ H.T @ np.linalg.inv(S)
        self._inject(K @ (z_world - self.v))
        I_KH = np.eye(21) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rn @ K.T
