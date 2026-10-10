"""
Absolute-pose aiding for BaseESKF: closes the documented gap "global x, y and yaw are NOT observable (drift slowly)".

Any external fix - a SLAM pose (perception/ekf_slam.py), a known landmark / AprilTag, visual odometry, UWB, a magnetometer - enters
as a position and/or yaw measurement in the SAME 21-dim error state, with Mahalanobis gating so one bad fix cannot yank the filter.
Also: `ContactDetector` from foot force/torque sensing (Fz hysteresis + CoP), which replaces the "contact flag from the planner".
Pure NumPy; works on any object with BaseESKF's p, R, P, _inject.
"""
from __future__ import annotations
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _update(f, H, r, Rn, gate_chi2=None):
    S = H @ f.P @ H.T + Rn
    if gate_chi2 is not None:
        m2 = float(r @ np.linalg.solve(S, r))
        if m2 > gate_chi2:
            return False, m2
    K = f.P @ H.T @ np.linalg.inv(S)
    f._inject(K @ r)
    I_KH = np.eye(f.P.shape[0]) - K @ H
    f.P = I_KH @ f.P @ I_KH.T + K @ Rn @ K.T
    return True, 0.0


def update_position(f, p_meas, sig, gate_chi2=16.27):
    """World position fix (3-vector, or a 2-vector for x,y only). chi2 gate: 16.27 = 99.9 % for 3 dof."""
    p_meas = np.asarray(p_meas, float); k = len(p_meas)
    H = np.zeros((k, f.P.shape[0])); H[:, 0:k] = np.eye(k)
    return _update(f, H, p_meas - f.p[:k], np.eye(k) * sig ** 2, gate_chi2 if k == 3 else 13.8)


def update_yaw(f, yaw_meas, sig, gate_chi2=10.83):
    """Heading fix (rad). Error state dtheta is body-frame, so the world-yaw error is  R[2,:] . dtheta."""
    H = np.zeros((1, f.P.shape[0])); H[0, 6:9] = f.R[2, :]
    r = np.array([wrap(yaw_meas - yaw_of(f.R))])
    return _update(f, H, r, np.array([[sig ** 2]]), gate_chi2)


def update_pose(f, p_meas, yaw_meas, sig_p, sig_yaw):
    a = update_position(f, p_meas, sig_p)
    b = update_yaw(f, yaw_meas, sig_yaw)
    return a[0], b[0]


def widen_for_aiding(f, p_sigma=0.5, yaw_sigma=0.3):
    """BaseESKF starts with TIGHT x,y,yaw priors (they were unobservable). When aiding is available, loosen them so fixes can correct."""
    f.P[0:2, 0:2] = np.eye(2) * p_sigma ** 2
    f.P[8, 8] = yaw_sigma ** 2


class ContactDetector:
    """Foot contact + centre of pressure from a 6-axis foot force/torque sensor (sole frame, z up).

    Contact when Fz > f_on, released when Fz < f_off (hysteresis, so noise does not chatter). CoP = (-tau_y/Fz, tau_x/Fz) in the
    sole frame (+ sensor-to-sole offset h):  px = (-tau_y - Fx*h)/Fz,  py = (tau_x - Fy*h)/Fz.
    Slip flag: |F_t| > mu * Fz.
    """

    def __init__(self, f_on=60.0, f_off=25.0, mu=0.6, sensor_h=0.0, debounce=2):
        self.f_on, self.f_off, self.mu, self.h, self.deb = f_on, f_off, mu, sensor_h, debounce
        self.state = False; self._cnt = 0

    def update(self, wrench):
        Fx, Fy, Fz, tx, ty, tz = np.asarray(wrench, float)
        want = (Fz > self.f_on) if not self.state else (Fz > self.f_off)
        self._cnt = 0 if want == self.state else self._cnt + 1
        if self._cnt >= self.deb:
            self.state = want; self._cnt = 0
        cop = None
        if Fz > 5.0:
            cop = np.array([(-ty - Fx * self.h) / Fz, (tx - Fy * self.h) / Fz])
        slip = bool(self.state and np.hypot(Fx, Fy) > self.mu * max(Fz, 1e-6))
        return dict(contact=self.state, cop=cop, slip=slip, fz=float(Fz))
