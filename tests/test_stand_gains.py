"""Physics sanity for the RL PD gains: standing on a planted foot needs the joint stiffness matrix Kp to dominate gravity's destabilising Hessian.
3-link sagittal model (shank, thigh, pelvis+torso+head+arms lumped), masses/lengths from model.humanoid_model.ModelParams defaults.
Regression for the real failure: zero-action stand fell after 3.3 s with ankle kp = 120 (v3.2 gains)."""
import numpy as np
import pytest
from rl.fullbody_env import CLASS_GAINS

G = 9.81
L1, L2, LT = 0.45, 0.45, 0.50
M1, M2, M3 = 2 * (1.4 + 0.45), 2 * 2.4, 3.0 + 5.64 + 1.3 + 2 * (0.9 + 0.65 + 0.4) + 8 * 0.2


def _V(q):
    p1, p2, p3 = q[0], q[0] + q[1], q[0] + q[1] + q[2]
    y1 = 0.5 * L1 * np.cos(p1); y2 = L1 * np.cos(p1) + 0.5 * L2 * np.cos(p2); y3 = L1 * np.cos(p1) + L2 * np.cos(p2) + 0.5 * LT * np.cos(p3)
    return G * (M1 * y1 + M2 * y2 + M3 * y3)


def _hess(f, n=3, e=1e-4):
    H = np.zeros((n, n)); I = np.eye(n) * e; q = np.zeros(n)
    for i in range(n):
        for j in range(n): H[i, j] = (f(q + I[i] + I[j]) - f(q + I[i] - I[j]) - f(q - I[i] + I[j]) + f(q - I[i] - I[j])) / (4 * e * e)
    return H


def min_eig(kp_ankle, kp_knee, kp_hip):
    return float(np.linalg.eigvalsh(np.diag([kp_ankle, kp_knee, kp_hip]) + _hess(_V)).min())


def test_gravity_destabilising_stiffness_is_about_m_g_h():
    H = _hess(_V); assert -H[0, 0] == pytest.approx(G * (M1 * 0.5 * L1 + M2 * (L1 + 0.5 * L2) + M3 * (L1 + L2 + 0.25)), rel=0.02)


def test_old_gains_were_statically_unstable_current_gains_are_stable_with_margin():
    assert min_eig(120, 350, 300) < -100                       # the v3.2 gains: negative eigenvalue -> topples (observed: fell at 3.3 s)
    cur = min_eig(CLASS_GAINS["ankle_pitch"][1], CLASS_GAINS["knee"][1], CLASS_GAINS["hip_pitch"][1])
    assert cur > 100                                           # positive-definite with a real margin


def test_every_class_has_positive_damping_ratio_budget():
    for k, (sc, kp, kd) in CLASS_GAINS.items():
        assert sc > 0 and kp > 0 and kd > 0 and kd < 0.2 * kp, k   # kd/kp < 0.2 s: damping not absurd relative to stiffness
