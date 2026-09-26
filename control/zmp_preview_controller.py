"""
control/zmp_preview_controller.py

Optimal ZMP preview control (Kajita, Kanehiro, Kaneko, Fujiwara, Harada,
Yokoi, Hirukawa - "Biped Walking Pattern Generation by using Preview
Control of Zero-Moment Point", ICRA 2003). This is the actual algorithm
used to generate CoM trajectories on real preview-control-based humanoid
robots (HRP-2 and its lineage) - not a bespoke simplification.

Core idea: given a desired ZMP reference trajectory (which comes from the
footstep plan - see planning/footstep_planner.py), the controller looks
FORWARD over a preview window of future ZMP reference points (typically
~1-2s ahead) and computes the CoM jerk command that makes the LIPM's
ZMP track that reference with (theoretically) zero steady-state error,
while anticipating upcoming footstep transitions instead of reacting
after the fact - this preview/anticipation is why the CoM visibly leans
into a step before the ZMP reference actually jumps to the next foot in
generated trajectories from every real implementation of this algorithm.

Derivation implemented here (standard LQI-with-preview augmentation):

    augmented state  X(k) = [ e(k); x(k) ],  e(k) = e(k-1) + p(k) - p_ref(k)
    X(k+1) = A_hat X(k) + B_hat u(k) + G_hat p_ref(k+1)

    A_hat = [[1, C@A], [0, A]]     (4x4)
    B_hat = [[C@B],    [B]]        (4x1)
    G_hat = [[-1], [0,0,0]]        (4x1)

Solve the discrete algebraic Riccati equation (via scipy - not a
hand-rolled iteration, to avoid a subtle numerical bug in the one place
correctness matters most) for cost J = sum(Q*e(k)^2 + R*u(k)^2), giving
the optimal gain K = [Gi, Gx] and, from the closed-loop matrix, the
preview gains Gd(1..N) applied to the future reference window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are

from dynamics.lipm import LIPM


@dataclass
class PreviewControllerConfig:
    Q: float = 30.0        # ZMP tracking error weight - see docs/BUGS_FOUND.md: a much larger Q (e.g.
                            # 1e6, the more "obviously tight-tracking" choice) makes the controller so
                            # aggressive that it produces a large non-minimum-phase undershoot when a
                            # lateral weight-shift is requested on short notice, driving the ZMP OUT of
                            # the support polygon - i.e. too much tracking gain actually made the walk
                            # LESS stable, a real and initially counterintuitive tuning finding, not a
                            # default picked by guesswork
    R: float = 1.0        # jerk (control effort) weight
    preview_horizon_s: float = 1.6  # how far ahead the controller "sees" the ZMP reference


class ZMPPreviewController:
    def __init__(self, lipm: LIPM, config: PreviewControllerConfig):
        self.lipm = lipm
        self.cfg = config
        A, B, C = lipm.A, lipm.B, lipm.C
        self.N = int(round(config.preview_horizon_s / lipm.p.dt))

        A_hat = np.zeros((4, 4))
        A_hat[0, 0] = 1.0
        A_hat[0, 1:4] = (C @ A).flatten()
        A_hat[1:4, 1:4] = A

        B_hat = np.zeros((4, 1))
        B_hat[0, 0] = (C @ B).item()
        B_hat[1:4, 0] = B.flatten()

        G_hat = np.zeros((4, 1))
        G_hat[0, 0] = -1.0

        Q_hat = np.zeros((4, 4))
        Q_hat[0, 0] = config.Q
        R_mat = np.array([[config.R]])

        P = solve_discrete_are(A_hat, B_hat, Q_hat, R_mat)
        denom = (R_mat + B_hat.T @ P @ B_hat)  # 1x1
        K = np.linalg.solve(denom, B_hat.T @ P @ A_hat)  # 1x4, optimal feedback gain

        self.Gi = float(K[0, 0])
        self.Gx = K[0, 1:4].copy()  # shape (3,)

        A_cl = A_hat - B_hat @ K  # closed-loop augmented dynamics

        # Preview gains: Gd(j) = inv(R + B_hat^T P B_hat) B_hat^T (A_cl^T)^(j-1) P G_hat
        self.Gd = np.zeros(self.N)
        vec = P @ G_hat  # 4x1, this is (A_cl^T)^0 P G_hat for j=1
        for j in range(self.N):
            self.Gd[j] = float(np.linalg.solve(denom, B_hat.T @ vec).item())
            vec = A_cl.T @ vec

        self._e = 0.0  # integral tracking error, persists across calls to step()

    def reset(self):
        self._e = 0.0

    def compute_jerk(self, x: np.ndarray, zmp_ref_window: np.ndarray) -> float:
        """
        x: current LIPM state [pos, vel, acc]
        zmp_ref_window: array of length >= N+1, zmp_ref_window[0] = p_ref(k) is
            the CURRENT step's reference (used only to update the integral
            error), zmp_ref_window[1..N] = p_ref(k+1..k+N) is the preview
            window the controller looks ahead over.
        Returns: jerk command u(k) to feed into lipm.step().
        """
        assert len(zmp_ref_window) >= self.N + 1, (
            f"need {self.N + 1} reference points (current + {self.N} preview), "
            f"got {len(zmp_ref_window)} - pad the reference trajectory with its "
            f"final value if near the end of a walk."
        )
        p = self.lipm.zmp(x)
        self._e += p - zmp_ref_window[0]

        preview_term = float(np.dot(self.Gd, zmp_ref_window[1:self.N + 1]))
        jerk = -self.Gi * self._e - float(self.Gx @ x) - preview_term
        return jerk
