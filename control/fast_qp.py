"""
Dense ADMM QP solver (OSQP algorithm, pure NumPy/SciPy) with Ruiz scaling, adaptive rho, warm start and active-set polish.

    min 0.5 x'Px + q'x   s.t.  l <= Ax <= u        (equalities: l == u)

Why it exists: the whole-body controller builds a NEW osqp problem for every one of ~5 priority levels every 5 ms, paying setup +
factorisation + thousands of iterations. For the ~100-variable dense QPs of a humanoid WBC, a cached, warm-started dense solver
is far cheaper, needs no compiled dependency (so it runs on the robot's own computer too) and lets us keep state between ticks.
Interface mimics `osqp` just enough for `WholeBodyController._solve_qp`:  r.x, r.y, r.info.status_val (1 solved, 2 inaccurate).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as sla

INF = 1e20


@dataclass
class _Info:
    status: str = "unsolved"
    status_val: int = 0
    iter: int = 0
    pri_res: float = np.inf
    dua_res: float = np.inf
    polished: bool = False


@dataclass
class QPResult:
    x: np.ndarray | None
    y: np.ndarray | None
    info: _Info = field(default_factory=_Info)


def _ruiz(P, A, iters=10):
    n, m = P.shape[0], A.shape[0]
    D = np.ones(n); E = np.ones(m)
    for _ in range(iters):
        Pc = np.abs(P).max(0) if n else np.zeros(0)
        Ac = np.abs(A).max(0) if m else np.zeros(n)
        cn = np.maximum(Pc, Ac); cn[cn < 1e-4] = 1.0
        rn = np.abs(A).max(1) if m else np.zeros(0); rn[rn < 1e-4] = 1.0
        dd = 1.0 / np.sqrt(cn); ee = 1.0 / np.sqrt(rn)
        P = P * dd[:, None] * dd[None, :]; A = A * ee[:, None] * dd[None, :]
        D *= dd; E *= ee
    return D, E


class FastQP:
    def __init__(self, eps=1e-6, max_iter=4000, rho=0.1, sigma=1e-6, alpha=1.6, polish=True, check_every=25):
        self.eps, self.max_iter, self.rho0, self.sigma, self.alpha = eps, max_iter, rho, sigma, alpha
        self.polish, self.check_every = polish, check_every
        self._ws = {}

    # ------------------------------------------------------------------
    def solve(self, P, q, A, l, u, x0=None, y0=None, key=None) -> QPResult:
        P = np.asarray(P, float); q = np.asarray(q, float); A = np.asarray(A, float)
        l = np.maximum(np.asarray(l, float), -INF); u = np.minimum(np.asarray(u, float), INF)
        P = 0.5 * (P + P.T)
        n, m = len(q), A.shape[0]
        if m == 0:
            A = np.zeros((0, n)); l = u = np.zeros(0)
        # ---- scaling
        D, E = _ruiz(P, A)
        Ps = P * D[:, None] * D[None, :]; As = A * E[:, None] * D[None, :]
        cs = 1.0 / max(np.abs(Ps).max() if n else 1.0, np.abs(q * D).max() if n else 1.0, 1e-6)
        Ps = cs * Ps; qs = cs * q * D
        ls = np.where(l <= -INF, -INF, E * l); us = np.where(u >= INF, INF, E * u)
        eq = (u - l) < 1e-9
        rho_vec = np.where(eq, 1e3 * self.rho0, self.rho0)
        # ---- warm start (explicit > cached by key)
        ws = self._ws.get(key) if key is not None else None
        x = np.zeros(n); y = np.zeros(m)
        if x0 is not None: x = np.asarray(x0, float) / D
        elif ws is not None and ws[0].shape == (n,) and ws[1].shape == (m,): x = ws[0] / D
        if y0 is not None: y = np.asarray(y0, float) / E * cs
        elif ws is not None and ws[0].shape == (n,) and ws[1].shape == (m,): y = ws[1] / E * cs
        z = np.clip(As @ x, ls, us)

        def factor(rv):
            K = Ps + self.sigma * np.eye(n) + As.T @ (rv[:, None] * As)
            return sla.cho_factor(K, lower=True, check_finite=False)
        cf = factor(rho_vec)
        info = _Info(); a = self.alpha
        for k in range(1, self.max_iter + 1):
            rhs = self.sigma * x - qs + As.T @ (rho_vec * z - y)
            xt = sla.cho_solve(cf, rhs, check_finite=False)
            zt = As @ xt
            x_new = a * xt + (1 - a) * x
            zr = a * zt + (1 - a) * z
            z_new = np.clip(zr + y / rho_vec, ls, us)
            y = y + rho_vec * (zr - z_new)
            x, z = x_new, z_new
            if k % self.check_every == 0 or k == 1:
                Ax = As @ x; Px = Ps @ x; Aty = As.T @ y
                pr = np.abs((Ax - z) / E).max() if m else 0.0
                dr = np.abs((Px + qs + Aty) / (D * cs)).max()
                n_p = max(np.abs(Ax / E).max() if m else 0.0, np.abs(z / E).max() if m else 0.0)
                n_d = max(np.abs(Px / (D * cs)).max(), np.abs(Aty / (D * cs)).max(), np.abs(qs / (D * cs)).max())
                if pr <= self.eps * (1 + n_p) and dr <= self.eps * (1 + n_d):
                    info.status, info.status_val = "solved", 1; break
                if m and k > 1:                                    # adaptive rho
                    ratio = np.sqrt((pr / (n_p + 1e-12)) / max(dr / (n_d + 1e-12), 1e-12))
                    if ratio > 5.0 or ratio < 0.2:
                        rho_vec = np.clip(rho_vec * ratio, 1e-6, 1e6)
                        rho_vec = np.where(eq, np.maximum(rho_vec, 1e3 * 1e-6), rho_vec)
                        cf = factor(rho_vec)
        else:
            info.status = "solved inaccurate" if (pr < 1e-3 and dr < 1e-3) else "max iter"
            info.status_val = 2 if (pr < 1e-3 and dr < 1e-3) else -2
        info.iter, info.pri_res, info.dua_res = k, float(pr), float(dr)
        xo = D * x; yo = E * y / cs
        res = QPResult(xo, yo, info)
        if self.polish and info.status_val > 0 and m:
            self._polish(res, P, q, A, l, u)
        if key is not None and info.status_val > 0:
            self._ws[key] = (res.x.copy(), res.y.copy())
        return res

    # ------------------------------------------------------------------
    @staticmethod
    def _polish(res, P, q, A, l, u, tol=1e-7, delta=1e-9):
        x, y = res.x, res.y
        Ax = A @ x
        lo = (Ax - l < -y) & (l > -INF)              # lower active
        hi = (u - Ax < y) & (u < INF)                # upper active
        act = lo | hi
        n = len(q); na = int(act.sum())
        Aa = A[act]; ba = np.where(lo[act], l[act], u[act])
        K0 = np.block([[P, Aa.T], [Aa, np.zeros((na, na))]])
        K = K0 + delta * np.diag(np.concatenate([np.ones(n), -np.ones(na)]))
        rhs = np.concatenate([-q, ba])
        try:
            sol = np.linalg.solve(K, rhs)
            for _ in range(3):                         # iterative refinement against the UNregularised KKT matrix
                sol += np.linalg.solve(K, rhs - K0 @ sol)
        except np.linalg.LinAlgError:
            return
        xp = sol[:n]; yp = np.zeros(len(l)); yp[act] = sol[n:]
        Axp = A @ xp
        viol = max(np.max(np.maximum(l - Axp, 0)) if len(l) else 0, np.max(np.maximum(Axp - u, 0)) if len(l) else 0)
        sign_ok = np.all(yp[lo & act] <= 1e-6) and np.all(yp[hi & act] >= -1e-6)
        old = max(np.max(np.maximum(l - Ax, 0)), np.max(np.maximum(Ax - u, 0)))
        if viol <= max(tol, old) and sign_ok and np.all(np.isfinite(xp)):
            res.x, res.y = xp, yp; res.info.polished = True


_default = FastQP()


def solve_qp(P, q, A, l, u, x0=None, key=None, solver: FastQP | None = None) -> QPResult:
    return (solver or _default).solve(P, q, A, l, u, x0=x0, key=key)
