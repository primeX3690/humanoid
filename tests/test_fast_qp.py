import time
import numpy as np
import pytest
from scipy.optimize import minimize
from control.fast_qp import FastQP, solve_qp


def rand_qp(n, m, n_eq, seed):
    r = np.random.default_rng(seed)
    M = r.normal(size=(n, n)); P = M @ M.T + 0.1 * np.eye(n); q = r.normal(size=n)
    A = r.normal(size=(m, n)); x0 = r.normal(size=n)
    l = A @ x0 - r.uniform(0.0, 1.0, m); u = A @ x0 + r.uniform(0.0, 1.0, m)
    l[:n_eq] = u[:n_eq] = A[:n_eq] @ x0
    return P, q, A, l, u


def kkt_ok(P, q, A, l, u, res, tol=1e-4):
    x, y = res.x, res.y; Ax = A @ x
    prim = max(np.max(np.maximum(l - Ax, 0)), np.max(np.maximum(Ax - u, 0)))
    dual = np.abs(P @ x + q + A.T @ y).max()
    comp = 0.0
    for i in range(len(l)):
        if y[i] > 1e-6: comp = max(comp, abs(Ax[i] - u[i]))
        if y[i] < -1e-6: comp = max(comp, abs(Ax[i] - l[i]))
    return prim < tol and dual < tol and comp < tol


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_matches_slsqp_on_small_problems(seed):
    P, q, A, l, u = rand_qp(8, 14, 2, seed)
    r = solve_qp(P, q, A, l, u)
    assert r.info.status_val == 1
    cons = [{"type": "ineq", "fun": lambda x: A @ x - l}, {"type": "ineq", "fun": lambda x: u - A @ x}]
    ref = minimize(lambda x: 0.5 * x @ P @ x + q @ x, np.zeros(8), jac=lambda x: P @ x + q, constraints=cons,
                   method="SLSQP", options=dict(ftol=1e-12, maxiter=500))
    assert np.allclose(r.x, ref.x, atol=1e-4)
    assert kkt_ok(P, q, A, l, u, r)


def test_wbc_sized_problem_satisfies_kkt_and_is_fast():
    P, q, A, l, u = rand_qp(90, 170, 40, 7)
    s = FastQP()
    t0 = time.time(); r = s.solve(P, q, A, l, u); dt = time.time() - t0
    assert r.info.status_val == 1 and kkt_ok(P, q, A, l, u, r, 1e-5)
    assert dt < 0.5


def test_warm_start_cuts_iterations_on_perturbed_problem():
    P, q, A, l, u = rand_qp(60, 100, 20, 3)
    s = FastQP(polish=False)
    r0 = s.solve(P, q, A, l, u, key="k")
    q2 = q + 1e-2 * np.random.default_rng(0).normal(size=len(q))
    cold = FastQP(polish=False).solve(P, q2, A, l, u)
    warm = s.solve(P, q2, A, l, u, key="k")
    assert warm.info.iter < cold.info.iter
    assert np.allclose(warm.x, cold.x, atol=1e-3)


def test_equality_constraints_are_tight_after_polish():
    P, q, A, l, u = rand_qp(30, 50, 25, 11)
    r = solve_qp(P, q, A, l, u)
    assert np.abs(A[:25] @ r.x - l[:25]).max() < 1e-7


def test_unconstrained_and_box_cases():
    P = np.diag([2.0, 4.0]); q = np.array([-2.0, -8.0])
    r = solve_qp(P, q, np.eye(2), np.array([-10.0, -10.0]), np.array([10.0, 1.5]))
    assert np.allclose(r.x, [1.0, 1.5], atol=1e-5)
    r = solve_qp(P, q, np.zeros((0, 2)), np.zeros(0), np.zeros(0))
    assert np.allclose(r.x, [1.0, 2.0], atol=1e-5)
