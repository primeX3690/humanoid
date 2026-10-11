"""
Records REAL whole-body-controller QPs (and OSQP's answers/iteration counts) so the fast solver can be benchmarked offline on actual data.

    python tools/dump_wbc_qps.py                    # ~2 min: stand 1 s + a 2-step walk, keeps ~60 problems
    python tools/dump_wbc_qps.py --every 7 --limit 120
Writes results/wbc_qps.npz (+ prints a summary). Send me that file: sim_smoke showed the 'fast' backend ~4.5x SLOWER than OSQP on the real
problems (1044 s vs 230 s), the opposite of what random test problems predicted - I need the real problems to fix it.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform.startswith("linux"): os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np


def dense(a): return a.toarray() if hasattr(a, "toarray") else np.asarray(a, float)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=5); ap.add_argument("--limit", type=int, default=60); ap.add_argument("--no-walk", action="store_true"); a = ap.parse_args()
    from control.whole_body_qp import WholeBodyController
    orig = WholeBodyController._solve_qp; store = []; n_calls = [0]; osqp_time = [0.0]

    def wrapped(self, P, q, A, l, u, x0=None):
        n_calls[0] += 1; t0 = time.perf_counter(); r = orig(self, P, q, A, l, u, x0); dt = time.perf_counter() - t0; osqp_time[0] += dt
        if n_calls[0] % a.every == 0 and len(store) < a.limit and r is not None:
            info = getattr(r, "info", None)
            store.append(dict(P=dense(P), q=np.asarray(q, float), A=dense(A), l=np.asarray(l, float), u=np.asarray(u, float), x=np.asarray(r.x, float),
                              iters=int(getattr(info, "iter", -1)), t=dt, status=int(getattr(info, "status_val", 0))))
        return r
    WholeBodyController._solve_qp = wrapped
    from simulation.sim_smoke import _stand
    from simulation.wbc_sim import WBCSim
    S = WBCSim(table=False); S.wbc.qp_backend = "osqp"; _stand(S, 1.0)
    if not a.no_walk:
        from simulation.walk_wbc import run
        run("repo", n_steps=2, verbose=False, qp_backend="osqp")
    WholeBodyController._solve_qp = orig
    out = {}
    for i, d in enumerate(store):
        for k, v in d.items(): out[f"{k}_{i}"] = v
    out["n"] = len(store); os.makedirs("results", exist_ok=True); np.savez_compressed("results/wbc_qps.npz", **out)
    sizes = sorted({(d["P"].shape[0], d["A"].shape[0]) for d in store})
    print(f"solves seen {n_calls[0]}, kept {len(store)}, total OSQP time {osqp_time[0]:.1f} s, problem sizes (n_vars, n_cons): {sizes}")
    print("OSQP iterations: median %d, max %d" % (np.median([d['iters'] for d in store]), max(d['iters'] for d in store)))
    print("saved results/wbc_qps.npz  <- send this file")


if __name__ == "__main__":
    main()
