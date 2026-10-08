"""Capability boundary of the 27 kg humanoid under the ASSUMED actuator limits. python -m simulation.jump_sweep"""
import json, numpy as np
from planning.centroidal_jump_to import JumpOptimizer

if __name__ == "__main__":
    o = JumpOptimizer(); rows = []
    print("--- vertical-ish jumps: minimum flight time forced ---")
    for tf in (0.2, 0.3, 0.4, 0.5, 0.6):
        r = o.solve(0.0, T_flight_min=tf)
        v = o.verify(r) if r["ok"] else {}
        rows.append(dict(kind="vertical", Tf_min=tf, ok=r["ok"], status=r["status"], **{k: (float(x) if np.ndim(x) == 0 else None) for k, x in v.items() if k != "final_state"}))
        print(tf, r["ok"], r["status"], {k: round(float(x), 3) for k, x in v.items() if k in ("apex_gain_m", "peak_GRF_over_mg_takeoff", "peak_knee_torque_Nm_both_legs", "final_err_x")})
    print("--- forward leaps ---")
    for dx in (0.4, 0.8, 1.0, 1.2, 1.5):
        r = o.solve(dx, T_flight_min=0.2)
        v = o.verify(r) if r["ok"] else {}
        rows.append(dict(kind="forward", dx=dx, ok=r["ok"], status=r["status"], **{k: (float(x) if np.ndim(x) == 0 else None) for k, x in v.items() if k != "final_state"}))
        print(dx, r["ok"], r["status"], {k: round(float(x), 3) for k, x in v.items() if k in ("takeoff_vx", "peak_GRF_over_mg_takeoff", "peak_knee_torque_Nm_both_legs", "final_err_x")})
    json.dump(rows, open("results/jump_sweep.json", "w"), indent=1)
