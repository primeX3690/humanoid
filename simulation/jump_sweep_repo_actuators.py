"""Can the humanoid-loco hardware (PH54-200, 44.7 Nm, 3.47 rad/s) do athletic motion?  python -m simulation.jump_sweep_repo_actuators"""
import json, numpy as np
from planning.centroidal_jump_to import JumpOptimizer, JumpConfig, repo_jump_config

def row(tag, cfg, dx, tf):
    o = JumpOptimizer(cfg); r = o.solve(dx, T_flight_min=tf)
    v = o.verify(r) if r["ok"] else {}
    d = dict(tag=tag, dx=dx, Tf_min=tf, ok=bool(r["ok"]), status=r["status"],
             **{k: float(x) for k, x in v.items() if k in ("apex_gain_m", "peak_knee_torque_Nm_both_legs", "peak_stance_joint_speed_rad_s", "peak_GRF_over_mg_takeoff", "final_err_x")})
    print(d, flush=True); return d

if __name__ == "__main__":
    rows = []
    torque_only = repo_jump_config(omega_max=0.0)            # repo torque limits, no speed limit
    both = repo_jump_config()                                 # repo torque + speed limits
    for tf in (0.15, 0.2, 0.25, 0.3, 0.4):
        rows.append(row("torque-only", torque_only, 0.0, tf))
        rows.append(row("torque+speed", both, 0.0, tf))
    for dx in (0.2, 0.4, 0.6):
        rows.append(row("torque-only", torque_only, dx, 0.15))
        rows.append(row("torque+speed", both, dx, 0.15))
    json.dump(rows, open("results/jump_repo_actuators.json", "w"), indent=1)
