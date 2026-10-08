"""Render the demo videos used for the IISc / grant material.  python -m simulation.make_videos  -> results/videos/*.mp4"""
import os, sys, json
from model.humanoid_model import ModelParams
from simulation.recorder import Recorder

OUT = "results/videos"

def _rec(S_model, name, **kw):
    return Recorder(S_model, os.path.join(OUT, name), **kw)

if __name__ == "__main__":
    import numpy as np
    from simulation import walk_wbc, pick_demo, push_recovery_exp
    from model.humanoid_model import make_model, repo_actuator_params
    which = sys.argv[1:] or ["push", "pick", "walk"]
    if "push" in which:
        m, _ = make_model(ModelParams(), table=False)
        for name, F, d, stepping in (("push_forward_120N_stepping.mp4", 120, (1, 0), True), ("push_forward_90N_standing_falls.mp4", 90, (1, 0), False),
                                     ("push_lateral_90N_crossover_step.mp4", 90, (0, 1), True)):
            rec = _rec(m, name, distance=3.2, azimuth=(135 if d[0] else 60))
            r = push_recovery_exp.run(F, d, stepping=stepping, T=3.5, recorder=rec)
            print(name, "fell" if r["fell"] else "ok", rec.close(), flush=True)
    if "pick" in which:
        m, _ = make_model(ModelParams(), table=True)
        rec = _rec(m, "pick_red_block_estimated_state.mp4", distance=1.9, azimuth=60, elevation=-20, lookat_z=0.9)
        res, ctx, S = pick_demo.run(estimated=True, verbose=False, recorder=rec)
        print("pick", res["success"], rec.close(), flush=True)
    if "walk" in which:
        m, _ = make_model(repo_actuator_params(), table=False)
        rec = _rec(m, "walk_8steps_repo_actuators_estimated_state.mp4", distance=3.4, azimuth=120, speed=1.0)
        r, S = walk_wbc.run("repo", 8, zc=0.89, verbose=False, estimated=True, recorder=rec)
        print("walk", "fell" if r["fell"] else "ok", rec.close(), flush=True)
