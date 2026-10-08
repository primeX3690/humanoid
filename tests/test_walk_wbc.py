"""Integration: humanoid-loco LIPM/ZMP plan -> whole-body QP -> MuJoCo full-body humanoid (needs the repo root on the path)."""
import numpy as np, pytest
pytest.importorskip("planning.footstep_planner")


def test_two_step_walk_tracks_lipm_plan_and_lands_on_target():
    from simulation.walk_wbc import run
    r, S = run(actuators="strong", n_steps=2, zc=0.85, verbose=False, max_time=4.0)
    assert not r["fell"]
    assert r["touchdowns"] == 2 and r["touchdown_err_mm_max"] < 20.0       # feet land within 2 cm of the plan
    assert r["com_track_rms_mm"] < 60.0                                    # CoM follows the LIPM reference
    assert r["distance_m"] > 0.3                                            # it actually moved forward
    assert r["peak_leg_speed_rad_s"] < 3.47                                 # within the PH54-200 no-load speed
