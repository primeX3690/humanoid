import numpy as np
import pytest
from rl.fullbody_env import FullBodyEnv, FakeBackend, gains_for, joint_class, N_J, CLASS_GAINS
from hardware.mass_budget import ACTUATED_30


def make(**kw):
    return FullBodyEnv(backend=FakeBackend(ACTUATED_30, seed=1), names=ACTUATED_30, seed=0, **kw)


def test_every_actuated_joint_has_gains_and_dims_match():
    sc, kp, kd = gains_for(ACTUATED_30)
    assert len(sc) == N_J and np.all(kp > 0) and np.all(kd > 0) and joint_class("L_wr_roll") == "wr" and joint_class("neck_pitch") == "neck" and joint_class("R_sh_yaw") == "sh"
    e = make(); obs, _ = e.reset(); assert obs.shape == (e.obs_dim,) == (101,) and np.all(np.isfinite(obs))


def test_step_contract_and_action_scaling():
    e = make(); e.reset(); obs, r, term, trunc, info = e.step(np.zeros(N_J))
    assert obs.shape == (101,) and np.isfinite(r) and isinstance(term, bool) and "track_v" in info
    e2 = make(); e2.reset(seed=0); e2.step(np.ones(N_J)); sc = e2.scale
    assert np.allclose(e2.last_action, 1.0) and np.all(sc[[i for i, n in enumerate(ACTUATED_30) if "knee" in n]] == 0.8)


def test_seeded_reset_is_reproducible():
    a, b = make(), make(); oa, _ = a.reset(seed=3); ob, _ = b.reset(seed=3)
    assert np.allclose(oa[-5:-2], ob[-5:-2])                          # same command


def test_unstable_base_eventually_terminates_with_penalty_and_truncation_works():
    e = make(max_steps=400); e.reset(seed=0); e.b.tilt[:] = [0.2, 0.0]; fell = False; r_last = 0
    for _ in range(400):
        obs, r, term, trunc, _ = e.step(np.zeros(N_J))
        if term: fell, r_last = True, r; break
    assert fell and r_last == -5.0
    e = make(max_steps=5); e.reset(); out = [e.step(np.zeros(N_J)) for _ in range(5)]; assert out[-1][3] and not out[-1][2]


def test_commanded_speed_tracking_term_prefers_matching_velocity():
    e = make(); e.reset(); e.cmd[:] = [0.4, 0, 0]; s = e.b.state(); a = np.zeros(N_J)
    s["lin_vel"] = np.array([0.4, 0, 0]); good = e._reward(s, a, 0.0)[1]["track_v"]
    s["lin_vel"] = np.array([0.0, 0, 0]); bad = e._reward(s, a, 0.0)[1]["track_v"]
    assert good > 3 * bad


def test_pushes_are_scheduled_only_when_enabled_and_randomization_samples_per_episode():
    e = make(push_force=50.0, push_interval=(0.05, 0.06)); e.reset(seed=1); assert np.isfinite(e.next_push)
    got = False
    for _ in range(30):
        e.step(np.zeros(N_J)); got |= e.push_left > 0 or e.t > e.next_push
    assert got
    quiet = make(); quiet.reset(); assert quiet.next_push == np.inf
    r = make(); r.reset(seed=1); s1 = r._sample_randomization(); s2 = r._sample_randomization()
    assert s1["seed"] != s2["seed"] and len(s1["joint_names"]) == N_J and make(randomize=False)._sample_randomization() is None
