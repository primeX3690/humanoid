"""tests/test_rl_env.py - verifies the MuJoCo biped model and RL environment
(rl/biped_model.py, rl/biped_env.py): the model is really built from this
repository's own parameters (mass, link lengths, actuator limit), the
environment obeys the Gymnasium API, actuator limits are respected, and a
plain PD hold of the nominal pose stands for 20 s - the sanity baseline any
learned policy must beat before a learning curve means anything."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

import mujoco
from gymnasium.utils.env_checker import check_env

from kinematics.leg_fk import LegParams
from dynamics.leg_dynamics import (
    RobotMassParams, THIGH_MASS_FRACTION, SHANK_MASS_FRACTION, FOOT_MASS_FRACTION,
)
from actuators.motor_specs import DYNAMIXEL_H54P_200
from rl.biped_model import (
    default_spec, KP_LEG, NOMINAL_LEG_POSE, JOINT_LIMITS_RAD, N_JOINTS,
)
from rl.biped_env import BipedEnv


def _body_pos(env, name):
    b = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return env.data.xpos[b].copy()


def test_total_mass_matches_the_repos_robot_mass_assumption():
    env = BipedEnv(randomize=False)
    assert np.sum(env.model.body_mass) == pytest.approx(RobotMassParams().total_mass_kg, rel=1e-6)


def test_segment_masses_use_the_de_leva_fractions():
    env = BipedEnv(randomize=False)
    total = RobotMassParams().total_mass_kg
    for name, frac in (("left_thigh", THIGH_MASS_FRACTION), ("left_shank", SHANK_MASS_FRACTION),
                       ("left_foot", FOOT_MASS_FRACTION)):
        b = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert env.model.body_mass[b] == pytest.approx(frac * total, rel=1e-6)


def test_link_lengths_match_legparams():
    """thigh/shank lengths in the MuJoCo model must equal the ones the
    kinematics, IK and dynamics modules use."""
    env = BipedEnv(randomize=False)
    env.reset(seed=0)
    leg = LegParams()
    # zero all joints so segments hang straight down, then measure body origins
    env.data.qpos[7:7 + N_JOINTS] = 0.0
    mujoco.mj_forward(env.model, env.data)
    hip, knee, ankle = (_body_pos(env, n) for n in ("left_thigh", "left_shank", "left_foot"))
    assert np.linalg.norm(hip - knee) == pytest.approx(leg.thigh_length_m, abs=1e-6)
    assert np.linalg.norm(knee - ankle) == pytest.approx(leg.shank_length_m, abs=1e-6)


def test_actuators_use_the_h54p_torque_limit():
    env = BipedEnv(randomize=False)
    assert env.model.nu == N_JOINTS
    limit = DYNAMIXEL_H54P_200.continuous_torque_nm
    assert np.allclose(env.model.actuator_forcerange[:, 1], limit)
    assert np.allclose(env.model.actuator_forcerange[:, 0], -limit)


def test_ankle_pd_stiffness_exceeds_gravitational_destabilising_stiffness():
    """m*g*h is the inverted-pendulum 'negative stiffness' of the standing
    body about the ankle; a PD ankle softer than that cannot hold the pose at
    all, however good the nominal joint angles are (an earlier version with
    kp_ankle=150 fell over from every pose tried)."""
    env = BipedEnv(randomize=False)
    env.reset(seed=0)
    m = RobotMassParams().total_mass_kg
    com_h = env.data.subtree_com[env._pelvis][2]
    assert KP_LEG[4] > m * 9.81 * com_h


def test_nominal_pose_is_inside_joint_limits():
    assert np.all(NOMINAL_LEG_POSE >= JOINT_LIMITS_RAD[:, 0])
    assert np.all(NOMINAL_LEG_POSE <= JOINT_LIMITS_RAD[:, 1])


def test_gymnasium_api_conformance():
    env = BipedEnv(randomize=True, push_velocity=0.5, seed=1)
    check_env(env, skip_render_check=True)


def test_reset_is_deterministic_given_a_seed():
    a = BipedEnv(seed=3)
    b = BipedEnv(seed=3)
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.allclose(oa, ob)
    for _ in range(20):
        act = np.full(N_JOINTS, 0.1)
        oa, ra, *_ = a.step(act)
        ob, rb, *_ = b.step(act)
    assert np.allclose(oa, ob) and ra == pytest.approx(rb)


@pytest.mark.parametrize("randomize", [False, True])
def test_zero_action_pd_hold_stands_for_20_seconds(randomize):
    """The baseline: with zero policy output the PD controller just holds the
    nominal pose. It must stay upright for the full 1000-step (20 s) episode."""
    env = BipedEnv(randomize=randomize, cmd_vx_range=(0.0, 0.0))
    for seed in range(3):
        env.reset(seed=seed)
        for _ in range(1000):
            _, _, terminated, truncated, _ = env.step(np.zeros(N_JOINTS))
            assert not terminated
            if truncated:
                break


def test_torque_never_exceeds_actuator_limit_even_under_extreme_actions():
    env = BipedEnv(randomize=False)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    limit = DYNAMIXEL_H54P_200.continuous_torque_nm
    for _ in range(100):
        env.step(rng.choice([-1.0, 1.0], size=N_JOINTS))
        assert np.max(np.abs(env.data.actuator_force)) <= limit + 1e-6


def test_push_disturbance_applies_a_velocity_impulse():
    env = BipedEnv(randomize=False, push_velocity=1.0, push_interval_s=(0.2, 0.2), cmd_vx_range=(0, 0))
    env.reset(seed=0)
    speeds = []
    for _ in range(30):
        env.step(np.zeros(N_JOINTS))
        speeds.append(np.linalg.norm(env.data.qvel[:2]))
    assert max(speeds) > 0.25


def test_observation_and_reward_are_finite_and_standing_earns_positive_reward():
    env = BipedEnv(randomize=False, cmd_vx_range=(0.0, 0.0))
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    total = 0.0
    for _ in range(100):
        obs, r, *_ = env.step(np.zeros(N_JOINTS))
        assert np.isfinite(r)
        total += r
    assert total > 0.0


def test_falling_terminates_the_episode():
    env = BipedEnv(randomize=False, cmd_vx_range=(0.0, 0.0))
    env.reset(seed=0)
    terminated = False
    for _ in range(300):
        _, _, terminated, _, info = env.step(np.full(N_JOINTS, -1.0))  # collapse the legs
        if terminated:
            assert info["fallen"]
            break
    assert terminated


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
