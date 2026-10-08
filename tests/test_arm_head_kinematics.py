import numpy as np, mujoco, pytest
from model.humanoid_model import make_model, ACTUATED, joint_qadr
from kinematics.arm_kinematics import Arm, ArmParams
from kinematics.head_neck_kinematics import HeadNeck


def _mj_fk(m, d, side, q):
    qa = joint_qadr(m, [f"{side}_{n}" for n in ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_yaw", "wr_pitch", "wr_roll"]])
    d.qpos[:] = 0; d.qpos[3] = 1
    d.qpos[qa] = q
    mujoco.mj_forward(m, d)
    tcp = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")].copy()
    Rt = d.site_xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{side}_tcp")].reshape(3, 3).copy()
    # torso frame: pelvis at origin (identity base), waist joints zero => torso frame = pelvis frame shifted by waist offset
    tb = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")].copy()
    return tcp - tb, Rt


@pytest.mark.parametrize("side", ["L", "R"])
def test_fk_matches_mujoco(side):
    m, d = make_model(table=False)
    arm = Arm(ArmParams(side=side))
    rng = np.random.default_rng(1)
    for _ in range(25):
        q = rng.uniform(arm.q_lo, arm.q_hi)
        p_mj, R_mj = _mj_fk(m, d, side, q)
        T = arm.fk(q)
        assert np.allclose(T[:3, 3], p_mj, atol=1e-9)
        assert np.allclose(T[:3, :3], R_mj, atol=1e-9)


def test_jacobian_finite_difference():
    arm = Arm(ArmParams(side="R"))
    rng = np.random.default_rng(2)
    q = rng.uniform(arm.q_lo, arm.q_hi)
    J = arm.jacobian(q)
    for i in range(7):
        dq = np.zeros(7); dq[i] = 1e-6
        dp = (arm.fk(q + dq)[:3, 3] - arm.fk(q - dq)[:3, 3]) / 2e-6
        assert np.allclose(J[:3, i], dp, atol=1e-6)


def test_ik_reaches_random_reachable_poses():
    arm = Arm(ArmParams(side="R"))
    rng = np.random.default_rng(3)
    ok = 0; N = 40
    for _ in range(N):
        qt = rng.uniform(arm.q_lo * 0.6, arm.q_hi * 0.6)
        Td = arm.fk(qt)
        q, conv, err = arm.ik(Td, q0=0.5 * (arm.q_lo + arm.q_hi) * 0.5, iters=600)
        ok += conv
    assert ok / N > 0.85      # honest threshold: DLS from one seed is not 100%


def test_ik_unreachable_reports_failure():
    arm = Arm(ArmParams(side="R"))
    T = np.eye(4); T[:3, 3] = [2.0, -0.17, 0.4]
    _, conv, err = arm.ik(T, q0=np.zeros(7), pos_only=True)
    assert not conv and err > 1.0


def test_head_lookat_matches_mujoco_camera():
    m, d = make_model(table=False)
    hn = HeadNeck()
    tgt_torso = np.array([0.6, 0.2, -0.1])
    yaw, pitch, ok = hn.look_at(tgt_torso)
    assert ok
    assert hn.gaze_error(yaw, pitch, tgt_torso) < 1e-6
    # same pose in MuJoCo: torso frame == world offset by torso body pos (zero waist, identity base)
    d.qpos[:] = 0; d.qpos[3] = 1
    d.qpos[joint_qadr(m, ["neck_yaw", "neck_pitch"])] = [yaw, pitch]
    mujoco.mj_forward(m, d)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cam_site")
    tb = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso")]
    T = hn.cam_pose(yaw, pitch)
    assert np.allclose(T[:3, 3], d.site_xpos[sid] - tb, atol=1e-9)
    assert np.allclose(T[:3, :3], d.site_xmat[sid].reshape(3, 3), atol=1e-9)
