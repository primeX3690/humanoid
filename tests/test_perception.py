import numpy as np, mujoco, pytest
from model.humanoid_model import make_model, stand_pose, joint_qadr
pytest.importorskip("mujoco")


def _scene(neck_pitch=0.9):
    m, d = make_model(); stand_pose(m, d)
    d.qpos[joint_qadr(m, ["neck_pitch"])] = neck_pitch
    mujoco.mj_forward(m, d)
    return m, d


def test_detects_red_box_and_blue_cylinder_with_cm_accuracy():
    from perception.render import HeadCameraRenderer
    from perception.object_detector import detect
    m, d = _scene()
    R = HeadCameraRenderer(m)
    rgb, depth = R.grab(d)
    p, Rm = R.cam_pose(d)
    dets = {x.label: x for x in detect(rgb, depth, R.cam, p, Rm)}
    assert set(dets) == {"red", "blue"}
    box = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "target_obj")]
    cyl = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "distractor")]
    assert np.linalg.norm(dets["red"].top_centroid[:2] - box[:2]) < 0.01     # <1 cm in the table plane (top-face centroid)
    assert abs(dets["red"].top_z - (box[2] + 0.03)) < 0.01                # top face height (box half-height 3 cm)
    assert np.linalg.norm(dets["blue"].centroid[:2] - cyl[:2]) < 0.02


def test_no_detection_when_not_looking_at_scene():
    from perception.render import HeadCameraRenderer
    from perception.object_detector import detect
    m, d = _scene(neck_pitch=-0.3)      # looking up at the sky
    R = HeadCameraRenderer(m)
    rgb, depth = R.grab(d)
    p, Rm = R.cam_pose(d)
    assert detect(rgb, depth, R.cam, p, Rm) == []


def test_projection_backprojection_roundtrip():
    from perception.object_detector import PinholeCamera
    cam = PinholeCamera(320, 240, 70)
    pts = np.array([[0.1, -0.05, -0.8], [-0.2, 0.1, -1.5]])
    u, v = cam.project(pts)
    back = cam.backproject(u, v, -pts[:, 2])
    assert np.allclose(back, pts, atol=1e-9)
