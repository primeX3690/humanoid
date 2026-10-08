"""Thin MuJoCo RGB-D renderer for the head camera (EGL/headless)."""
import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
from perception.object_detector import PinholeCamera


class HeadCameraRenderer:
    def __init__(self, model, width=320, height=240, camera="head_cam"):
        self.m = model
        self.r = mujoco.Renderer(model, height, width)
        self.cam_name = camera
        self.cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.cam = PinholeCamera(width, height, float(model.cam_fovy[self.cam_id]))

    def grab(self, data):
        self.r.disable_depth_rendering()
        self.r.update_scene(data, camera=self.cam_name)
        rgb = self.r.render().copy()
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.cam_name)
        depth = self.r.render().copy()
        self.r.disable_depth_rendering()
        return rgb, depth

    def cam_pose(self, data):
        return data.cam_xpos[self.cam_id].copy(), data.cam_xmat[self.cam_id].reshape(3, 3).copy()
