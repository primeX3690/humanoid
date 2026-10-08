"""MP4 (or GIF fallback) video recorder for any MuJoCo scenario in this repo (headless EGL on Linux)."""
import os, sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco


class Recorder:
    def __init__(self, model, path, fps=30, width=640, height=360, track="pelvis", distance=3.0, azimuth=135.0,
                 elevation=-15.0, lookat_z=0.8, speed=1.0):
        self.m, self.path, self.fps, self.speed = model, path, fps, speed
        self.r = mujoco.Renderer(model, height, width)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track)
        self.cam.distance, self.cam.azimuth, self.cam.elevation = distance, azimuth, elevation
        self.cam.lookat[:] = [0, 0, lookat_z]
        self.frames, self.next_t = [], 0.0

    def maybe(self, data, t):
        if t + 1e-9 >= self.next_t:
            self.r.update_scene(data, camera=self.cam)
            self.frames.append(self.r.render().copy())
            self.next_t += self.speed / self.fps

    def close(self):
        import imageio
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            imageio.mimsave(self.path, self.frames, fps=self.fps, macro_block_size=1)
        except Exception:
            self.path = self.path.rsplit(".", 1)[0] + ".gif"
            imageio.mimsave(self.path, self.frames, fps=self.fps)
        self.r.close()
        return self.path, len(self.frames)
