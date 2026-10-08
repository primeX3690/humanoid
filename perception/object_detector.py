"""
RGB-D object detection + 3-D localisation from the head camera (pinhole model).

Pipeline: colour segmentation (hue/saturation) -> connected components (scipy.ndimage) -> depth back-projection
-> world-frame point cloud -> centroid, top-surface height, principal-axis yaw.
Deliberately classical & CPU-cheap (no NN); the interface (`detect(rgb, depth, cam_pose)`) is what a learned
detector would replace later. Works on MuJoCo-rendered frames here; real sensors will need calibration + domain work.
"""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage


@dataclass
class Detection:
    label: str
    centroid: np.ndarray       # world frame, mean of visible-surface points
    top_z: float
    yaw: float                 # principal axis of the footprint (rad, world)
    n_pixels: int
    bbox: tuple                # (u0, v0, u1, v1)
    extent: np.ndarray         # (major, minor) footprint extent in metres
    top_centroid: np.ndarray = None   # centroid of the top face only (unbiased by visible side faces), world frame


COLOR_RULES = {   # simple, explicit, tweakable
    "red": lambda r, g, b: (r > 110) & (g < 0.45 * r) & (b < 0.45 * r),
    "blue": lambda r, g, b: (b > 110) & (r < 0.45 * b) & (g < 0.6 * b),
}


class PinholeCamera:
    def __init__(self, width, height, fovy_deg):
        self.w, self.h = width, height
        self.fy = 0.5 * height / np.tan(np.radians(fovy_deg) / 2)
        self.fx = self.fy
        self.cx, self.cy = (width - 1) / 2, (height - 1) / 2

    def backproject(self, u, v, depth):
        """MuJoCo camera frame: x right, y up, looks along -z. Returns (N,3) in camera frame."""
        x = (u - self.cx) / self.fx * depth
        y = -(v - self.cy) / self.fy * depth
        z = -depth
        return np.stack([x, y, z], axis=-1)

    def project(self, pc):
        """Camera-frame points (N,3) -> pixel (u, v)."""
        d = -pc[:, 2]
        return self.cx + self.fx * pc[:, 0] / d, self.cy - self.fy * pc[:, 1] / d


def detect(rgb, depth, cam: PinholeCamera, cam_pos, cam_mat, labels=("red", "blue"), min_pixels=30):
    """cam_pos (3,), cam_mat (3,3): camera frame -> world (MuJoCo convention, columns = camera x,y,z axes)."""
    r, g, b = [rgb[..., i].astype(float) for i in range(3)]
    out = []
    for lab in labels:
        mask = COLOR_RULES[lab](r, g, b) & np.isfinite(depth) & (depth > 0.05)
        mask = ndimage.binary_opening(mask, iterations=1)
        cc, n = ndimage.label(mask)
        if n == 0:
            continue
        sizes = ndimage.sum(mask, cc, index=np.arange(1, n + 1))
        k = 1 + int(np.argmax(sizes))
        if sizes[k - 1] < min_pixels:
            continue
        vs, us = np.nonzero(cc == k)
        pc_cam = cam.backproject(us.astype(float), vs.astype(float), depth[vs, us])
        pc = (cam_mat @ pc_cam.T).T + cam_pos
        c = pc.mean(0)
        xy = pc[:, :2] - pc[:, :2].mean(0)
        w, V = np.linalg.eigh(np.cov(xy.T) + 1e-12 * np.eye(2))
        yaw = float(np.arctan2(V[1, 1], V[0, 1]))
        proj = xy @ V
        ext = np.array([np.ptp(proj[:, 1]), np.ptp(proj[:, 0])])
        top_z = float(np.percentile(pc[:, 2], 95))
        top = pc[pc[:, 2] > top_z - 0.008]
        tc = top.mean(0) if len(top) >= 5 else c
        out.append(Detection(lab, c, top_z, yaw, int(sizes[k - 1]),
                             (int(us.min()), int(vs.min()), int(us.max()), int(vs.max())), ext, tc))
    return out
