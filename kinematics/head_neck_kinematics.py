"""Neck (yaw, pitch) look-at kinematics and torso+neck gaze coupling (pure NumPy)."""
import numpy as np
from kinematics.arm_kinematics import _rot, _T


class HeadNeck:
    def __init__(self, torso_h=0.42, cam_pos_in_head=(0.11, 0.0, 0.10), pitch_lim=(-0.6, 0.9), yaw_lim=(-1.2, 1.2)):
        self.torso_h = torso_h
        self.cam = np.array(cam_pos_in_head)
        self.pitch_lim, self.yaw_lim = pitch_lim, yaw_lim

    def cam_pose(self, yaw, pitch):
        """Camera pose in the torso frame. Camera forward axis = +x of the returned rotation."""
        T = _T(p=[0, 0, self.torso_h]) @ _T(R=_rot("z", yaw)) @ _T(R=_rot("y", pitch)) @ _T(p=self.cam)
        return T

    def look_at(self, target_torso):
        """Analytic yaw/pitch so camera forward axis points at target (torso frame). Returns (yaw, pitch, reachable)."""
        base = np.array([0, 0, self.torso_h])
        # first solve ignoring the small camera offset, then refine twice (offset depends on yaw/pitch)
        yaw, pitch = 0.0, 0.0
        for _ in range(30):
            cam = self.cam_pose(yaw, pitch)[:3, 3]
            d = target_torso - cam
            yaw = np.arctan2(d[1], d[0])
            # NOTE: positive pitch rotates +x toward -z (right-hand rule about +y) => pitch = atan2(-dz, horiz)
            pitch = np.arctan2(-d[2], np.hypot(d[0], d[1]))
        ok = (self.yaw_lim[0] <= yaw <= self.yaw_lim[1]) and (self.pitch_lim[0] <= pitch <= self.pitch_lim[1])
        yaw = float(np.clip(yaw, *self.yaw_lim)); pitch = float(np.clip(pitch, *self.pitch_lim))
        return yaw, pitch, bool(ok)

    def gaze_error(self, yaw, pitch, target_torso):
        T = self.cam_pose(yaw, pitch)
        d = target_torso - T[:3, 3]
        d /= np.linalg.norm(d)
        return float(np.arccos(np.clip(T[:3, 0] @ d, -1, 1)))
