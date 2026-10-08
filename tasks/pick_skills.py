"""Behaviour-tree skills for the pick task: gaze, perception, gripper, hand motion (all executed through the WBC)."""
import numpy as np, mujoco
from tasks.behavior_tree import Node, Status, Sequence, Fallback, Retry, Condition
from simulation.wbc_sim import min_jerk
from perception.object_detector import detect

# MuJoCo camera frame (x right, y up, looks -z) expressed in the head/site frame (site x = forward)
CAM_IN_SITE = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)


def camera_pose_from_controller(wbc):
    """Camera pose in world from the CONTROLLER's own state (so estimation error propagates into perception)."""
    sid = wbc.cam_site
    return wbc.d.site_xpos[sid].copy(), wbc.d.site_xmat[sid].reshape(3, 3) @ CAM_IN_SITE


class Ctx:
    """Shared blackboard. `state_fn` -> (qpos, qvel) as seen by the controller (truth or estimator)."""
    def __init__(self, S, renderer, goal, state_fn=None, verbose=True):
        self.S, self.renderer, self.goal = S, renderer, goal
        self.state_fn = state_fn
        self.t = 0.0
        self.verbose = verbose
        self.gaze_target = None
        self.hand = goal.hand or None
        self.hand_target = None            # (pos, R) desired TCP
        self.hand_traj = None
        self.obj = None                    # latest Detection
        self.com_des = None
        self.events = []
        self.lift_log = []
        self.lean_goal, self.lean_cur = 0.45, 0.0   # rad forward torso pitch: extends reach (arm alone is ~5 cm short)
    def log(self, s):
        self.events.append((self.t, s))
        if self.verbose:
            print(s, flush=True)
    def wbc_tasks(self, w):
        self.lean_cur += float(np.clip(self.lean_goal - self.lean_cur, -0.00125, 0.00125))   # 0.25 rad/s ramp @200 Hz
        c, sn = np.cos(self.lean_cur), np.sin(self.lean_cur)
        Ry = np.array([[c, 0, sn], [0, 1, 0], [-sn, 0, c]])
        ts = [w.task_com(self.com_des), w.task_torso_orient(Ry)]
        if self.hand_traj is not None and self.hand:
            pd, vd, Rd = self.hand_traj(self.t)
            ts.append(w.task_hand(self.hand, pd, Rd, vel_des=vd, level=2))
        if self.gaze_target is not None:
            ts.append(w.task_gaze(self.gaze_target, level=3))
        ts.append(w.task_posture(level=4))
        return ts


class Wait(Node):
    def __init__(self, dur, name="wait"):
        self.dur, self.name, self.t0 = dur, name, None
    def tick(self, ctx):
        if self.t0 is None:
            self.t0 = ctx.t
        if ctx.t - self.t0 >= self.dur:
            self.t0 = None
            return Status.SUCCESS
        return Status.RUNNING
    def reset(self):
        self.t0 = None


class LookAt(Node):
    def __init__(self, point_fn, name="LookAtTable", tol=0.30, settle=0.4, timeout=5.0):
        self.point_fn, self.name, self.tol, self.settle, self.timeout = point_fn, name, tol, settle, timeout
        self.t0 = None; self.t_ok = None
    def tick(self, ctx):
        if self.t0 is None:
            self.t0 = ctx.t; self.t_ok = None
            ctx.gaze_target = np.array(self.point_fn(ctx))
        err = ctx.S.wbc.task_gaze(ctx.gaze_target).err
        if err < self.tol:
            self.t_ok = self.t_ok or ctx.t
            if ctx.t - self.t_ok > self.settle:
                self.t0 = None
                return Status.SUCCESS
        else:
            self.t_ok = None
        if ctx.t - self.t0 > self.timeout:
            self.t0 = None
            return Status.FAILURE
        return Status.RUNNING
    def reset(self):
        self.t0 = None


class DetectObject(Node):
    name = "DetectObject"
    def tick(self, ctx):
        rgb, depth = ctx.renderer.grab(ctx.S.d)
        w = ctx.S.wbc
        qpos, qvel = ctx.state_fn() if ctx.state_fn else ctx.S.robot_state()
        w.set_state(qpos, qvel)
        p, Rm = camera_pose_from_controller(w)
        dets = {d.label: d for d in detect(rgb, depth, ctx.renderer.cam, p, Rm)}
        ctx.last_frame = (rgb, depth)
        if ctx.goal.target_label not in dets:
            ctx.log(f"[PERCEPTION] '{ctx.goal.target_label}' object not visible")
            return Status.FAILURE
        d = dets[ctx.goal.target_label]
        ctx.obj = d
        if ctx.hand is None:
            ctx.hand = "L" if d.centroid[1] > 0 else "R"
        ctx.log(f"[PERCEPTION] {d.label} at {np.round(d.centroid, 3)} top_z={d.top_z:.3f} ({d.n_pixels}px) -> hand {ctx.hand}")
        return Status.SUCCESS


class Gripper(Node):
    def __init__(self, open_, name):
        self.open_, self.name, self.t0 = open_, name, None
    def tick(self, ctx):
        side = 0 if ctx.hand == "L" else 1
        if self.t0 is None:
            self.t0 = ctx.t
            ctx.S.grip_cmd[side] = 0.004 if self.open_ else -0.040
        if ctx.t - self.t0 > (0.5 if self.open_ else 1.0):
            self.t0 = None
            return Status.SUCCESS
        return Status.RUNNING
    def reset(self):
        self.t0 = None


class MoveHand(Node):
    """Min-jerk TCP motion to a target built from the latest detection. Success: <6 mm and slow."""
    def __init__(self, target_fn, duration, name, tol=0.008):
        self.target_fn, self.duration, self.name, self.tol = target_fn, duration, name, tol
        self.t0 = None
    def tick(self, ctx):
        S = ctx.S
        if self.t0 is None:
            self.t0 = ctx.t
            p0 = S.tcp_pos(ctx.hand); pT, R = self.target_fn(ctx)
            ctx.hand_target = (pT, R)
            def traj(t, p0=p0, pT=pT, R=R, t0=self.t0, T=self.duration):
                s, v = min_jerk(t - t0, T)
                return p0 + (pT - p0) * s, (pT - p0) * v, R
            ctx.hand_traj = traj
        done = ctx.t - self.t0 >= self.duration
        err = np.linalg.norm(S.tcp_pos(ctx.hand) - ctx.hand_target[0])
        if done and err < self.tol:
            self.t0 = None
            return Status.SUCCESS
        if ctx.t - self.t0 > self.duration + 2.5:
            ctx.log(f"[SKILL] {self.name}: timeout, residual {err*1000:.1f} mm")
            self.t0 = None
            return Status.FAILURE
        return Status.RUNNING
    def reset(self):
        self.t0 = None


class VerifyLift(Node):
    name = "VerifyLift"
    def __init__(self, min_rise=0.07, hold=1.0):
        self.min_rise, self.hold, self.t0 = min_rise, hold, None
    def tick(self, ctx):
        S = ctx.S
        oz = S.d.xpos[mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_BODY, "target_obj")]
        rise = oz[2] - ctx.obj_rest_z
        near = np.linalg.norm(oz - S.tcp_pos(ctx.hand))
        if self.t0 is None:
            self.t0 = ctx.t
        ctx.lift_log.append((ctx.t, float(rise), float(near)))
        if ctx.t - self.t0 >= self.hold:
            self.t0 = None
            ok = rise > self.min_rise and near < 0.05
            ctx.log(f"[VERIFY] object rise {rise*100:.1f} cm, distance to TCP {near*100:.1f} cm -> {'HELD' if ok else 'NOT HELD'}")
            return Status.SUCCESS if ok else Status.FAILURE
        return Status.RUNNING
    def reset(self):
        self.t0 = None


def grasp_yaw(det):
    """Square-ish footprint -> yaw ambiguous -> use 0; elongated -> align jaws across the short axis."""
    major, minor = det.extent
    if major / max(minor, 1e-6) < 1.35:
        return 0.0
    y = det.yaw + np.pi / 2
    return float((y + np.pi / 4) % (np.pi / 2) - np.pi / 4)


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_tree(plan_names, table_center=(0.42, -0.10, 0.80)):
    """Map planner skill names -> BT nodes. Pregrasp: 10 cm above grasp pose; grasp TCP at object centre + 1.4 cm
    (finger tips 4 cm below the TCP must clear the table), centred between the open jaws (1.5 cm clearance each side)."""
    GRASP_DZ = 0.014
    def grasp_pose(ctx, dz=0.0):
        c = ctx.obj.centroid
        z_center = ctx.obj.top_z - 0.03                    # box half-height 3 cm (known object model)
        p = np.array([c[0], c[1], z_center + GRASP_DZ + dz])
        return p, Rz(grasp_yaw(ctx.obj))
    skills = {
        "LookAtTable": lambda: LookAt(lambda ctx: np.array(table_center), "LookAtTable"),
        "ScanForObject": lambda: Fallback([LookAt(lambda ctx: np.array([0.40, -0.25, 0.80]), "ScanRight"),
                                           LookAt(lambda ctx: np.array([0.40, 0.15, 0.80]), "ScanLeft")], "ScanForObject"),
        "DetectObject": lambda: Retry(Sequence([Wait(0.3, "settle"), DetectObject()], "detect_seq"), 3, "DetectRetry"),
        "OpenGripper": lambda: Gripper(True, "OpenGripper"),
        "MoveToPregrasp": lambda: MoveHand(lambda ctx: grasp_pose(ctx, 0.10), 2.5, "MoveToPregrasp"),
        "MoveToGrasp": lambda: MoveHand(lambda ctx: grasp_pose(ctx, 0.0), 1.5, "MoveToGrasp"),
        "CloseGripper": lambda: Gripper(False, "CloseGripper"),
        "LiftObject": lambda: Sequence([MoveHand(lambda ctx: (ctx.hand_target[0] + np.array([0, 0, 0.12]), ctx.hand_target[1]), 2.0, "LiftObject"),
                                        VerifyLift()], "lift_and_verify"),
    }
    return Sequence([skills[n]() for n in plan_names], "pick_task")
