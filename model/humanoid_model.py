"""
Full-body humanoid MuJoCo model (legs + waist + neck/head + 7-DOF arms + parallel-jaw hands).

Built programmatically so every number is visible and editable in ONE place (ModelParams).
Total mass is tuned to 25.0 kg (the humanoid-loco budget); arm/torso/head masses are ESTIMATES, leg masses follow the original design.
Actuator torque limits are ASSUMED values (see ModelParams) -- replace with your real
actuator datasheet numbers once the hardware is chosen.

Actuated joint order (30) + 2 grippers:
  L leg: hip_yaw, hip_roll, hip_pitch, knee, ankle_pitch, ankle_roll   (6)
  R leg: same                                                           (6)
  waist_yaw, waist_pitch                                                (2)
  neck_yaw, neck_pitch                                                  (2)
  L arm: sh_pitch, sh_roll, sh_yaw, elbow, wr_yaw, wr_pitch, wr_roll    (7)
  R arm: same                                                           (7)
  L_grip, R_grip (slide, servo-controlled)                              (2)
"""
import os
import sys
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering on Linux; Windows/macOS use MuJoCo defaults
from dataclasses import dataclass
import numpy as np
import mujoco


@dataclass
class ModelParams:
    thigh: float = 0.45
    shank: float = 0.45
    foot_len: float = 0.20
    foot_wid: float = 0.08          # = humanoid-loco LegParams.foot_width_m
    foot_h: float = 0.05            # = humanoid-loco LegParams.foot_height_m
    hip_half_width: float = 0.06
    pelvis_h: float = 0.08
    torso_h: float = 0.42
    shoulder_half_width: float = 0.17
    upper_arm: float = 0.28
    forearm: float = 0.26
    finger: float = 0.09
    m_pelvis: float = 3.0
    m_torso: float = 5.64
    m_head: float = 1.3
    m_thigh: float = 2.4
    m_shank: float = 1.4
    m_foot: float = 0.45
    m_small: float = 0.2
    m_uarm: float = 0.9
    m_farm: float = 0.65
    m_hand: float = 0.4
    tau_hip: float = 130.0
    tau_knee: float = 130.0
    tau_ankle: float = 70.0
    tau_waist: float = 70.0
    tau_neck: float = 8.0
    tau_shoulder: float = 45.0
    tau_elbow: float = 35.0
    tau_wrist: float = 12.0
    armature: float = 0.02
    foot_friction: float = 0.8
    gripper_force: float = 40.0
    dt: float = 0.001


def repo_actuator_params(**kw) -> "ModelParams":
    """Torque limits from the humanoid-loco hardware choice (actuators/motor_specs.py).
    Legs + waist : Dynamixel PH54-200-S500-R  -> 44.7 Nm continuous (official datasheet; its stall column is blank,
                   a distributor lists 73 Nm - NOT used, 44.7 is the only manufacturer-stated figure).
    Shoulders/elbow : ASSUMED PH54-100-S500-R (25.3 Nm continuous) - arms were out of the original scope.
    Wrists/neck  : ASSUMED MX-106-class (8.4 Nm stall)."""
    d = dict(tau_hip=44.7, tau_knee=44.7, tau_ankle=44.7, tau_waist=44.7, tau_shoulder=25.3, tau_elbow=25.3,
             tau_wrist=8.4, tau_neck=8.4)
    d.update(kw)
    return ModelParams(**d)


# no-load joint speed (rad/s) of the same actuators: PH54-200 33.1 rpm, PH54-100 33.3 rpm, MX-106 45 rpm
SPEED_LIMIT = {"leg": 33.1 * 2 * np.pi / 60, "waist": 33.1 * 2 * np.pi / 60, "shoulder": 33.3 * 2 * np.pi / 60,
               "elbow": 33.3 * 2 * np.pi / 60, "wrist": 45 * 2 * np.pi / 60, "neck": 45 * 2 * np.pi / 60}


def joint_speed_limits(names=None):
    names = names or ACTUATED
    out = []
    for n in names:
        b = n[2:] if n[:2] in ("L_", "R_") else n
        if b in LEG: out.append(SPEED_LIMIT["leg"])
        elif b.startswith("waist"): out.append(SPEED_LIMIT["waist"])
        elif b.startswith("neck"): out.append(SPEED_LIMIT["neck"])
        elif b in ("sh_pitch", "sh_roll", "sh_yaw"): out.append(SPEED_LIMIT["shoulder"])
        elif b == "elbow": out.append(SPEED_LIMIT["elbow"])
        else: out.append(SPEED_LIMIT["wrist"])
    return np.array(out)


LEG = ["hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle_pitch", "ankle_roll"]
ARM = ["sh_pitch", "sh_roll", "sh_yaw", "elbow", "wr_yaw", "wr_pitch", "wr_roll"]
ACTUATED = ([f"L_{n}" for n in LEG] + [f"R_{n}" for n in LEG] + ["waist_yaw", "waist_pitch"]
            + ["neck_yaw", "neck_pitch"] + [f"L_{n}" for n in ARM] + [f"R_{n}" for n in ARM])
GRIPPERS = ["L_grip", "R_grip"]

Q_STAND = {"hip_pitch": -0.30, "knee": 0.60, "ankle_pitch": -0.30}
Q_ARM_REST = {"sh_pitch": 0.0, "sh_roll": 0.12, "sh_yaw": 0.0, "elbow": -0.35,
              "wr_yaw": 0.0, "wr_pitch": 0.0, "wr_roll": 0.0}


def build_xml(p: ModelParams = ModelParams(), table: bool = True, object_xyz=(0.42, -0.10, 0.0),
              object_size=(0.025, 0.025, 0.03), object_mass=0.20) -> str:
    def lim(t):
        return f'ctrlrange="-{t} {t}" forcerange="-{t} {t}"'

    def leg(s: str, sgn: float) -> str:
        return f'''
      <body name="{s}_hip_yaw_l" pos="0 {sgn*p.hip_half_width} -0.04">
        <joint name="{s}_hip_yaw" axis="0 0 1" range="-0.6 0.6" armature="{p.armature}"/>
        <geom type="sphere" size="0.03" mass="{p.m_small}" rgba="0.4 0.4 0.5 1" contype="0" conaffinity="0"/>
        <body name="{s}_hip_roll_l">
          <joint name="{s}_hip_roll" axis="1 0 0" range="-0.5 0.5" armature="{p.armature}"/>
          <geom type="sphere" size="0.03" mass="{p.m_small}" rgba="0.4 0.4 0.5 1" contype="0" conaffinity="0"/>
          <body name="{s}_thigh">
            <joint name="{s}_hip_pitch" axis="0 1 0" range="-1.6 0.8" armature="{p.armature}"/>
            <geom type="capsule" fromto="0 0 0 0 0 -{p.thigh}" size="0.05" mass="{p.m_thigh}" rgba="0.6 0.6 0.7 1" contype="0" conaffinity="0"/>
            <body name="{s}_shank" pos="0 0 -{p.thigh}">
              <joint name="{s}_knee" axis="0 1 0" range="0.0 2.4" armature="{p.armature}"/>
              <geom type="capsule" fromto="0 0 0 0 0 -{p.shank}" size="0.04" mass="{p.m_shank}" rgba="0.6 0.6 0.7 1" contype="0" conaffinity="0"/>
              <body name="{s}_ankle_l" pos="0 0 -{p.shank}">
                <joint name="{s}_ankle_pitch" axis="0 1 0" range="-0.9 0.9" armature="{p.armature}"/>
                <geom type="sphere" size="0.025" mass="{p.m_small}" rgba="0.4 0.4 0.5 1" contype="0" conaffinity="0"/>
                <body name="{s}_foot">
                  <joint name="{s}_ankle_roll" axis="1 0 0" range="-0.4 0.4" armature="{p.armature}"/>
                  <geom name="{s}_sole" type="box" size="{p.foot_len/2} {p.foot_wid/2} 0.01"
                        pos="0.03 0 -{p.foot_h-0.01}" mass="{p.m_foot}" friction="{p.foot_friction} 0.005 0.0001"
                        rgba="0.2 0.2 0.2 1" condim="3"/>
                  <site name="{s}_c0" pos="{0.03+p.foot_len/2} {p.foot_wid/2} -{p.foot_h}" size="0.004"/>
                  <site name="{s}_c1" pos="{0.03+p.foot_len/2} -{p.foot_wid/2} -{p.foot_h}" size="0.004"/>
                  <site name="{s}_c2" pos="{0.03-p.foot_len/2} {p.foot_wid/2} -{p.foot_h}" size="0.004"/>
                  <site name="{s}_c3" pos="{0.03-p.foot_len/2} -{p.foot_wid/2} -{p.foot_h}" size="0.004"/>
                  <site name="{s}_sole_site" pos="0.03 0 -{p.foot_h}" size="0.004"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>'''

    def arm(s: str, sgn: float) -> str:
        roll_rng = "-0.3 2.0" if sgn > 0 else "-2.0 0.3"
        return f'''
        <body name="{s}_sh_l1" pos="0 {sgn*p.shoulder_half_width} {p.torso_h-0.07}">
          <joint name="{s}_sh_pitch" axis="0 1 0" range="-3.0 1.5" armature="{p.armature}"/>
          <geom type="sphere" size="0.035" mass="{p.m_small}" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
          <body name="{s}_sh_l2">
            <joint name="{s}_sh_roll" axis="1 0 0" range="{roll_rng}" armature="{p.armature}"/>
            <geom type="sphere" size="0.03" mass="{p.m_small}" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
            <body name="{s}_uarm">
              <joint name="{s}_sh_yaw" axis="0 0 1" range="-1.6 1.6" armature="{p.armature}"/>
              <geom type="capsule" fromto="0 0 0 0 0 -{p.upper_arm}" size="0.035" mass="{p.m_uarm}" rgba="0.7 0.7 0.8 1" contype="0" conaffinity="0"/>
              <body name="{s}_farm" pos="0 0 -{p.upper_arm}">
                <joint name="{s}_elbow" axis="0 1 0" range="-2.5 0.0" armature="{p.armature}"/>
                <geom type="capsule" fromto="0 0 0 0 0 -{p.forearm*0.8}" size="0.03" mass="{p.m_farm}" rgba="0.7 0.7 0.8 1" contype="0" conaffinity="0"/>
                <body name="{s}_wr_l1" pos="0 0 -{p.forearm}">
                  <joint name="{s}_wr_yaw" axis="0 0 1" range="-2.0 2.0" armature="{p.armature}"/>
                  <geom type="sphere" size="0.022" mass="{p.m_small*0.5}" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
                  <body name="{s}_wr_l2">
                    <joint name="{s}_wr_pitch" axis="0 1 0" range="-1.5 1.5" armature="{p.armature}"/>
                    <geom type="sphere" size="0.02" mass="{p.m_small*0.5}" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
                    <body name="{s}_hand">
                      <joint name="{s}_wr_roll" axis="1 0 0" range="-1.5 1.5" armature="{p.armature}"/>
                      <geom name="{s}_palm" type="box" size="0.02 0.05 0.015" pos="0 0 -0.02" mass="{p.m_hand*0.7}"
                            rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0"/>
                      <geom name="{s}_jaw_fixed" type="box" size="0.018 0.006 {p.finger/2}" pos="0 -0.046 -{0.035+p.finger/2}"
                            mass="0.05" rgba="0.2 0.2 0.25 1" friction="1.2 0.01 0.001" condim="4" contype="1" conaffinity="1"/>
                      <site name="{s}_tcp" pos="0 0 -{0.035+p.finger*0.55}" size="0.006" rgba="1 0 0 1"/>
                      <body name="{s}_jaw_mov" pos="0 0.046 0">
                        <joint name="{s}_grip" type="slide" axis="0 1 0" range="-0.040 0.004" armature="0.002" damping="2.0"/>
                        <geom name="{s}_jaw_mov_g" type="box" size="0.018 0.006 {p.finger/2}" pos="0 0 -{0.035+p.finger/2}"
                              mass="0.05" rgba="0.2 0.2 0.25 1" friction="1.2 0.01 0.001" condim="4" contype="1" conaffinity="1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>'''

    acts = []
    tl = {"hip_yaw": p.tau_hip, "hip_roll": p.tau_hip, "hip_pitch": p.tau_hip, "knee": p.tau_knee,
          "ankle_pitch": p.tau_ankle, "ankle_roll": p.tau_ankle}
    for s in ("L", "R"):
        for n in LEG:
            acts.append(f'<motor name="{s}_{n}" joint="{s}_{n}" {lim(tl[n])}/>')
    acts.append(f'<motor name="waist_yaw" joint="waist_yaw" {lim(p.tau_waist)}/>')
    acts.append(f'<motor name="waist_pitch" joint="waist_pitch" {lim(p.tau_waist)}/>')
    acts.append(f'<motor name="neck_yaw" joint="neck_yaw" {lim(p.tau_neck)}/>')
    acts.append(f'<motor name="neck_pitch" joint="neck_pitch" {lim(p.tau_neck)}/>')
    al = {"sh_pitch": p.tau_shoulder, "sh_roll": p.tau_shoulder, "sh_yaw": p.tau_shoulder,
          "elbow": p.tau_elbow, "wr_yaw": p.tau_wrist, "wr_pitch": p.tau_wrist, "wr_roll": p.tau_wrist}
    for s in ("L", "R"):
        for n in ARM:
            acts.append(f'<motor name="{s}_{n}" joint="{s}_{n}" {lim(al[n])}/>')
    for s in ("L", "R"):
        acts.append(f'<position name="{s}_grip" joint="{s}_grip" kp="900" ctrlrange="-0.040 0.004" '
                    f'forcerange="-{p.gripper_force} {p.gripper_force}"/>')

    ox, oy, _ = object_xyz
    hx, hy, hz = object_size
    table_xml = ""
    if table:
        table_xml = f'''
    <body name="table" pos="{ox} {oy} 0">
      <geom name="table_top" type="box" size="0.18 0.25 0.01" pos="0 0 0.78" rgba="0.55 0.4 0.25 1" friction="0.9 0.01 0.001"/>
      <geom type="box" size="0.01 0.01 0.385" pos="0.15 0.22 0.385" rgba="0.4 0.3 0.2 1"/>
      <geom type="box" size="0.01 0.01 0.385" pos="-0.15 0.22 0.385" rgba="0.4 0.3 0.2 1"/>
      <geom type="box" size="0.01 0.01 0.385" pos="0.15 -0.22 0.385" rgba="0.4 0.3 0.2 1"/>
      <geom type="box" size="0.01 0.01 0.385" pos="-0.15 -0.22 0.385" rgba="0.4 0.3 0.2 1"/>
    </body>
    <body name="target_obj" pos="{ox} {oy} {0.791+hz}">
      <freejoint name="obj_free"/>
      <geom name="target_geom" type="box" size="{hx} {hy} {hz}" mass="{object_mass}" rgba="0.9 0.1 0.1 1"
            friction="1.0 0.01 0.001" condim="4"/>
    </body>
    <body name="distractor" pos="{ox+0.02} {oy+0.17} {0.791+0.03}">
      <freejoint name="dist_free"/>
      <geom name="dist_geom" type="cylinder" size="0.03 0.03" mass="0.1" rgba="0.1 0.2 0.9 1" friction="1.0 0.01 0.001"/>
    </body>'''

    return f'''<mujoco model="prime_humanoid">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p.dt}" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual><global offwidth="640" offheight="480"/></visual>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1" friction="{p.foot_friction} 0.005 0.0001"/>
    <body name="pelvis" pos="0 0 1.0">
      <freejoint name="root"/>
      <geom name="pelvis_g" type="box" size="0.06 {p.hip_half_width+0.03} {p.pelvis_h/2}" pos="0 0 0.02" mass="{p.m_pelvis}" rgba="0.3 0.3 0.4 1" contype="0" conaffinity="0"/>
      <site name="imu" pos="0 0 0.02" size="0.01"/>
      {leg("L", 1.0)}
      {leg("R", -1.0)}
      <body name="waist_l1" pos="0 0 {p.pelvis_h/2+0.02}">
        <joint name="waist_yaw" axis="0 0 1" range="-0.8 0.8" armature="{p.armature}"/>
        <geom type="sphere" size="0.04" mass="{p.m_small}" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
        <body name="torso">
          <joint name="waist_pitch" axis="0 1 0" range="-0.5 0.8" armature="{p.armature}"/>
          <geom name="torso_g" type="box" size="0.08 0.15 {p.torso_h/2}" pos="0 0 {p.torso_h/2-0.04}" mass="{p.m_torso}" rgba="0.35 0.35 0.5 1" contype="0" conaffinity="0"/>
          {arm("L", 1.0)}
          {arm("R", -1.0)}
          <body name="neck_l1" pos="0 0 {p.torso_h}">
            <joint name="neck_yaw" axis="0 0 1" range="-1.2 1.2" armature="0.005"/>
            <geom type="sphere" size="0.02" mass="0.1" rgba="0.5 0.5 0.6 1" contype="0" conaffinity="0"/>
            <body name="head">
              <joint name="neck_pitch" axis="0 1 0" range="-0.6 0.9" armature="0.005"/>
              <geom type="sphere" size="0.09" pos="0 0 0.09" mass="{p.m_head}" rgba="0.8 0.8 0.85 1" contype="0" conaffinity="0"/>
              <camera name="head_cam" pos="0.11 0 0.10" xyaxes="0 -1 0 0 0 1" fovy="70"/>
              <site name="cam_site" pos="0.11 0 0.10" size="0.01"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    {table_xml}
  </worldbody>
  <actuator>
    {chr(10).join(acts)}
  </actuator>
  <sensor>
    <accelerometer name="imu_acc" site="imu"/>
    <gyro name="imu_gyro" site="imu"/>
  </sensor>
</mujoco>'''


def make_model(p: ModelParams = ModelParams(), **kw):
    m = mujoco.MjModel.from_xml_string(build_xml(p, **kw))
    return m, mujoco.MjData(m)


def _id(m, kind, n):
    return mujoco.mj_name2id(m, kind, n)


def joint_qadr(m, names):
    return np.array([m.jnt_qposadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])


def joint_dadr(m, names):
    return np.array([m.jnt_dofadr[_id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names])


def nominal_q_act(bend=None):
    """bend: leg flexion b -> hip_pitch=-b, knee=2b, ankle=-b (flat foot). Default 0.3 (CoM ~0.92 m)."""
    q = np.zeros(len(ACTUATED))
    st = dict(Q_STAND) if bend is None else {"hip_pitch": -bend, "knee": 2 * bend, "ankle_pitch": -bend}
    for i, n in enumerate(ACTUATED):
        if n[:2] in ("L_", "R_"):
            base = n[2:]
            if base in st:
                q[i] = st[base]
            elif base in Q_ARM_REST:
                q[i] = Q_ARM_REST[base]
                if base == "sh_roll" and n[0] == "R":
                    q[i] = -Q_ARM_REST[base]
    return q


def stand_pose(m, d, bend=None):
    """Nominal standing pose, soles exactly on the floor. Returns q_act (30,)."""
    mujoco.mj_resetData(m, d)
    qa = joint_qadr(m, ACTUATED)
    q = nominal_q_act(bend)
    d.qpos[qa] = q
    d.qpos[2] = 1.0
    d.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    zs = [d.site_xpos[_id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_c{k}")][2] for s in "LR" for k in range(4)]
    d.qpos[2] += -min(zs) - 0.0003
    mujoco.mj_forward(m, d)
    return q


if __name__ == "__main__":
    m, d = make_model()
    stand_pose(m, d)
    print("nq", m.nq, "nv", m.nv, "nu", m.nu, "mass(no table) %.2f kg" % m.body_subtreemass[1])
