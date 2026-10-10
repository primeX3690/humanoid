import numpy as np
import pytest
from model.urdf_export import (capsule_inertia_local, geom_mass_inertia, body_inertial, mjcf_to_urdf, validate_urdf, build_default_mjcf, _rot_z_to)
import xml.etree.ElementTree as ET


def test_capsule_with_zero_length_is_a_sphere_and_long_capsule_approaches_a_rod():
    Ip, Ia = capsule_inertia_local(2.0, 0.1, 0.0); assert Ip == pytest.approx(0.4 * 2.0 * 0.01) and Ia == pytest.approx(0.4 * 2.0 * 0.01)
    Ip, Ia = capsule_inertia_local(1.0, 1e-4, 1.0); assert Ip == pytest.approx(1.0 / 12, rel=1e-3) and Ia < 1e-6


def test_box_and_rotated_capsule_inertia():
    g = ET.fromstring('<geom type="box" size="0.1 0.2 0.3" mass="6"/>')
    m, c, I = geom_mass_inertia(g); assert np.allclose(np.diag(I), [6 * (0.16 + 0.36) / 12, 6 * (0.04 + 0.36) / 12, 6 * (0.04 + 0.16) / 12])
    cap = ET.fromstring('<geom type="capsule" fromto="0 0 0 1 0 0" size="0.05" mass="1"/>')     # lying along x
    m, c, I = geom_mass_inertia(cap); assert np.allclose(c, [0.5, 0, 0]) and I[0, 0] < I[1, 1] and I[1, 1] == pytest.approx(I[2, 2])


def test_parallel_axis_two_spheres():
    gs = [ET.fromstring('<geom type="sphere" size="0.1" mass="1" pos="1 0 0"/>'), ET.fromstring('<geom type="sphere" size="0.1" mass="1" pos="-1 0 0"/>')]
    M, com, I = body_inertial(gs); assert M == 2 and np.allclose(com, 0) and I[1, 1] == pytest.approx(2 * (0.4 * 0.01 + 1.0)) and I[0, 0] == pytest.approx(2 * 0.4 * 0.01)


def test_rotation_helper_maps_z_to_direction():
    d = np.array([1.0, 2.0, -0.5]); R = _rot_z_to(d); assert np.allclose(R @ [0, 0, 1], d / np.linalg.norm(d)) and np.allclose(R.T @ R, np.eye(3))


MINI = """<mujoco><worldbody><body name="base" pos="0 0 1"><freejoint name="root"/><geom type="box" size="0.1 0.1 0.1" mass="2"/>
 <body name="arm" pos="0 0.2 0"><joint name="j1" axis="0 1 0" range="-1 2" damping="0.1" armature="0.01"/><geom type="capsule" fromto="0 0 0 0 0 -0.3" size="0.03" mass="0.5"/>
  <body name="hand" pos="0 0 -0.3"><joint name="j2" type="slide" axis="1 0 0" range="0 0.04"/><geom type="sphere" size="0.02" mass="0.1"/></body></body></body></worldbody>
 <actuator><motor name="j1" joint="j1" forcerange="-12 12"/></actuator></mujoco>"""


def test_mini_robot_structure_limits_and_effort():
    u = mjcf_to_urdf(MINI, velocity_fn=lambda n: 4.0); v = validate_urdf(u)
    assert v["connected"] and v["roots"] == ["base"] and v["total_mass"] == pytest.approx(2.6) and v["joint_types"] == {"revolute": 1, "prismatic": 1}
    r = ET.fromstring(u); j1 = [j for j in r.findall("joint") if j.get("name") == "j1"][0]
    lim = j1.find("limit"); assert float(lim.get("lower")) == -1 and float(lim.get("upper")) == 2 and float(lim.get("effort")) == 12 and float(lim.get("velocity")) == 4.0
    assert j1.find("parent").get("link") == "base" and np.allclose([float(x) for x in j1.find("origin").get("xyz").split()], [0, 0.2, 0])


def test_full_humanoid_exports_a_valid_30dof_urdf_with_the_same_mass():
    x = build_default_mjcf(); u = mjcf_to_urdf(x); v = validate_urdf(u)
    assert v["connected"] and v["bad_inertia"] == []
    assert v["total_mass"] == pytest.approx(25.0, abs=1e-6)                               # the simulated robot is a 25 kg robot
    assert v["joint_types"]["revolute"] == 30 and v["joint_types"]["prismatic"] == 2
    assert {"L_knee", "R_hip_yaw", "waist_pitch", "neck_yaw", "L_wr_roll", "R_elbow"} <= set(v["joint_names"])
    assert "imu_link" in u
