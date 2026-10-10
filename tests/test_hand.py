import numpy as np
import pytest
import xml.etree.ElementTree as ET
from model.hand_model import DIGITS, FLEX_JOINTS, joint_names, hand_actuators_xml, hand_bodies_xml, hand_contact_xml
from manipulation.hand.finger_kinematics import tip, tip_jacobian, joint_positions
from manipulation.hand.grasp_analysis import is_force_closure, epsilon_quality, distribute_forces, friction_cone_edges
from manipulation.hand.hand_grasp import Obj, plan_grasp, close_digit, q_of, hold_forces


def test_default_xml_is_unchanged_and_dexterous_xml_adds_22_joints_after_the_first_32_actuators():
    from model.urdf_export import ensure_importable; ensure_importable()
    from model.humanoid_model import build_xml, ModelParams
    a = ET.fromstring(build_xml(ModelParams(), table=False)); b = ET.fromstring(build_xml(ModelParams(), table=False, hand="dexterous"))
    assert a.find("contact") is None
    na = [e.get("name") for e in a.find("actuator")]; nb = [e.get("name") for e in b.find("actuator")]
    assert nb[:len(na)] == na and len(nb) == len(na) + 22 and len(set(nb)) == len(nb)
    ja = {j.get("name") for j in a.iter("joint")}; jb = [j.get("name") for j in b.iter("joint")]
    assert len(jb) == len(set(jb)) == len(ja) + 22 and set(joint_names("L")) <= set(jb)
    assert all(float(g.get("mass", 0)) > 0 for g in b.iter("geom") if g.get("name", "").startswith("L_ix"))


def test_hand_xml_is_well_formed_and_exclusions_reference_existing_bodies():
    root = ET.fromstring("<r>" + hand_bodies_xml("L") + "</r>"); bodies = {b.get("name") for b in root.iter("body")}
    assert len(bodies) == sum(len(v) for v in FLEX_JOINTS.values())
    ex = ET.fromstring(hand_contact_xml("L")); assert all(e.get("body1") in bodies for e in ex.iter("exclude"))
    assert len(hand_actuators_xml("L")) == 11


def test_finger_fk_matches_straight_and_curled_geometry_and_jacobian():
    L = DIGITS["ix"][1]; base = np.array(DIGITS["ix"][0])
    assert np.allclose(tip("ix", [0, 0, 0]), base + [0, 0, -sum(L)])
    t = tip("ix", [np.pi / 2, 0, 0]); assert t[1] < base[1] - 0.05 and t[2] > base[2] - 0.05       # fingers curl toward -y
    assert tip("th", [np.pi / 2, 0])[1] > DIGITS["th"][0][1] + 0.04                                  # thumb curls toward +y
    q = np.array([0.4, 0.5, 0.3]); J = tip_jacobian("md", q); e = 1e-6
    Jn = np.stack([(tip("md", q + e * np.eye(3)[i]) - tip("md", q - e * np.eye(3)[i])) / (2 * e) for i in range(3)], 1)
    assert np.allclose(J, Jn, atol=1e-6)


def test_force_closure_known_cases():
    r = 0.04
    tri = [(r * np.array([np.cos(a), np.sin(a), 0]), -np.array([np.cos(a), np.sin(a), 0])) for a in (0, 2 * np.pi / 3, 4 * np.pi / 3)]
    assert is_force_closure(tri, 0.5)                             # 3 frictional contacts around a sphere equator DO close (z-forces via friction)
    assert not is_force_closure(tri, 0.0001)                      # ...but not when frictionless
    tet = [(r * np.array(v) / np.linalg.norm(v), -np.array(v) / np.linalg.norm(v)) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    assert is_force_closure(tet, 0.5) and epsilon_quality(tet, 0.5) > 0.0
    two = [(np.array([r, 0, 0]), np.array([-1.0, 0, 0])), (np.array([-r, 0, 0]), np.array([1.0, 0, 0]))]
    assert not is_force_closure(two, 0.5)
    assert epsilon_quality(tet, 0.8) > epsilon_quality(tet, 0.3)       # more friction -> better quality


def test_force_distribution_balances_weight_inside_friction_cones():
    r = 0.04; tet = [(r * np.array(v) / np.linalg.norm(v), -np.array(v) / np.linalg.norm(v)) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
    w = np.array([0, 0, -9.81, 0, 0, 0]); F, ok = distribute_forces(tet, 0.5, w)
    assert ok and np.allclose(F.sum(0), [0, 0, 9.81], atol=1e-4)
    for (p, n), f in zip(tet, F):
        fn = f @ n; assert fn >= -1e-9 and np.linalg.norm(f - fn * n) <= 0.5 * fn + 1e-6


def test_planner_wraps_a_mid_size_sphere_and_rejects_oversized_or_distant_objects():
    ball = Obj("sphere", np.array([0.0, 0.0, -0.085]), 0.030)
    p = plan_grasp(ball); assert p["success"] and set(p["joint_targets"]) == {"ix", "md", "rg", "th"} and p["quality"] > 0
    for d, q in p["joint_targets"].items():
        from manipulation.hand.finger_kinematics import surface_points
        dmin = min(np.linalg.norm(pt - ball.center) for pt in surface_points(d, q)); assert abs(dmin - (ball.radius + 0.008)) < 3e-3
    assert not plan_grasp(Obj("sphere", np.array([0, 0, -0.085]), 0.09))["success"]
    assert not plan_grasp(Obj("sphere", np.array([0.0, 0.0, -0.40]), 0.03))["success"]


def test_holding_force_scales_with_mass():
    p = plan_grasp(Obj("sphere", np.array([0.0, 0.0, -0.085]), 0.030))
    F1, ok1 = hold_forces(p, 0.2); F2, ok2 = hold_forces(p, 0.4)
    assert ok1 and ok2 and np.linalg.norm(F2) == pytest.approx(2 * np.linalg.norm(F1), rel=1e-3)
