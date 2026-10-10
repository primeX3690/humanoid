import json
import pytest
from hardware.design_search import search, self_consistent_mass, feasible, GROUPS, CATALOG, to_model_kwargs
from actuators.joint_actuator import PH54_200

REQ = json.load(open("results/fullbody/motor_requirements.json"))


def test_group_counts_cover_all_30_joints():
    assert sum(GROUPS.values()) == 30


def test_knee_is_the_binding_constraint_and_ph54_200_is_only_feasible_with_thin_margin_at_089():
    from hardware.design_search import Demand
    assert feasible(PH54_200, Demand(42.4, 2.4), margin=1.04) and not feasible(PH54_200, Demand(42.4, 2.4), margin=1.15)
    assert not feasible(PH54_200, Demand(52.2, 2.1), margin=1.04)           # CoM 0.85 m: knee 52 Nm


def test_verified_catalogue_cannot_close_the_mass_loop_but_a_stronger_knee_can():
    best_v, _ = self_consistent_mass(REQ, 0.89, 1.0, use_unverified=False)
    assert best_v is None
    best_u, _ = self_consistent_mass(REQ, 0.89, 1.0, use_unverified=True)
    assert best_u is not None and 26.0 < best_u["robot_mass_kg"] < 31.0 and best_u["choice"]["knee"] == "QDD-hyp"
    assert best_u["robot_mass_kg"] >= best_u["mass_needed_kg"] - 1e-9


def test_design_preset_scales_total_mass_to_target():
    from model.urdf_export import build_default_mjcf, mjcf_to_urdf, validate_urdf
    from model.design_v3 import design_v3_params
    p, d = design_v3_params(28.0)
    v = validate_urdf(mjcf_to_urdf(build_default_mjcf(p)))
    assert v["total_mass"] == pytest.approx(28.0, abs=1e-6) and p.tau_knee >= 50.0
