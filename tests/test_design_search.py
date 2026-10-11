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
    assert v["total_mass"] == pytest.approx(28.0, abs=1e-3) and p.tau_knee >= 50.0   # URDF masses are written with 6 significant digits


def test_evaluate_design_matrix_boundaries_for_the_32kg_decision():
    from hardware.design_search import evaluate_design
    assert not evaluate_design(REQ, 25, 0.89, 1.0, True, 1.04, 0.25)["closes"]                  # the original 25 kg robot does not close
    e = evaluate_design(REQ, 32, 0.89, 1.0, True, 1.04, 0.25)
    assert e["closes"] and e["structure_pct"] >= 25 and 55 < e["knee_peak_needed"] < 58 and e["choice"]["knee"] == "QDD-hyp"
    assert not evaluate_design(REQ, 30, 0.89, 1.0, True, 1.04, 0.25)["closes"] and evaluate_design(REQ, 30, 0.89, 1.0, True, 1.04, 0.20)["closes"]
    from hardware.design_search import GROUPS
    nowrist = evaluate_design(REQ, 32, 0.89, 1.0, True, 1.04, 0.25, groups=dict(GROUPS, wrist=0))
    assert nowrist["actuator_kg"] < e["actuator_kg"] and e["actuator_kg"] - nowrist["actuator_kg"] < 1.5     # dropping wrists saves < 1.5 kg: not worth the DoF


def test_measured_32kg_demand_confirms_the_mass_scaling_assumption_and_the_32kg_decision():
    from hardware.design_search import evaluate_design
    meas = json.load(open("results/design_requirements_measured32.json"))
    base = {r["com_height_m"]: r for r in REQ}
    for r in meas:                                            # M/25 scaling was within 10 % for every joint group that matters
        for k in ("knee_Nm", "hip_pitch_Nm", "ankle_pitch_Nm", "waist_Nm"):
            assert r[k] == pytest.approx(base[r["com_height_m"]][k] * 32 / 25, rel=0.1), k
    e = evaluate_design(meas, 32, 0.89, 1.0, True, 1.04, 0.25, ref_mass=32.0)
    assert e["closes"] and e["choice"]["knee"] == "QDD-hyp" and 56 < e["knee_peak_needed"] < 58 and e["structure_pct"] >= 25
