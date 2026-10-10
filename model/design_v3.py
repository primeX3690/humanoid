"""Design preset that resolves the mass/torque blocker found in v3 (see docs/DESIGN_DECISION.md).

`design_v3_params()` returns a ModelParams whose link masses are scaled to the self-consistent robot mass and whose joint torque limits are
the chosen actuators' peaks. The knee uses the HYPOTHETICAL QDD-class entry (hardware/design_search.py) - treat it as a *specification*
(>= ~50 Nm peak, >= ~25 Nm continuous, >= ~3.5 rad/s at 28 kg), not as a part number."""
from __future__ import annotations
import json, os

_BASE_MASS = 25.0
_MASS_FIELDS = ("m_pelvis", "m_torso", "m_head", "m_thigh", "m_shank", "m_foot", "m_small", "m_uarm", "m_farm", "m_hand")


def _fixed_mass():
    """Mass in the model that does NOT scale with the m_* fields (finger/gripper parts): total(s) = s*(25-f) + f, so f = 2*25 - total(2)."""
    from model.urdf_export import build_default_mjcf, mjcf_to_urdf, validate_urdf
    from model.humanoid_model import ModelParams
    p2 = ModelParams(**{m: getattr(ModelParams, m) * 2 for m in _MASS_FIELDS})
    return 2 * _BASE_MASS - validate_urdf(mjcf_to_urdf(build_default_mjcf(p2)))["total_mass"]


def design_v3_params(robot_mass_kg: float = 28.0, zc: float = 0.89, payload_kg: float = 1.0, req_json="results/fullbody/motor_requirements.json", **overrides):
    from model.urdf_export import ensure_importable; ensure_importable()
    from model.humanoid_model import ModelParams
    from hardware.design_search import search, to_model_kwargs
    with open(req_json) as f: req = json.load(f)
    rows = [dict(r, **{k: r[k] * robot_mass_kg / _BASE_MASS for k in ("knee_Nm", "hip_pitch_Nm", "hip_roll_Nm", "hip_yaw_Nm", "ankle_pitch_Nm", "ankle_roll_Nm", "waist_Nm")}) for r in req]
    d = search(rows, zc, payload_kg, use_unverified=True)
    if not d["feasible"]: raise ValueError(d["reason"])
    f = _fixed_mass(); s = (robot_mass_kg - f) / (_BASE_MASS - f)       # scale only the scalable part so the TOTAL hits the target
    kw = {m: getattr(ModelParams, m) * s for m in _MASS_FIELDS}
    kw.update(to_model_kwargs(d)); kw.update(overrides)
    return ModelParams(**kw), d
