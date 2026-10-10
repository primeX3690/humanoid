"""python tools/export_urdf.py [--out export/humanoid.urdf]   -> URDF of the simulated humanoid (+ a validation summary)."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.urdf_export import build_default_mjcf, mjcf_to_urdf, validate_urdf

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="export/humanoid.urdf"); a = ap.parse_args()
    mjcf = build_default_mjcf()                      # (installs a harmless mujoco stub if MuJoCo is absent)
    from model.humanoid_model import joint_speed_limits, ACTUATED
    vel = dict(zip(ACTUATED, joint_speed_limits()))
    urdf = mjcf_to_urdf(mjcf, velocity_fn=lambda n: vel.get(n, 3.0))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True); open(a.out, "w").write(urdf)
    print(json.dumps(validate_urdf(urdf), indent=1)); print("wrote", a.out)
