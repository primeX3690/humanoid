"""python tools/check_env.py  - tells you which layers of the repo can run on this machine."""
import importlib.util, sys
LAYERS = {"core (numpy/scipy)": ["numpy", "scipy", "matplotlib"], "tests": ["pytest"],
          "full-body sim": ["mujoco", "osqp", "casadi"], "RL": ["gymnasium", "stable_baselines3", "torch"]}
ok_all = True
print("python", sys.version.split()[0])
for layer, mods in LAYERS.items():
    miss = [m for m in mods if importlib.util.find_spec(m) is None]
    print(f"{'OK ' if not miss else 'MISSING'} {layer}" + (f"  -> pip install {' '.join(miss)}" if miss else ""))
    ok_all &= not miss
sys.exit(0 if ok_all else 1)
