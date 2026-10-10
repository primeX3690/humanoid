"""Shared pytest config: headless MuJoCo on Linux, repo root on sys.path, auto-skip of tests whose heavy deps are missing."""
import os, sys, importlib.util
if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pytest
except ImportError:      # allows `python tools/minipytest.py` on machines without pytest
    pytest = None

HEAVY = {"mujoco": "mujoco", "osqp": "osqp", "casadi": "casadi", "torch": "torch", "sb3": "stable_baselines3"}
HAVE = {k: importlib.util.find_spec(v) is not None for k, v in HEAVY.items()}

if pytest is not None:
    def pytest_configure(config):
        for k in HEAVY:
            config.addinivalue_line("markers", f"{k}: needs the optional '{HEAVY[k]}' package")
        config.addinivalue_line("markers", "slow: long-running test (deselect with -m 'not slow')")

    def pytest_collection_modifyitems(config, items):
        for it in items:
            for k, ok in HAVE.items():
                if k in it.keywords and not ok:
                    it.add_marker(pytest.mark.skip(reason=f"optional dependency '{HEAVY[k]}' not installed"))
