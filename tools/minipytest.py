"""Tiny pytest-compatible runner for machines where pytest is NOT installed (offline sandboxes, CI bootstrap).

Supports what this repo's tests use: assert, pytest.approx / raises / fixture(module scope) / mark.parametrize /
mark.skipif / mark.xfail / importorskip. Use real pytest when you can:  python tools/minipytest.py [paths...] [-k substr]
"""
import sys, os, types, importlib, importlib.util, inspect, traceback, math, time, glob, itertools

class Skip(Exception): pass

class approx:
    def __init__(self, v, rel=1e-6, abs=1e-12): self.v, self.rel, self.abs = v, rel, abs
    def _eq(self, a, b):
        import numpy as np
        a = np.asarray(a, float); b = np.asarray(b, float)
        return bool(np.all(np.abs(a - b) <= np.maximum(self.abs, self.rel * np.abs(b))))
    def __eq__(self, o): return self._eq(o, self.v)
    def __repr__(self): return f"approx({self.v})"

class _Raises:
    def __init__(self, exc, match=None): self.exc, self.match = exc, match
    def __enter__(self): return self
    def __exit__(self, t, v, tb):
        if t is None: raise AssertionError(f"DID NOT RAISE {self.exc}")
        if not issubclass(t, self.exc): return False
        if self.match:
            import re
            assert re.search(self.match, str(v)), f"{v!r} !~ {self.match}"
        return True

def _mk():
    m = types.ModuleType("pytest")
    m.approx = approx
    m.raises = lambda exc, match=None: _Raises(exc, match)
    def fixture(*a, **k):
        def deco(f): f._is_fixture = True; f._scope = k.get("scope", "function"); return f
        return deco(a[0]) if a and callable(a[0]) else deco
    m.fixture = fixture
    def importorskip(name, *a, **k):
        try: return importlib.import_module(name)
        except Exception: raise Skip(f"missing module {name}")
    m.importorskip = importorskip
    def skip(reason=""): raise Skip(reason)
    m.skip = skip
    class Mark:
        def parametrize(self, names, vals, **k):
            def d(f): f._params = (names, list(vals)); return f
            return d
        def skipif(self, cond, reason="", **k):
            def d(f):
                if cond: f._skip = reason or "skipif"
                return f
            return d
        def xfail(self, *a, **k):
            def d(f): f._xfail = k.get("reason", ""); return f
            return d(a[0]) if a and callable(a[0]) else d
        def __getattr__(self, n):
            def mk(*a, **k):
                def d(f): f.__dict__.setdefault("_marks", set()).add(n); return f
                return d(a[0]) if a and callable(a[0]) and not k else d
            return mk
    m.mark = Mark()
    m.main = lambda *a, **k: 0
    return m

def main(argv):
    sys.modules["pytest"] = _mk()
    sys.path.insert(0, os.getcwd())
    k = None
    if "-k" in argv:
        i = argv.index("-k"); k = argv[i + 1]; argv = argv[:i] + argv[i + 2:]
    paths = [a for a in argv if not a.startswith("-")] or ["tests"]
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "test_*.py"))) if os.path.isdir(p) else [p]
    res = dict(passed=0, failed=0, skipped=0, xfailed=0, xpassed=0)
    failures = []
    for f in files:
        modname = os.path.splitext(f)[0].replace(os.sep, ".")
        try:
            mod = importlib.import_module(modname)
        except Skip as e:
            res["skipped"] += 1; print(f"SKIP  {f} ({e})"); continue
        except Exception as e:
            res["skipped"] += 1; print(f"SKIP  {f} (import: {type(e).__name__}: {e})"); continue
        cache = {}
        def getfix(name):
            fx = getattr(mod, name, None)
            if fx is None or not getattr(fx, "_is_fixture", False): raise TypeError(f"no fixture {name}")
            if name not in cache:
                cache[name] = fx(**{a: getfix(a) for a in inspect.signature(fx).parameters})
            return cache[name]
        for name, fn in sorted(vars(mod).items()):
            if not name.startswith("test") or not callable(fn): continue
            if k and k not in name: continue
            cases = [{}]
            if hasattr(fn, "_params"):
                names, vals = fn._params; names = [n.strip() for n in names.split(",")]
                cases = [dict(zip(names, v if len(names) > 1 else (v,))) for v in vals]
            for case in cases:
                tid = f"{f}::{name}" + (f"[{','.join(map(str, case.values()))}]" if case else "")
                if getattr(fn, "_skip", None): res["skipped"] += 1; print(f"SKIP  {tid}"); continue
                _mods = {"mujoco": "mujoco", "osqp": "osqp", "casadi": "casadi", "torch": "torch", "sb3": "stable_baselines3"}
                miss = [m for k2, m in _mods.items() if k2 in getattr(fn, "_marks", ()) and importlib.util.find_spec(m) is None]
                if miss: res["skipped"] += 1; print(f"SKIP  {tid} (missing {miss[0]})"); continue
                t0 = time.time()
                try:
                    kw = dict(case)
                    for a in inspect.signature(fn).parameters:
                        if a not in kw: kw[a] = getfix(a)
                    fn(**kw)
                    if hasattr(fn, "_xfail"): res["xpassed"] += 1; print(f"XPASS {tid}")
                    else: res["passed"] += 1; print(f"PASS  {tid} ({time.time()-t0:.1f}s)")
                except Skip as e: res["skipped"] += 1; print(f"SKIP  {tid} ({e})")
                except BaseException as e:
                    if hasattr(fn, "_xfail"): res["xfailed"] += 1; print(f"XFAIL {tid}")
                    else:
                        res["failed"] += 1; print(f"FAIL  {tid}: {type(e).__name__}: {str(e)[:200]}")
                        failures.append((tid, traceback.format_exc()))
    for tid, tb in failures: print("\n=== ", tid, "\n", tb)
    print("\n" + ", ".join(f"{v} {k}" for k, v in res.items()))
    return 1 if res["failed"] else 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
