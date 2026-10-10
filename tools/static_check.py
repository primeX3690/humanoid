"""python tools/static_check.py [paths...]  - offline lint that needs no MuJoCo: (1) undefined global names (typos, missing imports), (2) `from repo_module import name`
where the name does not exist in that module, (3) syntax errors. Exit 1 on findings. Complements (does not replace) running the simulations."""
import ast, builtins, os, sys, symtable

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {".venv", "__pycache__", ".git", "runs", "export"}


def py_files(paths):
    for p in paths:
        if os.path.isfile(p): yield p; continue
        for d, ds, fs in os.walk(p):
            ds[:] = [x for x in ds if x not in SKIP]
            for f in fs:
                if f.endswith(".py"): yield os.path.join(d, f)


def module_names(tree):
    out = set()
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(t):
                    if isinstance(x, ast.Name): out.add(x.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.If, ast.Try, ast.With, ast.For)):
            for x in ast.walk(n):
                if isinstance(x, (ast.FunctionDef, ast.ClassDef)): out.add(x.name)
                elif isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store): out.add(x.id)
                elif isinstance(x, (ast.Import, ast.ImportFrom)):
                    for a in x.names: out.add((a.asname or a.name).split(".")[0])
    return out


def undefined(path, src):
    top = symtable.symtable(src, path, "exec"); defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    defined |= set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
    bad = []

    def walk(t):
        for s in t.get_symbols():
            if t.get_type() != "module" and s.is_global() and s.is_referenced() and s.get_name() not in defined and not s.is_assigned():
                bad.append((t.get_lineno(), s.get_name()))
        for c in t.get_children(): walk(c)
    walk(top); return bad


_cache = {}


def exports(modpath):
    if modpath not in _cache:
        f = os.path.join(ROOT, *modpath.split(".")) 
        cand = [f + ".py", os.path.join(f, "__init__.py")]
        fp = next((c for c in cand if os.path.exists(c)), None)
        if fp is None: _cache[modpath] = None
        else:
            try: tree = ast.parse(open(fp).read()); ex = module_names(tree)
            except SyntaxError: ex = None
            if ex is not None and os.path.isdir(f): ex |= {x[:-3] for x in os.listdir(f) if x.endswith(".py")} | {x for x in os.listdir(f) if os.path.isdir(os.path.join(f, x))}
            _cache[modpath] = ex
    return _cache[modpath]


def bad_imports(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            ex = exports(n.module)
            if ex is None: continue
            for a in n.names:
                if a.name != "*" and a.name not in ex: out.append((n.lineno, f"from {n.module} import {a.name}"))
    return out


def main(paths):
    n_bad = 0
    for f in sorted(py_files(paths or [ROOT])):
        src = open(f).read()
        try: tree = ast.parse(src)
        except SyntaxError as e: print(f"{f}:{e.lineno}: SYNTAX ERROR {e.msg}"); n_bad += 1; continue
        for ln, nm in undefined(f, src): print(f"{os.path.relpath(f, ROOT)}:{ln}: undefined name '{nm}'"); n_bad += 1
        for ln, what in bad_imports(tree): print(f"{os.path.relpath(f, ROOT)}:{ln}: {what}  <- name not found in that module"); n_bad += 1
    print(f"static_check: {n_bad} finding(s)"); return 1 if n_bad else 0


if __name__ == "__main__":
    sys.path.insert(0, ROOT); sys.exit(main(sys.argv[1:]))
