"""Maintainer tool: writes MANIFEST.json (sha256 of every source/doc file, line endings normalised) so tools/verify_install.py can detect stale files."""
import hashlib, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP = {"results", "runs", "export", "__pycache__", ".git", "venv", ".venv", "env", "site-packages"}
EXT = (".py", ".md", ".txt", ".ini")


def digest(path):
    b = open(path, "rb").read().replace(b"\r\n", b"\n"); return hashlib.sha256(b).hexdigest()[:16]


def files():
    for d, ds, fs in os.walk(ROOT):
        ds[:] = [x for x in ds if x not in SKIP]
        for f in fs:
            if f.endswith(EXT) and f != "MANIFEST.json": yield os.path.relpath(os.path.join(d, f), ROOT).replace(os.sep, "/")


if __name__ == "__main__":
    m = {f: digest(os.path.join(ROOT, f)) for f in sorted(files())}
    json.dump(m, open(os.path.join(ROOT, "MANIFEST.json"), "w"), indent=0); print(f"MANIFEST.json: {len(m)} files")
