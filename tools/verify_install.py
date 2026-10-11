"""python tools/verify_install.py  - compares YOUR tree with the manifest of the latest delivery and lists files that are STALE (different content,
e.g. an 'overwrite' that was skipped while unzipping) or MISSING. Run it after unzipping any update."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_manifest import ROOT, digest, files

if __name__ == "__main__":
    mf = os.path.join(ROOT, "MANIFEST.json")
    if not os.path.exists(mf): print("MANIFEST.json not found - unzip the latest update first"); sys.exit(2)
    want = json.load(open(mf)); have = set(files()); bad = 0
    for f, h in sorted(want.items()):
        p = os.path.join(ROOT, f)
        if not os.path.exists(p): print("MISSING", f); bad += 1
        elif digest(p) != h: print("STALE  ", f, " <- different from the delivered version (skipped overwrite, or locally edited)"); bad += 1
    print(f"verify_install: {bad} problem file(s) of {len(want)}"); sys.exit(1 if bad else 0)
