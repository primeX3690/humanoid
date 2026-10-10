"""
Dexterous hand variant (11 DoF per hand: 3 fingers x 3 + thumb x 2) that is ADDED to the existing parallel-jaw hand (so `{s}_grip`, `{s}_tcp` and every controller that
uses them keep working). Opposed layout in the palm frame (fingers point -z, jaws close along y):
    3 fingers on the +y side  (index / middle / ring, 3 flexion joints each: mcp, pip, dip) curling toward -y
    1 thumb   on the -y side  (3 flexion joints) curling toward +y
Enable with  build_xml(..., hand="dexterous").  Finger geoms use contype 4 / conaffinity 5 and explicit <exclude> pairs against the jaws, so they
collide with objects/floor but not with the jaw links they overlap. NOT compiled with MuJoCo in the sandbox this was written in - run
tests/test_hand_model.py + `python -m simulation.sim_smoke --hand` on your machine.
"""
from __future__ import annotations

DIGITS = {  # name: (base_pos in palm frame, phalanx lengths (m), side (+1 curls toward -y), radius)
    "ix": ((0.016, 0.040, -0.040), (0.045, 0.030, 0.022), +1, 0.0085),
    "md": ((0.000, 0.040, -0.040), (0.050, 0.033, 0.024), +1, 0.0085),
    "rg": ((-0.016, 0.040, -0.040), (0.045, 0.030, 0.022), +1, 0.0080),
    "th": ((0.000, -0.040, -0.040), (0.050, 0.034, 0.000), -1, 0.0095),
}
JOINT_RANGE = (0.0, 1.6)
FLEX_JOINTS = {"ix": ("mcp", "pip", "dip"), "md": ("mcp", "pip", "dip"), "rg": ("mcp", "pip", "dip"), "th": ("mcp", "ip")}
PHALANX_MASS = {"mcp": 0.012, "pip": 0.009, "dip": 0.006, "ip": 0.008}


def joint_names(side):
    return [f"{side}_{d}_{j}" for d, js in FLEX_JOINTS.items() for j in js]


def _digit(side, d):
    base, L, sgn, r = DIGITS[d]; js = FLEX_JOINTS[d]
    axis = "-1 0 0" if sgn > 0 else "1 0 0"
    out = []; close = []
    for i, j in enumerate(js):
        pos = f"{base[0]} {base[1]} {base[2]}" if i == 0 else f"0 0 -{L[i - 1]}"
        out.append(f'<body name="{side}_{d}_{j}_b" pos="{pos}"><joint name="{side}_{d}_{j}" axis="{axis}" range="{JOINT_RANGE[0]} {JOINT_RANGE[1]}" armature="0.0005" damping="0.02"/>'
                   f'<geom name="{side}_{d}_{j}_g" type="capsule" fromto="0 0 0 0 0 -{L[i]}" size="{r}" mass="{PHALANX_MASS[j]}" rgba="0.75 0.7 0.65 1" friction="1.2 0.01 0.001" condim="4" contype="4" conaffinity="5"/>')
        close.append("</body>")
    out.append(f'<site name="{side}_{d}_tip" pos="0 0 -{L[len(js) - 1]}" size="0.004" rgba="0 1 0 1"/>')
    return "".join(out) + "".join(reversed(close))


def hand_bodies_xml(side):
    return "".join(_digit(side, d) for d in DIGITS)


def hand_actuators_xml(side, kp=6.0, force=1.5):
    return [f'<position name="{n}" joint="{n}" kp="{kp}" ctrlrange="{JOINT_RANGE[0]} {JOINT_RANGE[1]}" forcerange="-{force} {force}"/>' for n in joint_names(side)]


def hand_contact_xml(side):
    """Exclude finger-vs-jaw/palm pairs (they overlap by design)."""
    ex = []
    for d, js in FLEX_JOINTS.items():
        for j in js:
            for other in (f"{side}_jaw_mov", f"{side}_hand"):
                ex.append(f'<exclude body1="{side}_{d}_{j}_b" body2="{other}"/>')
    return "<contact>" + "".join(ex) + "</contact>"
