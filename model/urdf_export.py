"""
MJCF -> URDF exporter (pure Python, no MuJoCo needed to RUN it; only to *generate* the MJCF if you start from ModelParams).

Why: the humanoid is defined in code (model/humanoid_model.py). Real hardware tooling (ROS 2, MoveIt, Pinocchio/RBDL, CAD import,
Isaac/Gazebo) wants a URDF. This turns the exact simulated robot into one, with inertias computed from the primitives.
Mapping: body -> link (mass/COM/inertia from its geoms, parallel-axis), joint -> revolute/prismatic with limits, effort (from the
<motor>/<position> force range) and velocity (callable), sites 'imu' & '*_tcp' and cameras -> fixed frames.
NOT in URDF (kept as comments): MuJoCo `armature`, contact params. Capsules are exported as cylinder + 2 spheres (URDF has no capsule).
"""
from __future__ import annotations
import sys, types
import numpy as np
import xml.etree.ElementTree as ET


def build_default_mjcf(params=None, **kw) -> str:
    if "mujoco" not in sys.modules:
        try:
            import mujoco  # noqa: F401
        except ImportError:
            sys.modules["mujoco"] = types.ModuleType("mujoco")      # build_xml is plain string generation; it never calls MuJoCo
    from model.humanoid_model import build_xml, ModelParams
    return build_xml(params or ModelParams(), table=False, **kw)


def _v(s, n=3, default=0.0):
    return np.array([float(x) for x in s.split()]) if s else np.full(n, default)


def _fmt(a): return " ".join(f"{x:.6g}" for x in a)


def _rot_z_to(d):
    d = d / np.linalg.norm(d); z = np.array([0, 0, 1.0])
    v = np.cross(z, d); c = z @ d
    if np.linalg.norm(v) < 1e-12: return np.eye(3) if c > 0 else np.diag([1, -1, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def _rpy(R):
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-9: return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arctan2(-R[2, 0], sy), np.arctan2(R[1, 0], R[0, 0])])
    return np.array([np.arctan2(-R[1, 2], R[1, 1]), np.arctan2(-R[2, 0], sy), 0.0])


def capsule_inertia_local(m, r, h):
    """Capsule (axis z): cylinder length h + 2 hemispheres. Returns (Ixx=Iyy, Izz) about the centre."""
    vc, vh = np.pi * r * r * h, 2.0 / 3.0 * np.pi * r ** 3
    rho = m / (vc + 2 * vh); mc, mh = rho * vc, rho * vh
    I_perp = mc * (3 * r * r + h * h) / 12 + 2 * (83.0 / 320.0 * mh * r * r + mh * (h / 2 + 3 * r / 8) ** 2)
    I_ax = mc * r * r / 2 + 2 * 0.4 * mh * r * r
    return I_perp, I_ax


def geom_mass_inertia(g):
    """-> (mass, center (3,), inertia about its own centre in the BODY frame (3x3))"""
    t = g.get("type", "sphere"); m = float(g.get("mass", 0.0)); sz = _v(g.get("size"), 3)
    if t == "sphere":
        r = sz[0]; return m, _v(g.get("pos")), np.eye(3) * 0.4 * m * r * r
    if t == "box":
        a, b, c = 2 * sz[:3]
        return m, _v(g.get("pos")), np.diag([m * (b * b + c * c) / 12, m * (a * a + c * c) / 12, m * (a * a + b * b) / 12])
    if t == "capsule":
        r = sz[0]
        if g.get("fromto"):
            ft = _v(g.get("fromto"), 6); p0, p1 = ft[:3], ft[3:]
        else:
            h2 = sz[1]; p0, p1 = _v(g.get("pos")) - [0, 0, h2], _v(g.get("pos")) + [0, 0, h2]
        d = p1 - p0; h = np.linalg.norm(d); R = _rot_z_to(d)
        Ip, Ia = capsule_inertia_local(m, r, h)
        return m, 0.5 * (p0 + p1), R @ np.diag([Ip, Ip, Ia]) @ R.T
    raise ValueError(f"unsupported geom type {t}")


def body_inertial(geoms):
    ms = [geom_mass_inertia(g) for g in geoms if g.get("type") != "plane" and float(g.get("mass", 0)) > 0]
    if not ms: return 1e-6, np.zeros(3), np.eye(3) * 1e-9
    M = sum(m for m, _, _ in ms); com = sum(m * c for m, c, _ in ms) / M
    I = np.zeros((3, 3))
    for m, c, Ig in ms:
        d = c - com; I += Ig + m * (d @ d * np.eye(3) - np.outer(d, d))
    return M, com, I


def _shape_xml(g, tag):
    t = g.get("type"); sz = _v(g.get("size"), 3); rgba = g.get("rgba", "0.7 0.7 0.7 1")
    mat = f'<material name="m_{abs(hash(rgba)) % 99991}"><color rgba="{rgba}"/></material>' if tag == "visual" else ""
    def one(geom_xml, xyz, rpy=(0, 0, 0)):
        return f'<{tag}><origin xyz="{_fmt(xyz)}" rpy="{_fmt(rpy)}"/><geometry>{geom_xml}</geometry>{mat}</{tag}>'
    if t == "sphere": return [one(f'<sphere radius="{sz[0]:.6g}"/>', _v(g.get("pos")))]
    if t == "box": return [one(f'<box size="{_fmt(2 * sz[:3])}"/>', _v(g.get("pos")))]
    if t == "capsule":
        if g.get("fromto"):
            ft = _v(g.get("fromto"), 6); p0, p1 = ft[:3], ft[3:]
        else:
            h2 = sz[1]; p0, p1 = _v(g.get("pos")) - [0, 0, h2], _v(g.get("pos")) + [0, 0, h2]
        d = p1 - p0; h = np.linalg.norm(d); R = _rot_z_to(d); rpy = _rpy(R); r = sz[0]
        return [one(f'<cylinder radius="{r:.6g}" length="{h:.6g}"/>', 0.5 * (p0 + p1), rpy),
                one(f'<sphere radius="{r:.6g}"/>', p0), one(f'<sphere radius="{r:.6g}"/>', p1)]
    return []


def mjcf_to_urdf(mjcf: str, robot_name="humanoid", velocity_fn=None, default_effort=100.0, default_velocity=3.0) -> str:
    root = ET.fromstring(mjcf)
    effort = {}
    for a in root.iter():
        if a.tag in ("motor", "position") and a.get("joint"):
            fr = a.get("forcerange") or a.get("ctrlrange")
            if fr: effort[a.get("joint")] = max(abs(float(x)) for x in fr.split())
    out = [f'<?xml version="1.0"?>', f'<robot name="{robot_name}">']
    wb = root.find("worldbody"); top = [b for b in wb.findall("body")]
    if len(top) != 1: raise ValueError("expected exactly one root body")

    def link(name, geoms, extra=""):
        M, com, I = body_inertial(geoms)
        s = [f'<link name="{name}"><inertial><origin xyz="{_fmt(com)}" rpy="0 0 0"/><mass value="{M:.6g}"/>'
             f'<inertia ixx="{I[0,0]:.6g}" ixy="{I[0,1]:.6g}" ixz="{I[0,2]:.6g}" iyy="{I[1,1]:.6g}" iyz="{I[1,2]:.6g}" izz="{I[2,2]:.6g}"/></inertial>']
        for g in geoms:
            if g.get("type") == "plane": continue
            s += _shape_xml(g, "visual")
            if g.get("contype") != "0" or g.get("name", "").endswith(("sole", "palm")): s += _shape_xml(g, "collision")
        s.append("</link>"); return "".join(s)

    def fixed(name, parent, xyz, rpy=(0, 0, 0)):
        return (f'<link name="{name}"/><joint name="{name}_fixed" type="fixed"><parent link="{parent}"/><child link="{name}"/>'
                f'<origin xyz="{_fmt(xyz)}" rpy="{_fmt(rpy)}"/></joint>')

    def emit(body, parent):
        name = body.get("name"); pos = _v(body.get("pos"))
        joints = [j for j in body.findall("joint")]
        geoms = body.findall("geom")
        if parent is None:
            out.append(link(name, geoms))
        else:
            # chain of joints: all but the last go through massless intermediate links
            prev = parent; xyz = pos
            for k, j in enumerate(joints):
                last = k == len(joints) - 1
                child = name if last else f"{name}__j{k}"
                out.append(link(name, geoms) if last else f'<link name="{child}"/>')
                jt = "prismatic" if j.get("type") == "slide" else "revolute"
                lo, hi = (float(x) for x in j.get("range", "-3.1416 3.1416").split())
                vel = velocity_fn(j.get("name")) if velocity_fn else default_velocity
                out.append(f'<joint name="{j.get("name")}" type="{jt}"><parent link="{prev}"/><child link="{child}"/><origin xyz="{_fmt(xyz)}" rpy="0 0 0"/>'
                           f'<axis xyz="{_fmt(_v(j.get("axis"), 3))}"/><limit lower="{lo:.6g}" upper="{hi:.6g}" effort="{effort.get(j.get("name"), default_effort):.6g}" velocity="{float(vel):.6g}"/>'
                           f'<dynamics damping="{float(j.get("damping", 0)):.6g}" friction="0"/></joint><!-- mujoco armature={j.get("armature", "0")} -->')
                prev = child; xyz = np.zeros(3)
            if not joints:
                out.append(link(name, geoms))
                out.append(f'<joint name="{name}_fixed" type="fixed"><parent link="{parent}"/><child link="{name}"/><origin xyz="{_fmt(pos)}" rpy="0 0 0"/></joint>')
        for s in body.findall("site"):
            if s.get("name") == "imu" or s.get("name", "").endswith("_tcp"): out.append(fixed(s.get("name") + "_link", name, _v(s.get("pos"))))
        for c in body.findall("camera"):
            xy = _v(c.get("xyaxes"), 6); x, y = xy[:3], xy[3:]; z = np.cross(x, y)
            out.append(fixed(c.get("name") + "_link", name, _v(c.get("pos")), _rpy(np.column_stack([x, y, z]))))
        for ch in body.findall("body"): emit(ch, name)

    emit(top[0], None)
    out.append("</robot>")
    return "\n".join(out)


def validate_urdf(urdf: str) -> dict:
    r = ET.fromstring(urdf)
    links = {l.get("name"): l for l in r.findall("link")}; joints = r.findall("joint")
    children = {j.find("child").get("link") for j in joints}; parents = {j.find("parent").get("link") for j in joints}
    roots = [n for n in links if n not in children]
    mass = 0.0; bad = []
    for n, l in links.items():
        i = l.find("inertial")
        if i is None: continue
        mass += float(i.find("mass").get("value"))
        a = i.find("inertia"); I = np.array([[float(a.get(k)) for k in ("ixx", "ixy", "ixz")], [float(a.get("ixy")), float(a.get("iyy")), float(a.get("iyz"))],
                                              [float(a.get("ixz")), float(a.get("iyz")), float(a.get("izz"))]])
        w = np.linalg.eigvalsh(I)
        if w.min() <= 0 or w[2] > w[0] + w[1] + 1e-9: bad.append(n)
    types_ = {}
    for j in joints: types_[j.get("type")] = types_.get(j.get("type"), 0) + 1
    return dict(n_links=len(links), n_joints=len(joints), joint_types=types_, roots=roots, total_mass=mass, bad_inertia=bad,
                joint_names=[j.get("name") for j in joints if j.get("type") != "fixed"], connected=len(roots) == 1 and set(children) <= set(links) and parents <= set(links))
