from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from .objects import ObjectSpec, object_xml

# Body-local collision proxies on the glove.  These are deliberately simple
# spheres (and one palm box) rather than 93 visual meshes: the visual meshes
# stay unchanged, but contacts are cheap and stable.  Positions are the visual
# geom centres from hand_closed_chain.xml.
GLOVE_COLLISIONS: tuple[tuple[str, str, tuple[float, float, float], float, str], ...] = (
    ("base_link", "palm", (0.0334, -0.0015, 0.0050), 0.030, "box"),
    ("thumb_mcp_link", "thumb", (0.0147, -0.0065, 0.0088), 0.017, "sphere"),
    ("thumb_mcp_link", "thumb", (0.0183, 0.0094, 0.0102), 0.014, "sphere"),
    ("thumb_cmc_abd_link", "thumb", (0.0108, 0.0039, -0.0141), 0.015, "sphere"),
    ("thumb_ip_link", "thumb", (0.0113, 0.0073, 0.0123), 0.014, "sphere"),
    ("index_mcp_link", "index", (0.0029, 0.0191, 0.0100), 0.014, "sphere"),
    ("index_pip_link", "index", (-0.0042, 0.0090, -0.0015), 0.014, "sphere"),
    ("index_dip_link", "index", (0.0037, 0.0185, 0.0100), 0.013, "sphere"),
    ("middle_mcp_link", "middle", (0.0049, 0.0191, 0.0100), 0.014, "sphere"),
    ("middle_pip_link", "middle", (-0.0031, 0.0102, -0.0015), 0.014, "sphere"),
    ("middle_dip_link", "middle", (0.0049, 0.0187, 0.0105), 0.013, "sphere"),
    ("ring_mcp_link", "ring", (0.0029, 0.0191, 0.0100), 0.014, "sphere"),
    ("ring_pip_link", "ring", (-0.0037, 0.0095, -0.0015), 0.014, "sphere"),
    ("ring_dip_link", "ring", (0.0042, 0.0185, 0.0100), 0.013, "sphere"),
    ("pinky_mcp_link", "pinky", (0.0005, 0.0221, 0.0100), 0.013, "sphere"),
    ("pinky_pip_link", "pinky", (-0.0007, 0.0055, -0.0015), 0.012, "sphere"),
    ("pinky_dip_link", "pinky", (-0.0014, 0.0124, 0.0090), 0.012, "sphere"),
)

# Collision proxy index -> MuJoCo body name, exposed to the force mapper.
COLLISION_BODY_TO_FINGER = {body: finger for body, finger, _, _, _ in GLOVE_COLLISIONS}


def _insert_child(parent: ET.Element, child: ET.Element) -> None:
    parent.append(child)


def _patch_glove(root: ET.Element, model_path: Path, mapping: Any, *, config_kinematic: bool = False,
                 timestep_s: float = 0.001, scene: dict[str, Any] | None = None,
                 collision_geometry: str = "mesh") -> None:
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", str(model_path.parent / "meshes"))
    compiler.set("texturedir", str(model_path.parent))
    compiler.set("angle", "radian")
    compiler.set("autolimits", "true")

    visual = root.find("visual")
    if visual is None:
        visual = ET.Element("visual")
        root.insert(0, visual)
    global_visual = visual.find("global")
    if global_visual is None:
        global_visual = ET.SubElement(visual, "global")
    global_visual.set("offwidth", "1280")
    global_visual.set("offheight", "960")
    global_visual.set("fovy", "45")
    headlight = visual.find("headlight")
    if headlight is None:
        headlight = ET.SubElement(visual, "headlight")
    headlight.set("ambient", "0.16 0.17 0.19")
    headlight.set("diffuse", "0.24 0.24 0.25")
    headlight.set("specular", "0.08 0.08 0.08")

    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("timestep", f"{float(timestep_s):.9g}")
    option.set("gravity", "0 0 0")
    option.set("integrator", "implicitfast")
    option.set("cone", "elliptic")
    option.set("jacobian", "dense")

    world = root.find("worldbody")
    if world is None:
        raise ValueError("手套 MJCF 缺少 worldbody")
    # The CAD MJCF is authored lying palm-down at the world origin.  Re-pose
    # the fixed base body (no joint) without touching the joint chain.
    base_link = next((body for body in world.findall("body")
                      if body.get("name") == "base_link"), None)
    if base_link is not None and scene:
        hand_pos = scene.get("hand_position")
        hand_quat = scene.get("hand_quat_wxyz")
        if hand_pos is not None:
            base_link.set("pos", " ".join(f"{float(v):.9g}" for v in hand_pos))
        if hand_quat is not None:
            q = np.asarray(hand_quat, dtype=float)
            q = q / np.linalg.norm(q)
            base_link.set("quat", " ".join(f"{float(v):.9g}" for v in q))
    bodies = {body.get("name"): body for body in world.iter("body") if body.get("name")}
    joint_limits = {m.model_joint: m.model_rad_limits() for m in mapping.motors}
    for joint in world.iter("joint"):
        name = joint.get("name")
        if name in joint_limits:
            lo, hi = joint_limits[name]
            joint.set("limited", "true")
            joint.set("range", f"{lo:.9g} {hi:.9g}")
    for cid, (body_name, finger, pos, radius, kind) in enumerate(GLOVE_COLLISIONS):
        body = bodies.get(body_name)
        if body is None:
            continue
        geom = ET.SubElement(body, "geom")
        # The index keeps names unique (some links carry two proxy spheres).
        geom.set("name", f"col_{finger}_{body_name}_{cid}")
        geom.set("type", kind)
        if kind == "box":
            # palm: half extents, not a radius
            geom.set("size", f"0.012 0.045 0.020")
        else:
            geom.set("size", f"{radius:.9g}")
        geom.set("pos", " ".join(f"{v:.9g}" for v in pos))
        # In "mesh" mode the proxies stay in the model (UI/tests use their
        # positions) but do not collide.
        collide = collision_geometry == "spheres"
        geom.set("contype", "1" if collide else "0")
        geom.set("conaffinity", "2" if collide else "0")
        geom.set("mass", "0")
        geom.set("rgba", "0.9 0.2 0.2 0.20")
        geom.set("group", "3")

    if collision_geometry == "mesh":
        # Collide with exactly what is drawn: every visual mesh of the glove
        # (convex hull per part).  The hand-placed spheres sat on the dorsal
        # motor housings and missed the palm-side finger rings entirely.
        for geom in world.iter("geom"):
            if geom.get("type") == "mesh" and (geom.get("name") or "").endswith("_visual"):
                geom.set("contype", "1")
                geom.set("conaffinity", "2")

    if not bool(config_kinematic):
        actuator = root.find("actuator")
        if actuator is None:
            actuator = ET.SubElement(root, "actuator")
        for motor in mapping.motors:
            act = ET.SubElement(actuator, "motor")
            act.set("name", f"act_{motor.motor_id}")
            act.set("joint", motor.model_joint)
            act.set("gear", "1")
            act.set("ctrlrange", "-12 12")
            act.set("forcerange", "-12 12")


def _add_objects(root: ET.Element, specs: list[ObjectSpec], config: dict[str, Any]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF 缺少 worldbody")
    objects_cfg = config.get("objects", {})
    solref_range = tuple(float(v) for v in objects_cfg.get("solref_range_s", (0.002, 0.30)))
    damping = float(objects_cfg.get("damping", 0.02))
    friction = tuple(float(v) for v in objects_cfg.get("friction", (1.0, 0.01, 0.001)))
    mesh_names: dict[str, str] = {}
    bodies_xml: list[str] = []
    for spec in specs:
        mesh_name = f"softmesh_{spec.name}"
        mesh_xml, body_xml = object_xml(
            spec, solref_range=solref_range, damping=damping, friction=friction, mesh_name=mesh_name
        )
        if mesh_xml:
            asset.append(ET.fromstring(mesh_xml))
        bodies_xml.append(body_xml)
        mesh_names[spec.name] = mesh_name
    # Objects are appended as parsed XML fragments.  They are top-level bodies
    # in the same world as the glove so MuJoCo computes cross-contacts.
    for body_xml in bodies_xml:
        world.append(ET.fromstring(body_xml))
    # Subtle reference grid: gives depth cues without the low-resolution
    # MuJoCo shadow-map artifacts that made the floor look dirty.
    grid_file = Path(__file__).resolve().parents[2] / "assets" / "textures" / "grid.png"
    if grid_file.is_file():
        texture = ET.SubElement(asset, "texture")
        texture.set("name", "floor_grid")
        texture.set("type", "2d")
        texture.set("file", str(grid_file))
        material = ET.SubElement(asset, "material")
        material.set("name", "floor_grid_material")
        material.set("texture", "floor_grid")
        material.set("texrepeat", "4 4")
        material.set("texuniform", "false")
        material.set("rgba", "1 1 1 1")
        material.set("specular", "0.05")
        material.set("shininess", "0.1")
    floor = ET.SubElement(world, "geom")
    floor.set("name", "floor_visual")
    floor.set("type", "plane")
    floor.set("size", "0.8 0.8 0.1")
    floor.set("pos", "0 0 -0.002")
    floor.set("contype", "0")
    floor.set("conaffinity", "0")
    floor.set("rgba", "0.065 0.080 0.100 1")
    if grid_file.is_file():
        floor.set("material", "floor_grid_material")
    light = ET.SubElement(world, "light")
    light.set("name", "key_light")
    light.set("pos", "0.35 -0.20 0.55")
    light.set("dir", "-0.5 0.3 -1")
    light.set("directional", "true")
    light.set("diffuse", "0.80 0.78 0.74")
    light.set("specular", "0.18 0.18 0.18")
    light.set("ambient", "0.08 0.085 0.095")
    fill = ET.SubElement(world, "light")
    fill.set("name", "fill_light")
    fill.set("pos", "-0.25 0.35 0.42")
    fill.set("dir", "0.55 -0.25 -1")
    fill.set("directional", "true")
    fill.set("diffuse", "0.16 0.18 0.22")
    fill.set("specular", "0.04 0.045 0.05")
    fill.set("ambient", "0.04 0.0425 0.0475")
    stat = root.find("statistic")
    if stat is None:
        stat = ET.SubElement(root, "statistic")
    stat.set("center", "0.14 0.0 0.11")
    stat.set("extent", "0.32")
    return mesh_names, [(b.get("name") if hasattr(b, "get") else "", "") for b in []]


def build_scene_xml(config: dict[str, Any], mapping: Any, specs: list[ObjectSpec]) -> str:
    model_path = Path(config["paths"]["glove_model"]).resolve()
    tree = ET.parse(model_path)
    root = tree.getroot()
    _patch_glove(root, model_path, mapping,
                 config_kinematic=bool(config.get("glove", {}).get("kinematic", True)),
                 timestep_s=float(config.get("simulation", {}).get("timestep_s", 0.001)),
                 scene=config.get("scene"),
                 collision_geometry=str(config.get("glove", {}).get("collision_geometry", "mesh")))
    _add_objects(root, specs, config)
    # Glove geoms use contype 1 / conaffinity 2 and objects 2 / 2, so the
    # glove never collides with itself, only with objects.
    return ET.tostring(root, encoding="unicode")


def compile_scene(config: dict[str, Any], mapping: Any, specs: list[ObjectSpec]):
    import mujoco

    xml = build_scene_xml(config, mapping, specs)
    try:
        model = mujoco.MjModel.from_xml_string(xml)
    except Exception:
        debug = Path(config["_sim_root"]) / "data" / "generated_scene.xml"
        debug.parent.mkdir(parents=True, exist_ok=True)
        debug.write_text(xml, encoding="utf-8")
        raise
    data = mujoco.MjData(model)
    return model, data, xml
