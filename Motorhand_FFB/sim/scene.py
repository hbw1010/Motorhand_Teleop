"""Dynamically driven vE3.3 glove interacting with a MuJoCo object.

Joint positions are the CAD tree angles in radians, not human-joint angles.
The object is a configurable primitive, fixed by default or optionally free
on a horizontal support under gravity.
All contact positions and forces returned by :meth:`GloveScene.step` are in
world coordinates; forces are in newtons and act *on the glove*.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from motorhand_teleop.domain import GLOVE_JOINTS
from motorhand_teleop.ve33_adapter import Ve33GloveModel


_FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# MuJoCo's positive-format solref is (time constant, damping ratio), in seconds.
_CONTACT_SOLREF = {"soft": (0.04, 1.0), "medium": (0.02, 1.0), "hard": (0.01, 1.0)}
# Uncalibrated Young's moduli (Pa); contact compliance is separate.
_FLEX_YOUNG = {"soft": 10000.0, "medium": 15000.0, "hard": 30000.0}
_FLEX_CONTACT_SOLREF = (0.004, 1.0)
_FLEX_POISSON = 0.3
_FLEX_DAMPING = 0.01
# Nearly collapsed or inverted tetrahedra invalidate the elastic force model.
_FLEX_MIN_VOLUME_RATIO = 0.02
# During a physical-output pause, bound virtual motor motion independently of
# the real motor torque limit. Unfollowed input is not accumulated in contact.
_EDIT_CLUTCH_SPEED_RAD_S = 0.5
_EDIT_CLUTCH_DRIVE_NM = 0.05


_NUMERICAL_WARNINGS = (
    mujoco.mjtWarning.mjWARN_INERTIA,
    mujoco.mjtWarning.mjWARN_CONTACTFULL,
    mujoco.mjtWarning.mjWARN_CNSTRFULL,
    mujoco.mjtWarning.mjWARN_BADQPOS,
    mujoco.mjtWarning.mjWARN_BADQVEL,
    mujoco.mjtWarning.mjWARN_BADQACC,
    mujoco.mjtWarning.mjWARN_BADCTRL,
)

# Visual and contact presets, not experimentally measured material properties.
_MATERIALS = {
    "foam": {"rgba": (0.25, 0.68, 0.95, 1.0), "stiffness": "soft",
             "friction": 0.9, "specular": 0.12, "shininess": 0.08},
    "rubber": {"rgba": (0.12, 0.72, 0.52, 1.0), "stiffness": "medium",
               "friction": 1.1, "specular": 0.25, "shininess": 0.18},
    "wood": {"rgba": (0.68, 0.39, 0.16, 1.0), "stiffness": "hard",
             "friction": 0.65, "specular": 0.18, "shininess": 0.15},
    "metal": {"rgba": (0.68, 0.72, 0.77, 1.0), "stiffness": "hard",
              "friction": 0.4, "specular": 0.85, "shininess": 0.8},
}
_OBJECT_SHAPES = ("sphere", "box", "cylinder")


def _dimensions(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("object dimensions must contain three positive finite values")
    return result.copy()


def _quaternion(values):
    result = np.asarray(values, dtype=float)
    if result.shape != (4,) or not np.isfinite(result).all():
        raise ValueError("object quaternion must contain four finite values")
    norm = np.linalg.norm(result)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("object quaternion must be nonzero")
    return result / norm


def _rotation(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


def _tetra_volumes(vertices, elements):
    points = vertices[elements]
    a = points[:, 1] - points[:, 0]
    b = points[:, 2] - points[:, 0]
    c = points[:, 3] - points[:, 0]
    return np.einsum("ij,ij->i", np.cross(a, b), c)


def _flex_centroid_jacobian(model, data, vertices, qpos_indices):
    """Recover the constant centroid map for the asset's sliding flex nodes."""
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos

    def centroid():
        mujoco.mj_kinematics(model, scratch)
        mujoco.mj_flex(model, scratch)
        return scratch.flexvert_xpos[vertices].mean(axis=0)

    base = centroid().copy()
    result = np.empty((3, len(qpos_indices)))
    # Quadratic interpolation is affine in the translation-only node qpos.
    for column, address in enumerate(qpos_indices):
        scratch.qpos[address] += 0.001
        result[:, column] = (centroid() - base) / 0.001
        scratch.qpos[address] -= 0.001
    return result


class GloveScene:
    """Real glove CAD geometry with a selectable contact object.

    ``object_position_m`` is the object centre in world coordinates. If not
    supplied, it is placed in the initial finger-flexion path. ``object_size_m``
    is the sphere diameter, cube edge length, or cylinder diameter and height.
    The glove collision surfaces are CAD mesh convex hulls, not human skin.
    Objects are fixed unless ``object_fixed=False`` is explicitly selected.
    ``stiffness`` selects an uncalibrated elastic modulus for rubber and foam
    spheres, and a contact response for rigid objects.

    Measured motor angles are interpolated as bounded PD torque targets. Contact
    can impede the simulated glove, whose closed-chain pose is reconstructed
    from the simulated motor angles after each frame.
    """

    def __init__(
        self,
        side: str = "right",
        scenario: str = "press",
        stiffness: str | None = None,
        *,
        object_position_m: Sequence[float] | None = None,
        initial_tree_positions_rad: Mapping[str, float] | None = None,
        object_shape: str = "sphere",
        object_material: str = "rubber",
        object_size_m: float = 0.05,
        object_mass_kg: float = 0.05,
        object_fixed: bool = True,
        timestep_s: float = 0.001,
        asset_root: str | Path | None = None,
        glove_model: Ve33GloveModel | None = None,
        object_dimensions_m: Sequence[float] | None = None,
    ) -> None:
        if side not in ("right", "left"):
            raise ValueError("side must be 'right' or 'left'")
        if scenario not in ("press", "grasp"):
            raise ValueError("scenario must be 'press' or 'grasp'")
        if object_shape not in _OBJECT_SHAPES:
            raise ValueError(f"object_shape must be one of {_OBJECT_SHAPES}")
        if object_material not in _MATERIALS:
            raise ValueError(f"unknown object material: {object_material}")
        material = _MATERIALS[object_material]
        selected_stiffness = material["stiffness"] if stiffness is None else stiffness
        if selected_stiffness not in _CONTACT_SOLREF:
            raise ValueError("stiffness must be 'soft', 'medium', or 'hard'")
        for name, value in (("object_size_m", object_size_m),
                            ("object_mass_kg", object_mass_kg),
                            ("timestep_s", timestep_s)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if not isinstance(object_fixed, bool):
            raise ValueError("object_fixed must be a boolean")
        fixed = object_fixed
        dimensions = _dimensions([object_size_m] * 3 if object_dimensions_m is None
                                 else object_dimensions_m)
        half_dimensions = dimensions / 2
        self.object_dimensions_m = dimensions
        self._asset_root = root = Path(asset_root) if asset_root is not None else Path(__file__).resolve().parents[1] / "assets" / "hand_motor_ve33"
        self._object_shape = object_shape
        self._object_material = object_material
        self._object_mass_kg = float(object_mass_kg)
        self._object_size_m = float(object_size_m)
        half_size = float(half_dimensions[2])
        source = root / ("hand_motor_ve33_R" if side == "right" else "hand_motor_ve33_L") / "hand_closed_chain.xml"
        xml = ET.parse(source).getroot()
        xml.find("compiler").set("meshdir", str((source.parent / "meshes").resolve()))
        deformable = object_shape == "sphere" and object_material in ("rubber", "foam")
        option = xml.find("option")
        option.set("timestep", str(timestep_s))
        option.set("gravity", "0 0 0" if fixed else "0 0 -9.81")
        option.set("integrator", "discrete" if deformable else "implicitfast")
        # Original CAD meshes are visual-only. Use the same physical meshes for
        # object contact, but ensure no glove/glove pairs can collide.
        solref = " ".join(map(str, _CONTACT_SOLREF[selected_stiffness]))
        friction = f"{material['friction']} 0.005 0.0001"
        for body in xml.iter("body"):
            if body.get("name", "").split("_", 1)[0] in _FINGERS:
                for geom in body.findall("geom"):
                    geom.set("contype", "1")
                    geom.set("conaffinity", "2")
                    geom.set("solref", solref)
                    geom.set("friction", friction)
        world = xml.find("worldbody")
        ET.SubElement(world, "light", name="scene_key_light", pos="0.1 -0.1 0.3",
                      dir="0 0 -1", diffuse="0.8 0.8 0.8", ambient="0.5 0.5 0.5")
        if not fixed:
            # The floor contacts the object, not the glove.
            ET.SubElement(world, "geom", name="scene_floor", type="plane",
                          pos="0 0 -0.1", size="0.4 0.4 0.01",
                          contype="4", conaffinity="2", friction="0.8 0.005 0.0001",
                          rgba="0.3 0.35 0.4 1")
        asset = xml.find("asset")
        ET.SubElement(asset, "material", name="scene_object_material",
                      rgba=" ".join(map(str, material["rgba"])),
                      specular=str(material["specular"]), shininess=str(material["shininess"]))
        obj = ET.SubElement(world, "body", name="scene_object", pos="0 0 0")
        if deformable:
            ball_asset = Path(__file__).with_name("assets") / "deformable_ball.xml"
            ball_root = ET.parse(ball_asset).getroot()
            ball_body = ball_root.find("./worldbody/body")
            flex = ball_root.find("./deformable/flex")
            if ball_body is None or flex is None:
                raise ValueError("deformable ball asset is missing its body or flex")
            nodes = ball_body.findall("body")
            center = ball_body.find("./body[@name='scene_soft_1_1_1']")
            if not nodes or center is None:
                raise ValueError("deformable ball asset is missing its control nodes")
            reference_radius = float(flex.get("radius"))
            reference_half_size = reference_radius + max(
                abs(float(value)) for node in nodes for value in node.get("pos", "0 0 0").split()
            )
            radius = reference_radius * min(half_dimensions) / reference_half_size
            scale = (half_dimensions - radius) / (reference_half_size - reference_radius)
            reference_mass = sum(float(node.find("inertial").get("mass")) for node in nodes)
            mass_scale = object_mass_kg / reference_mass
            inertia_scale = np.array([scale[1]**2 + scale[2]**2,
                                      scale[0]**2 + scale[2]**2,
                                      scale[0]**2 + scale[1]**2]) / 2
            for node in nodes:
                if node.get("pos") is not None:
                    node.set("pos", " ".join(str(float(value) * factor)
                                             for value, factor in zip(node.get("pos").split(), scale)))
                inertial = node.find("inertial")
                inertial.set("mass", str(float(inertial.get("mass")) * mass_scale))
                inertial.set("diaginertia", " ".join(
                    str(float(value) * mass_scale * factor)
                    for value, factor in zip(inertial.get("diaginertia").split(), inertia_scale)
                ))
                if node is center and not fixed:
                    for component, axis in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
                        ET.SubElement(node, "joint", name=f"scene_soft_1_1_1_{component}",
                                      type="slide", axis=axis)
                obj.append(node)
            flex.set("vertex", " ".join(str(float(value) * scale[i % 3])
                                        for i, value in enumerate(flex.get("vertex").split())))
            flex.set("radius", str(radius))
            flex.set("material", "scene_object_material")
            flex.set("rgba", " ".join(map(str, material["rgba"])))
            elasticity = flex.find("elasticity")
            elasticity.set("young", str(_FLEX_YOUNG[selected_stiffness]))
            elasticity.set("poisson", str(_FLEX_POISSON))
            elasticity.set("damping", str(_FLEX_DAMPING))
            contact = flex.find("contact")
            contact.set("solref", " ".join(map(str, _FLEX_CONTACT_SOLREF)))
            contact.set("friction", friction)
            deformable_xml = xml.find("deformable")
            if deformable_xml is None:
                deformable_xml = ET.SubElement(xml, "deformable")
            deformable_xml.append(flex)
            if fixed:
                # Three material-frame constraints suppress rigid rotation
                # without fixing the deformable surface.
                for section in ("tendon", "equality"):
                    target = xml.find(section)
                    if target is None:
                        target = ET.SubElement(xml, section)
                    for element in ball_root.find(section):
                        if section == "tendon":
                            rotation_axis = "xyz".index(element.get("name")[-1])
                            for joint in element.findall("joint"):
                                displacement_axis = "xyz".index(joint.get("joint")[-1])
                                reference_axis = 3 - rotation_axis - displacement_axis
                                joint.set("coef", str(float(joint.get("coef")) *
                                                      scale[reference_axis] / max(scale)))
                        target.append(element)
        else:
            if not fixed:
                ET.SubElement(obj, "freejoint", name="scene_object_free")
            geom_type = object_shape
            geom_attrs = {}
            if object_shape == "sphere":
                geom_type = "sphere" if np.all(dimensions == dimensions[0]) else "ellipsoid"
                geom_attrs["size"] = (str(half_dimensions[0]) if geom_type == "sphere" else
                                      " ".join(map(str, half_dimensions)))
            elif object_shape == "box":
                geom_attrs["size"] = " ".join(map(str, half_dimensions))
            elif dimensions[0] == dimensions[1]:
                geom_attrs["size"] = f"{half_dimensions[0]} {half_dimensions[2]}"
            else:
                # The mesh is both the rendered surface and the convex contact hull.
                geom_type = "mesh"
                vertices = [(half_dimensions[0] * math.cos(2 * math.pi * i / 64),
                             half_dimensions[1] * math.sin(2 * math.pi * i / 64), z)
                            for z in (-half_dimensions[2], half_dimensions[2]) for i in range(64)]
                ET.SubElement(asset, "mesh", name="scene_object_mesh",
                              vertex=" ".join(str(value) for vertex in vertices for value in vertex))
                geom_attrs["mesh"] = "scene_object_mesh"
            ET.SubElement(obj, "geom", name="scene_object_geom", type=geom_type,
                          mass=str(object_mass_kg), material="scene_object_material",
                          contype="2", conaffinity="5", solref=solref, friction=friction,
                          **geom_attrs)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        self.data = mujoco.MjData(self.model)
        self.model.eq_solref[:, :] = (0.002, 1.0)
        self.model.eq_solimp[:, :3] = (0.99, 0.9999, 0.00001)
        self.side, self.scenario, self.stiffness = side, scenario, selected_stiffness
        self.object_fixed = fixed
        self._deformable = deformable
        self._flex_id = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, "scene_soft")
                         if deformable else -1)
        self.timestep_s = float(timestep_s)
        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "scene_object")
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_object_geom")
        self._object_joint_id = (-1 if fixed or deformable else
                                 mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "scene_object_free"))
        self._flex_qpos = []
        self._flex_dofs = []
        self._joint_names = []
        self._qpos_indices = []
        self._dof_indices = []
        for joint_id in range(self.model.njnt):
            if joint_id == self._object_joint_id:
                continue
            joint_body = int(self.model.jnt_bodyid[joint_id])
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, joint_body) or ""
            if deformable and body_name.startswith("scene_soft_"):
                self._flex_qpos.append(int(self.model.jnt_qposadr[joint_id]))
                self._flex_dofs.append(int(self.model.jnt_dofadr[joint_id]))
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            if name is None or self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("glove model must contain named hinge joints only")
            self._joint_names.append(name)
            self._qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self._dof_indices.append(int(self.model.jnt_dofadr[joint_id]))
        self._qpos_indices = np.asarray(self._qpos_indices, dtype=int)
        self._dof_indices = np.asarray(self._dof_indices, dtype=int)
        self.model.dof_armature[self._dof_indices] += 1e-4
        self.glove_model = glove_model if glove_model is not None else Ve33GloveModel(side, asset_root=root)
        self.kinematics = self.glove_model.kinematics
        self._motor_names = tuple(self.glove_model.motor_names)
        self._tree_names = tuple(self.kinematics.tree_names)
        if (self.kinematics.hand != side or len(self._motor_names) != 12 or len(set(self._motor_names)) != 12
                or set(self._motor_names) != set(self.kinematics.motor_names)
                or set(self._tree_names) != set(self._joint_names)
                or len(self._tree_names) != len(self._joint_names)):
            raise ValueError("glove kinematics and MuJoCo tree do not match")
        name_to_row = {name: i for i, name in enumerate(self._joint_names)}
        self._motor_tree_rows = np.array([self._tree_names.index(name) for name in self._motor_names])
        self._motor_qpos = self._qpos_indices[[name_to_row[name] for name in self._motor_names]]
        self._motor_dofs = self._dof_indices[[name_to_row[name] for name in self._motor_names]]
        self._tree_qpos = self._qpos_indices[[name_to_row[name] for name in self._tree_names]]
        self._tree_dofs = self._dof_indices[[name_to_row[name] for name in self._tree_names]]
        self._body_finger = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            finger = name.split("_", 1)[0]
            if finger in _FINGERS:
                self._body_finger[body_id] = finger
        initial_positions = ({name: 0.0 for name in self._joint_names}
                             if initial_tree_positions_rad is None else initial_tree_positions_rad)
        self._virtual_pose = self.kinematics.evaluate_tree(initial_positions)
        self.data.qpos[self._tree_qpos] = [self._virtual_pose.all_positions[name] for name in self._tree_names]
        mujoco.mj_forward(self.model, self.data)
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "index_dip_link_visual")
        if geom_id < 0:
            raise ValueError("glove index distal visual mesh is missing")
        reference = self.data.geom_xpos[geom_id].copy()
        middle_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "middle_dip_link_visual")
        if middle_id < 0:
            raise ValueError("glove middle distal visual mesh is missing")
        middle = self.data.geom_xpos[middle_id].copy()
        if object_position_m is None:
            # Keep the top surface near the original 3 cm ball's interaction path.
            position = reference + np.array([0.0, 0.0, -0.052 - (half_size - 0.015)])
            if scenario == "grasp":
                position[0] -= 0.05
                position[1] += (reference[1] - middle[1]) * (2.0 / 3.0)
        else:
            position = np.asarray(object_position_m, dtype=float)
            if position.shape != (3,) or not np.all(np.isfinite(position)):
                raise ValueError("object_position_m must contain three finite values")
        if fixed or deformable:
            self.model.body_pos[self.object_body_id] = position
        if not fixed and not deformable:
            address = int(self.model.jnt_qposadr[self._object_joint_id])
            self.data.qpos[address:address + 3] = position
        if not fixed:
            floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
            self.model.geom_pos[floor_id, 2] = position[2] - half_size
        mujoco.mj_forward(self.model, self.data)
        if deformable:
            start = int(self.model.flex_vertadr[self._flex_id])
            count = int(self.model.flex_vertnum[self._flex_id])
            self._flex_vertices = slice(start, start + count)
            vertices = self.data.flexvert_xpos[self._flex_vertices]
            self._flex_rest_centered = (vertices - vertices.mean(axis=0)).copy()
            self._initial_radii = np.linalg.norm(vertices - vertices.mean(axis=0), axis=1)
            elem_start = int(self.model.flex_elemadr[self._flex_id])
            elem_count = int(self.model.flex_elemnum[self._flex_id])
            self._flex_tetrahedra = self.model.flex_elem.reshape(-1, 4)[
                elem_start:elem_start + elem_count
            ].copy()
            self._initial_flex_volumes = _tetra_volumes(vertices, self._flex_tetrahedra)
            if (not np.isfinite(self._initial_flex_volumes).all()
                    or np.any(self._initial_flex_volumes == 0)):
                raise ValueError("deformable object has invalid rest tetrahedra")
            self._initial_flex_qpos = self.data.qpos[self._flex_qpos].copy()
            self._centroid_jacobian = _flex_centroid_jacobian(
                self.model, self.data, self._flex_vertices, self._flex_qpos
            )
        self.metadata = {
            "side": side, "scenario": scenario, "stiffness": selected_stiffness,
            "model_xml": str(source.resolve()), "object_fixed": fixed,
            "object_shape": object_shape, "object_material": object_material,
            "object_size_m": float(object_size_m), "object_mass_kg": float(object_mass_kg),
            "object_dimensions_m": dimensions.tolist(),
            "object_model": "deformable" if deformable else "rigid",
            "object_deformation_definition": (
                "maximum absolute change in vertex distance from vertex centroid, in metres"
                if deformable else None),
            "object_flex_vertex_count": count if deformable else None,
            "object_flex_asset": str(ball_asset.resolve()) if deformable else None,
            "object_flex_node_count": int(self.model.flex_nodenum[self._flex_id]) if deformable else None,
            "object_flex_young_pa": _FLEX_YOUNG[selected_stiffness] if deformable else None,
            "object_flex_poisson": _FLEX_POISSON if deformable else None,
            "object_flex_damping_s": _FLEX_DAMPING if deformable else None,
            "object_flex_anchor": ("center fixed; three best-fit rotation constraints" if fixed else "none")
                                  if deformable else None,
            "object_flex_contact_solref_s": list(_FLEX_CONTACT_SOLREF) if deformable else None,
            "object_flex_collision_radius_m": radius if deformable else None,
            "object_flex_min_volume_ratio": _FLEX_MIN_VOLUME_RATIO if deformable else None,
            "object_rgba": list(material["rgba"]), "object_friction": material["friction"],
            "object_initial_position_m": position.tolist(),
            "reference_geom": "index_dip_link_visual",
            "reference_point_world_m": reference.tolist(),
            "contact_surface": "convex hulls of glove CAD meshes; not human skin",
            "contact_solref_s": list(_CONTACT_SOLREF[selected_stiffness]),
            "contact_material_calibrated": False,
            "glove_self_collision": False,
            "integrator": "discrete" if deformable else "implicitfast",
            "glove_armature_added_kg_m2": 1e-4,
            "equality_solref": [0.002, 1.0],
            "equality_solimp_first_three": [0.99, 0.9999, 0.00001],
            "motor_kp_nm_per_rad": 1.0,
            "motor_kd_nm_s_per_rad": 0.03,
            "motor_drive_limit_nm": 0.5,
            "gravity_m_s2": [0.0, 0.0, 0.0 if fixed else -9.81],
            "support": None if fixed else "horizontal plane; friction=0.8",
            "timestep_s": self.timestep_s,
        }
        self._reset_body_pos = self.model.body_pos[self.object_body_id].copy()
        self._reset_body_quat = self.model.body_quat[self.object_body_id].copy()
        self._previous = (None if initial_tree_positions_rad is None else
                          np.array([self._virtual_pose.motor_positions[name] for name in self._motor_names]))
        self._clutch_previous_input = None
        self._warning_counts = (0,) * len(_NUMERICAL_WARNINGS)
        self._check_dynamics()

    def object_pose(self):
        """Return the object's current world-space centre and orientation (wxyz)."""
        if not self._deformable:
            return (self.data.xpos[self.object_body_id].copy(),
                    self.data.xquat[self.object_body_id].copy())
        vertices = self.data.flexvert_xpos[self._flex_vertices]
        position = vertices.mean(axis=0)
        # The free flex has no rigid orientation joint. Its best-fit material
        # frame tracks rigid rotation without treating elastic strain as pose.
        covariance = (vertices - position).T @ self._flex_rest_centered
        u, _, vh = np.linalg.svd(covariance)
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(u @ vh)
        matrix = np.ascontiguousarray(u @ correction @ vh)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, matrix.ravel())
        return position.copy(), quat

    def object_max_penetration(self):
        """Maximum current penetration involving the edited object, in metres."""
        depth = 0.0
        for contact in self.data.contact[:self.data.ncon]:
            matches = (self._flex_id in contact.flex if self._deformable else
                       self.object_geom_id in contact.geom)
            if matches:
                depth = max(depth, -float(contact.dist))
        return depth

    def set_object_pose(self, position_m, quat_wxyz):
        """Move the physical object without changing its deformation or floor."""
        position = np.asarray(position_m, dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("object position must contain three finite values")
        quat = _quaternion(quat_wxyz)
        if self._deformable:
            old_position, old_quat = self.object_pose()
            delta = _rotation(quat) @ _rotation(old_quat).T
            body_position = self.model.body_pos[self.object_body_id].copy()
            body_rotation = _rotation(self.model.body_quat[self.object_body_id])
            self.model.body_pos[self.object_body_id] = position + delta @ (body_position - old_position)
            new_rotation = np.ascontiguousarray(delta @ body_rotation)
            mujoco.mju_mat2Quat(self.model.body_quat[self.object_body_id], new_rotation.ravel())
        elif self.object_fixed:
            self.model.body_pos[self.object_body_id] = position
            self.model.body_quat[self.object_body_id] = quat
        else:
            address = int(self.model.jnt_qposadr[self._object_joint_id])
            self.data.qpos[address:address + 3] = position
            self.data.qpos[address + 3:address + 7] = quat
        mujoco.mj_forward(self.model, self.data)
        if self._deformable:
            # The flex nodes have translation-only joints, so their centroid
            # Jacobian rotates with the material frame and is otherwise constant.
            self._centroid_jacobian = delta @ self._centroid_jacobian

    def resized(self, dimensions_m):
        """Rebuild the object at rest while retaining glove state and simulation time."""
        dimensions = _dimensions(dimensions_m)
        position, quat = self.object_pose()
        scene = GloveScene(
            self.side, self.scenario, self.stiffness,
            object_position_m=position, object_shape=self._object_shape,
            object_material=self._object_material, object_size_m=self._object_size_m,
            object_dimensions_m=dimensions, object_mass_kg=self._object_mass_kg,
            object_fixed=self.object_fixed, timestep_s=self.timestep_s,
            asset_root=self._asset_root, glove_model=self.glove_model)
        scene.data.qpos[scene._tree_qpos] = self.data.qpos[self._tree_qpos]
        scene.data.qvel[scene._tree_dofs] = self.data.qvel[self._tree_dofs]
        scene.data.qfrc_applied[scene._motor_dofs] = self.data.qfrc_applied[self._motor_dofs]
        scene.data.time = self.data.time
        scene._virtual_pose = self._virtual_pose
        scene._previous = None if self._previous is None else self._previous.copy()
        if not self.object_fixed:
            old_floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
            new_floor = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "scene_floor")
            scene.model.geom_pos[new_floor] = self.model.geom_pos[old_floor]
        scene._reset_body_pos = self._reset_body_pos.copy()
        scene._reset_body_quat = self._reset_body_quat.copy()
        scene.metadata["object_initial_position_m"] = list(self.metadata["object_initial_position_m"])
        scene.set_object_pose(position, quat)
        return scene

    def _check_dynamics(self) -> None:
        if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                or not np.isfinite(self.data.qacc).all() or not math.isfinite(self.data.time)):
            raise ValueError("MuJoCo dynamics produced a non-finite state")
        for warning, previous in zip(_NUMERICAL_WARNINGS, self._warning_counts):
            if self.data.warning[warning].number != previous:
                raise RuntimeError(f"MuJoCo numerical warning: {warning.name}")
        if self._deformable:
            volumes = _tetra_volumes(self.data.flexvert_xpos[self._flex_vertices], self._flex_tetrahedra)
            ratio = volumes / self._initial_flex_volumes
            if not np.isfinite(ratio).all() or np.any(ratio <= _FLEX_MIN_VOLUME_RATIO):
                raise RuntimeError("MuJoCo deformable ball tetrahedra are inverted or nearly collapsed")

    def synchronize_tree(self, tree_positions_rad: Mapping[str, float]) -> None:
        """Set a fresh startup pose without interpreting startup delay as velocity."""
        pose = self.kinematics.evaluate_tree(tree_positions_rad)
        # Explicit startup synchronization, never a per-frame target assignment.
        self.data.qpos[self._tree_qpos] = [pose.all_positions[name] for name in self._tree_names]
        self.data.qvel[self._tree_dofs] = 0.0
        self.data.qfrc_applied[self._motor_dofs] = 0.0
        self._virtual_pose = pose
        self._previous = np.array([pose.motor_positions[name] for name in self._motor_names])
        mujoco.mj_forward(self.model, self.data)

    def reset_object(self) -> None:
        """Return a free object to its recorded initial pose and zero velocity."""
        if self.object_fixed:
            return
        if self._deformable:
            old_rotation = _rotation(self.model.body_quat[self.object_body_id])
            reset_rotation = _rotation(self._reset_body_quat)
            self.model.body_pos[self.object_body_id] = self._reset_body_pos
            self.model.body_quat[self.object_body_id] = self._reset_body_quat
            self._centroid_jacobian = reset_rotation @ old_rotation.T @ self._centroid_jacobian
            self.data.qpos[self._flex_qpos] = self._initial_flex_qpos
            self.data.qvel[self._flex_dofs] = 0.0
        else:
            qpos = int(self.model.jnt_qposadr[self._object_joint_id])
            dof = int(self.model.jnt_dofadr[self._object_joint_id])
            self.data.qpos[qpos:qpos + 3] = self.metadata["object_initial_position_m"]
            self.data.qpos[qpos + 3:qpos + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _glove_object_contact(self, contact):
        geom1, geom2 = map(int, contact.geom)
        flex1, flex2 = map(int, contact.flex)
        object1 = flex1 == self._flex_id if self._deformable else geom1 == self.object_geom_id
        object2 = flex2 == self._flex_id if self._deformable else geom2 == self.object_geom_id
        if object1 and geom2 >= 0:
            glove_geom, sign = geom2, 1.0
        elif object2 and geom1 >= 0:
            glove_geom, sign = geom1, -1.0
        else:
            return None
        body_id = int(self.model.geom_bodyid[glove_geom])
        finger = self._body_finger.get(body_id)
        return None if finger is None else (body_id, finger, sign)

    def step(self, tree_positions_rad: Mapping[str, float], dt: float, *, edit_clutch: bool = False) -> dict:
        """Advance by ``dt`` seconds and return object state and glove contacts.

        Tree angles are continuous CAD coordinates; differences are not wrapped.
        Returned virtual positions are the actual collision-constrained state.
        The edit clutch bounds simulated actuation and drops input backlog in
        contact while physical output is paused. It is not a hardware limit.
        """
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        if set(tree_positions_rad) != set(self._joint_names):
            raise ValueError("tree_positions_rad must contain every glove tree joint exactly once")
        target = np.asarray([tree_positions_rad[name] for name in self._joint_names], dtype=float)
        if not np.all(np.isfinite(target)):
            raise ValueError("tree_positions_rad must be finite")
        self._check_dynamics()
        input_target = np.asarray([tree_positions_rad[name] for name in self._motor_names], dtype=float)
        current = self.data.qpos[self._motor_qpos].copy()
        if edit_clutch:
            increment = (np.zeros_like(input_target) if self._clutch_previous_input is None
                         else input_target - self._clutch_previous_input)
            self._clutch_previous_input = input_target.copy()
            in_contact = any(self._glove_object_contact(self.data.contact[i]) is not None
                             for i in range(self.data.ncon))
            desired_change = increment if in_contact else input_target - current
            target = current + np.clip(desired_change, -_EDIT_CLUTCH_SPEED_RAD_S * dt,
                                       _EDIT_CLUTCH_SPEED_RAD_S * dt)
            previous = current
            drive_limit = _EDIT_CLUTCH_DRIVE_NM
        else:
            target = input_target
            previous = current if self._clutch_previous_input is not None else (
                target if self._previous is None else self._previous)
            self._clutch_previous_input = None
            drive_limit = 0.5
        nsteps = max(1, math.ceil(dt / min(self.timestep_s, 0.001)))
        original_timestep = self.model.opt.timestep
        self.model.opt.timestep = dt / nsteps
        try:
            for index in range(nsteps):
                motor_target = previous + (target - previous) * ((index + 1) / nsteps)
                self.data.qfrc_applied[self._motor_dofs] = np.clip(
                    (motor_target - self.data.qpos[self._motor_qpos])
                    - 0.03 * self.data.qvel[self._motor_dofs], -drive_limit, drive_limit
                )
                start_time = self.data.time
                mujoco.mj_step(self.model, self.data)
                self._check_dynamics()
                if self.data.time <= start_time:
                    raise ValueError("MuJoCo dynamics produced an invalid state")
        finally:
            self.model.opt.timestep = original_timestep

        motors = dict(zip(self._motor_names, map(float, self.data.qpos[self._motor_qpos])))
        pose = self.kinematics.evaluate_motors(motors, seed=self._virtual_pose)
        derivative = self.kinematics._tree_derivatives(pose.all_positions)
        motor_derivative = derivative[self._motor_tree_rows, :]
        singular = np.linalg.svd(motor_derivative, compute_uv=False)
        if singular[0] == 0 or singular[-1] <= singular[0] * 1e-9:
            raise ValueError("motor-space closed-chain derivative is singular")
        try:
            tree_per_motor = np.linalg.solve(motor_derivative.T, derivative.T).T
        except np.linalg.LinAlgError as exc:
            raise ValueError("motor-space closed-chain derivative unavailable") from exc
        tree_velocity = tree_per_motor @ self.data.qvel[self._motor_dofs]
        if not np.isfinite(tree_velocity).all():
            raise ValueError("non-finite closed-chain velocity")
        self.data.qpos[self._tree_qpos] = [pose.all_positions[name] for name in self._tree_names]
        self.data.qvel[self._tree_dofs] = tree_velocity
        self._virtual_pose = pose
        self._previous = target.copy()
        # Recompute at the simulated closed-chain pose, including contact forces.
        mujoco.mj_forward(self.model, self.data)
        self._check_dynamics()
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            match = self._glove_object_contact(contact)
            if match is None:
                continue
            body_id, finger, sign = match
            wrench = np.zeros(6)
            mujoco.mj_contactForce(self.model, self.data, index, wrench)
            # Rows of frame are contact axes in world coordinates. MuJoCo's
            # contact wrench acts on geom2, hence negate it for geom1.
            force = sign * (contact.frame.reshape(3, 3).T @ wrench[:3])
            if self._deformable:
                # efc_vel is the actual contact Jacobian times qvel, including
                # MuJoCo's flex-element interpolation. Pyramid edges encode
                # normal +/- friction * tangent in consecutive row pairs.
                adr = int(contact.efc_address)
                if adr < 0 or contact.dim < 3:
                    raise RuntimeError("flex contact tangent velocity is unavailable")
                if self.model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL:
                    if contact.friction[0] <= 0 or contact.friction[1] <= 0:
                        raise RuntimeError("flex contact friction prevents tangent velocity reconstruction")
                    v = self.data.efc_vel
                    tangent_velocity = np.array([
                        (v[adr] - v[adr + 1]) / (2 * contact.friction[0]),
                        (v[adr + 2] - v[adr + 3]) / (2 * contact.friction[1]),
                    ])
                else:
                    tangent_velocity = self.data.efc_vel[adr + 1:adr + 3]
            else:
                glove_jac = np.zeros((3, self.model.nv))
                object_jac = np.zeros((3, self.model.nv))
                mujoco.mj_jac(self.model, self.data, glove_jac, None, contact.pos, body_id)
                mujoco.mj_jac(self.model, self.data, object_jac, None, contact.pos, self.object_body_id)
                relative_velocity = (glove_jac - object_jac) @ self.data.qvel
                normal = contact.frame[:3]
                tangent_velocity = relative_velocity - normal * np.dot(normal, relative_velocity)
            contacts.append({"finger": finger, "body_id": body_id,
                             "point_world": contact.pos.copy(),
                             "force_world_on_glove": force,
                             "normal_force_n": float(wrench[0]),
                             "tangential_force_n": float(np.linalg.norm(wrench[1:3])),
                             "relative_tangent_speed_m_s": float(np.linalg.norm(tangent_velocity)),
                             "penetration_m": max(0.0, -float(contact.dist))})
        if self._deformable:
            vertices = self.data.flexvert_xpos[self._flex_vertices]
            object_position = vertices.mean(axis=0)
            deformation = float(np.max(np.abs(
                np.linalg.norm(vertices - object_position, axis=1) - self._initial_radii)))
            object_velocity = self._centroid_jacobian @ self.data.qvel[self._flex_dofs]
        else:
            object_position = self.data.xpos[self.object_body_id].copy()
            deformation = None
            if self.object_fixed:
                object_velocity = np.zeros(3)
            else:
                address = int(self.model.jnt_dofadr[self._object_joint_id])
                object_velocity = self.data.qvel[address:address + 3].copy()
        return {"contacts": contacts, "sim_time_s": float(self.data.time),
                "object_position_m": object_position,
                "object_deformation_m": deformation,
                "object_velocity_m_s": object_velocity,
                "virtual_servo_mode": "edit_clutch" if edit_clutch else "tracking",
                "virtual_servo_limit_nm": drive_limit,
                "input_alignment_error_rad": float(np.max(np.abs(
                    input_target - self.data.qpos[self._motor_qpos]))),
                "virtual_motor_positions_rad": {
                    name: float(value) for name, value in zip(self._motor_names, self.data.qpos[self._motor_qpos])
                },
                "virtual_tree_positions_rad": {
                    name: float(value) for name, value in zip(self._tree_names, self.data.qpos[self._tree_qpos])
                },
                "virtual_joint_positions_rad": {
                    name: float(pose.joint_positions[f"{name}_joint"]) for name in GLOVE_JOINTS
                }}
