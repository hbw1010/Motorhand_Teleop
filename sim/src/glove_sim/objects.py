from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class ObjectSpec:
    name: str
    shape: str
    size_m: float
    stiffness: float
    position: tuple[float, float, float]
    deformable: bool = True
    quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectSpec":
        pos = tuple(float(v) for v in data.get("position", (0.0, 0.0, 0.2)))
        quat = tuple(float(v) for v in data.get("quat", (1.0, 0.0, 0.0, 0.0)))
        return cls(
            name=str(data["name"]),
            shape=str(data["shape"]),
            size_m=float(data["size_m"]),
            stiffness=float(data.get("stiffness", 0.5)),
            position=pos,
            deformable=bool(data.get("deformable", True)),
            quat=quat,
        )


@dataclass
class ObjectRuntime:
    spec: ObjectSpec
    body_name: str
    free_joint: str
    qposadr: int = -1
    flex_name: str = ""
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_quat: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))

    @property
    def name(self) -> str:
        return self.spec.name


def _box_mesh(size: float, n: int = 3) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    """Subdivided cube surface; unique vertices on shared edges.

    A coarse 8-vertex cube tetrahedralises into a very weak lattice and can
    stretch unrealistically when squeezed.  A subdivided surface keeps the
    same shape while giving the lattice enough constraints to deform locally.
    """
    s = float(size)
    index: dict[tuple[int, int, int], int] = {}
    verts: list[tuple[float, float, float]] = []

    def vid(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        if key not in index:
            index[key] = len(verts)
            verts.append((s * (2.0 * i / n - 1.0), s * (2.0 * j / n - 1.0),
                          s * (2.0 * k / n - 1.0)))
        return index[key]

    faces: list[tuple[int, int, int]] = []
    for face, fixed in (("x0", 0), ("x1", n), ("y0", 0), ("y1", n), ("z0", 0), ("z1", n)):
        axis = face[0]
        for a in range(n):
            for b in range(n):
                quad = []
                for da, db in ((0, 0), (1, 0), (1, 1), (0, 1)):
                    if axis == "x":
                        i, j, k = fixed, a + da, b + db
                    elif axis == "y":
                        i, j, k = a + da, fixed, b + db
                    else:
                        i, j, k = a + da, b + db, fixed
                    quad.append(vid(i, j, k))
                # Outward winding for the two faces of each axis.
                if (face in ("x0", "y0", "z0")) == (axis in "xyz"):
                    pass
                if face in ("x0", "y1", "z0"):
                    faces.append((quad[0], quad[2], quad[1]))
                    faces.append((quad[0], quad[3], quad[2]))
                else:
                    faces.append((quad[0], quad[1], quad[2]))
                    faces.append((quad[0], quad[2], quad[3]))
    return verts, faces


def _icosphere(size: float, subdivisions: int = 0) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    t = (1.0 + math.sqrt(5.0)) / 2.0
    verts = [
        (-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
        (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
        (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    verts = [tuple(float(v) / math.sqrt(sum(x * x for x in p)) for v in p) for p in verts]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            pa, pb = np.array(verts[a]), np.array(verts[b])
            p = pa + pb
            p /= np.linalg.norm(p)
            verts.append(tuple(float(x) for x in p))
            cache[key] = len(verts) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = new_faces
    return [(x * size, y * size, z * size) for x, y, z in verts], faces


def _cylinder_mesh(size: float, segments: int = 8) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    r = float(size)
    h = float(size) * 0.8
    verts: list[tuple[float, float, float]] = []
    for z in (-h, h):
        for i in range(segments):
            a = 2.0 * math.pi * i / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
    bottom_center = len(verts)
    verts.append((0.0, 0.0, -h))
    top_center = len(verts)
    verts.append((0.0, 0.0, h))
    faces: list[tuple[int, int, int]] = []
    for i in range(segments):
        j = (i + 1) % segments
        b0, b1 = i, j
        t0, t1 = segments + i, segments + j
        faces.append((b0, b1, t1))
        faces.append((b0, t1, t0))
        faces.append((bottom_center, b1, b0))
        faces.append((top_center, t0, t1))
    return verts, faces


def shape_mesh(shape: str, size_m: float) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    if shape == "sphere":
        # One subdivision keeps the soft ball round enough to avoid the
        # low-poly "gem" look while remaining stable at 1 kHz physics.
        return _icosphere(size_m * 0.5, subdivisions=1)
    if shape == "box":
        return _box_mesh(size_m * 0.5)
    if shape == "cylinder":
        return _cylinder_mesh(size_m * 0.5, segments=12)
    raise ValueError(f"unsupported object shape: {shape}")


def _xml_floats(values: Iterable[float]) -> str:
    return " ".join(f"{v:.9g}" for v in values)


def _xml_ints(values: Iterable[int]) -> str:
    return " ".join(str(int(v)) for v in values)


def object_xml(spec: ObjectSpec, *, solref_range: tuple[float, float], damping: float,
               friction: tuple[float, float, float], mesh_name: str) -> tuple[str, str]:
    """Return (asset mesh XML, worldbody object XML) for one soft object.

    The object is a tetrahedral flex lattice generated by flexcomp from a
    closed surface mesh (surface triangles + a centre origin are tetrahedralised
    by MuJoCo).  A few bottom vertices are pinned to the object's locked root
    body; the rest deform.  The root body pose is written every physics step,
    so the object is fixed unless the user drags it.
    """
    lo, hi = solref_range
    stiffness = max(0.0, min(1.0, float(spec.stiffness)))
    # Larger solref is softer.  High stiffness must map to low solref.
    solref = hi - (hi - lo) * stiffness
    radius = max(0.003, spec.size_m * 0.10)
    mass = max(0.02, 800.0 * spec.size_m ** 3)
    if spec.deformable:
        verts, faces = shape_mesh(spec.shape, spec.size_m)
        zs = np.array([p[2] for p in verts])
        threshold = zs.min() + 0.50 * (zs.max() - zs.min())
        pin_ids = [i for i, z in enumerate(zs) if z <= threshold]
        if not pin_ids:
            pin_ids = [int(np.argmin(zs))]
        pin_ids = pin_ids[:10]
        mesh_asset = (
            f'<mesh name="{mesh_name}" vertex="{_xml_floats(v for p in verts for v in p)}" '
            f'face="{_xml_ints(i for f in faces for i in f)}"/>'
        )
        body = (
            f'<body name="obj_{spec.name}" pos="{_xml_floats(spec.position)}" '
            f'quat="{_xml_floats(spec.quat)}">'
            f'<freejoint name="obj_{spec.name}_free"/>'
            f'<geom name="obj_{spec.name}_root_mass" type="sphere" size="0.001" mass="0.08" '
            f'contype="0" conaffinity="0" rgba="0 0 0 0"/>'
            f'<flexcomp name="flex_{spec.name}" type="mesh" mesh="{mesh_name}" dim="3" '
            f'origin="0 0 0" pos="0 0 0" radius="{radius:.9g}" mass="{mass:.9g}" '
            f'rgba="0.18 0.60 0.86 0.94">'
            f'<edge equality="true" solref="{solref:.9g} 1" damping="{damping:.9g}"/>'
            f'<contact contype="2" conaffinity="2" condim="3" '
            f'friction="{_xml_floats(friction)}" solref="0.01 1" solimp="0.9 0.95 0.001"/>'
            f'<pin id="{_xml_ints(pin_ids)}"/>'
            f'</flexcomp></body>'
        )
        return mesh_asset, body
    # Hard object: a rigid body with a compliant contact.  It is still
    # adjustable in size and contact stiffness, but does not tessellate a
    # soft lattice, which keeps the real-time budget bounded.
    # High stiffness must map to a stiffer contact (smaller solref).
    contact_solref = 0.0025 - 0.0020 * stiffness
    if spec.shape == "sphere":
        geom_type, geom_size = "sphere", f"{spec.size_m * 0.5:.9g}"
    elif spec.shape == "box":
        s = spec.size_m * 0.5
        geom_type, geom_size = "box", f"{s:.9g} {s:.9g} {s:.9g}"
    elif spec.shape == "cylinder":
        geom_type, geom_size = "cylinder", f"{spec.size_m * 0.5:.9g} {spec.size_m * 0.4:.9g}"
    else:
        raise ValueError(f"unsupported object shape: {spec.shape}")
    # Soft family stays cool teal; hard family uses warm amber.  This makes
    # the compliance difference readable before the force plot is inspected.
    object_color = ("0.20 0.62 0.66 0.96" if stiffness < 0.5
                    else "0.95 0.58 0.18 0.96")
    body = (
        f'<body name="obj_{spec.name}" pos="{_xml_floats(spec.position)}" '
        f'quat="{_xml_floats(spec.quat)}">'
        f'<freejoint name="obj_{spec.name}_free"/>'
        f'<geom name="obj_{spec.name}_geom" type="{geom_type}" size="{geom_size}" '
        f'mass="{mass:.9g}" contype="2" conaffinity="2" condim="3" '
        f'friction="{_xml_floats(friction)}" solref="{contact_solref:.9g} 1" '
        f'solimp="0.99 0.999 0.0001" rgba="{object_color}"/>'
        f'</body>'
    )
    return "", body
