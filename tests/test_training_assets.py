import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import pytest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "assets/training_models.json").read_text(encoding="utf-8"))
MODELS = MANIFEST["models"]


@pytest.mark.parametrize("entry", MODELS, ids=lambda entry: entry["id"])
def test_model_entry_and_local_references(entry):
    path = ROOT / entry["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    root = ET.parse(path).getroot()
    assert root.tag == ("robot" if entry["format"] == "urdf" else "mujoco")
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    meshes = list(root.iter("mesh"))
    assert meshes
    for mesh in meshes:
        filename = mesh.get("filename") or mesh.get("file")
        if filename is not None:
            assert not Path(filename).is_absolute()
            reference = path.parent / (meshdir if mesh.get("file") else "") / filename
            assert reference.is_file(), reference
    if entry["format"] == "urdf":
        links = {link.get("name") for link in root.findall("link")}
        assert links
        for joint in root.findall("joint"):
            assert joint.find("parent").get("link") in links
            assert joint.find("child").get("link") in links
        named_materials = {material.get("name") for material in root.iter("material")
                           if material.get("name") and len(material)}
        for material in root.iter("material"):
            if material.get("name") and not len(material):
                assert material.get("name") in named_materials
    else:
        assets = root.find("asset")
        materials = {material.get("name") for material in assets.findall("material")}
        for element in root.iter():
            if element.get("material"):
                assert element.get("material") in materials
        for include in root.iter("include"):
            assert (path.parent / include.attrib["file"]).is_file()


@pytest.mark.parametrize("entry", [entry for entry in MODELS if entry["format"] == "mjcf"],
                         ids=lambda entry: entry["id"])
def test_mjcf_loads_offline(entry):
    model = mujoco.MjModel.from_xml_path(str(ROOT / entry["path"]))
    assert model.nbody > 1
    if entry["id"].startswith("ve33-"):
        assert model.neq == 56
    else:
        assert model.nq == 20
