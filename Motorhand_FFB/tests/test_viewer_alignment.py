from pathlib import Path

import numpy as np

from glove_sim.config import load_config
from glove_sim.mapping import GloveMapping
from glove_sim.physics import Simulation
from motorhand_teleop.landmarks import Ve33LandmarkSource
from motorhand_teleop.viewer import SimulationGlovePose


def test_viewer_matches_simulation_motor_directions_and_limits():
    config = load_config()
    config["glove"]["collision_clamp"] = False
    mapping = GloveMapping.load(config["paths"]["glove_mapping"])
    asset_root = Path(config["paths"]["glove_model"]).parent.parent
    simulation = Simulation(config, mapping, [])
    viewer = SimulationGlovePose(Path(config["paths"]["glove_mapping"]), asset_root)
    landmarks = Ve33LandmarkSource("right", asset_root=asset_root)

    for motor in mapping.motors:
        for delta in (-2.0, 2.0):
            raw = {item.motor_id: item.zero_deg for item in mapping.motors}
            raw[motor.motor_id] += delta
            simulation.update_glove(np.asarray(mapping.raw_deg_to_model_rad(raw)))
            tree = viewer.solve(raw)
            displayed = np.array([
                simulation.data.joint(name).qpos[0]
                for name in viewer.kinematics.joint_names
            ])
            np.testing.assert_allclose(
                [tree[name] for name in viewer.kinematics.joint_names], displayed, atol=1e-8
            )
            frame = landmarks.evaluate_tree(tree)
            assert np.isfinite(frame.points_m).all()
            assert all(np.isfinite(value) for value in frame.glove_pose.joints.values())
