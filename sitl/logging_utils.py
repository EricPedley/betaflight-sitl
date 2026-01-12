import rerun as rr
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

file_path = Path(__file__).parent
logged_model_names = {}

fix_viz_rot = Rotation.from_euler("x", 180, degrees=True)
fix_model_rot = Rotation.identity()


def log_drone_pose(
    position: np.ndarray, quaternion: np.ndarray, model_name="drone/drone_model"
):
    if model_name not in logged_model_names:
        rr.log(model_name, rr.Asset3D(path=file_path / "Drone.obj"), static=True)
        logged_model_names[model_name] = True
    rr.log(
        model_name,
        rr.Transform3D(
            translation=fix_viz_rot.apply(position),
            quaternion=(fix_model_rot*fix_viz_rot * Rotation.from_quat(quaternion)).as_quat(),
        ),
        rr.TransformAxes3D(0.1),
        static=False,
    )


def log_gates(gate_info: list[tuple[np.ndarray, float]]):
    """
    Takes gate info as
    list of (position, yaw) tuples
    """
    for i, (gate_pos, gate_yaw) in enumerate(gate_info):
        # Log 3d gate model
        obj_file_path = file_path / "gate.obj"

        instance_path = f"gate_models/gate_{i}"
        rr.log(
            instance_path,
            rr.Transform3D(
                translation=fix_viz_rot.apply(gate_pos),
                rotation=rr.Quaternion(
                    xyzw=(
                        fix_viz_rot * Rotation.from_euler("XYZ", [0.0, 0.0, gate_yaw])
                    ).as_quat()
                ),
            ),
            static=True,
        )
        rr.log(
            f"{instance_path}/model",
            rr.Asset3D(path=obj_file_path, albedo_factor=[0.9, 0.9, 0.9, 1.0]),
            static=True,
        )


def log_velocity(
    position: np.ndarray, velocity: np.ndarray, model_name="drone/velocity"
):
    rr.log(
        model_name,
        rr.Arrows3D(
            origins=[fix_viz_rot.apply(position)],
            vectors=[fix_viz_rot.apply(velocity * 0.5)],  # Scale for visibility
            colors=[0, 0, 255],
        ),
    )
