import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from bop_toolkit_lib import inout, visualization
from bop_toolkit_lib.rendering import renderer

from aic_perception.utils.pose_estimator import PoseEstimator


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_camera(data_dir, scene_dir, camera_name, frame_id):
    intrinsics = load_json(data_dir / f"camera_{camera_name}.json")
    scene_camera = load_json(scene_dir / f"scene_camera_{camera_name}.json")[str(frame_id)]
    return {
        "intrinsics": intrinsics,
        "extrinsics": {
            "R_w2c": scene_camera["R_w2c"],
            "t_w2c": scene_camera["t_w2c"],
        },
    }


def load_camera_input(scene_dir, camera_name, frame_id):
    image_path = scene_dir / f"rgb_{camera_name}" / f"{frame_id:06d}.png"
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    scene_gt = load_json(scene_dir / f"scene_gt_{camera_name}.json")[str(frame_id)]
    masks = {}
    for instance_id, annotation in enumerate(scene_gt):
        mask_path = scene_dir / f"mask_{camera_name}" / f"{frame_id:06d}_{instance_id:06d}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        masks.setdefault(annotation["obj_id"], []).append(mask)

    return {
        "color": image,
        "masks": masks,
    }


def save_visualizations(data_dir, output_dir, cameras, camera_inputs, results, scene_id, frame_id):
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = inout.load_json(Path(visualization.__file__).parent / "colors.json")

    for camera_name, camera_results in results.items():
        image = (camera_inputs[camera_name]["color"] * 255).astype(np.uint8)
        intrinsics = cameras[camera_name]["intrinsics"]
        K = np.array(
            [
                [intrinsics["fx"], 0, intrinsics["cx"]],
                [0, intrinsics["fy"], intrinsics["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        ren = renderer.create_renderer(
            image.shape[1],
            image.shape[0],
            renderer_type="vispy",
            mode="rgb",
            shading="flat",
        )
        vis_poses = []

        for object_id, poses in camera_results.items():
            model_path = data_dir / "models" / f"obj_{object_id:06d}.ply"
            model_color = tuple(colors[(object_id - 1) % len(colors)])
            ren.add_object(object_id, str(model_path), surf_color=model_color)

            for instance_id, pose in enumerate(poses):
                if pose is None:
                    continue

                vis_poses.append(
                    {
                        "obj_id": object_id,
                        "R": pose["R_m2c"],
                        "t": pose["t_m2c"],
                        "text_info": [
                            {
                                "name": "",
                                "fmt": "",
                                "val": f"{object_id}:{pose['quality']:.0f}",
                            }
                        ],
                    }
                )

        vis = image
        if vis_poses:
            vis = visualization.vis_object_poses(
                poses=vis_poses,
                K=K,
                renderer=ren,
                rgb=image,
                vis_rgb=True,
                vis_rgb_resolve_visib=False,
            )["vis_im_rgb"]

        out_path = output_dir / f"scene_{scene_id:06d}_frame_{frame_id:06d}_{camera_name}.png"
        inout.save_im(out_path, vis)
        print(f"Saved visualization: {out_path}")


def main():
    # data_dir = Path(__file__).resolve().parent / "data" / "ic"
    # templates_dir = Path(__file__).resolve().parent / "data" / "templates"
    # output_dir = Path(__file__).resolve().parent / "visualizations"
    data_dir = Path("data") / "ic"
    templates_dir = Path("data") / "templates"
    output_dir = Path("visualizations")
    scene_id = 1
    frame_id = 0
    camera_names = ["center", "left", "right"]

    scene_dir = data_dir / "test" / f"{scene_id:06d}"
    cameras = {
        name: load_camera(data_dir, scene_dir, name, frame_id)
        for name in camera_names
    }
    camera_inputs = {
        name: load_camera_input(scene_dir, name, frame_id)
        for name in camera_names
    }

    estimator = PoseEstimator(
        cameras=cameras,
        templates_dir=templates_dir,
        models_dir=data_dir / "models",
    )

    start_time = time.time()
    results = estimator.estimate_pose(camera_inputs)
    end_time = time.time()
    print(f"Inference time: {end_time - start_time:.2f} seconds")
    
    save_visualizations(data_dir, output_dir, cameras, camera_inputs, results, scene_id, frame_id)

    for camera_name, camera_results in results.items():
        for object_id, poses in camera_results.items():
            for instance_id, pose in enumerate(poses):
                quality = None if pose is None else pose["quality"]
                print(camera_name, object_id, instance_id, quality)


if __name__ == "__main__":
    main()
