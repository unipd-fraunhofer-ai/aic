"""
A collection of utility functions for vision-based perception policies.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Transform, TransformStamped
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo

####################################################
# Vision configuations

# Mapping from object general class to object model ID (ply file name)
CLASS_NAMES_MAP = {
    "nic_card_mount": 4,
    "task_board_base": 1,
    "sc_port": 5,
}

# Predefined cable tip frames and their transforms relative to the gripper TCP
CABLE_TIP_FRAMES = {
    'sc_tip_link': {
        't_gripper_to_tip': [-0.0005699, -0.0005699, 0.0096407],
        'q_gripper_to_tip': [-0.2298133984379278, 0.22655156773159751, -0.6627472977643364, -0.6757412205316223],
    },
    'sfp_tip_link': {
        't_gripper_to_tip': [-0.000, -0.020687, 0.054119],
        'q_gripper_to_tip': [-0.17785966749625665, -0.00503708733179058, 0.027383843138112103, -0.983661891514159],
    }
}


####################################################
# Load functions
def load_intrinsics(camera_info_msg: CameraInfo):
    k = camera_info_msg.k
    fx = float(k[0])
    fy = float(k[4])
    cx = float(k[2])
    cy = float(k[5])
    return {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": camera_info_msg.width,
        "height": camera_info_msg.height,
        "depth_scale": 1.0,
    }

def load_model_frames(filename):
    with open(filename, 'r') as f:
        return json.load(f)

####################################################
# Pose estimation utilities
def get_object_model_id(class_name: str):
    object_model_name = '_'.join(class_name.split("_")[:-1]) if class_name[-1].isdigit() else class_name
    object_model_id = CLASS_NAMES_MAP.get(object_model_name, None)
    return object_model_id

def get_best_camera(camera_segmentation_results, quality_threshold=0.5):
    best_camera = None
    best_quality = -1.0
    for name, cam_mask in camera_segmentation_results.items():

        if cam_mask["names"]:
            # If we have detections, use the highest confidence one for pose estimation
            max_conf = max([max(confs) for confs in cam_mask["confs"].values()])
            if max_conf > best_quality:
                best_quality = max_conf
                best_camera = name
    print(f"Best camera: {best_camera} with quality {best_quality}")

    if best_quality < quality_threshold:
        print(f"Best camera quality {best_quality} is below threshold {quality_threshold}, rejecting all cameras")
        return None
    
    return best_camera
    
    
def get_best_pose(pose_results, best_camera_name=None, quality_threshold=0.5):

    best_pose_candidates = []

    for camera_name, camera_results in pose_results.items():
        if camera_name is not None and camera_name != best_camera_name:
            print(f"Skipping pose results from camera {camera_name} since it's not the best camera")
            continue

        print(f"Camera: {camera_name} - results: {camera_results.keys()}")
        
        for object_id, poses in camera_results.items():
            print(f"Camera: {camera_name}-{object_id} - results: {len(poses)}")
            
            for instance_id, pose in enumerate(poses):
                quality = None if pose is None else pose["quality"]
                print(camera_name, object_id, instance_id, quality)

            best_pose_candidates.append(pose)

    # Use the highest quality pose 
    sorted_poses = sorted(best_pose_candidates, key=lambda p: p["quality"] if p is not None else -1.0, reverse=True)
    best_pose = sorted_poses[0] if sorted_poses else None

    if best_pose is not None and best_pose["quality"] < quality_threshold:
        print(f"Best pose quality {best_pose['quality']} is below threshold {quality_threshold}, rejecting pose")
        best_pose = None

    return best_pose


###############################################
# Transform utilities
def transform_to_matrix(transform: Transform | TransformStamped) -> np.ndarray:
    if isinstance(transform, TransformStamped):
        transform = transform.transform
    """Convert a geometry_msgs Transform to a 4x4 homogeneous transformation matrix."""
    translation = transform.translation
    rotation = transform.rotation
    T = np.eye(4)
    T[0:3, 3] = [translation.x, translation.y, translation.z]
    r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
    T[0:3, 0:3] = r.as_matrix()
    return T

def matrix_to_transform(T: np.ndarray) -> Transform:
    """Convert a 4x4 homogeneous transformation matrix to a geometry_msgs Transform."""
    translation = T[0:3, 3]
    r = Rotation.from_matrix(T[0:3, 0:3])
    rotation = r.as_quat()  # returns (x, y, z, w)
    transform = Transform()
    transform.translation.x = translation[0]
    transform.translation.y = translation[1]
    transform.translation.z = translation[2]
    transform.rotation.x = rotation[0]
    transform.rotation.y = rotation[1]
    transform.rotation.z = rotation[2]
    transform.rotation.w = rotation[3]
    return transform

def matrix_from_Rt(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Construct a 4x4 homogeneous transformation matrix from rotation and translation."""
    R = np.asarray(R).reshape(3, 3)
    t = np.asarray(t).reshape(3)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t

    return T


###############################################
# Robot utilities
def random_pose_increment(current_pose: Pose, position_scale=0.01, orientation_scale=None) -> Pose:
    """Generate a random pose increment for exploration."""
    delta_position = np.random.uniform(-position_scale, position_scale, size=3)
    if orientation_scale is None:
        delta_orientation = np.array([0.0, 0.0, 0.0, 0.0])  # No rotation
    else:
        delta_orientation = Rotation.from_euler('xyz', np.random.uniform(-orientation_scale, orientation_scale, size=3)).as_quat()
    new_pose = Pose()
    new_pose.position.x = current_pose.position.x + delta_position[0]
    new_pose.position.y = current_pose.position.y + delta_position[1]
    new_pose.position.z = current_pose.position.z + delta_position[2]
    new_pose.orientation.x = current_pose.orientation.x + delta_orientation[0]
    new_pose.orientation.y = current_pose.orientation.y + delta_orientation[1]
    new_pose.orientation.z = current_pose.orientation.z + delta_orientation[2]
    new_pose.orientation.w = current_pose.orientation.w + delta_orientation[3]
    print(f"Generated random pose increment: Δposition={delta_position}, Δorientation={delta_orientation}")
    print(f"New pose: {new_pose}")
    return new_pose