"""
A collection of utility functions for vision-based perception policies.
"""
import json
import numpy as np
from scipy.spatial.transform import Rotation

from geometry_msgs.msg import Transform, TransformStamped
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
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T