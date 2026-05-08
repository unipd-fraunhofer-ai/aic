"""
To integrate a policy using ROS data structures, such as geometry_msgs.msg.Pose, sensor_msgs.msg.Image, and so on:

- define a Python class which derives from aic_model.Policy
- implement the insert_cable() method, which is called when aic_engine requests a new task.
- supply this Python class name as a parameter to aic_model at runtime.
"""

from unittest import result

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_control_interfaces.msg import (
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose, Quaternion, Transform
from geometry_msgs.msg import Vector3, Wrench
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from tf2_ros import TransformException
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from cv_bridge import CvBridge

from aic_perception.yolo_wrapper import YoloWrapper, plot_bboxes, plot_masks
from aic_perception.utils.pose_estimator import PoseEstimator

QuaternionTuple = tuple[float, float, float, float]

#########################################
# Utilities
#########################################
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

def transform_to_matrix(transform: Transform) -> np.ndarray:
    """Convert a geometry_msgs Transform to a 4x4 homogeneous transformation matrix."""
    translation = transform.translation
    rotation = transform.rotation
    T = np.eye(4)
    T[0:3, 3] = [translation.x, translation.y, translation.z]
    r = R.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
    T[0:3, 0:3] = r.as_matrix()
    return T


class VisionOnly(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.get_logger().info("VisionOnly.__init__()")
        
        self._task = None
        self.bridge = CvBridge()

        self.camera_names = ["center_camera", "left_camera", "right_camera"]
        self.camera_frames = {name: f"{name}/optical" for name in self.camera_names}
        
        self.class_names_map = {
            "nic_card_mount": 4,
            "task_board_base": 1
        }

        self.debug_mask = True

        # Path to your data folder
        policy_data_path = Path("/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data") #<--- CHANGE THIS TO YOUR LOCAL PATH
        yolo_checkpoint_path = policy_data_path / "weights_istances/yolo26_segment.pt"
        print(f"Loading YoloWrapper with checkpoint: {yolo_checkpoint_path}")

        self.yolo = YoloWrapper(yolo_checkpoint_path)   
        self.get_logger().info("Loaded YoloWrapper")
        
        self.pose_estimator = PoseEstimator(
            cameras={},
            templates_dir="templates_dir",      
            models_dir="models",
        )
        self.get_logger().info("Loaded PoseEstimator")
        self.mask_image_pub = {}
        for name in self.camera_names:
            self.mask_image_pub[name] = self._parent_node.create_publisher(Image, f"/pose_estimator/{name}_debug_mask_image", 10)


        
    def get_object_model_id(self, class_name: str):
        object_model_name = '_'.join(class_name.split("_")[:-1]) if class_name[-1].isdigit() else class_name
        object_model_id = self.class_names_map.get(object_model_name, None)
        return object_model_id


    def _wait_for_tf(
        self, target_frame: str, source_frame: str, timeout_sec: float = 10.0
    ) -> bool:
        """Wait for a TF frame to become available."""
        start = self.time_now()
        timeout = Duration(seconds=timeout_sec)
        attempt = 0
        while (self.time_now() - start) < timeout:
            try:
                self._parent_node._tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                )
                return True
            except TransformException:
                if attempt % 20 == 0:
                    self.get_logger().info(
                        f"Waiting for transform '{source_frame}' -> '{target_frame}'... -- are you running eval with `ground_truth:=true`?"
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False
    
    def _lookup_transform(self, target_frame: str, source_frame: str):
        """Lookup a TF transform, with error handling."""
        try:
            return self._parent_node._tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                Time(),
            )
        except TransformException as e:
            self.get_logger().error(
                f"Failed to lookup transform '{source_frame}' -> '{target_frame}': {e}"
            )
        return None

    def prepare_observations(self, obs_msg: Observation, world_frame: str = "world"):
        cameras = {}
        for name, frame in self.camera_frames.items():
            image_msg = getattr(obs_msg, name.replace("camera", "image"))
            camera_info_msg = getattr(obs_msg, name.replace("camera", "camera_info"))
            if image_msg is None or camera_info_msg is None:
                self.get_logger().error(f"Missing data for camera '{name}'")
                continue
            intrinsics = load_intrinsics(camera_info_msg)
            tf = self._lookup_transform(self.tool_frame, frame)
            
            T_world_camera = np.eye(4)
            if tf is not None:
                T_world_camera = transform_to_matrix(tf.transform)

            cameras[name] = {
                "image": image_msg,
                "intrinsics": intrinsics,
                "extrinsics": {
                    "R_w2c": T_world_camera[:3, :3].tolist(),
                    "t_w2c": T_world_camera[:3, 3].tolist(),
                },
            }
        return cameras
    
    def compute_masks(self, camera_name: str, image: Image):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        raw_results = self.yolo.predict(cv_image, keep_best=True)

        masks = {}
        names = {}
        for result in raw_results:
            class_name = result["class_name"]
            confidence = result["confidence"]
            class_id = result["class_id"]
            self.get_logger().info(f"[{camera_name}] YOLO result: {class_id} {class_name} ({confidence:.2f})")

            if class_name is None:
                self.get_logger().warning(f"Skipping YOLO result with no class name, class_id {class_id}, confidence {confidence:.2f}")
                continue

            object_model_id = self.get_object_model_id(class_name)
            if object_model_id not in self.class_names_map.values():
                self.get_logger().warning(f"Unknown object_model_id {object_model_id}, class_name {class_name} in YOLO results")
                continue

            if "mask" in result:
                masks.setdefault(object_model_id, []).append(result["mask"])
                names.setdefault(object_model_id, []).append(class_name)

        color = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0

        if self.debug_mask:
            image_mask = plot_masks(cv_image, raw_results)
            image_mask_msg = self.bridge.cv2_to_imgmsg(image_mask, encoding="bgr8")
            image_mask_msg.header = image.header
            self.mask_image_pub[camera_name].publish(image_mask_msg)

        return {
            "color": color,
            "masks": masks,
            "names": names,
        }


    """
    This method is called by aic_engine when a new task is requested.
    VisionOnly.insert_cable() task: aic_task_interfaces.msg.Task(id='task_1', cable_type='sfp_sc', cable_name='cable_0', plug_type='sfp', plug_name='sfp_tip', port_type='sfp', port_name='sfp_port_0', target_module_name='nic_card_mount_1', time_limit=180)
    """
    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"VisionOnly.insert_cable() task: {task}")
        self._task = task

        # Define the frames for the port and cable tip based on the task information
        port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"


        # Wait for camera extrinsics to be available in TF
        self.tool_frame = "tool0"
        for name, frame in self.camera_frames.items():
            if not self._wait_for_tf(self.tool_frame, frame):
                return False
            
        # Access to image and camera info
        cameras = self.prepare_observations(get_observation(), world_frame=self.tool_frame)
        print("Prepared camera observations")

        camera_inputs = {}
        for name, cam in cameras.items():

            cam_mask = self.compute_masks(name, cam["image"])
            camera_inputs[name] = cam_mask

        
        self.get_logger().info("VisionOnly.insert_cable() exiting...")
        return True
        