"""
To integrate a policy using ROS data structures, such as geometry_msgs.msg.Pose, sensor_msgs.msg.Image, and so on:

- define a Python class which derives from aic_model.Policy
- implement the insert_cable() method, which is called when aic_engine requests a new task.
- supply this Python class name as a parameter to aic_model at runtime.
"""

import json
from unittest import result

import numpy as np
import cv2
from pathlib import Path
from scipy.spatial.transform import Rotation

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
from geometry_msgs.msg import Point, Pose, Quaternion
from geometry_msgs.msg import TransformStamped, Transform
from geometry_msgs.msg import Vector3, Wrench
from rclpy.duration import Duration
from rclpy.time import Time
from rclpy.node import Node
from tf2_ros import TransformException
from tf2_ros import TransformBroadcaster
from transforms3d._gohlketransforms import quaternion_multiply, quaternion_slerp

from cv_bridge import CvBridge

from aic_perception.yolo_wrapper import YoloWrapper, plot_bboxes, plot_masks
from aic_perception.utils.pose_estimator import PoseEstimator
from aic_perception.utils.infer import save_visualizations

from aic_perception_policies.CheatCode import CheatCode

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
    r = Rotation.from_quat([rotation.x, rotation.y, rotation.z, rotation.w])
    T[0:3, 0:3] = r.as_matrix()
    return T


class VisionOnly(Policy):
    def __init__(self, parent_node: Node):
        super().__init__(parent_node)
        self.get_logger().info("VisionOnly.__init__()")
        
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None
        self.bridge = CvBridge()

        self.camera_names = ["center_camera", "left_camera", "right_camera"]
        self.camera_frames = {name: f"{name}/optical" for name in self.camera_names}
        
        self.class_names_map = {
            "nic_card_mount": 4,
            "task_board_base": 1,
            "sc_port": 5,
        }

        self.debug_mask = True
        

        # Path to your data folder
        self.policy_data_path = Path("/home/iaslab/ros2_ws/torch_ws/src/aic_perception/data") #<--- CHANGE THIS TO YOUR LOCAL PATH
        self.yolo_checkpoint_path = self.policy_data_path / "weights_istances/yolo26_segment.pt"
        print(f"Loading YoloWrapper with checkpoint: {self.yolo_checkpoint_path}")

        self.yolo = YoloWrapper(self.yolo_checkpoint_path)   
        self.get_logger().info("Loaded YoloWrapper")
        
        self.templates_dir = self.policy_data_path / "templates"
        self.models_dir = self.policy_data_path / "ic/models"

        self.nic_card_ports_filename = self.policy_data_path / "nic_card_merged_transforms.json"
        with open(self.nic_card_ports_filename, 'r') as f:
            self.nic_card_port_frames = json.load(f)

        self.sc_port_filename = self.policy_data_path / "sc_port_visual_pulito.json"
        with open(self.sc_port_filename, 'r') as f:
            self.sc_port_frames = json.load(f)

        self.pose_estimator = None
        # self.pose_estimator = PoseEstimator(
        #     cameras={},
        #     templates_dir="templates_dir",      
        #     models_dir="models",
        # )
        # self.get_logger().info("Loaded PoseEstimator")

        self.tf_broadcaster = TransformBroadcaster(self._parent_node)
        self.mask_image_pub = {}
        for name in self.camera_names:
            self.mask_image_pub[name] = self._parent_node.create_publisher(Image, f"/pose_estimator/{name}_debug_mask_image", 10)


        
    def get_object_model_id(self, class_name: str):
        object_model_name = '_'.join(class_name.split("_")[:-1]) if class_name[-1].isdigit() else class_name
        object_model_id = self.class_names_map.get(object_model_name, None)
        return object_model_id


    #######################################################################
    # Cheact code

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
    
    def calc_gripper_pose(
        self,
        port_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        reset_xy_integrator: bool = False,
    ) -> Pose:
        """Find the gripper pose that results in plug alignment."""
        q_port = (
            port_transform.rotation.w,
            port_transform.rotation.x,
            port_transform.rotation.y,
            port_transform.rotation.z,
        )
        plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            f"{self._task.cable_name}/{self._task.plug_name}_link",
            Time(),
        )
        q_plug = (
            plug_tf_stamped.transform.rotation.w,
            plug_tf_stamped.transform.rotation.x,
            plug_tf_stamped.transform.rotation.y,
            plug_tf_stamped.transform.rotation.z,
        )
        q_plug_inv = (
            -q_plug[0],
            q_plug[1],
            q_plug[2],
            q_plug[3],
        )
        q_diff = quaternion_multiply(q_port, q_plug_inv)
        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        q_gripper = (
            gripper_tf_stamped.transform.rotation.w,
            gripper_tf_stamped.transform.rotation.x,
            gripper_tf_stamped.transform.rotation.y,
            gripper_tf_stamped.transform.rotation.z,
        )
        q_gripper_target = quaternion_multiply(q_diff, q_gripper)
        q_gripper_slerp = quaternion_slerp(q_gripper, q_gripper_target, slerp_fraction)

        gripper_xyz = (
            gripper_tf_stamped.transform.translation.x,
            gripper_tf_stamped.transform.translation.y,
            gripper_tf_stamped.transform.translation.z,
        )
        port_xy = (
            port_transform.translation.x,
            port_transform.translation.y,
        )
        plug_xyz = (
            plug_tf_stamped.transform.translation.x,
            plug_tf_stamped.transform.translation.y,
            plug_tf_stamped.transform.translation.z,
        )
        plug_tip_gripper_offset = (
            gripper_xyz[0] - plug_xyz[0],
            gripper_xyz[1] - plug_xyz[1],
            gripper_xyz[2] - plug_xyz[2],
        )

        tip_x_error = port_xy[0] - plug_xyz[0]
        tip_y_error = port_xy[1] - plug_xyz[1]

        if reset_xy_integrator:
            self._tip_x_error_integrator = 0.0
            self._tip_y_error_integrator = 0.0
        else:
            self._tip_x_error_integrator = np.clip(
                self._tip_x_error_integrator + tip_x_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )
            self._tip_y_error_integrator = np.clip(
                self._tip_y_error_integrator + tip_y_error,
                -self._max_integrator_windup,
                self._max_integrator_windup,
            )

        self.get_logger().info(
            f"pfrac: {position_fraction:.3} xy_error: {tip_x_error:0.3} {tip_y_error:0.3}   integrators: {self._tip_x_error_integrator:.3} , {self._tip_y_error_integrator:.3}"
        )

        i_gain = 0.15

        target_x = port_xy[0] + i_gain * self._tip_x_error_integrator
        target_y = port_xy[1] + i_gain * self._tip_y_error_integrator
        target_z = port_transform.translation.z + z_offset - plug_tip_gripper_offset[2]

        blend_xyz = (
            position_fraction * target_x + (1.0 - position_fraction) * gripper_xyz[0],
            position_fraction * target_y + (1.0 - position_fraction) * gripper_xyz[1],
            position_fraction * target_z + (1.0 - position_fraction) * gripper_xyz[2],
        )

        return Pose(
            position=Point(
                x=blend_xyz[0],
                y=blend_xyz[1],
                z=blend_xyz[2],
            ),
            orientation=Quaternion(
                w=q_gripper_slerp[0],
                x=q_gripper_slerp[1],
                y=q_gripper_slerp[2],
                z=q_gripper_slerp[3],
            ),
        )
    
    #######################################################################
    
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

    def publish_object_tf(self, pose, frame_id, tf_name: str):
        tf_msg = TransformStamped()

        #tf_msg.header.stamp = header.stamp
        #tf_msg.header.frame_id = header.frame_id
        tf_msg.header.stamp = self.time_now().to_msg()
        tf_msg.header.frame_id = frame_id
        tf_msg.child_frame_id = tf_name

        tf_msg.transform = self.pose_to_transform(pose, scale_factor=1000.0)  # Convert from mm to m

        self.tf_broadcaster.sendTransform(tf_msg)

    def pose_to_transform(self, pose, scale_factor=1000.0) -> Transform:
        transform = Transform()

        #print(pose) A dict {'object_id', 'R_m2c', 't_m2c', 'T_m2c', 'T_m2w','quality', 'num_inliers', 'template_id', 'corresp_id'}
        t = pose['t_m2c'] / scale_factor  # meters
        q = Rotation.from_matrix(pose['R_m2c']).as_quat()  # x,y,z,w

        transform.translation.x = float(t[0])
        transform.translation.y = float(t[1])
        transform.translation.z = float(t[2])
        transform.rotation.x = float(q[0])
        transform.rotation.y = float(q[1])
        transform.rotation.z = float(q[2])
        transform.rotation.w = float(q[3])

        return transform

    def prepare_observations(self, obs_msg: Observation, world_frame: str = "world"):
        cameras = {}
        for name, frame in self.camera_frames.items():
            image_msg = getattr(obs_msg, name.replace("camera", "image"))
            camera_info_msg = getattr(obs_msg, name.replace("camera", "camera_info"))
            if image_msg is None or camera_info_msg is None:
                self.get_logger().error(f"Missing data for camera '{name}'")
                continue
            intrinsics = load_intrinsics(camera_info_msg)
            tf = self._lookup_transform(frame, world_frame) #world  wrt camera
            
            T_world_camera = np.eye(4)
            if tf is not None:
                T_world_camera = transform_to_matrix(tf.transform)

            # Pose estimator expect mm
            T_world_camera[:3, 3] = T_world_camera[:3, 3] * 1000.0 # Convert from m to mm

            cameras[name] = {
                "image": image_msg,
                "intrinsics": intrinsics,
                "extrinsics": {
                    "R_w2c": T_world_camera[:3, :3].tolist(),
                    "t_w2c": T_world_camera[:3, 3].tolist(),
                },
            }

            print(f"Camera '{name}': intrinsics: {intrinsics}, extrinsics (world to camera): R=\n{cameras[name]['extrinsics']['R_w2c']}, \nt=\n{cameras[name]['extrinsics']['t_w2c']}")
        return cameras
    
    def compute_masks(self, camera_name: str, image: Image, target_name=None):
        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        raw_results = self.yolo.predict(cv_image, keep_best=True)

        masks = {}
        names = {}
        confs = {}
        for result in raw_results:
            class_name = result["class_name"]
            confidence = result["confidence"]
            class_id = result["class_id"]
            self.get_logger().info(f"[{camera_name}] YOLO result: {class_id} {class_name} ({confidence:.2f})")

            # Search only for the target object if target_name is specified
            if target_name is not None and class_name != target_name:
                self.get_logger().warning(f"Skipping YOLO result with class_name {class_name} since it does not match target_name {target_name}")
                continue

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
                confs.setdefault(object_model_id, []).append(confidence)
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
            "confs": confs,
        }
    
    def compute_port_pose(self, best_pose, T_world_camera, T_model_to_port=None, scale_factor=1000.0):
        T_m2c = best_pose['T_m2c']

        T_m2c[0:3, 3] = T_m2c[0:3, 3] / scale_factor  # Convert from mm to m

        T_world_m = T_world_camera @ T_m2c
        
        if T_model_to_port is not None:
            T_world_m = T_world_m @ T_model_to_port

        transform = Transform()
        transform.translation.x = float(T_world_m[0, 3])
        transform.translation.y = float(T_world_m[1, 3])
        transform.translation.z = float(T_world_m[2, 3])
        r = Rotation.from_matrix(T_world_m[:3, :3])
        q = r.as_quat()  # x,y,z,w
        transform.rotation.x = float(q[0])
        transform.rotation.y = float(q[1])
        transform.rotation.z = float(q[2])
        transform.rotation.w = float(q[3])
        return transform

    def quaternion_to_rotation_matrix(self, q: Quaternion) -> np.ndarray:
        return Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    def transform_stamped_to_matrix(self, tf_msg: TransformStamped) -> np.ndarray:
        q = Quaternion()
        q.x = tf_msg.transform.rotation.x
        q.y = tf_msg.transform.rotation.y
        q.z = tf_msg.transform.rotation.z
        q.w = tf_msg.transform.rotation.w

        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.quaternion_to_rotation_matrix(q)
        T[:3, 3] = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ],
            dtype=np.float64,
        )
        return T


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
        self.world_frame = "base_link"
        for name, frame in self.camera_frames.items():
            if not self._wait_for_tf(self.world_frame, frame):
                return False
            
        # Access to image and camera info
        cameras = self.prepare_observations(get_observation(), world_frame=self.world_frame)
        print("Prepared camera observations")

        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator(
                cameras=cameras,
                templates_dir=self.templates_dir,      
                models_dir=self.models_dir,
            )
            self.get_logger().info("Loaded PoseEstimator")

        camera_inputs = {}
        best_camera = None
        best_quality = -1.0
        for name, cam in cameras.items():

            cam_mask = self.compute_masks(name, cam["image"], target_name=task.target_module_name)
            camera_inputs[name] = cam_mask

            if cam_mask["names"]:
                # If we have detections, use the highest confidence one for pose estimation
                max_conf = max([max(confs) for confs in cam_mask["confs"].values()])
                if max_conf > best_quality:
                    best_quality = max_conf
                    best_camera = name

        self.get_logger().info(f"Best camera for pose estimation: {best_camera} with quality {best_quality:.2f}")


        pose_results = self.pose_estimator.estimate_pose(camera_inputs)

        try:
            data_dir = self.policy_data_path / "ic"
            output_dir = self.policy_data_path  / "visualizations"
            scene_id = 1
            frame_id = 0  # Assuming single frame for now
            save_visualizations(data_dir, output_dir, cameras, camera_inputs, pose_results, scene_id, frame_id)
            scene_id += 1
        except Exception as e:
            self.get_logger().error(f"Failed to save visualizations: {e}")

        best_pose = None

        for camera_name, camera_results in pose_results.items():

            # Temporarily only use the best camera for pose estimation, since we don't have a good way to fuse multiple views yet.
            if camera_name != best_camera:
                self.get_logger().info(f"Skipping pose results from camera {camera_name} since it's not the best camera")
                continue

            print(f"Camera: {camera_name} - results: {camera_results.keys()}")

            for object_id, poses in camera_results.items():

                print(f"Camera: {camera_name}-{object_id} - results: {len(poses)}")

                for instance_id, pose in enumerate(poses):
                    quality = None if pose is None else pose["quality"]
                    print(camera_name, object_id, instance_id, quality)
                    # tf_name = camera_inputs[camera_name]["names"][object_id][instance_id] if object_id in camera_inputs[camera_name]["names"] else "unknown"
                    # print(camera_name, object_id, instance_id, quality, '->', tf_name)

                    tf_name = f"{camera_name}_object{object_id}_instance{instance_id}"

                    self.publish_object_tf(pose, self.camera_frames[camera_name], tf_name)

                    print(f"camera: {camera_name}")
                    print(pose['T_m2w'])

                # Use the best pose (highest quality) for this object to publish a TF for the object in the world frame
                sorted_poses = sorted(poses, key=lambda p: p["quality"] if p is not None else -1.0, reverse=True)
                best_pose = sorted_poses[0] if sorted_poses else None
                tf_name = f"{camera_name}_object{object_id}"
                self.publish_object_tf(best_pose, self.camera_frames[camera_name], tf_name)

        self.get_logger().info("VisionOnly.insert_cable() port pose estimated. Starting robot motion...")

        #port_transform = self.pose_to_transform(best_pose, scale_factor=1000.0)  # Convert from mm to m
        
        print(f"Best pose:\n{best_pose}")

        T_world_camera = self.transform_stamped_to_matrix(self._lookup_transform(self.world_frame, self.camera_frames[best_camera]))
        print(f"T_world_camera:\n{T_world_camera}")

        # Retrieve relative transofrm between module and port
        port_name = f'{task.port_name}_link'        
        #port_name = f'{task.port_name}_link_entrance'
        if 'nic_card_mount' in task.target_module_name:
            t_model_to_port = self.nic_card_port_frames[port_name]['t_ply']
            q_model_to_port = self.nic_card_port_frames[port_name]['q_ply']
        elif 'sc_port' in task.target_module_name:
            t_model_to_port = self.sc_port_frames[port_name]['t_ply']
            q_model_to_port = self.sc_port_frames[port_name]['q_ply']
        else:
            self.get_logger().error(f"Unknown target module {task.target_module_name} in task, cannot retrieve model to port transform")

        print(f"Relative transform from model {task.target_module_name} to port {port_name}\n t: {t_model_to_port}\n q: {q_model_to_port}")


        # Apply transform from model to port frame to get it in the world frame
        T_model_to_port = np.eye(4)
        T_model_to_port[:3, 3] = np.array(t_model_to_port)
        T_model_to_port[:3, :3] = Rotation.from_quat(q_model_to_port).as_matrix()


        port_transform = self.compute_port_pose(best_pose, T_world_camera, T_model_to_port)
        print(f"Computed port transform:\n{port_transform}")


        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.time_now().to_msg()
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = f'{tf_name}_port_entrance'
        tf_msg.transform = port_transform
        self.tf_broadcaster.sendTransform(tf_msg)


        ###################################################################
        z_offset = 0.2

        # Over five seconds, smoothly interpolate from the current position to
        # a position above the port.
        for t in range(0, 100):
            interp_fraction = t / 100.0
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(
                        port_transform,
                        slerp_fraction=interp_fraction,
                        position_fraction=interp_fraction,
                        z_offset=z_offset,
                        reset_xy_integrator=True,
                    ),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during interpolation: {ex}")
            self.sleep_for(0.05)

        # Descend until the cable is inserted into the port.
        while True:
            if z_offset < -0.015:
                break

            z_offset -= 0.0005
            self.get_logger().info(f"z_offset: {z_offset:0.5}")
            try:
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=self.calc_gripper_pose(port_transform, z_offset=z_offset),
                )
            except TransformException as ex:
                self.get_logger().warn(f"TF lookup failed during insertion: {ex}")
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for connector to stabilize...")
        self.sleep_for(5.0)
        ###################################################################

        
        self.get_logger().info("VisionOnly.insert_cable() exiting...")
        return True
        