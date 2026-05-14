
import json
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

import aic_perception_policies.vision_utils as vision_utils

QuaternionTuple = tuple[float, float, float, float]

#<--- CHANGE THIS TO YOUR LOCAL PATH 
POLICY_DATA_PATH = "aic_perception/data"
#------------------------------------------------------------

class VisionBase(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("VisionBase.__init__()")

        # Cheat code state for integrator in calc_gripper_pose
        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None

        # Path to your data folder
        self.policy_data_path = Path(POLICY_DATA_PATH)
        self.yolo_checkpoint_path = self.policy_data_path / "weights_istances/yolo26_segment.pt"
        self.templates_dir = self.policy_data_path / "templates"
        self.models_dir = self.policy_data_path / "ic/models"
        self.nic_card_ports_filename = self.policy_data_path / "nic_card_merged_transforms.json"
        self.sc_port_filename = self.policy_data_path / "sc_port_visual_pulito.json"

        # Perception variables
        self.camera_names = ["center_camera", "left_camera", "right_camera"]
        self.camera_frames = {name: f"{name}/optical" for name in self.camera_names}

        self.nic_card_port_frames = vision_utils.load_model_frames(self.nic_card_ports_filename)
        self.sc_port_frames = vision_utils.load_model_frames(self.sc_port_filename)

        self.pose_estimator = None
        self.yolo = YoloWrapper(self.yolo_checkpoint_path)  
        self.get_logger().info("Loaded YoloWrapper")

        # ROS variables
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self._parent_node)

        # Debug        
        self.debug_mask = True
        self.ground_truth_available = False

        self.mask_image_pub = {}
        for name in self.camera_names:
            self.mask_image_pub[name] = self._parent_node.create_publisher(Image, f"/pose_estimator/{name}_debug_mask_image", 10)

        self.get_logger().info("VisionBase policy initialized.")


    #######################################################################
    # CheatCode utiilities
    #######################################################################
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

        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )

        if self.ground_truth_available:
            plug_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                f"{self._task.cable_name}/{self._task.plug_name}_link",
                Time(),
            )
        else:
            # Compute tip pose in the world frame (when plug_tf_stamped not available in evalutation)
            plug_name = f"{self._task.plug_name}_link"
            t_gripper_to_tip = vision_utils.CABLE_TIP_FRAMES[plug_name]['t_gripper_to_tip']
            q_gripper_to_tip = vision_utils.CABLE_TIP_FRAMES[plug_name]['q_gripper_to_tip']

            T_gripper_to_tip = vision_utils.matrix_from_Rt(
                R=Rotation.from_quat(q_gripper_to_tip).as_matrix(),
                t=t_gripper_to_tip,
            )
            
            T_world_gripper = vision_utils.transform_to_matrix(gripper_tf_stamped.transform)
            T_world_tip = T_world_gripper @ T_gripper_to_tip

            plug_tf_stamped = TransformStamped()
            plug_tf_stamped.header.stamp = gripper_tf_stamped.header.stamp
            plug_tf_stamped.header.frame_id = gripper_tf_stamped.header.frame_id
            plug_tf_stamped.child_frame_id = plug_name
            plug_tf_stamped.transform = vision_utils.matrix_to_transform(T_world_tip)
            print(f"Computed plug_tf_stamped:\n{plug_tf_stamped}")

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
    # Vision utilities
    #######################################################################
    def get_camera_observations(self, obs_msg: Observation, world_frame: str = "world"):
        cameras = {}
        for name, frame in self.camera_frames.items():
            image_msg = getattr(obs_msg, name.replace("camera", "image"))
            camera_info_msg = getattr(obs_msg, name.replace("camera", "camera_info"))
            if image_msg is None or camera_info_msg is None:
                self.get_logger().error(f"Missing data for camera '{name}'")
                continue
            intrinsics = vision_utils.load_intrinsics(camera_info_msg)

            # NOTE: Pose estimator requires world wrt camera (in millimeters!)
            extrinsics = self._parent_node._tf_buffer.lookup_transform(
                frame,
                world_frame,
                Time(),
            )
            
            T_world_camera = vision_utils.transform_to_matrix(extrinsics.transform)
            T_world_camera[:3, 3] = T_world_camera[:3, 3] * 1000.0 # Convert from m to mm

            cameras[name] = {
                "image": image_msg,
                "intrinsics": intrinsics,
                "extrinsics": {
                    "R_w2c": T_world_camera[:3, :3].tolist(),
                    "t_w2c": T_world_camera[:3, 3].tolist(),
                },
            }
            #print(f"Camera '{name}': intrinsics: {intrinsics}, extrinsics (world to camera): R=\n{cameras[name]['extrinsics']['R_w2c']}, \nt=\n{cameras[name]['extrinsics']['t_w2c']}")
        return cameras
    
    def compute_segmentation_masks(self, camera_name: str, image: Image, target_name=None):
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

            object_model_id = vision_utils.get_object_model_id(class_name)
            if object_model_id not in vision_utils.CLASS_NAMES_MAP.values():
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
    
    def compute_poses(
            self, cameras, camera_inputs, 
            scale_factor=1000.0,
            mask_quality_threshold=0.5,
            pose_quality_threshold=0.5
        ):
        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator(
                cameras=cameras,
                templates_dir=self.templates_dir,      
                models_dir=self.models_dir,
            )
            self.get_logger().info("Loaded PoseEstimator")

        pose_detections = self.pose_estimator.estimate_pose(camera_inputs)

        # filter to select the best pose based on the best camera quality
        best_camera_name = vision_utils.get_best_camera(
            camera_inputs,
            quality_threshold=mask_quality_threshold
        )
        best_pose = vision_utils.get_best_pose(
            pose_detections,
            best_camera_name=best_camera_name,
            quality_threshold=pose_quality_threshold,
        )

        if best_pose is not None:
            best_pose['t_m2c'] = best_pose['t_m2c'] / scale_factor  # meters
            best_pose['T_m2c'] = vision_utils.matrix_from_Rt(
                R=Rotation.from_matrix(best_pose['R_m2c']).as_matrix(),
                t=np.array(best_pose['t_m2c']),
            )
            best_pose['camera_name'] = best_camera_name

        return best_pose
        

    def get_port_local_transform(self, target_module_name, port_name: str):
        if 'nic_card_mount' in target_module_name:
            t_model_to_port = self.nic_card_port_frames[port_name]['t_ply']
            q_model_to_port = self.nic_card_port_frames[port_name]['q_ply']
        elif 'sc_port' in target_module_name:
            t_model_to_port = self.sc_port_frames[port_name]['t_ply']
            q_model_to_port = self.sc_port_frames[port_name]['q_ply']
        else:
            self.get_logger().error(f"Unknown target module {target_module_name} in task, cannot retrieve model to port transform")
            return None
        
        T_model_to_port = vision_utils.matrix_from_Rt(
            R=Rotation.from_quat(q_model_to_port).as_matrix(),
            t=np.array(t_model_to_port),
        )
        return T_model_to_port
    
    def compute_port_transform(self, T_camera_model, T_world_camera, T_model_to_port=None):
        
        T_world_m = T_world_camera @ T_camera_model
        
        if T_model_to_port is not None:
            T_world_m = T_world_m @ T_model_to_port

        transform = vision_utils.matrix_to_transform(T_world_m)
        return transform    
    
    def publish_debug_tf(self, transform: Transform, frame_id: str, child_frame_id: str):
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = self.time_now().to_msg()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        tf_stamped.transform = transform
        self.tf_broadcaster.sendTransform(tf_stamped)
        
    #######################################################################

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ):
        self.get_logger().info(f"VisionBase.insert_cable() task: {task}")
        self._task = task

        # port_frame = f"task_board/{task.target_module_name}/{task.port_name}_link"
        # cable_tip_frame = f"{task.cable_name}/{task.plug_name}_link"
        # try:
        #     port_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
        #         "base_link",
        #         port_frame,
        #         Time(),
        #     )
        # except TransformException as ex:
        #     self.get_logger().error(f"Could not look up port transform: {ex}")
        #     return False
        # port_transform = port_tf_stamped.transform
        port_transform = None

        # TODO: move arm in predefined start configuration?

        ######################################### <-- Perception

        # Wait for camera extrinsics to be available in TF
        self.world_frame = "base_link"
        for name, frame in self.camera_frames.items():
            if not self._wait_for_tf(self.world_frame, frame):
                return False
        
        # Search until a high confidence pose is detected
        while port_transform is None:

            observation = get_observation()
            # Access to image and camera info
            cameras = self.get_camera_observations(observation, world_frame=self.world_frame)
            self.get_logger().info(f"[Perception] Camera observations obtained for cameras: {list(cameras.keys())}")

            # Segment only the required module in the task board to simplify pose estimation
            camera_inputs = {}
            for name, cam in cameras.items():
                camera_inputs[name] = self.compute_segmentation_masks(
                    camera_name=name, 
                    image=cam["image"], 
                    target_name=task.target_module_name
                )

            # Compute poses based on the segmented masks
            detected_pose = self.compute_poses(
                cameras, 
                camera_inputs,
                mask_quality_threshold=0.5,
                pose_quality_threshold=0.5,
            )

            if detected_pose is not None:
                T_world_camera = vision_utils.transform_to_matrix(
                    self._parent_node._tf_buffer.lookup_transform(
                        'base_link', 
                        self.camera_frames[detected_pose['camera_name']],
                        Time(),
                        )
                    )
                
                #port_name = f'{task.port_name}_link'  
                #port_name = f'{task.port_name}_link_entrance'

                T_model_to_port = self.get_port_local_transform(
                    target_module_name=task.target_module_name,
                    port_name=f"{task.port_name}_link",
                )
                port_transform = self.compute_port_transform(
                    T_camera_model=detected_pose['T_m2c'],
                    T_world_camera=T_world_camera,
                    T_model_to_port=T_model_to_port,
                )

                best_camera_name = detected_pose['camera_name']
                tf_name = f"{best_camera_name}_{task.port_name}_link"
                self.publish_debug_tf(port_transform, frame_id='base_link', child_frame_id=tf_name)
                break

            robot_pose = observation.controller_state.tcp_pose
            print(f"[Perception] Robot TCP pose: {robot_pose}")

            # Move the arm in the environment to explore and find board
            self.set_pose_target(
                move_robot=move_robot,
                pose=vision_utils.random_pose_increment(
                    robot_pose, 
                    position_scale=0.02, 
                    orientation_scale=None,
                    ),
            )
            self.sleep_for(0.25)
        
        
        ######################################### --> Perception

        self.get_logger().info(f"[Perception] Best pose found with quality above threshold, proceeding with insertion.")

        ######################################### <-- CheatCode policy
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
        ######################################### --> CheatCode policy

        self.get_logger().info("VisionBase.insert_cable() exiting...")
        return True
