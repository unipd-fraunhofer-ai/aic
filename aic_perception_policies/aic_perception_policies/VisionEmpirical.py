"""Vision-based empirical insertion policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Sequence


import numpy as np
from scipy.spatial.transform import Rotation
from transforms3d._gohlketransforms import quaternion_slerp

from aic_control_interfaces.msg import (
    JointMotionUpdate,
    MotionUpdate,
    TrajectoryGenerationMode,
)
from aic_model.policy import (
    GetObservationCallback,
    MoveRobotCallback,
    Policy,
    SendFeedbackCallback,
)
from aic_model_interfaces.msg import Observation
from aic_task_interfaces.msg import Task
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
    Wrench,
    WrenchStamped,
)
from rclpy.duration import Duration
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster, TransformException

# import cv2
# from cv_bridge import CvBridge

# from aic_perception.yolo_wrapper import YoloWrapper, plot_masks
# from aic_perception.utils.pose_estimator import PoseEstimator
import aic_perception_policies.vision_utils as vision_utils


#<--- CHANGE THIS TO YOUR LOCAL PATH 
POLICY_DATA_PATH = "aic_perception/data"
#------------------------------------------------------------


@dataclass(frozen=True)
class FixedTipTransform:
    """Fixed gripper TCP to connector-tip transform."""

    translation: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]

    def as_matrix(self) -> np.ndarray:
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(
            self.quaternion_xyzw,
        ).as_matrix()
        transform[:3, 3] = self.translation
        return transform

    def as_ros_transform(self) -> Transform:
        return vision_utils.matrix_to_transform(self.as_matrix())


class InsertCableState(Enum):
    """Empirical insertion state machine."""

    INIT = auto()
    WAIT_FOR_TFS = auto()
    LOOKUP_FIXED_TFS = auto()
    MOVE_ABOVE_PORT = auto()
    DESCEND_AND_INSERT = auto()
    STABILIZE_XY = auto()
    FIX_YAW = auto()
    UNTILT_INSERTED_CABLE = auto()
    REDESCEND = auto()
    STABILIZE = auto()
    DONE = auto()
    FAILED = auto()


class VisionEmpirical(Policy):
    """Detect the target port with vision, then run empirical insertion."""

    # Fixed measured gripper/tcp -> plug-tip transforms.  Do not use
    # sample_config.yaml cable poses here: those are cable spawn poses, not
    # TCP-to-tip transforms.
    _FIXED_TIP_TRANSFORMS = {
        "sfp": FixedTipTransform(
            translation=(-0.000000886478297101867, -0.020687488511920038,
                         0.054118867685087224),
            quaternion_xyzw=(-0.17785966749625665, -0.00503708733179058,
                             0.027383843138112103, -0.983661891514159),
        ),
        "sc": FixedTipTransform(
            translation=(-0.0005699007703731662, -0.01096440443128599,
                         0.009640708469970782),
            quaternion_xyzw=(-0.2298133984379278, 0.22655156773159751,
                             -0.6627472977643364, -0.6757412205316223),
        ),
    }
    _CONTROLLED_FRAME_OFFSET_TIP = np.array([-0.00, 0.006, 0.0])
    _FIXED_PORT_Z_BY_CONNECTOR = {
        "sfp": 0.133476,
        "sc": 0.0165,
    }
    _STARTING_JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    _STARTING_JOINT_POSITIONS = (
        -0.1597,
        -1.3542,
        -1.6648,
        -1.6933,
        1.5710,
        1.4110,
    )

    def import_dependencies(self):
        global cv2, CvBridge, YoloWrapper, plot_masks, PoseEstimator, vision_utils

        import cv2
        from cv_bridge import CvBridge
        from aic_perception.yolo_wrapper import YoloWrapper, plot_masks
        from aic_perception.utils.pose_estimator import PoseEstimator
        import aic_perception_policies.vision_utils as vision_utils


    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.get_logger().info("VisionEmpirical.__init__()")

        self._tip_x_error_integrator = 0.0
        self._tip_y_error_integrator = 0.0
        self._max_integrator_windup = 0.05
        self._task = None

        self.policy_data_path = Path(POLICY_DATA_PATH)
        self.get_logger().info(
            f"Using perception data path: {self.policy_data_path}"
        )
        self.yolo_checkpoint_path = (
            self.policy_data_path / "weights_istances/yolo26_segment.pt"
        )
        self.templates_dir = self.policy_data_path / "templates"
        self.models_dir = self.policy_data_path / "ic/models"
        self.nic_card_ports_filename = (
            self.policy_data_path / "nic_card_merged_transforms.json"
        )
        self.sc_port_filename = (
            self.policy_data_path / "sc_port_visual_pulito.json"
        )

        self.camera_names = ["center_camera", "left_camera", "right_camera"]
        self.camera_frames = {
            name: f"{name}/optical" for name in self.camera_names
        }
        self.nic_card_port_frames = vision_utils.load_model_frames(
            self.nic_card_ports_filename,
        )
        self.sc_port_frames = vision_utils.load_model_frames(
            self.sc_port_filename,
        )

        self.pose_estimator = None
        self.yolo = None
        
        from cv_bridge import CvBridge
        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self._parent_node)

        self.debug_mask = True
        self.enforce_port_zero_roll_pitch = True
        self.correct_port_roll_ambiguity = True
        self.mask_image_pub = {}
        for name in self.camera_names:
            self.mask_image_pub[name] = self._parent_node.create_publisher(
                Image,
                f"/pose_estimator/{name}_debug_mask_image",
                10,
            )

        self._wrist_to_tip = np.eye(4)
        self.print_insertion_loop_debug = False
        self.debug_tip_pose_error = False
        self._tip_pose_debug_step = 0
        self._tip_pose_debug_period = 20
        self._plug_wrench_pub = parent_node.create_publisher(
            WrenchStamped,
            "/plug_wrench",
            10,
        )
        self._plug_pose_pub = parent_node.create_publisher(
            PoseStamped,
            "/plug_pose",
            10,
        )
        self._plug_reference_pose_pub = parent_node.create_publisher(
            PoseStamped,
            "/plug_reference_pose",
            10,
        )

    def _connector_key(self, task: Task | None = None) -> str:
        """Return the connector key used by the fixed transform table."""
        task = task or self._task
        if task is None:
            raise ValueError("Task is not initialized.")

        plug_type = str(task.plug_type).strip().lower()
        if plug_type in self._FIXED_TIP_TRANSFORMS:
            return plug_type

        plug_name = str(task.plug_name).strip().lower()
        for connector_key in self._FIXED_TIP_TRANSFORMS:
            if plug_name.startswith(connector_key):
                return connector_key

        raise KeyError(
            "No fixed gripper-to-tip transform configured for "
            f"plug_type={task.plug_type!r}, plug_name={task.plug_name!r}."
        )

    def _fixed_gripper_tip_transform(self, task: Task | None = None) -> Transform:
        connector_key = self._connector_key(task)
        return self._FIXED_TIP_TRANSFORMS[connector_key].as_ros_transform()

    def _fixed_gripper_tip_matrix(self, task: Task | None = None) -> np.ndarray:
        connector_key = self._connector_key(task)
        return self._FIXED_TIP_TRANSFORMS[connector_key].as_matrix()

    def _fixed_port_z(self, task: Task) -> float:
        port_type = str(task.port_type).strip().lower()
        if port_type in self._FIXED_PORT_Z_BY_CONNECTOR:
            return self._FIXED_PORT_Z_BY_CONNECTOR[port_type]

        port_name = str(task.port_name).strip().lower()
        for connector_key, port_z in self._FIXED_PORT_Z_BY_CONNECTOR.items():
            if port_name.startswith(connector_key):
                return port_z

        raise KeyError(
            "No fixed port height configured for "
            f"port_type={task.port_type!r}, port_name={task.port_name!r}."
        )

    def get_current_plug_transform(
        self,
        gripper_tf_stamped: TransformStamped | None = None,
    ) -> TransformStamped:
        """Compute plug-tip pose from the robot TCP and fixed tip offset.

        Ground-truth cable-tip TFs are intentionally not used for control here.
        """
        if gripper_tf_stamped is None:
            gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                "gripper/tcp",
                Time(),
            )

        base_gripper = vision_utils.transform_to_matrix(
            gripper_tf_stamped.transform,
        )
        base_tip = base_gripper @ self._fixed_gripper_tip_matrix()

        plug_tf_stamped = TransformStamped()
        plug_tf_stamped.header.stamp = gripper_tf_stamped.header.stamp
        plug_tf_stamped.header.frame_id = gripper_tf_stamped.header.frame_id
        plug_tf_stamped.child_frame_id = f"{self._task.plug_name}_fixed_tip"
        plug_tf_stamped.transform = vision_utils.matrix_to_transform(base_tip)
        return plug_tf_stamped

    def _wait_for_tf(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float = 10.0,
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
                        "Waiting for transform "
                        f"'{source_frame}' -> '{target_frame}'..."
                    )
                attempt += 1
                self.sleep_for(0.1)
        self.get_logger().error(
            f"Transform '{source_frame}' not available after {timeout_sec}s"
        )
        return False

    def get_camera_observations(
        self,
        obs_msg: Observation,
        world_frame: str = "world",
    ):
        cameras = {}
        for name, frame in self.camera_frames.items():
            image_msg = getattr(obs_msg, name.replace("camera", "image"))
            camera_info_msg = getattr(
                obs_msg,
                name.replace("camera", "camera_info"),
            )
            if image_msg is None or camera_info_msg is None:
                self.get_logger().error(f"Missing data for camera '{name}'")
                continue

            intrinsics = vision_utils.load_intrinsics(camera_info_msg)
            extrinsics = self._parent_node._tf_buffer.lookup_transform(
                frame,
                world_frame,
                Time(),
            )
            world_camera = vision_utils.transform_to_matrix(
                extrinsics.transform,
            )
            world_camera[:3, 3] = world_camera[:3, 3] * 1000.0

            cameras[name] = {
                "image": image_msg,
                "intrinsics": intrinsics,
                "extrinsics": {
                    "R_w2c": world_camera[:3, :3].tolist(),
                    "t_w2c": world_camera[:3, 3].tolist(),
                },
            }
        return cameras

    def compute_segmentation_masks(
        self,
        camera_name: str,
        image: Image,
        target_name=None,
    ):
        if self.yolo is None:
            from aic_perception.yolo_wrapper import YoloWrapper
            self.yolo = YoloWrapper(self.yolo_checkpoint_path)
            self.get_logger().info("Loaded YoloWrapper")

        cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding="bgr8")
        raw_results = self.yolo.predict(cv_image, keep_best=True)

        masks = {}
        names = {}
        confs = {}
        for result in raw_results:
            class_name = result["class_name"]
            confidence = result["confidence"]
            class_id = result["class_id"]
            self.get_logger().info(
                f"[{camera_name}] YOLO result: "
                f"{class_id} {class_name} ({confidence:.2f})"
            )

            if target_name is not None and class_name != target_name:
                self.get_logger().warning(
                    "Skipping YOLO result with class_name "
                    f"{class_name} since it does not match target_name "
                    f"{target_name}"
                )
                continue

            if class_name is None:
                self.get_logger().warning(
                    "Skipping YOLO result with no class name, "
                    f"class_id {class_id}, confidence {confidence:.2f}"
                )
                continue

            object_model_id = vision_utils.get_object_model_id(class_name)
            if object_model_id not in vision_utils.CLASS_NAMES_MAP.values():
                self.get_logger().warning(
                    "Unknown object_model_id "
                    f"{object_model_id}, class_name {class_name} "
                    "in YOLO results"
                )
                continue

            if "mask" in result:
                masks.setdefault(object_model_id, []).append(result["mask"])
                names.setdefault(object_model_id, []).append(class_name)
                confs.setdefault(object_model_id, []).append(confidence)

        import cv2
        color = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        color = color.astype(np.float32) / 255.0

        if self.debug_mask:
            from aic_perception.yolo_wrapper import plot_masks
            image_mask = plot_masks(cv_image, raw_results)
            image_mask_msg = self.bridge.cv2_to_imgmsg(
                image_mask,
                encoding="bgr8",
            )
            image_mask_msg.header = image.header
            self.mask_image_pub[camera_name].publish(image_mask_msg)

        return {
            "color": color,
            "masks": masks,
            "names": names,
            "confs": confs,
        }

    def compute_poses(
        self,
        cameras,
        camera_inputs,
        scale_factor=1000.0,
        mask_quality_threshold=0.5,
        pose_quality_threshold=0.5,
    ):
        if self.pose_estimator is None:
            from aic_perception.utils.pose_estimator import PoseEstimator
            self.pose_estimator = PoseEstimator(
                cameras=cameras,
                templates_dir=self.templates_dir,
                models_dir=self.models_dir,
            )
            self.get_logger().info("Loaded PoseEstimator")

        pose_detections = self.pose_estimator.estimate_pose(camera_inputs)
        best_camera_name = vision_utils.get_best_camera(
            camera_inputs,
            quality_threshold=mask_quality_threshold,
        )
        best_pose = vision_utils.get_best_pose(
            pose_detections,
            best_camera_name=best_camera_name,
            quality_threshold=pose_quality_threshold,
        )

        if best_pose is not None:
            best_pose["t_m2c"] = best_pose["t_m2c"] / scale_factor
            best_pose["T_m2c"] = vision_utils.matrix_from_Rt(
                R=Rotation.from_matrix(best_pose["R_m2c"]).as_matrix(),
                t=np.array(best_pose["t_m2c"]),
            )
            best_pose["camera_name"] = best_camera_name

        return best_pose

    def get_port_local_transform(
        self,
        target_module_name,
        port_name: str,
    ):
        if "nic_card_mount" in target_module_name:
            model_to_port = self.nic_card_port_frames[port_name]
        elif "sc_port" in target_module_name:
            model_to_port = self.sc_port_frames[port_name]
        else:
            self.get_logger().error(
                f"Unknown target module {target_module_name} in task, "
                "cannot retrieve model to port transform"
            )
            return None

        return vision_utils.matrix_from_Rt(
            R=Rotation.from_quat(model_to_port["q_ply"]).as_matrix(),
            t=np.array(model_to_port["t_ply"]),
        )

    def apply_port_orientation_prior(self, world_port):
        """Project the port orientation onto the known roll=pitch=0 manifold."""
        constrained = np.array(world_port, dtype=float, copy=True)
        yaw = np.arctan2(constrained[1, 0], constrained[0, 0])
        constrained[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        return constrained

    def apply_port_plug_roll_prior(self, port_transform: Transform) -> Transform:
        if (
            not self.enforce_port_zero_roll_pitch
            or not self.correct_port_roll_ambiguity
        ):
            return port_transform

        try:
            plug_tf_stamped = self.get_current_plug_transform()
        except (TransformException, KeyError, ValueError) as ex:
            self.get_logger().warning(
                "Could not check port/plug roll alignment, keeping detected "
                f"port orientation: {ex}"
            )
            return port_transform

        world_port = vision_utils.transform_to_matrix(port_transform)
        world_plug = vision_utils.transform_to_matrix(plug_tf_stamped)
        world_port_rotation = world_port[:3, :3]
        world_plug_rotation = world_plug[:3, :3]
        raw_port_to_plug_rpy_deg = Rotation.from_matrix(
            world_port_rotation.T @ world_plug_rotation,
        ).as_euler("xyz", degrees=True)

        roll_candidates = (180.0,)
        candidate_results = []
        for roll_offset_deg in roll_candidates:
            candidate = np.array(world_port, dtype=float, copy=True)
            candidate[:3, :3] = (
                world_port_rotation
                @ Rotation.from_euler(
                    "x",
                    roll_offset_deg,
                    degrees=True,
                ).as_matrix()
            )
            rotation_to_candidate_deg = np.degrees(
                Rotation.from_matrix(
                    candidate[:3, :3] @ world_plug_rotation.T,
                ).magnitude()
            )
            candidate_results.append(
                (rotation_to_candidate_deg, roll_offset_deg, candidate),
            )

        rotation_to_target_deg, roll_offset_deg, world_port_aligned = min(
            candidate_results,
            key=lambda result: result[0],
        )
        candidate_summary = ", ".join(
            f"{candidate_roll:.0f}deg={candidate_angle:.2f}deg"
            for candidate_angle, candidate_roll, _ in candidate_results
        )

        self.get_logger().info(
            "[Perception] Port/plug roll alignment: "
            f"raw droll={raw_port_to_plug_rpy_deg[0]:.2f} deg, "
            f"dpitch={raw_port_to_plug_rpy_deg[1]:.2f} deg, "
            f"dyaw={raw_port_to_plug_rpy_deg[2]:.2f} deg; "
            f"selected local roll offset={roll_offset_deg:.0f} deg "
            f"(rotation-to-plug {rotation_to_target_deg:.2f} deg; "
            f"candidates: {candidate_summary})"
        )
        return vision_utils.matrix_to_transform(world_port_aligned)

    def compute_port_transform(
        self,
        T_camera_model,
        T_world_camera,
        T_model_to_port=None,
    ):
        world_model = T_world_camera @ T_camera_model
        if T_model_to_port is not None:
            world_model = world_model @ T_model_to_port

        if self.enforce_port_zero_roll_pitch:
            world_model = self.apply_port_orientation_prior(world_model)

        return vision_utils.matrix_to_transform(world_model)

    def log_port_detection_error(
        self,
        task: Task,
        detected_port_transform: Transform,
        camera_name: str | None = None,
    ) -> None:
        ground_truth_port_frame = (
            f"task_board/{task.target_module_name}/{task.port_name}_link"
        )
        try:
            ground_truth_port_tf = self._parent_node._tf_buffer.lookup_transform(
                self.world_frame,
                ground_truth_port_frame,
                Time(),
            )
        except TransformException:
            return

        world_detected = vision_utils.transform_to_matrix(
            detected_port_transform,
        )
        world_ground_truth = vision_utils.transform_to_matrix(
            ground_truth_port_tf,
        )

        position_error = world_detected[:3, 3] - world_ground_truth[:3, 3]
        position_error_mm = 1000.0 * position_error
        position_error_norm_mm = 1000.0 * np.linalg.norm(position_error)

        rotation_error = (
            world_detected[:3, :3] @ world_ground_truth[:3, :3].T
        )
        rotation_error_rpy_deg = Rotation.from_matrix(
            rotation_error,
        ).as_euler("xyz", degrees=True)
        angle_error_deg = np.degrees(
            Rotation.from_matrix(rotation_error).magnitude(),
        )

        self.get_logger().info(
            "\n[Perception] Port detection error vs ground truth "
            f"{ground_truth_port_frame}: "
            f"dx={position_error_mm[0]:.1f} mm, "
            f"dy={position_error_mm[1]:.1f} mm, "
            f"dz={position_error_mm[2]:.1f} mm, "
            f"|dpos|={position_error_norm_mm:.1f} mm, "
            f"dangle={angle_error_deg:.2f} deg, "
            f"droll={rotation_error_rpy_deg[0]:.2f} deg, "
            f"dpitch={rotation_error_rpy_deg[1]:.2f} deg, "
            f"dyaw={rotation_error_rpy_deg[2]:.2f} deg"
        )

        self.append_port_detection_error_log(
            task=task,
            camera_name=camera_name,
            detected_port_transform=detected_port_transform,
            ground_truth_port_tf=ground_truth_port_tf,
            position_error_mm=position_error_mm,
            position_error_norm_mm=position_error_norm_mm,
            angle_error_deg=angle_error_deg,
            rotation_error_rpy_deg=rotation_error_rpy_deg,
        )

    def _perception_error_log_path(self, task: Task) -> Path:
        task_id = str(task.id).strip() or "unknown_task"
        safe_task_id = "".join(
            char if char.isalnum() or char in ("-", "_") else "_"
            for char in task_id
        )
        log_dir = Path.home() / "aic_results"
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{safe_task_id}.txt"

    @staticmethod
    def _pose_components(transform: Transform | TransformStamped) -> list[float]:
        matrix = vision_utils.transform_to_matrix(transform)
        translation = matrix[:3, 3]
        quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        rpy_deg = Rotation.from_matrix(matrix[:3, :3]).as_euler(
            "xyz",
            degrees=True,
        )
        return [
            *translation.tolist(),
            *quaternion.tolist(),
            *rpy_deg.tolist(),
        ]

    def append_port_detection_error_log(
        self,
        task: Task,
        camera_name: str | None,
        detected_port_transform: Transform,
        ground_truth_port_tf: TransformStamped,
        position_error_mm: np.ndarray,
        position_error_norm_mm: float,
        angle_error_deg: float,
        rotation_error_rpy_deg: np.ndarray,
    ) -> None:
        try:
            board_tf = self._parent_node._tf_buffer.lookup_transform(
                self.world_frame,
                "task_board",
                Time(),
            )
            board_frame = "task_board"
            board_pose = self._pose_components(board_tf)
        except TransformException as ex:
            self.get_logger().warning(
                f"Could not log task board pose in {self.world_frame}: {ex}"
            )
            board_frame = "task_board_unavailable"
            board_pose = [float("nan")] * 10

        header = [
            "timestamp_ns",
            "task_id",
            "camera_name",
            "world_frame",
            "target_module_name",
            "port_name",
            "detected_x_m",
            "detected_y_m",
            "detected_z_m",
            "detected_qx",
            "detected_qy",
            "detected_qz",
            "detected_qw",
            "detected_roll_deg",
            "detected_pitch_deg",
            "detected_yaw_deg",
            "gt_x_m",
            "gt_y_m",
            "gt_z_m",
            "gt_qx",
            "gt_qy",
            "gt_qz",
            "gt_qw",
            "gt_roll_deg",
            "gt_pitch_deg",
            "gt_yaw_deg",
            "error_x_mm",
            "error_y_mm",
            "error_z_mm",
            "error_norm_mm",
            "error_angle_deg",
            "error_roll_deg",
            "error_pitch_deg",
            "error_yaw_deg",
            "board_frame",
            "board_x_m",
            "board_y_m",
            "board_z_m",
            "board_qx",
            "board_qy",
            "board_qz",
            "board_qw",
            "board_roll_deg",
            "board_pitch_deg",
            "board_yaw_deg",
        ]

        row = [
            str(self.time_now().nanoseconds),
            str(task.id),
            camera_name or "",
            self.world_frame,
            str(task.target_module_name),
            str(task.port_name),
            *self._pose_components(detected_port_transform),
            *self._pose_components(ground_truth_port_tf),
            *position_error_mm.tolist(),
            float(position_error_norm_mm),
            float(angle_error_deg),
            *rotation_error_rpy_deg.tolist(),
            board_frame,
            *board_pose,
        ]

        try:
            log_path = self._perception_error_log_path(task)
            file_needs_header = (
                not log_path.exists()
                or log_path.stat().st_size == 0
            )
            with log_path.open("a", encoding="utf-8") as log_file:
                if file_needs_header:
                    log_file.write(",".join(header) + "\n")
                log_file.write(
                    ",".join(
                        self._format_log_value(value)
                        for value in row
                    )
                )
                log_file.write("\n")
        except OSError as ex:
            self.get_logger().warning(
                f"Failed to append perception error log: {ex}"
            )

    @staticmethod
    def _format_log_value(value) -> str:
        if isinstance(value, float):
            return f"{value:.9g}"
        return str(value)

    def publish_debug_tf(
        self,
        transform: Transform,
        frame_id: str,
        child_frame_id: str,
    ) -> None:
        tf_stamped = TransformStamped()
        tf_stamped.header.stamp = self.time_now().to_msg()
        tf_stamped.header.frame_id = frame_id
        tf_stamped.child_frame_id = child_frame_id
        tf_stamped.transform = transform
        self.tf_broadcaster.sendTransform(tf_stamped)

    def _detect_port_transform(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> Transform | None:
        """Run the vision perception stack until a port pose is found."""
        self.world_frame = "base_link"
        for frame in self.camera_frames.values():
            if not self._wait_for_tf(self.world_frame, frame):
                return None

        while True:
            observation = get_observation()
            if observation is None:
                self.get_logger().warning("No observation available yet.")
                self.sleep_for(0.1)
                continue

            cameras = self.get_camera_observations(
                observation,
                world_frame=self.world_frame,
            )
            self.get_logger().info(
                "[Perception] Camera observations obtained for cameras: "
                f"{list(cameras.keys())}"
            )

            camera_inputs = {}
            for name, camera in cameras.items():
                camera_inputs[name] = self.compute_segmentation_masks(
                    camera_name=name,
                    image=camera["image"],
                    target_name=task.target_module_name,
                )

            detected_pose = self.compute_poses(
                cameras,
                camera_inputs,
                mask_quality_threshold=0.5,
                pose_quality_threshold=0.5,
            )
            if detected_pose is not None:
                return self._port_transform_from_detection(task, detected_pose)

            self.get_logger().info(
                "[Perception] No reliable port pose yet; exploring."
            )
            self.set_pose_target(
                move_robot=move_robot,
                pose=vision_utils.random_pose_increment(
                    observation.controller_state.tcp_pose,
                    position_scale=0.02,
                    orientation_scale=None,
                ),
            )
            self.sleep_for(0.25)

    def _port_transform_from_detection(
        self,
        task: Task,
        detected_pose,
    ) -> Transform | None:
        camera_name = detected_pose["camera_name"]
        camera_frame = self.camera_frames[camera_name]
        world_camera_tf = self._parent_node._tf_buffer.lookup_transform(
            self.world_frame,
            camera_frame,
            Time(),
        )
        world_camera = vision_utils.transform_to_matrix(
            world_camera_tf.transform,
        )
        model_to_port = self.get_port_local_transform(
            target_module_name=task.target_module_name,
            port_name=f"{task.port_name}_link",
        )
        if model_to_port is None:
            return None

        port_transform = self.compute_port_transform(
            T_camera_model=detected_pose["T_m2c"],
            T_world_camera=world_camera,
            T_model_to_port=model_to_port,
        )
        port_transform = self.apply_port_plug_roll_prior(port_transform)
        port_transform.translation.z = self._fixed_port_z(task)

        self.publish_debug_tf(
            port_transform,
            frame_id=self.world_frame,
            child_frame_id=f"{camera_name}_{task.port_name}_link",
        )
        self.log_port_detection_error(
            task,
            port_transform,
            camera_name=camera_name,
        )
        return port_transform

    def _compute_wrist_to_tip_matrix(self, task: Task) -> np.ndarray:
        """Compose wrist->TCP with the fixed TCP->tip transform."""
        wrist_gripper_tf = self._parent_node._tf_buffer.lookup_transform(
            "ati/tool_link",
            "gripper/tcp",
            Time(),
        )
        wrist_gripper = vision_utils.transform_to_matrix(
            wrist_gripper_tf.transform,
        )
        return wrist_gripper @ self._fixed_gripper_tip_matrix(task)

    def _observed_starting_joint_positions(
        self,
        observation: Observation | None,
    ) -> np.ndarray | None:
        if observation is None:
            return None

        joint_states = observation.joint_states
        target_len = len(self._STARTING_JOINT_POSITIONS)
        if len(joint_states.position) < target_len:
            return None

        if joint_states.name:
            positions_by_name = dict(zip(joint_states.name, joint_states.position))
            if all(name in positions_by_name for name in self._STARTING_JOINT_NAMES):
                return np.array(
                    [
                        positions_by_name[name]
                        for name in self._STARTING_JOINT_NAMES
                    ],
                    dtype=float,
                )

        return np.array(joint_states.position[:target_len], dtype=float)

    def _is_at_starting_pose(
        self,
        observation: Observation | None,
        tolerance: float = 0.03,
    ) -> bool:
        current_positions = self._observed_starting_joint_positions(observation)
        if current_positions is None:
            return False

        target_positions = np.array(self._STARTING_JOINT_POSITIONS, dtype=float)
        joint_errors = (
            current_positions - target_positions + np.pi
        ) % (2.0 * np.pi) - np.pi
        return bool(np.all(np.abs(joint_errors) <= tolerance))

    def _ensure_starting_pose(
        self,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
    ) -> bool:
        if self._is_at_starting_pose(get_observation()):
            return True

        joint_motion_update = JointMotionUpdate(
            target_stiffness=[200.0, 200.0, 200.0, 50.0, 50.0, 50.0],
            target_damping=[40.0, 40.0, 40.0, 15.0, 15.0, 15.0],
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        joint_motion_update.target_state.positions = list(
            self._STARTING_JOINT_POSITIONS,
        )

        for _ in range(40):
            try:
                if move_robot(joint_motion_update=joint_motion_update) is False:
                    return False
            except Exception as ex:  # noqa: BLE001 - policy should log failures.
                self.get_logger().info(f"move_robot exception: {ex}")
                return False

            self.sleep_for(0.1)
            if self._is_at_starting_pose(get_observation()):
                return True

        self.get_logger().error("Arm did not reach the starting pose.")
        return False

    def publish_tip_pose(self) -> PoseStamped:
        """Publish and return the computed plug-tip pose in base_link."""
        plug_tf_stamped = self.get_current_plug_transform()
        plug_pose = PoseStamped(
            header=Header(
                frame_id="base_link",
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=Pose(
                position=Point(
                    x=plug_tf_stamped.transform.translation.x,
                    y=plug_tf_stamped.transform.translation.y,
                    z=plug_tf_stamped.transform.translation.z,
                ),
                orientation=Quaternion(
                    x=plug_tf_stamped.transform.rotation.x,
                    y=plug_tf_stamped.transform.rotation.y,
                    z=plug_tf_stamped.transform.rotation.z,
                    w=plug_tf_stamped.transform.rotation.w,
                ),
            ),
        )
        self._plug_pose_pub.publish(plug_pose)
        return plug_pose

    def log_tip_pose_error_against_ground_truth(self, force: bool = False) -> None:
        """Compare fixed-offset tip pose with ground-truth TF when available."""
        # if not self.print_insertion_loop_debug or not self.debug_tip_pose_error:
        #     return
        
        if not self.debug_tip_pose_error:
            return

        self._tip_pose_debug_step += 1
        if (
            not force
            and self._tip_pose_debug_step % self._tip_pose_debug_period != 0
        ):
            return

        ground_truth_frame = f"{self._task.cable_name}/{self._task.plug_name}_link"
        try:
            computed_tip_tf = self.get_current_plug_transform()
            ground_truth_tip_tf = self._parent_node._tf_buffer.lookup_transform(
                "base_link",
                ground_truth_frame,
                Time(),
            )
        except TransformException:
            return

        computed_tip = vision_utils.transform_to_matrix(computed_tip_tf)
        ground_truth_tip = vision_utils.transform_to_matrix(ground_truth_tip_tf)

        position_error = computed_tip[:3, 3] - ground_truth_tip[:3, 3]
        position_error_mm = 1000.0 * position_error
        position_error_norm_mm = 1000.0 * np.linalg.norm(position_error)

        rotation_error = computed_tip[:3, :3] @ ground_truth_tip[:3, :3].T
        rotation_error_rpy_deg = Rotation.from_matrix(rotation_error).as_euler(
            "xyz",
            degrees=True,
        )
        angle_error_deg = np.degrees(
            Rotation.from_matrix(rotation_error).magnitude()
        )

        self.get_logger().info(
            "[Debug] Fixed tip pose error vs ground truth "
            f"{ground_truth_frame}: "
            f"dx={position_error_mm[0]:.1f} mm, "
            f"dy={position_error_mm[1]:.1f} mm, "
            f"dz={position_error_mm[2]:.1f} mm, "
            f"|dpos|={position_error_norm_mm:.1f} mm, "
            f"dangle={angle_error_deg:.2f} deg, "
            f"droll={rotation_error_rpy_deg[0]:.2f} deg, "
            f"dpitch={rotation_error_rpy_deg[1]:.2f} deg, "
            f"dyaw={rotation_error_rpy_deg[2]:.2f} deg"
        )

    def publish_tip_wrench(self, observation: Observation) -> Wrench:
        """Publish and return the wrist wrench expressed at the plug tip."""
        corrected_wrench = _subtract_wrench_offset(
            observation.wrist_wrench,
            observation.controller_state.fts_tare_offset,
        )
        plug_wrench = _wrench_at_tip_from_wrist(
            corrected_wrench,
            self._wrist_to_tip,
        )
        self._plug_wrench_pub.publish(
            WrenchStamped(
                header=Header(
                    frame_id=f"{self._task.plug_name}_fixed_tip",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                wrench=plug_wrench,
            )
        )
        return plug_wrench

    def calc_gripper_pose(
        self,
        port_transform: Transform,
        gripper_tip_transform: Transform,
        slerp_fraction: float = 1.0,
        position_fraction: float = 1.0,
        z_offset: float = 0.1,
        x_offset: float = 0.0,
        reset_xy_integrator: bool = False,
        tilt_roll: float = 0.0,
        tilt_pitch: float = 0.0,
        tilt_yaw: float = 0.0,
        rotate_tcp_in_place: bool = False,
        controlled_frame_offset_tip: np.ndarray | None = None,
    ) -> Pose:
        """Compute the TCP pose that brings the plug tip to the port."""
        if controlled_frame_offset_tip is None:
            controlled_frame_offset_tip = np.zeros(3)

        gripper_tip = vision_utils.transform_to_matrix(gripper_tip_transform)
        tip_controlled = np.eye(4)
        tip_controlled[:3, 3] = controlled_frame_offset_tip

        gripper_tf_stamped = self._parent_node._tf_buffer.lookup_transform(
            "base_link",
            "gripper/tcp",
            Time(),
        )
        base_gripper_current = vision_utils.transform_to_matrix(
            gripper_tf_stamped.transform,
        )
        base_tip_current = vision_utils.transform_to_matrix(
            self.get_current_plug_transform(gripper_tf_stamped),
        )
        base_controlled_current = base_tip_current @ tip_controlled

        base_port = vision_utils.transform_to_matrix(port_transform)
        base_port_rotation = base_port[:3, :3]
        base_port_position = base_port[:3, 3]

        tilt_rotation = _xyz_rpy_to_matrix(
            0.0,
            0.0,
            0.0,
            tilt_roll,
            tilt_pitch,
            tilt_yaw,
        )[:3, :3]
        base_controlled_desired_rotation = base_port_rotation @ tilt_rotation

        tip_x_error = base_port_position[0] - base_controlled_current[0, 3]
        tip_y_error = base_port_position[1] - base_controlled_current[1, 3]
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

        integrator_gain = 0.15
        base_controlled_desired_position = (
            base_port_position + np.array([x_offset, 0.0, z_offset])
        )
        base_controlled_desired_position[0] += (
            integrator_gain * self._tip_x_error_integrator
        )
        base_controlled_desired_position[1] += (
            integrator_gain * self._tip_y_error_integrator
        )

        base_controlled_desired = np.eye(4)
        base_controlled_desired[:3, :3] = base_controlled_desired_rotation
        base_controlled_desired[:3, 3] = base_controlled_desired_position

        self._plug_reference_pose_pub.publish(
            PoseStamped(
                header=Header(
                    frame_id="base_link",
                    stamp=self._parent_node.get_clock().now().to_msg(),
                ),
                pose=_matrix_to_pose(base_controlled_desired),
            )
        )

        base_tip_desired = base_controlled_desired @ np.linalg.inv(
            tip_controlled,
        )
        base_gripper_target = base_tip_desired @ np.linalg.inv(gripper_tip)

        current_position = base_gripper_current[:3, 3]
        target_position = base_gripper_target[:3, 3]
        if rotate_tcp_in_place:
            blended_position = current_position
        else:
            blended_position = (
                position_fraction * target_position
                + (1.0 - position_fraction) * current_position
            )

        current_quaternion = _transform_quaternion_wxyz(
            gripper_tf_stamped.transform,
        )
        target_quaternion = _matrix_quaternion_wxyz(
            base_gripper_target[:3, :3],
        )
        blended_quaternion = quaternion_slerp(
            current_quaternion,
            target_quaternion,
            slerp_fraction,
        )

        roll, pitch, yaw = Rotation.from_quat(
            _wxyz_to_xyzw(blended_quaternion),
        ).as_euler("xyz")
        if self.print_insertion_loop_debug:
            self.get_logger().info(
                "[CMD POSE] "
                f"x={blended_position[0]:.4f}, "
                f"y={blended_position[1]:.4f}, "
                f"z={blended_position[2]:.4f} | "
                f"roll={np.rad2deg(roll):.2f} deg, "
                f"pitch={np.rad2deg(pitch):.2f} deg, "
                f"yaw={np.rad2deg(yaw):.2f} deg"
            )

        return Pose(
            position=Point(
                x=float(blended_position[0]),
                y=float(blended_position[1]),
                z=float(blended_position[2]),
            ),
            orientation=Quaternion(
                w=float(blended_quaternion[0]),
                x=float(blended_quaternion[1]),
                y=float(blended_quaternion[2]),
                z=float(blended_quaternion[3]),
            ),
        )

    def _move_sc_to_vision_target(
        self,
        port_transform: Transform,
        gripper_tip_transform: Transform,
        move_robot: MoveRobotCallback,
    ) -> bool:
        """Run the CheatCode-style motion using the vision-derived port pose."""
        z_offset = 0.2
        cheatcode_stiffness = [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
        cheatcode_damping = [100.0, 100.0, 100.0, 40.0, 40.0, 40.0]
        cheatcode_wrench_feedback = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.get_logger().info(
            "SC plug detected; moving directly to the vision target."
        )

        for step in range(100):
            interpolation_fraction = step / 100.0
            try:
                pose = self.calc_gripper_pose(
                    port_transform,
                    gripper_tip_transform,
                    slerp_fraction=interpolation_fraction,
                    position_fraction=interpolation_fraction,
                    z_offset=z_offset,
                    reset_xy_integrator=True,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=cheatcode_stiffness,
                    damping=cheatcode_damping,
                    wrench_feedback_gains_at_tip=cheatcode_wrench_feedback,
                )
            except TransformException as ex:
                self.get_logger().warning(
                    "TF lookup failed during SC interpolation: "
                    f"{ex}"
                )
            self.sleep_for(0.05)

        while z_offset >= -0.015:
            z_offset -= 0.0005
            try:
                pose = self.calc_gripper_pose(
                    port_transform,
                    gripper_tip_transform,
                    z_offset=z_offset,
                )
                self.set_pose_target(
                    move_robot=move_robot,
                    pose=pose,
                    stiffness=cheatcode_stiffness,
                    damping=cheatcode_damping,
                    wrench_feedback_gains_at_tip=cheatcode_wrench_feedback,
                )
            except TransformException as ex:
                self.get_logger().warning(
                    f"TF lookup failed during SC descent: {ex}"
                )
            self.sleep_for(0.05)

        self.get_logger().info("Waiting for SC connector to stabilize.")
        self.sleep_for(5.0)
        return True

    def insert_cable(
        self,
        task: Task,
        get_observation: GetObservationCallback,
        move_robot: MoveRobotCallback,
        send_feedback: SendFeedbackCallback,
    ) -> bool:
        self.get_logger().info(f"VisionEmpirical.insert_cable() task: {task}")
        self._task = task

        state = InsertCableState.INIT
        port_transform = None
        gripper_tip_transform = None
        tip_initialized = False

        z_offset = 0.10
        z_force_threshold = -2.0
        roll_angle = np.deg2rad(15.0)
        pitch_angle = np.deg2rad(15.0)
        yaw_angle = 0.0
        untilt_step = np.deg2rad(0.5)

        move_above_step = 0
        stabilize_step = 0
        target_stabilize_steps = 50

        fix_yaw_direction = 1.0
        fix_yaw_attempt = 0
        fix_yaw_max_attempts = 2
        fix_yaw_best_z = None
        fix_yaw_no_improve_steps = 0
        fix_yaw_max_no_improve_steps = 100
        fix_yaw_step = np.deg2rad(0.2)
        fix_yaw_limit = np.deg2rad(10.0)
        fix_yaw_min_z_improvement = 0.001

        while state not in (InsertCableState.DONE, InsertCableState.FAILED):
            observation = get_observation()
            if observation is None:
                self.get_logger().warning("No observation available yet.")
                self.sleep_for(0.1)
                continue

            plug_pose = None
            plug_wrench = None
            if tip_initialized:
                plug_wrench = self.publish_tip_wrench(observation)
                plug_pose = self.publish_tip_pose()
                self.log_tip_pose_error_against_ground_truth()

            command_kwargs = {
                "stiffness": [200.0, 200.0, 200.0, 50.0, 50.0, 50.0],
                "wrench_feedback_gains_at_tip": [0.0] * 6,
                "slerp_fraction": 1.0,
                "position_fraction": 1.0,
                "z_offset": z_offset,
                "reset_xy_integrator": True,
                "tilt_roll": roll_angle,
                "tilt_pitch": pitch_angle,
                "tilt_yaw": yaw_angle,
            }
            should_send_command = False

            if state == InsertCableState.INIT:
                if not self._ensure_starting_pose(get_observation, move_robot):
                    state = InsertCableState.FAILED
                    continue
                self.sleep_for(1.0)
                port_transform = self._detect_port_transform(
                    task,
                    get_observation,
                    move_robot,
                )
                state = (
                    InsertCableState.WAIT_FOR_TFS
                    if port_transform is not None
                    else InsertCableState.FAILED
                )

            elif state == InsertCableState.WAIT_FOR_TFS:
                required_tfs = [
                    ("base_link", "gripper/tcp"),
                    ("ati/tool_link", "gripper/tcp"),
                ]
                for target_frame, source_frame in required_tfs:
                    if not self._wait_for_tf(target_frame, source_frame):
                        state = InsertCableState.FAILED
                        break
                else:
                    state = InsertCableState.LOOKUP_FIXED_TFS

            elif state == InsertCableState.LOOKUP_FIXED_TFS:
                try:
                    gripper_tip_transform = self._fixed_gripper_tip_transform(
                        task,
                    )
                    self._wrist_to_tip = self._compute_wrist_to_tip_matrix(
                        task,
                    )
                except (KeyError, TransformException, ValueError) as ex:
                    self.get_logger().error(
                        f"Fixed tip transform initialization failed: {ex}"
                    )
                    state = InsertCableState.FAILED
                    continue

                connector_key = self._connector_key(task)
                self.get_logger().info(
                    "Using fixed gripper/tcp -> plug-tip transform "
                    f"for {connector_key}."
                )
                if connector_key == "sc":
                    state = (
                        InsertCableState.DONE
                        if self._move_sc_to_vision_target(
                            port_transform,
                            gripper_tip_transform,
                            move_robot,
                        )
                        else InsertCableState.FAILED
                    )
                    continue

                move_above_step = 0
                tip_initialized = True
                state = InsertCableState.MOVE_ABOVE_PORT

            elif state == InsertCableState.MOVE_ABOVE_PORT:
                interpolation_fraction = move_above_step / 100.0
                command_kwargs.update(
                    stiffness=[100.0, 100.0, 100.0, 60.0, 60.0, 60.0],
                    slerp_fraction=interpolation_fraction,
                    position_fraction=interpolation_fraction,
                )
                should_send_command = True
                move_above_step += 1
                if move_above_step >= 100:
                    state = InsertCableState.DESCEND_AND_INSERT

            elif state == InsertCableState.DESCEND_AND_INSERT:
                if plug_wrench is None:
                    state = InsertCableState.FAILED
                    continue

                if plug_wrench.force.z < z_force_threshold:
                    self.get_logger().info(
                        f"Z contact detected: Fz={plug_wrench.force.z:.3f} N"
                    )
                    state = InsertCableState.STABILIZE_XY
                    continue

                if z_offset < -0.015:
                    self.get_logger().warning(
                        "Reached max insertion depth without detecting Z contact."
                    )
                    state = InsertCableState.STABILIZE
                    continue

                z_offset -= 0.001
                command_kwargs.update(
                    z_offset=z_offset,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                )
                should_send_command = True

            elif state == InsertCableState.STABILIZE_XY:
                if plug_pose is None:
                    state = InsertCableState.FAILED
                    continue

                if z_offset < -0.015:
                    self.get_logger().info("Insertion depth reached.")
                    port_transform.translation.x = plug_pose.pose.position.x
                    port_transform.translation.y = plug_pose.pose.position.y
                    if plug_pose.pose.position.z >= 0.1765:
                        self.get_logger().warning(
                            "Plug tip is not aligned with port; fixing yaw."
                        )
                        state = InsertCableState.FIX_YAW
                    else:
                        self.get_logger().info(
                            "Plug tip is aligned with port; reorienting."
                        )
                        state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                z_offset -= 0.001
                command_kwargs.update(
                    stiffness=[10.0, 10.0, 90.0, 5.0, 5.0, 5.0],
                    z_offset=z_offset,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                )
                should_send_command = True

            elif state == InsertCableState.FIX_YAW:
                if plug_pose is None:
                    state = InsertCableState.FAILED
                    continue

                current_z = plug_pose.pose.position.z
                if current_z < 0.1765:
                    self.get_logger().info("Yaw fixed.")
                    fix_yaw_best_z = None
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                if fix_yaw_best_z is None:
                    fix_yaw_best_z = current_z
                    fix_yaw_no_improve_steps = 0
                elif current_z < fix_yaw_best_z - fix_yaw_min_z_improvement:
                    fix_yaw_best_z = current_z
                    fix_yaw_no_improve_steps = 0
                else:
                    fix_yaw_no_improve_steps += 1

                yaw_angle += fix_yaw_direction * fix_yaw_step
                yaw_limit_reached = abs(yaw_angle) >= fix_yaw_limit
                no_improvement = (
                    fix_yaw_no_improve_steps >= fix_yaw_max_no_improve_steps
                )
                if yaw_limit_reached or no_improvement:
                    fix_yaw_attempt += 1
                    if fix_yaw_attempt < fix_yaw_max_attempts:
                        fix_yaw_direction *= -1.0
                        fix_yaw_best_z = None
                        fix_yaw_no_improve_steps = 0
                        continue

                    self.get_logger().error(
                        "Yaw adjustment failed in both directions."
                    )
                    state = InsertCableState.UNTILT_INSERTED_CABLE
                    continue

                command_kwargs.update(
                    stiffness=[90.0, 90.0, 90.0, 50.0, 50.0, 200.0],
                    z_offset=z_offset,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )
                should_send_command = True

            elif state == InsertCableState.UNTILT_INSERTED_CABLE:
                roll_done = abs(roll_angle) <= untilt_step
                pitch_done = abs(pitch_angle) <= untilt_step
                if not roll_done:
                    roll_angle -= np.sign(roll_angle) * untilt_step
                if not pitch_done:
                    pitch_angle -= np.sign(pitch_angle) * untilt_step

                if roll_done and pitch_done and z_offset >= 0.04:
                    state = InsertCableState.REDESCEND
                    continue

                z_offset += 0.001
                command_kwargs.update(
                    stiffness=[90.0, 90.0, 200.0, 50.0, 50.0, 200.0],
                    z_offset=z_offset,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )
                should_send_command = True

            elif state == InsertCableState.REDESCEND:
                z_offset -= 0.001
                command_kwargs.update(
                    stiffness=[90.0, 90.0, 90.0, 200.0, 200.0, 200.0],
                    z_offset=z_offset,
                    tilt_roll=roll_angle,
                    tilt_pitch=pitch_angle,
                    tilt_yaw=yaw_angle,
                )
                should_send_command = True
                if z_offset < -0.015:
                    self.get_logger().info("Redescend completed.")
                    state = InsertCableState.STABILIZE
                    continue

            elif state == InsertCableState.STABILIZE:
                stabilize_step += 1
                if stabilize_step == 1:
                    self.get_logger().info("Waiting for connector to stabilize.")
                if stabilize_step > target_stabilize_steps:
                    state = InsertCableState.DONE

            if should_send_command:
                if port_transform is None or gripper_tip_transform is None:
                    self.get_logger().error(
                        "Cannot command insertion before transforms are ready."
                    )
                    state = InsertCableState.FAILED
                    continue

                try:
                    pose = self.calc_gripper_pose(
                        port_transform,
                        gripper_tip_transform,
                        slerp_fraction=command_kwargs["slerp_fraction"],
                        position_fraction=command_kwargs["position_fraction"],
                        z_offset=command_kwargs["z_offset"],
                        reset_xy_integrator=(
                            command_kwargs["reset_xy_integrator"]
                        ),
                        tilt_roll=command_kwargs["tilt_roll"],
                        tilt_pitch=command_kwargs["tilt_pitch"],
                        tilt_yaw=command_kwargs["tilt_yaw"],
                        controlled_frame_offset_tip=(
                            self._CONTROLLED_FRAME_OFFSET_TIP
                        ),
                    )
                    self.set_pose_target(
                        move_robot=move_robot,
                        pose=pose,
                        stiffness=command_kwargs["stiffness"],
                        wrench_feedback_gains_at_tip=(
                            command_kwargs["wrench_feedback_gains_at_tip"]
                        ),
                    )
                except TransformException as ex:
                    self.get_logger().warning(
                        f"TF lookup failed in state {state.name}: {ex}"
                    )

            self.sleep_for(0.05)

        success = state == InsertCableState.DONE
        if success:
            self.get_logger().info("VisionEmpirical.insert_cable() completed.")
        else:
            self.get_logger().error("VisionEmpirical.insert_cable() failed.")
        return success

    def set_pose_target(
        self,
        move_robot: MoveRobotCallback,
        pose: Pose,
        frame_id: str = "base_link",
        stiffness: Sequence[float] | None = None,
        damping: Sequence[float] | None = None,
        wrench_feedback_gains_at_tip: Sequence[float] | None = None,
    ) -> None:
        """Send a Cartesian pose target with empirical controller gains."""
        if stiffness is None:
            stiffness = [90.0, 90.0, 90.0, 50.0, 50.0, 50.0]
        if damping is None:
            damping = [100.0, 100.0, 100.0, 40.0, 40.0, 40.0]
        if wrench_feedback_gains_at_tip is None:
            wrench_feedback_gains_at_tip = [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]

        motion_update = MotionUpdate(
            header=Header(
                frame_id=frame_id,
                stamp=self._parent_node.get_clock().now().to_msg(),
            ),
            pose=pose,
            target_stiffness=np.diag(stiffness).flatten(),
            target_damping=np.diag(damping).flatten(),
            feedforward_wrench_at_tip=Wrench(
                force=Vector3(x=0.0, y=0.0, z=0.0),
                torque=Vector3(x=0.0, y=0.0, z=0.0),
            ),
            wrench_feedback_gains_at_tip=list(wrench_feedback_gains_at_tip),
            trajectory_generation_mode=TrajectoryGenerationMode(
                mode=TrajectoryGenerationMode.MODE_POSITION,
            ),
        )
        try:
            move_robot(motion_update=motion_update)
        except Exception as ex:  # noqa: BLE001 - policy should log failures.
            self.get_logger().info(f"move_robot exception: {ex}")


def _xyz_rpy_to_matrix(
    x: float,
    y: float,
    z: float,
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler(
        "xyz",
        [roll, pitch, yaw],
    ).as_matrix()
    transform[:3, 3] = [x, y, z]
    return transform


def _matrix_to_pose(transform: np.ndarray) -> Pose:
    quaternion = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return Pose(
        position=Point(
            x=float(transform[0, 3]),
            y=float(transform[1, 3]),
            z=float(transform[2, 3]),
        ),
        orientation=Quaternion(
            x=float(quaternion[0]),
            y=float(quaternion[1]),
            z=float(quaternion[2]),
            w=float(quaternion[3]),
        ),
    )


def _matrix_quaternion_wxyz(rotation_matrix: np.ndarray) -> tuple[float, ...]:
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return (
        float(quaternion[3]),
        float(quaternion[0]),
        float(quaternion[1]),
        float(quaternion[2]),
    )


def _transform_quaternion_wxyz(transform: Transform) -> tuple[float, ...]:
    return (
        float(transform.rotation.w),
        float(transform.rotation.x),
        float(transform.rotation.y),
        float(transform.rotation.z),
    )


def _wxyz_to_xyzw(quaternion: Sequence[float]) -> tuple[float, ...]:
    return (
        float(quaternion[1]),
        float(quaternion[2]),
        float(quaternion[3]),
        float(quaternion[0]),
    )


def _subtract_wrench_offset(
    wrench: WrenchStamped,
    offset: WrenchStamped,
) -> WrenchStamped:
    return WrenchStamped(
        header=wrench.header,
        wrench=Wrench(
            force=Vector3(
                x=wrench.wrench.force.x - offset.wrench.force.x,
                y=wrench.wrench.force.y - offset.wrench.force.y,
                z=wrench.wrench.force.z - offset.wrench.force.z,
            ),
            torque=Vector3(
                x=wrench.wrench.torque.x - offset.wrench.torque.x,
                y=wrench.wrench.torque.y - offset.wrench.torque.y,
                z=wrench.wrench.torque.z - offset.wrench.torque.z,
            ),
        ),
    )


def _wrench_at_tip_from_wrist(
    wrist_wrench: WrenchStamped,
    wrist_to_tip: np.ndarray,
) -> Wrench:
    rotation_wrist_tip = wrist_to_tip[:3, :3]
    wrist_to_tip_translation = wrist_to_tip[:3, 3]

    force_wrist = np.array(
        [
            wrist_wrench.wrench.force.x,
            wrist_wrench.wrench.force.y,
            wrist_wrench.wrench.force.z,
        ]
    )
    torque_wrist = np.array(
        [
            wrist_wrench.wrench.torque.x,
            wrist_wrench.wrench.torque.y,
            wrist_wrench.wrench.torque.z,
        ]
    )

    force_tip = rotation_wrist_tip.T @ force_wrist
    torque_tip = rotation_wrist_tip.T @ (
        torque_wrist - np.cross(wrist_to_tip_translation, force_wrist)
    )

    return Wrench(
        force=Vector3(
            x=float(force_tip[0]),
            y=float(force_tip[1]),
            z=float(force_tip[2]),
        ),
        torque=Vector3(
            x=float(torque_tip[0]),
            y=float(torque_tip[1]),
            z=float(torque_tip[2]),
        ),
    )
