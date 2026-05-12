import threading
import time
import cv2
import numpy as np
from typing import Optional

import scipy

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

from aic_perception.yolo_wrapper import YoloWrapper, plot_bboxes, plot_masks
from aic_perception.utils.pose_estimator import PoseEstimator


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

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__("pose_estimator_node")

        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("output_topic", "/pose_estimator/poses")
        self.declare_parameter("debug_image_topic", "/pose_estimator/debug_image")
        self.declare_parameter("templates_dir", "data/templates")
        self.declare_parameter("models_dir", "data/ic/models")
        self.declare_parameter("yolo_checkpoint_path", "")
        self.declare_parameter("debug_bbox", False)
        self.declare_parameter("debug_mask", True)

        image_topic = self.get_parameter("image_topic").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        output_topic = self.get_parameter("output_topic").value
        debug_image_topic = self.get_parameter("debug_image_topic").value
        templates_dir = self.get_parameter("templates_dir").value
        models_dir = self.get_parameter("models_dir").value
        yolo_checkpoint_path = self.get_parameter("yolo_checkpoint_path").value
        self.debug_bbox = self.get_parameter("debug_bbox").value
        self.debug_mask = self.get_parameter("debug_mask").value

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.tf_broadcaster = TransformBroadcaster(self)

        self.latest_image_msg: Optional[Image] = None
        self.latest_camera_info_msg: Optional[CameraInfo] = None

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            sensor_qos,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            sensor_qos,
        )

        self.yolo = YoloWrapper(yolo_checkpoint_path)   

        # Wait until we receive the first camera info message to initialize the estimator.
        while rclpy.ok() and self.latest_camera_info_msg is None:
            self.get_logger().info("Waiting for camera info message...")
            rclpy.spin_once(self, timeout_sec=1.0)

        self.camera_name = "center"
        T_w2c = np.eye(4)
        cameras = {
            self.camera_name: {
                "intrinsics": load_intrinsics(self.latest_camera_info_msg),
                "extrinsics": {
                    "R_w2c": T_w2c[:3, :3].tolist(),
                    "t_w2c": T_w2c[:3, 3].tolist(),
                },
            }
        }

        self.estimator = PoseEstimator(
            cameras=cameras,
            templates_dir=templates_dir,
            models_dir=models_dir,
        )

        self.pose_pub = self.create_publisher(PoseArray, output_topic, 10)
        self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)
        self.bbox_image_pub = self.create_publisher(Image, "/pose_estimator/debug_bbox_image", 10)
        self.mask_image_pub = self.create_publisher(Image, "/pose_estimator/debug_mask_image", 10)

        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            f"PoseEstimatorNode ready | image={image_topic} | "
            f"camera_info={camera_info_topic} | output={output_topic} | debug_image={debug_image_topic}"
        )

    def image_callback(self, msg: Image):
        # Sovrascrive sempre: teniamo solo il frame più recente.
        with self.lock:
            self.latest_image_msg = msg
            #self.get_logger().info(f"Received image frame {msg.header.frame_id}")

    def camera_info_callback(self, msg: CameraInfo):
        # CameraInfo cambia raramente, quindi teniamo l'ultimo.
        with self.lock:
            self.latest_camera_info_msg = msg
            self.get_logger().debug(f"Received camera info for {msg.header.frame_id}")
        # Once we have the camera info, we can unsubscribe
        self.destroy_subscription(self.camera_info_sub)
        self.get_logger().info(f"Camera info saved! Subscriber removed.")

    def processing_loop(self):
        self.get_logger().info("Processing thread started.")
        while rclpy.ok() and not self.stop_event.is_set():
            with self.lock:
                #self.get_logger().info("Checking for new image frame...")
                #self.get_logger().info(f"Latest image msg: {self.latest_image_msg}")

                if self.latest_image_msg is None:
                    image_msg = None
                else:
                    image_msg = self.latest_image_msg

                    # Consuma il frame corrente.
                    # I frame che arrivano durante l'inference sovrascriveranno latest_image_msg.
                    self.latest_image_msg = None

            if image_msg is None:
                time.sleep(0.01)
                continue

            self.get_logger().info(f"Processing frame {image_msg.header.frame_id}...")

            try:
                start = time.perf_counter()

                poses = self.estimate_poses(image_msg)

                msg = PoseArray()
                msg.header = image_msg.header
                msg.poses = poses

                self.pose_pub.publish(msg)

                dt = time.perf_counter() - start
                self.get_logger().info(f"Pose estimation done in {dt:.2f} s")

            except Exception as exc:
                self.get_logger().error(f"Pose estimation failed: {exc}")

    def estimate_poses(self, image_msg: Image):
        frame = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

        # retrieve YOLO results (class_id, confidence, bbox, mask) for each detected object
        raw_results = self.yolo.predict(frame, keep_best=True)
        print(f"YOLO predicted {len(raw_results)} results")

        """
        {
            0: 'task_board_base', 
            1: 'nic_card_mount_0', 
            2: 'nic_card_mount_1', 
            3: 'nic_card_mount_2', 
            4: 'nic_card_mount_3', 
            5: 'nic_card_mount_4', 
            6: 'sc_port_0', 
            7: 'sc_port_1'
        }
        """
        
        class_names_map = {
            "nic_card_mount": 4,
            "task_board_base": 1
        }

        masks = {}
        names = {}
        for result in raw_results:
            #class_id = result["class_id"]
            class_name = result["class_name"]
            confidence = result["confidence"]
            class_id = result["class_id"]
            self.get_logger().info(f"YOLO result: {class_id} {class_name} ({confidence:.2f})")

            if class_name is None:
                self.get_logger().warning(f"Skipping YOLO result with no class name, class_id {class_id}, confidence {confidence:.2f}")
                continue

            object_model_name = '_'.join(class_name.split("_")[:-1]) if class_name[-1].isdigit() else class_name
            object_model_id = class_names_map.get(object_model_name, None)
            self.get_logger().info(f"{class_name} -> object_model_name: {object_model_name} with ID {object_model_id}")

            #self.get_logger().info(f"YOLO result: {object_model_id} ({confidence:.2f})")

            if object_model_id not in [1,4]:
                self.get_logger().warning(f"Unknown object_model_id {object_model_id}, class_name {class_name} in YOLO results")
                continue

            if "mask" in result:
                masks.setdefault(object_model_id, []).append(result["mask"])
                names.setdefault(object_model_id, []).append(class_name)

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0

        camera_inputs = {
            self.camera_name: {
                "color": image,
                "masks": masks,
                "names": names,
            }
        }

        results = self.estimator.estimate_pose(camera_inputs)

        for camera_name, camera_results in results.items():
            for object_id, poses in camera_results.items():
                for instance_id, pose in enumerate(poses):
                    quality = None if pose is None else pose["quality"]
                    tf_name = camera_inputs[camera_name]["names"][object_id][instance_id] if object_id in camera_inputs[camera_name]["names"] else "unknown"
                    print(camera_name, object_id, instance_id, quality, '->', tf_name)
                    self.publish_object_tf(pose, image_msg.header, tf_name)

        debug_image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        debug_image_msg.header = image_msg.header
        self.debug_image_pub.publish(debug_image_msg)

        if self.debug_bbox:
            image_bbox = plot_bboxes(frame, raw_results)
            image_bbox_msg = self.bridge.cv2_to_imgmsg(image_bbox, encoding="bgr8")
            image_bbox_msg.header = image_msg.header
            self.bbox_image_pub.publish(image_bbox_msg)

        if self.debug_mask:
            image_mask = plot_masks(frame, raw_results)
            image_mask_msg = self.bridge.cv2_to_imgmsg(image_mask, encoding="bgr8")
            image_mask_msg.header = image_msg.header
            self.mask_image_pub.publish(image_mask_msg)

        pose = Pose()
        pose.position.z = 1.0
        pose.orientation.w = 1.0

        return [pose]
    
    def publish_object_tf(self, pose: Pose, header, tf_name: str):
        tf_msg = TransformStamped()

        #print(pose) A dict {'object_id', 'R_m2c', 't_m2c', 'T_m2c', 'T_m2w','quality', 'num_inliers', 'template_id', 'corresp_id'}
        t = pose['t_m2c'] / 1000.0  # mm -> m
        q = scipy.spatial.transform.Rotation.from_matrix(pose['R_m2c']).as_quat()  # x,y,z,w

        tf_msg.header.stamp = header.stamp
        tf_msg.header.frame_id = header.frame_id
        tf_msg.child_frame_id = tf_name

        tf_msg.transform.translation.x = float(t[0])
        tf_msg.transform.translation.y = float(t[1])
        tf_msg.transform.translation.z = float(t[2])

        tf_msg.transform.rotation.x = float(q[0])
        tf_msg.transform.rotation.y = float(q[1])
        tf_msg.transform.rotation.z = float(q[2])
        tf_msg.transform.rotation.w = float(q[3])

        self.tf_broadcaster.sendTransform(tf_msg)

    def destroy_node(self):
        self.stop_event.set()
        if hasattr(self, "worker"):
            self.worker.join(timeout=1.0)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()