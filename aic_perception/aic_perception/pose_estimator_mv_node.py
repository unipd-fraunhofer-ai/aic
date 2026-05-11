from email.mime import image
import threading
import time
from tkinter.font import names
from matplotlib import image
import cv2
import numpy as np
from typing import Optional

import scipy
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped, Transform
from tf2_ros import TransformBroadcaster

from aic_perception.yolo_wrapper import YoloWrapper, plot_bboxes, plot_masks
from aic_perception.utils.pose_estimator import Path, PoseEstimator
from aic_perception.utils.infer import save_visualizations

from message_filters import Subscriber, ApproximateTimeSynchronizer

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


#ros2 run tf2_ros tf2_echo tool0 center_camera/optical
extrinsics_cache_w2c= {
    'center_camera': {
        "R_w2c": R.from_quat([-0.131, 0.000, -0.000, 0.991]).as_matrix().tolist(),    
        "t_w2c": np.array([-0.000, -0.116, -0.009]).tolist(),
    },
    'left_camera': {
        "R_w2c": R.from_quat([-0.113, 0.065, -0.496, 0.859]).as_matrix().tolist(),    
        "t_w2c": np.array([-0.101, -0.058, -0.009]).tolist(),
    },
    'right_camera': {
        "R_w2c": R.from_quat([-0.113, -0.065, 0.496, 0.859]).as_matrix().tolist(),    
        "t_w2c": np.array([0.101, -0.058, -0.009]).tolist(),
    },
}

#ros2 run tf2_ros tf2_echo center_camera/optical tool0
extrinsics_cache_c2w= {
    'center_camera': {
        "R_w2c": R.from_quat([0.131, -0.000, 0.000, 0.991]).as_matrix().tolist(),    
        "t_w2c": np.array([-0.000, 0.110, 0.039]).tolist(),
    },
    'left_camera': {
        "R_w2c": R.from_quat([0.113, -0.065, 0.496, 0.859]).as_matrix().tolist(),    
        "t_w2c": np.array([0.000, 0.110, 0.039]).tolist(),
    },
    'right_camera': {
        "R_w2c": R.from_quat([0.113, 0.065, -0.496, 0.859]).as_matrix().tolist(),    
        "t_w2c": np.array([-0.000, 0.110, 0.039]).tolist(),
    },
}

#ros2 run tf2_ros tf2_echo center_camera/optical base_link
extrinsics_cache_c2base= {
    'center_camera': {
        "R_w2c": R.from_quat([0.991, 0.000, -0.000, -0.131]).as_matrix().tolist(),    
        "t_w2c": np.array([0.372, 0.164, 0.588]).tolist(),
    },
    'left_camera': {
        "R_w2c": R.from_quat([0.859, 0.496, 0.065, -0.113]).as_matrix().tolist(),    
        "t_w2c": np.array([0.017, 0.381, 0.646]).tolist(),
    },
    'right_camera': {
        "R_w2c": R.from_quat([0.859, -0.496, -0.065, -0.113]).as_matrix().tolist(),    
        "t_w2c": np.array([0.354, -0.241, 0.480]).tolist(),
    },
}

def load_extrinsics(camera_name, scale_factor=1000.0):
    if camera_name in extrinsics_cache_c2base:
        extrinsics = extrinsics_cache_c2base[camera_name].copy()
        extrinsics["t_w2c"] = [t * scale_factor for t in extrinsics["t_w2c"]]
        return extrinsics
    else:
        raise ValueError(f"Unknown camera name {camera_name} for extrinsics")

class PoseEstimatorNode(Node):
    def __init__(self):
        super().__init__("pose_estimator_node")

        #self.declare_parameter("image_topic", "/camera/image_raw")
        #self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("output_topic", "/pose_estimator/poses")
        self.declare_parameter("debug_image_topic", "/pose_estimator/debug_image")
        self.declare_parameter("templates_dir", "data/templates")
        self.declare_parameter("models_dir", "data/ic/models")
        self.declare_parameter("yolo_checkpoint_path", "")
        self.declare_parameter("debug_bbox", False)
        self.declare_parameter("debug_mask", True)
        self.declare_parameter("policy_data_path", "data/")

        #image_topic = self.get_parameter("image_topic").value
        #camera_info_topic = self.get_parameter("camera_info_topic").value
        output_topic = self.get_parameter("output_topic").value
        debug_image_topic = self.get_parameter("debug_image_topic").value
        templates_dir = self.get_parameter("templates_dir").value
        models_dir = self.get_parameter("models_dir").value
        yolo_checkpoint_path = self.get_parameter("yolo_checkpoint_path").value
        self.debug_bbox = self.get_parameter("debug_bbox").value
        self.debug_mask = self.get_parameter("debug_mask").value
        self.policy_data_path = self.get_parameter("policy_data_path").value

        # PATHS
        self.policy_data_path = Path(self.policy_data_path)
        yolo_checkpoint_path = self.policy_data_path / "weights_istances/yolo26_segment.pt"
        templates_dir = self.policy_data_path / "templates"
        models_dir = self.policy_data_path / "ic/models"
        print(f"Loading YoloWrapper with checkpoint: {yolo_checkpoint_path}")

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        self.tf_broadcaster = TransformBroadcaster(self)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )


        # CAMERAS
        self.camera_names = ["center_camera", "left_camera", "right_camera"]
        self.camera_frames = {name: f"{name}/optical" for name in self.camera_names}
        self.camera_image_topics = {name: f"/{name}/image" for name in self.camera_names}
        self.camera_info_topics = {name: f"/{name}/camera_info" for name in self.camera_names}

        self.latest_image_msg = None
        self.latest_camera_info_msg = None

        self.image_sub = []
        self.camera_info_sub = []
        for camera_name in self.camera_names:
            image_topic = self.camera_image_topics[camera_name]
            camera_info_topic = self.camera_info_topics[camera_name]
            sub = Subscriber(self, Image, image_topic, qos_profile=sensor_qos)
            self.image_sub.append(sub)

            camera_info_sub = Subscriber(self, CameraInfo, camera_info_topic, qos_profile=sensor_qos)
            self.camera_info_sub.append(camera_info_sub)

        # Message filters for synchronized subscriptions
        self.image_ts = ApproximateTimeSynchronizer(
            self.image_sub,
            queue_size=10,
            slop=0.05  # 50ms tolerance for approximate sync
        )
        self.image_ts.registerCallback(self.image_callback)

        self.camera_info_ts = ApproximateTimeSynchronizer(
            self.camera_info_sub,
            queue_size=10,
            slop=0.05  # 50ms tolerance for approximate sync
        )
        self.camera_info_ts.registerCallback(self.camera_info_callback)

        self.yolo = YoloWrapper(yolo_checkpoint_path)   

        self.camera_info_received = False

        # Wait until we receive the first camera info message to initialize the estimator.
        while rclpy.ok() and self.latest_camera_info_msg is None:
            self.get_logger().info("Waiting for camera info message...")
            rclpy.spin_once(self, timeout_sec=1.0)

        #self.camera_name = "center_camera"
        T_w2c = np.eye(4)
        self.cameras = {}
        for camera_name in self.camera_names:
            self.cameras[camera_name] = {
                "intrinsics": load_intrinsics(self.latest_camera_info_msg[camera_name]),
                "extrinsics": load_extrinsics(camera_name),
            }

        self.estimator = PoseEstimator(
            cameras=self.cameras,
            templates_dir=templates_dir,
            models_dir=models_dir,
        )

        # self.pose_pub = self.create_publisher(PoseArray, output_topic, 10)
        # self.debug_image_pub = self.create_publisher(Image, debug_image_topic, 10)
        # self.bbox_image_pub = self.create_publisher(Image, "/pose_estimator/debug_bbox_image", 10)
        # self.mask_image_pub = self.create_publisher(Image, "/pose_estimator/debug_mask_image", 10)
        
        self.mask_image_pub = {}
        for name in self.camera_names:
            self.mask_image_pub[name] = self.create_publisher(Image, f"/pose_estimator/{name}_debug_mask_image", 10)

        self.worker = threading.Thread(target=self.processing_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(
            f"PoseEstimatorNode ready | image={image_topic} | "
            f"camera_info={camera_info_topic} | output={output_topic} | debug_image={debug_image_topic}"
        )

    def image_callback(self, *msgs: Image):
        # Sovrascrive sempre: teniamo solo il frame più recente.
        with self.lock:
            self.latest_image_msg = {camera_name: msg for camera_name, msg in zip(self.camera_names, msgs)}
            #self.get_logger().info(f"Received image frame {msg.header.frame_id}")

    def camera_info_callback(self, *msgs: CameraInfo):
        if self.camera_info_received:
            return

        self.camera_info_received = True

        with self.lock:
            self.latest_camera_info_msg = {camera_name: msg for camera_name, msg in zip(self.camera_names, msgs)}
            self.get_logger().debug(f"Received camera info")
        
        # Once we have the camera info, we can unsubscribe
        for sub in self.camera_info_sub:
            self.destroy_subscription(sub)
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

            self.get_logger().info(f"Processing frame {image_msg['center_camera'].header.frame_id}...")

            try:
                start = time.perf_counter()

                target_name = "nic_card_mount_0"

                camera_inputs = {}
                for camera_name, img_msg in image_msg.items():
                    self.get_logger().info(f"Received image frame from {camera_name}")
                    camera_inputs[camera_name] = self.estimate_masks(camera_name, img_msg, target_name=target_name)

            
                # Check num mask for cameras
                for camera_name, inputs in camera_inputs.items():
                    num_masks = sum(len(masks) for masks in inputs["masks"].values())
                    self.get_logger().info(f"Camera {camera_name} has {num_masks} masks detected")

                
                results = self.estimator.estimate_pose(camera_inputs)

                self.get_logger().info(f"Pose estimation results: {results}")

                for camera_name, camera_results in results.items():
                    for object_id, poses in camera_results.items():
                        for instance_id, pose in enumerate(poses):
                            quality = None if pose is None else pose["quality"]
                            print(camera_name, object_id, instance_id, quality)
                            tf_name = f"{camera_name}_object{object_id}_instance{instance_id}"
                            self.publish_object_tf(pose, self.camera_frames[camera_name], tf_name)

                

                print(results.keys())
                data_dir = self.policy_data_path / "ic"
                output_dir = self.policy_data_path  / "visualizations"
                scene_id = 1
                frame_id = 0  # Assuming single frame for now

                self.get_logger().info(f"Saving visualizations to {output_dir}...")
                save_visualizations(data_dir, output_dir, self.cameras, camera_inputs, results, scene_id, frame_id)
                



                dt = time.perf_counter() - start
                self.get_logger().info(f"Pose estimation done in {dt:.2f} s")

            except Exception as exc:
                self.get_logger().error(f"Pose estimation failed: {exc}")

    def estimate_masks(self, camera_name: str, image_msg: Image, target_name=None):
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

            # Search only for the target object if target_name is specified
            if target_name is not None and class_name != target_name:
                self.get_logger().warning(f"Skipping YOLO result with class_name {class_name} since it does not match target_name {target_name}")
                continue

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

        if self.debug_mask:
            image_mask = plot_masks(frame, raw_results)
            image_mask_msg = self.bridge.cv2_to_imgmsg(image_mask, encoding="bgr8")
            image_mask_msg.header = image_msg.header
            self.mask_image_pub[camera_name].publish(image_mask_msg)
        

        return {
            "color": image,
            "masks": masks,
            "names": names,
        }

        
    
    def publish_object_tf(self, pose, frame_id, tf_name: str):
        tf_msg = TransformStamped()

        #tf_msg.header.stamp = header.stamp
        #tf_msg.header.frame_id = header.frame_id
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = frame_id
        tf_msg.child_frame_id = tf_name

        tf_msg.transform = self.pose_to_transform(pose, scale_factor=1000.0)  # Convert from mm to m

        self.tf_broadcaster.sendTransform(tf_msg)

    def pose_to_transform(self, pose, scale_factor=1000.0) -> Transform:
        transform = Transform()

        #print(pose) A dict {'object_id', 'R_m2c', 't_m2c', 'T_m2c', 'T_m2w','quality', 'num_inliers', 'template_id', 'corresp_id'}
        t = pose['t_m2c'] / scale_factor  # meters
        q = R.from_matrix(pose['R_m2c']).as_quat()  # x,y,z,w

        transform.translation.x = float(t[0])
        transform.translation.y = float(t[1])
        transform.translation.z = float(t[2])
        transform.rotation.x = float(q[0])
        transform.rotation.y = float(q[1])
        transform.rotation.z = float(q[2])
        transform.rotation.w = float(q[3])

        return transform

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