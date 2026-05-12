#!/usr/bin/env python3

import glob
import os
from turtle import stamp
import cv2
import numpy as np
import yaml
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from rclpy.qos import qos_profile_sensor_data # -> BEST_EFFORT + depth=1

class TestCameraNode(Node):
	def __init__(self) -> None:
		super().__init__("test_camera_node")

		self.declare_parameter("data_dir", "data")
		self.declare_parameter("image_topic", "/camera/image_raw")
		self.declare_parameter("camera_info_topic", "/camera/camera_info")
		self.declare_parameter("frame_id", "camera_optical_frame")
		self.declare_parameter("fps", 30.0)

		data_dir = self.get_parameter("data_dir").get_parameter_value().string_value
		image_topic = self.get_parameter("image_topic").get_parameter_value().string_value
		camera_info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
		self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
		fps = float(self.get_parameter("fps").get_parameter_value().double_value)
		if fps <= 0.0:
			fps = 30.0

		self._images = self._load_images(data_dir)
		self._image_index = 0
		self._camera_info_template = self._load_camera_info(data_dir)

		self.image_pub = self.create_publisher(Image, image_topic, qos_profile_sensor_data)
		self.camera_info_pub = self.create_publisher(CameraInfo, camera_info_topic, 10)

		self._timer = self.create_timer(1.0 / fps, self._publish)

		self.get_logger().info(
			f"Publishing camera stream from '{data_dir}' on '{image_topic}' and '{camera_info_topic}' at {fps:.1f} FPS"
		)

	def _load_images(self, data_dir: str) -> list[str]:
		color_dir = os.path.join(data_dir, "color")
		patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
		images: list[str] = []
		for pattern in patterns:
			images.extend(sorted(glob.glob(os.path.join(color_dir, pattern))))
		if not images:
			raise FileNotFoundError(f"No images found in {color_dir}")
		return images

	def _load_camera_info(self, data_dir: str) -> CameraInfo:
		intrinsics_path = os.path.join(data_dir, "intrinsics.yaml")
		with open(intrinsics_path, "r", encoding="utf-8") as f:
			data = yaml.safe_load(f)

		fx = float(data["fx"])
		fy = float(data["fy"])
		cx = float(data["cx"])
		cy = float(data["cy"])

		camera_info = CameraInfo()
		camera_info.distortion_model = data.get("distortion_model", "plumb_bob")
		camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
		camera_info.d = (
			[float(data.get(key, 0.0)) for key in ("dist_k0", "dist_k1", "dist_px", "dist_py", "dist_k2")]
			if int(data.get("has_dist_coeff", 0))
			else [0.0, 0.0, 0.0, 0.0, 0.0]
		)
		camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
		camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
		camera_info.height = int(data.get("img_height", 0))
		camera_info.width = int(data.get("img_width", 0))
		return camera_info

	def _publish(self) -> None:
		stamp = self.get_clock().now().to_msg()
		image_path = self._images[self._image_index % len(self._images)]
		frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
		if frame is None:
			self.get_logger().warning(f"Failed to read image: {image_path}")
			return
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		self.get_logger().debug(f"Publishing image {self._image_index} -- {image_path} at time {stamp.sec}.{stamp.nanosec:09d}")

		image_msg = Image()
		image_msg.header.stamp = stamp
		image_msg.header.frame_id = self.frame_id
		image_msg.height = int(frame.shape[0])
		image_msg.width = int(frame.shape[1])
		image_msg.encoding = "rgb8"
		image_msg.is_bigendian = 0
		image_msg.step = int(frame.shape[1]) * 3
		image_msg.data = frame.tobytes()

		camera_info_msg = self._camera_info_template
		camera_info_msg.header.stamp = stamp
		camera_info_msg.header.frame_id = self.frame_id
		camera_info_msg.height = int(frame.shape[0])
		camera_info_msg.width = int(frame.shape[1])

		self.image_pub.publish(image_msg)
		self.camera_info_pub.publish(camera_info_msg)
		self._image_index += 1


def main(args: List[str] | None = None) -> None:
	rclpy.init(args=args)
	node = TestCameraNode()
	try:
		rclpy.spin(node)
	except KeyboardInterrupt:
		pass
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
	
"""
Usage:
ros2 run aic_perception test_camera_node --ros-args -p data_dir:=src/aic_perception/data
"""