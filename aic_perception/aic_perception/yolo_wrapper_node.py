import argparse
import time
from typing import Any, Dict, List, Tuple

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from aic_perception.yolo_wrapper import YoloWrapper, plot_bboxes, plot_masks


class YoloWrapperNode(Node):
	def __init__(self, checkpoint_path: str = "") -> None:
		super().__init__("yolo_wrapper_node")

		self.declare_parameter("checkpoint_path", checkpoint_path)
		self.declare_parameter("input_image_topic", "/camera/image_raw")
		self.declare_parameter("output_bbox_topic", "/yolo/detections")
		self.declare_parameter("output_bbox_image_topic", "/yolo/bbox_image")
		self.declare_parameter("output_mask_image_topic", "/yolo/mask_image")
		self.declare_parameter("debug_bbox", False)
		self.declare_parameter("debug_mask", False)

		self.checkpoint_path = self.get_parameter("checkpoint_path").get_parameter_value().string_value
		self.input_image_topic = self.get_parameter("input_image_topic").get_parameter_value().string_value
		self.output_bbox_topic = self.get_parameter("output_bbox_topic").get_parameter_value().string_value
		self.output_bbox_image_topic = self.get_parameter("output_bbox_image_topic").get_parameter_value().string_value
		self.output_mask_image_topic = self.get_parameter("output_mask_image_topic").get_parameter_value().string_value
		self.debug_bbox = self.get_parameter("debug_bbox").get_parameter_value().bool_value
		self.debug_mask = self.get_parameter("debug_mask").get_parameter_value().bool_value

		if not self.checkpoint_path:
			self.get_logger().error("checkpoint_path is empty. Pass it via --checkpoint-path or ROS parameter.")
			raise ValueError("checkpoint_path is required")

		self.bridge = CvBridge()
		self.yolo = YoloWrapper(self.checkpoint_path)

		self.bbox_pub = self.create_publisher(Detection2DArray, self.output_bbox_topic, 10)
		self.bbox_image_pub = self.create_publisher(Image, self.output_bbox_image_topic, 10)
		self.mask_image_pub = self.create_publisher(Image, self.output_mask_image_topic, 10)
		self.sub = self.create_subscription(Image, self.input_image_topic, self.image_callback, 10)

		self.get_logger().info(
			f"YOLO node ready | ckpt={self.checkpoint_path} | in={self.input_image_topic} | "
			f"bbox_out={self.output_bbox_topic} | img_out={self.output_bbox_image_topic}"
		)

	def image_callback(self, msg: Image) -> None:
		try:
			frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
		except Exception as exc:
			self.get_logger().error(f"Failed to convert image: {exc}")
			return
		
		start_time = time.time()

		try:
			raw_results = self.yolo.predict(frame)
		except Exception as exc:
			self.get_logger().error(f"Inference error: {exc}")
			return
		
		self.get_logger().info(f"Predicted {len(raw_results)} results")

		for result in raw_results:
			self.get_logger().info(f"Result: {result['class_name']} ({result['confidence']:.2f})")
		
		elapsed_time = time.time() - start_time
		self.get_logger().info(f"Inference time: {elapsed_time:.3f} seconds")

		if self.debug_bbox:
			image_bbox = plot_bboxes(frame, raw_results)
			self.bbox_image_pub.publish(self.bridge.cv2_to_imgmsg(image_bbox, encoding="bgr8"))

		if self.debug_mask:
			image_mask = plot_masks(frame, raw_results)
			self.mask_image_pub.publish(self.bridge.cv2_to_imgmsg(image_mask, encoding="bgr8"))

		detection_array = Detection2DArray()
		for mask_info in raw_results:
			detection_msg = self.build_detection_msg(mask_info)
			detection_array.detections.append(detection_msg)
		self.bbox_pub.publish(detection_array)

		elapsed_time = time.time() - start_time
		self.get_logger().info(f"Inference + Publishing time: {elapsed_time:.3f} seconds")




	def build_detection_msg(self, mask_info: Dict[str, Any]) -> Detection2D:
		"""Convert a single mask info dict to a Detection2D message."""
		x1, y1, x2, y2 = int(mask_info["x1"]), int(mask_info["y1"]), int(mask_info["x2"]), int(mask_info["y2"])
		class_name = str(mask_info["class_name"])
		confidence = float(mask_info["confidence"])

		detection = Detection2D()
		detection.bbox.center.position.x = float((x1 + x2) / 2.0)
		detection.bbox.center.position.y = float((y1 + y2) / 2.0)
		detection.bbox.center.theta = 0.0 
		detection.bbox.size_x = float(x2 - x1)
		detection.bbox.size_y = float(y2 - y1)

		results = ObjectHypothesisWithPose()
		results.hypothesis.class_id = class_name
		results.hypothesis.score = confidence
		detection.results.append(results)

		return detection
	
	def build_mask_msg(self, mask_info: Dict[str, Any]) -> Image:
		"""Convert a single mask info dict to an Image message."""
		pass


		
def main(args=None) -> None:
	parser = argparse.ArgumentParser(add_help=False)
	parser.add_argument("--checkpoint-path", default="", type=str)
	known_args, _ = parser.parse_known_args(args=args)

	rclpy.init(args=args)
	node = YoloWrapperNode(checkpoint_path=known_args.checkpoint_path)
	try:
		rclpy.spin(node)
	finally:
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()

