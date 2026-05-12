"""Simple YOLO model wrapper."""

from unittest import result

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ID to color mapping for consistent visualization
ID_COLOR_MAP = {}
rng = np.random.default_rng(42)

def get_color_by_id(class_id: int) -> tuple:
	"""Get consistent color for a class ID."""
	if class_id not in ID_COLOR_MAP:
		ID_COLOR_MAP[class_id] = tuple(rng.integers(0, 256, size=3, dtype=np.uint8).tolist())
	return ID_COLOR_MAP[class_id]

def plot_bboxes(image: np.ndarray, bboxes: list[dict]) -> np.ndarray:
	"""Plot bounding boxes on the image."""
	overlay = image.copy()
	for bbox in bboxes:
		x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
		confidence = bbox["confidence"]
		class_name = bbox["class_name"]
		class_id = bbox["class_id"]
		color = get_color_by_id(class_id)
		cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
		cv2.putText(overlay, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
	return overlay

def plot_masks(image: np.ndarray, masks: list[dict]) -> np.ndarray:
	"""Plot segmentation masks on the image."""
	overlay = image.copy()
	for mask_info in masks:
		if mask_info["mask"] is None:
			continue
		mask_bool = mask_info["mask"].astype(bool)
		class_id = mask_info["class_id"]
		color = get_color_by_id(class_id)
		colored = np.zeros_like(image, dtype=np.uint8)
		colored[mask_bool] = color
		overlay = np.where(mask_bool[..., None], (overlay * (1 - 0.5) + colored * 0.5).astype(np.uint8), overlay)

		x1, y1, x2, y2 = int(mask_info["x1"]), int(mask_info["y1"]), int(mask_info["x2"]), int(mask_info["y2"])
		cls_name = mask_info["class_name"]
		cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
		cv2.putText(overlay, cls_name, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
	return overlay


class YoloWrapper:
	"""Wrap a YOLO checkpoint and expose prediction helpers."""

	def __init__(self, checkpoint_path: str | Path) -> None:
		self.checkpoint_path = Path(checkpoint_path)
		if not self.checkpoint_path.exists():
			raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

		self.model = YOLO(self.checkpoint_path)

	def predict(self, image: np.ndarray, keep_best: bool = True) -> list[dict]:
		"""Run inference on the given image and return a list of predictions."""
		# results = self.model(image)
		# for result in results:
		# 	print(f'BBOX: {result.boxes.shape}')  # Boxes object for bounding box outputs
		# 	masks = result.masks  # Masks object for segmentation masks outputs
		# 	print(f'masks = {masks.shape if masks is not None else None}')
		# 	keypoints = result.keypoints  # Keypoints object for pose outputs
		# 	print(f'keypoints = {keypoints.shape if keypoints is not None else None}')
		# 	probs = result.probs  # Probs object for classification outputs
		# 	print(f'probs = {probs.shape if probs is not None else None}')
		# 	obb = result.obb  # Oriented boxes object for OBB outputs
		# 	print(f'obb = {obb.shape if obb is not None else None}') 
		# print('#########'*10)

		result = self.model.predict(image, conf=0.25, verbose=False)[0]  # Get the first (and only) result
		#print(result.names)

		return self._sort_by_confidence(self._parse_yolo_masks(result), keep_best=keep_best)  # Parse and sort masks by confidence


	def _parse_yolo_bboxes(self, result):
		"""Parse YOLO results into a list of bounding box dictionaries."""
		bboxes = []
		id2name = result.names  # Mapping from class ID to class name
		for box in result.boxes:
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
			confidence = box.conf[0].cpu().item()  # Confidence score
			class_id = int(box.cls[0].cpu().item())  # Class ID
			bboxes.append({
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"confidence": confidence,
				"class_id": class_id,
				"class_name": id2name[class_id],
				"mask": None
			})
		return bboxes
	
	def _parse_yolo_masks(self, result):
		"""Parse YOLO results into a list of mask dictionaries."""
		if result.masks is None:
			return self._parse_yolo_bboxes(result)  # Fallback to bounding boxes if no masks are available
		
		masks = result.masks.data.detach().cpu().numpy()  # [N, h, w]
		id2name = result.names  # Mapping from class ID to class name
		img_h, img_w = result.orig_shape

		out_masks = []
		for box, mask in zip(result.boxes, masks):
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Bounding box coordinates
			confidence = box.conf[0].cpu().item()  # Confidence score
			class_id = int(box.cls[0].cpu().item())  # Class ID

			mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
			if mask_uint8.shape[:2] != (img_h, img_w):
				mask_uint8 = cv2.resize(mask_uint8, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
									
			out_masks.append({
				"x1": x1,
				"y1": y1,
				"x2": x2,
				"y2": y2,
				"confidence": confidence,
				"class_id": class_id,
				"class_name": id2name[class_id],
				"mask": mask_uint8
			})
		return out_masks
	
	def _sort_by_confidence(self, results: list[dict], keep_best: bool = True) -> list[dict]:
		"""Sort bounding boxes by confidence score in descending order and append counter to duplicate class names."""
		sorted_results = sorted(results, key=lambda x: x["confidence"], reverse=True)
		class_name_counts = {}
		
		for result in sorted_results:
			class_name = result["class_name"]
			if class_name not in class_name_counts:
				class_name_counts[class_name] = 0
			elif keep_best:
				result["class_name"] = None
			else:
				class_name_counts[class_name] += 1
				result["class_name"] = f"{class_name}_{class_name_counts[class_name]}"
		
		return sorted_results