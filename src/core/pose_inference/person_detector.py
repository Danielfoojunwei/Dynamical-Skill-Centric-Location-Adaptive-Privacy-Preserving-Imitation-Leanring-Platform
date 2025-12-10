"""
Person Detection for Pose Estimation

Uses YOLOv8/YOLOv11 or RTMDet for person detection before pose estimation.
This is required for multi-person pose estimation scenarios.

Models available from:
- Ultralytics: yolov8n/s/m/l/x
- HuggingFace: various YOLO models
"""

import numpy as np
import os
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Person detection result."""
    bboxes: np.ndarray  # [N, 4] - x1, y1, x2, y2
    scores: np.ndarray  # [N] - confidence scores
    class_ids: np.ndarray  # [N] - class IDs (0 = person)
    timestamp: float
    image_size: Tuple[int, int]

    def get_person_bboxes(self, threshold: float = 0.5) -> np.ndarray:
        """Get person bboxes with scores above threshold."""
        person_mask = (self.class_ids == 0) & (self.scores > threshold)
        bboxes = self.bboxes[person_mask]
        scores = self.scores[person_mask]
        return np.hstack([bboxes, scores[:, np.newaxis]])


class PersonDetector:
    """
    Person detector using ONNX Runtime.

    Supports:
    - YOLOv8/v11 nano to xlarge
    - RTMDet for lightweight detection

    Usage:
        detector = PersonDetector(model_name="yolov8n")
        result = detector.detect(image)
        person_bboxes = result.get_person_bboxes()
    """

    YOLO_MODELS = {
        "yolov8n": {
            "url": "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx",
            "input_size": (640, 640),
        },
        "yolov8s": {
            "url": "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.onnx",
            "input_size": (640, 640),
        },
        "yolov8m": {
            "url": "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8m.onnx",
            "input_size": (640, 640),
        },
    }

    def __init__(
        self,
        model_name: str = "yolov8n",
        model_path: Optional[str] = None,
        device: str = "cuda",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.session = None
        self._initialized = False

        # Get model config
        if model_name in self.YOLO_MODELS:
            self.input_size = self.YOLO_MODELS[model_name]["input_size"]
        else:
            self.input_size = (640, 640)

        self._init_session(model_path)

    def _init_session(self, model_path: Optional[str] = None):
        """Initialize ONNX Runtime session."""
        try:
            import onnxruntime as ort

            if model_path and os.path.exists(model_path):
                path = model_path
            else:
                path = os.path.join("models", f"{self.model_name}.onnx")

            if not os.path.exists(path):
                # Try to download
                if self.model_name in self.YOLO_MODELS:
                    self._download_model(path)
                else:
                    logger.warning(f"Model not found: {path}")
                    return

            providers = (
                ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if "cuda" in self.device else ['CPUExecutionProvider']
            )

            self.session = ort.InferenceSession(path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self._initialized = True

            logger.info(f"PersonDetector initialized: {self.model_name}")

        except ImportError:
            logger.error("onnxruntime not installed")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")

    def _download_model(self, save_path: str):
        """Download YOLO model."""
        import urllib.request

        if self.model_name not in self.YOLO_MODELS:
            raise ValueError(f"Unknown model: {self.model_name}")

        url = self.YOLO_MODELS[self.model_name]["url"]
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        logger.info(f"Downloading {self.model_name}...")
        urllib.request.urlretrieve(url, save_path)
        logger.info(f"Saved to {save_path}")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Preprocess image for YOLO."""
        h, w = image.shape[:2]
        target_w, target_h = self.input_size

        # Letterbox resize
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2

        try:
            import cv2
            resized = cv2.resize(image, (new_w, new_h))
        except ImportError:
            from scipy.ndimage import zoom
            resized = zoom(image, (new_h / h, new_w / w, 1), order=1).astype(np.uint8)

        # Create letterbox
        canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # BGR to RGB, normalize
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0

        # HWC to CHW, add batch
        tensor = rgb.transpose(2, 0, 1)[np.newaxis, ...]

        transform_info = {
            "scale": scale,
            "pad": (pad_w, pad_h),
            "original_size": (w, h),
        }

        return tensor.astype(np.float32), transform_info

    def _postprocess(
        self,
        output: np.ndarray,
        transform_info: dict
    ) -> DetectionResult:
        """Postprocess YOLO output."""
        # Output shape: [1, 84, 8400] for YOLOv8
        # 84 = 4 (bbox) + 80 (classes)
        predictions = output[0].T  # [8400, 84]

        # Extract boxes and scores
        boxes = predictions[:, :4]  # cx, cy, w, h
        class_probs = predictions[:, 4:]

        # Get class with max probability
        class_ids = np.argmax(class_probs, axis=1)
        scores = np.max(class_probs, axis=1)

        # Filter by confidence and person class (class 0)
        person_mask = (class_ids == 0) & (scores > self.conf_threshold)
        boxes = boxes[person_mask]
        scores = scores[person_mask]
        class_ids = class_ids[person_mask]

        if len(boxes) == 0:
            return DetectionResult(
                bboxes=np.array([]).reshape(0, 4),
                scores=np.array([]),
                class_ids=np.array([]),
                timestamp=time.time(),
                image_size=transform_info["original_size"],
            )

        # Convert cx, cy, w, h to x1, y1, x2, y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        bboxes = np.stack([x1, y1, x2, y2], axis=1)

        # Transform back to original coordinates
        scale = transform_info["scale"]
        pad_w, pad_h = transform_info["pad"]
        orig_w, orig_h = transform_info["original_size"]

        bboxes[:, [0, 2]] = (bboxes[:, [0, 2]] - pad_w) / scale
        bboxes[:, [1, 3]] = (bboxes[:, [1, 3]] - pad_h) / scale

        # Clip to image bounds
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, orig_w)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, orig_h)

        # NMS
        keep = self._nms(bboxes, scores, self.iou_threshold)
        bboxes = bboxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        return DetectionResult(
            bboxes=bboxes,
            scores=scores,
            class_ids=class_ids,
            timestamp=time.time(),
            image_size=transform_info["original_size"],
        )

    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> np.ndarray:
        """Non-maximum suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=int)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep)

    def detect(self, image: np.ndarray) -> DetectionResult:
        """
        Detect persons in image.

        Args:
            image: BGR image [H, W, 3]

        Returns:
            DetectionResult with person bounding boxes
        """
        if not self._initialized:
            # Return empty result
            h, w = image.shape[:2]
            return DetectionResult(
                bboxes=np.array([[0, 0, w, h]]),
                scores=np.array([0.95]),
                class_ids=np.array([0]),
                timestamp=time.time(),
                image_size=(w, h),
            )

        tensor, transform_info = self._preprocess(image)
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        return self._postprocess(output, transform_info)
