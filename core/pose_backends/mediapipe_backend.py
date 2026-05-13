from typing import Any, Dict, Optional
import cv2
import mediapipe as mp

from core.pose_backends.base_backend import BasePoseBackend


class MediaPipePoseBackend(BasePoseBackend):
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        super().__init__(model_name="mediapipe")
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame) -> Dict[str, Any]:
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.pose.process(image_rgb)

        landmarks = None
        confidence = None
        pose_detected = results.pose_landmarks is not None

        if pose_detected:
            landmarks = results.pose_landmarks.landmark
            try:
                visible_points = [p.visibility for p in landmarks if hasattr(p, "visibility")]
                if visible_points:
                    confidence = float(sum(visible_points) / len(visible_points))
            except Exception:
                confidence = None

        return {
            "landmarks": landmarks,
            "pose_detected": pose_detected,
            "confidence": confidence,
            "raw_result": results
        }

    def draw(self, frame, result: Dict[str, Any]) -> None:
        raw_result = result.get("raw_result")
        if raw_result and raw_result.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                raw_result.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass