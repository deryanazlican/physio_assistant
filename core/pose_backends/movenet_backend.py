from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from core.pose_backends.base_backend import BasePoseBackend


class SimpleLandmark:
    def __init__(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        visibility: float = 1.0,
        presence: float = 1.0
    ):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


class MoveNetBackend(BasePoseBackend):
    """
    MoveNet SinglePose backend.
    17 keypoint çıktısını MediaPipe benzeri 33 landmark listesine map eder.
    """

    def __init__(self, variant: str = "lightning"):
        model_name = f"movenet_{variant}"
        super().__init__(model_name=model_name)

        if variant not in ("lightning", "thunder"):
            raise ValueError("variant 'lightning' veya 'thunder' olmalı")

        self.variant = variant
        self.input_size = 192 if variant == "lightning" else 256

        if variant == "lightning":
            model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
        else:
            model_url = "https://tfhub.dev/google/movenet/singlepose/thunder/4"

        self.module = hub.load(model_url)
        self.model = self.module.signatures["serving_default"]

    def process(self, frame) -> Dict[str, Any]:
        h, w = frame.shape[:2]

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(image, axis=0),
            self.input_size,
            self.input_size
        )
        input_image = tf.cast(input_image, dtype=tf.int32)

        outputs = self.model(input_image)
        key = list(outputs.keys())[0]
        keypoints_with_scores = outputs[key].numpy()[0, 0, :, :]  # (17, 3)

        landmarks = self._convert_movenet_to_mediapipe_style(
            keypoints_with_scores=keypoints_with_scores,
            width=w,
            height=h
        )

        scores = keypoints_with_scores[:, 2]
        confidence = float(np.mean(scores)) if len(scores) > 0 else None
        pose_detected = confidence is not None and confidence > 0.1

        return {
            "landmarks": landmarks if pose_detected else None,
            "pose_detected": pose_detected,
            "confidence": confidence,
            "raw_result": keypoints_with_scores
        }

    def draw(self, frame, result: Dict[str, Any]) -> None:
        keypoints_with_scores = result.get("raw_result")
        if keypoints_with_scores is None:
            return

        h, w = frame.shape[:2]

        # COCO edges
        edges = [
            (0, 1), (0, 2),
            (1, 3), (2, 4),
            (0, 5), (0, 6),
            (5, 7), (7, 9),
            (6, 8), (8, 10),
            (5, 6),
            (5, 11), (6, 12),
            (11, 12),
            (11, 13), (13, 15),
            (12, 14), (14, 16)
        ]

        points = []
        for kp in keypoints_with_scores:
            y, x, score = kp
            px, py = int(x * w), int(y * h)
            points.append((px, py, score))

        for i, (px, py, score) in enumerate(points):
            if score > 0.2:
                cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

        for a, b in edges:
            if points[a][2] > 0.2 and points[b][2] > 0.2:
                cv2.line(frame, (points[a][0], points[a][1]), (points[b][0], points[b][1]), (255, 255, 0), 2)

    def close(self):
        pass

    def _convert_movenet_to_mediapipe_style(
        self,
        keypoints_with_scores,
        width: int,
        height: int
    ) -> List[SimpleLandmark]:
        """
        MoveNet 17 keypoint -> MediaPipe benzeri 33 landmark listesi
        MoveNet formatı: [y, x, score]
        """

        def make_point(x_norm, y_norm, vis=1.0, z=0.0):
            return SimpleLandmark(
                x=float(x_norm),
                y=float(y_norm),
                z=float(z),
                visibility=float(vis),
                presence=float(vis)
            )

        mp_like = [SimpleLandmark(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(33)]

        # MoveNet indices:
        # 0 nose
        # 1 left_eye
        # 2 right_eye
        # 3 left_ear
        # 4 right_ear
        # 5 left_shoulder
        # 6 right_shoulder
        # 7 left_elbow
        # 8 right_elbow
        # 9 left_wrist
        # 10 right_wrist
        # 11 left_hip
        # 12 right_hip
        # 13 left_knee
        # 14 right_knee
        # 15 left_ankle
        # 16 right_ankle

        mapping = {
            0: 0,
            3: 7,
            4: 8,
            5: 11,
            6: 12,
            7: 13,
            8: 14,
            9: 15,
            10: 16,
            11: 23,
            12: 24,
            13: 25,
            14: 26,
            15: 27,
            16: 28,
        }

        for mv_idx, mp_idx in mapping.items():
            y, x, score = keypoints_with_scores[mv_idx]
            mp_like[mp_idx] = make_point(x, y, score)

        # bazı ek yüz noktaları yaklaşık
        try:
            y, x, score = keypoints_with_scores[0]  # nose
            mp_like[0] = make_point(x, y, score)
        except Exception:
            pass

        try:
            l_eye = keypoints_with_scores[1]
            r_eye = keypoints_with_scores[2]
            l_ear = keypoints_with_scores[3]
            r_ear = keypoints_with_scores[4]

            mp_like[2] = make_point(l_eye[1], l_eye[0], l_eye[2])
            mp_like[5] = make_point(r_eye[1], r_eye[0], r_eye[2])
            mp_like[7] = make_point(l_ear[1], l_ear[0], l_ear[2])
            mp_like[8] = make_point(r_ear[1], r_ear[0], r_ear[2])
        except Exception:
            pass

        return mp_like