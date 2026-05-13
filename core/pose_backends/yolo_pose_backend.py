from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from ultralytics import YOLO

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


class YOLOPoseBackend(BasePoseBackend):
    """
    YOLO Pose çıktısını MediaPipe benzeri landmark listesine çevirir.
    Böylece mevcut egzersiz modülleri minimum değişiklikle çalışabilir.
    """

    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        super().__init__(model_name="yolo_pose")
        self.model = YOLO(model_path)

    def process(self, frame) -> Dict[str, Any]:
        h, w = frame.shape[:2]
        results = self.model.predict(frame, verbose=False)

        if not results or len(results) == 0:
            return {
                "landmarks": None,
                "pose_detected": False,
                "confidence": None,
                "raw_result": None
            }

        result = results[0]

        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return {
                "landmarks": None,
                "pose_detected": False,
                "confidence": None,
                "raw_result": result
            }

        keypoints_xy = result.keypoints.xy[0].cpu().numpy()  # shape: (17, 2)

        conf_array = None
        try:
            if result.keypoints.conf is not None:
                conf_array = result.keypoints.conf[0].cpu().numpy()
        except Exception:
            conf_array = None

        landmarks = self._convert_yolo_to_mediapipe_style(
            keypoints_xy=keypoints_xy,
            conf_array=conf_array,
            width=w,
            height=h
        )

        confidence = None
        if conf_array is not None and len(conf_array) > 0:
            confidence = float(np.mean(conf_array))

        return {
            "landmarks": landmarks,
            "pose_detected": True,
            "confidence": confidence,
            "raw_result": result
        }

    def draw(self, frame, result: Dict[str, Any]) -> None:
        raw_result = result.get("raw_result")
        if raw_result is None:
            return

        try:
            plotted = raw_result.plot()
            frame[:] = plotted
        except Exception:
            pass

    def close(self):
        pass

    def _convert_yolo_to_mediapipe_style(
        self,
        keypoints_xy,
        conf_array,
        width: int,
        height: int
    ) -> List[SimpleLandmark]:
        """
        YOLO 17 keypoint -> MediaPipe 33 landmark benzeri liste

        COCO keypoints:
        0 nose
        1 left_eye
        2 right_eye
        3 left_ear
        4 right_ear
        5 left_shoulder
        6 right_shoulder
        7 left_elbow
        8 right_elbow
        9 left_wrist
        10 right_wrist
        11 left_hip
        12 right_hip
        13 left_knee
        14 right_knee
        15 left_ankle
        16 right_ankle
        """

        def make_point(x, y, vis=1.0, z=0.0):
            return SimpleLandmark(
                x=float(x) / float(width),
                y=float(y) / float(height),
                z=float(z),
                visibility=float(vis)
            )

        # MediaPipe 33 default dummy
        mp_like = [SimpleLandmark(0.0, 0.0, 0.0) for _ in range(33)]

        def get_conf(idx: int) -> float:
            if conf_array is None:
                return 1.0
            try:
                return float(conf_array[idx])
            except Exception:
                return 1.0

        # birebir eşleşenler
        mapping = {
            0: 0,    # nose -> nose
            5: 11,   # left_shoulder
            6: 12,   # right_shoulder
            7: 13,   # left_elbow
            8: 14,   # right_elbow
            9: 15,   # left_wrist
            10: 16,  # right_wrist
            11: 23,  # left_hip
            12: 24,  # right_hip
            13: 25,  # left_knee
            14: 26,  # right_knee
            15: 27,  # left_ankle
            16: 28   # right_ankle
        }

        for yolo_idx, mp_idx in mapping.items():
            x, y = keypoints_xy[yolo_idx]
            mp_like[mp_idx] = make_point(x, y, get_conf(yolo_idx))

        # Boyun için yaklaşık landmark üret
        try:
            nose = keypoints_xy[0]
            l_sh = keypoints_xy[5]
            r_sh = keypoints_xy[6]

            shoulder_center = (l_sh + r_sh) / 2.0
            neck = (nose + shoulder_center) / 2.0

            mp_like[0] = make_point(nose[0], nose[1], get_conf(0))
            mp_like[11] = make_point(l_sh[0], l_sh[1], get_conf(5))
            mp_like[12] = make_point(r_sh[0], r_sh[1], get_conf(6))

            # MediaPipe'te neck diye direkt nokta yok ama gerekirse 1'i yaklaşık doldurabiliriz
            mp_like[1] = make_point(neck[0], neck[1], min(get_conf(0), get_conf(5), get_conf(6)))
        except Exception:
            pass

        return mp_like