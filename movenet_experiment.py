import cv2
import time
import json
import os
import math
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


MAX_REPS = 10
TARGET_ANGLE = 0.10
EXERCISE_CODE = "ROM_ROT"
PATIENT_NAME = "DENEME"


def calculate_fps(prev_time):
    now = time.time()
    fps = 1.0 / max(now - prev_time, 1e-6)
    return now, fps


def smooth_value(prev_value, new_value, alpha=0.25):
    return (1 - alpha) * prev_value + alpha * new_value


def get_shoulder_width(l_sh, r_sh):
    w = abs(r_sh[0] - l_sh[0])
    return max(w, 1e-6)


def get_shoulder_center(l_sh, r_sh):
    return np.array([
        (l_sh[0] + r_sh[0]) / 2.0,
        (l_sh[1] + r_sh[1]) / 2.0,
        0.0
    ], dtype=float)


def calculate_flexion_components(nose, head_center, l_sh, r_sh):
    sh_center = get_shoulder_center(l_sh, r_sh)
    sh_w = get_shoulder_width(l_sh, r_sh)

    nose_dy = (nose[1] - sh_center[1]) / sh_w
    head_dy = (head_center[1] - sh_center[1]) / sh_w

    value = (nose_dy * 0.75 + head_dy * 0.25) * 70.0
    return float(value)


def midpoint(a, b):
    return (a + b) / 2.0

def calculate_rotation_ratio(nose, l_sh, r_sh):
    cx = (l_sh[0] + r_sh[0]) / 2.0
    sh_w = get_shoulder_width(l_sh, r_sh)
    return float((nose[0] - cx) / sh_w)

class MoveNetRunner:
    def __init__(self, variant="lightning"):
        self.input_size = 192 if variant == "lightning" else 256
        url = (
            "https://tfhub.dev/google/movenet/singlepose/lightning/4"
            if variant == "lightning"
            else "https://tfhub.dev/google/movenet/singlepose/thunder/4"
        )
        self.module = hub.load(url)
        self.model = self.module.signatures["serving_default"]
        self.model_name = f"movenet_{variant}"

    def infer(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_image = tf.image.resize_with_pad(
            tf.expand_dims(img, axis=0),
            self.input_size,
            self.input_size
        )
        input_image = tf.cast(input_image, dtype=tf.int32)
        outputs = self.model(input_image)
        key = list(outputs.keys())[0]
        kps = outputs[key].numpy()[0, 0, :, :]  # (17,3)
        return kps


def to_point(kp):
    # kp = [y, x, score]
    return np.array([kp[1], kp[0], 0.0], dtype=float), float(kp[2])


def main():
    cap = cv2.VideoCapture(0)
    runner = MoveNetRunner("lightning")

    reps_right = 0
    reps_left = 0
    state = "CENTER"
    baseline_flex = None
    baseline_samples = []
    smooth_angle = 0.0
    frame_logs = []

    start_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        prev_time, fps = calculate_fps(prev_time)

        kps = runner.infer(frame)

        nose, nose_conf = to_point(kps[0])
        l_eye, _ = to_point(kps[1])
        r_eye, _ = to_point(kps[2])
        l_ear, _ = to_point(kps[3])
        r_ear, _ = to_point(kps[4])
        l_sh, lsh_conf = to_point(kps[5])
        r_sh, rsh_conf = to_point(kps[6])

        raw_rot = calculate_rotation_ratio(nose, l_sh, r_sh)

        if baseline_flex is None:
            baseline_samples.append(raw_rot)
            if len(baseline_samples) >= 15:
                baseline_flex = float(np.mean(baseline_samples))

            msg = "Kalibrasyon..."
            angle = 0.0
        else:
            angle = raw_rot - baseline_flex
            smooth_angle = smooth_value(smooth_angle, angle, alpha=0.25)
            angle = smooth_angle

            abs_angle = abs(angle)

            if state == "CENTER":
                if angle >= 0.10:
                    state = "RIGHT"
                    msg = "Saga dondu, merkeze don"
                elif angle <= -0.10:
                    state = "LEFT"
                    msg = "Sola dondu, merkeze don"
                else:
                    msg = "Saga veya sola don"
            elif state == "RIGHT":
                if abs_angle <= 0.04:
                    reps_right += 1
                    state = "CENTER"
                    msg = f"Sag tekrar: {reps_right}/10 | Sol: {reps_left}/10"
                else:
                    msg = "Merkeze don"
            elif state == "LEFT":
                if abs_angle <= 0.04:
                    reps_left += 1
                    state = "CENTER"
                    msg = f"Sag: {reps_right}/10 | Sol tekrar: {reps_left}/10"
                else:
                    msg = "Merkeze don"

        frame_logs.append({
            "timestamp_sec": round(time.time() - start_time, 3),
            "fps": round(float(fps), 2),
            "angle": round(float(angle), 2),
            "reps_right": int(reps_right),
            "reps_left": int(reps_left),
            "reps_total": int(reps_right + reps_left),
            "confidence": round(float((nose_conf + lsh_conf + rsh_conf) / 3.0), 4),
            "is_complete": bool(angle >= TARGET_ANGLE)
        })

        cv2.putText(frame, f"Model: {runner.model_name}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Angle: {angle:.2f}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"RawRot: {raw_rot:.3f}", (20, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        if baseline_flex is not None:
            cv2.putText(frame, f"Baseline: {baseline_flex:.2f}", (20, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"R: {reps_right}/10  L: {reps_left}/10", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, msg, (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.imshow("MoveNet Experiment", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or (reps_right >= MAX_REPS and reps_left >= MAX_REPS):
            break

    cap.release()
    cv2.destroyAllWindows()

    angles = [x["angle"] for x in frame_logs]
    fpss = [x["fps"] for x in frame_logs]
    complete_count = sum(1 for x in frame_logs if x["is_complete"])

    summary = {
        "frame_count": len(frame_logs),
        "duration_sec": round(time.time() - start_time, 2),
        "completed_reps_right": reps_right,
        "completed_reps_left": reps_left,
        "completed_reps_total": reps_right + reps_left,
        "avg_angle": round(float(np.mean(angles)) if angles else 0.0, 2),
        "max_angle": round(float(np.max(angles)) if angles else 0.0, 2),
        "min_angle": round(float(np.min(angles)) if angles else 0.0, 2),
        "std_angle": round(float(np.std(angles)) if angles else 0.0, 2),
        "avg_fps": round(float(np.mean(fpss)) if fpss else 0.0, 2),
        "min_fps": round(float(np.min(fpss)) if fpss else 0.0, 2),
        "max_fps": round(float(np.max(fpss)) if fpss else 0.0, 2),
        "completion_rate": round(complete_count / len(frame_logs), 4) if frame_logs else 0.0,
        "missing_angle_to_target": round(max(0.0, TARGET_ANGLE - (float(np.max(angles)) if angles else 0.0)), 2),
        "pain_before": None,
        "pain_after": None
    }

    payload = {
        "patient_name": PATIENT_NAME,
        "exercise_code": EXERCISE_CODE,
        "model_name": runner.model_name,
        "target_angle": TARGET_ANGLE,
        "summary": summary,
        "frames": frame_logs
    }

    os.makedirs("experiment_logs", exist_ok=True)
    out_path = os.path.join(
        "experiment_logs",
        f"{PATIENT_NAME}_{EXERCISE_CODE}_{runner.model_name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Kaydedildi: {out_path}")


if __name__ == "__main__":
    main()