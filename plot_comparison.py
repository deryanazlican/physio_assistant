import json
import matplotlib.pyplot as plt
import numpy as np
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract(data):
    s = data["summary"]
    return {
        "model": data["model_name"],
        "fps": s["avg_fps"],
        "angle": s["max_angle"],
        "std": s["std_angle"]
    }


def main(paths):
    datas = [extract(load_json(p)) for p in paths]

    models = [d["model"] for d in datas]
    fps = [d["fps"] for d in datas]
    angle = [d["angle"] for d in datas]
    std = [d["std"] for d in datas]

    x = np.arange(len(models))

    # -------- FPS --------
    plt.figure()
    plt.bar(models, fps)
    plt.title("FPS Comparison")
    plt.ylabel("FPS")
    plt.savefig("fps.png", dpi=300)

    # -------- Angle --------
    plt.figure()
    plt.bar(models, angle)
    plt.title("Max Angle Comparison")
    plt.ylabel("Angle")
    plt.savefig("angle.png", dpi=300)

    # -------- Stability --------
    plt.figure()
    plt.bar(models, std)
    plt.title("Stability (Std Dev)")
    plt.ylabel("Std Dev")
    plt.savefig("stability.png", dpi=300)

    plt.show()


if __name__ == "__main__":
    main(sys.argv[1:])