"""Train the PPE detector on cropped person images. See PROJECT_STEPS.md, step 4."""
import argparse
import os
import shutil

from ultralytics import YOLO

from utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="configs/data_ppe.yaml")
    p.add_argument("--model", default=None, help="overrides training.ppe.model, e.g. yolov8s.pt")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]["ppe"]
    model_name = args.model or tcfg["model"]

    model = YOLO(model_name)
    model.train(
        data=args.data,
        imgsz=tcfg["imgsz"],
        epochs=tcfg["epochs"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        project=os.path.abspath("runs"),
        name="ppe_det",
        exist_ok=True,
    )

    best = os.path.join("runs", "ppe_det", "weights", "best.pt")
    dst = cfg["paths"]["ppe_weights"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(best, dst)
    print(f"copied {best} -> {dst}")


if __name__ == "__main__":
    main()
