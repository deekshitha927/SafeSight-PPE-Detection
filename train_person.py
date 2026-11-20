"""Train the full-image person detector. See PROJECT_STEPS.md, step 2."""
import argparse
import os
import shutil

from ultralytics import YOLO

from utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--data", default="configs/data_person.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    tcfg = cfg["training"]["person"]

    model = YOLO(tcfg["model"])
    model.train(
        data=args.data,
        imgsz=tcfg["imgsz"],
        epochs=tcfg["epochs"],
        batch=tcfg["batch"],
        patience=tcfg["patience"],
        project=os.path.abspath("runs"),
        name="person_det",
        exist_ok=True,
    )

    best = os.path.join("runs", "person_det", "weights", "best.pt")
    dst = cfg["paths"]["person_weights"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(best, dst)
    print(f"copied {best} -> {dst}")


if __name__ == "__main__":
    main()
