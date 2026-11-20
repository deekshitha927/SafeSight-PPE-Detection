"""Convert PascalVOC annotations to YOLOv8 format and build the person-detection dataset.

Also writes configs/split_map.json (source-image -> train/val/test), which
crop_for_ppe.py reuses so PPE crops never leak across splits. See
PROJECT_STEPS.md, step 1.
"""
import argparse
import os
import shutil

import yaml

from utils.config import load_config
from utils.geometry import xyxy_to_yolo
from utils.splits import build_image_split, save_split_map
from utils.voc import list_annotations, parse_voc_xml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--xml-dir", default=None, help="overrides paths.raw_labels")
    p.add_argument("--img-dir", default=None, help="overrides paths.raw_images")
    p.add_argument("--out-dir", default=None, help="overrides paths.person_dataset")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    xml_dir = args.xml_dir or cfg["paths"]["raw_labels"]
    img_dir = args.img_dir or cfg["paths"]["raw_images"]
    out_dir = args.out_dir or cfg["paths"]["person_dataset"]
    split_map_path = cfg["paths"]["split_map"]
    ratios = tuple(cfg["split"]["ratios"])
    seed = cfg["split"]["seed"]

    annotations = list_annotations(xml_dir)
    print(f"found {len(annotations)} annotation files in {xml_dir}")

    parsed = {}  # stem -> (image_path, w, h, objects)
    person_counts = {}
    n_missing_images = 0
    for stem, xml_path in annotations:
        try:
            image_path, w, h, objects = parse_voc_xml(xml_path, img_dir)
        except FileNotFoundError as e:
            n_missing_images += 1
            print(f"  skipping {stem}: {e}")
            continue
        parsed[stem] = (image_path, w, h, objects)
        person_counts[stem] = sum(1 for name, _ in objects if name == "person")

    if n_missing_images:
        print(f"skipped {n_missing_images} annotations with no matching image")

    image_ids = sorted(parsed.keys())
    split_map = build_image_split(image_ids, person_counts, ratios=ratios, seed=seed)
    os.makedirs(os.path.dirname(split_map_path), exist_ok=True)
    save_split_map(split_map, split_map_path)

    split_counts = {"train": 0, "val": 0, "test": 0}
    zero_person_images = 0
    total_person_boxes = 0

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    for stem in image_ids:
        image_path, w, h, objects = parsed[stem]
        split = split_map[stem]
        split_counts[split] += 1

        person_boxes = [box for name, box in objects if name == "person"]
        total_person_boxes += len(person_boxes)
        if not person_boxes:
            zero_person_images += 1

        ext = os.path.splitext(image_path)[1]
        dst_image = os.path.join(out_dir, "images", split, stem + ext)
        shutil.copyfile(image_path, dst_image)

        label_path = os.path.join(out_dir, "labels", split, stem + ".txt")
        lines = []
        for box in person_boxes:
            xc, yc, bw, bh = xyxy_to_yolo(box, w, h)
            lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

    data_yaml = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person"},
    }
    data_yaml_path = os.path.join(os.path.dirname(split_map_path), "data_person.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"person dataset written to {out_dir}")
    print(f"split sizes: {split_counts}")
    print(f"total person boxes: {total_person_boxes}, images with 0 persons: {zero_person_images}")
    print(f"wrote {data_yaml_path}")


if __name__ == "__main__":
    main()
