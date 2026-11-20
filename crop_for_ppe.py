"""Crop each person out of full images and build the PPE-detection dataset.

For every person box in a source image, crop the image (with padding) and
translate every PPE box that belongs to that person into crop-local YOLO
coordinates. A full image with N people yields N cropped training images.
Reuses the split assignment written by voc_to_yolo.py so crops of the same
source image never leak across train/val/test. See PROJECT_STEPS.md, step 3.
"""
import argparse
import os

import yaml
from PIL import Image

from utils.config import load_config
from utils.geometry import (
    box_area,
    clip_box,
    intersection_over_area,
    pad_box,
    to_crop_local,
    xyxy_to_yolo,
)
from utils.splits import load_split_map
from utils.voc import list_annotations, parse_voc_xml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--xml-dir", default=None, help="overrides paths.raw_labels")
    p.add_argument("--img-dir", default=None, help="overrides paths.raw_images")
    p.add_argument("--out-dir", default=None, help="overrides paths.ppe_dataset")
    p.add_argument("--pad", type=float, default=None, help="overrides cropping.pad_frac")
    return p.parse_args()


def assign_ppe_to_persons(person_boxes, ppe_objects):
    """For each ppe object, pick the person box with highest intersection-over-PPE-area.

    Returns {person_idx: [(class_name, box), ...]}. Boxes with zero overlap
    with any person (max IoA == 0) are dropped rather than force-assigned.
    """
    assignments = {i: [] for i in range(len(person_boxes))}
    for name, box in ppe_objects:
        best_idx, best_ioa = None, 0.0
        for i, person_box in enumerate(person_boxes):
            ioa = intersection_over_area(box, person_box)
            if ioa > best_ioa:
                best_idx, best_ioa = i, ioa
        if best_idx is not None:
            assignments[best_idx].append((name, box))
    return assignments


def main():
    args = parse_args()
    cfg = load_config(args.config)

    xml_dir = args.xml_dir or cfg["paths"]["raw_labels"]
    img_dir = args.img_dir or cfg["paths"]["raw_images"]
    out_dir = args.out_dir or cfg["paths"]["ppe_dataset"]
    pad_frac = args.pad if args.pad is not None else cfg["cropping"]["pad_frac"]
    min_crop_px = cfg["cropping"]["min_crop_px"]
    min_retained_area = cfg["cropping"]["min_retained_area"]

    ppe_classes = cfg["ppe_classes"]
    class_to_id = {name: i for i, name in enumerate(ppe_classes)}

    split_map = load_split_map(cfg["paths"]["split_map"])

    for split in ("train", "val", "test"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

    annotations = list_annotations(xml_dir)

    n_crops = 0
    n_skipped_small_person = 0
    n_ppe_kept = 0
    n_ppe_truncated_dropped = 0
    n_ppe_unassigned = 0
    per_class_counts = {name: 0 for name in ppe_classes}

    for stem, xml_path in annotations:
        if stem not in split_map:
            continue
        split = split_map[stem]

        image_path, w, h, objects = parse_voc_xml(xml_path, img_dir)
        person_boxes = [box for name, box in objects if name == "person"]
        if not person_boxes:
            continue

        ppe_objects = [(name, box) for name, box in objects if name in class_to_id]
        assignments = assign_ppe_to_persons(person_boxes, ppe_objects)
        n_ppe_unassigned += len(ppe_objects) - sum(len(v) for v in assignments.values())

        image = Image.open(image_path).convert("RGB")

        for i, person_box in enumerate(person_boxes):
            padded = pad_box(person_box, pad_frac, w, h)
            crop_xmin, crop_ymin = round(padded[0]), round(padded[1])
            crop_xmax, crop_ymax = round(padded[2]), round(padded[3])
            crop_w, crop_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
            if crop_w < min_crop_px or crop_h < min_crop_px:
                n_skipped_small_person += 1
                continue
            crop_box_int = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)

            label_lines = []
            for name, box in assignments[i]:
                clipped = clip_box(box, crop_box_int)
                if clipped is None:
                    continue
                retained = box_area(clipped) / box_area(box) if box_area(box) > 0 else 0
                if retained < min_retained_area:
                    n_ppe_truncated_dropped += 1
                    continue
                local_box = to_crop_local(clipped, crop_box_int)
                xc, yc, bw, bh = xyxy_to_yolo(local_box, crop_w, crop_h)
                label_lines.append(f"{class_to_id[name]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                per_class_counts[name] += 1
                n_ppe_kept += 1

            crop_name = f"{stem}_p{i}"
            crop_img = image.crop(crop_box_int)
            crop_img.save(os.path.join(out_dir, "images", split, crop_name + ".jpg"))
            with open(os.path.join(out_dir, "labels", split, crop_name + ".txt"), "w") as f:
                f.write("\n".join(label_lines))
                if label_lines:
                    f.write("\n")
            n_crops += 1

        image.close()

    data_yaml = {
        "path": os.path.abspath(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(ppe_classes)},
    }
    data_yaml_path = os.path.join(os.path.dirname(cfg["paths"]["split_map"]), "data_ppe.yaml")
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)

    print(f"ppe dataset written to {out_dir}")
    print(f"person crops created: {n_crops} (skipped {n_skipped_small_person} undersized persons)")
    print(f"ppe boxes kept: {n_ppe_kept}, dropped after truncation: {n_ppe_truncated_dropped}, "
          f"unassigned (no overlapping person): {n_ppe_unassigned}")
    print(f"per-class counts: {per_class_counts}")
    print(f"wrote {data_yaml_path}")


if __name__ == "__main__":
    main()
