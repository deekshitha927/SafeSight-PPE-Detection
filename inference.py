"""Two-stage inference: person model on full images, PPE model on person crops.

Takes an input image directory, runs both trained models, remaps PPE boxes
from crop-local back to full-image coordinates, and saves annotated images
(drawn with cv2.rectangle/cv2.putText only) to an output directory.
See PROJECT_STEPS.md, step 5.
"""
import argparse
import json
import os

import cv2
from ultralytics import YOLO

from utils.config import load_config
from utils.geometry import clip_box, pad_box, to_full_image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--person-weights", default=None)
    p.add_argument("--ppe-weights", default=None)
    p.add_argument("--save-json", action="store_true", help="also dump raw detections per image")
    return p.parse_args()


def draw_box(img, box, label, color):
    xmin, ymin, xmax, ymax = (int(round(v)) for v in box)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_top = max(ymin - th - baseline - 2, 0)
    cv2.rectangle(img, (xmin, label_top), (xmin + tw + 4, label_top + th + baseline + 2), color, -1)
    text_color = (0, 0, 0) if sum(color) > 380 else (255, 255, 255)
    cv2.putText(
        img, label, (xmin + 2, label_top + th + 1),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    person_weights = args.person_weights or cfg["paths"]["person_weights"]
    ppe_weights = args.ppe_weights or cfg["paths"]["ppe_weights"]
    pad_frac = cfg["cropping"]["pad_frac"]
    min_crop_px = cfg["cropping"]["min_crop_px"]
    person_conf = cfg["inference"]["person_conf"]
    ppe_conf = cfg["inference"]["ppe_conf"]
    colors_cfg = cfg["inference"]["colors"]
    person_color = tuple(colors_cfg.get("person", [255, 255, 255]))
    default_ppe_color = (0, 200, 255)

    os.makedirs(args.output_dir, exist_ok=True)

    person_model = YOLO(person_weights)
    ppe_model = YOLO(ppe_weights)

    image_files = sorted(
        fn for fn in os.listdir(args.input_dir) if fn.lower().endswith(IMAGE_EXTS)
    )
    print(f"found {len(image_files)} images in {args.input_dir}")

    for fn in image_files:
        path = os.path.join(args.input_dir, fn)
        img = cv2.imread(path)
        if img is None:
            print(f"  skipping unreadable image: {fn}")
            continue
        h, w = img.shape[:2]
        annotated = img.copy()
        detections = {"person": [], "ppe": []}

        person_result = person_model.predict(img, conf=person_conf, verbose=False)[0]
        for box, conf in zip(
            person_result.boxes.xyxy.tolist(), person_result.boxes.conf.tolist()
        ):
            draw_box(annotated, box, f"person {conf:.2f}", person_color)
            detections["person"].append({"box": box, "conf": conf})

            padded = pad_box(tuple(box), pad_frac, w, h)
            crop_xmin, crop_ymin = round(padded[0]), round(padded[1])
            crop_xmax, crop_ymax = round(padded[2]), round(padded[3])
            crop_w, crop_h = crop_xmax - crop_xmin, crop_ymax - crop_ymin
            if crop_w < min_crop_px or crop_h < min_crop_px:
                continue
            crop_box_int = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
            crop_img = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

            ppe_result = ppe_model.predict(crop_img, conf=ppe_conf, verbose=False)[0]
            for pbox, pconf, pcls in zip(
                ppe_result.boxes.xyxy.tolist(),
                ppe_result.boxes.conf.tolist(),
                ppe_result.boxes.cls.tolist(),
            ):
                class_name = ppe_model.names[int(pcls)]
                full_box = to_full_image(tuple(pbox), crop_box_int)
                full_box = clip_box(full_box, (0, 0, w, h))
                if full_box is None:
                    continue
                color = tuple(colors_cfg.get(class_name, default_ppe_color))
                draw_box(annotated, full_box, f"{class_name} {pconf:.2f}", color)
                detections["ppe"].append({"box": list(full_box), "conf": pconf, "class": class_name})

        out_path = os.path.join(args.output_dir, fn)
        cv2.imwrite(out_path, annotated)

        if args.save_json:
            json_path = os.path.join(args.output_dir, os.path.splitext(fn)[0] + ".json")
            with open(json_path, "w") as f:
                json.dump(detections, f, indent=2)

    print(f"annotated images written to {args.output_dir}")


if __name__ == "__main__":
    main()
