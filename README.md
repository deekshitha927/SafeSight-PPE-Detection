# SafeSight — PPE Compliance Detection

Two-stage YOLOv8 pipeline for automated PPE compliance monitoring on job sites:
person detection feeding a per-crop PPE classifier (hard-hat, vest, gloves,
boots, safety suit).

See `PROJECT_STEPS.md` for the full design rationale (data audit, class
selection, cropping/remap logic). This file is just the run order.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. VOC XML -> YOLO person dataset (+ configs/split_map.json, data_person.yaml)
python voc_to_yolo.py

# 2. Train person detector (full images)
python train_person.py

# 3. Crop each person out, remap PPE boxes -> YOLO PPE dataset (data_ppe.yaml)
python crop_for_ppe.py

# 4. Train PPE detector (cropped person images)
python train_ppe.py

# 5. Run both models over a directory of images
python inference.py --input-dir path/to/images --output-dir outputs

# 6. Evaluate both models on their test splits, build reports/report.pdf
python evaluate.py
```

## Layout

```
configs/            central config.yaml + generated data_person.yaml / data_ppe.yaml / split_map.json
utils/              shared VOC parsing, bbox geometry, split assignment, config loader
datasets/           raw images/labels (VOC) + generated person_det/ and ppe_det/ (YOLO format)
models/             promoted best.pt for each model
runs/               raw Ultralytics training/val logs and plots
outputs/            inference.py annotated images
reports/            metrics.json, report.md, report.pdf, figures/
```
