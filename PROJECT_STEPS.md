# Two-Stage PPE Detection Pipeline — Build Plan

## Data audit (from datasets/labels/*.xml, 416 images/XMLs)

```
person          1284  -> used for person model only
hard-hat         972  -> KEEP
boots            895  -> KEEP
gloves           447  -> KEEP
ppe-suit         240  -> KEEP
vest             194  -> KEEP
mask               8  -> DROP (too few instances)
glasses            0  -> DROP (not in dataset)
ear-protector      0  -> DROP (not in dataset)
safety-harness     0  -> DROP (not in dataset)
```

Result: PPE model trained on 5 classes (hard-hat, gloves, boots, vest, ppe-suit) — satisfies "at least 5 classes" requirement.

## Step 1 — voc_to_yolo.py

- Parse each XML, read width/height from `<size>` (fallback to actual image if missing/wrong).
- Build person-only YOLO dataset: single class (person, id 0), all images kept including the 7 with zero persons (negatives).
- Apply class filter/remap above for later PPE use (ids 0-4).
- Split 80/10/10 train/val/test, roughly stratified by person-count.
- Output: `datasets/person_det/{images,labels}/{train,val,test}` + `data_person.yaml`

## Step 2 — Train person detector

```
yolo detect train model=yolov8n.pt \
  data=datasets/person_det/data_person.yaml \
  imgsz=640 epochs=100 batch=16 patience=20 \
  project=runs name=person_det
```

## Step 3 — crop_for_ppe.py (crop + label remap logic)

- For each person box: crop image to that box + small padding, clipped to image bounds.
- For each PPE box: assign to the person crop where it has the largest intersection-over-PPE-area (handles overlapping people).
- Translate assigned box into crop-local coords, clip, re-normalize to YOLO xc/yc/w/h.
- Write one image+label per person: `001003_p0.jpg`, `001003_p1.jpg` ... Persons with no PPE still get a crop with empty label.
- Apply 5-class filter/remap.
- Output: `datasets/ppe_det/{images,labels}/{train,val,test}` + `data_ppe.yaml`

```
python crop_for_ppe.py --xml-dir datasets/labels \
  --img-dir datasets/images --out-dir datasets/ppe_det --pad 0.08
```

## Step 4 — Train PPE detector

- yolov8n/s, 5 classes, on datasets/ppe_det.
- Imbalance fix: oversample train images containing vest/ppe-suit (minority classes) rather than loss re-weighting.

```
yolo detect train model=yolov8n.pt \
  data=datasets/ppe_det/data_ppe.yaml \
  imgsz=512 epochs=150 batch=16 patience=25 \
  project=runs name=ppe_det
```

## Step 5 — inference.py (two-stage, image dir in -> annotated dir out)

- Run person model on full image -> person boxes.
- For each person box: crop with SAME padding convention as step 3, run PPE model on that crop.
- Remap: add crop origin offset to each PPE box to place it back in full-image coordinates (must mirror step 3 geometry exactly).
- Draw with cv2.rectangle + cv2.putText only (no YOLO plot()): one color for person, per-class colors for PPE, label+confidence text.
- Save annotated image to output dir.

```
python inference.py --input-dir path/to/images \
  --output-dir path/to/output \
  --person-weights runs/person_det/weights/best.pt \
  --ppe-weights runs/ppe_det/weights/best.pt
```

## Step 6 — Evaluation + report

- `model.val()` on both models: mAP50, mAP50-95, per-class P/R, confusion matrix.
- `report.md`: approach, dataset stats, class drop rationale, crop/remap logic, training config, metrics, limitations.
- Convert report.md -> report.pdf.

## Assumptions

- 5-class PPE model (mask + 3 zero-instance classes dropped)
- 80/10/10 split
- yolov8n as starting backbone for both models
- oversampling (not loss re-weighting) for class imbalance
