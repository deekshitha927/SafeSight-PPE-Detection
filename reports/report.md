# Person & PPE Detection — Evaluation Report
## 1. Approach
Two independently trained YOLOv8 detectors: (1) a single-class person detector trained on full images, and (2) a 5-class PPE detector trained only on cropped person regions produced from the same source annotations. At inference time, model 1 finds people in the full frame, each detected person is cropped (with padding matching training-time cropping), model 2 runs on that crop, and its detections are remapped back into full-image coordinates before being drawn.
## 2. Dataset & Class Selection
416 source images / PascalVOC XML annotations. Raw instance counts were audited before training:

| Class | Instances | Decision |
|---|---|---|
| person | 1284 | used for person model |
| hard-hat | 972 | kept |
| boots | 895 | kept |
| gloves | 447 | kept |
| ppe-suit | 240 | kept |
| vest | 194 | kept |
| mask | 8 | dropped — too few instances |
| glasses | 0 | dropped — not in dataset |
| ear-protector | 0 | dropped — not in dataset |
| safety-harness | 0 | dropped — not in dataset |

5 PPE classes remain: hard-hat, gloves, boots, vest, ppe-suit — meeting the "at least 5 classes" requirement with classes the data can actually support.
## 3. Cropping & Coordinate-Remap Logic
Each person box is padded by 8% before cropping. Each PPE box is assigned to the person crop with the largest intersection-over-PPE-area; boxes with no overlapping person are dropped. After clipping to the crop, a PPE box is discarded if less than 40% of its original area survives, to avoid training on truncated labels. Person crops smaller than 32px on a side are skipped entirely. The same padding/clipping geometry is reused unchanged in inference.py so crop-local and full-image coordinates always agree.
## 4. Training Configuration
- Person model: `yolov8s.pt`, imgsz 640, 150 epochs
- PPE model: `yolov8s.pt`, imgsz 512, 200 epochs
- Class imbalance (vest/ppe-suit minority classes) handled via oversampling in the cropped training split rather than loss re-weighting.
## 5. Evaluation Metrics (held-out test split)
### Person detector
| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.859 | 0.653 | 0.875 | 0.820 |
### PPE detector — overall
| mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|
| 0.532 | 0.372 | 0.604 | 0.491 |
### PPE detector — per class
| Class | AP50 | AP50-95 |
|---|---|---|
| hard-hat | 0.862 | 0.700 |
| gloves | 0.287 | 0.224 |
| boots | 0.655 | 0.394 |
| vest | 0.506 | 0.309 |
| ppe-suit | 0.351 | 0.235 |

## 6. Limitations & Learnings
- Dataset is small (416 source images); PPE model benefits from the crop-based approach since it turns each person instance into its own training example, but absolute data volume for minority classes (vest, ppe-suit) remains limited.
- Crowded scenes (images with 10+ people) yield small, low-resolution person crops; PPE items on these can be harder to detect even after upscaling.
- The intersection-over-PPE-area assignment heuristic can misassign items in extreme overlap cases (multiple people standing very close together); no such failures were observed as systematic in this dataset's scale but would need revisiting on denser crowd data.
