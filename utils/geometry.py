"""Bounding-box math shared by crop_for_ppe.py and inference.py.

Both scripts must apply the exact same crop/pad/clip/remap geometry —
this module is the single source of truth for that so the two never drift.
Boxes are always (xmin, ymin, xmax, ymax) in absolute pixel coordinates
unless noted otherwise.
"""


def pad_box(box, pad_frac, img_w, img_h):
    """Expand box by pad_frac of its own width/height, clipped to image bounds."""
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    pad_x, pad_y = w * pad_frac, h * pad_frac
    return (
        max(0.0, xmin - pad_x),
        max(0.0, ymin - pad_y),
        min(float(img_w), xmax + pad_x),
        min(float(img_h), ymax + pad_y),
    )


def box_area(box):
    xmin, ymin, xmax, ymax = box
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def intersection(box_a, box_b):
    xmin = max(box_a[0], box_b[0])
    ymin = max(box_a[1], box_b[1])
    xmax = min(box_a[2], box_b[2])
    ymax = min(box_a[3], box_b[3])
    if xmax <= xmin or ymax <= ymin:
        return None
    return (xmin, ymin, xmax, ymax)


def intersection_over_area(box_a, box_b):
    """area(box_a ∩ box_b) / area(box_a) — used to assign a PPE box to a person."""
    inter = intersection(box_a, box_b)
    area_a = box_area(box_a)
    if inter is None or area_a <= 0:
        return 0.0
    return box_area(inter) / area_a


def clip_box(box, crop_box):
    """Clip box to crop_box, both in the same absolute frame. None if no overlap."""
    return intersection(box, crop_box)


def to_crop_local(box, crop_box):
    """Translate an absolute-image box into coordinates local to crop_box's origin."""
    ox, oy = crop_box[0], crop_box[1]
    xmin, ymin, xmax, ymax = box
    return (xmin - ox, ymin - oy, xmax - ox, ymax - oy)


def to_full_image(box, crop_box):
    """Inverse of to_crop_local — used by inference.py to remap PPE detections."""
    ox, oy = crop_box[0], crop_box[1]
    xmin, ymin, xmax, ymax = box
    return (xmin + ox, ymin + oy, xmax + ox, ymax + oy)


def xyxy_to_yolo(box, img_w, img_h):
    """(xmin, ymin, xmax, ymax) -> normalized (xc, yc, w, h)."""
    xmin, ymin, xmax, ymax = box
    w, h = xmax - xmin, ymax - ymin
    xc, yc = xmin + w / 2, ymin + h / 2
    return (xc / img_w, yc / img_h, w / img_w, h / img_h)


def yolo_to_xyxy(box, img_w, img_h):
    """Normalized (xc, yc, w, h) -> (xmin, ymin, xmax, ymax)."""
    xc, yc, w, h = box
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
