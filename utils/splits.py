"""Single source of truth for train/val/test assignment.

Split is decided once, per *source image*, and reused by both voc_to_yolo.py
(person dataset) and crop_for_ppe.py (PPE dataset) — every crop of a given
source image must land in the same split as that image, otherwise crops of
the same scene can leak across train/val/test.
"""
import json
import random


def _bucket(person_count):
    if person_count == 0:
        return "0"
    if person_count == 1:
        return "1"
    if person_count <= 3:
        return "2-3"
    if person_count <= 6:
        return "4-6"
    return "7+"


def build_image_split(image_ids, person_counts, ratios=(0.8, 0.1, 0.1), seed=0):
    """Assign each image id to 'train' / 'val' / 'test'.

    Stratified by bucketed person_counts so multi-person-heavy images aren't
    concentrated in one split. Returns {image_id: split}.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6
    rng = random.Random(seed)

    buckets = {}
    for img_id in image_ids:
        buckets.setdefault(_bucket(person_counts[img_id]), []).append(img_id)

    split_map = {}
    for bucket_ids in buckets.values():
        bucket_ids = sorted(bucket_ids)
        rng.shuffle(bucket_ids)
        n = len(bucket_ids)
        n_train = round(n * ratios[0])
        n_val = round(n * ratios[1])
        # remainder goes to test so rounding never drops an image
        for i, img_id in enumerate(bucket_ids):
            if i < n_train:
                split_map[img_id] = "train"
            elif i < n_train + n_val:
                split_map[img_id] = "val"
            else:
                split_map[img_id] = "test"
    return split_map


def load_split_map(path):
    """Read a previously written split assignment (JSON: {image_id: split})."""
    with open(path, "r") as f:
        return json.load(f)


def save_split_map(split_map, path):
    """Persist the split assignment so voc_to_yolo.py and crop_for_ppe.py agree."""
    with open(path, "w") as f:
        json.dump(split_map, f, indent=2, sort_keys=True)
