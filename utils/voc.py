"""PascalVOC XML parsing shared by voc_to_yolo.py and crop_for_ppe.py."""
import os
import xml.etree.ElementTree as ET

from PIL import Image


def parse_voc_xml(xml_path, img_dir):
    """Return (image_path, width, height, objects).

    objects is a list of (class_name, (xmin, ymin, xmax, ymax)) in absolute
    pixel coordinates. Falls back to reading the actual image if <size> is
    missing or looks wrong, and falls back to the xml stem if <filename>
    doesn't resolve to a real file (some exports rename inconsistently).
    """
    root = ET.parse(xml_path).getroot()

    filename = root.findtext("filename", "").strip()
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    candidates = [filename] if filename else []
    candidates += [stem + ext for ext in (".jpg", ".jpeg", ".png")]

    image_path = None
    for cand in candidates:
        cand_path = os.path.join(img_dir, cand)
        if os.path.isfile(cand_path):
            image_path = cand_path
            break
    if image_path is None:
        raise FileNotFoundError(f"no image found for {xml_path} (tried {candidates})")

    size_node = root.find("size")
    width = height = None
    if size_node is not None:
        w_text, h_text = size_node.findtext("width"), size_node.findtext("height")
        if w_text and h_text:
            width, height = int(w_text), int(h_text)

    if not width or not height:
        with Image.open(image_path) as im:
            width, height = im.size
    else:
        # sanity-check against the real file; VOC exports occasionally have stale sizes
        with Image.open(image_path) as im:
            real_w, real_h = im.size
        if (width, height) != (real_w, real_h):
            width, height = real_w, real_h

    objects = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        # clip to image bounds and drop degenerate boxes
        xmin, xmax = sorted((max(0, min(xmin, width)), max(0, min(xmax, width))))
        ymin, ymax = sorted((max(0, min(ymin, height)), max(0, min(ymax, height))))
        if xmax - xmin < 1 or ymax - ymin < 1:
            continue
        objects.append((name, (xmin, ymin, xmax, ymax)))

    return image_path, width, height, objects


def list_annotations(xml_dir):
    """Sorted list of (stem, xml_path) for every .xml in xml_dir."""
    items = []
    for fn in os.listdir(xml_dir):
        if fn.lower().endswith(".xml"):
            stem = os.path.splitext(fn)[0]
            items.append((stem, os.path.join(xml_dir, fn)))
    return sorted(items)
