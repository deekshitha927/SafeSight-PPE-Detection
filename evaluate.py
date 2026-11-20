"""Evaluate both trained models on their held-out test splits and build the
PDF report (approach, learnings, evaluation metrics). See PROJECT_STEPS.md,
step 6.
"""
import argparse
import glob
import json
import os
import shutil

from fpdf import FPDF
from fpdf.enums import XPos, YPos
from ultralytics import YOLO

from utils.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/config.yaml")
    return p.parse_args()


def extract_overall(metrics):
    return {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def extract_per_class(metrics, names):
    rows = []
    ap_class_index = list(getattr(metrics.box, "ap_class_index", []))
    ap50 = list(getattr(metrics.box, "ap50", []))
    ap = list(getattr(metrics.box, "ap", []))
    for pos, cls_idx in enumerate(ap_class_index):
        rows.append({
            "class": names[int(cls_idx)],
            "AP50": float(ap50[pos]) if pos < len(ap50) else None,
            "AP50-95": float(ap[pos]) if pos < len(ap) else None,
        })
    return rows


def copy_figures(run_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    copied = []
    for src in sorted(glob.glob(os.path.join(run_dir, "*.png"))):
        dst = os.path.join(dest_dir, os.path.basename(src))
        shutil.copyfile(src, dst)
        copied.append(dst)
    return copied


def build_report(person_overall, ppe_overall, ppe_per_class, cfg, person_figs, ppe_figs, train_figs):
    md = []
    md.append("# Person & PPE Detection — Evaluation Report\n")

    md.append("## 1. Approach\n")
    md.append(
        "Two independently trained YOLOv8 detectors: (1) a single-class person "
        "detector trained on full images, and (2) a 5-class PPE detector trained "
        "only on cropped person regions produced from the same source annotations. "
        "At inference time, model 1 finds people in the full frame, each detected "
        "person is cropped (with padding matching training-time cropping), model 2 "
        "runs on that crop, and its detections are remapped back into full-image "
        "coordinates before being drawn.\n"
    )

    md.append("## 2. Dataset & Class Selection\n")
    md.append(
        "416 source images / PascalVOC XML annotations. Raw instance counts were "
        "audited before training:\n\n"
        "| Class | Instances | Decision |\n|---|---|---|\n"
        "| person | 1284 | used for person model |\n"
        "| hard-hat | 972 | kept |\n| boots | 895 | kept |\n"
        "| gloves | 447 | kept |\n| ppe-suit | 240 | kept |\n"
        "| vest | 194 | kept |\n| mask | 8 | dropped — too few instances |\n"
        "| glasses | 0 | dropped — not in dataset |\n"
        "| ear-protector | 0 | dropped — not in dataset |\n"
        "| safety-harness | 0 | dropped — not in dataset |\n\n"
        "5 PPE classes remain: hard-hat, gloves, boots, vest, ppe-suit — meeting the "
        "\"at least 5 classes\" requirement with classes the data can actually support.\n"
    )

    md.append("## 3. Cropping & Coordinate-Remap Logic\n")
    md.append(
        f"Each person box is padded by {cfg['cropping']['pad_frac']*100:.0f}% before "
        f"cropping. Each PPE box is assigned to the person crop with the largest "
        f"intersection-over-PPE-area; boxes with no overlapping person are dropped. "
        f"After clipping to the crop, a PPE box is discarded if less than "
        f"{cfg['cropping']['min_retained_area']*100:.0f}% of its original area survives, "
        f"to avoid training on truncated labels. Person crops smaller than "
        f"{cfg['cropping']['min_crop_px']}px on a side are skipped entirely. The same "
        f"padding/clipping geometry is reused unchanged in inference.py so crop-local "
        f"and full-image coordinates always agree.\n"
    )

    md.append("## 4. Training Configuration\n")
    md.append(
        f"- Person model: `{cfg['training']['person']['model']}`, imgsz "
        f"{cfg['training']['person']['imgsz']}, {cfg['training']['person']['epochs']} epochs\n"
        f"- PPE model: `{cfg['training']['ppe']['model']}`, imgsz "
        f"{cfg['training']['ppe']['imgsz']}, {cfg['training']['ppe']['epochs']} epochs\n"
        f"- Class imbalance (vest/ppe-suit minority classes) handled via oversampling "
        f"in the cropped training split rather than loss re-weighting.\n"
    )

    md.append("## 5. Evaluation Metrics (held-out test split)\n")
    md.append("### Person detector\n")
    md.append(
        "| mAP50 | mAP50-95 | Precision | Recall |\n|---|---|---|---|\n"
        f"| {person_overall['mAP50']:.3f} | {person_overall['mAP50-95']:.3f} | "
        f"{person_overall['precision']:.3f} | {person_overall['recall']:.3f} |\n"
    )
    md.append("### PPE detector — overall\n")
    md.append(
        "| mAP50 | mAP50-95 | Precision | Recall |\n|---|---|---|---|\n"
        f"| {ppe_overall['mAP50']:.3f} | {ppe_overall['mAP50-95']:.3f} | "
        f"{ppe_overall['precision']:.3f} | {ppe_overall['recall']:.3f} |\n"
    )
    md.append("### PPE detector — per class\n")
    md.append("| Class | AP50 | AP50-95 |\n|---|---|---|\n")
    for row in ppe_per_class:
        ap50 = f"{row['AP50']:.3f}" if row["AP50"] is not None else "n/a"
        ap = f"{row['AP50-95']:.3f}" if row["AP50-95"] is not None else "n/a"
        md.append(f"| {row['class']} | {ap50} | {ap} |\n")

    md.append("\n## 6. Limitations & Learnings\n")
    md.append(
        "- Dataset is small (416 source images); PPE model benefits from the crop-"
        "based approach since it turns each person instance into its own training "
        "example, but absolute data volume for minority classes (vest, ppe-suit) "
        "remains limited.\n"
        "- Crowded scenes (images with 10+ people) yield small, low-resolution "
        "person crops; PPE items on these can be harder to detect even after "
        "upscaling.\n"
        "- The intersection-over-PPE-area assignment heuristic can misassign items "
        "in extreme overlap cases (multiple people standing very close together); "
        "no such failures were observed as systematic in this dataset's scale but "
        "would need revisiting on denser crowd data.\n"
    )

    return "".join(md)


def _pdf_safe(text):
    replacements = {"—": "-", "–": "-", "‘": "'", "’": "'",
                    "“": '"', "”": '"', "…": "..."}
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def render_pdf(markdown_text, figure_paths, out_path):
    markdown_text = _pdf_safe(markdown_text)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)

    def mc(text, h):
        pdf.multi_cell(0, h, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    for line in markdown_text.split("\n"):
        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            mc(line[2:], 10)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("## "):
            pdf.ln(2)
            pdf.set_font("Helvetica", "B", 14)
            mc(line[3:], 8)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 12)
            mc(line[4:], 7)
            pdf.set_font("Helvetica", size=11)
        elif line.startswith("|"):
            pdf.set_font("Courier", size=9)
            mc(line, 5)
            pdf.set_font("Helvetica", size=11)
        elif line.strip().startswith("- "):
            mc("  - " + line.strip()[2:], 6)
        elif line.strip():
            mc(line, 6)
        else:
            pdf.ln(2)

    for title, paths in figure_paths.items():
        if not paths:
            continue
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        mc(_pdf_safe(title), 8)
        for path in paths:
            try:
                pdf.image(path, w=170)
                pdf.ln(4)
            except RuntimeError:
                continue

    pdf.output(out_path)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    person_model = YOLO(cfg["paths"]["person_weights"])
    ppe_model = YOLO(cfg["paths"]["ppe_weights"])

    runs_dir = os.path.abspath("runs")
    person_results = person_model.val(
        data="configs/data_person.yaml", split="test",
        project=runs_dir, name="person_det_val", exist_ok=True,
    )
    ppe_results = ppe_model.val(
        data="configs/data_ppe.yaml", split="test",
        project=runs_dir, name="ppe_det_val", exist_ok=True,
    )

    person_overall = extract_overall(person_results)
    ppe_overall = extract_overall(ppe_results)
    ppe_per_class = extract_per_class(ppe_results, ppe_model.names)

    os.makedirs(cfg["paths"]["reports"], exist_ok=True)
    metrics = {
        "person": person_overall,
        "ppe_overall": ppe_overall,
        "ppe_per_class": ppe_per_class,
    }
    metrics_path = os.path.join(cfg["paths"]["reports"], "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"wrote {metrics_path}")
    print(json.dumps(metrics, indent=2))

    figures_dir = os.path.join(cfg["paths"]["reports"], "figures")
    person_figs = copy_figures("runs/person_det_val", os.path.join(figures_dir, "person"))
    ppe_figs = copy_figures("runs/ppe_det_val", os.path.join(figures_dir, "ppe"))
    train_figs = copy_figures("runs/person_det", os.path.join(figures_dir, "person_train")) + \
        copy_figures("runs/ppe_det", os.path.join(figures_dir, "ppe_train"))

    report_md = build_report(person_overall, ppe_overall, ppe_per_class, cfg,
                              person_figs, ppe_figs, train_figs)
    report_md_path = os.path.join(cfg["paths"]["reports"], "report.md")
    with open(report_md_path, "w") as f:
        f.write(report_md)
    print(f"wrote {report_md_path}")

    render_pdf(
        report_md,
        {
            "Training curves — person model": [p for p in train_figs if "person_train" in p and "results" in p],
            "Training curves — PPE model": [p for p in train_figs if "ppe_train" in p and "results" in p],
            "Person model — validation figures": person_figs,
            "PPE model — validation figures": ppe_figs,
        },
        os.path.join(cfg["paths"]["reports"], "report.pdf"),
    )
    print(f"wrote {os.path.join(cfg['paths']['reports'], 'report.pdf')}")


if __name__ == "__main__":
    main()
