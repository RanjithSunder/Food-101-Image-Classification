"""
final_report_generator.py

Generates a short (1–2 page) PDF report for the Food-101 project.

Looks for:
  - outputs/f1.txt                  (optional, macro F1 score)
  - outputs/confusion_matrix.png    (optional, confusion matrix image)
  - outputs/sample_predictions.png  (optional, sample predictions image)

Output:
  - outputs/report.pdf

Usage:
    python final_report_generator.py

Requirements:
    pip install reportlab
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Image,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


def read_f1(out_dir: str) -> str:
    """
    Read macro F1 score from outputs/f1.txt if it exists.
    Returns a string (e.g. '0.472315') or 'N/A'.
    """
    path = os.path.join(out_dir, "f1.txt")
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                value = f.read().strip()
            return value if value else "N/A"
        except Exception:
            return "N/A"
    return "N/A"


def create_pdf(out_dir: str = "outputs", out_pdf: str | None = None) -> None:
    """
    Build the PDF report and save it to outputs/report.pdf.
    """
    if out_pdf is None:
        out_pdf = os.path.join(out_dir, "report.pdf")

    os.makedirs(out_dir, exist_ok=True)

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(out_pdf, pagesize=A4)
    elems = []

    # ---- Title ----
    elems.append(Paragraph("Food-101 Image Classification Report", styles["Title"]))
    elems.append(Spacer(1, 16))

    # ---- High-level summary ----
    intro_text = (
        "This report summarizes the results of a food image classification model "
        "trained on the Food-101 dataset using transfer learning (ResNet-50). "
        "The model predicts one of 101 food categories from an input image."
    )
    elems.append(Paragraph(intro_text, styles["BodyText"]))
    elems.append(Spacer(1, 12))

    # ---- Info table ----
    f1_val = read_f1(out_dir)
    info_data = [
        ["Dataset", "Food-101 (101 classes, ~101k images)"],
        ["Base Model", "ResNet-50 (ImageNet pretrained)"],
        ["Task", "Multiclass food image classification"],
        ["Validation Metric", "Macro F1-score"],
        ["Macro F1 (validation)", f1_val],
    ]

    info_table = Table(info_data, colWidths=[160, 330])
    info_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    elems.append(info_table)
    elems.append(Spacer(1, 16))

    # ---- Optional: Confusion Matrix ----
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    if os.path.exists(cm_path):
        elems.append(Paragraph("Confusion Matrix", styles["Heading2"]))
        elems.append(Spacer(1, 6))
        try:
            # Adjust size if needed
            elems.append(Image(cm_path, width=430, height=360))
        except Exception as e:
            elems.append(
                Paragraph(
                    f"(Could not embed confusion matrix image: {e})", styles["BodyText"]
                )
            )
        elems.append(Spacer(1, 16))
    else:
        elems.append(
            Paragraph(
                "<i>No confusion_matrix.png found in outputs/ (skipping confusion matrix)</i>",
                styles["BodyText"],
            )
        )
        elems.append(Spacer(1, 12))

    # ---- Optional: Sample Predictions ----
    sp_path = os.path.join(out_dir, "sample_predictions.png")
    if os.path.exists(sp_path):
        elems.append(Paragraph("Sample Predictions", styles["Heading2"]))
        elems.append(Spacer(1, 6))
        try:
            elems.append(Image(sp_path, width=430, height=320))
        except Exception as e:
            elems.append(
                Paragraph(
                    f"(Could not embed sample_predictions image: {e})",
                    styles["BodyText"],
                )
            )
        elems.append(Spacer(1, 16))
    else:
        elems.append(
            Paragraph(
                "<i>No sample_predictions.png found in outputs/ "
                "(skipping sample prediction visualization)</i>",
                styles["BodyText"],
            )
        )
        elems.append(Spacer(1, 12))

    # ---- Short conclusion ----
    conclusion = (
        "The model demonstrates reasonable performance on the Food-101 dataset, as "
        "summarized by the macro F1-score. Further improvements could be achieved by "
        "fine-tuning deeper layers, experimenting with stronger data augmentations, "
        "trying alternative architectures (e.g., EfficientNet, Vision Transformers), "
        "and applying techniques like mixup/cutmix or test-time augmentation."
    )
    elems.append(Paragraph("Conclusion & Future Work", styles["Heading2"]))
    elems.append(Spacer(1, 6))
    elems.append(Paragraph(conclusion, styles["BodyText"]))

    # ---- Build PDF ----
    doc.build(elems)
    print(f"Report generated at: {out_pdf}")


if __name__ == "__main__":
    create_pdf()
