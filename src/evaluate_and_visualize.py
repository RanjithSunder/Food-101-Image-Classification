"""
evaluate_and_visualize.py

Loads outputs/best_model.pth, runs evaluation on test set (ImageFolder test/)
- computes macro F1 and writes outputs/f1.txt
- saves confusion matrix (outputs/confusion_matrix.png)
- saves sample_predictions (outputs/sample_predictions.png)

Usage:
    python src/evaluate_and_visualize.py --data-root data/food-101 --num-samples 8
"""

import os
import argparse
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib.pyplot as plt
import random
from PIL import Image

from model import get_model


def evaluate_and_save(model, dataloader, device, classes):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(y.numpy())
    f1 = f1_score(gts, preds, average="macro")
    return f1, gts, preds


def save_confusion_matrix(gts, preds, classes, out_path="outputs/confusion_matrix.png"):
    cm = confusion_matrix(gts, preds)
    plt.figure(figsize=(18, 16))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.title("Confusion matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved confusion matrix to", out_path)


def save_sample_predictions(
    model,
    data_root,
    classes,
    device,
    num_samples=8,
    out_path="outputs/sample_predictions.png",
):
    # collect image paths from test set
    all_paths = []
    for cls in classes:
        folder = os.path.join(data_root, "test", cls)
        if os.path.exists(folder):
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith(".jpg")
            ]
            all_paths.extend(files)
    if len(all_paths) == 0:
        print("No test images found to sample.")
        return

    samples = random.sample(all_paths, min(num_samples, len(all_paths)))
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    cols = min(4, len(samples))
    rows = (len(samples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()
    for i, p in enumerate(samples):
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            pred = int(out.argmax(1).item())
        axes[i].imshow(img)
        axes[i].set_title(f"Pred: {classes[pred]}")
        axes[i].axis("off")
    # hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print("Saved sample predictions to", out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/food-101")
    parser.add_argument("--checkpoint", type=str, default="outputs/best_model.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-samples", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load classes
    cls_path = os.path.join("outputs", "classes.txt")
    if os.path.exists(cls_path):
        with open(cls_path) as f:
            classes = [l.strip() for l in f]
    else:
        # fallback - read from ImageFolder
        ds = ImageFolder(os.path.join(args.data_root, "test"))
        classes = ds.classes

    # load model
    model = get_model(
        "resnet50", num_classes=len(classes), pretrained=True, fine_tune=False
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)

    val_ds = ImageFolder(
        os.path.join(args.data_root, "test"),
        transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    f1, gts, preds = evaluate_and_save(model, val_loader, device, classes)

    # write f1
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/f1.txt", "w") as f:
        f.write(f"{f1:.6f}")
    print("Macro F1:", f1)

    # save confusion matrix and sample predictions
    save_confusion_matrix(gts, preds, classes, out_path="outputs/confusion_matrix.png")
    save_sample_predictions(
        model,
        args.data_root,
        classes,
        device,
        num_samples=args.num_samples,
        out_path="outputs/sample_predictions.png",
    )


if __name__ == "__main__":
    main()
