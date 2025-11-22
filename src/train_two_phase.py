"""
train_two_phase.py

Two-phase training pipeline for Food-101:
 - Phase 1: Freeze backbone, train classifier (warm-up)
 - Phase 2: Unfreeze (fine-tune) full network with LR scheduler
Features:
 - Mixed precision training (torch.cuda.amp)
 - CosineAnnealingLR for phase2
 - Early stopping on val macro-F1
 - Save best model to outputs/best_model.pth
 - Save outputs/classes.txt and outputs/training_log.csv
Usage:
    python src/train_two_phase.py --data-root /path/to/data/food-101 --phase1-epochs 8 --phase2-epochs 20
"""

import os
import time
import argparse
import csv
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Use existing get_transforms from utils for consistency
from utils import get_transforms, evaluate

# import model factory
from model import get_model


def save_classes(root, out_path="outputs/classes.txt"):
    ds = ImageFolder(os.path.join(root, "train"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        for c in ds.classes:
            f.write(c + "\n")
    print("Saved classes to", out_path)
    return ds.classes


def get_loaders(root, batch_size, num_workers, image_size):
    train_tfms, val_tfms = get_transforms(image_size)
    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_tfms)
    val_ds = ImageFolder(os.path.join(root, "test"), transform=val_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes


def train_phase(
    model, train_loader, optimizer, criterion, device, scaler, epoch, log_every=50
):
    model.train()
    running_loss = 0.0
    seen = 0
    pbar = tqdm(
        enumerate(train_loader), total=len(train_loader), desc=f"Train Epoch {epoch+1}"
    )
    for i, (imgs, labels) in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device != "cpu")):
            out = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        seen += bs
        if (i + 1) % log_every == 0:
            pbar.set_postfix(loss=running_loss / seen)
    avg_loss = running_loss / seen
    return avg_loss


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Prepare outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Save class names
    classes = save_classes(args.data_root, out_path="outputs/classes.txt")

    # Dataloaders
    train_loader, val_loader, _ = get_loaders(
        args.data_root, args.batch_size, args.num_workers, args.image_size
    )

    # Build model (initially frozen in phase1)
    model = get_model(
        name=args.model, num_classes=len(classes), pretrained=True, fine_tune=False
    )
    model = model.to(device)

    # Phase 1: freeze backbone (get only classifier params)
    print("Phase 1: training classifier only (backbone frozen)")
    criterion = nn.CrossEntropyLoss()
    # Ensure only parameters with requires_grad True are optimized
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=args.phase1_lr, weight_decay=args.weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

    best_f1 = 0.0
    history = []

    # Training loop for phase1
    for epoch in range(args.phase1_epochs):
        t0 = time.time()
        loss = train_phase(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        f1, _, _ = evaluate(model, val_loader, device=device)
        t = time.time() - t0
        print(
            f"Phase1 Epoch {epoch+1}/{args.phase1_epochs} | loss {loss:.4f} | val_f1 {f1:.4f} | time {t:.1f}s"
        )
        history.append(("phase1", epoch + 1, loss, f1))

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("Saved best model (phase1) with f1", best_f1)

    # Phase 2: unfreeze backbone and fine-tune
    print("\nPhase 2: fine-tuning full model (unfreeze backbone)")
    for p in model.parameters():
        p.requires_grad = True

    # Recreate optimizer for all parameters
    optimizer = AdamW(
        model.parameters(), lr=args.phase2_lr, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device != "cpu"))

    # Early stopping
    stale = 0
    best_f1_phase2 = best_f1
    for epoch in range(args.phase2_epochs):
        t0 = time.time()
        loss = train_phase(
            model, train_loader, optimizer, criterion, device, scaler, epoch
        )
        f1, gts, preds = evaluate(model, val_loader, device=device)
        scheduler.step()
        t = time.time() - t0
        print(
            f"Phase2 Epoch {epoch+1}/{args.phase2_epochs} | loss {loss:.4f} | val_f1 {f1:.4f} | time {t:.1f}s | lr {scheduler.get_last_lr()[0]:.2e}"
        )
        history.append(("phase2", epoch + 1, loss, f1))

        # Save training log CSV progressively
        with open("outputs/training_log.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["phase", "epoch", "loss", "val_macro_f1"])
            for rec in history:
                writer.writerow(rec)

        if f1 > best_f1_phase2:
            best_f1_phase2 = f1
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("Saved best model (phase2) with f1", best_f1_phase2)
            stale = 0
        else:
            stale += 1
            print("Stale epochs:", stale)

        # early stopping
        if stale >= args.patience:
            print("Early stopping triggered (patience reached).")
            break

    # Final save of classes.txt (already saved) and f1
    with open("outputs/f1.txt", "w") as f:
        f.write(f"{best_f1_phase2:.6f}")

    print("Training complete. Best val macro F1:", best_f1_phase2)
    print(
        "Artifacts in outputs/: best_model.pth, training_log.csv, f1.txt, classes.txt"
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="data/food-101")
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--phase1-epochs", type=int, default=8)
    p.add_argument("--phase2-epochs", type=int, default=20)
    p.add_argument("--phase1-lr", type=float, default=1e-4)
    p.add_argument("--phase2-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
