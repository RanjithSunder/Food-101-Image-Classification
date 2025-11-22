"""
train_food101.py

Main training script for Food-101 classification using Transfer Learning (ResNet50).

Example usage:
    python src/train_food101.py --epochs 10 --batch-size 32 --lr 1e-4
"""

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
import time
import os

from model import get_model
from utils import get_dataloaders, evaluate


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    # Load dataset
    train_loader, val_loader, classes = get_dataloaders(
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    # Load model
    model = get_model(
        name=args.model,
        num_classes=len(classes),
        pretrained=True,
        fine_tune=False,  # freeze backbone
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best_f1 = 0.0

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Evaluate on validation set
        f1, gts, preds = evaluate(model, val_loader, device=device)
        duration = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss:.4f} | "
            f"Val F1: {f1:.4f} | Time: {duration:.1f}s"
        )

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), "outputs/best_model.pth")
            print("✔ Best model updated!")

    print("\nTraining Completed!")
    print("Best Validation Macro F1 Score:", best_f1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Food-101 classifier")

    parser.add_argument(
        "--data-root",
        type=str,
        default=r"C:\Ranjith\Exercise\DS Project\Final_Project\Food101_Project\data\food-101",
        help="Path to Food-101 dataset root (after extraction)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--model", type=str, default="resnet50", help="Model name: resnet50 (default)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
