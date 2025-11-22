"""
utils.py

Helper functions:
- Data transforms (train & validation)
- Data loaders
- Model evaluation (macro F1)
- Checkpoint saving

Used by train_food101.py
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score
import os


def get_transforms(image_size=224):
    """
    Returns train and validation transforms.
    """

    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return train_tfms, val_tfms


def get_dataloaders(root="data/food-101", batch_size=32, num_workers=4, image_size=224):
    """
    Loads train and test sets using ImageFolder.
    """

    train_tfms, val_tfms = get_transforms(image_size)

    train_ds = ImageFolder(os.path.join(root, "train"), transform=train_tfms)
    val_ds = ImageFolder(os.path.join(root, "test"), transform=val_tfms)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, train_ds.classes


def save_checkpoint(state, is_best, filename="outputs/checkpoint.pth"):
    """
    Saves model checkpoint.
    """

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

    if is_best:
        best_path = os.path.join(os.path.dirname(filename), "best_model.pth")
        torch.save(state["model_state_dict"], best_path)


def evaluate(model, dataloader, device="cuda"):
    """
    Computes macro F1 score for validation set.
    Returns (f1_score, ground_truths, predictions)
    """

    model.eval()
    preds = []
    gts = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            gts.extend(y.numpy())

    macro_f1 = f1_score(gts, preds, average="macro")
    return macro_f1, gts, preds
