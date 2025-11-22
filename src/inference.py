"""
inference.py

Runs prediction on a single image using the trained Food-101 model.

Example:
    python src/inference.py --checkpoint outputs/best_model.pth --image myfood.jpg
"""

import argparse
import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import os
import json


def load_image(path, image_size=224):
    """
    Loads and preprocesses a single image.
    """
    img = Image.open(path).convert("RGB")

    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )

    return tfm(img).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to best_model.pth"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to input food image"
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--model", type=str, default="resnet50")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = get_model(
        name=args.model, num_classes=101, pretrained=True, fine_tune=False
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    # Load image
    img_tensor = load_image(args.image, args.image_size).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_index = output.argmax(1).item()

    print("\n============================")
    print(" Predicted Class Index:", pred_index)
    print("============================\n")

    print("Note: If you want class names instead of index,")
    print("load dataset classes using ImageFolder and map index → label.")


if __name__ == "__main__":
    main()
