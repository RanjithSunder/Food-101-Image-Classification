"""
dataset.py

Usage:
    python src/dataset.py --root data/food-101

This script converts the Food-101 folder structure into
a PyTorch ImageFolder-compatible format:

Creates:
data/food-101/train/<class>/*.jpg
data/food-101/test/<class>/*.jpg
"""

import os
import shutil
import argparse


def create_imagefolder(
    root_dir=r"C:\Ranjith\Exercise\DS Project\Final_Project\Food101_Project\data\food-101",
):
    images_dir = os.path.join(root_dir, "images")
    meta_dir = os.path.join(root_dir, "meta")

    # Validate folders
    if not os.path.exists(images_dir) or not os.path.exists(meta_dir):
        raise FileNotFoundError(
            f"Expected folders 'images/' and 'meta/' inside {root_dir}"
        )

    # Read train/test splits
    with open(os.path.join(meta_dir, "train.txt"), "r") as f:
        train_list = f.read().splitlines()

    with open(os.path.join(meta_dir, "test.txt"), "r") as f:
        test_list = f.read().splitlines()

    # Create ImageFolder type structure
    for split, split_list in [("train", train_list), ("test", test_list)]:
        print(f"Processing {split} set...")
        for item in split_list:
            cls, img_name = item.split("/")
            src_img_path = os.path.join(images_dir, cls, img_name + ".jpg")
            dst_cls_dir = os.path.join(root_dir, split, cls)

            os.makedirs(dst_cls_dir, exist_ok=True)
            dst_img_path = os.path.join(dst_cls_dir, img_name + ".jpg")

            # Avoid overwriting if file exists
            if not os.path.exists(dst_img_path):
                shutil.copyfile(src_img_path, dst_img_path)

    print("ImageFolder dataset created successfully!")
    print("Train directory:", os.path.join(root_dir, "train"))
    print("Test directory:", os.path.join(root_dir, "test"))


def main():
    parser = argparse.ArgumentParser(
        description="Convert Food-101 dataset to ImageFolder format"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=r"C:\Ranjith\Exercise\DS Project\Final_Project\Food101_Project\data\food-101",
        help="Path to dataset root",
    )
    args = parser.parse_args()

    create_imagefolder(args.root)


if __name__ == "__main__":
    main()
