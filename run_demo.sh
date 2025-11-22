#!/bin/bash
set -e

echo "Step 1: Preparing ImageFolder format (if not done earlier)..."
python src/dataset.py --root data/food-101

echo "Step 2: Starting training (example run)..."
python src/train_food101.py --epochs 5 --batch-size 32 --lr 1e-4

echo "Step 3: Running inference demo..."
python src/inference.py --checkpoint outputs/best_model.pth --image demo/sample.jpg
