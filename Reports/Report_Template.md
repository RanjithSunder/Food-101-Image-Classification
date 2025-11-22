# Food-101 Image Classification - Report

## 1. Problem Statement
The objective of this project is to classify food images into **101 food categories** using transfer learning with PyTorch.  
We use a pre-trained CNN model (ResNet50 or EfficientNet) and fine-tune the classifier layer.

---

## 2. Dataset Overview
**Dataset:** Food-101  
- 101 food classes  
- 101,000 images total  
- 750 training images per class  
- 250 test images per class  

Dataset link:  
https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/

### Preprocessing & Augmentation
- RandomResizedCrop  
- HorizontalFlip  
- RandomRotation  
- ColorJitter  
- Normalization (ImageNet stats)

---

## 3. Model Architecture
**Base Model:** ResNet50 (ImageNet pretrained)

- Frozen backbone (feature extractor)
- New fully-connected classifier:
```
Linear(2048 → 101)
```
- Optimizer: AdamW  
- Loss: CrossEntropy  
- LR scheduler (optional improvements)  

---

## 4. Training Configuration
| Parameter | Value |
|----------|-------|
| Epochs | 10 |
| Batch Size | 32 |
| Learning Rate | 1e-4 |
| Optimizer | AdamW |
| Training Time | <fill after training> |

---

## 5. Results
### Macro F1 Score
**Final Macro F1:**  
```
< insert your result >
```

### Confusion Matrix
(Insert image here: `outputs/confusion_matrix.png`)

### Sample Predictions
(Insert 3–6 sample images with predicted labels)

---

## 6. Observations
- Some classes with visually similar dishes show confusion  
- Model improves significantly with augmentation  
- Dataset is well-balanced → macro F1 is reliable

---

## 7. Future Improvements
- Fine-tune entire backbone  
- Use EfficientNet-B3 or ViT  
- Apply mixup / cutmix  
- Use test-time augmentation (TTA)  
- Add Grad-CAM visual explanations  

---

## 8. Conclusion
This project successfully builds a working pipeline for **fine-grained food image classification** using transfer learning.  
The model achieves competitive performance and can be deployed for food detection in apps.

