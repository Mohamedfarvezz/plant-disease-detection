
# Plant Disease Detection using Deep Learning (CNN, ResNet50, DenseNet121 & Grad-CAM)

This project implements a deep learning pipeline to classify plant diseases using leaf images from the PlantVillage dataset(https://github.com/spMohanty/PlantVillage-Dataset). It implements a baseline CNN model and enhances it with advanced augmentation and K-Fold cross-validation, then introduces transfer learning using ResNet50 and DenseNet121. Interpretability is added using Grad-CAM to visualize what regions influence the model's decisions.

---

## Research Question

> **"How can deep learning models be optimized through transfer learning and interpretability techniques to improve plant disease classification in real-world agricultural applications?"**

---

## Objectives

- Build a baseline CNN to classify diseases on PlantVillage.
- Improve generalization with **advanced augmentation** and **Stratified K-Fold validation**.
- Use **transfer learning** with **ResNet50** and **DenseNet121**.
- Apply **Grad-CAM** for explainability and trust.
- Evaluate performance using accuracy, F1-score, and confusion matrix.

---

##  Project Structure

| File                          | Description |
|-------------------------------|-------------|
| `baseline_cnn`               | Early baseline model with basic augmentation |
| `CNN_baseline_final`         | Final version of the baseline CNN |
| `CNN_kfold_n_aug`           | CNN using **advanced augmentation** and **K-Fold** cross-validation |
| `ResNet50_model`             | Transfer learning with ResNet50 (fixed image size & pooling layer) |
| `DenseNet121_80_20_split`    | DenseNet121 using 80/20 split – best model version |
| `DenseNet121_Kfold`          | DenseNet121 with 3-fold cross-validation |
| `Grad-CAM`                   | Grad-CAM visualisation of correct/wrong predictions |

---

## Methodology

- Used **colored** PlantVillage images for training.
- Resized images to **128x128** for CNN and **224x224** for transfer learning models.
- Applied **advanced augmentations** (rotation, color jitter, noise, blur, etc.).
- Implemented **custom K-Fold generator** for Stratified K-Fold validation.
- Applied Transfer learning with **ResNet50** and **DenseNet121**.
- Integrated **Grad-CAM** to highlight regions influencing predictions.

---

## Dataset

- **Source**: [PlantVillage GitHub Repository](https://github.com/spMohanty/PlantVillage-Dataset)
- **Total Images**: 52,306
- **Classes**: 38 (including healthy/diseased across 14 crops)

---

## 🛠️ Requirements

Install the following packages (via pip):

```bash
pip install tensorflow scikit-learn opencv-python matplotlib seaborn pillow
```

---

## How to Run

Each model is written as a separate script. Example:

```bash
python DenseNet121_80_20_split.py
```

Make sure your dataset is organized in folders by class inside a root directory (e.g., `PlantVillage/color`).

---

## Folder Setup

```
plant-disease-detection/
│
├── CNN_baseline_final
├── CNN_kfold_n_aug
├── DenseNet121_80_20_split
├── DenseNet121_Kfold
├── ResNet50_model
├── Grad-CAM
├── baseline_cnn
└── README.md
```

Run each script seperately and get accuracy and classification metrics.

For grad_CAM, Use the saved model to visualise correctly classdified and misclassified images.

---

## Citation

If you use this code, please cite:

- Mohanty, S.P., Hughes, D.P. & Salathé, M. (2016). Using Deep Learning for Image-Based Plant Disease Detection. *Frontiers in Plant Science*.
---

## Author

 **Mohamed Farvez Noordeen Badrudeen**  
Dissertation project at Oxford Brookes University  
**Project Title**: *Plant Disease Detection using Deep Learning*
