## Lung Cancer Histopathology Classification

Deep learning classifier for lung cancer histopathology images (LC25000 dataset). Achieves 99.9% validation accuracy using ResNet50 transfer learning.

## Results

| Model  | Accuracy | AUC (ACA) | AUC (SCC) |
| ------------- | ------------- | ------------- | ------------- |
| Control (5 batches)  | 77.0%  | 0.85 | 0.98 |
| Final (10 epochs)  | 99.9%  | 1.00 | 1.00 |

Grad-CAM visualizations show the final model focuses on relevant histopathological features:

<img width="987" height="471" alt="image" src="https://github.com/user-attachments/assets/d4148187-7b63-484a-8496-b844182fe4e9" />


## Installation

```bash
git clone https://github.com/yourusername/lung-cancer-classification.git
cd lung-cancer-classification
pip install -r requirements.txt
```

**Dataset:** Download LC25000 to data/lung_colon_image_set/lung_image_sets/

## Usage

### Train

```bash
python scripts/train.py --config config/config.yaml --epochs 10
```

### Evaluate

```bash
python scripts/evaluate.py --config config/config.yaml
```

### Grad-CAM visualization

```bash
python scripts/visualize.py --config config/config.yaml
```

## Features

- Transfer learning with ResNet50/EfficientNet-B0
- Stratified Sequential Block Split to prevent Identity Leakage
- Focal Loss for class imbalance handling
- Grad-CAM explainability for model interpretability
- Data augmentation (flips, rotations, color jitter)

## Citation

**Borkowski, A. A., Bui, M. M., Thomas, L. B., Wilson, C. P., DeLand, L. A., & Mastorides, S. M. (2019). Lung and Colon Cancer Histopathological Image Dataset (LC25000). *arXiv preprint arXiv:1912.12142*.**
