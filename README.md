# Robust CIFAR-10 classification with 500 training images

Small Inception-style CNN (about 0.24M parameters, trained from scratch) that must work on a shifted CIFAR-10 test set (rotation, brightness/contrast, blur, noise, occlusion and a hidden colour corruption) after seeing only 500 clean images.

## Repository contents
- `cifar10_robust.ipynb` - full Section B notebook (data, model, baseline, augmentation, ablation, error analysis, TTA)
- `ablation_table.csv` - produced by the notebook (Part D)
- `README.md` - this file

## Setup that follows the assignment rules
- No pretrained weights; at most 1,000,000 trainable parameters (asserted in the notebook).
- Inception-style module: 1x1, 3x3 and 5x5 branches with 1x1 reductions, plus a pooling branch. Global Average Pooling replaces large FC layers.
- 500 images (50 per class, seed 42) split 400 train / 100 validation; all hyperparameters chosen on validation only.
- Augmentation is applied on the fly to training data only; no augmented copies are saved.
- 60 epochs maximum, fixed seeds (0, 1, 2); results reported as mean ± std.

## How to reproduce
**Kaggle (recommended)**
1. Create a new notebook, upload `cifar10_robust.ipynb`.
2. Settings: Accelerator = GPU (T4 is enough), Internet = On (CIFAR-10 is downloaded by torchvision).
3. Run All. Expect roughly 20-40 minutes.

**Local**
```bash
pip install torch torchvision matplotlib scikit-learn pandas jupyter
jupyter notebook cifar10_robust.ipynb   # then Run All
```

## Results
Fill in from the executed notebook:

| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation | | | |
| Geometric only | | | |
| Photometric only | | | |
| Geometric + photometric | | | |
| + Cutout / Random Erasing | | | |
| + Mixup + label smoothing | | | |
| Final model | | | |

## Note on the hidden corruption
The assignment does not state the hidden corruption. The notebook uses a stand-in (random grayscale / channel swap) to build its own shifted test set, and also evaluates on unseen corruptions (invert, solarize, posterize, hue shift) to check generalisation honestly.
