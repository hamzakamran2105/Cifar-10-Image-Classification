# Robust CIFAR-10 Classification with 500 Training Images

A small Inception-style CNN (242,258 trainable parameters, trained from scratch) that is trained on only 500 clean CIFAR-10 images and tested on a shifted test set (rotation, brightness/contrast change, blur, noise, occlusion and a hidden colour corruption).

> **GitHub repository:** `<paste your repository link here>`

## Repository contents
| File | Description |
|---|---|
| `cifar10_robust.ipynb` | Executed Section B notebook with all outputs (data, model, baseline, augmentation, ablation, error analysis, TTA) |
| `ablation_table.csv` | Ablation table produced by the notebook (Part D) |
| `README.md` | This file |
| `summary.docx` | One-page summary (final score, configuration, most important change) |

## How the assignment rules are met
| Rule | How |
|---|---|
| No pretrained weights | All models are trained from scratch |
| At most 1,000,000 parameters | 242,258 parameters (asserted in the notebook) |
| Inception-style module | 3 Inception modules: 1x1, 3x3 and 5x5 branches with 1x1 reductions, plus a pooling branch. Global Average Pooling replaces large FC layers |
| No tuning on test data | 500 images split 400 train / 100 validation (stratified); all hyperparameters chosen on validation only |
| On-the-fly augmentation, training data only | Transforms run inside the dataset `__getitem__`; no augmented copies are saved |
| Max 60 epochs, fixed seeds, 3 runs | 60 epochs, seeds 0, 1, 2, results reported as mean ± std |

## How to reproduce
**Kaggle (recommended)**
1. Create a new notebook and upload `cifar10_robust.ipynb`.
2. Settings: Accelerator = GPU (T4 is enough), Internet = On (torchvision downloads CIFAR-10).
3. Click Run All. It takes roughly 20-40 minutes.

**Local**
```bash
pip install torch torchvision matplotlib scikit-learn pandas jupyter
jupyter notebook cifar10_robust.ipynb   # then Run All
```
Everything is seeded (`set_seed`), so reruns on the same hardware reproduce the numbers up to small GPU non-determinism.

## Method summary
- **Data:** 50 images per class from the CIFAR-10 training set (seed 42), split 40/10 per class into 400 train and 100 validation. Test sets: the 10,000 clean test images and a shifted copy built once with a fixed seed.
- **Model:** conv stem, then 3 Inception modules with max-pooling between them, Global Average Pooling, Dropout, one linear layer. BatchNorm after every convolution.
- **Augmentation (final model):** random crop with padding, horizontal flip (no vertical flip), affine (rotation up to 30 degrees, translation, scale), colour jitter, random grayscale, channel shuffle, blur, noise, Cutout / Random Erasing, Mixup (alpha 0.4) with label smoothing 0.1.
- **Training:** AdamW (lr 2e-3, weight decay 5e-4), one-cycle learning-rate schedule, batch size 64, Dropout 0.4 only before the classifier, best-validation checkpoint (early stopping), 60 epochs.
- **Selection:** dropout, weight decay and Mixup strength were chosen from a small grid on the validation split only (seed 0).

## Results (mean ± std over 3 seeds, accuracy in %)

### Part A: baseline (no augmentation)
Train accuracy reaches about 100% while validation stays near 48%, so the network memorises the 400 images (overfitting). Shifted-test accuracy is far below clean-test accuracy, which is covariate shift on top of the overfitting.

### Part D: ablation
| Configuration | Val Acc | Clean Test | Shifted Test |
|---|---|---|---|
| No augmentation | 49.3 ± 2.9 | 44.1 ± 2.8 | 16.4 ± 0.2 |
| Geometric only | 53.0 ± 1.4 | 45.1 ± 1.1 | 17.2 ± 0.4 |
| Photometric only | 42.0 ± 2.9 | 37.8 ± 1.3 | 17.2 ± 0.9 |
| Geometric + photometric | 48.3 ± 1.9 | 42.7 ± 1.6 | **20.7 ± 0.2** |
| + Cutout / Random Erasing | 48.0 ± 2.4 | 43.5 ± 0.6 | 20.6 ± 0.7 |
| + Mixup + label smoothing | 47.3 ± 2.1 | 41.4 ± 1.3 | 20.3 ± 0.8 |
| **Final model** | 46.0 ± 1.4 | 41.0 ± 1.3 | 19.7 ± 0.9 |

**Final score:** shifted test **19.69 ± 0.88**, clean test 40.99 ± 1.34, validation 46.00 ± 1.41, up from 16.4 on the shifted test with no augmentation (+3.3 points).

**Single change that helped most:** combining geometric and photometric augmentation (+3.5 points over the best single family, 17.2 to 20.7). Each family alone added only about +0.8 points. The shifted test applies rotation, brightness/contrast change, blur and noise to the same image, so both kinds of invariance are needed together.

Cutout (-0.15) and Mixup with label smoothing (-0.29) did not help. With the combined augmentations, training accuracy is only 52-64%, so the small model is already underfitting. The differences among the last four rows are within about one standard deviation. Clean accuracy falls as augmentation grows (44.1 to 41.0), the usual robustness versus clean-accuracy trade-off.

The Final model scored slightly below "Geometric + photometric". It is reported as the final model because it was chosen on validation only; picking a row by test accuracy would break the no-tuning-on-test rule.

### Part E: error analysis (final model, shifted test)
Per-class accuracy: ship 67.9, bird 35.1, airplane 24.3, cat 16.6, truck 15.1, automobile 14.8, horse 6.3, deer 5.6, frog 5.1, dog 3.7.

Most confused pairs (true to predicted, fraction of the true class): truck to ship 0.54, airplane to ship 0.50, automobile to ship 0.49, deer to bird 0.36, bird to ship 0.34, frog to bird 0.32.

Errors are dominated by a collapse onto a few default classes (ship, bird) rather than classic fine-grained confusion; cat/dog are also confused (dog to cat 0.19). The eight most confident errors are all airplane or truck images predicted as ship (confidence 0.72-0.87). Occlusion hides the wings, tail or cab, blur and noise remove detail, grayscale removes colour, and what remains is a dark horizontal shape on a bright smooth background, which the model reads as a ship.

**Hidden-corruption test** (clean test with only the corruption applied):
| Test | No augmentation | Final model |
|---|---|---|
| Clean test (reference) | 44.1 ± 2.8 | 41.0 ± 1.3 |
| Hidden stand-in only (gray / channel swap) | 35.8 ± 2.2 | 32.7 ± 1.3 |
| Unseen: colour invert | 11.6 ± 0.8 | 10.6 ± 2.2 |
| Unseen: solarize | 24.2 ± 1.0 | 24.5 ± 2.4 |
| Unseen: posterize (3 bits) | 41.7 ± 2.4 | 40.0 ± 1.5 |
| Unseen: hue shift | 30.1 ± 1.6 | 26.4 ± 2.1 |

The augmentation did not generalise to colour-type corruptions: the final model's drop from clean accuracy is about the same as the baseline's in every row. It helped on the combined shifted test but not on colour changes. Likely reasons are underfitting and the low probability of the colour augmentations.

### Part F (bonus): test-time augmentation
| TTA | Shifted accuracy | Time for 10k images |
|---|---|---|
| None (N=1) | 19.69 ± 0.88 | 0.48 s |
| Flip (N=2) | 19.70 ± 1.09 | 0.96 s |
| Flip + 4 shifts (N=10) | 19.74 ± 1.08 | 4.56 s |

TTA gave no real gain (+0.05 points at best, inside the standard deviation) at 2x to 9.5x the inference cost. Flips and small shifts are already covered by the training augmentation, and every view makes the same wrong decision on the dominant failure cases.

## Limitations and notes
- **Hidden corruption:** the assignment does not say what it is. This notebook builds its own shifted test set with a stand-in (random grayscale or channel swap), so absolute shifted-test numbers will differ from the official hidden test set. The unseen-corruption rows (invert, solarize, posterize, hue shift) are the more honest generalisation check.
- **Accuracy:** shifted-test accuracy is low (about 20%) because the model is trained from scratch on 400 images. Training accuracy under augmentation is only about 55-60%, so the model underfits; a wider network (only about 24% of the 1M-parameter budget is used), more optimisation steps per epoch, or input normalisation that removes brightness/contrast shift are the next things to try.
- **Noise:** the validation split has only 100 images, so validation differences of a few points are noise, and differences under about 3 points between ablation rows are not statistically significant with 3 seeds.
