# SDANet: Spectral Disentanglement Adversarial Network for Cross-Center Ultrasound Cancer Diagnosis

> **Spectral Disentanglement Adversarial Network for Robust Cross-Center Ultrasound Cancer Diagnosis**
> *Frontiers in Medical Technology*
> Yongju Tian, Lu Huang, Jiqing Xuan, Peng Li, Qi Han, Dongjing Shan

---

## Overview

SDANet addresses the cross-center domain shift problem in multi-center ultrasound cancer diagnosis through principled **frequency-domain decomposition** combined with **dual-stream adversarial training**.

The core insight is that, in the Fourier representation of an image, the **amplitude spectrum** encodes low-level domain-specific style (scanner texture, contrast), while the **phase spectrum** preserves high-level semantic structure (tissue boundaries, lesion morphology). SDANet exploits this decomposition to explicitly separate and suppress domain-specific features without discarding pathologically informative content.

### Architecture

```
Input image x [B, 3, H, W]
        │
  FrequencyDecomposer (2D FFT)
   ├── log_amplitude  [B, 3, H, W]   ──►  AmplitudeEncoder (ResNet-50)
   │                                           │
   │                                        GRL (α-scheduled)
   │                                           │
   │                                    DomainDiscriminator  ──► L_adv
   │
   └── phase_img      [B, 3, H, W]   ──►  PhaseEncoder (Swin-T)

        amp_feat ◄──── ConsistencyLoss (cosine) ────► phase_feat
                                    │
                             Concat [B, 2D]
                                    │
                            ClassificationHead  ──► L_cls

  L_total = L_cls + λ_adv · L_adv + λ_cons · L_cons
```

**Key components:**
- **FrequencyDecomposer** — parameter-free 2D FFT splitting each image into log-amplitude and phase-reconstructed streams
- **AmplitudeEncoder** — ResNet-50 backbone with projection head; trained adversarially via a Gradient Reversal Layer (GRL) with progressive λ-scheduling to produce center-invariant representations
- **PhaseEncoder** — Swin-Transformer-Tiny backbone capturing long-range structural semantics
- **DomainDiscriminator** — multi-layer MLP predicting source center identity; drives the adversarial objective
- **Cross-stream consistency loss** — cosine similarity between amplitude and phase features, preventing the adversarial objective from discarding pathologically informative amplitude content

---

## Results

Evaluated on two source centers (The First Affiliated Hospital of Chongqing Medical University; Jincheng People's Hospital) and an unseen three-hospital target cohort.

### Cross-Center Generalization

| Method | Accuracy | AUC | F1 | Sensitivity | Specificity |
|---|---|---|---|---|---|
| Swin-T | 0.5084 | 0.4816 | 0.5556 | 0.6322 | 0.3913 |
| DANN | 0.6145 | 0.4783 | **0.7013** | **0.9310** | 0.3152 |
| **SDANet (ours)** | **0.6872** | **0.7090** | 0.6744 | 0.6667 | **0.7065** |

### Ablation Study

| Variant | Freq. | Dual | Adv. | Cons. | Accuracy | AUC | F1 | Sens. | Spec. |
|---|:---:|:---:|:---:|:---:|---|---|---|---|---|
| A1: SDANet (full) | ✓ | ✓ | ✓ | ✓ | 0.6872 | **0.7090** | **0.6744** | **0.6667** | 0.7065 |
| A2: w/o Consistency | ✓ | ✓ | ✓ | | **0.7151** | 0.7143 | 0.6577 | 0.5632 | **0.8587** |
| A3: w/o Adversarial | ✓ | ✓ | | ✓ | 0.6089 | 0.6316 | 0.5395 | 0.4713 | 0.7391 |
| A4: w/o Freq. Decomp. | | ✓ | ✓ | ✓ | 0.3352 | 0.3140 | 0.3568 | 0.3793 | 0.2935 |
| A5: Phase-Only† | ✓ | | | | 0.5140 | 0.7276 | 0.0000 | 0.0000 | 1.0000 |
| A6: Amplitude-Only | ✓ | | ✓ | | 0.6704 | 0.6810 | 0.6194 | 0.5517 | 0.7826 |

† A5 degenerates to predicting all samples as Normal under class imbalance.

---

## Requirements

```
Python >= 3.8
torch >= 2.0.0
torchvision >= 0.15.0
timm >= 0.9.0
numpy >= 1.24.0
scikit-learn >= 1.2.0
Pillow >= 9.0.0
tqdm >= 4.65.0
matplotlib >= 3.7.0
```

Install all dependencies:

```bash
pip install -r requirements.txt
```

> **Note on pretrained weights:** `timm` downloads Swin-Transformer and ResNet-50 ImageNet weights on first use. On servers without internet access, pre-download the weights on a connected machine and copy the HuggingFace cache (`~/.cache/huggingface/hub`) to the server before training.

---

## Data Preparation

Organize each clinical center as a directory with `cancer/` and `normal/` subfolders:

```
data/
├── chongqing_c/          # Source center 1 (cancer cases only)
│   ├── cancer/
│   │   ├── img_001.png
│   │   └── ...
│   └── normal/           # May be empty; handled automatically
├── jincheng_nc/          # Source center 2
│   ├── cancer/
│   └── normal/
└── sw_2zibo/             # Target cohort (test only, not used in training)
    ├── cancer/
    └── normal/
```

Update the paths in `config.py`:

```python
center_dirs: List[str] = [
    "/path/to/data/chongqing_c",
    "/path/to/data/jincheng_nc",
]
target_dir: str = "/path/to/data/sw_2zibo"
output_dir: str = "/path/to/outputs"
```

Images are resized to `224×224` and normalized with ImageNet statistics. Training augmentations include random horizontal/vertical flips, rotation (±15°), and color jitter.

---

## Training

### Full model training

```bash
python train.py
```

Override any hyperparameter from the command line:

```bash
python train.py --epochs 100 --batch_size 16 --lambda_adv 1.0 --lambda_cons 0.5
```

Resume from a checkpoint:

```bash
python train.py --resume outputs/checkpoints/best_model.pth
```

Key hyperparameters (see `config.py` for full list):

| Parameter | Default | Description |
|---|---|---|
| `num_epochs` | 100 | Maximum training epochs |
| `batch_size` | 16 | Batch size |
| `lr_encoder` | 1e-4 | Encoder learning rate (fine-tuning) |
| `lr_head` | 1e-3 | Classification head / discriminator LR |
| `lambda_adv` | 1.0 | Adversarial loss weight |
| `lambda_cons` | 0.5 | Consistency loss weight |
| `grl_gamma` | 10.0 | GRL λ-scheduling coefficient |
| `warmup_epochs` | 5 | Linear LR warmup epochs |

Training uses cosine annealing after warmup and early stopping (patience=15, metric=val_AUC).

### Ablation study

Run all six ablation variants sequentially:

```bash
python ablation_train.py
```

Run a single variant:

```bash
python ablation_train.py --variant full       # Full SDANet (baseline)
python ablation_train.py --variant no_cons    # Remove consistency loss
python ablation_train.py --variant no_adv     # Remove adversarial training
python ablation_train.py --variant no_freq    # Remove FFT decomposition
python ablation_train.py --variant phase_only # Phase stream only
python ablation_train.py --variant amp_only   # Amplitude stream only
```

Results are saved to `outputs/ablation_results.csv`.

---

## Evaluation

Evaluate the best checkpoint on the held-out target cohort:

```bash
python test.py
```

Reported metrics: Accuracy, AUC (primary), F1, Sensitivity, Specificity.

---

## Project Structure

```
SDANet/
├── config.py               # Global configuration (paths, hyperparameters)
├── train.py                # Main training script
├── test.py                 # Evaluation on target cohort
├── ablation_train.py       # Ablation study runner
├── requirements.txt
├── models/
│   ├── dann.py             # FreqDANN / SDANet model (ablation variants supported)
│   ├── encoders.py         # PhaseEncoder (Swin-T) and AmplitudeEncoder (ResNet-50)
│   ├── frequency.py        # FrequencyDecomposer (FFT amplitude–phase split)
│   ├── grl.py              # Gradient Reversal Layer with λ-scheduling
│   └── discriminator.py    # Multi-domain discriminator
├── datasets/
│   └── dataset.py          # MedicalImageDataset, build_dataloaders, transforms
├── losses/
│   └── losses.py           # TotalLoss (cls + adv + consistency)
├── utils/
│   └── utils.py            # EarlyStopping, MetricTracker, checkpointing, etc.
└── outputs/
    ├── checkpoints/        # Saved model weights
    └── logs/               # Training curves
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{tian2026sdanet,
  title   = {Spectral Disentanglement Adversarial Network for Robust Cross-Center Ultrasound Cancer Diagnosis},
  author  = {Tian, Yongju and Huang, Lu and Xuan, Jiqing and Li, Peng and Han, Qi and Shan, Dongjing},
  journal = {in Medical Technology},
  year    = {2026}
}
```

---

## License

This project is released under the MIT License.
