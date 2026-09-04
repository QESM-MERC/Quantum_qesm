# QESM

<p align="right">
  <strong>English</strong> | <a href="README_zh.md">简体中文</a>
</p>

**A Quantum-Inspired Network with Dynamic Fusion and Entangled Measurement for Multimodal Emotion Recognition**

[![Task](https://img.shields.io/badge/task-multimodal%20emotion%20recognition-5c6ac4)](https://github.com/QESM-MERC/Quantum_qesm)
[![Framework](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Tests](https://github.com/QESM-MERC/Quantum_qesm/actions/workflows/ci.yml/badge.svg)](https://github.com/QESM-MERC/Quantum_qesm/actions/workflows/ci.yml)

> [!IMPORTANT]
> This initial release publishes the project overview first. Source code, frozen configurations,
> raw-feature checkpoints, publication metadata, and the open-source license are being prepared
> for the subsequent release.

QESM is a classical PyTorch model inspired by quantum mathematical structures; it does not require quantum hardware. It represents text, audio, and visual utterance features as complex states, evolves them over dialogue context, models cross-modal perturbations, and performs a joint emotion measurement.

![QESM architecture](overview.png)

## Architecture

The processing order used in the experiments is:

```text
Quantum State Preparation -> UTE_1 -> OCS -> UTE_2 -> Density-Mixture Fusion -> EBM
```

- **Unitary Temporal Evolution (UTE)** propagates speaker-aware context through norm-preserving phase evolution.
- **OTOC Cross-Modal Scrambling (OCS)** models dynamic cross-modal complementarity through an out-of-time-order-correlator-inspired interaction.
- **Entangled Born Measurement (EBM)** combines cross-modal phase coherence with trimodal class agreement for emotion classification.

## Results

QESM is evaluated with weighted F1 (WF1) on IEMOCAP and MELD.

| Dataset | Utterances | Classes | Representative WF1 | Five-run WF1 |
|---|---:|---:|---:|---:|
| IEMOCAP | 7,433 | 6 | **73.23** | **72.82 ± 0.23** |
| MELD | 13,708 | 7 | **67.31** | **67.17 ± 0.11** |

The five-run statistics use seeds `{0, 1, 2, 3, 6}` and report the sample standard deviation. Exact configuration snapshots will be added before publication.

## Repository layout

```text
qotoc/          Core complex-valued QESM modules and model
configs/        Versioned JSON examples for the training CLI
dataset.py      IEMOCAP/MELD feature loading and split construction
train.py        Training and evaluation entry point
evaluate.py     Raw CFN-ESA checkpoint inference entry point
tests/          Numerical and model-integration tests
```

The internal Python package retains the historical `qotoc` name. A checkpoint must still match
the saved architecture, preprocessing protocol, and feature dimensions; inference validates these
requirements strictly.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data preparation

The default path reads the original combined CFN-ESA feature pickle directly. No converted
M3Net files or replacement embeddings are needed:

```text
data/cfn_esa/
├── iemocap_multimodal_features.pkl
└── meld_multimodal_features.pkl
```

The loader accepts the CFN-ESA IEMOCAP 12-field schema and MELD 13/14-field schemas. It uses
the four 1024-dimensional RoBERTa layers, the original acoustic features (1582 for IEMOCAP,
300 for MELD), and the original 342-dimensional visual features. External text, audio, or
visual replacements are rejected in this mode. The datasets and cached features are not
redistributed. Python pickle files can execute code while loading; use only files from a
source you trust.

## Training

```bash
python train.py --dataset iemocap --cfn-pkl data/cfn_esa/iemocap_multimodal_features.pkl --epochs 80
python train.py --dataset meld --cfn-pkl data/cfn_esa/meld_multimodal_features.pkl --epochs 40 --batch-size 32
python train.py --config configs/iemocap_example.json --seed 1
```

Explicit command-line options override values loaded from `--config`. The checked-in JSON files are runnable examples, not yet the frozen configurations behind the headline table.

Session-wise IEMOCAP LOSO evaluation is also supported. For example:

```bash
python train.py --dataset iemocap --data-dir data/m3net \
  --loso-test-session 5 --loso-internal-valid-frac 0.1
```

Run outputs, checkpoints, reports, and metrics are written under `results/<run-name>/`.

In raw CFN-ESA mode, `--merge-valid` uses the complete train pool stored in the combined pickle.

## Checkpoint inference

Evaluate a checkpoint trained on the original CFN-ESA feature dimensions with only the
combined pickle, the checkpoint, and its saved config (a training JSON or `metrics.json`).
The checked-in `*_cfnesa_raw.json` files capture the two verified raw-checkpoint architectures:

```bash
python evaluate.py \
  --checkpoint results/iemocap_raw/best_test.pt \
  --config configs/iemocap_cfnesa_raw.json \
  --cfn-pkl data/cfn_esa/iemocap_multimodal_features.pkl
```

The evaluator loads weights strictly and checks every modality dimension before inference.
A checkpoint trained with augmented or replacement features is rejected rather than padded,
truncated, or silently adapted. It reports test utterance count, weighted F1, macro F1, and
accuracy.

## Tests

```bash
python -m pytest -q
```

The tests cover the closed-form OTOC kernel, unitary phase rotation, complex-state normalization,
Born probabilities, padding invariance, finite backward gradients, both CFN-ESA schemas,
raw-only enforcement, checkpoint compatibility, and end-to-end inference.

## Citation

The manuscript is under review. Publication metadata and a BibTeX entry will be added when available. Until then, please link to:

```text
https://github.com/QESM-MERC/Quantum_qesm
```

## License

License selection is pending. Until a `LICENSE` file is added, all rights remain with the authors and the repository is not yet an open-source distribution.
