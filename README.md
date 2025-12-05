# From Wireless to Wellness: Transfer Learning with WaveFM for Sleep Apnea Detection

This repository contains the code for our ENEL 645 project:

> **From Wireless to Wellness: Transfer Learning with WaveFM for Sleep Apnea Detection**

The project evaluates whether **WaveFM**, a foundation model pretrained on wireless communication signals, can be transferred to **healthcare radar sensing** for **obstructive sleep apnea (OSA)** detection using the **APNIWAVE** UWB radar dataset. We compare four adaptation strategies:

1. Training from scratch  
2. Patch-embed + head fine-tuning (frozen encoder)  
3. Full model fine-tuning  
4. Parameter-efficient fine-tuning with **LoRA** :contentReference[oaicite:0]{index=0}  

The experiments, methodology, and results are described in the accompanying project report PDF in this repo. :contentReference[oaicite:1]{index=1}  

---

## Contents

- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Data format](#data-format)
- [Running experiments](#running-experiments)
- [Fine-tuning modes](#fine-tuning-modes)
- [Reproducing paper results](#reproducing-paper-results)
- [Citation](#citation)

---

## Repository structure

### Core files

#### `config.py`
- Loads global configuration from `config.yml` using **pydantic**.
- Defines:
  - `SpectrogramConfig`: STFT parameters (`sample_rate`, `window_size`, `window_overlap`).
  - `WandBConfig`: Weights & Biases project name and API key (read from `WANDB_KEY`).
  - `Config`: top-level config including spectrogram settings, wandb config, and `checkpoint_dir`.
- Exposes a single global object: `CONFIG`, used across the codebase.

#### `dataset.py`
- Implements the **`SleepApneaDataset`** class (PyTorch `Dataset`) for APNIWAVE radar signals.
- For each sample:
  - Computes a spectrogram via `scipy.signal.spectrogram` using parameters from `CONFIG.spectrogram`.
  - Optionally applies a TorchVision transform pipeline (e.g., log-dB, resize, normalization).
  - Returns `(spectrogram, label)` as tensors.
- `get_statistics()`:
  - Computes dataset-wide mean and standard deviation **after** applying the chosen transform.
  - Used to build normalization transforms for training/validation.

#### `lora.py`
- Contains a lightweight **LoRA** implementation for ViT:
  - `LoRALayer`: low-rank adapter parameterized by rank `r` and scaling factor `α`.
  - `LinearWithLoRA`: wraps an existing `nn.Linear` layer and adds the LoRA residual.
  - `create_lora_model(model, lora_rank, lora_alpha)`:
    - Iterates over all ViT blocks.
    - Replaces attention and MLP linear layers (`qkv`, `proj`, `fc1`, `fc2`) with `LinearWithLoRA`.
- Used when `finetuning_method="lora"` to enable parameter-efficient fine-tuning.

#### `model.py`
- Defines the model architecture and LightningModule used for training and evaluation.

Key components:

- **`VisionTransformer`** (subclass of `timm.models.vision_transformer.VisionTransformer`)
  - Adds flexible classification head (optional MLP stack) on top of the ViT embedding.
  - Supports:
    - `global_pool` selection.
    - `unfreeze_patch_embed()` to train the patch embedding.
    - `freeze_encoder()` to freeze all or part of the encoder blocks.
    - `freeze_encoder_lora()` to freeze base weights but unfreeze only LoRA parameters.

- **Factory functions**:
  - `vit_small_patch16(...)`
  - `vit_medium_patch16(...)`
  - `vit_large_patch16(...)`  
  Each creates a ViT with different embed dimensions, depth, and head counts, all with `in_chans=5` for 5-channel spectrogram input.

- **`SleepApneaModel` (LightningModule)**
  - Wraps a ViT backbone and implements:
    - Forward pass.
    - `training_step` and `validation_step` with cross-entropy loss and accuracy metrics (`torchmetrics`).
    - `configure_optimizers()` using `AdamW` + `ReduceLROnPlateau`.
  - `_setup_fintuneing(...)` (note typo in name) configures one of four modes:
    - `"scratch"` – randomly initialized, no pretrained weights.
    - `"head"` – load pretrained checkpoint, freeze encoder, train patch-embed + head only.
    - `"full"` – load pretrained checkpoint, fine-tune all weights.
    - `"lora"` – wrap encoder with LoRA adapters, freeze base weights, train adapters + head.

#### `main.py`
- CLI entry point implemented with **Typer**.
- Handles:
  - Loading CSV data and labels (`--data-path`, `--labels-path`).
  - Reshaping raw radar time series into segments and channels.
  - Train/validation split with stratification.
  - Building **transform pipelines**:
    - log-dB conversion.
    - Resize to 224×224.
    - Dataset-specific standardization.
    - Optional time and frequency masking augmentation.
  - Constructing `SleepApneaDataset` and PyTorch `DataLoader`s.
  - Initializing **Weights & Biases** logging and **Lightning Trainer** with:
    - Checkpointing on `val_acc`.
    - Early stopping on `val_acc`.
    - LR monitor and rich progress bar.
  - Instantiating `SleepApneaModel` with user-specified:
    - `vit_size` (`small`, `medium`, `large`)
    - `finetuning_method` (`scratch`, `head`, `full`, `lora`)
    - LoRA `rank`/`alpha`, learning rate, label smoothing, etc.
- Command is registered as `app.main` and executed when running `python main.py`.

### Auxiliary / implied files

- **`config.yml`**  
  YAML configuration file loaded by `config.py`. Example keys:
  ```yaml
  spectrogram:
    sample_rate: 20
    window_size: 256
    window_overlap: 128

  wandb:
    project_name: wavefm-sleep-apnea
    WANDB_KEY: "YOUR_WANDB_KEY_HERE"

  checkpoint_dir: "checkpoints/"
