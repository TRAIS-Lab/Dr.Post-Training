# SFT Experiments

This folder contains the training code and experiment configurations for Supervised Fine-Tuning.

## Naming Convention

Experiments follow the pattern: `{train}_{task}-{selection}-{compression}-{model}-{training_type}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

| Component       | Options                     | Description                                   |
| --------------- | --------------------------- | --------------------------------------------- |
| `selection`     | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression`   | `NA`, `GraSS`, `LoGra`      | Gradient compression (implies MeSO optimizer) |
| `training_type` | `full`, `lora`              | Full fine-tuning or LoRA                      |

### Selection Modes
- **NA**: No data selection (baseline)
- **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
- **GREATS**: Global selection - accumulates scores across all layers (two-pass)

### Compression Methods
- **NA**: No compression - uses full gradients and standard AdamW optimizer
- **GraSS**: Gradient Sparsification with Sketching - uses MeSO optimizer
- **LoGra**: Low-rank Gradient compression - uses MeSO optimizer

## Experiment Configurations

### Full Configuration Matrix (16 experiments)

| #   | Selection | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | full     | Baseline full fine-tuning                 |
| 1b  | NA        | NA          | lora     | Baseline LoRA fine-tuning                 |
| 2a  | Streaming | NA          | full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | lora     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | lora     | Global selection, full gradients, LoRA    |
| 4a  | NA        | GraSS       | full     | MeSO only with GraSS (no selection)       |
| 4b  | NA        | LoGra       | full     | MeSO only with LoGra (no selection)       |
| 5a  | Streaming | GraSS       | full     | Per-layer selection + MeSO (GraSS)        |
| 5b  | Streaming | LoGra       | full     | Per-layer selection + MeSO (LoGra)        |
| 6a  | GREATS    | GraSS       | full     | Global selection + MeSO (GraSS)           |
| 6b  | GREATS    | LoGra       | full     | Global selection + MeSO (LoGra)           |
| 7a  | Streaming | GraSS       | full     | Per-layer + MeSO + second-order           |
| 7b  | GREATS    | GraSS       | full     | Global + MeSO + second-order              |

## Running Experiments

### Launch All Configurations

```bash
# Dry run
bash SFT/train/launch_experiments.sh --task samsum --train openhermes --dry-run

# Run experiments
bash SFT/train/launch_experiments.sh --task samsum --train openhermes \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 16 --percentage 0.05

# Submit to SLURM
bash SFT/train/launch_experiments.sh --task samsum --train openhermes --sbatch
```

### Launch Single Configuration

```bash
# Baseline
bash SFT/train/train.sh --task samsum --train openhermes --model llama3-1b

# Streaming with compression
bash SFT/train/train.sh --task samsum --train openhermes --model llama3-1b \
    --data_selection Streaming --selection_mode per_layer --compression GraSS

# GREATS with compression
bash SFT/train/train.sh --task samsum --train openhermes --model llama3-1b \
    --data_selection Streaming --selection_mode global --compression GraSS
```

## Experiment Summary

The following experiments have been run and can be rerun with the commands below.

| Train Dataset | Eval Task | Percentage           | Batch | N_val           | LR (full) | LR (LoRA) |
| ------------- | --------- | -------------------- | ----- | --------------- | --------- | --------- |
| alpaca        | samsum    | 0.9                  | 8     | 8               | 5e-06     | 1e-04     |
| alpaca        | tydiqa    | 0.9                  | 8     | 8               | 5e-06     | 1e-04     |
| less          | samsum    | 0.02, 0.05, 0.1      | 8     | 8 (16 for 0.05) | 5e-06     | 1e-04     |
| less          | tydiqa    | 0.02, 0.05, 0.1      | 8     | 8 (16 for 0.05) | 5e-06     | 1e-04     |
| wizardlm      | samsum    | 0.05, 0.2            | 8     | 8               | 5e-06     | 1e-04     |
| wizardlm      | tydiqa    | 0.05, 0.2            | 8     | 8               | 5e-06     | 1e-04     |
| tulu3         | samsum    | 0.02, 0.05, 0.1, 0.2 | 8     | 8               | 5e-06     | 1e-04     |
| tulu3         | tydiqa    | 0.02, 0.05, 0.1, 0.2 | 8     | 8               | 5e-06     | 1e-04     |

### Launch Commands

```bash
# Alpaca -> SAMSum/TyDiQA (p=0.9)
bash SFT/train/launch_experiments.sh --task samsum --train alpaca \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.9 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train alpaca \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.9 --seed 42

# LESS -> SAMSum/TyDiQA (p=0.1)
bash SFT/train/launch_experiments.sh --task samsum --train less \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.1 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train less \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.1 --seed 42

# WizardLM -> SAMSum/TyDiQA (p=0.2)
bash SFT/train/launch_experiments.sh --task samsum --train wizardlm \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.2 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train wizardlm \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.2 --seed 42

# Tulu3 -> SAMSum/TyDiQA (p=0.05)
bash SFT/train/launch_experiments.sh --task samsum --train tulu3 \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.03 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train tulu3 \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.03 --seed 42
```

## Output Directory

Results are saved to: `/scratch/pbb/Project/Gradient-Streaming/SFT/`

Each experiment creates a folder with the naming pattern above, containing:
- `train.log` - Training logs
- Model checkpoints (if save enabled)
- Evaluation results
