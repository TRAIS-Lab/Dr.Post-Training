# SFT Experiments

This folder contains the training code and experiment configurations for Supervised Fine-Tuning.

## Naming Convention

Experiments follow the pattern: `{train}_{task}-{selection}-{compression}-{model}-{training_type}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

| Component       | Options                     | Description                                   |
| --------------- | --------------------------- | --------------------------------------------- |
| `selection`     | `NA`, `Streaming`, `GREATS` | Data selection method                         |
| `compression`   | `NA`, `LoGra`               | Gradient compression (implies MeSO optimizer) |
| `training_type` | `full`, `lora`              | Full fine-tuning or LoRA                      |

> **Note:** GraSS compression is also available (`--compression GraSS`) but not used in default experiments.

### Selection Modes
- **NA**: No data selection (baseline)
- **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
- **GREATS**: Global selection - accumulates scores across all layers (two-pass)

### Compression Methods
- **NA**: No compression - uses full gradients and standard AdamW optimizer
- **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
- **GraSS**: Gradient Sparsification with Sketching (available but not used in default experiments)

## Experiment Configurations

### Full Configuration Matrix (10 experiments)

| #   | Selection | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | full     | Baseline full fine-tuning                 |
| 1b  | NA        | NA          | lora     | Baseline LoRA fine-tuning                 |
| 2a  | Streaming | NA          | full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | lora     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | lora     | Global selection, full gradients, LoRA    |
| 4   | Streaming | LoGra       | full     | Per-layer selection + MeSO                |
| 5   | GREATS    | LoGra       | full     | Global selection + MeSO                   |
| 6a  | Streaming | LoGra       | full     | Per-layer + MeSO + second-order           |
| 6b  | GREATS    | LoGra       | full     | Global + MeSO + second-order              |

> **Note:** LoRA doesn't need compression (already low-rank). Compression without selection doesn't provide meaningful benefit.

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
    --data_selection Streaming --compression LoGra

# GREATS with compression
bash SFT/train/train.sh --task samsum --train openhermes --model llama3-1b \
    --data_selection GREATS --compression LoGra
```

## Experiment Summary

The following experiments have been run and can be rerun with the commands below.

| Train Dataset | Eval Task       | Percentage | Batch | N_val | LR (full) | LR (LoRA) |
| ------------- | --------------- | ---------- | ----- | ----- | --------- | --------- |
| alpaca        | samsum / tydiqa | 0.4        | 8     | 8     | 5e-06     | 1e-04     |
| less          | samsum / tydiqa | 0.05       | 8     | 8     | 5e-06     | 1e-04     |
| wizardlm      | samsum / tydiqa | 0.1        | 8     | 8     | 5e-06     | 1e-04     |
| tulu3         | samsum / tydiqa | 0.01       | 8     | 8     | 5e-06     | 1e-04     |

### Launch Commands

Training commands for each experiment:

```bash
# Alpaca -> SAMSum/TyDiQA (p=0.4)
bash SFT/train/launch_experiments.sh --task samsum --train alpaca \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.4 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train alpaca \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.4 --seed 42

# LESS -> SAMSum/TyDiQA (p=0.05)
bash SFT/train/launch_experiments.sh --task samsum --train less \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.05 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train less \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.05 --seed 42

# WizardLM -> SAMSum/TyDiQA (p=0.1)
bash SFT/train/launch_experiments.sh --task samsum --train wizardlm \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.1 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train wizardlm \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.1 --seed 42

# Tulu3 -> SAMSum/TyDiQA (p=0.01)
bash SFT/train/launch_experiments.sh --task samsum --train tulu3 \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.01 --seed 42

bash SFT/train/launch_experiments.sh --task tydiqa --train tulu3 \
    --lr 5e-06 --lr_lora 1e-04 --batch_size 8 --n_val 8 --percentage 0.01 --seed 42
```

Evaluation commands for each experiment:

```bash
# Alpaca -> SAMSum/TyDiQA
bash SFT/eval/eval.sh --train alpaca --task samsum --batch_size 64
bash SFT/eval/eval.sh --train alpaca --task tydiqa --batch_size 64

# LESS -> SAMSum/TyDiQA
bash SFT/eval/eval.sh --train less --task samsum --batch_size 64
bash SFT/eval/eval.sh --train less --task tydiqa --batch_size 64

# WizardLM -> SAMSum/TyDiQA
bash SFT/eval/eval.sh --train wizardlm --task samsum --batch_size 64
bash SFT/eval/eval.sh --train wizardlm --task tydiqa --batch_size 64

# Tulu3 -> SAMSum/TyDiQA
bash SFT/eval/eval.sh --train tulu3 --task samsum --batch_size 64
bash SFT/eval/eval.sh --train tulu3 --task tydiqa --batch_size 64
```

## Output Directory

Results are saved to: `/scratch/pbb/Project/Gradient-Streaming/SFT/`

Each experiment creates a folder with the naming pattern above, containing:
- `train.log` - Training logs
- Model checkpoints (if save enabled)
- Evaluation results
