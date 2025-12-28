# SFT Experiments

This folder contains the training and evaluation code and method configurations for Supervised Fine-Tuning.

## Data Preparation

Download and prepare datasets using the unified data preparation script:

```bash
# See available datasets and options
python SFT/data/prepare_datasets.py -h

# Download specific datasets
python SFT/data/prepare_datasets.py --datasets mmlu bbh tydiqa

# Download all evaluation datasets
python SFT/data/prepare_datasets.py --datasets mmlu bbh tydiqa gsm8k math500 samsum

# Download training datasets
python SFT/data/prepare_datasets.py --datasets alpaca dolly flan_v2 cot oasst1
```

<details>
  <summary>Available Datasets</summary>

### Evaluation Datasets

| Dataset   | Task Type          | Description                                              |
| --------- | ------------------ | -------------------------------------------------------- |
| `samsum`  | Summarization      | SamSUM dialogue summarization                            |
| `tydiqa`  | Question Answering | Typologically Diverse QA (9 languages)                   |
| `mmlu`    | Multiple Choice    | Massive Multitask Language Understanding (57 subjects)   |
| `bbh`     | Reasoning          | BIG-Bench Hard (23 challenging reasoning tasks with CoT) |
| `gsm8k`   | Math               | Grade School Math (8K problems)                          |
| `math500` | Math               | MATH benchmark (500 competition problems)                |

### Training Datasets

| Dataset      | Size | Description                           |
| ------------ | ---- | ------------------------------------- |
| `less`       | 1M   | LESS-selected instruction data        |
| `alpaca`     | 52K  | Stanford Alpaca instruction-following |
| `tulu3`      | 939K | Tulu-3 SFT mixture                    |
| `dolly`      | 15K  | Databricks Dolly 2.0                  |
| `flan_v2`    | 100K | FLAN v2 instruction tuning mixture    |
| `cot`        | 100K | Chain-of-Thought reasoning examples   |
| `oasst1`     | 88K  | OpenAssistant conversations           |
| `vicuna`     | 125K | ShareGPT-based conversations          |
| `wizardlm`   | 196K | WizardLM evolved instructions         |
| `openhermes` | 1M   | OpenHermes 2.5 diverse instructions   |
</details>

## Experiment Summary

The following methods have been run and can be rerun with the commands below.

| Train Dataset | Eval Task | Percentage | Batch | Val Size | LoRA Rank |
| ------------- | --------- | ---------- | ----- | -------- | --------- |
| Alpaca        | SamSUM    | 0.4        | 8     | 32       | 32        |
| Tulu3         | TydiQA    | 0.01       | 8     | 32       | 32        |
| LESS          | MMLU      | 0.05       | 8     | 32       | 128       |
| LESS          | BBH       | 0.05       | 8     | 32       | 128       |

### Experiment Configurations

We consider the following 8 methods for each of the training datasets above:

| #   | Selection | Compression | Training | Description                               |
| --- | --------- | ----------- | -------- | ----------------------------------------- |
| 1a  | NA        | NA          | Full     | Baseline full fine-tuning                 |
| 1b  | NA        | NA          | LoRA     | Baseline LoRA fine-tuning                 |
| 2a  | Streaming | NA          | Full     | Per-layer selection, full gradients       |
| 2b  | Streaming | NA          | LoRA     | Per-layer selection, full gradients, LoRA |
| 3a  | GREATS    | NA          | Full     | Global selection, full gradients          |
| 3b  | GREATS    | NA          | LoRA     | Global selection, full gradients, LoRA    |
| 4   | Streaming | LoGra       | Full     | Per-layer selection + MeSO                |
| 5   | GREATS    | LoGra       | Full     | Global selection + MeSO                   |

> Experiments follow the pattern: `{train}_{task}-{model}-{selection}-{compression}-{training_type}-p{pct}-lr{lr}-b{batch}-v{nval}-s{seed}`

1. Selection Modes
   - **NA**: No data selection (baseline)
   - **Streaming**: Per-layer selection - each layer independently selects samples (single-pass)
   - **GREATS**: Global selection - accumulates scores across all layers (two-pass)
2. Compression Methods
   - **NA**: No compression - uses full gradients and standard AdamW optimizer
   - **LoGra**: Low-rank Gradient compression (Gaussian projection) - uses MeSO optimizer
   - **GraSS**: Gradient Sparsification with Sketching (available but not used in default methods)
3. Training Types
   - **Full**: Full fine-tuning of all model parameters
   - **LoRA**: LoRA fine-tuning of low-rank adapters only

### LR Sweep Commands

The `SFT/train/lr/` folder contains tools for finding optimal learning rates, where learning rates are managed via `SFT/train/lr/config.json`. Run LR sweep before full training to find optimal learning rates:

```bash
# Alpaca -> SamSUM
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train alpaca --task samsum --batch_size 8 --n_val 8 --sweep_percentage 0.04 --seed 2

# Tulu3 -> TydiQA
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train tulu3 --task tydiqa --batch_size 8 --n_val 8 --sweep_percentage 0.001 --seed 2

# LESS -> MMLU/BBH
bash SFT/train/lr/lr_sweep.sh --mode binary --methods all --train less --task mmlu --subject sociology --batch_size 8 --n_val 8 --sweep_percentage 0.005 --seed 2 --lora_r 128
```

You can also run grid search via `--mode grid`.

### Training Commands

All methods are launched using the unified `train.sh` script. Training commands for each experiment (LRs loaded from lr_config.json):

```bash
# Alpaca -> SamSUM
bash SFT/train/train.sh --methods all --train alpaca --task samsum --batch_size 8 --n_val 32 --percentage 0.4 --seed 42

# Tulu3 -> TydiQA
bash SFT/train/train.sh --methods all --train tulu3 --task tydiqa --batch_size 8 --n_val 32 --percentage 0.01 --seed 42

# LESS -> MMLU/BBH
bash SFT/train/train.sh --methods all --train less --task mmlu --subject sociology --batch_size 8 --n_val 32 --percentage 0.05 --seed 42 --lora_r 128
```


<details>
  <summary>Detailed Training Script Configuration</summary>

#### Run Multiple Experiments
Run multiple methods with the `--methods` flag:

```bash
# Run all 8 methods
bash SFT/train/train.sh --methods all --task mmlu --subject sociology

# Run by category
bash SFT/train/train.sh --methods baseline --task mmlu      # NA-NA-Full, NA-NA-LoRA
bash SFT/train/train.sh --methods streaming --task mmlu     # All Streaming-* variants
bash SFT/train/train.sh --methods greats --task mmlu        # All GREATS-* variants
bash SFT/train/train.sh --methods compression --task mmlu   # *-LoGra-* variants
bash SFT/train/train.sh --methods lora --task mmlu          # All *-LoRA variants
bash SFT/train/train.sh --methods full --task mmlu          # All *-Full variants

# Run specific methods
bash SFT/train/train.sh --methods "NA-NA-Full,Streaming-LoGra-Full" --task mmlu

# Combine categories
bash SFT/train/train.sh --methods "baseline,streaming" --task mmlu

# Dry run - preview commands without executing
bash SFT/train/train.sh --methods all --task mmlu --dry-run

# Submit to SLURM
bash SFT/train/train.sh --methods all --task mmlu --sbatch
```

Available Categories:

| Category         | Experiments                                                |
| ---------------- | ---------------------------------------------------------- |
| `all`            | All 8 methods                                              |
| `baseline`       | NA-NA-Full, NA-NA-LoRA                                     |
| `streaming`      | Streaming-NA-Full, Streaming-NA-LoRA, Streaming-LoGra-Full |
| `greats`         | GREATS-NA-Full, GREATS-NA-LoRA, GREATS-LoGra-Full          |
| `full`           | All *-Full methods (5 total)                               |
| `lora`           | All *-LoRA methods (3 total)                               |
| `compression`    | Streaming-LoGra-Full, GREATS-LoGra-Full                    |
| `no-compression` | All methods without compression (6 total)                  |

#### Parameters

The unified training script accepts the following arguments:

1. Task Arguments
   - `--task <task>` - Evaluation task: `mmlu`, `bbh`, `tydiqa`, `gsm8k`, `math500`, `samsum`
   - `--subject <subject>` - Subject for MMLU/BBH (default: `sociology`)
   - `--train <dataset>` - Training dataset (optional, overrides task-based default):
     - `alpaca`, `dolly`, `flan_v2`, `cot`, `oasst1` - Instruction tuning
     - `gsm8k` - Math training data
     - `vicuna`, `wizardlm`, `openhermes`, `tulu3`, `less` - Large-scale instruction data
2. Data Selection Arguments
   - `--data_selection <method>` - Data selection method:
     - `NA` - No selection (baseline, default)
     - `Streaming` - Per-layer selection (single-pass)
     - `GREATS` - Global selection (two-pass)
   - `--use_second_order` - Enable greedy selection with second-order interactions (enabled by default for all selection methods)
3. Compression Arguments
   - `--compression <method>` - Gradient compression method (implies MeSO optimizer):
     - `LoGra` - Low-rank Gradient compression (Gaussian projection, default)
     - `GraSS` - Gradient Sparsification with Sketching (available but not used in default methods)
     - If not specified, uses full gradients and standard AdamW optimizer
   - `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)
4. Core Training Arguments
   - `--model <model>` - HuggingFace model path (default: `meta-llama/Llama-3.2-1B`)
   - `--lr <lr>` - Learning rate override (if not specified, looked up from `lr_config.json`; fallback: `5e-05` for full, `2e-04` for LoRA)
   - `--lr_config <path>` - LR config file path (default: `SFT/train/lr/config.json`)
   - `--batch_size <size>` - Batch size (default: `8`)
   - `--seed <seed>` - Random seed (default: `42`)
   - `--gradient_accumulation_steps <steps>` - Gradient accumulation (default: `1`)
5. Data Arguments
   - `--percentage <pct>` - Data sampling, e.g., `0.05` for 5% (default: `0.05`)
   - `--n_val <n>` - Validation examples for data selection (default: `8`)
   - `--n_eval <n>` - Evaluation examples (default: `500`)
   - `--val_batch_size <size>` - Validation batch size for data selection (default: `1`)
   - `--data_dir <dir>` - Data directory (default: `SFT/data`)
6. LoRA Arguments
   - `--lora` - Enable LoRA fine-tuning (flag, omit for full fine-tuning)
   - `--lora_alpha <alpha>` - LoRA alpha (default: `1`)
   - `--lora_r <r>` - LoRA rank (default: `32`)
   - `--lora_dropout <dropout>` - LoRA dropout (default: `0.1`)
   - `--lora_target_modules <modules>` - Target modules for LoRA (default: `q_proj k_proj v_proj o_proj`)
7. Model Arguments
   - `--flash_attention` - Enable Flash Attention 2 (default: enabled)
</details>

### Evaluation Commands
Evaluation commands for each experiment:

```bash
# Alpaca -> SamSUM
bash SFT/eval/eval.sh --train alpaca --task samsum --batch_size 64

# Tulu3 -> TyDiQA
bash SFT/eval/eval.sh --train tulu3 --task tydiqa --batch_size 64 --n_test 500

# LESS -> MMLU/BBH
bash SFT/eval/eval.sh --train less --task mmlu --subject sociology --batch_size 64
```