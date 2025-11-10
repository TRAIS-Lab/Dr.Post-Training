# Efficient Fine-Tuning

This repository implements memory-efficient fine-tuning with **compressed optimizer** support, enabling full model fine-tuning on limited GPU memory.

## Quick Start

```bash
pip install -r requirements.txt
```

Download data at this [link](https://drive.google.com/file/d/1oJw_3V-ALHHJMFQq8c3QJ4GNTGC8vw6s/view?usp=sharing), and put it in the `experiment/data` folder.

### Recommended Environment Setup

It's **not** required to follow the exact same steps in this section. But this is a verified environment setup flow that may help users to avoid most of the issues during the installation.

```bash
conda create -n IF python=3.10
conda activate IF

conda install -c "nvidia/label/cuda-11.8.0" cudatoolkit
pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Running Experiments

Navigate to the experiment directory:
```bash
cd experiment/job
```

### Unified Training Script

Single script for all tasks and training modes with named arguments:

```bash
sbatch train.sh [options]
```

#### Examples

```bash
# Baseline
# Full fine-tuning (without MeSO and data selection)
sbatch train.sh --task mmlu --subject sociology --model llama3-1b

# LoRA fine-tuning (without MeSO and data selection)
sbatch train.sh --task mmlu --subject sociology --model llama3-1b --lora

# Data Selection
# Full fine-tuning with GREATS data selection
sbatch train.sh --task mmlu --subject sociology --model llama3-1b --data_selection GREATS --compression <GraSS/LoGra>

# LoRA fine-tuning with GREATS data selection
sbatch train.sh --task mmlu --subject sociology --model llama3-1b --lora --data_selection GREATS --compression <GraSS/LoGra>

# MeSO Fine-Tuning
# MeSO fine-tuning without data selection
sbatch train.sh --task mmlu --subject sociology --model llama3-1b --MeSO --compression <GraSS/LoGra>

# Both MeSO and Data Selection
# MeSO fine-tuning with GREATS data selection
sbatch train.sh --task mmlu --subject sociology --model llama3-1b --MeSO --data_selection GREATS --compression <GraSS/LoGra>
```

#### Parameters

The unified training script accepts the following arguments:

##### Required Arguments

- `--task <task>` - Task name: `mmlu`, `samsum`, `tydiqa`, `bbh`

###### Task-Specific Arguments

- `--subject <subject>` - Subject for MMLU/BBH (default: `world_religions`)

##### Core Training Arguments

- `--data_selection <method>` - Data selection: `NA`, `GREATS`, `GradNorm` (default: `NA`)
- `--optimizer <type>` - Optimizer: `Regular`, `MeSO` (compressed optimizer) (default: `Regular`)
- `--model <model>` - Model: `llama3-1b`, `llama2-7b`, `mistral-7b` (default: `llama3-1b`)
- `--lr <lr>` - Learning rate (default: `5e-05`)
- `--batch_size <size>` - Batch size (default: `2`)
- `--seed <seed>` - Random seed (default: `42`)
- `--gradient_accumulation_steps <steps>` - Gradient accumulation (default: `1`)

##### Data Arguments

- `--percentage <pct>` - Data sampling, e.g., `0.05` for 5% (default: `0.05`)
- `--n_val <n>` - Validation examples (default: `5`)
- `--n_eval <n>` - Evaluation examples (default: `500`)
- `--data_dir <dir>` - Data directory (default: `data`)

##### Compression Arguments

- `--compression <method>` - Gradient compression method (auto-enabled when needed)
  - `GraSS` - Gradient Sparsification with Sketching (RandomMask-128×128 + SJLT-4096) [default]
  - `LoGra` - Low-rank Gradient compression (Gaussian-64×64)
  - Auto-enabled when `--data_selection` is not `NA` or `--optimizer` is `MeSO`
- `--update_compressor_freq <steps>` - Projector refresh interval (default: `200`)
  - Resamples random projections every N steps
  - Helps maintain compression quality throughout training
  - Set to large value (e.g., `1000000`) to disable refresh

##### LoRA Arguments

- `--lora` - Enable LoRA fine-tuning (flag, omit for full fine-tuning)
- `--lora_alpha <alpha>` - LoRA alpha (default: `1`)
- `--lora_r <r>` - LoRA rank (default: `256`)

## Evaluation

Trained models are automatically evaluated after training. Results are saved in `/scratch/pbb/Project/Efficient-Fine-Tuning/<JOB_NAME>/`.