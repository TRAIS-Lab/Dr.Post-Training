# Efficient Fine-Tuning

## Quick Start

```bash
pip install -r requirements.txt
```

Download data at this [link](https://drive.google.com/file/d/1oJw_3V-ALHHJMFQq8c3QJ4GNTGC8vw6s/view?usp=sharing), and put it in the `experiment/data` folder.

## Running Experiments

Navigate to the experiment directory:
```bash
cd experiment
```

### MMLU Task

Run MMLU experiments with different configurations:

```bash
# Regular training (no compression)
sh job/mmlu.sh Regular 2 0.05 5 500 llama3-1b 1 5e-05 42 1 sociology

# Training with GraSS/LoGra compression
sh job/mmlu.sh GREATS 2 0.05 5 500 llama3-1b 1 5e-05 42 1 sociology <GraSS/LoGra>
```

### SAMSUM Task

```bash
# Regular training
sh job/samsum.sh Regular 2 0.05 5 500 llama3-1b 1 5e-05 42 1

# Training with GraSS/LoGra compression
sh job/samsum.sh GREATS 2 0.05 5 500 llama3-1b 1 5e-05 42 1 <GraSS/LoGra>
```

### TyDiQA Task

```bash
# Regular training
sh job/tydiqa.sh Regular 2 0.05 5 500 llama3-1b 1 5e-05 42 1

# Training with GraSS/LoGra compression
sh job/tydiqa.sh GREATS 2 0.05 5 500 llama3-1b 1 5e-05 42 1 <GraSS/LoGra>
```

> [!Note]
> To make it compatible with gradient accumulation, the current implementation still uses 2 forward-backward passes.

## Parameters

### MMLU

```bash
sh job/mmlu.sh \
    <method>                        # Batch selection strategy. Options: Regular, GREATS, GradNorm, etc.
    <batch_size>                    # Batch size for training (e.g., 2, 4, 8)
    <percentage>                    # Percentage of dataset to use (e.g., 0.05 for 5%)
    <n_val>                         # Number of validation examples for in-context learning
    <n_eval>                        # Number of evaluation examples
    <model>                         # Model identifier (e.g., llama3-1b)
    <lora_alpha>                    # LoRA alpha parameter (e.g., 1)
    <lr>                            # Learning rate (e.g., 5e-05)
    [seed]                          # Random seed for reproducibility (default: 42)
    [gradient_accumulation_steps]  # Gradient accumulation steps (default: 1)
    [subject]                       # MMLU subject (default: world_religions)
    [compression]                   # Compression method: Vanilla, GraSS, or LoGra (default: Vanilla)
```

**Compression Methods:**
- **Vanilla**: No gradient compression
- **GraSS**: Gradient Sparsification with Sketching (RandomMask-128*128 + SJLT-4096)
- **LoGra**: Low-rank Gradient compression (Gaussian-64*64 projection)

### SAMSUM

```bash
sh job/samsum.sh \
    <method>                        # Batch selection strategy
    <batch_size>                    # Batch size for training
    <percentage>                    # Percentage of dataset to use
    <n_val>                         # Number of validation examples
    <n_eval>                        # Number of evaluation examples (default: 500)
    <model>                         # Model identifier
    <lora_alpha>                    # LoRA alpha parameter
    <lr>                            # Learning rate
    [seed]                          # Random seed (default: 42)
    [gradient_accumulation_steps]  # Gradient accumulation steps (default: 1)
    [compression]                   # Compression method (default: Vanilla)
```

### TyDiQA

```bash
sh job/tydiqa.sh \
    <method>                        # Batch selection strategy
    <batch_size>                    # Batch size for training
    <percentage>                    # Percentage of dataset to use
    <n_val>                         # Number of validation examples
    <n_eval>                        # Number of evaluation examples (default: 500)
    <model>                         # Model identifier
    <lora_alpha>                    # LoRA alpha parameter
    <lr>                            # Learning rate
    [seed]                          # Random seed (default: 42)
    [gradient_accumulation_steps]  # Gradient accumulation steps (default: 1)
    [compression]                   # Compression method (default: Vanilla)
```

## Evaluation

Trained models are automatically evaluated after training. Results are saved in `/scratch/pbb/Project/Efficient-Fine-Tuning/<JOB_NAME>/`.