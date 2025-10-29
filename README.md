# Efficient Fine-Tuning:

## Quick Start

```bash
pip install -r requirements.txt
```

Download data at this [link](https://drive.google.com/file/d/1L8IE7_9R-8zamRrR-69PRIB9WoEHF5XY/view?usp=sharing), and put it in the folder of codebase.

Run experiments using:
```bash
sh mmlu.sh Regular 8 0.05 4 mmlu llama2 1 2e-05 11 1 sociology
sh mmlu.sh GREATS 8 0.05 4 mmlu llama2 1 2e-05 11 1 sociology "RandomMask-128*128" "SJLT-4096"
sh mmlu.sh GREATS 8 0.05 4 mmlu llama2 1 2e-05 11 1 sociology "" "Gaussian-64*64"
```

> [!Note]
> To make it compatible with gradient accumulation, in the current implementation still uses 2 forward-backward passes.


### Parameters

```bash
sh mmlu.sh \
    <selection_method>  # Batch selection strategy. Options: Regular, GREATS, GradNorm, MaxLoss, RHO-Loss, SBERT.
    <batch_size>        # Batch size for training.
    <data_percentage>   # Percentage of the full dataset used for training (for faster test).
    <validation_size>   # Size of the validation set.
    <task>              # Task name for the model (e.g., a classification or QA task).
    <model>             # Model name or path to the pretrained model.
    <lora_alpha>        # LoRA hyperparameter (if applicable).
    <learning_rate>     # Learning rate for the optimizer.
    [seed]              # Random seed for reproducibility.
    [gradient_accumulation_steps]  # Number of gradient accumulation steps.
    [subject]           # Dataset subject.
	[sparsify]          # Sparsification method.
	[projection]        # Projection method.
```