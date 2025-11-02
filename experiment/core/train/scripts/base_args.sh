#!/bin/bash

ID=$RANDOM
# Generate a unique port for this job to avoid conflicts with other jobs
PORT=$((29400 + RANDOM % 10000))
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d --rdzv-endpoint=localhost:$PORT \
-m core.train.train"

export base_training_args="--do_train=True \
--do_eval=True \
--max_seq_length=512 \
--use_fast_tokenizer=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--eval_steps=100 \
--eval_strategy=steps \
--save_strategy=epoch \
--num_train_epochs=1 \
--bf16=True \
--tf32=False \
--fp16=False \
--overwrite_output_dir=True \
--report_to=none \
--optim=adamw_torch \
--seed=0 \
--percentage=1.0 \
--lora=True \
--lora_r=256 \
--lora_dropout=0.1 \
--fracinv=2.0"
