#!/bin/bash

ID=$RANDOM
export header="torchrun --nproc_per_node 1 --nnodes 1 \
--rdzv-id=$ID --rdzv_backend c10d \
-m core.train.train"

export base_training_args="--do_train=True \
--do_eval=True \
--max_seq_length=512 \
--use_fast_tokenizer=True \
--lr_scheduler_type=linear \
--warmup_ratio=0.03 \
--weight_decay=0.0 \
--logging_steps=1 \
--n_eval=500 \
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
--lora_r=64 \
--lora_dropout=0.1 \
--fracinv=2.0"
