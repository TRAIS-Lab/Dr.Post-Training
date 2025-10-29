#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import time

import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import datasets
import torch
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    DataCollatorForSeq2Seq
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from experiment.utils.common import (
    SubsetSampler,
    replace_conv1d_modules,
    setup_compression_kwargs,
    validate_dataset_args,
    standardize_dataset_columns,
    is_sft
)
from experiment.utils import(
    prepare_datasets,
    preprocess_sft_dataset,
    create_data_collator
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.46.0")

pre_init_logger = logging.getLogger(__name__)
logger = get_logger(__name__)

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.0,
        help="Ratio of total training steps to use for warmup (e.g., 0.03 for 3% warmup). Overrides num_warmup_steps if set."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )

    # >>>>>>>>>>>>>>>>>>>>> Customize Argument begins here >>>>>>>>>>>>>>>>>>>>>
    parser.add_argument(
        "--force_sft",
        action="store_true",
        help="Force supervised fine-tuning mode regardless of dataset.",
    )
    parser.add_argument(
        "--force_lm",
        action="store_true",
        help="Force standard language modeling mode regardless of dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device to be used",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Record profiling results.",
    )
    parser.add_argument(
        "--tda",
        type=str,
        default="IF-RAW",
        help="Specify which mode we want to run the data attribution method. Available options: IF-RAW.",
    )
    parser.add_argument(
        "--layer",
        type=str,
        default="Linear",
        help="Layer used for attribution. Available Layers: Linear and LayerNorm",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode.",
    )
    parser.add_argument(
        "--projection",
        type=str,
        default=None,
        help="The projection method to be used when attributing. Basic format: 'proj_method-proj_dim' for non-factorized gradient and 'proj_method-proj_dim*proj_dim' for factorized gradient.",
    )
    parser.add_argument(
        "--sparsification",
        type=str,
        default=None,
        help="The first stage of the gradient compression algorithm. Basic format: 'sparsification_method-proj_dim' for non-factorized gradient and 'sparsification_method-proj_dim*proj_dim' for factorized gradient.",
    )
    parser.add_argument(
        "--output_influence_score_dir",
        type=str,
        default=None,
        help="Path to final influence score file.",
    )

    # Mixture dataset arguments
    parser.add_argument(
        "--master_train_size",
        type=int,
        default=5000,
        help="Size of the master original subset."
    )
    parser.add_argument(
        "--mixture_sizes",
        type=str,
        default="5000",
        help="Comma-separated sizes for mixture training (e.g., '3000,1000,500' for original,influential,synthetic)."
    )
    parser.add_argument(
        "--mixture_components",
        type=str,
        default="original",
        help="Comma-separated components for mixture training."
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=200,
        help="Size of validation set (split from HuggingFace validation set)"
    )
    parser.add_argument(
        "--evaluation_size",
        type=int,
        default=500,
        help="Size of evaluation set (split from HuggingFace validation set)"
    )
    parser.add_argument(
        "--influence_score_file",
        type=str,
        default=None,
        help="Path to influence score file (output from score.py, .pt format). Required if mixture includes 'influential' component.",
    )
    parser.add_argument(
        "--synthetic_dataset_dir",
        type=str,
        default=None,
        help="Path to synthetic dataset directory. Required if mixture includes 'synthetic' component.",
    )
    parser.add_argument(
        "--synthetic_samples_file",
        type=str,
        default=None,
        help="Path to synthetic samples JSON file. Alternative to synthetic_dataset_dir.",
    )

    parser.add_argument(
        "--evaluation_metrics",
        type=str,
        default="rouge_l",
        help="Comma-separated list of evaluation metrics to use. Available: rouge_l, test_accuracy, bleu. Use 'auto' to use dataset-specific defaults."
    )
    parser.add_argument(
        "--evaluation_frequency",
        type=int,
        default=1,
        help="Frequency of evaluation in training steps. Evaluation occurs at the beginning of epochs when step count reaches multiples of this value."
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Return random score instead of actual influence score. Useful for ablation study.",
    )

    parser.add_argument(
        "--num_evaluation_runs",
        type=int,
        default=5,
        help="Number of evaluation runs to average over (to account for LLM randomness). Set to 1 for single run."
    )

    args = parser.parse_args()

    # Validate dataset arguments using shared function
    validate_dataset_args(args)

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    return args

def result_filename(args) -> str:
        """
        Mirror args.output_dir under an `influence_score` root and return
        `<sparsification_name>-><projection_name>.pt` inside that folder.
        """
        sparsification_name = args.sparsification or "NA"
        projection_name = args.projection or "NA"

        out_path = Path(args.output_dir).expanduser()

        # Replace the FIRST "checkpoints" segment with "influence_score"
        parts = list(out_path.parts)
        try:
            idx = parts.index("checkpoints")
            parts[idx] = "influence_score"
            result_dir = Path(*parts)          # e.g. ./save/influence_score/…/…
        except ValueError:
            # Fallback: if "checkpoints" is missing, stick save/influence_score in front
            result_dir = Path("influence_score") / out_path

        # Ensure the directory exists
        result_dir.mkdir(parents=True, exist_ok=True)

        if args.random:
            # If random mode, save as random_score.pt
            return str(result_dir / "random_score.pt")

        return str(result_dir / f"{sparsification_name}->{projection_name}.pt")


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        set_seed(args.seed)

    # Handle the repository creation
    api = None
    repo_id = None
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    train_dataset, val_dataset, eval_dataset = prepare_datasets(args)

    # Add this to print master indices:
    def print_master_indices(dataset, dataset_name):
        """Print master indices for a dataset"""
        if len(dataset) == 0:
            logger.info(f"{dataset_name}: No samples")
            return

        # Check if master indices exist
        if '_master_index' not in dataset[0]:
            logger.info(f"{dataset_name}: No master indices found")
            return

        # Extract all master indices
        master_indices = [dataset[i]['_master_index'] for i in range(len(dataset))]

        logger.info(f"{dataset_name} master indices:")
        logger.info(f"  Total samples: {len(master_indices)}")
        logger.info(f"  Master index range: {min(master_indices)} to {max(master_indices)}")
        logger.info(f"  Master indices: {master_indices}")

        # Show some sample indices if dataset is large
        if len(master_indices) > 50:
            logger.info(f"  First 10: {master_indices[:25]}")
            logger.info(f"  Last 10: {master_indices[-25:]}")

    # Print indices for all datasets
    print_master_indices(train_dataset, "Training Dataset")
    print_master_indices(val_dataset, "Validation Dataset")
    print_master_indices(eval_dataset, "Evaluation Dataset")

    # Determine SFT mode
    sample = train_dataset[0] if len(train_dataset) > 0 else None

    use_sft = is_sft(
        dataset_name=args.dataset_name,
        train_file=args.train_file,
        sample_data=sample,
        force_sft=args.force_sft,
        force_lm=args.force_lm
    )

    if use_sft:
        logger.info("Using supervised fine-tuning (SFT) mode")
    else:
        logger.info("Using standard language modeling mode")


    # Always standardize column schemas to remove metadata columns (but preserve _master_index)
    base_columns = train_dataset.column_names
    metadata_columns = {
        "_seed_index", "_generation_id", "_synthetic", "_seed_score",
        "category",
        "_original_seed_index", "_timestamp", "_sample_index", "_seed_master_index"
    }
    base_columns = [col for col in base_columns if col not in metadata_columns]

    # Standardize each dataset based on its own columns
    train_target_columns = [col for col in train_dataset.column_names if col not in metadata_columns]
    val_target_columns = [col for col in val_dataset.column_names if col not in metadata_columns]
    eval_target_columns = [col for col in eval_dataset.column_names if col not in metadata_columns]

    train_dataset = standardize_dataset_columns(train_dataset, train_target_columns)
    val_dataset = standardize_dataset_columns(val_dataset, val_target_columns)
    eval_dataset = standardize_dataset_columns(eval_dataset, eval_target_columns)

    # Convert to DatasetDict for compatibility
    raw_datasets = datasets.DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "evaluation": eval_dataset
    })

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code, padding_side="left"
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code, padding_side="left"
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)

    # Setup tokenizer for SFT
    if use_sft:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Ensure EOS token is properly set
        if tokenizer.eos_token is None:
            logger.error("No EOS token found - this will cause generation issues")
            raise ValueError("EOS token must be available for SFT")

        logger.info(f"Tokenizer setup: pad_token='{tokenizer.pad_token}', eos_token='{tokenizer.eos_token}'")
        logger.info(f"Tokenizer IDs: pad_token_id={tokenizer.pad_token_id}, eos_token_id={tokenizer.eos_token_id}")

    # Set model generation config
    if tokenizer.eos_token_id is not None:
        model.config.eos_token_id = tokenizer.eos_token_id
        if hasattr(model, 'generation_config'):
            model.generation_config.eos_token_id = tokenizer.eos_token_id
            model.generation_config.pad_token_id = tokenizer.pad_token_id
    else:
        logger.warning("No EOS token found in tokenizer - model may not learn to stop properly")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

    # Different tokenization for SFT vs standard LM
    if use_sft:
        # For SFT, use the specialized preprocessing
        def tokenize_function(examples):
            tokenized = preprocess_sft_dataset(
                examples,
                tokenizer,
                args.block_size or tokenizer.model_max_length
            )
            # Preserve original indices if present
            if '_master_index' in examples:
                tokenized['_master_index'] = examples['_master_index']
            return tokenized
    else:
        text_column_name = "text" if "text" in column_names else column_names[0]
        def tokenize_function(examples):
            tokenized = tokenizer(examples[text_column_name])
            # Preserve original indices if present
            if '_master_index' in examples:
                tokenized['_master_index'] = examples['_master_index']
            return tokenized

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=[col for col in column_names if col != '_master_index'],
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Original block_size processing for standard LM
    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > config.max_position_embeddings:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
            )
            block_size = min(1024, config.max_position_embeddings)
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    model.config.max_position_embeddings = block_size

    # Only do block_size processing for standard LM
    if use_sft:
        # For SFT, we don't need to group texts
        lm_datasets = tokenized_datasets
    else:
        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys() if k != '_master_index'}
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()

            # Handle indices for grouped texts
            if '_master_index' in examples:
                # For grouped texts, we'll lose the exact mapping, but keep the first index for each group
                original_indices = examples['_master_index']
                num_groups = len(result["input_ids"])
                result['_master_index'] = [original_indices[0]] * num_groups

            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]
    eval_dataset = lm_datasets["evaluation"]

    # Use different collators for SFT vs standard LM
    data_collator = create_data_collator(tokenizer, model, use_sft)
    logger.info("Using master index preserving data collator")


    # Capture training dataset metadata right after preparation
    training_master_indices = []
    training_dataset_info = {}

    if len(train_dataset) > 0:
        # Extract master indices from training dataset
        training_master_indices = [train_dataset[i]['_master_index'] for i in range(len(train_dataset))]

        # Capture training dataset composition info
        training_dataset_info = {
            "total_samples": len(train_dataset),
            "master_indices": training_master_indices,
            "master_index_range": [min(training_master_indices), max(training_master_indices)] if training_master_indices else [None, None],
            "sample_keys": list(train_dataset[0].keys()) if len(train_dataset) > 0 else [],
            "use_sft": use_sft,
            "preprocessing_info": {
                "block_size": block_size,
                "tokenizer_name": args.model_name_or_path or args.tokenizer_name,
            }
        }

        logger.info(f"Captured training metadata: {len(training_master_indices)} samples, "
                    f"master index range: {training_dataset_info['master_index_range']}")
    else:
        logger.warning("Empty training dataset - no metadata to capture")

    # >>>>>>>>>>>>>>>>>>>>> Customized Code begins here >>>>>>>>>>>>>>>>>>>>>
    if args.device.startswith("cuda"):
        # Check if GPU is available
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your CUDA installation.")
        device = torch.device(args.device)
    else:
        assert args.device == "cpu", "Invalid device. Choose from 'cuda' or 'cpu'."
        device = torch.device("cpu")

    torch.cuda.set_device(device)

    # Create dataloaders
    train_sampler = SubsetSampler(range(len(train_dataset)))
    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
    )
    test_dataloader = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False
    )

    throughput_stats = {}

    sparsifier_kwargs, projector_kwargs = setup_compression_kwargs(args, device)

    # Logging setting
    logger.info("***** Running attribution *****")
    logger.info(f"  TDA Method: {args.tda}")
    logger.info(f"  Batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Compression: Sparsifier: {sparsifier_kwargs}|Projector: {projector_kwargs}")
    logger.info(f"  Layer: {args.layer}")

    from _GradComp.utils.common import find_layers
    from _GradComp.attributor.attributor import IFAttributor

    # get which Hessian to use
    tda, hessian = args.tda.split("-")
    hessian = hessian.lower()
    assert tda == "IF", "GradComp only supports Influence Function now."

    checkpoint = f"{args.output_dir}"
    model = AutoModelForCausalLM.from_pretrained(checkpoint)
    model = replace_conv1d_modules(model)
    layer_names = find_layers(model, args.layer, return_type="name")

    attributor = IFAttributor(
        setting="GPT2_wikitext",
        model=model,
        layer_names=layer_names,
        hessian=hessian,
        profile=args.profile,
        device=device,
        sparsifier_kwargs=sparsifier_kwargs,
        projector_kwargs=projector_kwargs,
        use_sft_loss=use_sft,
        offload="cpu",
        cache_dir="./save/cache"
    )

    # Measure cache throughput
    torch.cuda.synchronize(device)
    cache_start_time = time.time()
    attributor.cache_gradients(train_dataloader=train_dataloader)
    torch.cuda.synchronize(device)
    cache_end_time = time.time()

    torch.cuda.synchronize(device)
    attribute_start_time = time.time()

    # Compute preconditioners for best damping value
    attributor.compute_preconditioners()
    attributor.compute_ifvp()

    if args.profile:
        self_if, profile = attributor.compute_self_attribution()
    else:
        self_if = attributor.compute_self_attribution()

    # Measure attribute throughput
    torch.cuda.synchronize(device)
    attribute_start_time = time.time()
    if args.profile:
        score, profile = attributor.attribute(test_dataloader=test_dataloader, cosine=True)
    else:
        score = attributor.attribute(test_dataloader=test_dataloader, cosine=True)
    torch.cuda.synchronize(device)
    attribute_end_time = time.time()

    # Calculate throughput
    train_tokens = block_size * len(train_dataset)
    train_test_pairs = len(train_dataset) * len(val_dataset)

    cache_duration = cache_end_time - cache_start_time
    cache_throughput = train_tokens / cache_duration
    throughput_stats["cache"] = {
        "tokens": train_tokens,
        "duration_seconds": cache_duration,
        "throughput_tokens_per_second": cache_throughput
    }

    attribute_duration = attribute_end_time - attribute_start_time
    attribute_throughput = train_test_pairs / attribute_duration
    throughput_stats["attribute"] = {
        "train_test_pairs": train_test_pairs,
        "duration_seconds": attribute_duration,
        "throughput_pair_per_second": attribute_throughput
    }

    logger.info("***** Attribution finished *****")

    training_metadata = {
        # Core dataset arguments needed for reconstruction
        "dataset_args": {
            "dataset_name": args.dataset_name,
            "dataset_config_name": getattr(args, 'dataset_config_name', None),
            "train_file": getattr(args, 'train_file', None),
            "validation_file": getattr(args, 'validation_file', None),
            "validation_split_percentage": getattr(args, 'validation_split_percentage', 5),
            "no_keep_linebreaks": getattr(args, 'no_keep_linebreaks', False),
            "trust_remote_code": getattr(args, 'trust_remote_code', False),
            "seed": args.seed,
            "master_train_size": args.master_train_size,
        },

        # Mixture configuration
        "mixture_config": {
            "components": args.mixture_components,
            "sizes": args.mixture_sizes,
            "validation_size": args.validation_size,
            "evaluation_size": args.evaluation_size,
        },

        # Exact training dataset info (ground truth)
        "training_dataset_info": training_dataset_info,

        # Attribution metadata
        "attribution_info": {
            "tda_method": args.tda,
            "layer": args.layer,
            "projection": args.projection,
            "sparsification": args.sparsification,
            "model_name_or_path": args.model_name_or_path,
            "attribution_timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        },

        # Version info for debugging
        "version_info": {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
    }

    if args.random:
        # replace score with random values for ablation study
        logger.info("Random score mode enabled - replacing actual scores with random values")
        score = torch.rand_like(self_if)  # Random scores with the same shape as self_if
        self_if = torch.rand_like(self_if)  # Random self scores as well

    result = {
        "score": score,
        "self_score": self_if,
        "profile": profile if args.profile else None,
        "throughput": throughput_stats,
        "training_metadata": training_metadata
    }
    logger.info(result)

    logger.info(f"Saving attribution results with training metadata to {result_filename(args)}")
    logger.info(f"Training metadata includes {len(training_metadata['training_dataset_info']['master_indices'])} master indices")

    if not args.debug: # only save the results when not in debug mode
        # first make sure the output directory exists
        torch.save(result, result_filename(args))

if __name__ == "__main__":
    main()