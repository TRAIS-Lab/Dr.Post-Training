#!/usr/bin/env python3
"""
Script to download and prepare datasets for training and evaluation.
Supports: TyDiQA, SamSUM, TriviaQA, NQ-open (eval) + NQ-open, TriviaQA, SQuAD, Alpaca, Dolly, FLAN-v2, CoT, OASST1, Tulu3 (train).
"""

import glob
import json
import os
import random
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# Fixed seed for shuffling eval splits. Decoupled from any per-run seed so the
# val/lr/test partition is reproducible across runs.
EVAL_SHUFFLE_SEED = 42


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def shuffled_examples(examples, seed=EVAL_SHUFFLE_SEED):
    """Return a shuffled copy of `examples` using a fixed seed.

    Many HF datasets are ordered by article/language/topic (e.g., SQuAD validation
    is grouped by article, TyDiQA test is grouped by language). Slicing the first
    N examples therefore produces a topically-narrow eval set that doesn't
    represent the full distribution. Shuffle once before slicing val/lr/test.
    """
    examples = list(examples)
    random.Random(seed).shuffle(examples)
    return examples


def prepare_nq_open_train(output_dir):
    """
    Prepare NaturalQuestions open (closed-book) as a SFT training set.
    Q -> A pairs, ~88K examples.
    """
    print("Preparing NaturalQuestions (nq_open) training data...")
    ds = load_dataset("nq_open", split="train")

    output_file = os.path.join(output_dir, "train", "nq_open", "nq_open_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(ds)):
            question = ex["question"]
            answers = ex["answer"]
            if not answers:
                continue
            answer = answers[0]
            data = {
                "dataset": "nq_open",
                "id": f"nq_open_{idx}",
                "messages": [
                    {"role": "user", "content": f"Answer the following question.\nQuestion: {question}"},
                    {"role": "assistant", "content": answer},
                ],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    print(f"NQ-open training data saved to {output_file}")
    return output_file


def prepare_nq_open_eval(output_dir):
    """
    Prepare NaturalQuestions open (closed-book) for evaluation.
    Splits the validation set into val/lr/test for n_val/extra-dev/final-eval.
    Same Q->A format as nq_open_train, with answer aliases stored in metadata.
    """
    print("Preparing NaturalQuestions (nq_open) eval splits...")
    ds = shuffled_examples(load_dataset("nq_open", split="validation"))  # 3.6k, shuffled

    output_dir_nq = os.path.join(output_dir, "eval", "nq_open")
    ensure_dir(output_dir_nq)

    val_size = 100
    lr_size = 100
    test_size = 1000

    val_records, lr_records, test_records = [], [], []
    for idx, ex in enumerate(ds):
        question = ex["question"]
        answers = ex["answer"]  # list of valid answer strings
        if not answers:
            continue
        primary = answers[0]

        if len(val_records) < val_size:
            split, bucket = "val", val_records
        elif len(lr_records) < lr_size:
            split, bucket = "lr", lr_records
        elif len(test_records) < test_size:
            split, bucket = "test", test_records
        else:
            break

        rec = {
            "dataset": "nq_open",
            "id": f"nq_open_{split}_{len(bucket)}",
            "messages": [
                {"role": "user", "content": f"Answer the following question.\nQuestion: {question}"},
                {"role": "assistant", "content": primary},
            ],
            "metadata": {
                "primary_answer": primary,
                "aliases": list(set(answers)),
            },
        }
        bucket.append(rec)

    val_file = os.path.join(output_dir_nq, "nq_open_validation_data.jsonl")
    lr_file = os.path.join(output_dir_nq, "nq_open_lr_data.jsonl")
    test_file = os.path.join(output_dir_nq, "nq_open_test_data.jsonl")
    for fname, recs in [(val_file, val_records), (lr_file, lr_records), (test_file, test_records)]:
        with open(fname, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"NQ-open eval data saved:")
    print(f"  Validation: {val_file} ({len(val_records)})")
    print(f"  lr (dev):   {lr_file} ({len(lr_records)})")
    print(f"  Test:       {test_file} ({len(test_records)})")
    return val_file, lr_file, test_file


def prepare_triviaqa_train(output_dir):
    """
    Prepare TriviaQA closed-book (rc.nocontext) train split as a SFT training pool.
    Q -> A pairs, ~78K examples. Same format as nq_open_train so curation
    (TriviaQA -> NQ) parallels (NQ -> TriviaQA).
    """
    print("Preparing TriviaQA (rc.nocontext) training data...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")

    output_file = os.path.join(output_dir, "train", "triviaqa", "triviaqa_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    n = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(ds)):
            question = ex["question"]
            ans = ex["answer"]
            primary = ans.get("value", "") if isinstance(ans, dict) else ""
            if not primary:
                continue
            data = {
                "dataset": "triviaqa",
                "id": f"triviaqa_train_{idx}",
                "messages": [
                    {"role": "user", "content": f"Answer the following question.\nQuestion: {question}"},
                    {"role": "assistant", "content": primary},
                ],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            n += 1
    print(f"TriviaQA training data saved to {output_file} ({n} examples)")
    return output_file


def prepare_triviaqa_eval(output_dir):
    """
    Prepare TriviaQA closed-book (rc.nocontext) eval splits.
    Splits validation set into val/lr/test for n_val/extra-dev/final-eval.
    Each record has metadata.aliases (the answer aliases) for best-alias scoring.
    """
    print("Preparing TriviaQA (rc.nocontext) eval splits...")
    ds = shuffled_examples(load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation"))

    output_dir_tq = os.path.join(output_dir, "eval", "triviaqa")
    ensure_dir(output_dir_tq)

    val_size, lr_size, test_size = 100, 100, 1000
    val_records, lr_records, test_records = [], [], []
    for ex in ds:
        question = ex["question"]
        ans = ex["answer"]
        primary = ans.get("value", "") if isinstance(ans, dict) else ""
        aliases = ans.get("aliases", []) if isinstance(ans, dict) else []
        # de-dup, drop empties; ensure primary is included
        seen = set()
        clean = []
        for a in [primary] + list(aliases):
            if a and a not in seen:
                seen.add(a); clean.append(a)
        if not clean:
            continue

        if len(val_records) < val_size:
            split, bucket = "val", val_records
        elif len(lr_records) < lr_size:
            split, bucket = "lr", lr_records
        elif len(test_records) < test_size:
            split, bucket = "test", test_records
        else:
            break

        rec = {
            "dataset": "triviaqa",
            "id": f"triviaqa_{split}_{len(bucket)}",
            "messages": [
                {"role": "user", "content": f"Answer the following question.\nQuestion: {question}"},
                {"role": "assistant", "content": clean[0]},
            ],
            "metadata": {"primary_answer": clean[0], "aliases": clean},
        }
        bucket.append(rec)

    val_file = os.path.join(output_dir_tq, "triviaqa_validation_data.jsonl")
    lr_file = os.path.join(output_dir_tq, "triviaqa_lr_data.jsonl")
    test_file = os.path.join(output_dir_tq, "triviaqa_test_data.jsonl")
    for fname, recs in [(val_file, val_records), (lr_file, lr_records), (test_file, test_records)]:
        with open(fname, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TriviaQA eval data saved:")
    print(f"  Validation: {val_file} ({len(val_records)})")
    print(f"  lr (dev):   {lr_file} ({len(lr_records)})")
    print(f"  Test:       {test_file} ({len(test_records)})")
    return val_file, lr_file, test_file


def prepare_squad_eval(output_dir):
    """
    Prepare SQuAD as a closed-book QA eval target (no context).
    Strips the context from each SQuAD validation example so the model is
    asked Q -> A only. Same Q->A messages format as nq_open eval.
    answers.text (1-3 clean strings) is stored as `aliases` in metadata.
    """
    print("Preparing SQuAD eval splits (no context)...")
    ds = shuffled_examples(load_dataset("rajpurkar/squad", split="validation"))  # ~10.5K, shuffled

    output_dir_squad = os.path.join(output_dir, "eval", "squad")
    ensure_dir(output_dir_squad)

    val_size = 100
    lr_size = 100
    test_size = 1000

    val_records, lr_records, test_records = [], [], []
    for idx, ex in enumerate(ds):
        question = ex["question"]
        answers = ex["answers"]["text"]
        if not answers:
            continue
        primary = answers[0]

        if len(val_records) < val_size:
            split, bucket = "val", val_records
        elif len(lr_records) < lr_size:
            split, bucket = "lr", lr_records
        elif len(test_records) < test_size:
            split, bucket = "test", test_records
        else:
            break

        rec = {
            "dataset": "squad",
            "id": f"squad_{split}_{len(bucket)}",
            "messages": [
                {"role": "user", "content": f"Answer the following question.\nQuestion: {question}"},
                {"role": "assistant", "content": primary},
            ],
            "metadata": {
                "primary_answer": primary,
                "aliases": list(set(answers)),
            },
        }
        bucket.append(rec)

    val_file = os.path.join(output_dir_squad, "squad_validation_data.jsonl")
    lr_file = os.path.join(output_dir_squad, "squad_lr_data.jsonl")
    test_file = os.path.join(output_dir_squad, "squad_test_data.jsonl")
    for fname, recs in [(val_file, val_records), (lr_file, lr_records), (test_file, test_records)]:
        with open(fname, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"SQuAD (no-context) eval data saved:")
    print(f"  Validation: {val_file} ({len(val_records)})")
    print(f"  lr (dev):   {lr_file} ({len(lr_records)})")
    print(f"  Test:       {test_file} ({len(test_records)})")
    return val_file, lr_file, test_file


def prepare_squad(output_dir):
    """
    Prepare SQuAD as a SFT training pool (with context).
    Reading-comprehension format: (context + question) -> answer.
    """
    print("Preparing SQuAD training data...")
    ds = load_dataset("rajpurkar/squad", split="train")  # 87.6k

    output_file = os.path.join(output_dir, "train", "squad", "squad_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    n = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, ex in enumerate(tqdm(ds)):
            question = ex["question"]
            context = ex["context"]
            answers = ex["answers"]["text"]
            if not answers:
                continue
            answer = answers[0]
            user_content = (
                f"Answer the question based on the given context.\n\n"
                f"Context: {context}\n\nQuestion: {question}"
            )
            data = {
                "dataset": "squad",
                "id": f"squad_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer},
                ],
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            n += 1
    print(f"SQuAD training data saved to {output_file} ({n} examples)")
    return output_file


def prepare_tydiqa(output_dir):
    """
    Prepare TyDiQA validation, lr (extra dev), and test data in unified JSONL format.

    Uses the HuggingFace validation split and divides into validation/lr/test.
    Preserves the few-shot prompts file (tydiqa_fewshot.json) for downstream evaluation.

    Output files:
    - tydiqa_validation_data.jsonl: For data selection
    - tydiqa_lr_data.jsonl: Extra held-out dev split (small)
    - tydiqa_test_data.jsonl: For final evaluation
    - tydiqa_fewshot.json: One-shot prompts (preserved/renamed)
    """
    print("Preparing TyDiQA validation, LR, and test data...")

    output_dir_tydiqa = os.path.join(output_dir, "eval", "tydiqa")
    ensure_dir(output_dir_tydiqa)

    existing_one_shot = os.path.join(output_dir_tydiqa, "tydiqa-one-shot.json")

    # Load TyDiQA goldp (gold passage) task
    print("Loading TyDiQA from HuggingFace...")
    dataset = load_dataset("tydiqa", "secondary_task")

    # Use validation split (TyDiQA doesn't have a test split in secondary_task).
    # Shuffle before slicing — TyDiQA's HF validation order is grouped by language,
    # so the first 500 are all Arabic without this shuffle.
    all_data = shuffled_examples(dataset["validation"])

    # Split: first 100 for validation, next 100 for the extra dev partition, rest for test
    val_size = 100
    lr_size = 100
    val_data = all_data[:val_size]
    lr_data = all_data[val_size:val_size + lr_size]
    test_data = all_data[val_size + lr_size:]

    def format_qa(example):
        context = example.get('context', '')
        question = example.get('question', '')
        answers = example.get('answers', {})
        # Get the first answer text
        answer_texts = answers.get('text', [])
        answer = answer_texts[0] if answer_texts else ''

        user_content = f"Answer the question based on the given context.\n\nContext: {context}\n\nQuestion: {question}"
        return user_content, answer

    # Write validation data
    val_file = os.path.join(output_dir_tydiqa, "tydiqa_validation_data.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(val_data, desc="Validation")):
            user_content, answer = format_qa(example)
            data = {
                "dataset": "tydiqa",
                "id": f"tydiqa_val_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {
                    "language": example.get('id', '').split('-')[0] if example.get('id') else 'unknown'
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Write lr (dev) split
    lr_file = os.path.join(output_dir_tydiqa, "tydiqa_lr_data.jsonl")
    with open(lr_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(lr_data, desc="LR")):
            user_content, answer = format_qa(example)
            data = {
                "dataset": "tydiqa",
                "id": f"tydiqa_lr_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {
                    "language": example.get('id', '').split('-')[0] if example.get('id') else 'unknown'
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Write test data
    test_file = os.path.join(output_dir_tydiqa, "tydiqa_test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(test_data, desc="Test")):
            user_content, answer = format_qa(example)
            data = {
                "dataset": "tydiqa",
                "id": f"tydiqa_test_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": answer}
                ],
                "metadata": {
                    "language": example.get('id', '').split('-')[0] if example.get('id') else 'unknown'
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Handle few-shot prompts file (organized by language for multilingual evaluation)
    fewshot_file = os.path.join(output_dir_tydiqa, "tydiqa_fewshot.json")
    if os.path.exists(existing_one_shot) and not os.path.exists(fewshot_file):
        shutil.copy(existing_one_shot, fewshot_file)
        print(f"  Few-shot prompts copied to: {fewshot_file}")
    elif os.path.exists(fewshot_file):
        print(f"  Few-shot prompts already exist: {fewshot_file}")
    else:
        # Generate fewshot file organized by language (one example per language)
        print("  Generating few-shot prompts organized by language...")

        # Group examples by language
        examples_by_lang = {}
        for example in all_data:
            # Extract language from ID (format: "lang-xxx-yyy")
            example_id = example.get('id', '')
            lang = example_id.split('-')[0] if example_id else 'unknown'
            if lang not in examples_by_lang:
                examples_by_lang[lang] = []
            examples_by_lang[lang].append(example)

        # Create one-shot example for each language
        fewshot_data = {}
        for lang, examples in examples_by_lang.items():
            if examples:
                ex = examples[0]  # Use first example for each language
                context = ex.get('context', '')
                question = ex.get('question', '')
                answers = ex.get('answers', {})
                answer_texts = answers.get('text', [])
                answer = answer_texts[0] if answer_texts else ''

                fewshot_data[lang] = [{
                    "id": ex.get('id', ''),
                    "lang": lang,
                    "context": context,
                    "question": question,
                    "answers": [{"answer_start": answers.get('answer_start', [0])[0] if answers.get('answer_start') else 0,
                                 "text": answer}]
                }]

        with open(fewshot_file, 'w', encoding='utf-8') as f:
            json.dump(fewshot_data, f, ensure_ascii=False, indent=2)
        print(f"  Few-shot prompts generated: {fewshot_file} ({len(fewshot_data)} languages)")

    print(f"TyDiQA data saved:")
    print(f"  Validation: {val_file} ({len(val_data)} examples)")
    print(f"  lr (dev): {lr_file} ({len(lr_data)} examples)")
    print(f"  Test: {test_file} ({len(test_data)} examples)")
    return val_file, test_file


def prepare_alpaca(output_dir):
    """Prepare Alpaca instruction-following dataset."""
    print("Preparing Alpaca training data...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    output_file = os.path.join(output_dir, "train", "alpaca", "alpaca_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            instruction = example['instruction']
            input_text = example.get('input', '')
            output_text = example['output']

            # Combine instruction and input
            if input_text:
                user_content = f"{instruction}\n\nInput: {input_text}"
            else:
                user_content = instruction

            data = {
                "dataset": "alpaca",
                "id": f"alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": output_text}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Alpaca training data saved to {output_file}")
    return output_file


def prepare_dolly(output_dir):
    """Prepare Databricks Dolly dataset."""
    print("Preparing Dolly training data...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    output_file = os.path.join(output_dir, "train", "dolly", "dolly_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            instruction = example['instruction']
            context = example.get('context', '')
            response = example['response']

            # Combine instruction and context
            if context:
                user_content = f"{instruction}\n\nContext: {context}"
            else:
                user_content = instruction

            data = {
                "dataset": "dolly",
                "id": f"dolly_{idx}",
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Dolly training data saved to {output_file}")
    return output_file


def prepare_flan_v2(output_dir):
    """Prepare FLAN-v2 dataset."""
    print("Preparing FLAN-v2 training data...")
    print("Note: FLAN-v2 is very large. Using a subset for practicality.")

    # Using a subset of FLAN - the full dataset is too large
    dataset = load_dataset("Muennighoff/flan", split="train", streaming=True)

    output_file = os.path.join(output_dir, "train", "flan_v2", "flan_v2_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    max_examples = 100000  # Limit to 100k examples
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset, total=max_examples)):
            if idx >= max_examples:
                break

            inputs = example.get('inputs', '')
            targets = example.get('targets', '')

            data = {
                "dataset": "flan_v2",
                "id": f"flan_v2_{idx}",
                "messages": [
                    {"role": "user", "content": inputs},
                    {"role": "assistant", "content": targets}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"FLAN-v2 training data saved to {output_file}")
    return output_file


def prepare_cot(output_dir):
    """Prepare Chain-of-Thought dataset.

    `kaist-ai/CoT-Collection` is published as a dataset script, which `datasets`
    versions >= 3.0 refuse to execute. Use the auto-converted parquet revision
    (HF maintains this on `refs/convert/parquet` for any script-based dataset)
    so loading works without enabling code execution.
    """
    print("Preparing CoT training data...")
    dataset = load_dataset(
        "kaist-ai/CoT-Collection", revision="refs/convert/parquet", split="train"
    )

    output_file = os.path.join(output_dir, "train", "cot", "cot_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            source = example.get('source', '')
            rationale = example.get('rationale', '')

            data = {
                "dataset": "cot",
                "id": f"cot_{idx}",
                "messages": [
                    {"role": "user", "content": source},
                    {"role": "assistant", "content": rationale}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"CoT training data saved to {output_file}")
    return output_file


def prepare_oasst1(output_dir):
    """Prepare Open Assistant dataset."""
    print("Preparing OASST1 training data...")
    dataset = load_dataset("OpenAssistant/oasst1", split="train")

    output_file = os.path.join(output_dir, "train", "oasst1", "oasst1_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    # OASST1 has a tree structure, we need to flatten conversations
    # Group by message_tree_id to get full conversations
    conversations = {}
    for example in dataset:
        tree_id = example['message_tree_id']
        if tree_id not in conversations:
            conversations[tree_id] = []
        conversations[tree_id].append(example)

    with open(output_file, 'w', encoding='utf-8') as f:
        idx = 0
        for tree_id, messages in tqdm(conversations.items()):
            # Sort by parent_id to reconstruct conversation order
            # For simplicity, take user-assistant pairs
            messages = sorted(messages, key=lambda x: x.get('created_date', ''))

            current_messages = []
            for msg in messages:
                role = msg['role']
                text = msg['text']

                if role == 'prompter':
                    current_messages.append({"role": "user", "content": text})
                elif role == 'assistant':
                    current_messages.append({"role": "assistant", "content": text})

            # Only save if we have at least one exchange
            if len(current_messages) >= 2:
                data = {
                    "dataset": "oasst1",
                    "id": f"oasst1_{idx}",
                    "messages": current_messages
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
                idx += 1

    print(f"OASST1 training data saved to {output_file}")
    return output_file


def prepare_samsum(output_dir):
    """
    Prepare SamSUM dialogue summarization dataset.

    Checks if data already exists in the expected format before downloading.

    Output files:
    - samsum_train_data.jsonl: Training data
    - samsum_validation_data.jsonl: For data selection
    - samsum_lr_data.jsonl: Extra held-out dev split (small)
    - samsum_test_data.jsonl: For final evaluation
    """
    print("Preparing SamSUM data...")

    train_file = os.path.join(output_dir, "train", "samsum", "samsum_train_data.jsonl")
    val_file = os.path.join(output_dir, "eval", "samsum", "samsum_validation_data.jsonl")
    lr_file = os.path.join(output_dir, "eval", "samsum", "samsum_lr_data.jsonl")
    test_file = os.path.join(output_dir, "eval", "samsum", "samsum_test_data.jsonl")

    # Check if data already exists
    if os.path.exists(val_file) and os.path.exists(lr_file) and os.path.exists(test_file):
        print(f"SamSUM evaluation data already exists:")
        print(f"  Validation: {val_file}")
        print(f"  lr (dev): {lr_file}")
        print(f"  Test: {test_file}")

        # Check if train data exists
        if os.path.exists(train_file):
            print(f"  Train: {train_file}")
        else:
            print(f"  Train: Not found (run with --datasets samsum to download)")

        return train_file if os.path.exists(train_file) else val_file

    # Try to download from HuggingFace
    print("Downloading SamSUM from HuggingFace...")
    ensure_dir(os.path.dirname(train_file))
    ensure_dir(os.path.dirname(val_file))

    # Load from knkarthick/samsum (shuffle val + test for consistency with other tasks)
    train_dataset = load_dataset("knkarthick/samsum", split="train")
    val_dataset = shuffled_examples(load_dataset("knkarthick/samsum", split="validation"))
    test_dataset = shuffled_examples(load_dataset("knkarthick/samsum", split="test"))

    # Split test into lr-dev (first 100) and final test (rest)
    test_list = list(test_dataset)
    lr_size = 100
    lr_examples = test_list[:lr_size]
    test_examples = test_list[lr_size:]

    # Training data
    with open(train_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(train_dataset, desc="Train")):
            dialogue = example['dialogue']
            summary = example['summary']

            data = {
                "dataset": "samsum",
                "id": f"samsum_train_{idx}",
                "messages": [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Validation data
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(val_dataset, desc="Validation")):
            dialogue = example['dialogue']
            summary = example['summary']

            data = {
                "dataset": "samsum",
                "id": f"samsum_val_{idx}",
                "messages": [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # lr (dev) split
    with open(lr_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(lr_examples, desc="LR")):
            dialogue = example['dialogue']
            summary = example['summary']

            data = {
                "dataset": "samsum",
                "id": f"samsum_lr_{idx}",
                "messages": [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Test data
    with open(test_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(test_examples, desc="Test")):
            dialogue = example['dialogue']
            summary = example['summary']

            data = {
                "dataset": "samsum",
                "id": f"samsum_test_{idx}",
                "messages": [
                    {"role": "user", "content": f"Summarize the following dialogue:\n\n{dialogue}"},
                    {"role": "assistant", "content": summary}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"SamSUM data saved:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  lr (dev): {lr_file} ({len(lr_examples)} examples)")
    print(f"  Test: {test_file} ({len(test_examples)} examples)")
    return train_file


def prepare_tulu3(output_dir):
    """Prepare Tulu-3 SFT mixture training data."""
    print("Preparing Tulu-3 training data...")
    dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train")

    output_file = os.path.join(output_dir, "train", "tulu3", "tulu3_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            messages_raw = example.get('messages', [])

            messages = []
            for msg in messages_raw:
                role = msg.get('role', 'user')
                if role == 'system':
                    continue  # Skip system messages
                content = msg.get('content', '')
                messages.append({"role": role, "content": content})

            if len(messages) >= 2:
                data = {
                    "dataset": "tulu3",
                    "id": f"tulu3_{idx}",
                    "messages": messages
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Tulu-3 training data saved to {output_file}")
    return output_file


def prepare_truthfulqa(output_dir):
    """
    Prepare TruthfulQA (domenicrosati/TruthfulQA) validation, LR sweep, and test data.

    The dataset has a single 'train' split with 817 examples. We split it into:
    - validation: first 50 examples (for data curation during training)
    - lr: next 100 examples (for LR sweep)
    - test: remaining ~667 examples (for final evaluation)

    Each example is formatted as a Question -> Best Answer pair using the
    unified 'messages' format. Other fields (Type, Category, Correct/Incorrect
    Answers, Source) are preserved in metadata.
    """
    print("Preparing TruthfulQA validation, LR, and test data...")

    output_dir_truthfulqa = os.path.join(output_dir, "eval", "truthfulqa")
    ensure_dir(output_dir_truthfulqa)

    print("Loading domenicrosati/TruthfulQA from HuggingFace...")
    dataset = load_dataset("domenicrosati/TruthfulQA", split="train")
    all_data = list(dataset)

    val_size = 50
    lr_size = 100
    val_data = all_data[:val_size]
    lr_data = all_data[val_size:val_size + lr_size]
    test_data = all_data[val_size + lr_size:]

    def format_qa(example):
        question = example.get('Question', '')
        answer = example.get('Best Answer', '')
        user_content = f"Answer the following question truthfully and concisely.\n\nQuestion: {question}"
        return user_content, answer

    def write_split(file_path, data, split_name):
        with open(file_path, 'w', encoding='utf-8') as f:
            for idx, example in enumerate(tqdm(data, desc=split_name.capitalize())):
                user_content, answer = format_qa(example)
                entry = {
                    "dataset": "truthfulqa",
                    "id": f"truthfulqa_{split_name}_{idx}",
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {
                        "type": example.get('Type', ''),
                        "category": example.get('Category', ''),
                        "correct_answers": example.get('Correct Answers', ''),
                        "incorrect_answers": example.get('Incorrect Answers', ''),
                        "source": example.get('Source', '')
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    val_file = os.path.join(output_dir_truthfulqa, "truthfulqa_validation_data.jsonl")
    lr_file = os.path.join(output_dir_truthfulqa, "truthfulqa_lr_data.jsonl")
    test_file = os.path.join(output_dir_truthfulqa, "truthfulqa_test_data.jsonl")

    write_split(val_file, val_data, "val")
    write_split(lr_file, lr_data, "lr")
    write_split(test_file, test_data, "test")

    print(f"TruthfulQA data saved:")
    print(f"  Validation: {val_file} ({len(val_data)} examples)")
    print(f"  LR sweep: {lr_file} ({len(lr_data)} examples)")
    print(f"  Test: {test_file} ({len(test_data)} examples)")
    return val_file, lr_file, test_file


# =============================================================================
# Downstream benchmarks (Dolci capability setting)
#
# A *benchmark* file holds prompts plus the metadata its official verifier needs.
# It is scored by generation (SFT/eval/eval.sh) and is a different object from a
# *target* split (validation = D*, test = loss held-out). Layout:
#   eval/<bench>/<bench>_bench_data.jsonl
# Revisions are pinned.
# =============================================================================

BENCHMARK_PINS = {
    "ifeval": {"repo": "google/IFEval", "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84", "split": "train"},
    "ifbench": {"repo": "allenai/IFBench_test", "revision": "2e8a48de45ff3bf41242f927254ca81b59ca3ae2", "split": "train"},
    "math500": {"repo": "HuggingFaceH4/MATH-500", "revision": "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be", "split": "test"},
    "gsm8k": {"repo": "openai/gsm8k", "revision": "740312add88f781978c0658806c59bc2815b9866", "split": "test", "config": "main"},
}
MBPP_PLUS_DATASET_VERSION = "v0.2.0"


def _bench_path(output_dir, task):
    path = os.path.join(output_dir, "eval", task, f"{task}_bench_data.jsonl")
    ensure_dir(os.path.dirname(path))
    return path


def _write_bench(path, rows, task):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{task} benchmark saved: {path} ({len(rows)} rows)")
    return path


def _clean_kwargs(raw):
    if not isinstance(raw, (list, tuple)):
        return []
    return [{k: v for k, v in entry.items() if v is not None} if isinstance(entry, dict) else {} for entry in raw]


def prepare_if_benchmark(output_dir, task):
    """IFEval / IFBench: prompt + instruction_id_list + kwargs for the official verifiers."""
    pin = BENCHMARK_PINS[task]
    print(f"Preparing {task} benchmark from {pin['repo']}@{pin['revision'][:8]}...")
    dataset = load_dataset(pin["repo"], split=pin["split"], revision=pin["revision"])
    rows = []
    for idx, example in enumerate(dataset):
        prompt = (example.get("prompt") or "").strip()
        if not prompt:
            continue
        key = example.get("key", idx)
        rows.append({
            "dataset": task,
            "id": f"{task}::{key}",
            "messages": [{"role": "user", "content": prompt}],
            "metadata": {
                "key": key,
                "prompt": prompt,
                "instruction_id_list": list(example.get("instruction_id_list") or []),
                "kwargs": _clean_kwargs(example.get("kwargs")),
                "source_repo": pin["repo"],
                "source_revision": pin["revision"],
                "source_split": pin["split"],
            },
        })
    return _write_bench(_bench_path(output_dir, task), rows, task)


def prepare_math500_bench(output_dir):
    """MATH-500 (all 500 problems) with the gold answer for math-verify / boxed matching."""
    pin = BENCHMARK_PINS["math500"]
    print(f"Preparing math500 benchmark from {pin['repo']}@{pin['revision'][:8]}...")
    dataset = load_dataset(pin["repo"], split=pin["split"], revision=pin["revision"])
    rows = []
    for idx, example in enumerate(dataset):
        problem = (example.get("problem") or "").strip()
        if not problem:
            continue
        unique_id = str(example.get("unique_id") or idx)
        rows.append({
            "dataset": "math500",
            "id": f"math500::{unique_id}",
            "messages": [{"role": "user", "content": problem}],
            "metadata": {
                "problem": problem,
                "answer": example.get("answer"),
                "solution": example.get("solution"),
                "subject": example.get("subject"),
                "level": example.get("level"),
                "unique_id": unique_id,
                "source_repo": pin["repo"],
                "source_revision": pin["revision"],
                "source_split": pin["split"],
            },
        })
    return _write_bench(_bench_path(output_dir, "math500"), rows, "math500")


def prepare_gsm8k_bench(output_dir):
    """GSM8K test (1319 problems); gold answer = the number after '####' in the reference solution."""
    pin = BENCHMARK_PINS["gsm8k"]
    print(f"Preparing gsm8k benchmark from {pin['repo']}@{pin['revision'][:8]}...")
    dataset = load_dataset(pin["repo"], pin["config"], split=pin["split"], revision=pin["revision"])
    rows = []
    for idx, example in enumerate(dataset):
        problem = (example.get("question") or "").strip(); solution = (example.get("answer") or "").strip()
        if not problem or "####" not in solution:
            continue
        answer = solution.rsplit("####", 1)[1].strip().replace(",", "")
        rows.append({"dataset": "gsm8k", "id": f"gsm8k::{idx}", "messages": [{"role": "user", "content": problem}],
                     "metadata": {"problem": problem, "answer": answer, "solution": solution, "source_repo": pin["repo"],
                                  "source_revision": pin["revision"], "source_split": pin["split"]}})
    return _write_bench(_bench_path(output_dir, "gsm8k"), rows, "gsm8k")


def prepare_mbpp_plus_bench(output_dir):
    """MBPP+ tasks from the evalplus package (needs `pip install evalplus`).

    ``canonical_solution`` is stored so that limited smoke runs can fill the
    unscored tasks EvalPlus insists on receiving.
    """
    try:
        from evalplus.data import get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError("mbpp_plus preparation requires the evalplus package: pip install evalplus==0.3.1") from exc
    import importlib.metadata
    version = importlib.metadata.version("evalplus")
    print(f"Preparing mbpp_plus benchmark with evalplus {version} (dataset {MBPP_PLUS_DATASET_VERSION})...")
    tasks = get_mbpp_plus(version=MBPP_PLUS_DATASET_VERSION)
    rows = []
    for task_id in sorted(tasks):
        task = dict(tasks[task_id])
        prompt = (task.get("prompt") or "").strip()
        if not prompt:
            continue
        metadata = json.loads(json.dumps(task, ensure_ascii=False, default=str))
        metadata.update({
            "task_id": str(task_id),
            "evalplus_version": version,
            "evalplus_dataset_version": MBPP_PLUS_DATASET_VERSION,
            "source_repo": "evalplus/mbppplus",
            "source_revision": MBPP_PLUS_DATASET_VERSION,
        })
        rows.append({
            "dataset": "mbpp_plus",
            "id": f"mbpp_plus::{task_id}",
            "messages": [{"role": "user", "content": prompt}],
            "metadata": metadata,
        })
    return _write_bench(_bench_path(output_dir, "mbpp_plus"), rows, "mbpp_plus")


# =============================================================================
# Dolci capability setting: train pools + targets
#
# Benchmarks are reserved first, targets second, general pools last.
#   * Pools are 32,000-row uniform samples from allenai/Dolci-Instruct-SFT over a
#     domain group (DOLCI_POOL_DOMAINS), restricted to plain user/assistant chats
#     (no tool-use payloads) of <= DOLCI_MAX_CHARS characters.
#   * Targets are small "messages" splits: `validation` = D* (n_val rows drive
#     selection + the val_loss curve), `test` = loss-only held-out (n_eval rows).
#     precise_if comes from the Dolci "Precise IF" source (as in Next), math from
#     MATH train, mbpp from MBPP train minus every MBPP+ task id.
#   * Decontamination (ported from Next): a row is dropped if its normalised user
#     prompt equals a reference prompt, or if >= DECONTAM_THRESHOLD of its word
#     8-grams are shared with one reference prompt. Targets are decontaminated
#     against the four benchmarks; pools against the benchmarks AND all targets,
#     and precise_if target rows are additionally excluded from pools by id.
# =============================================================================

DOLCI_PIN = {
    "repo": "allenai/Dolci-Instruct-SFT",
    "revision": "bd3c8f3a9b2cc5a9682e44b96ddd0bb2ff027221",
    "split": "train",
}
MATH_TRAIN_PIN = {
    "repo": "EleutherAI/hendrycks_math",
    "revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "configs": ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                "number_theory", "prealgebra", "precalculus"],
    "split": "train",
}
MBPP_TRAIN_PIN = {
    "repo": "google-research-datasets/mbpp",
    "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
    "config": "full",
    "split": "train",
}

DOLCI_POOL_SIZE = 32_000
DOLCI_SAMPLE_SEED = 42
DOLCI_MAX_CHARS = 16_000          # ~4k tokens; longer rows cannot fit max_seq_length=4096
DOLCI_EXCLUDED_DOMAINS = {"Tool Use", "Hardcoded Data"}
DOLCI_INSTRUCTION_DOMAINS = {"Chat", "Precise IF", "Other", "Multilingual", "Safety"}
DOLCI_REASONING_DOMAINS = {"Math", "Coding", "Reasoning", "Science"}
DOLCI_POOL_DOMAINS = {
    "dolci_instruction": DOLCI_INSTRUCTION_DOMAINS,
    "dolci_reasoning": DOLCI_REASONING_DOMAINS,
    "dolci_mixed": DOLCI_INSTRUCTION_DOMAINS | DOLCI_REASONING_DOMAINS,
}
PRECISE_IF_SOURCE = "Dolci Instruct Precise IF"
PRECISE_IF_MIN_ASCII_RATIO = 0.95  # IFEval/IFBench are English; keep D* English too

DOLCI_BENCHMARKS = ("ifeval", "ifbench", "math500", "gsm8k", "mbpp_plus")
DOLCI_TARGETS = ("precise_if", "math_ref", "mbpp")   # pools are decontaminated against these
DOLCI_AUDIT_TARGETS = DOLCI_TARGETS + ("math", "math_v2", "math_bench", "math_pool", "math_persona")   # built by SFT/data/build_math{,_pool,_persona}_target.py
DECONTAM_NGRAM = 8
DECONTAM_THRESHOLD = 0.8   # shared unique 8-grams / min(candidate, reference)

# Target splits: validation = D* (>= 3*n_val so length rejection has slack), test = held-out.
# None = "everything that is left" (MBPP train has 374 rows, 108 of which are MBPP+ tasks).
TARGET_SPLIT_SIZES = {
    "precise_if": {"validation": 128, "test": 472},   # 600 reserved rows total (all excluded from the pools)
    "math_ref": {"validation": 64, "test": 500},    # MATH-train references; only used for pool decontamination
    "math_ref128": {"validation": 128, "test": None},  # math_ref re-split 128 / 436: base of the curated math_ref128_gen32b target
    "mbpp": {"validation": 128, "test": None},
}


try:
    from SFT.data.decontam import PromptDecontaminator, user_prompts as _user_prompts
except ImportError:  # run as `python SFT/data/prepare_datasets.py` (repo root not on sys.path)
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from SFT.data.decontam import PromptDecontaminator, user_prompts as _user_prompts


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _reference_prompts(output_dir, benchmarks=DOLCI_BENCHMARKS, targets=()):
    """(id, user prompt) pairs from benchmark files and target splits already on disk."""
    refs = []
    for bench in benchmarks:
        path = _bench_path(output_dir, bench)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"benchmark file missing: {path} (run --datasets {bench} first)")
        for row in _read_jsonl(path):
            for i, p in enumerate(_user_prompts(row.get("messages"))):
                refs.append((f"{bench}:{row.get('id')}:{i}", p))
    for target in targets:
        for split in ("validation", "test"):
            path = os.path.join(output_dir, "eval", target, f"{target}_{split}_data.jsonl")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"target split missing: {path} (run --datasets {target} first)")
            for row in _read_jsonl(path):
                for i, p in enumerate(_user_prompts(row.get("messages"))):
                    refs.append((f"{target}/{split}:{row.get('id')}:{i}", p))
    return refs


def _dolci_clean_messages(messages):
    """Return [{role, content}] if the row is a plain chat conversation, else None.

    Accepts an optional leading system turn, then strictly alternating
    user/assistant turns ending with a non-empty assistant turn. Rows carrying
    tool-use payloads (function_calls / functions) are rejected.
    """
    if not messages:
        return None
    cleaned = []
    for m in messages:
        if (m.get("function_calls") or "").strip() or (m.get("functions") or "").strip():
            return None
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role not in ("system", "user", "assistant") or not content:
            return None
        cleaned.append({"role": role, "content": content})
    start = 1 if cleaned[0]["role"] == "system" else 0
    turns = cleaned[start:]
    if len(turns) < 2 or turns[-1]["role"] != "assistant":
        return None
    for i, m in enumerate(turns):
        if m["role"] != ("user" if i % 2 == 0 else "assistant"):
            return None
    return cleaned


def _dolci_row_features(example):
    """Per-row features used by the pool/target filters (runs inside Dataset.map)."""
    cleaned = _dolci_clean_messages(example.get("messages") or [])
    if cleaned is None:
        return {"eligible": False, "n_chars": 0, "n_turns": 0, "ascii_ratio": 0.0}
    n_chars = sum(len(m["content"]) for m in cleaned)
    text = "".join(m["content"] for m in cleaned)
    ascii_ratio = sum(1 for ch in text if ord(ch) < 128) / max(1, len(text))
    return {
        "eligible": (example.get("domain") not in DOLCI_EXCLUDED_DOMAINS) and n_chars <= DOLCI_MAX_CHARS,
        "n_chars": n_chars,
        "n_turns": len([m for m in cleaned if m["role"] != "system"]),
        "ascii_ratio": ascii_ratio,
    }


def _write_messages_jsonl(path, rows, label):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  {label}: {path} ({len(rows)} rows)")
    return path


def _split_target(rows, target, output_dir):
    sizes = TARGET_SPLIT_SIZES[target]
    n_val = sizes["validation"]
    n_test = sizes["test"]
    if len(rows) < n_val + (n_test or 1):
        raise RuntimeError(f"{target}: only {len(rows)} clean rows; need {n_val} validation + held-out")
    val_rows = rows[:n_val]
    test_rows = rows[n_val:] if n_test is None else rows[n_val:n_val + n_test]
    out_dir = os.path.join(output_dir, "eval", target)
    return (
        _write_messages_jsonl(os.path.join(out_dir, f"{target}_validation_data.jsonl"), val_rows, "validation (D*)"),
        _write_messages_jsonl(os.path.join(out_dir, f"{target}_test_data.jsonl"), test_rows, "test (held-out)"),
    )


def _load_dolci(num_proc):
    pin = DOLCI_PIN
    print(f"Loading {pin['repo']}@{pin['revision'][:8]} ({pin['split']}) ...")
    dataset = load_dataset(pin["repo"], split=pin["split"], revision=pin["revision"])
    print(f"  {len(dataset):,} rows; computing row features with {num_proc} workers ...")
    dataset = dataset.map(_dolci_row_features, num_proc=num_proc, desc="dolci features")
    return dataset


def prepare_precise_if_target(output_dir, dataset=None, num_proc=16):
    """D* / held-out for the IFEval-style target from Dolci Precise-IF rows (as in Next).

    Single-turn, mostly-ASCII rows, decontaminated against the four benchmarks.
    Returns (paths, held_out_ids) so prepare_dolci_pools can exclude the rows by id.
    """
    if dataset is None:
        dataset = _load_dolci(num_proc)
    blocker = PromptDecontaminator(_reference_prompts(output_dir))
    sizes = TARGET_SPLIT_SIZES["precise_if"]
    n_total = sizes["validation"] + sizes["test"]
    pif = dataset.filter(
        lambda ex: ex["eligible"] and ex["source_dataset"] == PRECISE_IF_SOURCE
        and ex["n_turns"] == 2 and ex["ascii_ratio"] >= PRECISE_IF_MIN_ASCII_RATIO,
        num_proc=num_proc, desc="precise_if candidates",
    )
    print(f"precise_if: {len(pif):,} single-turn English Precise-IF rows; decontaminating against "
          f"{len(blocker)} benchmark prompts and sampling {n_total}")
    pif = pif.shuffle(seed=DOLCI_SAMPLE_SEED)

    rows, held_out_ids, blocked = [], set(), 0
    for example in pif:
        if len(rows) >= n_total:
            break
        cleaned = _dolci_clean_messages(example["messages"])
        if blocker.blocked_messages(cleaned):
            blocked += 1
            continue
        held_out_ids.add(example["id"])
        rows.append({
            "dataset": "precise_if",
            "id": f"precise_if::{example['id']}",
            "messages": cleaned,
            "metadata": {
                "source_id": example["id"],
                "source_dataset": example["source_dataset"],
                "domain": example["domain"],
                "source_repo": DOLCI_PIN["repo"],
                "source_revision": DOLCI_PIN["revision"],
            },
        })
    print(f"precise_if: dropped {blocked} benchmark-overlapping rows while selecting {len(rows)}")
    return _split_target(rows, "precise_if", output_dir), held_out_ids


def prepare_dolci_pools(output_dir, pools=None, num_proc=16):
    """Build the 32K-row Dolci training pools (and the precise_if target they exclude).

    Requires the benchmark files and the math/mbpp targets to exist already
    (`--datasets math mbpp ifeval ifbench math500 mbpp_plus`), because pools are
    decontaminated against all of them.
    """
    pools = list(pools or DOLCI_POOL_DOMAINS)
    dataset = _load_dolci(num_proc)
    _, held_out_ids = prepare_precise_if_target(output_dir, dataset=dataset, num_proc=num_proc)
    blocker = PromptDecontaminator(_reference_prompts(output_dir, targets=DOLCI_TARGETS))
    print(f"dolci: decontaminating pools against {len(blocker)} benchmark + target prompts")

    eligible = dataset.filter(
        lambda ex, ids=held_out_ids, b=blocker: ex["eligible"] and ex["id"] not in ids
        and not b.blocked_messages(ex["messages"]),
        num_proc=num_proc, desc="eligible rows",
    )
    print(f"dolci: {len(eligible):,} eligible rows after format/length/held-out/decontamination filtering")

    outputs = {}
    for pool in pools:
        domains = DOLCI_POOL_DOMAINS[pool]
        cand = eligible.filter(lambda ex, d=domains: ex["domain"] in d, num_proc=num_proc, desc=f"{pool} candidates")
        if len(cand) < DOLCI_POOL_SIZE:
            raise RuntimeError(f"{pool}: only {len(cand)} candidates for a {DOLCI_POOL_SIZE}-row pool")
        sampled = cand.shuffle(seed=DOLCI_SAMPLE_SEED).select(range(DOLCI_POOL_SIZE))
        rows, by_domain = [], {}
        for idx, example in enumerate(sampled):
            by_domain[example["domain"]] = by_domain.get(example["domain"], 0) + 1
            rows.append({
                "dataset": pool,
                "id": f"{pool}_{idx}",
                "messages": _dolci_clean_messages(example["messages"]),
                "metadata": {
                    "source_id": example["id"],
                    "source_dataset": example["source_dataset"],
                    "domain": example["domain"],
                },
            })
        print(f"{pool}: {len(cand):,} candidates -> {len(rows):,} rows; composition: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items(), key=lambda kv: -kv[1])))
        outputs[pool] = _write_messages_jsonl(
            os.path.join(output_dir, "train", pool, f"{pool}_data.jsonl"), rows, pool)
    return outputs


# =============================================================================
# Tulu 3 general SFT mixture pool
#
# `tulu3_general` = a 32,000-row uniform sample of allenai/tulu-3-sft-mixture, the
# standard general-purpose SFT recipe, filtered and decontaminated exactly like the
# Dolci pools (plain alternating chats <= DOLCI_MAX_CHARS chars; prompts blocked
# against the benchmarks and the precise_if / math_ref / mbpp / math_persona targets).
# Compared with dolci_mixed it dilutes precise IF (~3 % of rows vs 7 %) and
# concentrates math (~36 % vs 14 %), so the two targets bracket the "how much of the
# pool is on-target" question. The dataset is read from the pinned snapshot as parquet
# when it is on disk (shared read-only hub), otherwise through load_dataset(revision).
# =============================================================================
TULU3_PIN = {
    "repo": "allenai/tulu-3-sft-mixture",
    "revision": "b14afda60f1bbebe55d5d2fa1e4df5042f97f8be",
    "split": "train",
    "snapshot": os.environ.get(
        "TULU3_SNAPSHOT",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
                     "hub/datasets--allenai--tulu-3-sft-mixture/snapshots/b14afda60f1bbebe55d5d2fa1e4df5042f97f8be")),
}
TULU3_POOL = "tulu3_general"
TULU3_POOL_TARGETS = DOLCI_TARGETS + ("math_persona",)   # math_persona rows come from Tulu 3 persona sources
# Tulu 3 `source` -> domain label (reporting / metadata only; the sample is uniform over all sources)
TULU3_DOMAIN_OF = {
    "ai2-adapt-dev/personahub_math_v5_regen_149960": "Math",
    "ai2-adapt-dev/numinamath_tir_math_decontaminated": "Math",
    "ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k": "Math",
    "allenai/tulu-3-sft-personas-math-grade": "Math",
    "ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k": "Math",
    "ai2-adapt-dev/evol_codealpaca_heval_decontaminated": "Coding",
    "ai2-adapt-dev/personahub_code_v2_34999": "Coding",
    "ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980": "Precise IF",
    "ai2-adapt-dev/tulu_v3.9_wildchat_100k": "Chat",
    "ai2-adapt-dev/no_robots_converted": "Chat",
    "ai2-adapt-dev/oasst1_converted": "Chat",
    "ai2-adapt-dev/flan_v2_converted": "Other",
    "ai2-adapt-dev/tulu_v3.9_table_gpt_5k": "Other",
    "ai2-adapt-dev/tulu_hard_coded_repeated_10": "Other",
    "ai2-adapt-dev/tulu_v3.9_sciriff_10k": "Science",
    "ai2-adapt-dev/tulu_v3.9_aya_100k": "Multilingual",
    "ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k": "Safety",
    "ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k": "Safety",
    "ai2-adapt-dev/coconot_converted": "Safety",
}


def _load_tulu3(num_proc):
    pin = TULU3_PIN
    snapshot = pin["snapshot"]
    files = sorted(glob.glob(os.path.join(snapshot, "data", "*.parquet"))) if snapshot else []
    if files:
        print(f"Loading {pin['repo']}@{pin['revision'][:8]} from the local snapshot ({len(files)} parquet files) ...")
        dataset = load_dataset("parquet", data_files=files, split="train")
    else:
        print(f"Loading {pin['repo']}@{pin['revision'][:8]} ({pin['split']}) from the hub ...")
        dataset = load_dataset(pin["repo"], split=pin["split"], revision=pin["revision"])
    print(f"  {len(dataset):,} rows; computing row features with {num_proc} workers ...")
    return dataset.map(_dolci_row_features, num_proc=num_proc, desc="tulu3 features")


def prepare_tulu3_pool(output_dir, num_proc=16):
    """Build the 32K-row Tulu 3 general pool (see the module comment above).

    Requires the benchmark files and the precise_if / math_ref / mbpp / math_persona
    target splits on disk (the pool is decontaminated against all of them).
    """
    dataset = _load_tulu3(num_proc)
    blocker = PromptDecontaminator(_reference_prompts(output_dir, targets=TULU3_POOL_TARGETS))
    print(f"{TULU3_POOL}: decontaminating against {len(blocker)} benchmark + target prompts")
    eligible = dataset.filter(
        lambda ex, b=blocker: ex["eligible"] and not b.blocked_messages(ex["messages"]),
        num_proc=num_proc, desc="eligible rows",
    )
    print(f"{TULU3_POOL}: {len(eligible):,} of {len(dataset):,} rows eligible after format/length/decontamination filtering")
    if len(eligible) < DOLCI_POOL_SIZE:
        raise RuntimeError(f"{TULU3_POOL}: only {len(eligible)} candidates for a {DOLCI_POOL_SIZE}-row pool")
    sampled = eligible.shuffle(seed=DOLCI_SAMPLE_SEED).select(range(DOLCI_POOL_SIZE))
    rows, by_domain, by_source = [], {}, {}
    for idx, example in enumerate(sampled):
        source = example.get("source", "")
        domain = TULU3_DOMAIN_OF.get(source, "Other")
        by_domain[domain] = by_domain.get(domain, 0) + 1
        by_source[source] = by_source.get(source, 0) + 1
        rows.append({
            "dataset": TULU3_POOL,
            "id": f"{TULU3_POOL}_{idx}",
            "messages": _dolci_clean_messages(example["messages"]),
            "metadata": {
                "source_id": example["id"],
                "source_dataset": source,
                "domain": domain,
                "source_repo": TULU3_PIN["repo"],
                "source_revision": TULU3_PIN["revision"],
            },
        })
    print(f"{TULU3_POOL}: {len(rows):,} rows; domains: "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_domain.items(), key=lambda kv: -kv[1])))
    print(f"{TULU3_POOL}: sources: " + ", ".join(f"{k.split('/')[-1]}={v}" for k, v in sorted(by_source.items(), key=lambda kv: -kv[1])))
    return _write_messages_jsonl(os.path.join(output_dir, "train", TULU3_POOL, f"{TULU3_POOL}_data.jsonl"), rows, TULU3_POOL)


def prepare_math_target(output_dir):
    """D* / held-out for the MATH500 target from the MATH *train* split (all 7 subjects).

    User turn is the raw problem (the math500 evaluator adds its own instruction
    prefix at generation time, as in Next); rows are decontaminated against the benchmarks.
    """
    pin = MATH_TRAIN_PIN
    print(f"Preparing math target from {pin['repo']}@{pin['revision'][:8]} ({pin['split']}) ...")
    blocker = PromptDecontaminator(_reference_prompts(output_dir))
    examples = []
    for config in pin["configs"]:
        ds = load_dataset(pin["repo"], config, split=pin["split"], revision=pin["revision"])
        for ex in ds:
            ex = dict(ex); ex["_config"] = config
            examples.append(ex)
    examples = shuffled_examples(examples, seed=DOLCI_SAMPLE_SEED)
    sizes = TARGET_SPLIT_SIZES["math_ref"]
    rows, blocked = [], 0
    for ex in examples:
        problem = (ex.get("problem") or "").strip()
        solution = (ex.get("solution") or "").strip()
        if not problem or not solution or "\\boxed" not in solution:
            continue
        if blocker.match(problem) is not None:
            blocked += 1
            continue
        rows.append({
            "dataset": "math_ref",
            "id": f"math_ref::{ex['_config']}::{len(rows)}",
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "metadata": {
                "subject": ex.get("type") or ex["_config"],
                "level": str(ex.get("level") or ""),
                "source_repo": pin["repo"],
                "source_revision": pin["revision"],
                "source_split": pin["split"],
            },
        })
        if len(rows) >= sizes["validation"] + sizes["test"]:
            break
    print(f"math: dropped {blocked} benchmark-overlapping rows while selecting {len(rows)}")
    return _split_target(rows, "math_ref", output_dir)


def _mbpp_task_number(task_id):
    """`Mbpp/123` or `123` -> 123 (None if unparsable)."""
    tail = str(task_id).rsplit("/", 1)[-1]
    return int(tail) if tail.isdigit() else None


def render_mbpp_prompt(text, tests):
    """Standard MBPP prompt (same as Next's target rows)."""
    return (
        "You are an expert Python programmer, and here is your task:\n"
        f"{text.strip()}\nYour code should pass these tests:\n\n"
        + "\n".join(str(t) for t in tests) + "\n"
    )


def prepare_math_ref128_target(output_dir):
    """`math_ref128`: the 128-row MATH-train target from which the curated `math_ref128_gen32b` target ("Math (curated)"
    in the paper) is built by SFT/data/gen_target_candidates.py + build_rewrite_target.py.

    Rows = the 64 D* rows of `math_ref` followed by its 500 held-out rows, in file order, re-split 128 / 436, i.e. D* =
    the math_ref D* plus its first 64 held-out rows. Only `dataset`, the id prefix and `metadata.source_target` change,
    so every row keeps its MATH-train provenance and stays inside the pools' decontamination set.
    """
    src = os.path.join(output_dir, "eval", "math_ref")
    rows = []
    for split in ("validation", "test"):
        path = os.path.join(src, f"math_ref_{split}_data.jsonl")
        if not os.path.exists(path):
            raise RuntimeError(f"math_ref128 needs {path}; run --datasets math_ref first")
        for row in _read_jsonl(path):
            row = dict(row)
            row["dataset"] = "math_ref128"
            row["id"] = "math_ref128::" + row["id"].split("::", 1)[1]
            row["metadata"] = dict(row.get("metadata") or {}, source_target="math_ref")
            rows.append(row)
    print(f"math_ref128: {len(rows)} MATH-train rows from eval/math_ref (64 D* + 500 held-out), re-split 128 / {len(rows) - 128}")
    return _split_target(rows, "math_ref128", output_dir)


def prepare_mbpp_target(output_dir):
    """D* / held-out for the MBPP+ target from MBPP *train* minus every MBPP+ task id.

    MBPP+ is built from the sanitized MBPP set, which spans the original
    train/test/validation splits, so 108 of the 374 train rows are benchmark
    tasks and must be excluded (as Next does).
    """
    pin = MBPP_TRAIN_PIN
    print(f"Preparing mbpp target from {pin['repo']}:{pin['config']}@{pin['revision'][:8]} ({pin['split']}) ...")
    blocker = PromptDecontaminator(_reference_prompts(output_dir))
    blocked_ids = set()
    for row in _read_jsonl(_bench_path(output_dir, "mbpp_plus")):
        number = _mbpp_task_number((row.get("metadata") or {}).get("task_id"))
        if number is not None:
            blocked_ids.add(number)
    ds = load_dataset(pin["repo"], pin["config"], split=pin["split"], revision=pin["revision"])
    examples = shuffled_examples(list(ds), seed=DOLCI_SAMPLE_SEED)
    rows, id_blocked, prompt_blocked = [], 0, 0
    for ex in examples:
        # MBPP ships CRLF line endings; normalise so the fenced code is clean.
        text = (ex.get("text") or "").replace("\r\n", "\n").strip()
        code = (ex.get("code") or "").replace("\r\n", "\n").strip()
        tests = [t.replace("\r\n", "\n") for t in (ex.get("test_list") or [])]
        if not text or not code or not tests:
            continue
        if _mbpp_task_number(ex.get("task_id")) in blocked_ids:
            id_blocked += 1
            continue
        prompt = render_mbpp_prompt(text, tests)
        if blocker.match(prompt) is not None or blocker.match(text) is not None:
            prompt_blocked += 1
            continue
        rows.append({
            "dataset": "mbpp",
            "id": f"mbpp::{ex.get('task_id')}",
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": f"```python\n{code}\n```"},
            ],
            "metadata": {
                "task_id": str(ex.get("task_id")),
                "test_list": tests,
                "test_setup_code": (ex.get("test_setup_code") or "").replace("\r\n", "\n"),
                "source_repo": pin["repo"],
                "source_revision": pin["revision"],
                "source_split": pin["split"],
            },
        })
    print(f"mbpp: dropped {id_blocked} MBPP+ task ids and {prompt_blocked} benchmark-overlapping prompts; {len(rows)} rows remain")
    return _split_target(rows, "mbpp", output_dir)


def audit_dolci_leakage(output_dir):
    """Re-check every pool/target row against the benchmark (and target) prompts."""
    bench = PromptDecontaminator(_reference_prompts(output_dir))
    full = PromptDecontaminator(_reference_prompts(output_dir, targets=DOLCI_TARGETS))
    problems = 0
    for target in DOLCI_AUDIT_TARGETS:
        for split in ("validation", "test"):
            path = os.path.join(output_dir, "eval", target, f"{target}_{split}_data.jsonl")
            if not os.path.isfile(path):
                continue
            rows = _read_jsonl(path)
            hits = sum(bench.blocked_messages(r["messages"]) for r in rows)
            problems += hits
            print(f"  {target}/{split}: {len(rows)} rows, {hits} benchmark matches")
    for pool in DOLCI_POOL_DOMAINS:
        path = os.path.join(output_dir, "train", pool, f"{pool}_data.jsonl")
        if not os.path.isfile(path):
            continue
        rows = _read_jsonl(path)
        hits = sum(full.blocked_messages(r["messages"]) for r in rows)
        problems += hits
        print(f"  {pool}: {len(rows)} rows, {hits} benchmark/target matches")
    print("Leakage audit:", "CLEAN" if problems == 0 else f"{problems} MATCHES")
    return problems == 0


def main():
    parser = argparse.ArgumentParser(
        description="Prepare datasets for training and evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Datasets:

  Evaluation Datasets (validation/lr/test splits):
    tydiqa         - TyDiQA: Typologically Diverse QA (multilingual)
    samsum         - SamSUM: Dialogue summarization (includes train split)
    nq_open_eval   - NQ-open: eval splits (val/lr/test from HF nq_open validation)
    squad_eval     - SQuAD: closed-book eval splits (NO context; answer.text list)
    triviaqa_eval  - TriviaQA: closed-book eval splits (val/lr/test from rc.nocontext validation)
    truthfulqa     - TruthfulQA: adversarial truthfulness QA (50 val / 100 lr / 667 test)

  Downstream Benchmarks (eval/<bench>/<bench>_bench_data.jsonl; scored by SFT/eval/eval.sh):
    ifeval    - IFEval (541 prompts; official verifier vendored)
    ifbench   - IFBench (300 prompts; needs an allenai/IFBench checkout at eval time)
    math500   - MATH-500 (500 problems; math-verify or boxed-answer scoring)
    mbpp_plus - MBPP+ via the evalplus package (378 tasks)
    gsm8k     - GSM8K test (1319 problems; math-verify scoring)

  Dolci capability setting (Qwen3 + Dolci-Instruct pools; see SFT/README.md):
    Order matters: benchmarks -> math_ref mbpp -> dolci_pools (pools are decontaminated against all of them).
    math_ref    - eval/math_ref/ MATH-train reference solutions (D* 64 / held-out 500); not a training target, but
                  kept: the pools were decontaminated against it
    math_ref128 - eval/math_ref128/: math_ref re-split 128 / 436 (D* = math_ref D* + first 64 held-out rows); the base of
                  the curated MATH target math_ref128_gen32b (SFT/data/build_curated_math_target.sh)
    mbpp        - eval/mbpp/ from MBPP train minus MBPP+ task ids (D* 128 / held-out ~137)
    dolci_pools - train/dolci_{instruction,reasoning,mixed}/<name>_data.jsonl (32,000 rows each)
    tulu3_pool  - train/tulu3_general/tulu3_general_data.jsonl (32,000-row Tulu 3 SFT mixture sample; needs the
                  benchmarks + precise_if/math_ref/mbpp/math_persona targets on disk)
                  + eval/precise_if/ (Dolci Precise-IF rows; D* 128 / held-out 472, excluded from the pools)
    precise_if  - only the precise_if target (same rows as dolci_pools writes)
    dolci_audit - re-check pools/targets for exact or near-duplicate (8-gram) benchmark prompts

  Training Pools:
    nq_open         - NQ-open: train pool (~88K Q->A pairs)
    triviaqa_train  - TriviaQA: train pool (~138K Q->A pairs from rc.nocontext)
    squad           - SQuAD: with-context reading-comprehension (~88K)
    tulu3           - Tulu-3 SFT mixture (~939K)
    alpaca          - Stanford Alpaca (~52K)
    dolly, flan_v2, cot, oasst1 - LESS-mix components (~1.96M combined)
"""
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        required=True,
        metavar='DATASET',
        choices=['tydiqa', 'samsum', 'nq_open_eval', 'squad_eval', 'triviaqa_eval', 'truthfulqa',
                 'nq_open', 'triviaqa_train', 'squad', 'tulu3', 'alpaca',
                 'dolly', 'flan_v2', 'cot', 'oasst1',
                 'ifeval', 'ifbench', 'math500', 'gsm8k', 'mbpp_plus',
                 'dolci_pools', 'precise_if', 'math_ref', 'math_ref128', 'mbpp', 'dolci_audit', 'tulu3_pool'],
        help="Datasets to prepare (see list below)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="SFT/data",
        help="Output directory for prepared data (default: SFT/data)"
    )

    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="Worker processes for the Dolci pool filters (default: 16)"
    )

    args = parser.parse_args()
    datasets_to_prepare = args.datasets

    print(f"Preparing datasets: {datasets_to_prepare}")
    print(f"Output directory: {args.output_dir}")
    print()

    results = {}

    # Evaluation datasets (validation / lr / test splits)
    if 'tydiqa' in datasets_to_prepare:
        val_file, test_file = prepare_tydiqa(args.output_dir)
        results['tydiqa_validation'] = val_file
        results['tydiqa_test'] = test_file

    if 'samsum' in datasets_to_prepare:
        results['samsum'] = prepare_samsum(args.output_dir)

    if 'nq_open_eval' in datasets_to_prepare:
        val_file, lr_file, test_file = prepare_nq_open_eval(args.output_dir)
        results['nq_open_validation'] = val_file
        results['nq_open_lr'] = lr_file
        results['nq_open_test'] = test_file

    if 'squad_eval' in datasets_to_prepare:
        val_file, lr_file, test_file = prepare_squad_eval(args.output_dir)
        results['squad_validation'] = val_file
        results['squad_lr'] = lr_file
        results['squad_test'] = test_file

    if 'triviaqa_eval' in datasets_to_prepare:
        val_file, lr_file, test_file = prepare_triviaqa_eval(args.output_dir)
        results['triviaqa_validation'] = val_file
        results['triviaqa_lr'] = lr_file
        results['triviaqa_test'] = test_file

    if 'truthfulqa' in datasets_to_prepare:
        val_file, lr_file, test_file = prepare_truthfulqa(args.output_dir)
        results['truthfulqa_validation'] = val_file
        results['truthfulqa_lr'] = lr_file
        results['truthfulqa_test'] = test_file

    # Downstream benchmarks (generation + official verifier; see SFT/eval/tasks)
    if 'ifeval' in datasets_to_prepare:
        results['ifeval_bench'] = prepare_if_benchmark(args.output_dir, 'ifeval')

    if 'ifbench' in datasets_to_prepare:
        results['ifbench_bench'] = prepare_if_benchmark(args.output_dir, 'ifbench')

    if 'math500' in datasets_to_prepare:
        results['math500_bench'] = prepare_math500_bench(args.output_dir)

    if 'gsm8k' in datasets_to_prepare:
        results['gsm8k_bench'] = prepare_gsm8k_bench(args.output_dir)

    if 'mbpp_plus' in datasets_to_prepare:
        results['mbpp_plus_bench'] = prepare_mbpp_plus_bench(args.output_dir)

    # Dolci capability setting
    if 'dolci_pools' in datasets_to_prepare:
        results.update(prepare_dolci_pools(args.output_dir, num_proc=args.num_proc))
        results['precise_if'] = os.path.join(args.output_dir, "eval", "precise_if")
    elif 'precise_if' in datasets_to_prepare:
        results['precise_if'] = prepare_precise_if_target(args.output_dir, num_proc=args.num_proc)[0]

    if 'math_ref' in datasets_to_prepare:
        results['math_ref'] = prepare_math_target(args.output_dir)

    if 'math_ref128' in datasets_to_prepare:
        results['math_ref128'] = prepare_math_ref128_target(args.output_dir)

    if 'mbpp' in datasets_to_prepare:
        results['mbpp'] = prepare_mbpp_target(args.output_dir)

    if 'tulu3_pool' in datasets_to_prepare:
        results[TULU3_POOL] = prepare_tulu3_pool(args.output_dir, num_proc=args.num_proc)

    if 'dolci_audit' in datasets_to_prepare:
        results['dolci_audit'] = "clean" if audit_dolci_leakage(args.output_dir) else None

    # Training pools
    if 'nq_open' in datasets_to_prepare:
        results['nq_open_train'] = prepare_nq_open_train(args.output_dir)

    if 'triviaqa_train' in datasets_to_prepare:
        results['triviaqa_train'] = prepare_triviaqa_train(args.output_dir)

    if 'squad' in datasets_to_prepare:
        results['squad'] = prepare_squad(args.output_dir)

    if 'tulu3' in datasets_to_prepare:
        results['tulu3'] = prepare_tulu3(args.output_dir)

    if 'alpaca' in datasets_to_prepare:
        results['alpaca'] = prepare_alpaca(args.output_dir)

    # LESS-mix components (used as combined train pool for less_tydiqa)
    if 'dolly' in datasets_to_prepare:
        results['dolly'] = prepare_dolly(args.output_dir)

    if 'flan_v2' in datasets_to_prepare:
        results['flan_v2'] = prepare_flan_v2(args.output_dir)

    if 'cot' in datasets_to_prepare:
        results['cot'] = prepare_cot(args.output_dir)

    if 'oasst1' in datasets_to_prepare:
        results['oasst1'] = prepare_oasst1(args.output_dir)

    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("="*50)
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"{status} {name}: {path if path else 'Failed or requires manual setup'}")


if __name__ == "__main__":
    main()
