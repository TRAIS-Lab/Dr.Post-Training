#!/usr/bin/env python3
"""
Script to download and prepare datasets for training and evaluation.
Supports: TyDiQA, SamSUM, TriviaQA, NQ-open (eval) + NQ-open, TriviaQA, SQuAD, Alpaca, Dolly, FLAN-v2, CoT, OASST1, Tulu3 (train).
"""

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
    Splits the validation set into val/lr/test for n_val/LR-sweep/final-eval.
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
    print(f"  LR sweep:   {lr_file} ({len(lr_records)})")
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
    print(f"  LR sweep:   {lr_file} ({len(lr_records)})")
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
    Prepare TyDiQA validation, LR sweep, and test data in unified JSONL format.

    Uses the HuggingFace validation split and divides into validation/lr/test.
    Preserves the few-shot prompts file (tydiqa_fewshot.json) for downstream evaluation.

    Output files:
    - tydiqa_validation_data.jsonl: For data selection
    - tydiqa_lr_data.jsonl: For LR sweep evaluation
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

    # Split: first 100 for validation, next 100 for LR sweep, rest for test
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

    # Write LR sweep data
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
    print(f"  LR sweep: {lr_file} ({len(lr_data)} examples)")
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
    """Prepare Chain-of-Thought dataset."""
    print("Preparing CoT training data...")
    dataset = load_dataset("kaist-ai/CoT-Collection", split="train")

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
    - samsum_lr_data.jsonl: For LR sweep evaluation
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
        print(f"  LR sweep: {lr_file}")
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

    # Split test into LR sweep (first 100) and final test (rest)
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

    # LR sweep data
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
    print(f"  LR sweep: {lr_file} ({len(lr_examples)} examples)")
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
        choices=['tydiqa', 'samsum', 'nq_open_eval', 'squad_eval',
                 'nq_open', 'triviaqa_train', 'squad', 'tulu3', 'alpaca',
                 'dolly', 'flan_v2', 'cot', 'oasst1'],
        help="Datasets to prepare (see list below)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="SFT/data",
        help="Output directory for prepared data (default: SFT/data)"
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
