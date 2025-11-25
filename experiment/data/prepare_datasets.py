#!/usr/bin/env python3
"""
Script to download and prepare datasets for training and evaluation.
Supports: MMLU, BBH, TyDiQA, MATH500, GSM8K, SAMSum (eval) + Alpaca, Dolly, FLAN-v2, CoT, OASST1, Vicuna, WizardLM, OpenHermes, Tulu3 (train)
"""

import json
import os
import argparse
import shutil
import urllib.request
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


# BBH CoT prompts URL from the original BIG-Bench Hard repository
BBH_COT_PROMPTS_BASE_URL = "https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/"


def _generate_bbh_fewshot_from_cot_prompts(cot_prompts_dir):
    """
    Generate BBH few-shot JSON from existing cot-prompts directory.

    Args:
        cot_prompts_dir: Directory containing .txt files with CoT prompts

    Returns:
        Dictionary mapping task names to CoT prompt strings
    """
    fewshot_data = {}
    for filename in os.listdir(cot_prompts_dir):
        if filename.endswith('.txt'):
            task_name = filename.replace('.txt', '')
            filepath = os.path.join(cot_prompts_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                # Remove the canary string if present
                lines = content.split('\n')
                if lines and 'canary' in lines[0].lower():
                    content = '\n'.join(lines[2:]).strip()
                fewshot_data[task_name] = content
    return fewshot_data


def ensure_dir(path):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def prepare_gsm8k_train(output_dir):
    """Prepare GSM8K training data."""
    print("Preparing GSM8K training data...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")

    output_file = os.path.join(output_dir, "train", "gsm8k", "gsm8k_train_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            question = example['question']
            answer = example['answer']

            data = {
                "dataset": "gsm8k",
                "id": f"gsm8k_train_{idx}",
                "messages": [
                    {"role": "user", "content": f"Solve the following math problem step by step:\n{question}"},
                    {"role": "assistant", "content": answer}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"GSM8K training data saved to {output_file}")
    return output_file


def prepare_gsm8k_test(output_dir):
    """
    Prepare GSM8K validation and test data in unified JSONL format.

    Splits the test set into validation (first 100) and test (remaining).
    Uses unified format with 'messages' field.
    """
    print("Preparing GSM8K validation and test data...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    output_dir_gsm8k = os.path.join(output_dir, "eval", "gsm8k")
    ensure_dir(output_dir_gsm8k)

    # Split: first 100 for validation, rest for test
    val_size = 100
    all_examples = list(dataset)
    val_examples = all_examples[:val_size]
    test_examples = all_examples[val_size:]

    # Validation data
    val_file = os.path.join(output_dir_gsm8k, "gsm8k_validation_data.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(val_examples):
            data = {
                "dataset": "gsm8k",
                "id": f"gsm8k_val_{idx}",
                "messages": [
                    {"role": "user", "content": f"Solve the following math problem step by step:\n{example['question']}"},
                    {"role": "assistant", "content": example['answer']}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Test data
    test_file = os.path.join(output_dir_gsm8k, "gsm8k_test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_examples):
            data = {
                "dataset": "gsm8k",
                "id": f"gsm8k_test_{idx}",
                "messages": [
                    {"role": "user", "content": f"Solve the following math problem step by step:\n{example['question']}"},
                    {"role": "assistant", "content": example['answer']}
                ]
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"GSM8K data saved:")
    print(f"  Validation: {val_file} ({len(val_examples)} examples)")
    print(f"  Test: {test_file} ({len(test_examples)} examples)")
    return val_file, test_file


def prepare_mmlu(output_dir):
    """
    Prepare MMLU (Massive Multitask Language Understanding) dataset.

    Downloads from cais/mmlu on HuggingFace and saves in unified JSONL format:
    - mmlu_validation_data.jsonl: For data selection
    - mmlu_test_data.jsonl: For evaluation

    Uses unified format with 'messages' field, same as other datasets.
    """
    print("Preparing MMLU dataset...")

    output_dir_mmlu = os.path.join(output_dir, "eval", "mmlu")
    ensure_dir(output_dir_mmlu)

    # MMLU subjects
    subjects = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_european_history", "high_school_geography",
        "high_school_government_and_politics", "high_school_macroeconomics",
        "high_school_mathematics", "high_school_microeconomics",
        "high_school_physics", "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history", "human_aging",
        "human_sexuality", "international_law", "jurisprudence",
        "logical_fallacies", "machine_learning", "management", "marketing",
        "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory", "professional_accounting",
        "professional_law", "professional_medicine", "professional_psychology",
        "public_relations", "security_studies", "sociology", "us_foreign_policy",
        "virology", "world_religions"
    ]

    choices_letters = ["A", "B", "C", "D"]
    val_examples = []
    test_examples = []

    def format_question(question, choices):
        """Format MMLU question with choices."""
        prompt = f"The following is a multiple choice question. Answer with A, B, C, or D.\n\n{question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{choices_letters[i]}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt

    def format_answer(answer_idx):
        """Convert answer index to letter."""
        if isinstance(answer_idx, int):
            return choices_letters[answer_idx]
        return answer_idx

    for subject in tqdm(subjects, desc="Downloading MMLU subjects"):
        try:
            # Load subject data from HuggingFace
            dataset = load_dataset("cais/mmlu", subject)

            # Process validation split
            if "validation" in dataset:
                for idx, example in enumerate(dataset["validation"]):
                    val_examples.append({
                        "dataset": "mmlu",
                        "id": f"mmlu_val_{subject}_{idx}",
                        "subject": subject,
                        "messages": [
                            {"role": "user", "content": format_question(example["question"], example["choices"])},
                            {"role": "assistant", "content": format_answer(example["answer"])}
                        ]
                    })

            # Process test split
            if "test" in dataset:
                for idx, example in enumerate(dataset["test"]):
                    test_examples.append({
                        "dataset": "mmlu",
                        "id": f"mmlu_test_{subject}_{idx}",
                        "subject": subject,
                        "messages": [
                            {"role": "user", "content": format_question(example["question"], example["choices"])},
                            {"role": "assistant", "content": format_answer(example["answer"])}
                        ]
                    })

        except Exception as e:
            print(f"  Warning: Failed to load subject '{subject}': {e}")
            continue

    # Write validation data
    val_file = os.path.join(output_dir_mmlu, "mmlu_validation_data.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Write test data
    test_file = os.path.join(output_dir_mmlu, "mmlu_test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    print(f"MMLU data saved:")
    print(f"  Validation: {val_file} ({len(val_examples)} examples)")
    print(f"  Test: {test_file} ({len(test_examples)} examples)")
    print(f"  Subjects: {len(subjects)}")
    return val_file, test_file


def prepare_math500(output_dir):
    """
    Prepare MATH500 validation and test data in unified JSONL format.

    Splits the test set into validation (first 50) and test (remaining).
    Uses unified format with 'messages' field.
    """
    print("Preparing MATH500 validation and test data...")
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    output_dir_math = os.path.join(output_dir, "eval", "math500")
    ensure_dir(output_dir_math)

    # Split: first 50 for validation, rest for test
    val_size = 50
    all_examples = list(dataset)
    val_examples = all_examples[:val_size]
    test_examples = all_examples[val_size:]

    def format_problem(example):
        level = example.get('level', 'unknown')
        prob_type = example.get('type', 'unknown')
        return f"Solve the following Level {level} {prob_type} problem:\n{example['problem']}"

    # Validation data
    val_file = os.path.join(output_dir_math, "math500_validation_data.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(val_examples):
            data = {
                "dataset": "math500",
                "id": f"math500_val_{idx}",
                "messages": [
                    {"role": "user", "content": format_problem(example)},
                    {"role": "assistant", "content": example['solution']}
                ],
                "metadata": {
                    "level": example.get('level', 'unknown'),
                    "type": example.get('type', 'unknown')
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # Test data
    test_file = os.path.join(output_dir_math, "math500_test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(test_examples):
            data = {
                "dataset": "math500",
                "id": f"math500_test_{idx}",
                "messages": [
                    {"role": "user", "content": format_problem(example)},
                    {"role": "assistant", "content": example['solution']}
                ],
                "metadata": {
                    "level": example.get('level', 'unknown'),
                    "type": example.get('type', 'unknown')
                }
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"MATH500 data saved:")
    print(f"  Validation: {val_file} ({len(val_examples)} examples)")
    print(f"  Test: {test_file} ({len(test_examples)} examples)")
    return val_file, test_file


def prepare_bbh(output_dir):
    """
    Prepare BIG-Bench Hard (BBH) validation and test data in unified JSONL format.

    Reads from existing test/*.json files and splits into validation/test.
    Preserves the few-shot prompts file (bbh_fewshot.json) for downstream evaluation.

    Output files:
    - bbh_validation_data.jsonl: For data selection (n_val examples per task)
    - bbh_test_data.jsonl: For evaluation
    - bbh_fewshot.json: Few-shot prompts for each task (preserved/renamed)
    """
    print("Preparing BBH validation and test data...")

    output_dir_bbh = os.path.join(output_dir, "eval", "bbh")
    ensure_dir(output_dir_bbh)

    # Check if we have existing test data in test/*.json format
    test_dir = os.path.join(output_dir_bbh, "test")
    existing_three_shot = os.path.join(output_dir_bbh, "bbh-three-shot.json")

    # Try to load from existing test/*.json files first
    if os.path.exists(test_dir) and any(f.endswith('.json') for f in os.listdir(test_dir)):
        print("Found existing BBH test data, loading from test/*.json files...")
        all_tasks_data = {}
        for filename in os.listdir(test_dir):
            if filename.endswith('.json') and not filename.startswith('.'):
                task_name = filename.replace('.json', '')
                filepath = os.path.join(test_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    task_data = json.load(f)
                    # Handle the canary format
                    if 'examples' in task_data:
                        all_tasks_data[task_name] = task_data['examples']
                    else:
                        all_tasks_data[task_name] = task_data
    else:
        # Download from HuggingFace - need to load each task config separately
        print("Downloading BBH from HuggingFace...")
        bbh_tasks = [
            'boolean_expressions', 'causal_judgement', 'date_understanding',
            'disambiguation_qa', 'dyck_languages', 'formal_fallacies',
            'geometric_shapes', 'hyperbaton', 'logical_deduction_five_objects',
            'logical_deduction_seven_objects', 'logical_deduction_three_objects',
            'movie_recommendation', 'multistep_arithmetic_two', 'navigate',
            'object_counting', 'penguins_in_a_table', 'reasoning_about_colored_objects',
            'ruin_names', 'salient_translation_error_detection', 'snarks',
            'sports_understanding', 'temporal_sequences',
            'tracking_shuffled_objects_five_objects', 'tracking_shuffled_objects_seven_objects',
            'tracking_shuffled_objects_three_objects', 'web_of_lies', 'word_sorting'
        ]
        all_tasks_data = {}
        for task_name in tqdm(bbh_tasks, desc="Downloading BBH tasks"):
            try:
                dataset = load_dataset("lukaemon/bbh", task_name)
                all_tasks_data[task_name] = [
                    {"input": ex.get('input', ''), "target": ex.get('target', '')}
                    for ex in dataset['test']
                ]
            except Exception as e:
                print(f"  Warning: Failed to load BBH task '{task_name}': {e}")

    # Split each task: first 3 for validation, rest for test
    # (BBH tasks typically have ~250 examples each)
    val_per_task = 3
    val_examples = []
    test_examples = []

    for task_name, examples in all_tasks_data.items():
        task_val = examples[:val_per_task]
        task_test = examples[val_per_task:]

        for idx, ex in enumerate(task_val):
            val_examples.append({
                "dataset": "bbh",
                "id": f"bbh_val_{task_name}_{idx}",
                "task": task_name,
                "messages": [
                    {"role": "user", "content": ex.get('input', '')},
                    {"role": "assistant", "content": ex.get('target', '')}
                ]
            })

        for idx, ex in enumerate(task_test):
            test_examples.append({
                "dataset": "bbh",
                "id": f"bbh_test_{task_name}_{idx}",
                "task": task_name,
                "messages": [
                    {"role": "user", "content": ex.get('input', '')},
                    {"role": "assistant", "content": ex.get('target', '')}
                ]
            })

    # Write validation data
    val_file = os.path.join(output_dir_bbh, "bbh_validation_data.jsonl")
    with open(val_file, 'w', encoding='utf-8') as f:
        for example in val_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Write test data
    test_file = os.path.join(output_dir_bbh, "bbh_test_data.jsonl")
    with open(test_file, 'w', encoding='utf-8') as f:
        for example in test_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')

    # Handle few-shot prompts file (Chain-of-Thought prompts for downstream evaluation)
    fewshot_file = os.path.join(output_dir_bbh, "bbh_fewshot.json")
    cot_prompts_dir = os.path.join(output_dir_bbh, "cot-prompts")

    if os.path.exists(existing_three_shot) and not os.path.exists(fewshot_file):
        # Copy/rename the existing few-shot file
        shutil.copy(existing_three_shot, fewshot_file)
        print(f"  Few-shot prompts copied to: {fewshot_file}")
    elif os.path.exists(fewshot_file):
        print(f"  Few-shot prompts already exist: {fewshot_file}")
    else:
        # Try to generate from existing cot-prompts directory
        if os.path.exists(cot_prompts_dir) and any(f.endswith('.txt') for f in os.listdir(cot_prompts_dir)):
            print("  Generating few-shot prompts from existing cot-prompts...")
            fewshot_data = _generate_bbh_fewshot_from_cot_prompts(cot_prompts_dir)
        else:
            # Download CoT prompts from GitHub
            print("  Downloading Chain-of-Thought prompts from BIG-Bench-Hard repository...")
            ensure_dir(cot_prompts_dir)
            fewshot_data = {}

            for task_name in all_tasks_data.keys():
                url = f"{BBH_COT_PROMPTS_BASE_URL}{task_name}.txt"
                cot_file = os.path.join(cot_prompts_dir, f"{task_name}.txt")

                try:
                    urllib.request.urlretrieve(url, cot_file)
                    # Parse the CoT prompt file
                    with open(cot_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Remove the canary string (first two lines)
                        lines = content.split('\n')
                        if lines and 'canary' in lines[0].lower():
                            # Skip canary and separator line
                            content = '\n'.join(lines[2:]).strip()
                        fewshot_data[task_name] = content
                except Exception as e:
                    print(f"    Warning: Failed to download CoT prompt for '{task_name}': {e}")
                    # Fallback: use raw input/target (less useful but better than nothing)
                    examples = all_tasks_data[task_name][:3]
                    fallback_prompt = f"Answer the following questions.\n\n"
                    for ex in examples:
                        fallback_prompt += f"Q: {ex.get('input', '')}\nA: {ex.get('target', '')}\n\n"
                    fewshot_data[task_name] = fallback_prompt.strip()

        with open(fewshot_file, 'w', encoding='utf-8') as f:
            json.dump(fewshot_data, f, ensure_ascii=False, indent=4)
        print(f"  Few-shot prompts saved to: {fewshot_file}")

    print(f"BBH data saved:")
    print(f"  Validation: {val_file} ({len(val_examples)} examples)")
    print(f"  Test: {test_file} ({len(test_examples)} examples)")
    print(f"  Tasks: {len(all_tasks_data)}")
    return val_file, test_file


def prepare_tydiqa(output_dir):
    """
    Prepare TyDiQA validation and test data in unified JSONL format.

    Uses the HuggingFace validation split and divides into validation/test.
    Preserves the few-shot prompts file (tydiqa_fewshot.json) for downstream evaluation.

    Output files:
    - tydiqa_validation_data.jsonl: For data selection
    - tydiqa_test_data.jsonl: For evaluation
    - tydiqa_fewshot.json: One-shot prompts (preserved/renamed)
    """
    print("Preparing TyDiQA validation and test data...")

    output_dir_tydiqa = os.path.join(output_dir, "eval", "tydiqa")
    ensure_dir(output_dir_tydiqa)

    existing_one_shot = os.path.join(output_dir_tydiqa, "tydiqa-one-shot.json")

    # Load TyDiQA goldp (gold passage) task
    print("Loading TyDiQA from HuggingFace...")
    dataset = load_dataset("tydiqa", "secondary_task")

    # Use validation split (TyDiQA doesn't have a test split in secondary_task)
    all_data = list(dataset["validation"])

    # Split: first 100 for validation, rest for test
    val_size = 100
    val_data = all_data[:val_size]
    test_data = all_data[val_size:]

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
    Prepare SAMSum dialogue summarization dataset.

    Checks if data already exists in the expected format before downloading.
    """
    print("Preparing SAMSum data...")

    train_file = os.path.join(output_dir, "train", "samsum", "samsum_train_data.jsonl")
    val_file = os.path.join(output_dir, "eval", "samsum", "samsum_validation_data.jsonl")
    test_file = os.path.join(output_dir, "eval", "samsum", "samsum_test_data.jsonl")

    # Check if data already exists
    if os.path.exists(val_file) and os.path.exists(test_file):
        print(f"SAMSum evaluation data already exists:")
        print(f"  Validation: {val_file}")
        print(f"  Test: {test_file}")

        # Check if train data exists
        if os.path.exists(train_file):
            print(f"  Train: {train_file}")
        else:
            print(f"  Train: Not found (run with --datasets samsum to download)")

        return train_file if os.path.exists(train_file) else val_file

    # Try to download from HuggingFace
    print("Downloading SAMSum from HuggingFace...")
    ensure_dir(os.path.dirname(train_file))
    ensure_dir(os.path.dirname(val_file))

    # Load from knkarthick/samsum
    train_dataset = load_dataset("knkarthick/samsum", split="train")
    val_dataset = load_dataset("knkarthick/samsum", split="validation")
    test_dataset = load_dataset("knkarthick/samsum", split="test")

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

    # Test data
    with open(test_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(test_dataset, desc="Test")):
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

    print(f"SAMSum data saved:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  Test: {test_file}")
    return train_file


def prepare_vicuna(output_dir):
    """
    Prepare Vicuna (ShareGPT) training data.
    Using Anon8231489123/ShareGPT_Vicuna_unfiltered dataset.
    """
    print("Preparing Vicuna training data...")
    print("Note: This dataset is large and may take some time to download.")

    # Using the filtered ShareGPT dataset
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

    output_file = os.path.join(output_dir, "train", "vicuna", "vicuna_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            # ShareGPT format has 'conversations' field
            conversations = example.get('conversations', [])

            messages = []
            for conv in conversations:
                role = conv.get('from', '')
                content = conv.get('value', '')

                if role == 'human':
                    role = 'user'
                elif role == 'gpt':
                    role = 'assistant'
                else:
                    continue

                messages.append({"role": role, "content": content})

            if len(messages) >= 2:
                data = {
                    "dataset": "vicuna",
                    "id": f"vicuna_{idx}",
                    "messages": messages
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"Vicuna training data saved to {output_file}")
    return output_file


def prepare_wizardlm(output_dir):
    """Prepare WizardLM training data."""
    print("Preparing WizardLM training data...")
    dataset = load_dataset("WizardLMTeam/WizardLM_evol_instruct_V2_196k", split="train")

    output_file = os.path.join(output_dir, "train", "wizardlm", "wizardlm_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            conversations = example.get('conversations', [])

            messages = []
            for conv in conversations:
                role = conv.get('from', conv.get('role', ''))
                content = conv.get('value', conv.get('content', ''))

                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant', 'system']:
                    role = 'assistant'
                else:
                    continue

                messages.append({"role": role, "content": content})

            if len(messages) >= 2:
                data = {
                    "dataset": "wizardlm",
                    "id": f"wizardlm_{idx}",
                    "messages": messages
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"WizardLM training data saved to {output_file}")
    return output_file


def prepare_openhermes(output_dir):
    """Prepare OpenHermes2.5 training data."""
    print("Preparing OpenHermes2.5 training data...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")

    output_file = os.path.join(output_dir, "train", "openhermes", "openhermes_data.jsonl")
    ensure_dir(os.path.dirname(output_file))

    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, example in enumerate(tqdm(dataset)):
            conversations = example.get('conversations', [])

            messages = []
            for conv in conversations:
                role = conv.get('from', conv.get('role', ''))
                content = conv.get('value', conv.get('content', ''))

                if role in ['human', 'user']:
                    role = 'user'
                elif role in ['gpt', 'assistant']:
                    role = 'assistant'
                else:
                    continue

                messages.append({"role": role, "content": content})

            if len(messages) >= 2:
                data = {
                    "dataset": "openhermes",
                    "id": f"openhermes_{idx}",
                    "messages": messages
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    print(f"OpenHermes training data saved to {output_file}")
    return output_file


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

  Evaluation Datasets (with validation/test splits):
    mmlu      - MMLU: Massive Multitask Language Understanding (57 subjects)
    bbh       - BBH: BIG-Bench Hard (23 reasoning tasks with CoT prompts)
    tydiqa    - TyDiQA: Typologically Diverse QA (9 languages)
    gsm8k     - GSM8K: Grade School Math (includes train split)
    math500   - MATH500: Competition math problems
    samsum    - SAMSum: Dialogue summarization (includes train split)

  Training Datasets:
    alpaca    - Stanford Alpaca (52K instruction-following examples)
    dolly     - Databricks Dolly 2.0 (15K examples)
    flan_v2   - FLAN v2 instruction mixture (100K examples)
    cot       - Chain-of-Thought reasoning (100K examples)
    oasst1    - OpenAssistant conversations (88K examples)
    vicuna    - ShareGPT-based conversations (125K examples)
    wizardlm  - WizardLM evolved instructions (196K examples)
    openhermes - OpenHermes 2.5 (1M examples)
    tulu3     - Tulu-3 SFT mixture (939K examples)
"""
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        required=True,
        metavar='DATASET',
        choices=['mmlu', 'gsm8k', 'math500', 'bbh', 'tydiqa', 'vicuna', 'wizardlm', 'openhermes', 'tulu3',
                 'alpaca', 'dolly', 'flan_v2', 'cot', 'oasst1', 'samsum'],
        help="Datasets to prepare (see list below)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiment/data",
        help="Output directory for prepared data (default: experiment/data)"
    )

    args = parser.parse_args()
    datasets_to_prepare = args.datasets

    print(f"Preparing datasets: {datasets_to_prepare}")
    print(f"Output directory: {args.output_dir}")
    print()

    results = {}

    # Evaluation datasets (with validation/test split)
    if 'mmlu' in datasets_to_prepare:
        val_file, test_file = prepare_mmlu(args.output_dir)
        results['mmlu_validation'] = val_file
        results['mmlu_test'] = test_file

    if 'gsm8k' in datasets_to_prepare:
        results['gsm8k_train'] = prepare_gsm8k_train(args.output_dir)
        val_file, test_file = prepare_gsm8k_test(args.output_dir)
        results['gsm8k_validation'] = val_file
        results['gsm8k_test'] = test_file

    if 'math500' in datasets_to_prepare:
        val_file, test_file = prepare_math500(args.output_dir)
        results['math500_validation'] = val_file
        results['math500_test'] = test_file

    if 'bbh' in datasets_to_prepare:
        val_file, test_file = prepare_bbh(args.output_dir)
        results['bbh_validation'] = val_file
        results['bbh_test'] = test_file

    if 'tydiqa' in datasets_to_prepare:
        val_file, test_file = prepare_tydiqa(args.output_dir)
        results['tydiqa_validation'] = val_file
        results['tydiqa_test'] = test_file

    # New training datasets
    if 'vicuna' in datasets_to_prepare:
        results['vicuna'] = prepare_vicuna(args.output_dir)

    if 'wizardlm' in datasets_to_prepare:
        results['wizardlm'] = prepare_wizardlm(args.output_dir)

    if 'openhermes' in datasets_to_prepare:
        results['openhermes'] = prepare_openhermes(args.output_dir)

    if 'tulu3' in datasets_to_prepare:
        results['tulu3'] = prepare_tulu3(args.output_dir)

    # Existing training datasets (LESS + others)
    if 'alpaca' in datasets_to_prepare:
        results['alpaca'] = prepare_alpaca(args.output_dir)

    if 'dolly' in datasets_to_prepare:
        results['dolly'] = prepare_dolly(args.output_dir)

    if 'flan_v2' in datasets_to_prepare:
        results['flan_v2'] = prepare_flan_v2(args.output_dir)

    if 'cot' in datasets_to_prepare:
        results['cot'] = prepare_cot(args.output_dir)

    if 'oasst1' in datasets_to_prepare:
        results['oasst1'] = prepare_oasst1(args.output_dir)

    if 'samsum' in datasets_to_prepare:
        results['samsum'] = prepare_samsum(args.output_dir)

    print("\n" + "="*50)
    print("Dataset preparation complete!")
    print("="*50)
    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"{status} {name}: {path if path else 'Failed or requires manual setup'}")


if __name__ == "__main__":
    main()
