import torch
import tqdm
import json
import time
import asyncio
import os
from importlib import import_module
from transformers import StoppingCriteria


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


def dump_generations(args, task, rows, tokenizer=None):
    """Write <args.output_dir>/<task>_generations.jsonl (one row per test example: prediction, raw generation, references, score)
    next to <task>_results.json, so the metric can be decomposed post hoc (SFT/tables/qa_generation_stats.py). Adds n_tokens of the
    raw generation when a tokenizer is given (>= max_new_tokens - 1 means the generation never emitted a stop token). Never raises."""
    try:
        out_dir = getattr(args, "output_dir", None)
        if not out_dir or not rows:
            return
        path = os.path.join(out_dir, f"{task}_generations.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                r = dict(r)
                r.setdefault("idx", i)
                if tokenizer is not None and "raw" in r and "n_tokens" not in r:
                    r["n_tokens"] = len(tokenizer(r["raw"], add_special_tokens=False).input_ids)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Saved {len(rows)} generations to {path}")
    except Exception as e:   # diagnostics only; the evaluation result must not depend on it
        print(f"[warn] could not save generations for {task}: {e}")


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, add_special_tokens=False, disable_tqdm=False, **generation_kwargs):
    """Batched generation for chat-template-rendered prompts.

    ``add_special_tokens=False`` (default) matches the training encoding: every supervised sequence in this repo is the rendered
    chat template tokenized with ``add_special_tokens=False`` (SFT/data/chat_format.py, SFT/data/get_val_dataset.py), so the
    Llama tokenizer's ``<|begin_of_text|>`` is never part of a training sequence. In an earlier version the default was ``True``,
    which prepended that token to every evaluation prompt of the Llama-3.2-1B question-answering runs (a train/eval mismatch; the
    Qwen3 tokenizer adds no BOS, so the Qwen benchmarks were unaffected). Result files produced with the old encoding are kept as
    ``<task>_results.bos.json`` by SFT/eval/qa_reeval_task.sh.
    """
    generations = []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Generating Completions")

    # Set default max_new_tokens if not provided
    if "max_new_tokens" not in generation_kwargs:
        generation_kwargs["max_new_tokens"] = 20

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    # Remove pad_token_id from generation_kwargs if present, since we pass it explicitly
    generation_kwargs.pop("pad_token_id", None)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask
        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )

            # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
            # so some outputs still have the stop sequence, which we need to remove.
            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx+len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            # remove the prompt from the output
            # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
            # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
            # space is important for some tasks (e.g., code completion).
            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            # duplicate the prompts to match the number of return sequences
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            # Never substitute empty completions: a swallowed CUDA out-of-memory error silently zeroes the metric (observed
            # on samsum: a Rouge-L of 2.6 instead of 20.7). Fail loudly so the run is re-evaluated.
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            raise

        generations += batch_generations

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)

        if not disable_tqdm:
            progress.update(len(batch_prompts)//num_return_sequences)

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


@torch.no_grad()
def get_next_word_predictions(model, tokenizer, prompts,
                              candidate_token_ids=None,
                              batch_size=1,
                              return_token_predictions=False,
                              add_special_tokens=True,
                              disable_tqdm=True):

    predictions, probs = [], []
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(prompts), desc="Getting Predictions")

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i+batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=add_special_tokens)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        outputs = model(input_ids=batch_input_ids, attention_mask=attention_mask)
        batch_logits = outputs.logits[:, -1, :]
        batch_probs = torch.softmax(batch_logits, dim=-1)

        if candidate_token_ids is not None:
            batch_probs = batch_probs[:, candidate_token_ids]
        batch_prediction_indices = torch.argmax(batch_probs, dim=-1)

        if return_token_predictions:
            if candidate_token_ids is not None:
                candidate_tokens = tokenizer.convert_ids_to_tokens(candidate_token_ids)
                batch_predictions = [candidate_tokens[idx] for idx in batch_prediction_indices]
            else:
                batch_predictions = tokenizer.convert_ids_to_tokens(batch_prediction_indices)
            predictions += batch_predictions
        else:
            predictions += batch_prediction_indices.tolist()
        probs += batch_probs.tolist()

        if not disable_tqdm:
            progress.update(len(batch_prompts))

    assert len(predictions) == len(prompts), "number of predictions should be equal to number of prompts"
    return predictions, probs


def get_eos_token_ids(tokenizer):
    """Ids that should stop generation: the tokenizer eos plus the chat template's end-of-turn token.

    Qwen3-Base's eos is ``<|endoftext|>`` but an SFT'd model terminates assistant
    turns with ``<|im_end|>``; both are returned so ``model.generate`` stops on either.
    Returns an int when there is a single id, else a list (both accepted by generate).
    """
    from SFT.data.chat_format import get_eos_token_ids as _ids
    ids = _ids(tokenizer)
    if not ids:
        return tokenizer.eos_token_id
    return ids if len(ids) > 1 else ids[0]
