"""
Reward model utilities for RLHF training.

This module provides functions to load and use reward models for PPO training.
The primary reward model for toxicity experiments is the hate speech classifier.
"""

import logging
from typing import Optional, List, Tuple, Any
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    pipeline,
)

logger = logging.getLogger(__name__)


class _CrossVocabBackbone(nn.Module):
    """
    Backbone module for CrossVocabRewardWrapper that handles vocab translation.

    This is a separate class to avoid circular references when PyTorch
    traverses the module tree during .to() calls.
    """

    def __init__(self, reward_model, reward_tokenizer, policy_tokenizer, score, config):
        super().__init__()
        # Store the actual reward model as a submodule
        self.reward_model = reward_model
        # Store tokenizers as non-module attributes
        self._reward_tokenizer = reward_tokenizer
        self._policy_tokenizer = policy_tokenizer
        # Store reference to score layer (not as submodule - parent owns it)
        self._score = score
        self._config = config

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        Decode policy tokens, re-tokenize for reward model, compute reward.
        Returns a fake output with 'last_hidden_state' that TRL expects.
        """
        device = input_ids.device

        # Decode policy tokens to text
        texts = self._policy_tokenizer.batch_decode(
            input_ids, skip_special_tokens=True
        )

        # Re-tokenize for reward model
        reward_inputs = self._reward_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        # Get reward model outputs
        with torch.no_grad():
            outputs = self.reward_model(**reward_inputs)

        # Use raw logits like the reference implementation
        # Model labels: class 0 = "nothate", class 1 = "hate"
        # Higher logit for "nothate" (class 0) = higher reward = less toxic
        logits = outputs.logits.float()
        rewards = logits[:, 0]  # Use nothate logit as reward

        # Create fake hidden state that encodes our rewards
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_size = self._config.hidden_size

        fake_hidden = torch.zeros(
            batch_size, seq_len, hidden_size,
            device=device, dtype=self.reward_model.dtype
        )

        # Encode rewards so score layer produces correct values
        score_weight = self._score.weight.data  # [1, hidden_size]
        non_zero_idx = torch.abs(score_weight[0]).argmax().item()
        scale = score_weight[0, non_zero_idx].item()

        if abs(scale) > 1e-8:
            fake_hidden[:, -1, non_zero_idx] = rewards / scale
        else:
            scale = score_weight[0].mean().item()
            if abs(scale) > 1e-8:
                fake_hidden[:, -1, :] = rewards.unsqueeze(-1) / scale

        # Return object with hidden_states that TRL expects
        # TRL calls output.hidden_states[-1] to get the last layer hidden states
        class FakeOutput:
            def __init__(self, hidden_states):
                # hidden_states should be a tuple/list for TRL to index with [-1]
                self.hidden_states = hidden_states

        return FakeOutput(hidden_states=(fake_hidden,))


class CrossVocabRewardWrapper(nn.Module):
    """
    Wrapper for reward models that handles vocabulary mismatch between policy and reward models.

    TRL's experimental PPO passes policy model token IDs to reward models.
    This wrapper decodes those tokens to text and re-tokenizes for the reward model.

    Args:
        reward_model: The actual reward model (e.g., RoBERTa toxicity classifier)
        reward_tokenizer: Tokenizer for the reward model
        policy_tokenizer: Tokenizer for the policy model
        device: Device to run on
    """

    def __init__(
        self,
        reward_model: nn.Module,
        reward_tokenizer,
        policy_tokenizer,
        device: str = "cuda",
    ):
        super().__init__()

        # TRL expects these attributes
        self.config = reward_model.config
        # Use a fake base_model_prefix that TRL will use to get the backbone
        self.base_model_prefix = "reward_backbone"

        # Create score layer (TRL expects this)
        hidden_size = reward_model.config.hidden_size
        self.score = nn.Linear(hidden_size, 1, bias=False)

        # Initialize score weights from the classifier if available
        if hasattr(reward_model, 'classifier') and hasattr(reward_model.classifier, 'out_proj'):
            out_proj = reward_model.classifier.out_proj
            with torch.no_grad():
                if out_proj.out_features >= 2:
                    # Use class 0 (nothate) weights as reward
                    # Higher logit for nothate = higher reward = less toxic
                    self.score.weight.data = out_proj.weight[0:1, :]

        self.score = self.score.to(device)

        # Create the backbone module that TRL will call
        # Note: reward_model is passed to backbone, not stored here (avoids duplication)
        self.reward_backbone = _CrossVocabBackbone(
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            policy_tokenizer=policy_tokenizer,
            score=self.score,
            config=self.config,
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward through the backbone."""
        return self.reward_backbone(input_ids, attention_mask=attention_mask, **kwargs)


class RewardModelWrapper(nn.Module):
    """
    Wrapper for reward models that adds a .score() method for TRL 0.26.1 compatibility.

    TRL's experimental PPO expects models to have a .score() method that:
    - Takes hidden states of shape (batch, seq_len, hidden_size)
    - Returns logits of shape (batch, seq_len, 1)

    This wrapper provides that interface for models like RoBERTa that use
    a different classifier interface.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

        # Copy important attributes
        self.config = model.config
        self.base_model_prefix = getattr(model, 'base_model_prefix', 'model')

        # Create a proper score layer that preserves sequence dimension
        # TRL expects score(hidden_states) -> (batch, seq_len, 1)
        hidden_size = model.config.hidden_size

        # Initialize score from the classifier's output projection if available
        if hasattr(model, 'classifier') and hasattr(model.classifier, 'out_proj'):
            # For RoBERTa, the out_proj maps hidden_size -> num_labels
            # We want to map to a single value for reward
            out_proj = model.classifier.out_proj
            # Create score that maps to single value
            self.score = nn.Linear(hidden_size, 1, bias=False)
            with torch.no_grad():
                # Use class 0 (nothate) weights as reward
                # Higher logit for nothate = higher reward = less toxic
                if out_proj.out_features >= 2:
                    self.score.weight.data = out_proj.weight[0:1, :]
        elif hasattr(model, 'score'):
            self.score = model.score
        else:
            # Create a simple linear score head
            self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped model
        if name in ['model', 'config', 'base_model_prefix', 'score']:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def load_toxicity_model(
    model_id: str = "facebook/roberta-hate-speech-dynabench-r4-target",
    device: str = "cuda",
    use_fp16: bool = False,
    wrap_for_trl: bool = True,
) -> Tuple[Any, Any]:
    """
    Load the toxicity classification model (RoBERTa hate speech classifier).

    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        use_fp16: Whether to use FP16 for memory efficiency
        wrap_for_trl: Whether to wrap model for TRL 0.26.1 compatibility

    Returns:
        Tuple of (tokenizer, model)
    """
    logger.info(f"Loading toxicity model: {model_id}")

    tokenizer = RobertaTokenizer.from_pretrained(model_id)

    dtype = torch.float16 if use_fp16 else torch.float32
    model = RobertaForSequenceClassification.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)

    model.eval()

    # Wrap for TRL 0.26.1 compatibility (adds .score() method)
    if wrap_for_trl:
        model = RewardModelWrapper(model)
        logger.info("Wrapped model for TRL compatibility (added .score() method)")

    logger.info(f"Loaded toxicity model on {device}")

    return tokenizer, model


def load_reward_pipeline(
    model_name: str,
    device: str = "cuda",
    use_bfloat16: bool = True,
) -> Tuple[Any, Any]:
    """
    Load reward model as a HuggingFace pipeline.

    This is a more general function that can load any sequence classification
    model as a reward model.

    Args:
        model_name: HuggingFace model name
        device: Device string or index
        use_bfloat16: Use bfloat16 precision

    Returns:
        Tuple of (tokenizer, pipeline)
    """
    logger.info(f"Loading reward pipeline: {model_name}")

    # Special handling for hate speech model
    if "hate" in model_name:
        return load_toxicity_model(model_name, device)

    # General pipeline loading
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if use_bfloat16 else torch.float32

    reward_pipe = pipeline(
        "sentiment-analysis",
        model=model_name,
        device_map={"": device} if isinstance(device, str) else device,
        model_kwargs={"torch_dtype": dtype},
        tokenizer=tokenizer,
        return_token_type_ids=False,
    )

    # Ensure pad token is set in pipeline
    reward_pipe.tokenizer.pad_token = reward_pipe.tokenizer.eos_token
    reward_pipe.tokenizer.pad_token_id = tokenizer.eos_token_id
    reward_pipe.model.config.pad_token_id = tokenizer.eos_token_id

    return tokenizer, reward_pipe


def compute_toxicity_scores(
    texts: List[str],
    tokenizer: Any,
    model: Any,
    batch_size: int = 32,
    max_length: int = 512,
) -> List[float]:
    """
    Compute toxicity scores for a list of texts.

    Returns negative toxicity scores so that lower toxicity = higher reward.

    Args:
        texts: List of text strings
        tokenizer: Toxicity model tokenizer
        model: Toxicity model
        batch_size: Batch size for inference
        max_length: Maximum sequence length

    Returns:
        List of reward scores (negative toxicity)
    """
    all_scores = []
    device = next(model.parameters()).device

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)

            # Use raw logits like the reference implementation
            # Model labels: class 0 = "nothate", class 1 = "hate"
            # Higher logit for "nothate" (class 0) = higher reward = less toxic
            logits = outputs.logits.float()
            rewards = logits[:, 0].cpu().tolist()
            all_scores.extend(rewards)

    return all_scores


def compute_reward_scores(
    texts: List[str],
    reward_tokenizer: Any,
    reward_model: Any,
    model_type: str = "toxicity",
    **kwargs,
) -> List[float]:
    """
    Compute reward scores for a list of texts.

    This is the main entry point for computing rewards during PPO training.

    Args:
        texts: List of text strings (query + response concatenated)
        reward_tokenizer: Tokenizer for reward model
        reward_model: Reward model or pipeline
        model_type: Type of reward model ('toxicity', 'sentiment', 'general')
        **kwargs: Additional arguments passed to specific reward function

    Returns:
        List of reward scores
    """
    if model_type == "toxicity":
        return compute_toxicity_scores(
            texts, reward_tokenizer, reward_model, **kwargs
        )
    elif model_type == "sentiment":
        # For sentiment pipeline (like IMDB)
        outputs = reward_model(texts, **kwargs)
        # Convert to scores (positive sentiment = positive reward)
        scores = []
        for out in outputs:
            if out["label"] == "POSITIVE":
                scores.append(out["score"])
            else:
                scores.append(-out["score"])
        return scores
    else:
        # Generic pipeline
        outputs = reward_model(texts, **kwargs)
        return [out["score"] for out in outputs]


def format_for_reward_model(
    queries: List[str],
    responses: List[str],
    format_type: str = "concat",
) -> List[str]:
    """
    Format query-response pairs for reward model input.

    Args:
        queries: List of query strings
        responses: List of response strings
        format_type: How to format ('concat', 'qa', 'chat')

    Returns:
        List of formatted strings
    """
    if format_type == "concat":
        # Simple concatenation
        return [q + r for q, r in zip(queries, responses)]
    elif format_type == "qa":
        # Question-Answer format
        return [f"Question: {q}\n\nAnswer: {r}" for q, r in zip(queries, responses)]
    elif format_type == "chat":
        # Chat format
        return [f"Human: {q}\n\nAssistant: {r}" for q, r in zip(queries, responses)]
    else:
        return [q + r for q, r in zip(queries, responses)]
