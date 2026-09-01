# Method Configs

Each YAML file defines a training method as `{CurationMethod}-{FinetuningMethod}`.

## Fields

| Field | Values | Description |
|---|---|---|
| `method` | Standard, LayerWiseSubset, GlobalSubset | Data curation method |
| `finetuning` | Full, LoRA, MeSO, MeSO-LoRA | Training approach |
| `lora_r`, `lora_alpha`, `lora_dropout` | int, int, float | LoRA hyperparameters |

## Dolci capability settings

`dolci_inst_if`, `dolci_reason_math`, `dolci_reason_code`, `dolci_mixed_if`,
`dolci_mixed_math` train Qwen3-1.7B-Base on a Dolci pool toward a target whose
benchmark is scored post hoc (see `SFT/README.md`). Their `defaults.yaml` add these keys:

| Key | Values | Description |
|---|---|---|
| `gradient_checkpointing` | true, false | Non-reentrant activation checkpointing (required for 4096-token sequences on a 48 GB GPU) |
| `val_seq_length_multiplier` | float | D* rejection threshold as a multiple of the average train length; `0` disables |
| `eval_split` | validation, lr, test | Split used for the `eval_loss` curve (default `test`; CLI `--eval_split` overrides) |

## Gradient Compression

Both `score_grad_compression` and `opt_grad_compression` use the same two-stage pipeline:

```yaml
score_grad_compression:   # compresses gradients for influence score computation
  sparsifier: normal-64*64
  projector: none

opt_grad_compression:     # compresses gradients for MeSO optimizer updates
  sparsifier: normal-512*512
  projector: none
```

**Stage 1 — Sparsifier** (factorized random projection):
Reduces each layer's gradient from full dimension to a low-rank sketch.
Format: `METHOD-DIM*DIM` or `none`.

**Stage 2 — Projector** (non-factorized final projection):
Further compresses the sparsified intermediate representation.
Format: `METHOD-DIM` or `none`.

### Named Compression Schemes

| Name | Sparsifier | Projector | Description |
|---|---|---|---|
| **LoGra** | `normal-D*D` | `none` | Gaussian random projection only (default for MeSO) |
| **GraSS** | `random_mask-D*D` | `sjlt-K` | Sparse mask + sparse JL transform |

Examples:
- LoGra with 512×512: `sparsifier: normal-512*512`, `projector: none`
- GraSS with 1024×1024 + 262144: `sparsifier: random_mask-1024*1024`, `projector: sjlt-262144`

### Design Rules

- **score_grad_compression**: Used for influence score computation in LayerWiseSubset/GlobalSubset curation.
  Set `sparsifier: none` for exact scoring (higher accuracy, more memory).
- **opt_grad_compression**: Used by MeSO optimizer for memory-efficient updates.
  When both sections use the same sparsifier value, compressor objects are shared (zero overhead).
- **MeSO + curation**: If `opt_grad_compression` is set and `score_grad_compression` is not,
  scoring uses full (uncompressed) gradients. To share MeSO compression for scoring,
  set `score_grad_compression.sparsifier` to the same value as `opt_grad_compression.sparsifier`.
- **Identity fallback**: If the compression dimension exceeds the layer's actual feature dimension,
  the compressor automatically falls back to identity (no-op).
