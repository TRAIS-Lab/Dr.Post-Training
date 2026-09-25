# Method Configs

Each YAML file defines a training method as `{CurationMethod}-{FinetuningMethod}`.

## Fields

| Field | Values | Description |
|---|---|---|
| `method` | Standard, LayerWiseSubset, GlobalSubset, GroupWiseSubset, BlockWiseSubset, SublayerWiseSubset | Data curation method (`BlockWiseSubset` / `SublayerWiseSubset` = `GroupWiseSubset` with `selection_granularity` block / sublayer) |
| `selection_granularity` | layer, sublayer, block, global, custom | GroupWiseSubset only: which hooked layers select jointly (default block) |
| `selection_groups` | `name=member,..;name=..` | GroupWiseSubset only: custom per-block groups (implies custom); no `:` or quotes in the value |
| `finetuning` | Full, LoRA, MeSO, MeSO-LoRA | Training approach |
| `lora_r`, `lora_alpha`, `lora_dropout` | int, int, float | LoRA hyperparameters |

## Question-answering matrix (Llama-3.2-1B)

`alpaca_samsum`, `less_tydiqa`, `triviaqa_nq`, `less_squad` hold one yaml per cell of the appendix matrix
{Full, LoRA, MeSO} × {FullTraining, GlobalSubset, BlockWiseSubset, SublayerWiseSubset, LayerWiseSubset} × {exact GIP,
compressed 64×64} plus the controls (`SFT/train/qa_matrix_task.sh`, `SFT/tables/qa_downstream.py`):

| Yaml | Cell |
|---|---|
| `FullTraining-<FT>` | no curation |
| `GlobalSubset-<FT>` | exact PIP (numerically identical to GIP) |
| `LayerWiseSubset-<FT>` (Full, LoRA), `{Block,Sublayer}WiseSubset-LoRA` | compressed 64×64 |
| `<Method>-<FT>-gip`, `<Method>-<FT>-cmp` | every other exact / compressed cell (MeSO's compressed Layer-Wise cell is `LayerWiseSubset-MeSO-cmp`; `LayerWiseSubset-MeSO` scores with the optimizer's 512×512 compressor and is not part of the matrix) |
| `GlobalSubset-<FT>-random`, `LayerWiseSubset-<FT>-random` | Random Subset controls (`selection_mode: random`, shared / per-layer subsets) |
| `GlobalSubset-<FT>-lnorm` | layer-normalized Global Subset (`score_normalization: layer_meanabs`, compressed) |

Learning rates: Full 1e-5, LoRA 1e-4, MeSO 5e-5 (`alpaca_samsum_meso_ablation/`: the Full-Training MeSO sweep
over the learning rate, the sketch size and the refresh that selected 5e-5). `alpaca_samsum_diag/` holds the Layer-Wise LoRA
runs with `record_selections: true` behind `SFT/benchmark/diag_compression_agreement.py` (exact vs 64/128/256/512 compressed
scores on identical batches, quoted in the appendix).

## Dolci capability settings

`dolci_inst_if`, `dolci_mixed_if`, `tulu3_if` (Precise IF target), `dolci_reason_mathpersona`,
`dolci_mixed_mathpersona`, `tulu3_mathpersona` (Dolci math target) and `dolci_reason_mathref128_gen32b`
(curated MATH-train target) train Qwen3-1.7B-Base on a 32K-row pool toward a target whose benchmark is
scored post hoc (see `SFT/README.md`). Each holds `defaults.yaml` plus the seven paper arms
`FullTraining-Full-eot`, `{Global,LayerWise}Subset-Full[-f25|-f75]-eot` (k = 4 / 2 / 6 of 8). Their
`defaults.yaml` add these keys:

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
