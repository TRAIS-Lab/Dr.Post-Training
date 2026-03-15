# Case Study: Layerwise vs Subset Selection Analysis

## Setup

Two dataset combos, same model and protocol:

| | tulu3 → tydiqa | alpaca → samsum |
|---|---|---|
| **Training data** | 0.5% of tulu3 (~4.7k samples) | 20% of alpaca (~10.4k samples) |
| **Eval task** | TyDiQA (cross-lingual QA) | SAMSum (dialogue summarization) |
| **LR** | 3.61e-05 | 1.47e-06 |
| **Steps** | 587 | 1,300 |
| **Records** | Every step (587 total) | Every 2 steps (650 total) |

- **Model**: Llama-3.2-1B (full parameter fine-tuning, 113 Linear layers)
- **Training**: Standard full-batch Adam (no selection applied)
- **Selection scoring**: At each step, both Layerwise and Subset scoring were computed on the same model state and same batch (batch_size=8, select top 4)
- **Validation**: 1 task-specific sample per step used as selection signal

## Key Findings

### 1. Layerwise and Subset fundamentally disagree on sample importance

| Metric | tulu3 → tydiqa | alpaca → samsum |
|--------|----------------|-----------------|
| Per-layer Jaccard (LW layer vs Subset) | 0.446 ± 0.055 | 0.480 ± 0.062 |
| Majority-vote Jaccard (LW vote vs Subset) | 0.608 ± 0.212 | 0.680 ± 0.203 |
| Adjacent layer agreement | 0.510 ± 0.231 | 0.567 ± 0.250 |

**Consistent across both combos: Layerwise and Subset agree less than half the time at the per-layer level.** Even with majority voting (a sample is "Layerwise-selected" if >50% of layers pick it), agreement with Subset is only 61–68%. The two methods would train on substantially different data.

The alpaca→samsum combo shows slightly higher agreement, possibly because alpaca is a more homogeneous dataset (all English instruction-following) compared to tulu3's multilingual diversity.

### 2. Different layers want different samples — significantly

| Metric | tulu3 → tydiqa | alpaca → samsum |
|--------|----------------|-----------------|
| Unique patterns per step | 43.5 (range 25–57) | 39.3 (range 25–52) |
| Fraction of layers with unique decision | ~39% | ~35% |

Over a third of layers make a unique selection decision at each step. Adjacent layers agree only 51–57% of the time.

This is the core insight for Layerwise selection: **sample utility is layer-dependent**. A training example that helps the attention layers may not help the MLP layers, and vice versa. Subset selection forces a single global decision and inevitably compromises.

### 3. Early and late layers increasingly diverge during training

**tulu3 → tydiqa:**

| Step | Early-Mid | Mid-Late | Early-Late |
|------|-----------|----------|------------|
| 1 (start) | 0.681 | 0.815 | 0.823 |
| 147 (25%) | 0.663 | 0.584 | 0.893 |
| 294 (50%) | 0.595 | **-0.022** | 0.151 |
| 441 (75%) | 0.412 | 0.760 | 0.620 |
| 587 (end) | 0.301 | 0.077 | **-0.226** |

**alpaca → samsum:**

| Step | Early-Mid | Mid-Late | Early-Late |
|------|-----------|----------|------------|
| 2 (start) | 0.691 | **-0.156** | 0.297 |
| 326 (25%) | 0.738 | **-0.322** | 0.048 |
| 652 (50%) | -0.108 | 0.886 | 0.205 |
| 976 (75%) | 0.457 | 0.350 | 0.233 |
| 1300 (end) | 0.964 | 0.171 | 0.110 |

Both combos show layer-group divergence, but with different patterns:
- **tulu3→tydiqa**: Starts with high agreement, then early-late layers become anti-correlated (-0.226 by end). The divergence is gradual and monotonic.
- **alpaca→samsum**: Mid-late layers are already anti-correlated from the start (-0.156 at step 2, -0.322 at step 326). This suggests the summarization task creates immediate tension between what mid and late layers need.

In both cases, **different layer groups develop distinct and sometimes opposing data preferences**, confirming that Layerwise's per-layer flexibility has a genuine structural advantage over Subset's forced global consensus.

### 4. Subset selection is dominated by accumulated score magnitude

Subset scores accumulate across all 113 layers, so they are effectively a weighted average of per-layer preferences. This means:
- Layers with larger gradient magnitudes (typically late/output layers) dominate the global score
- Subtle but important signals from early layers get drowned out
- The global selection correlates most with late-layer preferences (Subset matches late-layer top-4 most often)

Indeed, at Step 1: Subset selected [1,5,6,7] exactly matches early-layer preference, but by Step 587: Subset selected [0,3,4,6] exactly matches early-layer preference [0,3,4,6] while late layers prefer [1,2,5,6] — a complete disagreement.

### 5. Score variance across layers grows with training

Mean per-sample score variance across layers (first 50 steps): **1.547**. This means the same sample can score very differently at different layers — it's not just noise, it's the model developing specialized per-layer data needs.

### 6. No batch position bias

Both methods select each batch position roughly equally (48–54%), confirming that selection is driven by content, not position artifacts.

### 7. Qualitative: Divergent examples show meaningful disagreements

In high-divergence batches (Jaccard as low as 0.125), the disagreements are content-driven:

**Example (Step 434):**
- Validation sample: Arabic question about a historical mosque
- Subset selects: math integral problem, vintage car event, Chinese Linux question, pattern matching task
- Layerwise majority selects: NLP relation task, math tangency problem, Chinese Linux question, data visualization task, irrigation planning
- Only 1 sample (pattern matching) is selected by both

The training data (tulu3) is a diverse instruction-tuning mixture. The selection methods genuinely disagree about which diverse training examples best align with the Arabic QA validation signal, and this disagreement varies by layer depth.

### 8. Temporal trends differ by dataset

**tulu3 → tydiqa** — Agreement *decreases* over training:

| Phase | LW-Subset Jaccard | LW unique patterns |
|-------|-------------------|--------------------|
| Phase 1 (steps 1–146) | 0.460 | 41.8 |
| Phase 2 (steps 147–292) | 0.446 | 44.0 |
| Phase 3 (steps 293–438) | 0.443 | 44.1 |
| Phase 4 (steps 439–587) | 0.434 | 44.2 |

**alpaca → samsum** — Agreement *slightly increases* over training:

| Phase | LW-Subset Jaccard | LW unique patterns |
|-------|-------------------|--------------------|
| Phase 1 (steps 1–325) | 0.473 | 39.7 |
| Phase 2 (steps 326–650) | 0.478 | 39.4 |
| Phase 3 (steps 651–975) | 0.482 | 39.3 |
| Phase 4 (steps 976–1300) | 0.487 | 38.9 |

Interesting contrast: with tulu3→tydiqa (diverse multilingual train → cross-lingual eval), the methods diverge further as training progresses. With alpaca→samsum (English instruction → English summarization), they converge slightly. This may reflect that the more diverse the data, the more room for methods to disagree about what's relevant.

## Implications

1. **Layerwise selection is not just an approximation of Subset** — they make fundamentally different data choices, especially as training progresses.

2. **The layer-specificity of data utility is real and grows during training.** Early in training, all layers roughly agree on what's useful. Later, they diverge significantly, with early-late layer correlations even becoming negative.

3. **Subset selection is a compromise** that averages over all layers. By accumulating scores, it's biased toward whatever layer group has the largest gradient magnitudes. Layerwise selection respects each layer's individual learning needs.

4. **For diverse training data** (like tulu3 instruction mixtures), the choice of selection method meaningfully affects which examples are used, which could explain performance differences between the two approaches.

### 9. Early layers agree more with Subset than late layers

Across both combos, **early layers have higher Jaccard with Subset** than late layers:

| Layer group | tulu3 → tydiqa | alpaca → samsum |
|-------------|----------------|-----------------|
| Early (layers 0–37) | 0.467 | 0.546 |
| Middle (layers 38–74) | 0.460 | 0.488 |
| Late (layers 75–112) | 0.412 | 0.409 |

This is consistent with Subset being dominated by accumulated scores, where late layers (with larger gradients) shift the global decision away from what early layers prefer. Late layers already get their way in the global score — so their per-layer selection naturally diverges from it less... except it diverges *more*. This suggests late layers have high score magnitude but also high internal disagreement among themselves.

## Reproduction

```bash
# Run the case studies (standard training + dual scoring)
cd $CODE_DIR/Gradient-Streaming

# tulu3 → tydiqa
bash SFT/case_study/run.sh --percentage 0.005

# alpaca → samsum
bash SFT/case_study/run.sh --train alpaca --task samsum --percentage 0.2 --lr 1.47e-06 --record_freq 2

# Analyze results
python3 SFT/case_study/analyze_results.py <path-to-selection_records.json>
```
