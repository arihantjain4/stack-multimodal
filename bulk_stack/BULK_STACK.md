# STACK on bulk RNA-seq: diagnosis, brainstorm, and adaptations

This folder asks whether STACK can take bulk RNA-seq as input, or reason natively in both single-cell and bulk space. It explains why the released model does badly on bulk, compares it across dataset types, and records every idea from a separate deep-thinking brainstorm. It then implements the most promising ideas as wrappers and adapters around the frozen checkpoint and benchmarks all of them. Nothing in `src/stack` was modified. Every new file lives in `bulk_stack/`.

The full brainstorm, written by the brainstorm agent before any adapter was trained, is kept verbatim in [`docs/brainstorm_agent.md`](docs/brainstorm_agent.md). This document summarizes it, says what happened to each idea, and reports the results.

## 1. Short answer

**Can STACK accept bulk RNA-seq today?** Mechanically, yes: a bulk count vector over the 15,012 model genes is a valid input. In practice it fails, and the failure is sharp. At native bulk depth (1–2 × 10⁷ reads), hidden-gene reconstruction correlation collapses to 0.29 on GENtoniK and 0.43 on MCF7/A549 bulk. Simply copying the population mean of the other samples gives 0.95 and 0.82. STACK does not reason natively in both modalities, and the cause is mostly one specific thing.

**Why it fails.** The input is `log1p(raw counts)` with no library-size normalization, and the first layer has a strong "total expression" direction. ‖W·1‖ is 15.5× what a random vector would give. Bulk depth adds about 9 nats to roughly 90% of genes. Along that direction the shift is 82× larger than the actual expression content, so every bulk sample becomes the same extreme "cell" before attention runs. Pairwise cosine between bulk first-layer pre-activations is 0.997, against 0.913 between real cells. Pseudobulks of 400 real cells fail the same way, but the same pseudobulks thinned back to single-cell depth work as well as single cells. So averaging many cells is harmless; the extreme count scale breaks the model. After depth is fixed, a second limit remains: the decoder was only ever trained on sparse single-cell targets, so it cannot reproduce fine quantitative bulk profiles.

**What fixes it.**
- *No training (depth harmonization):* rescaling bulk counts to single-cell depth before the log is a one-line change that recovers most of the loss. GENtoniK goes from 0.29 to 0.84 and the cell lines from 0.43 to 0.79, and single-cell scores are unchanged.
- *No training (context decoding):* decoding STACK's output relative to the sample's peers gives 0.955 on GENtoniK (population mean 0.954) and 0.872 on the cell lines (population mean 0.823). This is the first configuration that matches or beats the population mean.
- *Small trained adapters on real ARCHS4 bulk:* a 0.59M-parameter bulk output head, and a 1.0M-parameter full-depth side channel feeding that head. They give positive gains over the population mean on the cell lines (+0.044 and +0.070) and on organoid pb400 depth sweeps, without the hand-built context decoder.
- *Single-cell behaviour:* unchanged by the training-free variants and by the gated side channel's input path. The trained output head costs about 0.02–0.03 in single-cell reconstruction, discussed in Section 7.

**Caveat on downstream biology.** None of the STACK variants recovers GENtoniK drug-response fold changes as well as a plain 64-component linear (PCA-ridge) imputer trained on the same bulk pool. On imputed genes, the PCA imputer reaches r = 0.43 with the measured fold change, against 0.24–0.26 for the best STACK variants. A frozen STACK made bulk-safe is useful for embeddings and imputation, but it is not yet better than a simple bulk-native model at bulk-specific quantitative tasks. Getting there needs the architecture changes in Section 9.

## 2. What was compared

### Datasets (all real, fetched by rule from public sources)

| set family | source | what it is | profiles | median reads/profile |
|---|---|---|---|---|
| `organoid_10x/cells` | SCP1756 cortical organoids (23 days, 5 months), 10x | single cells, 24 (age, cell type) groups | 2,304 | 2,356 |
| `organoid_10x/pb{5,20,100,400}` | same cells | pseudobulk: sums of k real cells from one group | 576 / 576 / 576 / 504 | 1.4e4 → 1.2e6 |
| `organoid_10x/pb400_thinned` | same | pb400 binomially thinned to the single-cell median | 504 | 2,364 |
| `sciplex_nuclei/cells` | sci-Plex3 vehicle controls (A549, MCF7, K562), sci-RNA-seq nuclei | single nuclei | 288 | 854 |
| `sciplex_nuclei/pb*`, `pb400_thinned` | same | pseudobulk and thinned pseudobulk | 72 each | 5.0e3 → 3.7e5 |
| `gentonik_bulk/native` | ARCHS4, GEO GSE172544 (GENtoniK cocktail vs DMSO in 4 hPSC/iPSC-derived neuron lines) | real bulk | 29 | 2.0e7 |
| `celllines_bulk/native` | ARCHS4, MCF7 and A549 bulk from many GEO series (≤ 3 per series) | real bulk, same lines as sci-Plex | 96 | 1.3e7 |
| `*_bulk/thinned` | same bulk | thinned to the reference single-cell median | 29 / 96 | 2,360 / 837 |
| `*_bulk/depth_x{1,10,100,1000,10000}` | same bulk | thinned to multiples of the reference median (depth sweep) | 29 / 96 each | 2.4e3 → 2.0e7 |
| training pool | ARCHS4: 384 bulk samples, one per GEO series, series disjoint from every evaluation set and never mentioning MCF7/A549 | used only to train adapters | 384 | ≥ 5e6 |

Selection rules, file names, column names and regexes are all in [`config.yaml`](config.yaml). ARCHS4 columns are streamed remotely from `human_gene_v2.latest.h5`, and no sample list is hand-written. GENtoniK labels are parsed from the GEO `characteristics_ch1` field. The published GENtoniK DESeq2 tables are downloaded from the GEO supplement of GSE172544.

### Metrics (`evaluate.py`)

Profiles are run in windows of up to 64 profiles from one set, matching STACK's cross-cell attention. 20% of genes are hidden from every profile in the window, with 3 mask draws.

- `recon_r`: Pearson correlation, in log(1 + CPM), between the decoded and true values of the hidden genes. Only genes with window-mean CPM ≥ 1 in the truth are scored. **The truth is always the set's own counts**, so for `*/native` sets every variant is scored against native-depth truth, even when its input was transformed.
- `baseline_r`: the same score for the leave-one-out mean of the other profiles in the window (the "population mean" bar).
- `recon_gain` = recon_r − baseline_r, with a 1,000-draw bootstrap 95% CI over profiles. Positive means STACK adds information beyond the cohort mean.
- `resid_r`: correlation of (prediction − LOO mean) with (truth − LOO mean), which measures sample-specific signal.
- `visible_r`: decoded versus input with nothing hidden, i.e. how faithfully the decoder can reproduce what it was shown.
- `knn_stack` / `knn_pca`: 5-NN label accuracy on STACK embeddings versus on 50 PCs of log-CPM.
- `knn_series_*`: the same, with the GEO series as the label, a batch-confounding check on the cell lines.
- `emb_cosine`, `emb_pc1`: mean pairwise cosine of embeddings and the variance share of PC1, which together detect collapse.
- `preact_mean`, `preact_std`, `gelu_active`: statistics of the first-layer pre-activation (`gene_reduction[0]`), the direct readout of the depth shift.
- `cross_modal` (separate JSON): each bulk sample is assigned to the nearest single-cell group centroid (cosine on STACK embeddings) and, independently, by Spearman correlation of expression. Two references are used: single cells and depth-matched pseudobulks (`pb400_thinned`).

## 3. Why STACK does not work on bulk: diagnosis

These ten mechanisms come from measurements on the baseline and from a weight analysis of the checkpoint (`docs/brainstorm_agent.md` §1–2). They are ordered by measured impact.

1. **The input is not depth-invariant, and depth is encoded as a feature.** `features = log1p(counts)` with no CPM step. In single-cell data depth tracks cell size and quality, and the first Linear (15,012 → 1,600) learned a dedicated "sum over genes" direction: ‖W·1‖ / E‖W·r‖ = 15.5. At bulk depth, the uniform log-depth shift W·(log(lib/D)·1[x>0]) has 82× the norm of the depth-normalized content and a cosine of 0.96 with the full pre-activation. Measured effect: preact_std rises from 4.5 (cells) to 212.7 (GENtoniK), preact_mean falls to −53, GELU-active units fall from 51% to 38%, and visible_r falls from 0.65–0.80 (at 2e4–2e5 reads) to 0.29.
2. **No normalization before the first attention.** The layers are post-norm (`LN(x + attn(x))`), so the first cell-attention sees unnormalized tokens and its logits scale with the square of the input scale. The LayerNorm runs after the residual add, when the depth direction already dominates.
3. **Zero means three things.** Unobserved, masked and truly absent genes are all `0` in log space, and there is no mask token. With 91–95% zeros in single cells a masked gene looks like a dropout, which is harmless. With 10–20% zeros in bulk a masked gene looks like a confidently silent gene.
4. **Masks are shared across the window** in both pretraining and finetuning. Cross-cell attention therefore never learned to copy a gene's value from a peer that has it. Bulk imputation is exactly "this sample lacks a gene its peers have", which is why the LOO mean is so hard to beat and why peer-aware decoding (context decoder) helps.
5. **The decoder's resolution was learned on sparse targets.** An NB likelihood over 10³–10⁴ UMIs teaches "which 1–3k genes are non-zero, and roughly how many". Relative abundances finer than about 1/library were never supervised. Even with every gene visible and depth fixed, visible_r tops out at 0.85–0.89.
6. **The information is in the detection pattern, not in quantities.** A cell is about 1.4k detected genes, while a bulk sample has about 11k genes at ≥ 1 CPM. Harmonizing bulk to single-cell depth maps it back into the detection-pattern code but keeps only 3.3% (GENtoniK) or 0.8% (cell lines) of those genes at ≥ 1 expected count. A frozen STACK sees bulk through a keyhole.
7. **A mixture is not a cell.** Bulk is a library-weighted sum over many cell types. A depth-matched bulk is a draw from the *average* cell, which is not a real state for heterogeneous tissue, so the embedding lands between types. There is also a Jensen gap: log1p(Σ cells) ≠ mean(log1p(cell)).
8. **Cross-cell attention assumes the window is one population.** Training windows were 256–512 consecutive cells of one file. Bulk windows are small cohorts of different donors, series and treatments, and the embedding frame partly depends on window composition.
9. **Gene panel and protocol.** The 15,012 genes are single-cell HVGs, and full-length bulk protocols carry gene-length and capture biases. This is measurable but secondary: thinned bulk behaves like cells, and cell-line cross-modal accuracy is 0.96.
10. **No modality, depth or aggregation signal.** The model has no input that says "this is an aggregate of N cells at depth D". Finetuning showed that a 1,600-parameter additive embedding can switch STACK's role, but nothing like it exists for bulk.

## 4. Evidence across dataset types (unmodified STACK)

From `results/baseline_sets.csv`; figure `figures/dataset_types.png`.

| set | reads | recon_r | LOO mean | gain | visible_r | knn (STACK / PCA) | preact_std |
|---|---|---|---|---|---|---|---|
| organoid cells | 2.4e3 | 0.421 | 0.395 | **+0.026** | 0.452 | 0.80 / 0.80 | 4.5 |
| organoid pb5 | 1.4e4 | 0.608 | 0.606 | +0.003 | 0.650 | 0.91 / 0.95 | 13.8 |
| organoid pb20 | 5.7e4 | 0.720 | 0.727 | −0.007 | 0.766 | 0.99 / 1.00 | 29.5 |
| organoid pb100 | 2.9e5 | 0.753 | 0.809 | −0.057 | 0.804 | 1.00 / 1.00 | 57.5 |
| organoid pb400 | 1.2e6 | 0.689 | 0.859 | **−0.170** | 0.727 | 1.00 / 1.00 | 89.1 |
| organoid pb400 thinned | 2.4e3 | 0.408 | 0.417 | −0.009 | 0.440 | 0.94 / 0.93 | 3.9 |
| sci-Plex nuclei | 854 | 0.256 | 0.223 | **+0.032** | 0.274 | 0.99 / 0.99 | 2.4 |
| sci-Plex pb400 | 3.7e5 | 0.557 | 0.836 | **−0.279** | 0.590 | 1.00 / 1.00 | 75.8 |
| sci-Plex pb400 thinned | 833 | 0.234 | 0.203 | +0.031 | 0.250 | 0.97 / 1.00 | 1.8 |
| GENtoniK bulk native | 2.0e7 | 0.293 | 0.954 | **−0.660** | 0.287 | 0.83 / 0.97 | 212.7 |
| GENtoniK bulk thinned | 2.4e3 | 0.337 | 0.329 | +0.008 | 0.350 | 0.59 / 0.21 | 4.5 |
| cell-line bulk native | 1.3e7 | 0.427 | 0.823 | **−0.397** | 0.411 | 0.96 / 0.97 | 161.9 |
| cell-line bulk thinned | 837 | 0.248 | 0.216 | +0.033 | 0.259 | 0.96 / 0.94 | 1.6 |

How to read this:
- STACK only adds information beyond the cohort mean at single-cell depth. The gain is +0.026 to +0.035 on cells, thinned pseudobulk and thinned bulk.
- The gain falls monotonically with reads per profile, whether the depth comes from summing cells (pb5 → pb400) or from real bulk. The **depth sweep** (`figures/depth_sweep.png`) puts the same bulk samples at 1×, 10×, … 10,000× the single-cell median. As depth rises, GENtoniK visible_r peaks at about 10² × (0.66) and then collapses to 0.29, and pre-activation mean and std blow up in step.
- Thinned pseudobulk behaves like single cells, and so does thinned real bulk. The failure therefore comes from depth and count scale, not from aggregation, protocol or "bulk-ness".
- Embeddings degrade less than reconstruction: kNN on native bulk is still 0.83 / 0.96, because label identity survives in the residual directions. Cross-modal assignment does break. On the cell lines with the depth-matched pseudobulk reference, nearest-centroid accuracy is 0.68 for native bulk versus 0.96 for thinned bulk. GENtoniK native bulk scatters across 5–7 organoid groups, while Spearman correlation assigns all 29 samples to 5-month CPN.

## 5. The brainstorm

A separate deep-thinking agent was given the baseline results and the code, and asked to propose changes to architecture, input, output, training and evaluation. It ran its own small read-only experiments first (input-transform sweeps, residual-to-context decoding, mixed windows, weight analysis) to rule ideas in or out, then wrote 49 proposals in seven families plus a ranked shortlist (S1–S6). Its full text, including per-idea mechanism, expected metrics, cost, risks and test plan, is [`docs/brainstorm_agent.md`](docs/brainstorm_agent.md).

The experiments it ran before proposing anything changed the design in four ways:
- **T1:** any depth-harmonizing transform at 1–10× single-cell depth fixes the first-order problem. Above about 10×, everything degrades.
- **T2:** reconstruction is best at 1×, while label identity in the embedding is best at 10–100× (GENtoniK kNN 0.90–0.93).
- **T3:** the frozen decoder is the ceiling (visible_r ≤ 0.89), even though STACK's *relative* sample-versus-peers signal is real.
- **T4:** sparsity is not the problem. Dense non-integer rescaled inputs work as well as binomially thinned integer inputs.

### Status of every brainstormed idea

Legend: **Implemented** means it is a benchmarked variant or a component of one. **Measured** means it was tested in the brainstorm's own experiments (numbers in its §1) but not carried forward as a variant. **Not done** gives the reason. **Arch** means it needs re-pretraining the backbone, which is out of scope with a frozen checkpoint on CPU.

| id | idea | status | where / why |
|---|---|---|---|
| A1 | Deterministic depth rescaling, `log1p(x·min(1, D/lib))` | **Implemented** | `depth_scale_x1`, `_x10`, `_x100` (`adapters.depth_scale_transform`); also the input of every trained adapter |
| A2 | Binomial thinning, K-draw ensemble | **Implemented** | `thin_ensemble_x1`, `_x10` (K = 8) |
| A3 | Mean-field (Poisson-expected) thinning | Measured | ≈ rescale (0.840 vs 0.839 on GENtoniK); no reason to keep a second deterministic transform |
| A4 | Clip-shift (sparsity-imitating rescale) | Measured | worse than rescale at 1× (0.72); only useful as an embedding view at 100× |
| A5 | Multi-resolution fusion (decode at 1×, embed at 10–100×) | Not done | the trade-off is documented by the depth sweep; the fused embedding would not be comparable to single-cell embeddings |
| A6 | Context-mean imputation of masked inputs | Not done | negligible at 1× by construction (hidden genes are mostly zero anyway); D1 uses the same peer information at the output instead |
| A7 | Per-gene protocol offset | Partly implemented | as learned per-gene terms: input affine in `film_lora_adapter`, logit offset δ in `bulk_head`; no matched-pair estimate, which would leak evaluation series |
| A8 | Moment-matched tokens (training-free input LayerNorm) | Not done | predicted no-op after A1, and the shift is a direction, not a scale |
| A9 | Project out the depth direction | Measured | weight analysis; analytically equivalent to A1, so it serves as the explanation rather than a method |
| B1 | Depth FiLM on first-layer pre-activations | **Implemented** | `side_channel` (`film_scale`, `film_shift` × s) and `film_lora_adapter` (MLP of log10 library) |
| B2 | Modality / aggregation embedding | Not done | the learned depth gate of the side channel already acts as the modality switch |
| B3 | Bulk as ICL queries with a single-cell prompt | Measured (variant) | mixed windows of bulk plus reference cells changed kNN and assignment very little |
| B4 | Effective-heterogeneity conditioning | Not done | needs C5 first |
| C1 | Full-depth side channel into tokens, low rank, depth-gated | **Implemented** | `side_channel` (`adapters.SideChannel`) |
| C2 | LoRA on `gene_reduction[0]` | Not done | C1 chosen instead: it carries the 97% of information that harmonization throws away, whereas C2 only re-weights harmonized input |
| C3 | Window composition: (a) cohort, (b) mixed, (c) self-windows, (d) size | Partly implemented | (c) is `pseudocell_x10`; (b) measured; nearest-neighbour cohort windows used in adapter training (`adapter.pool.window_composition`) |
| C4 | Aggregate-to-aggregate cross-modal reference | **Implemented** | `evaluate.cross_modal`, reference `pb_thinned` (`benchmark.cross_modal_references`) |
| C5 | Deconvolution head over single-cell centroids | Not done | needs held-out real-cell mixtures; out of scope for this pass |
| C6 | Cross-sample attention LoRA with per-sample masking | Not done | the pool has one sample per series, so there are no batch-matched peers to learn copying from (needs E6) |
| C7 | Gene-identity tokenization with depth-invariant values | Arch | re-pretraining |
| C8 | Pre-norm layers with input LayerNorm | Arch | re-pretraining |
| C9 | Hierarchical cell → sample model | Arch | new model |
| D1 | Residual-to-context decoding, p ∝ C·(px / C̃)^β | **Implemented** | `context_x1` (window LOO prior), `context_x1_pool_prior` (training-pool prior); β fit on the pool |
| D2 | Copy path for visible genes | Not done | would make visible_r meaningless, and it does not change hidden genes |
| D3 | Per-gene bulk logit offset δ | **Implemented** | `BulkHead.delta` |
| D4 | Low-rank residual logits head | **Implemented** | `BulkHead.down/up` (rank 32) |
| D5 | Bulk likelihood (NB with per-gene bulk dispersion + correlation loss) | **Implemented** | `train_bulk.bulk_loss`: NB NLL normalized by the LOO-prior NLL + (1 − r) on hidden log-CPM; θ initialized by method of moments on the pool |
| D6 | Aggregation-consistency loss | Not done | needs the constituent cells' decodes; out of scope |
| D7 | Temperature on px | **Implemented** | `BulkHead.tau`; learned τ = −0.27, i.e. the head *softens* frozen logits |
| E1 | Masked-gene bulk reconstruction on the ARCHS4 pool | **Implemented** | `train_bulk.py`, `train_adapter.py`; random mask fraction in [0.1, 0.4] |
| E2 | Depth-consistency (multi-view) loss | Not done | kNN already depth-stable after A1 |
| E3 | Cross-modal anchoring | Not done | risk of leakage with the evaluation cells |
| E4 | Distill the thinning ensemble | Not done | the ensemble gain over A1 is small (+0.018 on GENtoniK) |
| E5 | Single-cell preservation | Partly implemented | as a learned depth gate g(s) on the side channel (closed for cells); the output head is not gated (see §7) |
| E6 | Complete held-out series for cohort training | Not done | download size; nearest-neighbour windows used as a substitute |
| E7 | Test-time self-supervised adaptation | Not done | transductive |
| F1 | Mask-ensemble imputation | Not done | |
| F2 | Depth-matched peers | **Implemented** (implicitly) | A1 puts every bulk sample at the same D |
| F3 | Hybrid decode (keep measured genes) | Not done | an API choice, not a model change |
| F4 | Uncertainty from thinning draws | Not done | |
| G1 | Truth = native when the input is transformed | **Implemented** | all `*/native` rows |
| G2 | resid_r | **Implemented** | `evaluate.run_set` |
| G3 | Deconvolution accuracy | Not done | needs C5 |
| G4 | DE recovery on GENtoniK versus published DESeq2 | **Implemented** | `treatment_effect.py` |
| G5 | Bulk ICL perturbation prediction | Not done | |
| G6 | Batch confounding (kNN by GEO series) | **Implemented** | `knn_series_stack` / `knn_series_pca` |
| G7 | Model-free controls trained on the same data | **Implemented** (i) | `pca_imputer` (64-component ridge); the LOO mean is the built-in second control |
| G8 | Single-cell regression check | **Implemented** | every variant reruns every cell and pseudobulk set |
| G9 | Bootstrap CIs | **Implemented** | `gain_ci_low/high` over profiles |

Shortlist mapping: S1 → `depth_scale_*` / `thin_ensemble_*`; S2 → `context_x1*`; S3 → `bulk_head`; S4 → `side_channel`; S5 → `pb_thinned` cross-modal reference. S6 (attention LoRA) was not attempted, for the reason given under C6.

## 6. What was implemented

Every variant is a *runner*: a function from a window of counts (and a hidden-gene mask) to embeddings and px. One benchmark (`evaluate.py`) scores all of them identically. All runners are built from `config.yaml` (`adapter.variants`) in `run_variants.make_runner`. The reference depth D is `auto`: the median library of all benchmark single cells (2,205.5 reads). Every target depth is a multiple of it.

### Training-free

1. **`depth_scale_x{1,10,100}`** (A1). The input becomes `log1p(x · min(1, D·m / lib))`. Profiles already at or below the target depth are untouched, so most single cells pass through. This is a one-line change to the input, implemented through the existing `run_window(..., transform=)` hook.
2. **`thin_ensemble_x{1,10}`** (A2). Draw K = 8 binomial thinnings to D·m, run each, and average px and embeddings.
3. **`pseudocell_x10`** (C3c). Place the K thinnings of every sample in *one* window, so cross-cell attention sees a population of "cells" drawn from each bulk sample, then average per sample.
4. **`context_x1`** (D1). On top of `depth_scale_x1`: `log p̂ = log(C + ε) + β·(log(px + ε) − log(C̃ + ε))`, renormalized. C is the leave-one-out mean of the peers' observed fractions, C̃ is the LOO mean of the peers' px, and ε is one read divided by the library. β is chosen by a grid search on the ARCHS4 training pool (`adapter.context.beta_grid`); it selected β = 1.0 (validation r 0.874 vs 0.806 at β = 0).
5. **`context_x1_pool_prior`**. The single-sample version of D1. C and C̃ come from the training-pool mean fraction and mean px (`results/pool_stats.npz`) instead of from window peers, so no evaluation peer is used.

### Trained adapters (frozen backbone)

All adapters are trained only on the 384-sample ARCHS4 pool. 10% is held out for validation, and every sample is from a different GEO series, so the split is by series. Every evaluation series and every sample mentioning MCF7/A549 is excluded. Training windows are an anchor sample plus its nearest pool neighbours (cohort-like). Each step hides a random 10–40% of genes and mixes native samples with copies thinned to random depths between 0.5·D and native. Every output-changing parameter starts at zero, so the untrained adapter reproduces the checkpoint exactly.

6. **`film_lora_adapter`** (first attempt, built before the brainstorm; `train_adapter.py`, 667k parameters). Depth-normalized input (D·10) with a per-gene affine (A7), a depth FiLM MLP on the first-layer pre-activation (B1), and LoRA rank 16 on the last `output_mlp` Linear (D4-like). It is trained with multinomial NLL + hidden-gene log-CPM MSE + a small visible-gene term, for 400 steps.
7. **`bulk_head`** (S3; `adapters.BulkHead`, 591,665 parameters, trained on cached embeddings in about 1 minute). The input is harmonized to D, and the backbone and `output_mlp` are frozen. The head is
   `logits' = (1+τ)·logits + δ + U·GELU(V·e) + α ⊙ log(C+ε) + γ ⊙ log(C̃+ε)`,
   with rank-32 U/V (D4), per-gene δ (D3), τ (D7), and per-gene α, γ (a learned D1), plus a per-gene bulk NB dispersion θ (D5).
   - Loss: NB NLL on hidden genes divided by the NLL of the LOO prediction (a skill score, so abundant genes don't dominate), plus 1 − Pearson r of hidden-gene log-CPM.
   - Training: 12 epochs; validation r 0.791 → 0.793, NB skill 0.83 → 0.78.
   - Learned values: τ = −0.27, mean α = 0.057, mean γ = 0.017.
8. **`side_channel`** (S4/C1; `adapters.SideChannel`, 1,016,802 parameters plus the head above).
   - Backbone input: the profile harmonized to D.
   - Side path: the *native* profile as per-gene standardized log-CPM (μ_g, σ_g from the pool), with hidden genes zeroed and a hidden-gene indicator concatenated, goes through a rank-32 GELU bottleneck into a token delta.
   - Gating: the delta is added after `gene_reduction` with gate `sigmoid(k·(s − s0))`, where `s = log(lib / D)`.
   - Depth FiLM: `pre ⊙ (1 + a·s) + b·s` on the first-layer pre-activation.
   - Training: 600 steps (99 s) jointly with a fresh copy of the bulk head; validation r 0.777 → 0.797.
   - Learned gate: slope 3.80, centre 2.15 (initial 4.0, ln 10). It is about 3 × 10⁻⁴ for single cells (s ≈ 0) and ≈ 1 for bulk (s ≈ 9), so the side path only touches deep profiles.
9. **`pca_imputer`** (G7, model-free control; `adapters.PCAImputer`). 64 principal axes of log-CPM fitted on the same pool. A ridge fit (λ = 1) of the visible genes onto those axes is used to reconstruct the hidden genes. If a trained STACK adapter cannot beat this, STACK itself is not contributing.

## 7. Results

Figures: `figures/variant_summary.png` (hidden-gene r per variant with the population-mean line), `figures/dataset_types.png` (all metrics, all set types, main variants), `figures/depth_sweep.png` (metrics vs reads per sample), `figures/embedding_map.png` (PCA of embeddings for cells, native bulk and thinned bulk; baseline vs `depth_scale_x1`).

### 7.1 Native-depth bulk: hidden-gene reconstruction

recon_r, with the gain over the leave-one-out population mean in parentheses. The 95% CIs on the gain are ±0.002–0.007 throughout (`results/*_sets.csv`).

| variant | trained params | GENtoniK native (LOO 0.954) | cell lines native (LOO 0.823) | organoid pb400 (LOO 0.859) | sci-Plex pb400 (LOO 0.836) |
|---|---|---|---|---|---|
| baseline (unmodified) | 0 | 0.293 (−0.660) | 0.427 (−0.397) | 0.689 (−0.170) | 0.557 (−0.279) |
| depth_scale_x1 | 0 | 0.840 (−0.114) | 0.787 (−0.036) | 0.858 (−0.002) | 0.663 (−0.173) |
| depth_scale_x10 | 0 | 0.828 (−0.126) | 0.763 (−0.060) | 0.853 (−0.006) | 0.659 (−0.177) |
| depth_scale_x100 | 0 | 0.662 (−0.291) | 0.707 (−0.116) | 0.797 (−0.062) | 0.617 (−0.219) |
| thin_ensemble_x1 | 0 | 0.858 (−0.095) | 0.778 (−0.045) | 0.857 (−0.002) | 0.754 (−0.082) |
| thin_ensemble_x10 | 0 | 0.840 (−0.113) | 0.768 (−0.055) | 0.857 (−0.002) | 0.696 (−0.140) |
| pseudocell_x10 | 0 | 0.839 (−0.114) | 0.766 (−0.057) | 0.856 (−0.003) | 0.696 (−0.139) |
| **context_x1** (peers) | 0 (β fit on pool) | **0.955 (+0.001)** | 0.872 (+0.049) | **0.916 (+0.057)** | **0.866 (+0.030)** |
| context_x1_pool_prior | 0 | 0.869 (−0.085) | 0.833 (+0.010) | 0.843 (−0.016) | 0.672 (−0.164) |
| film_lora_adapter | 667k | 0.865 (−0.088) | 0.874 (+0.051) | 0.845 (−0.014) | 0.735 (−0.101) |
| bulk_head | 592k | 0.872 (−0.081) | 0.867 (+0.044) | 0.832 (−0.027) | 0.717 (−0.119) |
| **side_channel** | 1.02M + 592k | 0.898 (−0.055) | **0.893 (+0.070)** | 0.826 (−0.033) | 0.730 (−0.105) |
| pca_imputer (no STACK) | 64 × 15k basis | 0.824 (−0.130) | 0.849 (+0.026) | 0.694 (−0.165) | 0.676 (−0.160) |

visible_r on native bulk (decoder fidelity with nothing hidden) goes from 0.29 / 0.41 (baseline) to 0.85 / 0.80 (`depth_scale_x1`), 0.88 / 0.88 (`bulk_head`), 0.90 / 0.89 (`side_channel`) and 0.96 / 0.89 (`context_x1`). resid_r, the sample-specific signal, goes from −0.03 / 0.02 to 0.14 / 0.30 (`depth_scale_x1`), 0.24 / 0.61 (`side_channel`) and 0.25 / 0.51 (`context_x1`). Pre-activation std on native GENtoniK returns from 212.7 to 5.3 (`depth_scale_x1`) and 5.9 (`side_channel`), inside the single-cell range.

What this shows:
- **Depth harmonization at 1× is the single biggest fix** (+0.55 / +0.36 r). 10× is nearly as good; 100× is already too deep for the frozen backbone (finding T1).
- **Thinning ensembles add a little over deterministic rescaling** (+0.018 on GENtoniK, +0.09 on sci-Plex pb400). Putting the draws in one window (`pseudocell`) adds nothing over separate windows.
- **The context decoder is the only variant that reaches the population mean on GENtoniK.** It is the strongest on homogeneous pseudobulk. It uses the peers' values of the hidden genes, which is the same information the LOO bar uses. Without peers (`pool_prior`) it still beats rescaling alone (0.869 vs 0.840; cell lines turn positive).
- **The trained adapters beat the population mean on the heterogeneous cell-line cohort without peer decoding** (side channel +0.070, CI 0.064–0.076). They also beat the model-free PCA imputer trained on the same pool on every set (cell lines 0.893 vs 0.849, GENtoniK 0.898 vs 0.824). So STACK's latent space contributes beyond a linear model of bulk.
- **The trained adapters do not beat context decoding on homogeneous cohorts** (GENtoniK, pb400). The pool has one sample per series, so the adapters never saw batch-matched peers and their learned α term underuses them. This is the E6/C6 gap.

### 7.2 Single-cell non-regression

| variant | organoid cells recon (gain) | sci-Plex nuclei recon (gain) | organoid pb400_thinned (gain) | organoid cells kNN |
|---|---|---|---|---|
| baseline | 0.421 (+0.026) | 0.256 (+0.032) | 0.408 (−0.009) | 0.797 |
| depth_scale_x1 / x10 / x100, thin_ensemble, pseudocell | 0.421 (+0.026) | 0.256–0.258 (+0.032–0.034) | 0.408 (−0.009) | 0.797–0.802 |
| context_x1 | 0.511 (+0.116) | 0.352 (+0.129) | 0.506 (+0.090) | 0.799 |
| context_x1_pool_prior | 0.475 (+0.080) | 0.316 (+0.093) | 0.465 (+0.049) | 0.799 |
| film_lora_adapter | 0.430 (+0.034) | 0.241 (+0.018) | 0.417 (+0.001) | 0.769 |
| bulk_head | 0.395 (+0.000) | 0.245 (+0.022) | 0.380 (−0.036) | 0.799 |
| side_channel | 0.396 (+0.000) | 0.249 (+0.026) | 0.380 (−0.036) | 0.801 |
| pca_imputer | 0.288 (−0.107) | 0.158 (−0.065) | 0.272 (−0.144) | 0.594 |

- The training-free input transforms leave every profile at or below D untouched. 55% of organoid cells and 11% of sci-Plex nuclei are deeper than D = 2,205.5 and are scaled or thinned down. That lowers their first-layer pre-activation std (organoid 4.5 → 3.4) but changes reconstruction and kNN by at most 0.005.
- Context decoding also *helps* single cells (+0.09 over STACK alone). It is a general way to use STACK's relative signal, not a bulk-only trick.
- The side channel's input path is gated off for cells: embeddings and kNN are unchanged. Its output head, and the stand-alone `bulk_head`, are **not** gated and cost 0.01–0.03 r on cells. Gating the head by the same depth gate (E5) is the obvious next step, so that cells use the frozen decoder.
- `film_lora_adapter` slightly lowers organoid kNN (0.80 → 0.77). Its native cell-line cross-modal accuracy (next section) also drops, because its FiLM is not gated.

### 7.3 Cross-modal: placing bulk samples in single-cell space

Nearest-centroid accuracy of MCF7/A549 bulk against sci-Plex groups, STACK cosine vs Spearman correlation (`results/*_cross_modal.json`):

| variant | native bulk vs cells | native bulk vs pb400_thinned | thinned bulk vs cells |
|---|---|---|---|
| baseline | 0.948 / 0.990 | **0.677** / 0.969 | 0.969 / 0.969 |
| depth_scale_x1 | 0.969 / 0.990 | **0.969** / 0.969 | 0.958 / 0.969 |
| thin_ensemble_x1 | 0.969 / 0.990 | 0.979 / 0.969 | 0.958 / 0.969 |
| context_x1, bulk_head (same embeddings as depth_scale_x1) | 0.969 / 0.990 | 0.969 / 0.969 | 0.958 / 0.969 |
| side_channel | 0.958 / 0.990 | 0.938 / 0.969 | 0.958 / 0.969 |
| film_lora_adapter | 0.771 / 0.990 | 0.927 / 0.969 | 0.938 / 0.969 |

Depth harmonization fixes the aggregate-matched failure (0.68 → 0.97) and brings STACK level with correlation. The embedding map (`figures/embedding_map.png`) shows the same: after `depth_scale_x1`, native bulk lands on top of its own thinned copies, instead of in a separate cluster away from every cell group. On GENtoniK there are no matched organoid labels. Correlation assigns all 29 samples to 5-month CPN. STACK variants assign most samples to "23 days | FOXG1− EMX1− neurons" (11–16 of 29) and spread the rest over neighbouring neuronal groups. Neither is ground truth for iPSC-derived 2D neurons, which matches the mixture-versus-cell problem (finding 7).

### 7.4 Batch confounding

The cell-line bulk spans 83 GEO series. Only the 21 samples whose series has another member in the set (8 series) can be scored for kNN with the series as label. On native bulk, that score is 0.24–0.38 for every variant, against 0.24 for PCA of log-CPM. Across all depth-sweep rows it is 0.14–0.48. Label kNN stays at 0.95–0.99. No adapter learned to cluster by batch (series) instead of biology (cell line).

### 7.5 Downstream bulk task: GENtoniK drug response (`treatment_effect.py`)

In one window of all 29 GENtoniK samples, 20% of genes are hidden (5 draws) and imputed. The per-line log2 fold change (GENtoniK cocktail vs DMSO) of the imputed genes is compared with the fold change measured from the real counts. For three lines it is also compared with the published DESeq2 table. Sanity check: the measured fold changes agree with the published ones at r = 0.91 (GM03348), 0.86 (MSK-SRF001) and 0.79 (WA09).

| variant | r vs measured log2FC (mean of 4 lines) | r vs published log2FC (3 lines) | sign agreement on published padj < 0.05 genes |
|---|---|---|---|
| population mean (LOO) | −0.855 | −0.628 | 0.006 |
| baseline | 0.073 | 0.072 | 0.558 |
| depth_scale_x1 | 0.239 | 0.215 | 0.679 |
| thin_ensemble_x1 | 0.262 | 0.256 | 0.704 |
| context_x1 | 0.139 | 0.153 | 0.605 |
| bulk_head | 0.221 | 0.221 | 0.720 |
| side_channel | 0.177 | 0.203 | 0.719 |
| pca_imputer | **0.429** | **0.384** | **0.854** |

- The LOO mean is anti-correlated with the true fold change by construction. A treated sample's LOO mean contains all the controls of its line, and vice versa, so it predicts the opposite direction. This is why a high reconstruction score near the population mean says nothing about biology.
- Depth harmonization turns STACK from uninformative (0.07) into weakly informative (0.24–0.26) about drug response, and trained heads reach 72% sign agreement on significant genes.
- Context decoding *hurts* this task relative to plain rescaling, because it anchors to the peers' mean.
- The side channel is unstable across lines: 0.29 on WA09 but 0.01 on GM03348.
- The linear PCA imputer is clearly better. Quantitative within-cohort contrasts depend on the mid- and low-abundance genes that harmonized input cannot see (finding 6), and the side channel is too small or too undertrained to recover them.

## 8. What we learned

1. Most of STACK's bulk problem is one input-scale defect, not missing biology. A one-line depth harmonization recovers most of it, with zero training and zero single-cell cost. It should be the default whenever an input profile is deeper than the single-cell reference.
2. STACK's *relative* signal (how a sample differs from its peers) is real and useful in both modalities. Decoding relative to context is the most effective single change measured, and it also improves single cells.
3. Its *absolute* resolution is limited by a decoder trained on sparse targets. Small bulk-trained heads raise it, and beat a linear model trained on the same data, but they don't reach bulk-native quality on fine quantitative contrasts.
4. The remaining gap is architectural: the backbone never sees most bulk information (harmonization keeps about 1–3% of expressed genes quantitatively), and cross-sample attention never learned to borrow values from peers.

## 9. Recommendations for a natively dual-modality STACK

From the brainstorm's §4, ordered by expected impact given the measurements above:

1. **A depth-invariant value encoding plus an explicit depth input.** Use normalized values (log-CPM or rank) as content and log(depth) as a separate token or FiLM scalar (fixes findings 1–2). Make the first layer pre-norm (C8).
2. **Gene-identity tokens with an explicit mask token and missing-gene token** (C7), distinct from a measured zero (finding 3). This also lets gene panels differ between protocols (finding 9).
3. **Per-sample masking with peers in pretraining**, so cross-sample attention learns to copy a gene's value from similar samples. This replaces the hand-built D1 (findings 4, 8).
4. **A likelihood and training targets that span 10³–10⁸ reads**: NB with modality- or depth-dependent dispersion, trained on native-depth targets. Without them the decoder never learns sub-1/library resolution (finding 5).
5. **Mixture semantics.** Add an "aggregated N cells" token, and losses that enforce aggregation consistency in decoding (D6) and embedding (E3). Pretrain on single cells, pseudobulks across k, depth thinnings, and real ARCHS4 bulk with protocol tokens, using series-level splits.
6. **A two-level context** (C9): within-sample cell attention, plus attention across samples and cross-attention from bulk tokens to a single-cell bank (deconvolution, C5).

Cheaper next steps on the frozen checkpoint:
- Gate the bulk head by depth so cells keep the frozen decoder (E5).
- Train on complete held-out series so the adapters learn cohort context (E6, then C6).
- Make the side channel larger and train it longer, reporting the PCA control alongside.
- Add the multi-depth embedding view (A5) for identity tasks.

## 10. Reproduce

```bash
cd bulk_stack
# Data (ARCHS4 is streamed remotely; SCP1756 needs the same access as engram_inference)
../.venv/bin/python fetch_archs4.py          # GENtoniK, MCF7/A549 bulk, 384-sample training pool
../.venv/bin/python fetch_singlecell.py      # organoid cells and sci-Plex controls (raw downloads shared with engram_inference/data)
../.venv/bin/python profiles.py              # data/profiles.h5ad: cells, pseudobulks, thinned, bulk, depth sweep
# Unmodified STACK across dataset types
../.venv/bin/python diagnose.py baseline
# Training on the ARCHS4 pool only
../.venv/bin/python train_bulk.py            # pool stats, context beta, bulk_head, side_channel, pca_imputer
../.venv/bin/python train_adapter.py         # film_lora_adapter
# Every variant on every set, then the downstream task, figures and tables
../.venv/bin/python run_variants.py          # all variants in config adapter.variants (or name some)
../.venv/bin/python treatment_effect.py
../.venv/bin/python figures.py
../.venv/bin/python summarize.py baseline depth_scale_x1 context_x1 bulk_head side_channel
```

Everything runs on CPU. On 4 cores, `run_variants.py` takes about 40 minutes and training about 15 minutes. Run jobs one at a time: concurrent PyTorch processes oversubscribe the CPU badly. The model checkpoint and gene list are shared with `engram_inference/models`.

### Files

| file | purpose |
|---|---|
| `config.yaml` | every dataset rule, benchmark setting, adapter hyperparameter and variant definition |
| `common.py` | config, paths, threads, frozen STACK loading, gene-symbol mapping |
| `fetch_archs4.py` | remote ARCHS4 metadata filter and parallel column reader |
| `fetch_singlecell.py` | organoid and sci-Plex reference cells |
| `profiles.py` | builds every evaluation set |
| `stack_ops.py` | window inference, masking, metric helpers |
| `evaluate.py` | the benchmark (Section 2 metrics, cross-modal) |
| `diagnose.py` | baseline run |
| `adapters.py` | all training-free runners and trained modules |
| `train_bulk.py` | pool statistics, β fit, bulk head, side channel, PCA control |
| `train_adapter.py` | the first FiLM + LoRA adapter |
| `run_variants.py` | evaluates every configured variant |
| `treatment_effect.py` | GENtoniK fold-change task, with published DESeq2 comparison |
| `figures.py`, `summarize.py` | figures and tables |
| `docs/brainstorm_agent.md` | the full brainstorm |
| `results/` | per-variant `*_sets.csv` and `*_cross_modal.json`, training histories, trained weights (`*.pt`, `*.npz`), `treatment_effect*.csv` |

## 11. Limitations

- **Small cohorts.** Two real bulk cohorts with labels (29 and 96 samples) and one bulk downstream task. GENtoniK differences of ±0.01–0.02 are within mask-draw noise, even though the per-profile bootstrap CIs are narrower.
- **Peer information.** The LOO baseline and the context decoder both use the peers' values of hidden genes. Both are fair within-cohort imputation settings, but neither applies to a single isolated sample; `context_x1_pool_prior` is the single-sample variant.
- **Small training pool.** 384 samples, one per series. The trained adapters are small and trained briefly. More data, complete series and longer training are likely to help more than architecture tweaks at this scale.
- **Gene panel.** All bulk profiles are mapped to STACK's 15,012 genes by symbol, and genes outside that panel are dropped.
- **Unused information.** The cell-type labels of the organoid groups are used only for evaluation. GENtoniK has no single-cell ground truth, so its cross-modal assignment is descriptive.
