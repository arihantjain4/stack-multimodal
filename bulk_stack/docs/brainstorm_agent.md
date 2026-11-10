# Making STACK work on bulk RNA-seq: design brainstorm

Scope: a design brainstorm grounded in the baseline benchmark (`results/baseline_sets.csv`, `results/baseline_cross_modal.json`) plus a handful of new, small, read-only experiments run for this note (Section 1). No file in `src/stack` or `bulk_stack` was changed. Every idea below is meant to wrap, subclass or compose around the published `Stack-Large-Aligned` checkpoint.

## TL;DR

1. **Normalizing depth alone, with no training, recovers most of the lost reconstruction quality.** Scaling a bulk profile down to the single-cell median depth before `log1p` lifts hidden-gene correlation against the *native* bulk truth on GENtoniK from 0.27 to 0.84. Thinning gives 0.85, and a mean of 8 thinning draws gives 0.86. Visible-gene correlation goes from 0.29 to 0.85–0.86. On the cell lines it goes from 0.41 to 0.78, and on organoid pb400 from 0.68 to 0.85. Deterministic rescaling is as good as binomial thinning: the fear that "non-integer, non-sparse input is out of distribution" did not materialize at single-cell depth. Thinning ensembles mainly help the embeddings: GENtoniK kNN rises from 0.66 with one draw to 0.90 with eight.
2. **Even after depth harmonization, STACK alone does not beat the population mean on bulk** (0.84 vs 0.95 on GENtoniK, 0.78 vs 0.82 on the cell lines). At single-cell depth, only 3% (GENtoniK) and 1% (cell lines) of the genes a bulk sample expresses at ≥1 CPM have an expected count of at least 1. The harmonized input therefore throws away almost all quantitative bulk information.
3. **A training-free "residual-to-context" decoder gives the first positive gain over the population baseline.** It multiplies the window's leave-one-out (LOO) population mean by STACK's *relative* prediction, px_i divided by the LOO mean of px. Results: GENtoniK 0.957 vs 0.953, cell lines 0.870 vs 0.824. STACK carries useful *relative* (sample-vs-peers) signal but poor *absolute* resolution.
4. **The root cause is visible in the weights.** The first Linear layer has a strong "sum over genes" direction: ‖W·1‖ is 15.5× the norm expected for a random sign vector. In native bulk, the uniform log-depth shift accounts for a cosine of 0.96 of the pre-activation and has 82× the norm of the depth-normalized content. Pairwise cosine between bulk pre-activations is 0.997, versus 0.913 between single cells, so bulk profiles collapse at layer 0, before any attention runs.
5. **Recommended order.** First the depth harmonizer (training-free), then the context decoder (training-free). Then a small bulk-aware output head (about 0.59M parameters, NB likelihood with bulk dispersion, trained on the ARCHS4 pool on cached embeddings). Then a full-depth side-channel input adapter (about 0.54M parameters) with depth FiLM. Then an aggregate-matched cross-modal reference. Optionally, LoRA on cross-sample attention trained with per-sample masking. Every trained adapter must be compared against a **model-free** linear imputer trained on the same pool, so we know STACK adds something.

---

## 1. New measurements made for this brainstorm

All runs use the unmodified checkpoint, windows of up to 64 profiles from one set, 20% of genes hidden (2–3 mask draws), and the `evaluate.py` metric definitions. **There is one key difference: the truth is always the native-depth profile**, while only the *input* is transformed. The existing `*/thinned` rows compare against thinned truth, which mixes up "the model is bad" with "the target is noisy". Mask draws differ from the baseline run, so identity rows differ from `baseline_sets.csv` by up to about 0.025.

Columns: `resid_r` is the Pearson correlation, over hidden expressed genes, between (prediction − LOO mean) and (truth − LOO mean), both in log-CPM. It measures sample-specific information beyond the population prior. `x1` is the reference single-cell median depth (2,356 for organoid, 854 for sci-Plex). `rescale` is `log1p(x·min(1, D/lib))`. `clipshift` is `max(log1p(x) + log(min(1, D/lib)), 0)`. `meanfield` is `E[log1p(N)]` with N ~ Poisson(x·D/lib), the analytic mean over thinning draws. `thin` is binomial thinning with K draws, averaging px and embeddings.

**GENtoniK bulk (native depth 2.0e7, n=29, LOO baseline 0.953; kNN label = line; knn_pca = 0.97)**

| input transform | depth | K | recon_r | resid_r | visible_r | knn | emb_cos | preact_std |
|---|---|---|---|---|---|---|---|---|
| identity | native | 1 | 0.273 | −0.035 | 0.287 | 0.83 | 0.73 | 212.7 |
| rescale | x1 | 1 | 0.839 | 0.147 | 0.850 | 0.83 | 0.67 | 5.5 |
| rescale | x10 | 1 | 0.821 | 0.117 | 0.829 | 0.86 | 0.65 | 24.9 |
| rescale | x100 | 1 | 0.644 | −0.018 | 0.649 | 0.90 | 0.67 | 68.3 |
| clipshift | x1 | 1 | 0.723 | 0.088 | 0.729 | 0.79 | 0.62 | 1.4 |
| clipshift | x100 | 1 | 0.807 | 0.063 | 0.815 | 0.93 | 0.64 | 62.5 |
| meanfield | x1 | 1 | 0.840 | 0.149 | 0.851 | 0.79 | 0.67 | 4.5 |
| meanfield | x10 | 1 | 0.830 | 0.135 | 0.839 | 0.86 | 0.66 | 22.7 |
| thin | x1 | 1 | 0.851 | 0.152 | 0.858 | 0.66 | 0.55 | 4.5 |
| thin | x1 | 8 | **0.856** | **0.154** | **0.864** | 0.90 | 0.79 | 4.5 |
| thin | x10 | 1 | 0.833 | 0.147 | 0.843 | 0.83 | 0.60 | 22.8 |
| thin | x100 | 4 | 0.720 | 0.025 | 0.698 | 0.86 | 0.68 | 66.5 |

**Cell-line bulk (native 1.3e7, n=96, LOO baseline 0.824, knn_pca 0.97)**

| input transform | depth | K | recon_r | resid_r | visible_r | knn | emb_cos |
|---|---|---|---|---|---|---|---|
| identity | native | 1 | 0.413 | 0.023 | 0.411 | 0.96 | 0.64 |
| rescale | x1 | 1 | **0.775** | **0.278** | **0.797** | 0.99 | 0.49 |
| rescale | x10 | 1 | 0.766 | 0.233 | 0.784 | 0.98 | 0.49 |
| rescale | x100 | 1 | 0.731 | 0.153 | 0.746 | 0.98 | 0.49 |
| meanfield | x1 | 1 | 0.771 | 0.273 | 0.795 | 0.98 | 0.50 |
| thin | x1 | 8 | 0.750 | 0.236 | 0.780 | 0.99 | 0.66 |
| thin | x10 | 1 | 0.774 | 0.258 | 0.785 | 0.98 | 0.48 |

**Organoid pb400 (1.2e6 reads, 128 profiles, LOO baseline 0.857)**: identity 0.678 / resid 0.139 / visible 0.727. Rescale x1: 0.850 / 0.494 / 0.889. Thin x1 K=8: 0.859 / 0.493 / 0.890. Thin x10: 0.858 / 0.495 / 0.887. Rescale x100: 0.782 / 0.306 / 0.829.

**Residual-to-context decoding** (rescale x1 input, 3 mask draws). The prediction is `LOO_mean_frac · (px / LOO_mean_px)^β`, renormalized:

| set | STACK px | LOO mean | β=1 | β=0.5 | 50/50 blend |
|---|---|---|---|---|---|
| GENtoniK | 0.837 | 0.953 | 0.955 | **0.957** | 0.923 |
| cell lines | 0.781 | 0.824 | **0.870** | 0.858 | 0.827 |

**Mixed windows** (32 rescaled bulk + n real reference cells in the same window, no attention mask). GENtoniK kNN is 0.83 / 0.90 / 0.86 for n = 0 / 32 / 96. Nearest-centroid assignment stays dominated by "23days | FOXG1− EMX1− neurons" (14 / 10 / 17 of 29). Cell lines: kNN 0.98 / 0.98 / 0.95, with assignments about 49 MCF7 / 46 A549 / 1 K562 in every case. Mixing windows barely changes anything.

**Information retained by harmonization.** A bulk sample expresses about 11.0k (GENtoniK) or 9.6k (cell lines) model genes at ≥1 CPM, versus 1,396 detected genes per organoid cell. The fraction of those genes with expected count ≥1 after harmonization:

| depth | GENtoniK | cell lines |
|---|---|---|
| x1 | 3.3% | 0.8% |
| x10 | 44% | 20% |
| x100 | 84% | 72% |

**Checkpoint weights.** Parameters: gene_reduction 24.0M, attention layers 92.2M, output_mlp 101.2M, cls head 0.32M; 217.8M in total. In the first Linear, ‖W·1‖ / E‖W·r‖ = 15.5 for a random sign vector r. For native GENtoniK, decompose the pre-activation into a shift part, W·(log(lib/D)·1[x>0]), and a content part, W·log1p(x·D/lib). The shift has 82× the norm of the content and a cosine of 0.96 with the full pre-activation. Pairwise cosine of bulk pre-activations is 0.997, against 0.913 for organoid cells. The output logit bias is negligible (std 0.019; correlation 0.51 with the log pseudobulk fraction), so px is driven entirely by the embedding. There is no LayerNorm before the first `cell_attn`: the layers are post-norm, `LN(x + attn(x))`, so layer 1's attention scores scale with the square of the input scale.

CPU timing: one forward pass of 64 profiles takes about 0.4 s on 2 threads. The memory-mapped checkpoint loads in 2 s.

Takeaways that change the design:
- (T1) The depth problem is solved to first order by *any* depth-harmonizing transform at x1–x10. Above about x10 (more than ~3e4 reads per profile) everything degrades, and that sets the operating envelope of the frozen backbone.
- (T2) There is a task trade-off. Reconstruction peaks at x1, while line identity in the embedding peaks at x100 (kNN 0.90–0.93), because more genes are quantitatively visible there. That motivates fusing multiple depths or adding a full-depth side channel.
- (T3) The frozen decoder is the ceiling: visible_r is at most 0.86–0.89 even with every gene visible. STACK's relative signal is useful (resid_r 0.15–0.49, and context decoding beats the LOO mean), but its absolute profile is over-smoothed toward a prior.
- (T4) Sparsity was not the problem. `rescale` (dense, non-integer) ≈ `meanfield` ≈ `thin` K=1 for reconstruction. `clipshift`, which imitates sparsity deterministically, is worse at x1.

---

## 2. Why STACK cannot natively reason across single-cell and bulk today

Each item points to the measurement that supports it.

1. **The input is not depth-invariant, and depth is an explicit feature.** `features = log1p(counts)` without library normalization. In single-cell data, depth correlates with cell size and quality, and the model uses it: ‖W·1‖ is 15.5× the random level, a dedicated "total expression / detected genes" direction. Bulk adds about log(2e7/2.4e3) ≈ 9 nats to ~90% of genes. That one direction then dominates the 1,600 pre-activations (cosine 0.96, 82× norm), so every bulk sample looks like the same extreme cell (pairwise cosine 0.997). This is the measured collapse: preact_std grows 47×, and visible_r falls from 0.65 to 0.29.
2. **There is no normalization before the first attention.** The layers are post-norm, so the first `cell_attn` sees unnormalized tokens. Its attention logits grow quadratically with input scale and its softmax becomes near-argmax. LayerNorm only rescales *after* the residual add, by which point the direction is already dominated by the shift.
3. **Zero means three things.** "Not detected", "masked" and "absent" are all `0` in log space; there is no mask token. In single-cell data ~91–95% of entries are zero, so a masked gene is indistinguishable from a dropout, which is benign. In bulk ~10–20% are zero, so a masked gene looks like a confidently silent gene. At x1 harmonization this is benign again; at x10 and above it is not.
4. **Masks are shared across the whole window** in both pretraining (`apply_mask`) and finetuning (`apply_finetune_mask`). Cross-cell attention therefore never learned to *copy* a gene's value from a peer; it learned population context at the embedding level. Bulk imputation is naturally "this sample is missing a gene the peers have", and STACK has no learned mechanism for it. The LOO baseline exploits exactly that information, which is why the model alone cannot beat it, while context decoding (which injects peers' values for that gene) can.
5. **Decoder resolution was learned on sparse targets.** The NB NLL with ~1e3–1e4 UMIs mostly teaches "which ~1–3k genes are non-zero and roughly how many". Relative abundances finer than about 1/lib were never supervised. Hence visible_r ≤ 0.89 even with full information, and absolute bulk profiles (baseline r 0.95) are out of reach without a new head.
6. **Information is in the detection pattern, not in quantities.** A single cell is about 1.4k detected genes, so the representation is largely a detection-pattern code. A bulk sample has ~11k genes quantitatively expressed. Harmonizing to x1 maps bulk back into the detection-pattern code but keeps only the top ~3% of genes quantitatively (Section 1). A frozen STACK can only see bulk through a keyhole.
7. **Mixture versus cell.** A bulk profile is a library-weighted *sum* over cells. A thinned bulk is a draw from the *average* cell of that population, which is not a real cell state for heterogeneous tissue. Its embedding sits between types, so "nearest single-cell centroid" is ill-posed. GENtoniK scatters across 5–7 organoid groups, while correlation picks one. There is also a Jensen gap: `log1p(Σ cells) ≠ mean(log1p(cell))`, so averaged single-cell features differ from bulk features even at matched depth.
8. **Cross-cell attention semantics.** Training windows are 256–512 cells drawn consecutively from one file: a single biological population plus batch. Bulk windows are cohorts of populations (different donors, series, treatments) and are small (29 here). The SW regularizer works on *window-centered* embeddings, so the embedding frame is partly defined by window composition. Mixed windows did not fix assignment (Section 1).
9. **Gene panel and protocol.** The 15,012 genes are single-cell HVGs; many bulk-informative low-abundance genes and long or non-polyA transcripts are rare or absent in single-cell data. Full-length protocols add gene-length bias. Evidence says this is secondary (thinned bulk behaves like cells; cell-line cross-modal accuracy is 0.96), but it bounds absolute accuracy.
10. **No modality, depth or aggregation token.** The finetune stage showed that a 1,600-parameter additive embedding (`query_pos_embedding`) can switch the model's role. Nothing analogous exists that tells the model "this is an aggregate of N cells at depth D, measured by protocol P".

---

## 3. Idea catalogue

Format per idea: **Mechanism / Addresses / Expected metrics / CPU cost / Risks / Test**. Ideas marked **[arch]** require changing or re-pretraining the backbone and are not feasible here. "Measured" means a number from Section 1.

### A. Input representation and normalization

**A1. Deterministic depth rescaling (`rescale`)**
- Mechanism: `log1p(x · min(1, D/lib))` before `gene_reduction`, with D = `depth_multiplier` × the reference single-cell median library size, derived from data.
- Addresses: finding 1 (depth shift) and finding 2.
- Expected metrics (measured): GENtoniK recon 0.27→0.84, visible 0.29→0.85; cell lines recon 0.41→0.78; pb400 0.68→0.85; preact_std back to the single-cell range (5.5 vs 4.5). kNN unchanged or improved.
- CPU cost: zero.
- Risks: discards fine quantitative information (only 3% of expressed genes have ≥1 expected count at x1); non-integer inputs (not a problem as measured); D is task-dependent (see T2).
- Test: already measured. Sweep `depth_multiplier` ∈ {1, 2, 5, 10} on the *training pool* (not the evaluation sets) to pick it.

**A2. Binomial thinning, single draw or K-draw ensemble (`thin`)**
- Mechanism: binomial thinning to D, K independent draws, average px and embeddings.
- Addresses: findings 1 and 6; exactly in-distribution sparsity.
- Expected metrics (measured): recon equal to or 0.01 above rescale. kNN goes 0.66 (K=1) → 0.90 (K=8) on GENtoniK because the single-draw noise in embeddings averages out. emb_cos rises (0.55→0.79) since averaging contracts embeddings, so read emb_cos as non-comparable across K.
- CPU cost: K× inference.
- Risks: stochastic; uses ~K·D reads out of 1e7 (gene-level information loss); averaging embeddings is not equivalent to embedding the average.
- Test: K ∈ {1, 4, 8, 16}; also report the per-draw variance of embeddings as an uncertainty estimate.

**A3. Mean-field (Rao–Blackwellized) thinning (`meanfield`)**
- Mechanism: feed `E[log1p(N)]`, N ~ Poisson(x·D/lib), computed analytically. `gene_reduction[0]` is linear, so this gives *exactly* the mean first-layer pre-activation of an infinite thinning ensemble in one pass.
- Addresses: findings 1 and 6, deterministically.
- Expected metrics (measured): ≈ rescale (0.840 / 0.771 / 0.849).
- CPU cost: one pass plus a Poisson expectation table (vectorized; cheap).
- Risks: the nonlinearity after the Linear means it is not the ensemble's *output* mean; non-sparse input.
- Test: compare with thin K=16 on embeddings (kNN) as well as reconstruction.

**A4. Clip-shift (sparsity-imitating deterministic rescale)**
- Mechanism: `max(log1p(x) − log(lib/D), 0)`, which is the log-ratio for high genes and zero for genes below about one expected count.
- Addresses: finding 6, with a sparsity pattern that mimics single-cell data.
- Expected metrics (measured): worse at x1 (0.72 / 0.66); at x100 it is good for embeddings (GENtoniK kNN 0.93) and reconstruction 0.81.
- CPU cost: zero.
- Risks: a hard threshold distorts mid-range genes.
- Test: as a second "embedding view" in A5.

**A5. Multi-resolution fusion**
- Mechanism: run the model at two harmonization depths (x1 for decoding, x10–x100 for embeddings), or concatenate or average embeddings across depths. Decode from x1 but take identity features from the high-depth view.
- Addresses: T2 (reconstruction peaks at x1, identity at x100).
- Expected metrics: recon as x1, kNN as x100 (0.90–0.93 on GENtoniK).
- CPU cost: 2–3×.
- Risks: a concatenated embedding is a new space, not comparable to single-cell embeddings.
- Test: knn_stack with each view and with the fusion; cross-modal assignment with each view.

**A6. Context-mean imputation of masked inputs (training-free mask token)**
- Mechanism: instead of `0` for hidden genes, input `log1p(D · C_g)`, where C is the window's peer-mean fraction (or a training-pool prior for single-sample inference).
- Addresses: finding 3 (zero means masked) at x10+ depths, where zeros are informative.
- Expected metrics: at x1 negligible (hidden genes are mostly 0 anyway); at x10 recon should move toward the LOO baseline. Risk of simply copying the prior (resid_r → 0).
- CPU cost: zero.
- Risks: leaks peers' values into the input (fine when the benchmark baseline already uses them, but state it explicitly).
- Test: x10 harmonization with and without A6, reporting recon_r and resid_r.

**A7. Per-gene protocol offset δ_g (bulk ↔ single-cell)**
- Mechanism: subtract a per-gene log offset (gene-length or full-length vs 3′ bias, kallisto vs CellRanger) from bulk log-CPM before harmonizing. δ_g is estimated from data as the median over matched pairs of (log bulk CPM − log single-cell pseudobulk CPM), or learned as part of an adapter.
- Addresses: finding 9; the GENtoniK cross-modal mismatch.
- Expected metrics: small reconstruction gain; better cross-modal assignment.
- CPU cost: zero at inference.
- Risks: **leakage**. The only matched pairs here are the evaluation sets (MCF7/A549 bulk vs sci-Plex), so estimate δ on one half of the GEO series and evaluate on the other, or learn it from the training pool by self-supervision. δ also absorbs real biology (bulk tissue vs single-cell line).
- Test: series-split cross-validation on cell lines; then apply to GENtoniK unchanged.

**A8. Moment-matched tokens (training-free "input LayerNorm")**
- Mechanism: after `gene_reduction`, standardize each of the 1,600 token dimensions to the reference single-cell token mean and std. Statistics come from reference single-cell windows and from the bulk set, or from the training pool.
- Addresses: finding 2.
- Expected metrics: on native-depth input, partial recovery but worse than A1, because the shift is a *direction*, not a scale (cosine 0.96), and moment matching does not remove a direction. After A1 it is a no-op.
- CPU cost: zero.
- Risks: removes real signal.
- Test: identity vs A8 vs A1 on depth_x100 and native.

**A9. Project out the depth direction** (the analytic equivalent of A1)
- Mechanism: subtract W·(log(lib/D)·1[x>0]). This equals rescale for expressed genes up to the `log1p` curvature, so it is a sanity check rather than a new method. It is worth stating because it shows *exactly* which subspace carries the failure.

### B. Depth, modality and aggregation conditioning

**B1. Depth FiLM on pre-activations or tokens (trained)**
- Mechanism: s = log(native lib / D). Apply `pre' = pre·(1 + a·s) + b·s` with a, b ∈ R^1600, zero-initialized (3,200 parameters). Alternatively a 2-layer MLP on s.
- Addresses: lets the harmonized model *know* the true depth (and therefore its confidence) without the depth flooding the input.
- Expected metrics: small reconstruction gain; prevents the adapter from treating x1-harmonized bulk as a low-quality cell.
- CPU cost: trivial; needs backprop through the backbone.
- Risks: s is constant within a bulk cohort, so the model cannot learn much from it unless training spans depths (thinned pool samples at random depths).
- Test: ablate on/off within the trained input adapter (S4 below).

**B2. Modality or aggregation embedding (1,600 parameters)**
- Mechanism: a learned `(100, 16)` additive token embedding for "aggregate profile", mirroring `query_pos_embedding`. Optionally scale it by log(number of cells) for pseudobulk.
- Addresses: finding 10.
- Expected metrics: lets the backbone switch to "mixture mode"; may improve cross-modal assignment if trained with E3.
- CPU cost: trivial.
- Risks: so few parameters may underfit; it might also shift bulk *away* from single-cell space, which would hurt cross-modal mapping. Use E3 to anchor it.
- Test: train it alone with the frozen backbone and context decoder.

**B3. Bulk as ICL queries with a single-cell prompt**
- Mechanism: windows of reference cells as prompt plus harmonized bulk as query, with `query_pos_embedding` added to bulk and the finetune attention mask (prompt cannot see queries).
- Addresses: finding 8; uses the model's ICL machinery for "describe this bulk sample in terms of these cells".
- Expected metrics: unknown. The unmasked mixed-window test changed little, so expect small changes.
- CPU cost: zero.
- Risks: `query_pos_embedding` was trained for generation targets, not bulk.
- Test: cross-modal assignment with and without the mask and embedding.

**B4. Effective-heterogeneity conditioning**
- Mechanism: estimate how heterogeneous the bulk mixture is (for example the entropy of a deconvolution, C5) and condition on it with FiLM.
- Addresses: finding 7.
- Cost / risk: needs C5 first. Speculative.

### C. Tokenization and architecture

**C1. Full-depth side channel into tokens (trained; low-rank)**
- Mechanism: z = per-gene standardized `log1p(CPM)` of the *native* profile (μ_g, σ_g from the training pool). Then Δ = U·GELU(V·z), reshaped to (100, 16) and added to the tokens after `gene_reduction` of the harmonized input. V: 15012→r, U: r→1600, U zero-initialized.
- Addresses: finding 6 and T2 (gives the backbone the 97% of quantitative information that harmonization discards) while keeping the backbone's input in-distribution.
- Expected metrics: higher resid_r and recon (target: beat LOO without context decoding); kNN toward knn_pca.
- CPU cost: about 0.53M parameters at r=32, needs backprop through 9 layers; roughly 0.6 s per step at window 32.
- Risks: it could learn a shortcut that bypasses STACK (needs the model-free control, G7); overfits with ~300 training samples; may drift single-cell embeddings unless gated by depth (zero for cells; see E5).
- Test: S4 in the shortlist.

**C2. LoRA on `gene_reduction[0]` with harmonized input**
- Mechanism: W' = W + B·A, rank r. A cheaper alternative to C1 that reuses the input path.
- Addresses: finding 1 (learned depth-robust projection).
- Expected metrics: like C1 but limited to the harmonized information.
- CPU cost: about 0.5M parameters at r=32.
- Risks: changes single-cell behaviour too, unless gated.

**C3. Window composition strategies (training-free)**
- Mechanism, four variants: (a) cohort windows by GEO series or batch; (b) mixed windows with reference cells (measured: little effect); (c) "self windows": K thinned draws of the *same* sample as a homogeneous population, with the embedding taken as the mean (an attention-aware version of A2); (d) window-size sweep (8 / 16 / 32 / 64) to learn how much context matters for bulk.
- Addresses: finding 8.
- Expected metrics: (c) might improve emb stability; (a) is needed so peers are batch-matched for context decoding.
- CPU cost: zero.
- Risks: (c) creates a population with zero heterogeneity, which is out of distribution for cross-cell attention.
- Test: report each metric versus window composition.

**C4. Aggregate-to-aggregate cross-modal reference (training-free)**
- Mechanism: compare harmonized bulk embeddings with centroids of **thinned pseudobulk** reference groups (pb400_thinned or pbk_thinned), not single cells. Both sides are then "average-of-population" draws at the same depth.
- Addresses: finding 7 (mixture vs cell).
- Expected metrics: better GENtoniK assignment consistency, with correlation (5-month CPN) as the reference. Cell lines should stay at ≥0.96.
- CPU cost: zero.
- Risks: true heterogeneous tissue still has no single best group (use C5 instead).
- Test: `cross_modal` with pb_thinned centroids.

**C5. Deconvolution head: bulk attends to a bank of single-cell embeddings (trained, small)**
- Mechanism: a cross-attention or softmax-weights module from the harmonized bulk embedding (query) to reference group centroids (keys). Its output is mixing weights w. Loss: bulk fractions ≈ Σ_k w_k · group pseudobulk fractions (multinomial or log-CPM loss).
- Data: mixtures of real cells from several groups with known proportions (sums of real cells; allowed), plus the ARCHS4 pool for self-supervision.
- Addresses: finding 7; gives a principled cross-modal readout instead of nearest centroid.
- Expected metrics: new metric, proportion error on held-out real-cell mixtures.
- CPU cost: under 0.1M parameters on cached embeddings.
- Risks: identifiability with correlated groups; reference/bulk protocol mismatch (A7).
- Test: G3.

**C6. LoRA on cross-sample attention (`gene_attn.qkv`, `gene_attn.proj`) trained with per-sample masking**
- Mechanism: rank-r LoRA in all 9 layers. Train with masks that hide a gene only in some samples of the window, so attention can learn to pull peer values.
- Addresses: findings 4 and 8; learns what context decoding does by hand.
- Expected metrics: recon above LOO without the hand-made composition.
- CPU cost: about 0.23M parameters at r=4 on qkv only, or 0.35M with proj as well, with backprop through the layers.
- Risks: the training pool is one sample per series, so windows are heterogeneous cohorts and the learned peer-copying may not transfer to batch-matched cohorts. Needs a few complete held-out series (E6).
- Test: per-sample-mask variant of `run_set`.

**C7. [arch] Gene-identity tokenization with continuous, depth-invariant values**
- Mechanism: replace `Linear(15012→1600)` with gene embeddings × value encoding (log-CPM bins or continuous value MLP), pooled into 100 tokens by a Perceiver-style cross-attention.
- Addresses: findings 1, 3 (explicit mask token), 6 and 9 (any gene panel).
- Cost: re-pretraining. Not feasible here.

**C8. [arch] Pre-norm layers with an input LayerNorm**
- Mechanism: standard pre-LN, so no layer sees unnormalized activations.
- Addresses: finding 2.
- Cost: re-pretraining.

**C9. [arch] Hierarchical cell → sample model**
- Mechanism: a sample-level transformer over sets of cell embeddings (single-cell) or over a mixture token (bulk). A bulk token is trained to equal the pooled embedding of its constituent cells.
- Addresses: findings 7 and 8; natively dual (see Section 4).
- Cost: new model.

### D. Output head and likelihood

**D1. Residual-to-context decoding (training-free)**
- Mechanism: `p̂_i ∝ C_i ⊙ (px_i / C̃_i)^β`. C_i is the LOO peer mean of observed fractions; C̃_i is the LOO mean of the model's px over the same peers. β is a config value fit on the training pool.
- Addresses: findings 4 and 5 and T3. It uses STACK only for *relative* deviations, where it has signal.
- Expected metrics (measured): GENtoniK 0.957 (+0.004 over LOO), cell lines 0.870 (+0.046).
- CPU cost: zero.
- Risks: needs peers that have the gene measured (fine for cohort imputation; not for single-sample use, which needs a prior from the training pool). Gains come partly from peers being heterogeneous (cell lines are mixed MCF7/A549); in homogeneous cohorts the gain is small.
- Test: also report it per window composition (C3a) and against a model-free kNN-peer baseline (G7).

**D2. Copy path for visible genes**
- Mechanism: `logits' = logits + κ·m_vis ⊙ log(frac_obs + ε)`, with ε = one read divided by the native library size (derived), κ fixed to 1 or learned.
- Addresses: T3 (the decoder cannot reproduce visible input; visible_r ≤ 0.89).
- Expected metrics: visible_r → about 1. Hidden genes are unaffected unless trained jointly, because the softmax renormalizes. This matters for ICL generation of bulk (keep measured genes, change the rest).
- CPU cost: zero.
- Risks: makes visible_r meaningless as a metric (report it separately).

**D3. Per-gene bulk offset in the logits (15,012 parameters)**
- Mechanism: `logits' = logits + δ_g`, learned on the training pool with the frozen backbone.
- Addresses: findings 5 and 9 (systematic single-cell-vs-bulk differences in abundance).
- Expected metrics: a direct recon gain. The output logit bias is near 0 and correlates 0.51 with single-cell abundance but only 0.37 with bulk, so there is room.
- CPU cost: trivial; train on cached embeddings.
- Risks: absorbs biology of the training pool's tissue mix.

**D4. Low-rank residual logits head (trained)**
- Mechanism: `logits' = logits + U2·GELU(V2·e)` on the 1,600-dim embedding, U2 zero-initialized, rank r. Equivalent to LoRA-in-spirit on `output_mlp`, but parallel to it, so the frozen path stays intact.
- Addresses: finding 5.
- Expected metrics: recon and resid_r gains; visible_r gains.
- CPU cost: about 0.55M parameters at r=32. Cheap because embeddings can be cached (no backprop through the backbone).
- Risks: overfitting with ~300 training samples; use mask draws and thinned views as augmentation.

**D5. Bulk likelihood**
- Mechanism, options considered: (a) NB with mean px·lib_native and per-gene bulk dispersion θ_g (DESeq2-style; covers both thinned and native depths because variance is μ + μ²/θ); (b) Dirichlet-multinomial with a sample-level concentration; (c) heteroscedastic Gaussian on log-CPM with per-gene σ_g; (d) the Pearson-correlation loss on hidden log-CPM (aligned with recon_r).
- Recommendation: (a) + λ·(d). NB keeps STACK's likelihood family, handles count noise at low depth, and with learned θ_g models biological between-sample variance at high depth. Normalize the NB NLL per gene by the NLL of the LOO-mean prediction (a skill score) so that high-abundance genes do not dominate.
- Addresses: finding 5.
- Risks: STACK's own dispersion head was trained on single-cell data; do **not** reuse it for bulk.

**D6. Aggregation-consistency loss**
- Mechanism: for a pseudobulk of k real cells, the decoded px of the pseudobulk should equal the library-weighted mean of the decoded px of its constituent cells (or of their truth).
- Addresses: findings 7 and 10; ties the bulk and single-cell heads together.
- CPU cost: requires running the constituent cells, k up to 400, so subsample cells.
- Risks: the single-cell decoder is itself smoothed; prefer the truth target.

**D7. Temperature or sharpening of px**
- Mechanism: `px ∝ px^τ` with a scalar τ fit on the training pool. The decoder is over-smoothed toward the prior (T3).
- Expected metrics: a small visible and recon gain.
- CPU cost: zero.
- Risks: it is partially subsumed by D1's β.

### E. Training objectives and data

**E1. Masked-gene bulk reconstruction on the ARCHS4 pool**
- Mechanism: hide genes (fraction ~ U[`mask_fraction_min`, `mask_fraction_max`]), feed the harmonized input (plus the side channel), and predict the hidden genes at *native* depth with D5. Augment with thinned copies at log-uniform depths so the adapter also works at intermediate depths.
- Addresses: findings 5 and 6.
- Risks: leakage between pool and evaluation (Section 6).

**E2. Depth-consistency (multi-view) loss**
- Mechanism: embeddings of the same sample at depths d1 and d2 (two thinning draws or rescales) should match. Use InfoNCE across samples in the window so it cannot collapse.
- Addresses: finding 1 and emb_cos collapse; makes kNN depth-invariant across the sweep.
- CPU cost: two passes.
- Risks: forces invariance to genuine depth-dependent information, which is fine for identity.

**E3. Cross-modal anchoring**
- Mechanism: the embedding of a pseudobulk of real cells (at a bulk-like depth, harmonized) should be close to the mean embedding of its constituent cells, or to the group centroid.
- Addresses: findings 7 and 10; cross-modal assignment.
- Risks: leakage if the evaluation cells are used (split cells per group).

**E4. Distill the thinning ensemble**
- Mechanism: the teacher is STACK on thin K=8 at x1 (best measured reconstruction). The student is a single deterministic pass with the side channel. Then add E1 to go beyond the teacher.
- Addresses: inference cost and the noise of A2.
- Risks: inherits the teacher's ceiling unless E1 is added.

**E5. Single-cell preservation regularizer**
- Mechanism: on real single cells, KL(px_adapter ‖ px_frozen) plus L2 on embeddings. Alternatively gate every adapter by s = log(lib/D) > 0, so cells pass through untouched.
- Addresses: keeping single-cell metrics intact.

**E6. Complete held-out series for cohort-context training**
- Mechanism: the pool has one sample per series, so windows contain no batch-matched peers. Add a few complete series (disjoint from GENtoniK and the cell lines) to learn cohort context (C6, D1-learned).
- Risks: download size; series selection rules must be config-driven.

**E7. Test-time self-supervised adaptation**
- Mechanism: on an unlabeled bulk cohort, fit only D3 (δ_g) or B1 for a few masked-reconstruction steps.
- Expected metrics: cohort-specific gain.
- CPU cost: seconds.
- Risks: transductive. Report it separately and never tune on evaluation-hidden genes.

### F. Inference-time strategies (no training), in addition to A1–A6, A8, C3, C4, D1, D2 and D7

**F1. Mask-ensemble imputation**
- Mechanism: average px for a target gene over several random masks of *other* genes, which reduces variance.

**F2. Depth-matched peers**
- Mechanism: harmonize every profile in a window to the same D, so cross-cell attention compares like with like. This is automatic in A1 because min(1, D/lib) gives the same depth for all bulk samples.

**F3. Hybrid decode**
- Mechanism: keep measured genes (copy path) and replace only the hidden ones with D1 predictions. This is the natural "imputation API" for bulk.

**F4. Uncertainty from thinning draws**
- Mechanism: use the per-gene variance of px across K draws as a confidence estimate, for example to decide where to trust STACK over the LOO prior in D1. β could become per-gene: β_g ∝ the stability of STACK's deviation.

### G. Evaluation (what to add to `evaluate.py` without changing it: a new evaluator that imports it)

- **G1. Truth = native when the input is transformed.** Without this, the thinned-set recon_r (0.32) understates what the model recovers (0.85).
- **G2. resid_r.** Sample-specific signal beyond the population prior. Already prototyped.
- **G3. Deconvolution accuracy** on held-out mixtures of real cells from known groups with known proportions (cells disjoint from the training mixtures).
- **G4. DE recovery on GENtoniK.** log2FC (treatment vs DMSO, per line) computed from decoded or imputed profiles versus the published DESeq2 tables already in `data/gentonik/`. This tests quantitative resolution at the scale bulk users care about (|log2FC| of about 0.2–1).
- **G5. Bulk ICL perturbation prediction.** Leave one line out: predict line X's treated profile from its DMSO profile plus other lines' treated/DMSO pairs. Baseline: X's DMSO plus the mean shift of the other lines.
- **G6. Batch confounding.** kNN by GEO series vs by biological label on the cell-line set (series_id is in `obs`). An adapter that clusters by series has learned batch, not biology.
- **G7. Model-free controls with the same training data:** (i) ridge or low-rank linear imputation (visible genes → hidden genes) trained on the pool; (ii) kNN-peer imputation. Any trained adapter must beat these, otherwise STACK adds nothing.
- **G8. Single-cell regression check.** Rerun all cell and pseudobulk sets; deltas must be about 0.
- **G9. Uncertainty.** Bootstrap over samples (GENtoniK n=29) and over mask draws; report 95% intervals. Many differences above are ~0.01.

---

## 4. What a natively dual-modality STACK would need

1. **A depth-invariant value encoding plus explicit depth.** Normalized values (log-CPM or rank) as content, and log(depth) as a separate token or FiLM scalar. The model can then use depth as a confidence signal without it flooding the content (fixes findings 1 and 2).
2. **An explicit mask token and a missing-gene token**, distinct from a measured zero (finding 3), and a gene-identity vocabulary so panels can differ (finding 9).
3. **Mixture semantics.** A profile token carries "number of cells aggregated" (1 for a cell, k for pseudobulk, unknown or large for bulk). Training enforces **aggregation consistency** in both decoding (D6) and embedding (E3): the bulk token decodes to the library-weighted mixture of its cells' decodes, and embeds near the pooled cell embeddings.
4. **A two-level context.** Within-sample cell attention (current cross-cell attention) and across-sample attention over sample tokens (bulk, pseudobulk, or pooled cells), with cross-attention from sample tokens to a cell bank (deconvolution, C5). This replaces "window = population" with "window = cohort of populations + reference cells".
5. **A likelihood that spans both regimes.** NB with modality-specific dispersion (or a depth-dependent θ), and training targets at native depth across the whole range 1e3–1e8. Without them the decoder never learns sub-1/lib resolution.
6. **Per-sample masking with peers** in pretraining, so cross-sample attention learns to borrow values of the same gene from similar samples (what D1 does by hand).
7. **Pretraining data across aggregation levels.** Single cells, pseudobulks of real cells across k, binomial thinnings across depths, and real bulk (ARCHS4) with protocol tokens (3′ / full-length / nuclei). Use series-level splits to avoid batch memorization.
8. **Window-independent embedding geometry.** A regularizer on *uncentered* embeddings or an anchor set, so that centroids from different windows (and modalities) are comparable (finding 8).

A cheap approximation of 1–3 on top of the frozen checkpoint is exactly the shortlist below: harmonize, add a depth/modality signal and a full-depth side channel, add a bulk head, and decode relative to context.

---

## 5. Ranked shortlist to implement now (frozen backbone, CPU)

All new code goes in `bulk_stack/` (for example `bulk_stack/adapters.py` and `bulk_stack/train_adapter.py`), wrapping STACK via the existing `run_window(..., transform=...)` hook plus a new runner for heads. Proposed config keys extend the `adapter:` block in `config.yaml`. Every numeric value is either a config key or derived from data or the checkpoint.

### Rank 1: S1, depth harmonizer (training-free)

- Wraps: counts → log features, before `gene_reduction` (the `transform` argument of `run_window`). Native `lib` is still used for NB means.
- Spec: `D = depth_multiplier × median_library(reference single cells)`, `f = min(1, D/lib)`. Three modes:
  - `rescale`: `log1p(x·f)`
  - `meanfield`: `E[log1p(Poisson(x·f))]`
  - `thin`: K binomial draws, averaging px and embeddings
- Parameters: 0.
- Config:
  ```yaml
  adapter:
    harmonize:
      mode: rescale            # rescale | meanfield | thin
      reference: auto          # median library of the set's reference single-cell data
      depth_multiplier: 1      # chosen by a sweep on the training pool
      n_draws: 8               # thin only
      embed_depth_multiplier: 1  # second view for embeddings (A5); equal = single view
  ```
- Expected metrics (measured at x1): GENtoniK recon 0.27→0.84–0.86 (vs native truth), cell lines 0.41→0.75–0.78, pb400 0.68→0.85. preact_std back in the single-cell range. kNN unchanged to improved (thin K=8: 0.90 on GENtoniK).
- Ablation: identity → rescale x1 → thin K=8 → meanfield → multiplier sweep (1, 2, 5, 10) → two-view (x1 decode, x10–x100 embed).
- Pitfall: the native-depth `*/native` sets must be scored against native truth (G1), not against harmonized truth.

### Rank 2: S2, residual-to-context decoder (training-free)

- Wraps: px_scale output (post-softmax), per window.
- Spec: `p̂_i ∝ C_i ⊙ (px_i / C̃_i)^β`. C_i is the LOO peer mean of observed fractions (genes hidden in i are taken from peers). C̃_i is the LOO mean of px over the same peers. When no peers exist (single sample), C = the training-pool mean fraction and C̃ = the mean px of the training pool.
- Parameters: 1 (β), fit by grid search on the training pool (`beta_grid`).
- Config:
  ```yaml
  adapter:
    context_decoder:
      enabled: true
      beta: auto              # fit on training pool over beta_grid
      beta_grid: [0.25, 0.5, 0.75, 1.0]
      prior: window_loo       # window_loo | training_pool
  ```
- Expected metrics (measured, rescale x1): GENtoniK 0.957 vs LOO 0.953; cell lines 0.870 vs 0.824. recon_gain turns positive.
- Pitfall: this uses peers' values of the hidden genes, exactly like `baseline_r` does. State it, and also report the single-sample variant (`prior: training_pool`), where recon_gain must be computed against the training-pool prior instead.

### Rank 3: S3, bulk-aware output head (trained, cached embeddings, no backprop through the backbone)

- Wraps: the 1,600-dim embedding plus the frozen logits (pre-softmax output_mlp[...,0]). It does not modify `output_mlp`. The frozen logits are recomputed or cached.
- Spec: `logits' = logits_frozen + δ + U2·GELU(V2·e) + α ⊙ log(C_i + ε)`, then `p' = softmax(logits')` and `μ = p'·lib_native`. ε is one read divided by lib (derived). The α term is the learned version of S2, using peers when available or the training-pool prior otherwise.
  - δ ∈ R^15012 (D3), zero-init.
  - V2: 1600→r, U2: r→15012, zero-init (D4).
  - α ∈ R^15012, zero-init.
  - ψ ∈ R^15012 with `θ_g = softplus(ψ_g)`, initialized from method-of-moments dispersions on the training pool.
- Parameter count at r=32: δ 15,012 + V2 (51,200 + 32) + U2 (480,384 + 15,012) + α 15,012 + ψ 15,012 ≈ **591,664**.
- Loss: on hidden genes with truth CPM ≥ `min_cpm_for_metric`, `L = mean_g[NB_NLL(y_g | μ_g, θ_g) / NB_NLL(y_g | μ_g^LOO, θ_g)] + λ_corr · (1 − r_logCPM(hidden))`, plus `λ_l2·‖U2‖²`.
- Training data: ARCHS4 pool, split into `train` and `val` (each sample is its own series, so a sample split is a series split). Native samples plus `n_depth_views` thinned copies at log-uniform depth in [`min_depth`, `max_depth`]. For thinned views the target is the thinned counts, at their own lib. Mask fraction ~ U[`mask_fraction_min`, `mask_fraction_max`], with K mask draws per sample cached. Windows of `window_size` pool samples.
- Embeddings are cached once: about 384 × (1 + n_depth_views) × K_mask passes of 32-profile windows. At about 0.25 s per pass this takes a few minutes. Head training then takes seconds per hundred steps.
- Config:
  ```yaml
  adapter:
    head:
      rank: 32
      use_context_term: true
      likelihood: nb_bulk      # nb_bulk | dirichlet_multinomial | logcpm_gaussian
      lambda_corr: 1.0
      lambda_l2: 1.0e-4
      n_mask_draws: 4
      n_depth_views: 2
      val_fraction: 0.2
    train:                     # existing keys reused: steps, window_size, learning_rate,
      ...                      # weight_decay, mask_fraction_min/max, min_depth, max_depth, eval_every
  ```
  Change `min_depth` from its current 5e5 to a derived value (`reference median × min multiplier`) so training covers the 1e4–1e5 range the benchmark sweeps.
- Expected metrics: recon_r ≥ the S2 numbers; resid_r up; visible_r up (δ and D4 sharpen the over-smoothed decoder). No change to embeddings or kNN.
- Ablation: δ only → δ + D4 → + α (context) → likelihood variants → compare with G7 controls.

### Rank 4: S4, full-depth side channel + depth FiLM input adapter (trained, backprop through the frozen backbone)

- Wraps: the tokens after `gene_reduction(harmonized log features)`, before `_run_attention_layers`. It does not modify the backbone; it reimplements the 5-line encode path as `stack_ops.encode` already does.
- Spec:
  - `z_g = (log1p(CPM_g) − μ_g)/σ_g`, where μ_g and σ_g are training-pool statistics (σ floored at the pool's median σ, derived).
  - `Δ = reshape(U·GELU(V·z_masked), (100,16))`. Hidden genes are zeroed in z and a binary mask-indicator vector is concatenated: V takes 2×15012 inputs.
  - Depth FiLM on pre-activations: `pre' = pre ⊙ (1 + a·s) + b·s` with `s = log(lib_native/D)` and `a, b ∈ R^1600`.
  - `tokens' = GELU(pre') reshaped + g(s)·Δ`, where `g(s) = sigmoid(k·(s − s0))` with k and s0 learned (it starts closed for cells, s ≈ 0) to keep single-cell inputs unchanged.
  - Optional B2 modality embedding of 1,600 parameters.
- Parameter count at r=32: V 30,024×32 + 32 = 960,800; U 32×1,600 + 1,600 = 52,800; FiLM 3,200; gate 2; modality 1,600; total ≈ **1.02M**. Without the mask-indicator half of V: ≈ 0.54M.
- Loss: the S3 loss (train S3 and S4 jointly, or S4 with S3 frozen), plus E5 single-cell preservation (KL of px and embedding L2 on held-out reference cells not used in evaluation sets), plus optional E2 depth consistency (InfoNCE, τ as config).
- Hyperparameters: `steps` 400–1000, `window_size` 32, `learning_rate` 3e-3 (adapter) / 1e-3 (head), AdamW, gradient clipping 1.0, `eval_every` 100. About 0.6 s per step, so 400 steps take roughly 5 minutes on 4 cores.
- Config: `adapter.side_channel.{rank, use_mask_indicator, gate, film, modality_embedding, lambda_sc_preserve, lambda_depth_consistency, temperature}`.
- Expected metrics: the first configuration that can raise *STACK's own* recon_r above LOO without D1; kNN toward knn_pca; cross-modal unchanged or improved (gated).
- Risks: shortcut learning (V→U→tokens→head could act as a linear imputer that bypasses STACK), so G7 is mandatory; overfitting with 307 training samples (use thinned views and mask draws as augmentation; early stopping on `val`).

### Rank 5: S5, aggregate-matched cross-modal readout (training-free; optional small trained deconvolution)

- Wraps: the embeddings from S1.
- Spec: build reference centroids from `pb{k}_thinned` profiles of each group (k = `cross_modal.reference_aggregation`, default the largest aggregation level), harmonized to the same D as the bulk. Assign by cosine. Optional C5: softmax weights over group centroids, trained on held-out real-cell mixtures (about 38k parameters for 24 groups as a linear map from the 1,600-dim embedding to group logits; under 0.1M as a small MLP).
- Config: `adapter.cross_modal.{reference: pb_thinned, reference_aggregation: max, deconvolution: false}`.
- Expected metrics: GENtoniK assignments concentrate (target: agreement with correlation's 5-month CPN, or a defensible neuronal mixture); cell lines ≥0.96.
- Pitfall: the reference pseudobulks must be built from cells that do not appear in training mixtures.

### Rank 6 (stretch): S6, cross-sample attention LoRA with per-sample masking

- Wraps: `layers[i].gene_attn.qkv` and `.proj` via forward hooks / parametrization (no source edit).
- Parameters at r=4: 9 × [4×(1600+4800) + 4×(1600+1600)] ≈ **345,600**.
- Data: needs batch-matched cohorts (E6); otherwise it learns to copy from unrelated samples.
- Expected metrics: recon above LOO without the hand-made D1.
- Only attempt this after S1–S4 are in place.

### Ablation order (one change at a time, same seeds, same windows)

| step | configuration |
|---|---|
| 0 | baseline (identity) |
| 1 | S1 rescale x1; then thin K=8; then meanfield; then multiplier sweep chosen on the pool (not on evaluation sets) |
| 2 | S1 + S2 (β fit on pool) |
| 3 | S1 + S3 with δ only |
| 4 | S1 + S3 with δ + low-rank |
| 5 | S1 + S3 full (+α) |
| 6 | S1 + S3 + S4 (side channel; then + FiLM; then + mask indicator; then + modality embedding) |
| 7 | each of steps 2–6, ± S2 on top (check additivity) |
| 8 | S5 cross-modal on the S1 and S4 embeddings |
| 9 | G7 model-free controls trained on identical data; G8 single-cell regression; G4/G5 bulk tasks; G6 series confounding; G9 bootstrap CIs |

---

## 6. Pitfalls

- **Thinning loses gene-level information.** At x1, only 3.3% (GENtoniK) and 0.8% (cell lines) of expressed genes have ≥1 expected read, and K=8 draws use under 0.1% of the reads. Ensembles reduce *sampling* variance but cannot recover genes whose expected count is ≪1. Mid- and low-abundance genes and small fold changes are invisible to the backbone. Only a side channel (S4) or a decoder that reads the native profile (S2/S3 context term) can use them.
- **Deterministic rescaling yields non-integer, non-sparse inputs.** At x1 this was harmless (rescale ≈ thin). At higher multipliers, the many small-but-nonzero entries are unlike single-cell zeros. Also, the meanfield/rescale identity holds only at the first Linear (it is linear), not after the GELU.
- **Truth must be native.** Scoring harmonized inputs against harmonized truth conflates input noise with model error (0.32 vs 0.85 on the same GENtoniK input). Scoring against native truth also means `baseline_r` (0.95) is the bar, and in homogeneous cohorts it is very hard to beat; report resid_r and the cell-line set, where heterogeneity gives room.
- **The context baseline uses peers' hidden genes.** Masks are shared across the window, but the LOO baseline reads peers' truth. Any method using C_i must be compared with that baseline, never with STACK-only numbers. Also report single-sample variants.
- **Leakage between the training pool and evaluation series.** Specifically:
  - ARCHS4 has SuperSeries/SubSeries duplicates: the same GSM appears under multiple GSE accessions. De-duplicate by GSM *and* by an expression hash.
  - Exclude the evaluation series, all series sharing a submitter/contact with them, and any sample whose characteristics mention MCF7/A549 or cortical neurons from hPSC (or document the overlap).
  - Fit μ_g, σ_g, θ_g, δ_g and β on the pool only.
  - Never choose `depth_multiplier` on evaluation sets.
- **Leakage between pseudobulk training and evaluation sets.** `profiles.py` builds evaluation pseudobulks from the same single-cell files that adapter training would use. Split cells per group (for example a hash of the barcode) and rebuild evaluation sets from the held-out half before training on real-cell mixtures.
- **Small-n noise.** GENtoniK has n=29; differences of ±0.01–0.03 in recon_r arise from mask draws alone (compare `baseline_sets.csv` with Section 1 identity rows). Use G9.
- **emb_cos is not comparable across K.** Averaging embeddings over draws raises cosine (0.55→0.79) without meaning collapse. Use emb_pc1 and kNN alongside.
- **The shortcut risk of trained adapters.** A side channel plus a head can solve imputation as a linear model of the pool without using STACK. The G7 controls are the guard. Also ablate by shuffling the STACK embedding (keeping the adapter) to measure STACK's contribution.
- **Breaking single-cell behaviour.** Any adapter not gated by depth will change cell embeddings; check G8 after every trained step.
- **Batch rather than biology.** The ARCHS4 pool is maximally batch-diverse. An adapter can learn series signatures; check G6 on the cell lines, where `series_id` varies.
- **Protocol offsets absorb biology.** δ_g learned on a tissue-diverse pool mixes protocol effects with average tissue composition. Keep it small (L2) and test cross-modal with and without it.

---

## Appendix: how the Section 1 numbers were produced

The scratch scripts, kept outside the repository, load the checkpoint memory-mapped and read `data/profiles.h5ad`, `data/baseline_embeddings.npz` and `profiles.uns['reference_depth']`. They reuse `stack_ops.make_windows / masked_log_cpm / row_pearson / loo_mean` and `evaluate.knn_accuracy / mean_pairwise_cosine`.

- **Forward pass:** `gene_reduction[0] → GELU → reshape → _run_attention_layers → _compute_nb_parameters` (identical to `stack_ops.run_window`). Hidden genes are zeroed in log space after the transform.
- **Windows:** 64 (cohorts smaller than 64 use one window); 2 mask draws (3 for the context test); 20% of genes hidden; expressed = window mean CPM ≥ 1.
- **Data:** GENtoniK all 29; cell lines all 96; organoid pb400 a random 128.
- **Cross-modal centroids:** from baseline single-cell embeddings (single-cell windows, untransformed).
- **Mixed windows:** 32 bulk + n randomly drawn reference cells per window; bulk embeddings read from the bulk positions.
- **Weight analysis:** `W = gene_reduction[0].weight`. The shift part is `(log(lib/D)·1[x>0]) @ Wᵀ`; the content part is `log1p(x·D/lib) @ Wᵀ`; D = organoid median library.
