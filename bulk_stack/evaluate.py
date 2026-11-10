"""Benchmark one STACK variant on every profile set.

Metrics per set:
  recon_r            Pearson r of log-CPM over hidden genes, model prediction vs truth
  baseline_r         same for the leave-one-out mean of the other profiles in the window
  recon_gain         recon_r - baseline_r (what the model adds over the population average)
  visible_r          Pearson r of the decoded profile vs the input when nothing is hidden
  knn_stack          leave-one-out kNN label accuracy in STACK embedding space
  knn_pca            the same in a PCA of log-CPM (a model-free reference)
  emb_cosine         mean pairwise cosine similarity of embeddings (collapse indicator)
  emb_pc1            variance fraction on the first principal component of embeddings
  preact_mean/std    statistics of the first Linear layer output (input-shift indicator)
  gelu_active        fraction of first-layer units that are positive
Cross-modal: bulk samples are assigned to the nearest reference group centroid
in STACK space and, as a reference, by Spearman correlation to group pseudobulks.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut

from stack_ops import loo_mean, make_windows, masked_log_cpm, row_pearson, run_window

Runner = Callable[[np.ndarray, np.ndarray | None], "object"]


def knn_accuracy(features: np.ndarray, labels: np.ndarray, k: int, metric: str) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    counts = pd.Series(labels).value_counts()
    if counts.min() < 2:
        keep = np.isin(labels, counts[counts >= 2].index)
        features, labels = features[keep], labels[keep]
    k = min(k, len(labels) - 1)
    clf = KNeighborsClassifier(n_neighbors=k, metric=metric)
    cv = LeaveOneOut() if len(labels) <= 200 else 10
    predicted = cross_val_predict(clf, features, labels, cv=cv)
    return float((predicted == labels).mean())


def pca_features(log_cpm: np.ndarray, n: int = 50) -> np.ndarray:
    n = max(1, min(n, log_cpm.shape[0] - 1, log_cpm.shape[1]))
    return PCA(n_components=n, random_state=0).fit_transform(log_cpm)


def mean_pairwise_cosine(emb: np.ndarray) -> float:
    unit = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    sims = unit @ unit.T
    n = len(unit)
    return float((sims.sum() - n) / max(n * (n - 1), 1))


def pc1_fraction(emb: np.ndarray) -> float:
    if emb.shape[0] < 3:
        return float("nan")
    pca = PCA(n_components=min(10, emb.shape[0] - 1)).fit(emb)
    return float(pca.explained_variance_ratio_[0])


def bootstrap_ci(values: np.ndarray, n_boot: int, level: float, generator: np.random.Generator) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan"), float("nan")
    means = generator.choice(values, (n_boot, values.size), replace=True).mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def run_set(runner: Runner, counts: np.ndarray, bench: dict, generator: np.random.Generator) -> dict:
    n, n_genes = counts.shape
    windows = make_windows(n, int(bench["window_size"]), generator)
    emb = np.zeros((n, 0), dtype=np.float32)
    stats = {k: np.zeros(n) for k in ("preact_mean", "preact_std", "gelu_active", "visible_r")}
    recon, base, resid = [], [], []
    seen = np.zeros(n, dtype=bool)
    cpm_floor = float(bench["min_cpm_for_metric"])
    for window in windows:
        out = runner(counts[window], None)
        if emb.shape[1] == 0:
            emb = np.zeros((n, out.embeddings.shape[1]), dtype=np.float32)
        fresh = ~seen[window]
        idx = window[fresh]
        emb[idx] = out.embeddings[fresh]
        stats["preact_mean"][idx] = out.preact_mean[fresh]
        stats["preact_std"][idx] = out.preact_std[fresh]
        stats["gelu_active"][idx] = out.gelu_active[fresh]
        truth = counts[window]
        all_genes = np.arange(n_genes)
        expressed = (1e6 * truth / np.maximum(truth.sum(1, keepdims=True), 1)).mean(0) >= cpm_floor
        stats["visible_r"][idx] = row_pearson(
            masked_log_cpm(out.px_scale, all_genes[expressed]), masked_log_cpm(truth, all_genes[expressed])
        )[fresh]
        seen[window] = True
        for _ in range(int(bench["mask_repeats"])):
            hidden = np.zeros(n_genes, dtype=bool)
            hidden[generator.choice(n_genes, int(n_genes * float(bench["mask_fraction"])), replace=False)] = True
            genes = np.flatnonzero(hidden & expressed)
            if genes.size < 10:
                continue
            masked = runner(truth, hidden)
            fractions = truth / np.maximum(truth.sum(1, keepdims=True), 1)
            target = masked_log_cpm(truth, genes)
            predicted = masked_log_cpm(masked.px_scale, genes)
            prior = masked_log_cpm(loo_mean(fractions), genes)
            recon.append(row_pearson(predicted, target)[fresh])
            base.append(row_pearson(prior, target)[fresh])
            resid.append(row_pearson(predicted - prior, target - prior)[fresh])
    if not recon:
        nan = float("nan")
        return {"embeddings": emb, "recon_r": nan, "baseline_r": nan, "recon_gain": nan, "resid_r": nan,
                "gain_ci_low": nan, "gain_ci_high": nan, **{k: float(np.nanmean(v)) for k, v in stats.items()}}
    recon_all, base_all = np.concatenate(recon), np.concatenate(base)
    low, high = bootstrap_ci(recon_all - base_all, int(bench["bootstrap"]), float(bench["ci_level"]), generator)
    return {
        "embeddings": emb,
        "recon_r": float(np.nanmean(recon_all)),
        "baseline_r": float(np.nanmean(base_all)),
        "recon_gain": float(np.nanmean(recon_all - base_all)),
        "resid_r": float(np.nanmean(np.concatenate(resid))),
        "gain_ci_low": low,
        "gain_ci_high": high,
        **{k: float(np.nanmean(v)) for k, v in stats.items()},
    }


def log_cpm(counts: np.ndarray) -> np.ndarray:
    return np.log1p(1e6 * counts / np.maximum(counts.sum(1, keepdims=True), 1))


def cross_modal(profiles: ad.AnnData, embeddings: dict, bench: dict) -> list[dict]:
    rows = []
    top = max(int(k) for k in bench["aggregation_levels"])
    suffixes = {"cells": "cells", "pb_thinned": f"pb{top}_thinned"}
    pairs = [
        (bulk_name, spec, f"{spec['reference']}/{suffixes[kind]}")
        for bulk_name, spec in bench["bulk"].items()
        for kind in bench["cross_modal_references"]
    ]
    for bulk_name, spec, ref_set in pairs:
        ref_mask = profiles.obs["set"].eq(ref_set).to_numpy()
        ref_labels = profiles.obs.loc[ref_mask, "label"].to_numpy()
        ref_counts = profiles.X[ref_mask].toarray()
        groups = np.unique(ref_labels)
        stack_centroids = np.vstack([embeddings[ref_set][ref_labels == g].mean(0) for g in groups])
        pseudo = np.vstack([ref_counts[ref_labels == g].sum(0) for g in groups])
        pseudo_log = log_cpm(pseudo)
        for variant in ("native", "thinned"):
            set_name = f"{bulk_name}/{variant}"
            mask = profiles.obs["set"].eq(set_name).to_numpy()
            labels = profiles.obs.loc[mask, "label"].to_numpy()
            emb = embeddings[set_name]
            unit_c = stack_centroids / np.linalg.norm(stack_centroids, axis=1, keepdims=True)
            unit_e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
            stack_pick = groups[np.argmax(unit_e @ unit_c.T, axis=1)]
            bulk_log = log_cpm(profiles.X[mask].toarray())
            rho = spearmanr(bulk_log, pseudo_log, axis=1)[0][: len(bulk_log), len(bulk_log) :]
            corr_pick = groups[np.argmax(rho, axis=1)]
            row = {"bulk_set": set_name, "reference": ref_set, "n": int(mask.sum())}
            shared = set(labels) & set(groups)
            if shared:
                row["stack_accuracy"] = float(np.mean(stack_pick == labels))
                row["correlation_accuracy"] = float(np.mean(corr_pick == labels))
            row["stack_assignment"] = pd.Series(stack_pick).value_counts().to_dict()
            row["correlation_assignment"] = pd.Series(corr_pick).value_counts().to_dict()
            rows.append(row)
    return rows


def evaluate(runner: Runner, profiles: ad.AnnData, cfg: dict, generator: np.random.Generator, sets: list[str] | None = None):
    bench = cfg["benchmark"]
    k = int(bench["knn_k"])
    rows, embeddings = [], {}
    for set_name in sets or list(dict.fromkeys(profiles.obs["set"])):
        mask = profiles.obs["set"].eq(set_name).to_numpy()
        obs = profiles.obs[mask]
        counts = profiles.X[mask].toarray() if sparse.issparse(profiles.X) else np.asarray(profiles.X[mask])
        result = run_set(runner, counts, bench, generator)
        emb = result.pop("embeddings")
        embeddings[set_name] = emb
        labels = obs["label"].to_numpy()
        row = {
            "set": set_name,
            "source": obs["source"].iloc[0],
            "kind": obs["kind"].iloc[0],
            "level": obs["level"].iloc[0],
            "n": int(mask.sum()),
            "n_labels": int(len(np.unique(labels))),
            "median_library": float(np.median(obs["library_size"])),
            "zero_fraction": float((counts == 0).mean()),
            "mean_log1p": float(np.log1p(counts).mean()),
            **result,
            "knn_stack": knn_accuracy(emb, labels, k, "cosine"),
            "knn_pca": knn_accuracy(pca_features(log_cpm(counts)), labels, k, "euclidean"),
            "emb_cosine": mean_pairwise_cosine(emb),
            "emb_pc1": pc1_fraction(emb),
        }
        if "series_id" in obs and obs["series_id"].notna().all():
            series = obs["series_id"].astype(str).str.split(",").str[0].to_numpy()
            row["knn_series_stack"] = knn_accuracy(emb, series, k, "cosine")
            row["knn_series_pca"] = knn_accuracy(pca_features(log_cpm(counts)), series, k, "euclidean")
        print(json.dumps({key: (round(v, 4) if isinstance(v, float) else v) for key, v in row.items()}), flush=True)
        rows.append(row)
    return pd.DataFrame(rows), embeddings
