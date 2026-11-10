"""Figures from results/<tag>_sets.csv (and the baseline embeddings)."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from common import load_config, path

METRICS = [
    ("recon_gain", "Masked-gene gain over\npopulation mean (Δr)"),
    ("visible_r", "Decoded vs input\n(r, nothing hidden)"),
    ("knn_stack", "kNN label accuracy\n(STACK embedding)"),
    ("emb_cosine", "Mean pairwise cosine\nof embeddings"),
    ("preact_mean", "First-layer\npre-activation mean"),
    ("gelu_active", "First-layer units\nactive (>0)"),
]


def set_order(table: pd.DataFrame, source: str) -> list[str]:
    rows = table[table["source"].eq(source) & ~table["kind"].eq("depth_sweep")]
    return rows["set"].tolist()


def short(name: str) -> str:
    return name.split("/", 1)[1]


def draw_dataset_types(cfg: dict, tables: dict[str, pd.DataFrame], out: Path) -> Path:
    base = tables["baseline"]
    sources = list(dict.fromkeys(base["source"]))
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(15, 2.3 * len(METRICS)), sharex=True)
    names = [s for src in sources for s in set_order(base, src)]
    x = np.arange(len(names))
    width = 0.8 / len(tables)
    for ax, (metric, label) in zip(axes, METRICS):
        for i, (tag, table) in enumerate(tables.items()):
            values = table.set_index("set").reindex(names)[metric].to_numpy()
            ax.bar(x + (i - (len(tables) - 1) / 2) * width, values, width, label=tag)
        ax.set_ylabel(label, fontsize=8)
        ax.axhline(0, color="black", linewidth=0.5)
        boundaries = np.cumsum([len(set_order(base, s)) for s in sources])[:-1] - 0.5
        for b in boundaries:
            ax.axvline(b, color="grey", linestyle=":")
    axes[0].legend(fontsize=8, ncol=len(tables))
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    fig.suptitle("STACK across dataset types: single cells, pseudobulk of k cells, depth-matched pseudobulk, real bulk")
    fig.tight_layout()
    target = out / "dataset_types.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def draw_depth_sweep(cfg: dict, tables: dict[str, pd.DataFrame], out: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (metric, label) in zip(axes.ravel(), METRICS):
        for tag, table in tables.items():
            sweep = table[table["kind"].eq("depth_sweep")]
            for source, block in sweep.groupby("source", sort=False):
                ax.plot(block["median_library"], block[metric], marker="o", label=f"{source} · {tag}")
            for source, reference in cfg["benchmark"]["bulk"].items():
                cells = table[table["set"].eq(f"{reference['reference']}/cells")]
                if not cells.empty and tag == "baseline":
                    ax.axhline(float(cells[metric].iloc[0]), linestyle=":", color="grey", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("reads per sample (thinned real bulk)")
        ax.set_title(label, fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Same bulk samples thinned to single-cell depth and back up (dotted: single-cell reference value)")
    fig.tight_layout()
    target = out / "depth_sweep.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def draw_embedding_map(cfg: dict, tags: list[str], out: Path) -> Path | None:
    data = path(cfg, "data")
    panels = []
    for tag in tags:
        file = data / f"{tag}_embeddings.npz"
        if file.exists():
            panels.append((tag, np.load(file)))
    if not panels:
        return None
    import anndata as ad

    obs = ad.read_h5ad(data / "profiles.h5ad", backed="r").obs
    pairs = list(cfg["benchmark"]["bulk"].items())
    fig, axes = plt.subplots(len(pairs), len(panels), figsize=(6 * len(panels), 5 * len(pairs)), squeeze=False)
    for row, (bulk_name, spec) in enumerate(pairs):
        sets = [f"{spec['reference']}/cells", f"{bulk_name}/native", f"{bulk_name}/thinned"]
        for col, (tag, store) in enumerate(panels):
            ax = axes[row, col]
            blocks = [store[s.replace("/", "__")] for s in sets]
            coords = PCA(n_components=2, random_state=0).fit_transform(np.vstack(blocks))
            start = 0
            for s, block, marker in zip(sets, blocks, ["o", "^", "s"]):
                labels = obs.loc[obs["set"].eq(s), "label"].to_numpy()
                part = coords[start : start + len(block)]
                start += len(block)
                if s.endswith("/cells") and len(np.unique(labels)) > 8:
                    labels = np.array([lab.split("|")[0] for lab in labels])
                for lab in np.unique(labels):
                    pick = labels == lab
                    ax.scatter(part[pick, 0], part[pick, 1], s=8 if marker == "o" else 40, marker=marker, alpha=0.6, label=f"{short(s)}:{lab}")
            ax.set_title(f"{tag}: {spec['reference']} cells (o), {bulk_name} native (^), thinned (s)", fontsize=8)
            ax.legend(fontsize=5, markerscale=0.8, ncol=2)
    fig.tight_layout()
    target = out / "embedding_map.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def draw_variant_summary(cfg: dict, tables: dict[str, pd.DataFrame], out: Path) -> Path:
    """Hidden-gene r per variant on the deep sets, against the population-mean baseline."""
    top = max(int(k) for k in cfg["benchmark"]["aggregation_levels"])
    sets = [f"{name}/native" for name in cfg["benchmark"]["bulk"]]
    sets += [f"{name}/pb{top}" for name in cfg["benchmark"]["singlecell"]]
    sets += [f"{name}/cells" for name in cfg["benchmark"]["singlecell"]]
    fig, axes = plt.subplots(1, len(sets), figsize=(3.2 * len(sets), 4.2), sharey=False)
    tags = list(tables)
    for ax, set_name in zip(axes, sets):
        values = [float(tables[t].set_index("set").loc[set_name, "recon_r"]) for t in tags]
        low = [float(tables[t].set_index("set").loc[set_name, "gain_ci_low"]) for t in tags]
        high = [float(tables[t].set_index("set").loc[set_name, "gain_ci_high"]) for t in tags]
        base = float(tables[tags[0]].set_index("set").loc[set_name, "baseline_r"])
        gain = np.array(values) - base
        yerr = np.vstack([gain - np.array(low), np.array(high) - gain]).clip(min=0)
        ax.bar(np.arange(len(tags)), values, color=["#999999"] + ["#4C72B0"] * (len(tags) - 1), yerr=yerr, capsize=2)
        ax.axhline(base, color="#C44E52", linestyle="--", linewidth=1, label="population mean (LOO)")
        ax.set_xticks(np.arange(len(tags)))
        ax.set_xticklabels(tags, rotation=70, ha="right", fontsize=7)
        ax.set_title(set_name, fontsize=8)
        floor = min(min(values), base)
        ax.set_ylim(max(0.0, floor - 0.1), 1.0)
    axes[0].set_ylabel("hidden-gene r (log-CPM)")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Masked-gene reconstruction by variant (error bars: 95% bootstrap CI of the gain over the population mean)")
    fig.tight_layout()
    target = out / "variant_summary.png"
    fig.savefig(target, dpi=130)
    plt.close(fig)
    return target


def draw_all(cfg: dict, tags: list[str]) -> list[Path]:
    results = path(cfg, "results")
    out = path(cfg, "figures")
    tables = {tag: pd.read_csv(results / f"{tag}_sets.csv") for tag in tags if (results / f"{tag}_sets.csv").exists()}
    figures = [draw_dataset_types(cfg, tables, out), draw_depth_sweep(cfg, tables, out)]
    if len(tables) > 1:
        figures.append(draw_variant_summary(cfg, tables, out))
    embedding = draw_embedding_map(cfg, list(tables)[: int(cfg["figures"]["embedding_panels"])], out)
    if embedding is not None:
        figures.append(embedding)
    return figures


if __name__ == "__main__":
    cfg = load_config()
    for figure in draw_all(cfg, sys.argv[1:] or cfg["figures"]["variants"]):
        print(figure)
