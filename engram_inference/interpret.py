"""Compare STACK generations with the measured query cells and save figures."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from sklearn.decomposition import PCA

from prepare_sciplex import load_symbols, model_columns, read_counts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
ARTIFACTS = Path("/opt/cursor/artifacts")
GENES = [
    "SOX2", "PAX6", "VIM", "HES1", "EOMES", "NEUROG2", "NHLH1", "DCX", "TUBB3",
    "KRT19", "KRT8", "GATA3", "EGFR", "VEGFA", "KDM6B", "KDM1A",
]


def dense(adata: ad.AnnData) -> np.ndarray:
    matrix = adata.X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def logmean(matrix: np.ndarray) -> np.ndarray:
    return np.log1p(matrix).mean(axis=0)


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(stats.pearsonr(left, right).statistic)


def matched_cell_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    scores = []
    for row_observed, row_predicted in zip(np.log1p(observed), np.log1p(predicted), strict=True):
        scores.append(pearson(row_observed, row_predicted))
    return float(np.median(scores))


def gene_index(names: pd.Index) -> dict[str, int]:
    return {str(name).upper(): i for i, name in enumerate(names)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    organoid_test = ad.read_h5ad(ROOT / "data/cortical_organoid/stack_inputs/test.h5ad")
    organoid_base = ad.read_h5ad(ROOT / "data/cortical_organoid/stack_inputs/base.h5ad")
    pred_23 = ad.read_h5ad(ROOT / "outputs/cortical_organoid/23days.h5ad")
    pred_5 = ad.read_h5ad(ROOT / "outputs/cortical_organoid/5month.h5ad")
    if not organoid_test.obs_names.equals(pred_23.obs_names):
        raise RuntimeError("Organoid prediction cells are not the query cells")

    obs_ip = dense(organoid_test)
    p23 = dense(pred_23)
    p5 = dense(pred_5)
    base = dense(organoid_base)
    arg23 = base[organoid_base.obs["age"].astype(str).eq("23days").to_numpy()]
    arg5 = base[organoid_base.obs["age"].astype(str).eq("5month").to_numpy()]
    means = {
        "ip": logmean(obs_ip),
        "pred23": logmean(p23),
        "pred5": logmean(p5),
        "arg23": logmean(arg23),
        "arg5": logmean(arg5),
    }
    organoid = {
        "n_query": int(obs_ip.shape[0]),
        "n_context_per_age": int(arg23.shape[0]),
        "library_observed_median": float(np.median(obs_ip.sum(1))),
        "library_pred23_median": float(np.median(p23.sum(1))),
        "library_pred5_median": float(np.median(p5.sum(1))),
        "profile_corr_pred23_vs_ip": pearson(means["pred23"], means["ip"]),
        "profile_corr_pred5_vs_ip": pearson(means["pred5"], means["ip"]),
        "profile_corr_pred23_vs_arg23": pearson(means["pred23"], means["arg23"]),
        "profile_corr_pred5_vs_arg5": pearson(means["pred5"], means["arg5"]),
        "profile_corr_pred23_vs_pred5": pearson(means["pred23"], means["pred5"]),
        "profile_corr_ip_vs_arg5": pearson(means["ip"], means["arg5"]),
        "profile_corr_arg23_vs_arg5": pearson(means["arg23"], means["arg5"]),
        "cell_median_corr_pred23": matched_cell_correlation(obs_ip, p23),
        "cell_median_corr_pred5": matched_cell_correlation(obs_ip, p5),
        "logit_median_23": float(pred_23.obs["gen_logit"].median()),
        "logit_median_5": float(pred_5.obs["gen_logit"].median()),
        "logit_frac_negative_23": float((pred_23.obs["gen_logit"] < 0).mean()),
        "logit_frac_negative_5": float((pred_5.obs["gen_logit"] < 0).mean()),
    }
    names = gene_index(organoid_test.var_names)
    markers = {}
    for gene in ["SOX2", "PAX6", "VIM", "HES1", "EOMES", "NEUROG2", "NHLH1", "DCX", "TUBB3"]:
        if gene not in names:
            continue
        column = names[gene]
        markers[gene] = {
            "ip": float(obs_ip[:, column].mean()),
            "pred23": float(p23[:, column].mean()),
            "pred5": float(p5[:, column].mean()),
            "arg23": float(arg23[:, column].mean()),
            "arg5": float(arg5[:, column].mean()),
        }
    organoid["markers_mean_counts"] = markers

    # sci-Plex reference profiles for the same dose and exposure, outside the query window.
    screen = ad.read_h5ad(ROOT / "data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad", backed="r")
    symbols = load_symbols(
        ROOT / "data/sciplex/GSM4150378_sciPlex3_A549_MCF7_K562_screen_gene.annotations.txt.gz",
        screen.var["ensembl_id"],
    )
    model_genes = list(organoid_test.var_names.astype(str))
    columns = model_columns(symbols, model_genes)
    positions = pd.Series(np.arange(screen.n_obs), index=screen.obs_names)
    obs = screen.obs
    compounds = ["Ki8751", "Tranylcypromine (2-PCPA) HCl", "GSK J1"]
    rng = np.random.default_rng(0)
    reference = {}
    for label, mask in {
        "mcf7_vehicle": obs["cell_line"].eq("MCF7") & obs["time"].eq(24) & obs["perturbation"].eq("control"),
        "a549_vehicle": obs["cell_line"].eq("A549") & obs["time"].eq(24) & obs["perturbation"].eq("control"),
        **{
            f"mcf7_{name}": obs["cell_line"].eq("MCF7") & obs["time"].eq(24) & obs["dose_value"].eq(10) & obs["perturbation"].eq(name)
            for name in compounds
        },
        **{
            f"a549_{name}": obs["cell_line"].eq("A549") & obs["time"].eq(24) & obs["dose_value"].eq(10) & obs["perturbation"].eq(name)
            for name in compounds
        },
    }.items():
        index = obs.index[mask.to_numpy()]
        if index.size > 200:
            index = rng.choice(index.to_numpy(), size=200, replace=False)
        rows = positions.loc[index].to_numpy()
        matrix = read_counts(
            ROOT / "data/sciplex/SrivatsanTrapnell2020_sciplex3.h5ad",
            rows,
            columns,
            screen.n_vars,
        ).toarray()
        reference[label] = logmean(matrix)
        print(label, "cells", len(index), flush=True)

    sp_test = ad.read_h5ad(ROOT / "data/sciplex/stack_inputs/test.h5ad")
    predictions = {
        "control": dense(ad.read_h5ad(ROOT / "outputs/sciplex/control.h5ad")),
        "Ki8751": dense(ad.read_h5ad(ROOT / "outputs/sciplex/Ki8751.h5ad")),
        "Tranylcypromine (2-PCPA) HCl": dense(ad.read_h5ad(ROOT / "outputs/sciplex/Tranylcypromine_2-PCPA_HCl.h5ad")),
        "GSK J1": dense(ad.read_h5ad(ROOT / "outputs/sciplex/GSK_J1.h5ad")),
    }
    observed = dense(sp_test)
    pred_means = {name: logmean(matrix) for name, matrix in predictions.items()}
    sciplex = {
        "n_query": int(observed.shape[0]),
        "library_observed_median": float(np.median(observed.sum(1))),
        "profile_corr_vs_mcf7_vehicle": {
            name: pearson(mean, reference["mcf7_vehicle"]) for name, mean in pred_means.items()
        },
        "profile_corr_vs_a549_same_context": {
            "control": pearson(pred_means["control"], reference["a549_vehicle"]),
            **{name: pearson(pred_means[name], reference[f"a549_{name}"]) for name in compounds},
        },
        "profile_corr_vs_query_window": {name: pearson(mean, logmean(observed)) for name, mean in pred_means.items()},
        "cell_median_corr_vs_query": {name: matched_cell_correlation(observed, matrix) for name, matrix in predictions.items()},
        "delta_corr_vs_real_mcf7": {},
        "delta_corr_vs_real_a549": {},
    }
    expressed = reference["mcf7_vehicle"] > np.log1p(0.05)
    for name in compounds:
        pred_delta = pred_means[name] - pred_means["control"]
        real_mcf7 = reference[f"mcf7_{name}"] - reference["mcf7_vehicle"]
        real_a549 = reference[f"a549_{name}"] - reference["a549_vehicle"]
        sciplex["delta_corr_vs_real_mcf7"][name] = pearson(pred_delta[expressed], real_mcf7[expressed])
        sciplex["delta_corr_vs_real_a549"][name] = pearson(pred_delta[expressed], real_a549[expressed])
        sciplex[f"real_mcf7_vs_a549_delta_{name}"] = pearson(real_mcf7[expressed], real_a549[expressed])

    summary = {"organoid": organoid, "sciplex": sciplex}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    for ax, key, title in zip(axes, ["pred23", "pred5"], ["Context: 23-day aRG", "Context: 5-month aRG"], strict=True):
        ax.scatter(means["ip"], means[key], s=4, alpha=0.25, c="#3d5a80", linewidths=0)
        limit = max(means["ip"].max(), means[key].max()) * 1.02
        ax.plot([0, limit], [0, limit], color="#888888", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Measured IP, mean log1p")
        ax.set_ylabel("Generated IP, mean log1p")
        ax.text(0.04, 0.96, f"r = {organoid['profile_corr_pred23_vs_ip' if key=='pred23' else 'profile_corr_pred5_vs_ip']:.2f}", transform=ax.transAxes, va="top")
    fig.suptitle("Cortical organoid: generated intermediate progenitors", y=1.02)
    fig.tight_layout()
    for folder in (OUT, ARTIFACTS):
        fig.savefig(folder / "organoid_profile.png", dpi=140, bbox_inches="tight")
    plt.close()

    # Shared PCA of the organoid cells that entered the run.
    blocks = [np.log1p(arg23), np.log1p(arg5), np.log1p(obs_ip), np.log1p(p23), np.log1p(p5)]
    labels = (
        ["aRG, 23 days"] * arg23.shape[0]
        + ["aRG, 5 months"] * arg5.shape[0]
        + ["IP, measured"] * obs_ip.shape[0]
        + ["IP, from 23-day aRG"] * p23.shape[0]
        + ["IP, from 5-month aRG"] * p5.shape[0]
    )
    stacked = np.vstack(blocks)
    variance = stacked.var(axis=0)
    keep = np.argsort(variance)[-2000:]
    pcs = PCA(n_components=2, random_state=0).fit_transform(stacked[:, keep])
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    colors = {
        "aRG, 23 days": "#8d99ae",
        "aRG, 5 months": "#2b2d42",
        "IP, measured": "#ee6c4d",
        "IP, from 23-day aRG": "#3d5a80",
        "IP, from 5-month aRG": "#98c1d9",
    }
    for label, color in colors.items():
        mask = np.array(labels) == label
        ax.scatter(pcs[mask, 0], pcs[mask, 1], s=12, alpha=0.75, c=color, label=label, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Organoid cells and generated IP profiles")
    ax.legend(frameon=False, markerscale=1.6)
    fig.tight_layout()
    for folder in (OUT, ARTIFACTS):
        fig.savefig(folder / "organoid_pca.png", dpi=140, bbox_inches="tight")
    plt.close()

    short = {"Ki8751": "Ki8751", "Tranylcypromine (2-PCPA) HCl": "Tranylcypromine", "GSK J1": "GSK J1"}
    observed_mean = logmean(observed)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4), sharex=True, sharey=True)
    for ax, name, title in zip(
        axes,
        ["control", "Ki8751"],
        ["A549 vehicle", "A549 Ki8751, 10 nM"],
        strict=True,
    ):
        predicted_mean = pred_means[name]
        score = pearson(observed_mean, predicted_mean)
        ax.scatter(observed_mean, predicted_mean, s=4, alpha=0.25, c="#3d5a80", linewidths=0)
        limit = max(float(observed_mean.max()), float(predicted_mean.max())) * 1.05
        ax.plot([0, limit], [0, limit], color="#888888", lw=1)
        ax.set_title(title)
        ax.set_xlabel("Measured untreated MCF7, mean log1p")
        ax.set_ylabel("Generated MCF7, mean log1p")
        ax.text(0.04, 0.96, f"r = {score:.2f}", transform=ax.transAxes, va="top")
    fig.suptitle("sci-Plex: generated profiles against the measured query", y=1.02)
    fig.tight_layout()
    for folder in (OUT, ARTIFACTS):
        fig.savefig(folder / "sciplex_profile.png", dpi=140, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    x = np.arange(len(compounds))
    width = 0.36
    mcf7_delta = [sciplex["delta_corr_vs_real_mcf7"][name] for name in compounds]
    a549_delta = [sciplex["delta_corr_vs_real_a549"][name] for name in compounds]
    ax.bar(x - width / 2, mcf7_delta, width, label="Measured MCF7 drug effect", color="#ee6c4d")
    ax.bar(x + width / 2, a549_delta, width, label="Measured A549 drug effect", color="#3d5a80")
    ax.axhline(0, color="#888888", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short[name] for name in compounds])
    ax.set_ylabel("Correlation with the generated drug-versus-vehicle shift")
    ax.set_title("sci-Plex drug shift compared with each measured line")
    ax.legend(frameon=False)
    fig.tight_layout()
    for folder in (OUT, ARTIFACTS):
        fig.savefig(folder / "sciplex_shift.png", dpi=140, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
