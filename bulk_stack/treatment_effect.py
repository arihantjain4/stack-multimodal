"""Does imputation keep the biology bulk users care about? GENtoniK vs DMSO fold changes.

For every variant: hide a random gene subset in all 29 GENtoniK samples (one window),
impute it, and compute per-line log2 fold change (treated vs DMSO) on the hidden,
expressed genes. Compare with the fold change measured from the real counts, and with
the published DESeq2 tables. The leave-one-out population mean is scored as well.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from common import load_config, load_stack, path, rng, set_threads


def log2fc(values: np.ndarray, treated: np.ndarray, control: np.ndarray, genes: np.ndarray, pseudocount: float) -> np.ndarray:
    sub = values[:, genes]
    cpm = 1e6 * sub / np.maximum(sub.sum(axis=1, keepdims=True), 1e-12)
    return np.log2(cpm[treated].mean(0) + pseudocount) - np.log2(cpm[control].mean(0) + pseudocount)


def fetch_published(cfg: dict, spec: dict) -> Path:
    target = path(cfg, "data") / spec["file"]
    if not target.exists():
        series = cfg["archs4"]["gentonik"]["series"]
        url = cfg["treatment_effect"]["geo_supplement_url"].format(stub=series[:-3], series=series, file=Path(spec["file"]).name)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        urllib.request.urlretrieve(url, partial)
        partial.rename(target)
    return target


def published_table(cfg: dict, spec: dict, symbols_by_ensembl: dict) -> pd.DataFrame:
    table = pd.read_csv(fetch_published(cfg, spec))
    ids = table[spec["id_column"]].astype(str)
    if spec["id_type"] == "ensembl":
        ids = ids.str.split(".").str[0].map(symbols_by_ensembl)
    elif spec["id_type"] != "symbol":
        raise ValueError(f"Unsupported id type {spec['id_type']!r}")
    out = pd.DataFrame({"symbol": ids.str.upper(), "lfc": table[spec["lfc_column"]], "padj": table[spec["padj_column"]]})
    # DESeq2 leaves log2FC empty for genes filtered by independent filtering or with zero counts.
    return out.dropna(subset=["symbol", "lfc"]).drop_duplicates("symbol").set_index("symbol")


def main() -> None:
    set_threads()
    from run_variants import make_runner
    from stack_ops import loo_mean, run_window
    from train_adapter import reference_depth

    cfg = load_config()
    task = cfg["treatment_effect"]
    profiles = ad.read_h5ad(path(cfg, "data") / "profiles.h5ad")
    mask_rows = profiles.obs["set"].eq(task["set"]).to_numpy()
    obs = profiles.obs[mask_rows]
    counts = profiles.X[mask_rows].toarray()
    genes_upper = np.array([g.upper() for g in profiles.var_names])
    bulk = ad.read_h5ad(path(cfg, "data") / cfg["benchmark"]["bulk"][task["set"].split("/")[0]]["file"], backed="r")
    symbols_by_ensembl = dict(zip(bulk.var.index.astype(str), bulk.var["symbol"].astype(str)))
    published = {line: published_table(cfg, spec, symbols_by_ensembl) for line, spec in task["published"].items()}

    model = load_stack(cfg)
    depth = reference_depth(cfg)
    runners = {"population_mean": None}
    for name in task["variants"]:
        runners[name] = (lambda c, m: run_window(model, c, m)) if name == "baseline" else make_runner(cfg, name, model, depth)

    pc = float(task["pseudocount_cpm"])
    lines = obs[task["group_column"]].astype(str).to_numpy()
    treatment = obs[task["treatment_column"]].astype(str).to_numpy()
    rows = []
    sanity = []
    for line in sorted(set(lines)):
        treated = (lines == line) & (treatment == task["treated"])
        control = (lines == line) & (treatment == task["control"])
        if treated.sum() == 0 or control.sum() == 0:
            continue
        expressed = (1e6 * counts / counts.sum(1, keepdims=True))[treated | control].mean(0) >= float(task["min_cpm"])
        measured_all = log2fc(counts, treated, control, np.flatnonzero(expressed), pc)
        if line in published:
            joined = pd.DataFrame({"measured": measured_all}, index=genes_upper[expressed]).join(published[line], how="inner")
            sanity.append({"line": line, "n_genes": len(joined), "r_measured_vs_published": float(pearsonr(joined["measured"], joined["lfc"])[0])})
    print(pd.DataFrame(sanity).to_string(index=False), flush=True)

    for name, runner in runners.items():
        generator = rng(cfg, 91)
        collected: dict[str, list] = {}
        for _ in range(int(task["mask_repeats"])):
            hidden = np.zeros(counts.shape[1], dtype=bool)
            hidden[generator.choice(counts.shape[1], int(counts.shape[1] * float(cfg["benchmark"]["mask_fraction"])), replace=False)] = True
            if runner is None:
                predicted = loo_mean(counts / counts.sum(1, keepdims=True))
            else:
                predicted = runner(counts, hidden).px_scale
            for line in sorted(set(lines)):
                treated = (lines == line) & (treatment == task["treated"])
                control = (lines == line) & (treatment == task["control"])
                if treated.sum() == 0 or control.sum() == 0:
                    continue
                expressed = (1e6 * counts / counts.sum(1, keepdims=True))[treated | control].mean(0) >= float(task["min_cpm"])
                genes = np.flatnonzero(hidden & expressed)
                bucket = collected.setdefault(line, [[], [], []])
                bucket[0].append(log2fc(predicted, treated, control, genes, pc))
                bucket[1].append(log2fc(counts, treated, control, genes, pc))
                bucket[2].append(genes_upper[genes])
        for line, (pred, meas, names) in collected.items():
            pred, meas, names = np.concatenate(pred), np.concatenate(meas), np.concatenate(names)
            row = {"variant": name, "line": line, "n_genes": int(pred.size), "r_vs_measured": float(pearsonr(pred, meas)[0])}
            if line in published:
                frame = pd.DataFrame({"pred": pred, "meas": meas}, index=names).join(published[line], how="inner")
                significant = frame[frame["padj"] < float(task["padj_threshold"])]
                row["r_vs_published"] = float(pearsonr(frame["pred"], frame["lfc"])[0])
                row["n_significant"] = int(len(significant))
                row["sign_agreement_significant"] = float((np.sign(significant["pred"]) == np.sign(significant["lfc"])).mean()) if len(significant) else float("nan")
            rows.append(row)
            print(row, flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(path(cfg, "results") / "treatment_effect.csv", index=False)
    pd.DataFrame(sanity).to_csv(path(cfg, "results") / "treatment_effect_sanity.csv", index=False)
    summary = table.groupby("variant", sort=False)[[c for c in ("r_vs_measured", "r_vs_published", "sign_agreement_significant") if c in table]].mean()
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"), flush=True)


if __name__ == "__main__":
    main()
