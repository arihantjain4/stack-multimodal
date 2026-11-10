"""Build every evaluation profile set on the STACK gene panel.

Sets (all from real measurements; pseudobulk is a sum of real cells and
thinning is binomial subsampling of real reads):
  <sc>/cells            single cells or nuclei
  <sc>/pb<k>            pseudobulk of k cells from one group
  <sc>/pb<k>_thinned    the largest pseudobulk thinned to the reference's median depth
  <bulk>/native         real bulk RNA-seq (ARCHS4 counts)
  <bulk>/thinned        real bulk thinned to the matched reference's median depth
  <bulk>/depth_x<m>     real bulk thinned to m times that median (depth sweep)
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from common import load_config, load_gene_list, path, rng, symbol_matrix


def library(x) -> np.ndarray:
    return np.asarray(x.sum(axis=1)).ravel()


def thin(counts: np.ndarray, target: np.ndarray, generator: np.random.Generator) -> np.ndarray:
    """Binomial thinning of integer counts so each row has about `target` reads."""
    counts = np.rint(counts).astype(np.int64)
    totals = counts.sum(axis=1)
    p = np.clip(target / np.maximum(totals, 1), 0.0, 1.0)
    return generator.binomial(counts, p[:, None]).astype(np.float32)


def group_labels(obs: pd.DataFrame, columns: list[str]) -> pd.Series:
    return obs[columns].astype(str).agg("|".join, axis=1)


def pseudobulk(x: sparse.csr_matrix, groups: pd.Series, k: int, n_profiles: int, generator) -> tuple[np.ndarray, list[str]]:
    rows, labels = [], []
    for label, members in groups.groupby(groups, sort=True):
        positions = np.flatnonzero(groups.to_numpy() == label)
        if positions.size < k:
            continue
        for _ in range(n_profiles):
            chosen = generator.choice(positions, k, replace=False)
            rows.append(np.asarray(x[chosen].sum(axis=0)).ravel())
            labels.append(label)
    return np.vstack(rows).astype(np.float32), labels


def frame(name: str, source: str, kind: str, level, labels, x: np.ndarray, extra: dict | None = None) -> ad.AnnData:
    obs = pd.DataFrame(
        {"set": name, "source": source, "kind": kind, "level": str(level), "label": [str(v) for v in labels]}
    )
    for key, values in (extra or {}).items():
        obs[key] = [str(v) for v in values]
    obs["library_size"] = library(x)
    obs.index = [f"{name}:{i}" for i in range(len(obs))]
    return ad.AnnData(X=sparse.csr_matrix(x), obs=obs)


def main() -> None:
    cfg = load_config()
    bench = cfg["benchmark"]
    data = path(cfg, "data")
    model_genes = load_gene_list(cfg)
    generator = rng(cfg, 41)
    pieces: list[ad.AnnData] = []
    reference_depth: dict[str, float] = {}

    for name, spec in bench["singlecell"].items():
        cells = ad.read_h5ad(data / spec["file"])
        if list(cells.var_names) != model_genes:
            raise RuntimeError(f"{name} is not on the model gene panel")
        x = sparse.csr_matrix(cells.X, dtype=np.float32)
        groups = group_labels(cells.obs, spec["group_columns"])
        reference_depth[name] = float(np.median(library(x)))
        levels = sorted(int(k) for k in bench["aggregation_levels"])
        for k in levels:
            if k == 1:
                keep = np.concatenate(
                    [
                        generator.choice(np.flatnonzero(groups.to_numpy() == g), min(n, int(bench["cells_per_group_eval"])), replace=False)
                        for g, n in groups.value_counts().sort_index().items()
                    ]
                )
                pieces.append(frame(f"{name}/cells", name, "cell", 1, groups.to_numpy()[keep], x[keep].toarray()))
                continue
            profiles, labels = pseudobulk(x, groups, k, int(bench["profiles_per_group"]), generator)
            pieces.append(frame(f"{name}/pb{k}", name, "pseudobulk", k, labels, profiles))
        top = max(levels)
        largest = pieces[-1]
        thinned = thin(largest.X.toarray(), np.full(largest.n_obs, reference_depth[name]), generator)
        pieces.append(frame(f"{name}/pb{top}_thinned", name, "pseudobulk_thinned", top, largest.obs["label"], thinned))
        print(name, "median depth", reference_depth[name], flush=True)

    for name, spec in bench["bulk"].items():
        bulk = ad.read_h5ad(data / spec["file"])
        mapper = symbol_matrix(bulk.var["symbol"].tolist(), model_genes)
        x = np.asarray((sparse.csr_matrix(bulk.X) @ mapper).todense(), dtype=np.float32)
        matched = int((np.asarray(mapper.sum(axis=0)).ravel() > 0).sum())
        print(name, "gene overlap", matched, "/", len(model_genes), flush=True)
        labels = bulk.obs[spec["label_column"]].astype(str).to_numpy()
        extra = {column: bulk.obs[column].to_numpy() for column in bulk.obs.columns if column in ("treatment", "series_id", "geo_accession")}
        pieces.append(frame(f"{name}/native", name, "bulk", "native", labels, x, extra))
        depth = reference_depth[spec["reference"]]
        pieces.append(frame(f"{name}/thinned", name, "bulk_thinned", "ref_depth", labels, thin(x, np.full(len(x), depth), generator), extra))
        for multiplier in bench["depth_multipliers"]:
            target = np.minimum(depth * float(multiplier), library(x))
            pieces.append(
                frame(f"{name}/depth_x{multiplier}", name, "depth_sweep", multiplier, labels, thin(x, target, generator), extra)
            )

    profiles = ad.concat(pieces, axis=0, join="outer")
    profiles.var_names = model_genes
    profiles.uns["reference_depth"] = reference_depth
    profiles.write_h5ad(data / "profiles.h5ad")
    summary = profiles.obs.groupby("set", sort=False).agg(n=("label", "size"), groups=("label", "nunique"), median_library=("library_size", "median"))
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
