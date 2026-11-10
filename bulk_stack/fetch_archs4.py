"""Pull real bulk RNA-seq counts from ARCHS4 by reading only the needed columns.

Two sets are taken by rule:
* every sample of the GENtoniK series (hESC and iPSC cortical neurons, DMSO or
  maturation compounds), the Group A bulk dataset from the Engram meeting;
* bulk MCF7 and A549 samples, the two sci-Plex lines, spread across series.
"""

from __future__ import annotations

import json
import re
import sys

import anndata as ad
import fsspec
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from common import load_config, path, rng


def decode(values) -> np.ndarray:
    return np.array([v.decode() if isinstance(v, bytes) else str(v) for v in values], dtype=object)


def _read_chunk(args) -> np.ndarray:
    url, block_kb, indices = args
    handle = fsspec.open(url, "rb", block_size=int(block_kb) * 2**10, cache_type="blockcache").open()
    expression = h5py.File(handle, "r")["data/expression"]
    return np.stack([expression[:, int(i)] for i in indices], axis=1)


def read_columns(a4: dict, indices: np.ndarray) -> np.ndarray:
    """Read sample columns with several independent HTTP readers (the reads are latency-bound)."""
    from multiprocessing import Pool

    order = np.argsort(indices)
    parts = np.array_split(indices[order], min(int(a4["parallel_readers"]), indices.size))
    with Pool(len(parts)) as pool:
        blocks = pool.map(_read_chunk, [(a4["url"], a4["expression_block_size_kb"], part) for part in parts])
    columns = np.empty((blocks[0].shape[0], indices.size), dtype=np.uint32)
    columns[:, order] = np.concatenate(blocks, axis=1)
    return columns


def add_labels(frame: pd.DataFrame, gt: dict) -> pd.DataFrame:
    for label, regex in gt["labels"].items():
        values = frame[gt["field"]].str.extract(re.compile(regex), expand=False).str.strip()
        if values.isna().any():
            raise RuntimeError(f"Missing {label} for {frame.loc[values.isna(), 'geo_accession'].tolist()}")
        frame[label] = values
    return frame


def relabel_saved(cfg: dict) -> None:
    """Re-derive GENtoniK labels in an existing download from its stored GEO characteristics."""
    target = path(cfg, "data") / "bulk" / "gentonik_archs4.h5ad"
    adata = ad.read_h5ad(target)
    adata.obs = add_labels(adata.obs.copy(), cfg["archs4"]["gentonik"])
    adata.write_h5ad(target)
    print(adata.obs.groupby(list(cfg["archs4"]["gentonik"]["labels"]), observed=True).size().to_string(), flush=True)


def main() -> None:
    cfg = load_config()
    a4 = cfg["archs4"]
    out = path(cfg, "data") / "bulk"
    out.mkdir(parents=True, exist_ok=True)
    if "--relabel" in sys.argv[1:]:
        relabel_saved(cfg)
        return
    handle = fsspec.open(a4["url"], "rb", block_size=int(a4["block_size_mb"]) * 2**20, cache_type="blockcache").open()
    h5 = h5py.File(handle, "r")
    samples = h5["meta/samples"]
    meta = pd.DataFrame(
        {
            "geo_accession": decode(samples["geo_accession"][:]),
            "series_id": decode(samples["series_id"][:]),
            "title": decode(samples["title"][:]),
            "library_strategy": decode(samples["library_strategy"][:]),
            "singlecellprobability": samples["singlecellprobability"][:],
            "alignedreads": samples["alignedreads"][:],
        }
    )
    print("archs4 samples", len(meta), flush=True)
    bulk_ok = meta["library_strategy"].eq(a4["library_strategy"]) & (
        meta["singlecellprobability"] < float(a4["max_singlecell_probability"])
    )
    genes = pd.DataFrame(
        {
            "symbol": decode(h5["meta/genes/symbol"][:]),
            "ensembl_gene": decode(h5["meta/genes/ensembl_gene"][:]),
            "biotype": decode(h5["meta/genes/biotype"][:]),
        }
    )

    gt = a4["gentonik"]
    cl = a4["cell_lines"]
    for field_name in {gt["field"], cl["field"]}:
        meta[field_name] = decode(samples[field_name][:])
    series_mask = meta["series_id"].str.split(",").apply(lambda ids: gt["series"] in [i.strip() for i in ids])
    gentonik = add_labels(meta[series_mask].copy(), gt)
    print("gentonik samples", len(gentonik), gentonik.groupby(list(gt["labels"])).size().to_dict(), flush=True)

    picked = []
    generator = rng(cfg, 11)
    for line, regex in cl["lines"].items():
        hits = meta[bulk_ok & meta[cl["field"]].str.contains(regex, regex=True) & (meta["alignedreads"] >= float(cl["min_aligned_reads"]))]
        others = [r for name, r in cl["lines"].items() if name != line]
        for other in others:
            hits = hits[~hits[cl["field"]].str.contains(other, regex=True)]
        hits = hits.assign(first_series=hits["series_id"].str.split(",").str[0])
        shuffled = hits.sample(frac=1.0, random_state=int(generator.integers(2**31)))
        capped = shuffled.groupby("first_series", sort=False).head(int(cl["max_per_series"]))
        chosen = capped.head(int(cl["n_per_line"])).assign(cell_line=line)
        print(line, "candidates", len(hits), "series", hits["first_series"].nunique(), "chosen", len(chosen), flush=True)
        picked.append(chosen)
    lines_meta = pd.concat(picked)

    tr = a4["training"]
    held_out_series = set()
    for frame in (gentonik, lines_meta):
        for ids in frame["series_id"]:
            held_out_series.update(i.strip() for i in ids.split(","))
    series_sets = meta["series_id"].str.split(",").apply(lambda ids: {i.strip() for i in ids})
    mentions_line = np.zeros(len(meta), dtype=bool)
    for regex in cl["lines"].values():
        mentions_line |= meta[cl["field"]].str.contains(regex, regex=True).to_numpy()
    eligible = (
        bulk_ok
        & (meta["alignedreads"] >= float(tr["min_aligned_reads"]))
        & ~series_sets.apply(lambda ids: bool(ids & held_out_series))
        & ~mentions_line
    )
    pool_meta = meta[eligible].assign(first_series=lambda f: f["series_id"].str.split(",").str[0])
    shuffled = pool_meta.sample(frac=1.0, random_state=int(generator.integers(2**31)))
    training = shuffled.groupby("first_series", sort=False).head(int(tr["max_per_series"])).head(int(tr["n_samples"]))
    print("training candidates", len(pool_meta), "chosen", len(training), "series", training["first_series"].nunique(), flush=True)

    for name, frame in [("gentonik", gentonik), ("cell_lines", lines_meta), ("training", training)]:
        if (out / f"{name}_archs4.h5ad").exists():
            continue
        indices = frame.index.to_numpy()
        print(f"reading {indices.size} columns for {name}", flush=True)
        counts = read_columns(a4, indices)
        obs = frame.copy()
        obs.index = frame["geo_accession"].astype(str).to_numpy()
        obs.index.name = "geo_accession"
        var = genes.copy()
        var.index = pd.Index(genes["ensembl_gene"].astype(str), name="ensembl_gene")
        adata = ad.AnnData(X=sparse.csr_matrix(counts.T.astype(np.float32)), obs=obs, var=var)
        adata.uns["source"] = json.dumps({"archs4": a4["url"]})
        adata.write_h5ad(out / f"{name}_archs4.h5ad")
        print(name, adata.shape, "median library", float(np.median(np.asarray(adata.X.sum(1)).ravel())), flush=True)


if __name__ == "__main__":
    main()
