"""Build STACK base/test h5ad files from the sci-Plex 3 screen.

The query is untreated target-line cells. Context is the source line at the
same exposure: vehicle plus the compounds that share one dose. The dose is the
one with the most compounds that clear the configured cell-count minimum in
both lines. Counts stay raw. Gene symbols come from the GEO annotation file,
because the h5ad index collapsed missing symbols.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from common import ROOT, group_rng, load_config, load_gene_list, read_model_config, take, window_sizes

DATA = ROOT / "data" / "sciplex"
INPUTS = DATA / "stack_inputs"


def ensure_h5ad(cfg: dict) -> Path:
    path = DATA / cfg["h5ad_name"]
    if path.exists():
        return path
    import urllib.request

    article = cfg["figshare_article_id"]
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{article}", timeout=60) as response:
        payload = json.load(response)
    matches = [item for item in payload.get("files", []) if item.get("name") == cfg["h5ad_name"]]
    if len(matches) != 1:
        raise RuntimeError(f"Figshare article {article} does not list {cfg['h5ad_name']}")
    url = matches[0]["download_url"]
    partial = path.with_suffix(path.suffix + ".partial")
    import subprocess

    subprocess.run(
        ["curl", "-fL", "--retry", "5", "--retry-all-errors", "-C", "-", "-o", str(partial), url],
        check=True,
    )
    partial.replace(path)
    return path


def load_symbols(annotation: Path, ensembl: pd.Series) -> list[str | None]:
    """Symbol for every matrix column.

    The h5ad var table carries the annotation's header line as an extra first row,
    so var row i + 1 names the gene, but matrix column i holds annotation gene i
    (the var `ncounts` of row i are the totals of column i). The trailing column
    has no annotation entry.
    """
    ids: list[str] = []
    symbols: list[str] = []
    with gzip.open(annotation, "rt") as handle:
        header = handle.readline()
        if "gene_short_name" not in header:
            raise RuntimeError(f"Unexpected gene annotation header: {header!r}")
        for line in handle:
            ensembl_id, symbol = line.rstrip("\n").split(None, 1)
            ids.append(ensembl_id.split(".", 1)[0])
            symbols.append(symbol)
    observed = ensembl.astype(str).to_numpy()
    if observed.shape[0] != len(ids) + 1:
        raise RuntimeError(
            f"Annotation has {len(ids)} genes but the matrix has {observed.shape[0]} columns"
        )
    if not str(observed[1]).startswith(ids[0]):
        raise RuntimeError("Matrix gene columns are not in GEO annotation order")
    return [*symbols, None]


def select_groups(obs: pd.DataFrame, cfg: dict, n_base: int, n_test: int, seed: int):
    columns = cfg["columns"]
    cell_line = columns["cell_line"]
    time_col = columns["time"]
    dose_col = columns["dose"]
    perturbation = columns["perturbation"]
    source = cfg["source_cell_type"]
    target = cfg["target_cell_type"]
    minimum = int(cfg["min_cells_per_group"])
    n_compounds = int(cfg["n_compounds"])

    work = obs.dropna(subset=[cell_line, time_col, dose_col, perturbation]).copy()
    control_dose = work[dose_col].min()
    at_control_dose = work.loc[work[dose_col].eq(control_dose), perturbation]
    control_label = at_control_dose.value_counts().idxmax()

    control_cells = work[work[perturbation].eq(control_label)]
    exposures = []
    for time, block in control_cells.groupby(time_col):
        n_source = int(block[cell_line].eq(source).sum())
        n_target = int(block[cell_line].eq(target).sum())
        if n_source and n_target:
            exposures.append((n_target, n_source, time))
    if not exposures:
        raise RuntimeError(f"No shared exposure with {control_label!r} cells in {source} and {target}")
    exposure = max(exposures)[-1]

    treated = work[work[time_col].eq(exposure) & work[perturbation].ne(control_label)]
    counts = (
        treated.groupby([dose_col, perturbation, cell_line], observed=True)
        .size()
        .unstack(cell_line, fill_value=0)
    )
    dose_scores = []
    for dose, block in counts.groupby(level=0):
        block = block.copy()
        if isinstance(block.index, pd.MultiIndex):
            block.index = block.index.droplevel(0)
        if source not in block.columns or target not in block.columns:
            continue
        n_source = block[source].astype(int)
        n_target = block[target].astype(int)
        keep = (n_source >= minimum) & (n_target >= minimum)
        eligible = []
        total = 0
        for compound in block.index[keep]:
            source_cells = int(n_source.loc[compound])
            target_cells = int(n_target.loc[compound])
            eligible.append((min(source_cells, target_cells), str(compound), source_cells, target_cells))
            total += source_cells + target_cells
        if eligible:
            dose_scores.append((len(eligible), total, -float(dose), dose, eligible))
    if not dose_scores:
        raise RuntimeError(f"No dose at exposure {exposure} has compounds with {minimum} cells in both lines")
    _n, _total, _neg_dose, dose, eligible = max(dose_scores)
    eligible.sort(key=lambda item: (-item[0], item[1]))
    compounds = [item[1] for item in eligible[:n_compounds]]

    base_parts = []
    for label in [control_label, *compounds]:
        mask = (
            work[cell_line].eq(source)
            & work[time_col].eq(exposure)
            & work[perturbation].eq(label)
        )
        if label != control_label:
            mask = mask & work[dose_col].eq(dose)
        chosen = take(work.index[mask].to_numpy(), n_base, group_rng(seed, f"sciplex-base-{label}"))
        base_parts.append(work.loc[chosen])
    base = pd.concat(base_parts)
    query_mask = (
        work[cell_line].eq(target) & work[time_col].eq(exposure) & work[perturbation].eq(control_label)
    )
    query_index = take(work.index[query_mask].to_numpy(), n_test, group_rng(seed, "sciplex-query"))
    test = work.loc[query_index]
    details = {
        "control_label": str(control_label),
        "control_dose": float(control_dose),
        "exposure": float(exposure) if isinstance(exposure, float) else exposure,
        "dose": float(dose),
        "compounds": compounds,
        "compound_support": [
            {"perturbation": item[1], "min_cells": int(item[0]), "source_cells": int(item[2]), "target_cells": int(item[3])}
            for item in eligible[:n_compounds]
        ],
        "n_base": int(base.shape[0]),
        "n_query": int(test.shape[0]),
    }
    return base, test, details


def model_columns(symbols: list[str | None], model_genes: list[str]) -> list[list[int]]:
    by_symbol: dict[str, list[int]] = {}
    for column, symbol in enumerate(symbols):
        if symbol:
            by_symbol.setdefault(symbol.upper(), []).append(column)
    columns: list[list[int]] = []
    matched = 0
    for gene in model_genes:
        hits = by_symbol.get(gene.upper(), [])
        columns.append(hits)
        if hits:
            matched += 1
    if matched == 0:
        raise RuntimeError("No sci-Plex gene symbols matched the model gene list")
    print(f"sciplex gene overlap {matched}/{len(model_genes)}", flush=True)
    return columns


def read_counts(
    h5ad_path: Path,
    row_positions: np.ndarray,
    columns: list[list[int]],
    n_source_genes: int,
) -> sparse.csr_matrix:
    """Pull selected rows from the backed CSR matrix and sum onto model genes."""
    n_obs = row_positions.size
    n_vars = len(columns)
    destinations: list[list[int]] = [[] for _ in range(n_source_genes)]
    for model_index, cols in enumerate(columns):
        for column in cols:
            destinations[column].append(model_index)

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    order = np.argsort(row_positions, kind="mergesort")
    with h5py.File(h5ad_path, "r") as handle:
        indptr = handle["X/indptr"][:]
        indices = handle["X/indices"]
        values = handle["X/data"]
        for out_row, source_row in zip(order, row_positions[order], strict=True):
            start = int(indptr[int(source_row)])
            end = int(indptr[int(source_row) + 1])
            if start == end:
                continue
            gene_ids = indices[start:end]
            counts = values[start:end]
            for gene_id, count in zip(gene_ids, counts, strict=True):
                if count < 0:
                    raise RuntimeError("Negative count in sci-Plex matrix")
                gene = int(gene_id)
                if gene < 0 or gene >= n_source_genes:
                    continue
                for model_index in destinations[gene]:
                    rows.append(int(out_row))
                    cols.append(model_index)
                    data.append(float(count))
    if not data:
        return sparse.csr_matrix((n_obs, n_vars), dtype=np.float32)
    return sparse.coo_matrix(
        (np.asarray(data, dtype=np.float32), (np.asarray(rows), np.asarray(cols))),
        shape=(n_obs, n_vars),
        dtype=np.float32,
    ).tocsr()


def write_split(frame: pd.DataFrame, matrix: sparse.csr_matrix, model_genes: list[str], keep: list[str], path: Path) -> None:
    obs = frame.loc[:, keep].copy()
    obs.index = frame.index.astype(str)
    obs.index.name = frame.index.name
    var = pd.DataFrame(index=pd.Index(model_genes, name="gene"))
    ad.AnnData(X=matrix, obs=obs, var=var).write_h5ad(path)


def main() -> None:
    cfg = load_config()
    screen = cfg["sciplex"]
    stack = cfg["stack"]
    model_dir = ROOT / "models"
    model_config = read_model_config(model_dir / stack["checkpoint_name"])
    model_genes = load_gene_list(model_dir / stack["genelist_name"])
    n_base, n_test = window_sizes(model_config, stack)
    h5ad_path = ensure_h5ad(screen)
    adata = ad.read_h5ad(h5ad_path, backed="r")
    base_obs, test_obs, details = select_groups(adata.obs, screen, n_base, n_test, int(stack["random_seed"]))
    print("sciplex selection", json.dumps(details), flush=True)
    symbols = load_symbols(DATA / Path(screen["files"]["genes"]).name, adata.var["ensembl_id"])
    columns = model_columns(symbols, model_genes)
    positions = pd.Series(np.arange(adata.n_obs), index=adata.obs.index)
    base_rows = positions.loc[base_obs.index].to_numpy()
    test_rows = positions.loc[test_obs.index].to_numpy()
    print(f"reading {base_rows.size + test_rows.size} sci-Plex cells", flush=True)
    base_x = read_counts(h5ad_path, base_rows, columns, adata.n_vars)
    test_x = read_counts(h5ad_path, test_rows, columns, adata.n_vars)
    keep = [
        screen["columns"]["cell_line"],
        screen["columns"]["time"],
        screen["columns"]["dose"],
        screen["columns"]["perturbation"],
    ]
    INPUTS.mkdir(parents=True, exist_ok=True)
    write_split(base_obs, base_x, model_genes, keep, INPUTS / "base.h5ad")
    write_split(test_obs, test_x, model_genes, keep, INPUTS / "test.h5ad")
    manifest = {
        "dataset": "sciplex",
        "gsm": screen["gsm"],
        "h5ad": str(h5ad_path),
        "selection": details,
        "base_h5ad": str(INPUTS / "base.h5ad"),
        "test_h5ad": str(INPUTS / "test.h5ad"),
        "n_genes": len(model_genes),
        "base_nnz": int(base_x.nnz),
        "test_nnz": int(test_x.nnz),
    }
    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("sciplex inputs ready", INPUTS / "base.h5ad", INPUTS / "test.h5ad", flush=True)


if __name__ == "__main__":
    main()
