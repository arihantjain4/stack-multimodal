"""Single-cell references for the bulk comparison, on the STACK gene panel.

* SCP1756 cortical organoids (10x 3' scRNA-seq): every annotated cell type at two
  ages, up to `cells_per_group` cells each.
* sci-Plex 3 (sci-RNA-seq3, single nuclei): vehicle-treated cells of every line
  at the configured exposure.

Raw downloads are cached in engram_inference/data so the two folders share them.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd

from common import engram_module, load_config, load_gene_list, path, rng

organoid = engram_module("prepare_organoid")
sciplex = engram_module("prepare_sciplex")


def download_objects(token: str, bucket: str, names: list[str], dest: Path) -> None:
    jobs = []
    header = dest / ".auth.header"
    dest.mkdir(parents=True, exist_ok=True)
    header.write_text(f"Authorization: Bearer {token}\n")
    header.chmod(0o600)
    try:
        for name in names:
            target = dest / name
            if target.exists():
                continue
            partial = target.with_suffix(target.suffix + ".partial")
            url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{quote(name, safe='')}?alt=media"
            command = ["curl", "-fsSL", "--retry", "5", "--retry-all-errors", "-C", "-", "-H", f"@{header}", "-o", str(partial), url]
            jobs.append((target, partial, subprocess.Popen(command)))
        for target, partial, process in jobs:
            if process.wait() != 0:
                raise RuntimeError(f"Download failed for {target.name}")
            partial.replace(target)
            print("downloaded", target.name, flush=True)
    finally:
        header.unlink(missing_ok=True)


def build_organoid(cfg: dict, model_genes: list[str], out: Path) -> None:
    scp = cfg["scp1756"]
    data_dir = organoid.DATA
    raw_dir = organoid.RAW
    token = organoid.renew_token(scp["accession"])
    bucket = organoid.resolve_bucket(token, scp["terra_namespace"], scp["terra_workspace"], data_dir / "bucket.txt")
    for age in scp["ages"]:
        names = [template.format(age=age) for template in scp["objects"]]
        expression = [n for n in names if n.startswith("expression_")]
        download_objects(token, bucket, [n for n in names if n not in expression], data_dir)
        download_objects(token, bucket, expression, raw_dir)

    exclude = re.compile(scp["exclude_cell_type_regex"])
    generator = rng(cfg, 21)
    parts = []
    for age in scp["ages"]:
        umap_name, genes_name, barcodes_name, expression_name = (t.format(age=age) for t in scp["objects"])
        clusters = pd.read_csv(data_dir / umap_name, sep="\t", skiprows=[1])
        labels = clusters[scp["cell_type_column"]].astype(str)
        clusters = clusters[~labels.map(lambda v: bool(exclude.search(v)))]
        chosen = []
        for cell_type, block in clusters.groupby(scp["cell_type_column"]):
            if len(block) < int(scp["min_cells_per_group"]):
                continue
            take = min(len(block), int(scp["cells_per_group"]))
            chosen.append(block.loc[generator.choice(block.index.to_numpy(), take, replace=False)])
        frame = pd.concat(chosen).reset_index(drop=True)
        frame = frame[[scp["barcode_column"], scp["cell_type_column"]]].rename(
            columns={scp["barcode_column"]: "barcode", scp["cell_type_column"]: "cell_type"}
        )
        frame["age"] = age
        barcodes = organoid.barcode_index(data_dir / barcodes_name)
        gene_map, n_source_genes = organoid.gene_column_map(data_dir / genes_name, model_genes)
        raw_path = raw_dir / expression_name
        gzipped, dim_a, dim_b, _ = organoid.matrix_header(raw_path)
        swap = (dim_a, dim_b) == (len(barcodes), n_source_genes)
        if not swap and (dim_a, dim_b) != (n_source_genes, len(barcodes)):
            raise RuntimeError(f"{expression_name} has unexpected dimensions {(dim_a, dim_b)}")
        cell_map = raw_dir / f"bulk_cells_{age}.tsv"
        with cell_map.open("w") as handle:
            for row, barcode in enumerate(frame["barcode"]):
                handle.write(f"{barcodes[barcode]}\tcells\t{row}\n")
        triplets = raw_dir / f"bulk_{age}.triplets.tsv"
        print("filtering", expression_name, len(frame), "cells", flush=True)
        organoid.filter_matrix(raw_path, gzipped, swap, gene_map, cell_map, triplets)
        matrix = organoid.triplets_by_role(triplets, {"cells": (len(frame), len(model_genes))})["cells"]
        obs = frame.set_index(frame["age"] + ":" + frame["barcode"])
        parts.append(ad.AnnData(X=matrix, obs=obs[["cell_type", "age"]], var=pd.DataFrame(index=pd.Index(model_genes, name="gene"))))
    merged = ad.concat(parts, axis=0)
    merged.obs["dataset"] = "organoid_10x"
    merged.write_h5ad(out / "organoid_cells.h5ad")
    print("organoid", merged.shape, merged.obs.groupby(["age", "cell_type"], observed=True).size().to_dict(), flush=True)


def build_sciplex(cfg: dict, model_genes: list[str], out: Path) -> None:
    sp = cfg["sciplex"]
    sciplex.DATA.mkdir(parents=True, exist_ok=True)
    h5ad_path = sciplex.ensure_h5ad(sp)
    annotation = sciplex.DATA / Path(sp["geo_gene_annotation"]).name
    if not annotation.exists():
        subprocess.run(["curl", "-4", "--http1.1", "-A", "Mozilla/5.0", "-fsSL", "-o", str(annotation), sp["geo_gene_annotation"]], check=True)
    backed = ad.read_h5ad(h5ad_path, backed="r")
    columns = sp["columns"]
    obs = backed.obs
    keep = obs[columns["perturbation"]].eq(sp["control_label"]) & obs[columns["time"]].eq(float(sp["exposure"]))
    generator = rng(cfg, 31)
    chosen = []
    for line, block in obs[keep].groupby(columns["cell_line"], observed=True):
        take = min(len(block), int(sp["cells_per_line"]))
        chosen.append(block.loc[generator.choice(block.index.to_numpy(), take, replace=False)])
    selected = pd.concat(chosen)
    symbols = sciplex.load_symbols(annotation, backed.var["ensembl_id"])
    model_cols = sciplex.model_columns(symbols, model_genes)
    positions = pd.Series(np.arange(backed.n_obs), index=obs.index)
    matrix = sciplex.read_counts(h5ad_path, positions.loc[selected.index].to_numpy(), model_cols, backed.n_vars)
    meta = pd.DataFrame(
        {"cell_type": selected[columns["cell_line"]].astype(str).to_numpy()},
        index=pd.Index(selected.index.astype(str), name="cell"),
    )
    meta["dataset"] = "sciplex_nuclei"
    adata = ad.AnnData(X=matrix, obs=meta, var=pd.DataFrame(index=pd.Index(model_genes, name="gene")))
    adata.write_h5ad(out / "sciplex_cells.h5ad")
    print("sciplex", adata.shape, meta["cell_type"].value_counts().to_dict(), flush=True)


def main() -> None:
    cfg = load_config()
    model_genes = load_gene_list(cfg)
    out = path(cfg, "data") / "singlecell"
    out.mkdir(parents=True, exist_ok=True)
    build_sciplex(cfg, model_genes, out)
    build_organoid(cfg, model_genes, out)


if __name__ == "__main__":
    main()
