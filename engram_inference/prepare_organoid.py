"""Build STACK base/test h5ad files from the SCP1756 raw count matrices.

Context cells are the class with the most cells at the earliest retained age.
Query cells are a different class, the one with the most cells at the latest
retained age. Ages are the earliest and latest cluster files that already have
gene and barcode sidecars. Counts stay raw; genes are the model list.
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import quote

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from common import ROOT, group_rng, load_config, load_gene_list, read_model_config, take, window_sizes

DATA = ROOT / "data" / "cortical_organoid"
RAW = DATA / "raw"
INPUTS = DATA / "stack_inputs"
AGE_RE = re.compile(r"^(\d+(?:\.\d+)?)(days|month)$")


def age_sort_key(label: str) -> float:
    match = AGE_RE.fullmatch(label)
    if match is None:
        raise ValueError(f"Unrecognized age label {label!r}")
    value = float(match.group(1))
    return value if match.group(2) == "days" else value * 30.0


def renew_token(accession: str) -> str:
    url = (
        "https://singlecell.broadinstitute.org/single_cell/api/v1/site/studies/"
        f"{accession}/renew_read_only_access_token"
    )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    token = payload.get("access_token")
    if not token:
        raise RuntimeError("SCP read-only token response had no access_token")
    return token


def resolve_bucket(token: str, namespace: str, workspace: str, cache: Path) -> str:
    if cache.exists():
        cached = cache.read_text().strip()
        if cached:
            return cached
    request = urllib.request.Request(
        "https://api.firecloud.org/api/workspaces",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json", "User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        workspaces = json.load(response)
    for item in workspaces:
        ws = item.get("workspace", item)
        if ws.get("namespace") == namespace and ws.get("name") == workspace:
            bucket = ws.get("bucketName")
            if not bucket:
                raise RuntimeError(f"Workspace {namespace}/{workspace} has no bucketName")
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(bucket + "\n")
            return bucket
    raise RuntimeError(f"Terra workspace {namespace}/{workspace} was not visible with the SCP token")


def discover_ages(cfg: dict) -> list[str]:
    ages = []
    for cluster in sorted(DATA.glob("umap_*.txt")):
        age = cluster.name[len("umap_") : -len(".txt")]
        genes = DATA / cfg["genes_filename_template"].format(age=age)
        barcodes = DATA / cfg["barcodes_filename_template"].format(age=age)
        if genes.exists() and barcodes.exists() and AGE_RE.fullmatch(age):
            ages.append(age)
    if len(ages) < 1:
        raise RuntimeError(f"No cluster/gene/barcode sets found in {DATA}")
    ages.sort(key=age_sort_key)
    if len(ages) == 1:
        return ages
    return [ages[0], ages[-1]]


def load_clusters(cfg: dict, ages: list[str]) -> dict[str, pd.DataFrame]:
    frames = {}
    barcode_column = cfg["barcode_column"]
    cell_type_column = cfg["cell_type_column"]
    for age in ages:
        path = DATA / cfg["cluster_filename_template"].format(age=age)
        frame = pd.read_csv(path, sep="\t", skiprows=[1])
        missing = {barcode_column, cell_type_column} - set(frame.columns)
        if missing:
            raise RuntimeError(f"{path.name} is missing columns {sorted(missing)}")
        frames[age] = frame
    return frames


def select_cells(cfg: dict, frames: dict[str, pd.DataFrame], n_base: int, n_test: int, seed: int):
    ages = list(frames)
    earliest, latest = ages[0], ages[-1]
    cell_type_column = cfg["cell_type_column"]
    barcode_column = cfg["barcode_column"]
    exclude = re.compile(cfg["exclude_cell_type_regex"])
    minimum = int(cfg["min_cells_per_age"])
    counts: dict[str, dict[str, int]] = {}
    for age, frame in frames.items():
        labels = frame[cell_type_column].astype(str)
        labels = labels[~labels.map(lambda value: bool(exclude.search(value)))]
        counts[age] = labels.value_counts().to_dict()
    eligible = set(counts[earliest])
    for age in ages[1:]:
        eligible &= set(counts[age])
    eligible = {label for label in eligible if all(counts[age].get(label, 0) >= minimum for age in ages)}
    if len(eligible) < 2:
        raise RuntimeError(f"Need two cell types present at {ages}; found {sorted(eligible)}")
    context = min(eligible, key=lambda label: (-counts[earliest][label], label))
    query = min(eligible - {context}, key=lambda label: (-counts[latest][label], label))

    base_rows = []
    for age in ages:
        frame = frames[age]
        mask = frame[cell_type_column].astype(str).eq(context)
        chosen = take(frame.index[mask].to_numpy(), n_base, group_rng(seed, f"organoid-base-{age}-{context}"))
        part = frame.loc[chosen, [barcode_column, cell_type_column]].copy()
        part["age"] = age
        part["role"] = "context"
        base_rows.append(part)
    base = pd.concat(base_rows, ignore_index=True)

    query_frame = frames[latest]
    query_mask = query_frame[cell_type_column].astype(str).eq(query)
    query_index = take(
        query_frame.index[query_mask].to_numpy(),
        n_test,
        group_rng(seed, f"organoid-query-{latest}-{query}"),
    )
    test = query_frame.loc[query_index, [barcode_column, cell_type_column]].copy()
    test["age"] = latest
    test["role"] = "query"
    test = test.reset_index(drop=True)

    selection = {
        "ages": ages,
        "context_cell_type": context,
        "query_cell_type": query,
        "context_counts": {age: int(counts[age][context]) for age in ages},
        "query_counts": {age: int(counts[age][query]) for age in ages},
        "n_base_per_age": int(base.groupby("age").size().min()),
        "n_query": int(test.shape[0]),
    }
    return base, test, selection


def barcode_index(path: Path) -> dict[str, int]:
    mapping = {}
    with path.open() as handle:
        for number, line in enumerate(handle, start=1):
            mapping[line.rstrip("\n")] = number
    return mapping


def gene_column_map(genes_path: Path, model_genes: list[str]) -> tuple[Path, int]:
    source = [line.rstrip("\n") for line in genes_path.read_text().splitlines() if line.strip()]
    upper_to_model: dict[str, list[int]] = {}
    for index, gene in enumerate(model_genes):
        upper_to_model.setdefault(gene.upper(), []).append(index)
    map_path = RAW / "gene_to_model.tsv"
    RAW.mkdir(parents=True, exist_ok=True)
    kept = 0
    with map_path.open("w") as handle:
        for source_index, gene in enumerate(source, start=1):
            for model_index in upper_to_model.get(gene.upper(), []):
                handle.write(f"{source_index}\t{model_index}\n")
                kept += 1
    if kept == 0:
        raise RuntimeError("No organoid genes matched the model gene list")
    return map_path, len(source)


def is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def matrix_header(path: Path) -> tuple[bool, int, int, int]:
    gzipped = is_gzip(path)
    opener = gzip.open if gzipped else open
    with opener(path, "rt") as handle:
        for line in handle:
            if line.startswith("%"):
                continue
            nrow, ncol, nnz = (int(part) for part in line.split()[:3])
            return gzipped, nrow, ncol, nnz
    raise RuntimeError(f"{path} has no Matrix Market size line")


def filter_matrix(path: Path, gzipped: bool, swap: bool, gene_map: Path, cell_map: Path, dest: Path) -> None:
    """Cell map columns are source_index, role, row_id. Output is role, row, model_col, count."""
    awk = r"""
BEGIN {
  while ((getline line < gene_file) > 0) {
    split(line, a, "\t"); genes[a[1]] = a[2]
  }
  while ((getline line < cell_file) > 0) {
    split(line, a, "\t"); cells[a[1]] = a[2] SUBSEP a[3]
  }
  seen = 0
}
$0 ~ /^%/ { next }
seen == 0 { seen = 1; next }
{
  g = swap ? $2 : $1
  c = swap ? $1 : $2
  if ((g in genes) && (c in cells)) {
    split(cells[c], loc, SUBSEP)
    print loc[1], loc[2], genes[g], $3
  }
}
"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    reader = ["gzip", "-dc", str(path)] if gzipped else ["cat", str(path)]
    source = subprocess.Popen(reader, stdout=subprocess.PIPE)
    with dest.open("w") as handle:
        awk_proc = subprocess.Popen(
            ["awk", "-v", f"gene_file={gene_map}", "-v", f"cell_file={cell_map}", "-v", f"swap={1 if swap else 0}", awk],
            stdin=source.stdout,
            stdout=handle,
        )
        source.stdout.close()
        if awk_proc.wait() != 0 or source.wait() != 0:
            raise RuntimeError(f"Filtering {path.name} failed")


def triplets_by_role(path: Path, shapes: dict[str, tuple[int, int]]) -> dict[str, sparse.csr_matrix]:
    buckets: dict[str, list[tuple[int, int, float]]] = {role: [] for role in shapes}
    if path.stat().st_size:
        with path.open() as handle:
            for line in handle:
                role, row, col, value = line.split()
                buckets[role].append((int(row), int(col), float(value)))
    matrices = {}
    for role, shape in shapes.items():
        entries = buckets[role]
        if not entries:
            matrices[role] = sparse.csr_matrix(shape, dtype=np.float32)
            continue
        rows, cols, data = zip(*entries)
        data_array = np.asarray(data, dtype=np.float32)
        if data_array.min() < 0:
            raise RuntimeError(f"Negative counts in {path} ({role})")
        matrices[role] = sparse.coo_matrix(
            (data_array, (np.asarray(rows), np.asarray(cols))),
            shape=shape,
            dtype=np.float32,
        ).tocsr()
    return matrices


def write_h5ad(frame: pd.DataFrame, matrix: sparse.csr_matrix, model_genes: list[str], barcode_column: str, path: Path) -> None:
    obs = frame.drop(columns=[barcode_column]).copy()
    obs.index = frame[barcode_column].astype(str).to_numpy()
    obs.index.name = barcode_column
    var = pd.DataFrame(index=pd.Index(model_genes, name="gene"))
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)


def main() -> None:
    cfg = load_config()
    organoid = cfg["cortical_organoid"]
    stack = cfg["stack"]
    model_dir = ROOT / "models"
    checkpoint = model_dir / stack["checkpoint_name"]
    model_config = read_model_config(checkpoint)
    model_genes = load_gene_list(model_dir / stack["genelist_name"])
    if len(model_genes) != int(model_config["n_genes"]):
        raise RuntimeError("Gene list length does not match checkpoint n_genes")
    n_base, n_test = window_sizes(model_config, stack)
    ages = discover_ages(organoid)
    frames = load_clusters(organoid, ages)
    base_obs, test_obs, selection = select_cells(organoid, frames, n_base, n_test, int(stack["random_seed"]))
    print("organoid selection", json.dumps(selection), flush=True)

    token = renew_token(organoid["study_accession"])
    bucket = resolve_bucket(
        token,
        organoid["terra_namespace"],
        organoid["terra_workspace"],
        DATA / "bucket.txt",
    )
    gene_map, n_source_genes = gene_column_map(DATA / organoid["genes_filename_template"].format(age=ages[0]), model_genes)

    barcode_column = organoid["barcode_column"]
    downloads = []
    for age in ages:
        object_name = organoid["expression_object_template"].format(age=age)
        raw_path = RAW / object_name
        if raw_path.exists():
            continue
        url = f"https://storage.googleapis.com/download/storage/v1/b/{bucket}/o/{quote(object_name, safe='')}?alt=media"
        print(f"downloading {object_name}", flush=True)
        header = raw_path.with_suffix(raw_path.suffix + ".header")
        partial = raw_path.with_suffix(raw_path.suffix + ".partial")
        header.parent.mkdir(parents=True, exist_ok=True)
        header.write_text(f"Authorization: Bearer {token}\n")
        header.chmod(0o600)
        command = ["curl", "-fL", "--retry", "5", "--retry-all-errors"]
        if partial.exists():
            command.extend(["-C", "-"])
        command.extend(["-H", f"@{header}", "-o", str(partial), url])
        downloads.append((age, raw_path, header, partial, subprocess.Popen(command)))
    for age, raw_path, header, partial, process in downloads:
        if process.wait() != 0:
            raise RuntimeError(f"Download failed for {raw_path.name}")
        header.unlink(missing_ok=True)
        partial.replace(raw_path)
        print(f"downloaded {raw_path.name}", flush=True)

    base_paths = []
    test_matrix = None
    for age in ages:
        barcodes = barcode_index(DATA / organoid["barcodes_filename_template"].format(age=age))
        object_name = organoid["expression_object_template"].format(age=age)
        raw_path = RAW / object_name
        gzipped, dim_a, dim_b, _nnz = matrix_header(raw_path)
        n_barcodes = len(barcodes)
        if (dim_a, dim_b) == (n_source_genes, n_barcodes):
            swap = False
        elif (dim_a, dim_b) == (n_barcodes, n_source_genes):
            swap = True
        else:
            raise RuntimeError(
                f"{object_name} dimensions {(dim_a, dim_b)} match neither genes {n_source_genes} nor cells {n_barcodes}"
            )
        age_base = base_obs[base_obs["age"].eq(age)].reset_index(drop=True)
        age_test = test_obs[test_obs["age"].eq(age)].reset_index(drop=True)
        pieces = [("base", age_base)]
        if not age_test.empty:
            pieces.append(("test", age_test))
        cell_map = RAW / f"cells_{age}.tsv"
        with cell_map.open("w") as handle:
            for role, frame in pieces:
                for row_id, barcode in enumerate(frame[barcode_column].astype(str)):
                    source_index = barcodes.get(barcode)
                    if source_index is None:
                        raise RuntimeError(f"{barcode} is not in the {age} barcodes")
                    handle.write(f"{source_index}\t{role}\t{row_id}\n")
        triplets = RAW / f"{age}.triplets.tsv"
        print(f"filtering {object_name}", flush=True)
        filter_matrix(raw_path, gzipped, swap, gene_map, cell_map, triplets)
        shapes = {role: (frame.shape[0], len(model_genes)) for role, frame in pieces}
        matrices = triplets_by_role(triplets, shapes)
        for role, frame in pieces:
            matrix = matrices[role]
            print(f"{age} {role} nnz {matrix.nnz}", flush=True)
            if role == "base":
                out = INPUTS / f"base_{age}.h5ad"
                write_h5ad(frame, matrix, model_genes, barcode_column, out)
                base_paths.append(out)
            else:
                test_matrix = (frame, matrix)

    if test_matrix is None:
        raise RuntimeError("Query cells were not in a downloaded age matrix")
    base_adatas = [ad.read_h5ad(path) for path in base_paths]
    base = ad.concat(base_adatas, axis=0, join="outer")
    base.write_h5ad(INPUTS / "base.h5ad")
    test_frame, test_x = test_matrix
    write_h5ad(test_frame, test_x, model_genes, barcode_column, INPUTS / "test.h5ad")
    manifest = {
        "dataset": "cortical_organoid",
        "study_accession": organoid["study_accession"],
        "bucket": bucket,
        "selection": selection,
        "base_h5ad": str(INPUTS / "base.h5ad"),
        "test_h5ad": str(INPUTS / "test.h5ad"),
        "n_base_cells": int(base.n_obs),
        "n_test_cells": int(test_frame.shape[0]),
        "n_genes": len(model_genes),
    }
    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("organoid inputs ready", INPUTS / "base.h5ad", INPUTS / "test.h5ad", flush=True)


if __name__ == "__main__":
    main()
