"""Shared paths, config loading, and model loading for the bulk investigation."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def load_config() -> dict:
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def path(cfg: dict, key: str) -> Path:
    target = Path(cfg["paths"][key])
    if not target.is_absolute():
        target = ROOT / target
    target.mkdir(parents=True, exist_ok=True)
    return target


def set_threads() -> None:
    threads = str(os.cpu_count() or 1)
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(variable, threads)
    import torch

    torch.set_num_threads(int(threads))


def load_gene_list(cfg: dict) -> list[str]:
    with (path(cfg, "models") / cfg["stack"]["genelist_name"]).open("rb") as handle:
        return [str(gene) for gene in pickle.load(handle)]


def load_stack(cfg: dict):
    """Load the published checkpoint into the unmodified STACK class."""
    from stack.model_loading import load_model_from_checkpoint

    model = load_model_from_checkpoint(
        str(path(cfg, "models") / cfg["stack"]["checkpoint_name"]),
        model_class="ICL_FinetunedModel",
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def engram_module(name: str):
    """Import a module from engram_inference, whose own `common` shadows ours."""
    import importlib

    ours = sys.modules.pop("common", None)
    folder = str(REPO / "engram_inference")
    sys.path.insert(0, folder)
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.remove(folder)
        sys.modules.pop("common", None)
        if ours is not None:
            sys.modules["common"] = ours
    return module


def symbol_matrix(symbols, model_genes: list[str]):
    """Sparse (n_source, n_model) matrix that sums source columns onto model genes by symbol."""
    from scipy import sparse

    position = {}
    for index, gene in enumerate(model_genes):
        position.setdefault(gene.upper(), index)
    rows, cols = [], []
    for source_index, symbol in enumerate(symbols):
        if symbol is None:
            continue
        target = position.get(str(symbol).upper())
        if target is not None:
            rows.append(source_index)
            cols.append(target)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(symbols), len(model_genes))
    )


def rng(cfg: dict, offset: int = 0) -> np.random.Generator:
    return np.random.default_rng(int(cfg["stack"]["random_seed"]) + offset)
