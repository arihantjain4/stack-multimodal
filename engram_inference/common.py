"""Shared helpers for the Engram STACK input and generation runs."""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent


def load_config() -> dict:
    with (ROOT / "config.yaml").open() as handle:
        return yaml.safe_load(handle)


def read_model_config(checkpoint: Path) -> dict:
    """Read Lightning hyperparameters without materializing the weights."""
    import torch

    original = torch._utils._rebuild_tensor_v2

    def _rebuild_stub(storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None):
        return None

    torch._utils._rebuild_tensor_v2 = _rebuild_stub

    class _Unpickler(pickle.Unpickler):
        def persistent_load(self, pid):
            return None

    try:
        with zipfile.ZipFile(checkpoint) as archive:
            with archive.open("archive/data.pkl") as handle:
                checkpoint_obj = _Unpickler(handle).load()
    finally:
        torch._utils._rebuild_tensor_v2 = original

    model_config = checkpoint_obj["hyper_parameters"]["model_config"]
    if "n_cells" not in model_config or "n_genes" not in model_config:
        raise KeyError("Checkpoint model_config is missing n_cells or n_genes")
    return model_config


def load_gene_list(path: Path) -> list[str]:
    with path.open("rb") as handle:
        genes = pickle.load(handle)
    return [str(gene) for gene in genes]


def window_sizes(model_config: dict, stack_cfg: dict) -> tuple[int, int]:
    """Cells kept per context split and in the shared query.

    ``n_test`` is the query block of one generation window at ``context_ratio``.
    Earlier MDM steps ask for a larger query block and repeat these cells.
    ``n_base`` is the context block of that same window.
    """
    n_cells = int(model_config["n_cells"])
    prompt = float(stack_cfg["prompt_ratio"])
    context = float(stack_cfg["context_ratio"])
    n_windows = int(stack_cfg["n_query_windows"])
    if n_windows < 1:
        raise ValueError("n_query_windows must be at least 1")
    ratio = prompt + context
    if not 0 < ratio < 1:
        raise ValueError("prompt_ratio + context_ratio must be between 0 and 1")
    n_test = max(1, int(n_cells * (1.0 - ratio)))
    n_base = n_cells - n_test
    return n_base * n_windows, n_test * n_windows


def group_rng(seed: int, key: str) -> np.random.Generator:
    import zlib

    mixed = (int(seed) + zlib.adler32(key.encode())) % (2**32)
    return np.random.default_rng(mixed)


def take(indices: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    indices = np.asarray(indices)
    if indices.size <= n:
        return np.sort(indices)
    chosen = rng.choice(indices, size=n, replace=False)
    return np.sort(chosen)
