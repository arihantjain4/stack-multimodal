"""Window-level forward passes through the unmodified STACK modules.

These follow the same steps as `get_latent_representation` and the masked
pretraining objective (log1p of raw counts, gene_reduction, attention layers,
output_mlp), but expose the intermediate quantities needed for diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WindowOutput:
    embeddings: np.ndarray      # (W, n_hidden*token_dim)
    px_scale: np.ndarray        # (W, n_genes)
    preact_mean: np.ndarray     # (W,) mean of the first Linear output
    preact_std: np.ndarray      # (W,)
    gelu_active: np.ndarray     # (W,) fraction of first-layer units with positive input


def make_windows(n: int, size: int, generator: np.random.Generator) -> list[np.ndarray]:
    """Shuffle profiles into windows; the last window wraps so every window is full."""
    order = generator.permutation(n)
    size = min(size, n)
    windows = []
    for start in range(0, n, size):
        window = order[start : start + size]
        if window.size < size:
            window = np.concatenate([window, order[: size - window.size]])
        windows.append(window)
    return windows


def encode(model, log_features: torch.Tensor):
    """log_features: (1, W, G). Returns tokens after gene_reduction plus pre-activation."""
    linear = model.gene_reduction[0]
    pre = linear(log_features)
    tokens = model.gene_reduction[2](model.gene_reduction[1](pre))
    batch, n_cells, _ = log_features.shape
    return tokens.reshape(batch, n_cells, model.n_hidden, model.token_dim), pre


@torch.no_grad()
def run_window(model, counts: np.ndarray, gene_mask: np.ndarray | None = None, transform=None) -> WindowOutput:
    """counts: (W, G) raw counts. gene_mask: (G,) bool, genes hidden from the model.

    `transform`, when given, maps (counts tensor, log features tensor) to new log
    features; it is how adapters plug in without changing STACK itself.
    """
    x = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32)).unsqueeze(0)
    log_x = torch.log1p(x)
    if transform is not None:
        log_x = transform(x, log_x)
    if gene_mask is not None:
        log_x = log_x.masked_fill(torch.from_numpy(gene_mask)[None, None, :], 0.0)
    tokens, pre = encode(model, log_x)
    hidden = model._run_attention_layers(tokens)
    emb = hidden.reshape(1, x.shape[1], -1)
    lib = x.sum(-1, keepdim=True).clamp(min=1.0)
    _, _, px_scale = model._compute_nb_parameters(emb, lib)
    return WindowOutput(
        embeddings=emb[0].numpy(),
        px_scale=px_scale[0].numpy(),
        preact_mean=pre[0].mean(-1).numpy(),
        preact_std=pre[0].std(-1).numpy(),
        gelu_active=(pre[0] > 0).float().mean(-1).numpy(),
    )


def masked_log_cpm(values: np.ndarray, genes: np.ndarray) -> np.ndarray:
    sub = values[:, genes]
    total = sub.sum(axis=1, keepdims=True)
    return np.log1p(1e6 * sub / np.maximum(total, 1e-12))


def row_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    denom = np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return np.where(denom > 0, (a * b).sum(1) / np.maximum(denom, 1e-12), np.nan)


def loo_mean(fractions: np.ndarray) -> np.ndarray:
    """Leave-one-out mean of normalized profiles within a window."""
    n = fractions.shape[0]
    if n == 1:
        return fractions.copy()
    total = fractions.sum(axis=0, keepdims=True)
    return (total - fractions) / (n - 1)
