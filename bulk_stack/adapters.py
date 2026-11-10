"""Ways to run STACK on bulk profiles without touching src/stack.

Every variant is a *runner*: a callable (counts (W, G), hidden-gene mask or None)
-> WindowOutput, so evaluate.py scores all of them with one benchmark.

Training-free:
  depth_scale       rescale counts to a single-cell-like library before log1p
  thin_ensemble     K binomial thinnings to that library, outputs averaged
  pseudocell        K thinnings of each sample placed in one window as a
                    population of "cells", outputs averaged per sample
  context_decoder   decode relative to the window's peers:
                    p ∝ C ⊙ (px / C̃)^β, C = leave-one-out mean of observed
                    fractions, C̃ = leave-one-out mean of px (or pool priors)
Trained (frozen backbone):
  BulkAdapter       (first attempt) depth-normalized input + per-gene affine +
                    depth FiLM on the first-layer pre-activation + LoRA on output_mlp
  BulkHead          residual bulk output head on frozen logits/embeddings with a
                    per-gene bulk NB dispersion and learned context terms
  SideChannel       full-depth standardized log-CPM fed through a low-rank,
                    depth-gated path into the tokens, plus depth FiLM, + BulkHead
Model-free control:
  PCAImputer        low-rank linear imputation fitted on the same bulk pool
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stack_ops import WindowOutput, run_window


def thin_counts(counts: np.ndarray, target: float, generator: np.random.Generator) -> np.ndarray:
    counts = np.rint(counts).astype(np.int64)
    totals = counts.sum(axis=1)
    p = np.clip(target / np.maximum(totals, 1), 0.0, 1.0)
    return generator.binomial(counts, p[:, None]).astype(np.float32)


def depth_scale_transform(target: float):
    def transform(x: torch.Tensor, log_x: torch.Tensor) -> torch.Tensor:
        lib = x.sum(-1, keepdim=True).clamp(min=1.0)
        factor = torch.clamp(target / lib, max=1.0)
        return torch.log1p(x * factor)

    return transform


def average(outputs: list[WindowOutput]) -> WindowOutput:
    return WindowOutput(
        embeddings=np.mean([o.embeddings for o in outputs], axis=0),
        px_scale=np.mean([o.px_scale for o in outputs], axis=0),
        preact_mean=np.mean([o.preact_mean for o in outputs], axis=0),
        preact_std=np.mean([o.preact_std for o in outputs], axis=0),
        gelu_active=np.mean([o.gelu_active for o in outputs], axis=0),
    )


def depth_scale_runner(model, target: float):
    transform = depth_scale_transform(target)
    return lambda counts, mask: run_window(model, counts, mask, transform=transform)


def thin_ensemble_runner(model, target: float, draws: int, generator: np.random.Generator):
    def runner(counts, mask):
        if counts.sum(axis=1).max() <= target:
            return run_window(model, counts, mask)
        return average([run_window(model, thin_counts(counts, target, generator), mask) for _ in range(draws)])

    return runner


def pseudocell_runner(model, target: float, draws: int, generator: np.random.Generator):
    def runner(counts, mask):
        if counts.sum(axis=1).max() <= target:
            return run_window(model, counts, mask)
        n = counts.shape[0]
        stacked = np.concatenate([thin_counts(counts, target, generator) for _ in range(draws)], axis=0)
        out = run_window(model, stacked, mask)
        split = lambda a: a.reshape(draws, n, *a.shape[1:]).mean(axis=0)
        return WindowOutput(
            embeddings=split(out.embeddings),
            px_scale=split(out.px_scale),
            preact_mean=split(out.preact_mean),
            preact_std=split(out.preact_std),
            gelu_active=split(out.gelu_active),
        )

    return runner


class LoRALinear(nn.Module):
    """y = base(x) + (alpha / rank) * B(A(x)); B starts at zero so the start is the checkpoint."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / float(rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.up(self.down(x))


class BulkAdapter(nn.Module):
    """Trainable pieces around a frozen STACK.

    input:  log1p(x * min(1, target / lib)) * (1 + gene_scale) + gene_shift
    first layer: pre-activation + film(log10 lib) (zero-initialized MLP)
    output: LoRA on the last output_mlp Linear (gene logits and dispersions)
    """

    def __init__(self, model, n_genes: int, target_depth: float, film_hidden: int, lora_rank: int, lora_alpha: float) -> None:
        super().__init__()
        self.model = model
        self.target_depth = float(target_depth)
        self.gene_scale = nn.Parameter(torch.zeros(n_genes))
        self.gene_shift = nn.Parameter(torch.zeros(n_genes))
        width = model.gene_reduction[0].out_features
        self.film = nn.Sequential(nn.Linear(1, film_hidden), nn.GELU(), nn.Linear(film_hidden, 2 * width))
        nn.init.zeros_(self.film[2].weight)
        nn.init.zeros_(self.film[2].bias)
        last = model.output_mlp[3]
        self.output_lora = LoRALinear(last, lora_rank, lora_alpha)

    def trainable(self):
        return [p for name, p in self.named_parameters() if not name.startswith("model.") and p.requires_grad]

    def forward(self, counts: torch.Tensor, hidden: torch.Tensor | None = None):
        """counts: (B, W, G) raw counts. hidden: (G,) or (B, W, G) bool of genes hidden from the model."""
        model = self.model
        lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
        factor = torch.clamp(self.target_depth / lib, max=1.0)
        log_x = torch.log1p(counts * factor) * (1.0 + self.gene_scale) + self.gene_shift
        if hidden is not None:
            log_x = log_x.masked_fill(hidden if hidden.dim() == 3 else hidden[None, None, :], 0.0)
        linear = model.gene_reduction[0]
        pre = linear(log_x)
        gamma, beta = self.film(torch.log10(lib)).chunk(2, dim=-1)
        pre = pre * (1.0 + gamma) + beta
        tokens = model.gene_reduction[2](model.gene_reduction[1](pre))
        batch, n_cells, _ = counts.shape
        tokens = tokens.reshape(batch, n_cells, model.n_hidden, model.token_dim)
        hidden_states = model._run_attention_layers(tokens)
        emb = hidden_states.reshape(batch, n_cells, -1)
        mlp = model.output_mlp
        h = mlp[2](mlp[1](mlp[0](emb)))
        out = self.output_lora(h).reshape(batch, n_cells, model.n_genes, 2)
        logits = out[..., 0]
        return {"embeddings": emb, "logits": logits, "px_scale": F.softmax(logits, dim=-1), "pre": pre}

    @torch.no_grad()
    def run_window(self, counts: np.ndarray, mask: np.ndarray | None) -> WindowOutput:
        x = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32)).unsqueeze(0)
        hidden = None if mask is None else torch.from_numpy(mask)
        out = self.forward(x, hidden)
        pre = out["pre"][0]
        return WindowOutput(
            embeddings=out["embeddings"][0].numpy(),
            px_scale=out["px_scale"][0].numpy(),
            preact_mean=pre.mean(-1).numpy(),
            preact_std=pre.std(-1).numpy(),
            gelu_active=(pre > 0).float().mean(-1).numpy(),
        )


def loo(values: np.ndarray) -> np.ndarray:
    n = values.shape[0]
    if n == 1:
        return values.copy()
    return (values.sum(axis=0, keepdims=True) - values) / (n - 1)


def fractions(counts: np.ndarray) -> np.ndarray:
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)


def context_terms(counts: np.ndarray, px: np.ndarray, prior: str, pool_prior: dict | None):
    """Peer context C (observed fractions) and C̃ (model px) for each profile."""
    if prior == "window_loo":
        return loo(fractions(counts)), loo(px)
    if prior == "training_pool":
        n = counts.shape[0]
        return np.repeat(pool_prior["fraction"][None], n, 0), np.repeat(pool_prior["px"][None], n, 0)
    raise ValueError(f"Unknown context prior {prior!r}")


def context_decode(px: np.ndarray, c_obs: np.ndarray, c_model: np.ndarray, beta: float, eps: np.ndarray) -> np.ndarray:
    log_p = np.log(c_obs + eps) + beta * (np.log(px + eps) - np.log(c_model + eps))
    log_p -= log_p.max(axis=1, keepdims=True)
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def context_decoder_runner(base, beta: float, prior: str, pool_prior: dict | None = None):
    def runner(counts, mask):
        out = base(counts, mask)
        c_obs, c_model = context_terms(counts, out.px_scale, prior, pool_prior)
        eps = 1.0 / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
        out.px_scale = context_decode(out.px_scale, c_obs, c_model, beta, eps)
        return out

    return runner


def nb_nll(y: torch.Tensor, mu: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    log_theta_mu = torch.log(theta + mu + 1e-8)
    return -(
        theta * (torch.log(theta + 1e-8) - log_theta_mu)
        + y * (torch.log(mu + 1e-8) - log_theta_mu)
        + torch.lgamma(y + theta)
        - torch.lgamma(theta)
        - torch.lgamma(y + 1.0)
    )


class BulkHead(nn.Module):
    """logits' = (1+τ)·logits + δ + U·GELU(V·e) + α⊙log(C+ε) + γ⊙log(C̃+ε); NB(θ_g) for bulk counts.

    Everything that changes the output starts at zero, so the untrained head is the checkpoint.
    """

    def __init__(self, n_genes: int, emb_dim: int, rank: int, init_dispersion: torch.Tensor) -> None:
        super().__init__()
        self.tau = nn.Parameter(torch.zeros(1))
        self.delta = nn.Parameter(torch.zeros(n_genes))
        self.down = nn.Linear(emb_dim, rank)
        self.up = nn.Linear(rank, n_genes, bias=False)
        nn.init.zeros_(self.up.weight)
        self.alpha = nn.Parameter(torch.zeros(n_genes))
        self.gamma = nn.Parameter(torch.zeros(n_genes))
        self.log_theta = nn.Parameter(torch.log(init_dispersion.clone()))

    def forward(self, logits, emb, c_obs, c_model, eps):
        out = (1.0 + self.tau) * logits + self.delta + self.up(F.gelu(self.down(emb)))
        out = out + self.alpha * torch.log(c_obs + eps) + self.gamma * torch.log(c_model + eps)
        return out

    def theta(self) -> torch.Tensor:
        return torch.exp(self.log_theta)


def torch_loo(values: torch.Tensor) -> torch.Tensor:
    """values: (B, W, G) -> leave-one-out mean over W."""
    n = values.shape[1]
    if n == 1:
        return values
    return (values.sum(dim=1, keepdim=True) - values) / (n - 1)


def head_inputs(counts: torch.Tensor, px: torch.Tensor):
    lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
    return torch_loo(counts / lib), torch_loo(px), 1.0 / lib


def frozen_logits(model, emb: torch.Tensor) -> torch.Tensor:
    mlp = model.output_mlp
    out = mlp(emb).reshape(*emb.shape[:-1], model.n_genes, 2)
    return out[..., 0]


def harmonized_log(counts: torch.Tensor, target: float) -> torch.Tensor:
    lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
    return torch.log1p(counts * torch.clamp(target / lib, max=1.0))


def backbone(model, log_x: torch.Tensor, token_delta=None, film=None):
    """gene_reduction -> (FiLM) -> GELU -> (+ token delta) -> attention layers. Returns (emb, pre)."""
    pre = model.gene_reduction[0](log_x)
    if film is not None:
        pre = film(pre)
    tokens = model.gene_reduction[2](model.gene_reduction[1](pre))
    if token_delta is not None:
        tokens = tokens + token_delta
    batch, n_cells, _ = log_x.shape
    hidden = model._run_attention_layers(tokens.reshape(batch, n_cells, model.n_hidden, model.token_dim))
    return hidden.reshape(batch, n_cells, -1), pre


class SideChannel(nn.Module):
    """Harmonized input for the backbone plus a depth-gated low-rank path for the full-depth profile.

    z = (log1p(CPM) - μ_g) / σ_g from the training pool, hidden genes zeroed, and a
    hidden-gene indicator; Δ = U·GELU(V·[z, m]) is added to the tokens with gate
    g(s) = sigmoid(k (s - s0)), s = log(lib / D), so single cells (s ≈ 0) pass through.
    Depth FiLM: pre' = pre ⊙ (1 + a s) + b s.
    """

    def __init__(self, model, head: BulkHead, target_depth: float, gene_mean: torch.Tensor, gene_std: torch.Tensor,
                 rank: int, gate_slope: float, gate_center: float) -> None:
        super().__init__()
        self.model = model
        self.head = head
        self.target_depth = float(target_depth)
        self.register_buffer("gene_mean", gene_mean)
        self.register_buffer("gene_std", gene_std)
        n_genes = gene_mean.numel()
        width = model.gene_reduction[0].out_features
        self.down = nn.Linear(2 * n_genes, rank)
        self.up = nn.Linear(rank, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.film_scale = nn.Parameter(torch.zeros(width))
        self.film_shift = nn.Parameter(torch.zeros(width))
        self.gate_slope = nn.Parameter(torch.tensor(float(gate_slope)))
        self.gate_center = nn.Parameter(torch.tensor(float(gate_center)))

    def trainable(self):
        return [p for name, p in self.named_parameters() if not name.startswith("model.")]

    def forward(self, counts: torch.Tensor, hidden: torch.Tensor | None):
        model = self.model
        lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
        s = torch.log(lib / self.target_depth).clamp(min=0.0)
        hidden_mask = torch.zeros(counts.shape[-1], dtype=torch.bool) if hidden is None else hidden
        log_x = harmonized_log(counts, self.target_depth).masked_fill(hidden_mask[None, None, :], 0.0)
        visible_lib = (counts * ~hidden_mask).sum(-1, keepdim=True).clamp(min=1.0)
        z = (torch.log1p(1e6 * counts / visible_lib) - self.gene_mean) / self.gene_std
        z = z.masked_fill(hidden_mask[None, None, :], 0.0)
        indicator = hidden_mask.float().expand_as(z)
        gate = torch.sigmoid(self.gate_slope * (s - self.gate_center))
        delta = gate * self.up(F.gelu(self.down(torch.cat([z, indicator], dim=-1))))
        film = lambda pre: pre * (1.0 + self.film_scale * s) + self.film_shift * s
        emb, pre = backbone(model, log_x, token_delta=delta, film=film)
        logits = frozen_logits(model, emb)
        px = F.softmax(logits, dim=-1)
        c_obs, c_model, eps = head_inputs(counts, px)
        return {"embeddings": emb, "pre": pre, "logits": self.head(logits, emb, c_obs, c_model, eps)}

    @torch.no_grad()
    def run_window(self, counts: np.ndarray, mask: np.ndarray | None) -> WindowOutput:
        x = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32)).unsqueeze(0)
        out = self.forward(x, None if mask is None else torch.from_numpy(mask))
        pre = out["pre"][0]
        return WindowOutput(
            embeddings=out["embeddings"][0].numpy(),
            px_scale=F.softmax(out["logits"][0], dim=-1).numpy(),
            preact_mean=pre.mean(-1).numpy(),
            preact_std=pre.std(-1).numpy(),
            gelu_active=(pre > 0).float().mean(-1).numpy(),
        )


def bulk_head_runner(model, head: BulkHead, target_depth: float):
    @torch.no_grad()
    def runner(counts, mask):
        x = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.float32)).unsqueeze(0)
        log_x = harmonized_log(x, target_depth)
        if mask is not None:
            log_x = log_x.masked_fill(torch.from_numpy(mask)[None, None, :], 0.0)
        emb, pre = backbone(model, log_x)
        logits = frozen_logits(model, emb)
        c_obs, c_model, eps = head_inputs(x, F.softmax(logits, dim=-1))
        out = head(logits, emb, c_obs, c_model, eps)
        return WindowOutput(
            embeddings=emb[0].numpy(),
            px_scale=F.softmax(out[0], dim=-1).numpy(),
            preact_mean=pre[0].mean(-1).numpy(),
            preact_std=pre[0].std(-1).numpy(),
            gelu_active=(pre[0] > 0).float().mean(-1).numpy(),
        )

    return runner


class PCAImputer:
    """Model-free control: ridge fit of visible log-CPM on pool principal axes, then reconstruct."""

    def __init__(self, mean: np.ndarray, loadings: np.ndarray, ridge: float) -> None:
        self.mean = mean
        self.loadings = loadings  # (k, G)
        self.ridge = float(ridge)

    def run_window(self, counts: np.ndarray, mask: np.ndarray | None) -> WindowOutput:
        visible = np.ones(counts.shape[1], dtype=bool) if mask is None else ~mask
        lib = np.maximum(counts[:, visible].sum(axis=1, keepdims=True), 1.0)
        y = np.log1p(1e6 * counts[:, visible] / lib) - self.mean[visible]
        basis = self.loadings[:, visible]
        gram = basis @ basis.T + self.ridge * np.eye(basis.shape[0])
        coef = np.linalg.solve(gram, basis @ y.T).T
        recon = np.clip(np.expm1(self.mean + coef @ self.loadings), 0.0, None)
        px = recon / np.maximum(recon.sum(axis=1, keepdims=True), 1e-12)
        zeros = np.zeros(counts.shape[0])
        return WindowOutput(embeddings=coef.astype(np.float32), px_scale=px, preact_mean=zeros, preact_std=zeros, gelu_active=zeros)
