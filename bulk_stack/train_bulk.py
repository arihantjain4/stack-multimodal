"""Fit the brainstorm shortlist on the held-out ARCHS4 bulk pool (backbone frozen).

Steps, each saved under results/:
  pool_stats.npz      per-gene log-CPM mean/std and the pool prior fractions
  context.json        β of the context decoder, chosen on pool windows
  bulk_head.pt        BulkHead trained on cached embeddings of harmonized input
  side_channel.pt     SideChannel (+ BulkHead) trained end-to-end through the frozen backbone
  pca_imputer.npz     model-free control fitted on the same pool rows

Usage: python train_bulk.py [step ...]  (default: all steps in order)
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from adapters import (BulkHead, SideChannel, backbone, context_decode, frozen_logits, harmonized_log, head_inputs,
                      loo, fractions, nb_nll, thin_counts)
from common import load_config, load_stack, path, rng, set_threads
from stack_ops import masked_log_cpm, row_pearson
from train_adapter import load_pool, reference_depth

STEPS = ["stats", "context", "head", "side_channel", "pca"]


def split_pool(cfg: dict, pool: np.ndarray):
    spec = cfg["adapter"]["pool"]
    order = rng(cfg, 81).permutation(pool.shape[0])
    n_val = max(1, int(round(pool.shape[0] * float(spec["validation_fraction"]))))
    return order[n_val:], order[:n_val]


def log_cpm(counts: np.ndarray) -> np.ndarray:
    return np.log1p(1e6 * fractions(counts))


class WindowSampler:
    """Windows of pool samples: an anchor plus its nearest neighbours (by log-CPM correlation), or random."""

    def __init__(self, pool: np.ndarray, rows: np.ndarray, size: int, composition: str, generator) -> None:
        self.pool, self.rows, self.size, self.generator = pool, rows, min(size, rows.size), generator
        self.composition = composition
        if composition == "nearest_neighbors":
            self.neighbours = np.argsort(-np.corrcoef(log_cpm(pool[rows])), axis=1)
        elif composition != "random":
            raise ValueError(f"Unknown window composition {composition!r}")

    def __call__(self) -> np.ndarray:
        if self.composition == "random":
            return self.generator.choice(self.rows, self.size, replace=False)
        anchor = self.generator.integers(self.rows.size)
        return self.rows[self.neighbours[anchor, : self.size]]


def depth_view(counts: np.ndarray, depth: float, spec: dict, generator, native: bool) -> np.ndarray:
    if native:
        return counts
    lib = counts.sum(1)
    low = np.log(depth * float(spec["min_depth_multiplier"]))
    targets = np.exp(generator.uniform(low, np.log(np.maximum(lib, np.exp(low) + 1.0))))
    return np.vstack([thin_counts(counts[i : i + 1], targets[i], generator) for i in range(len(counts))])


def hidden_mask(n_genes: int, spec: dict, generator) -> np.ndarray:
    rate = generator.uniform(float(spec["mask_fraction_min"]), float(spec["mask_fraction_max"]))
    mask = np.zeros(n_genes, dtype=bool)
    mask[generator.choice(n_genes, int(rate * n_genes), replace=False)] = True
    return mask


def bulk_loss(logits: torch.Tensor, theta: torch.Tensor, counts: torch.Tensor, hidden: torch.Tensor, spec: dict) -> dict:
    lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
    frac = counts / lib
    expressed = frac.mean(dim=(0, 1)) >= float(spec["min_cpm_for_loss"]) / 1e6
    genes = hidden & expressed
    if int(genes.sum()) < 2:
        zero = logits.sum() * 0.0
        return {"loss": zero, "nb": zero, "corr": zero}
    p = F.softmax(logits, dim=-1)
    y = counts[..., genes]
    nll = nb_nll(y, p[..., genes] * lib, theta[genes]).mean()
    prior = ((frac.sum(1, keepdim=True) - frac) / max(frac.shape[1] - 1, 1))[..., genes]
    reference = nb_nll(y, prior * lib + 1e-8, theta[genes]).mean().detach()
    pred_sub = F.softmax(logits[..., genes], dim=-1)
    truth_sub = y / y.sum(-1, keepdim=True).clamp(min=1.0)
    a = torch.log1p(1e6 * pred_sub)
    b = torch.log1p(1e6 * truth_sub)
    a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
    corr = ((a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-8)).mean()
    loss = float(spec["nb_weight"]) * nll / reference.clamp(min=1e-8) + float(spec["corr_weight"]) * (1.0 - corr)
    return {"loss": loss, "nb": nll / reference.clamp(min=1e-8), "corr": corr}


def init_dispersion(cfg: dict, pool: np.ndarray, rows: np.ndarray) -> torch.Tensor:
    spec = cfg["adapter"]["head"]
    counts = pool[rows]
    scaled = fractions(counts) * np.median(counts.sum(1))
    mean, var = scaled.mean(0), scaled.var(0)
    excess = np.maximum(var - mean, 1e-12)
    theta = np.clip(mean**2 / excess, float(spec["dispersion_min"]), float(spec["dispersion_max"]))
    return torch.from_numpy(theta.astype(np.float32))


def step_stats(cfg, pool, train_rows, val_rows, model, depth):
    lc = log_cpm(pool[train_rows])
    std = lc.std(0)
    std = np.maximum(std, np.median(std))
    out = path(cfg, "results") / "pool_stats.npz"
    np.savez(out, gene_mean=lc.mean(0).astype(np.float32), gene_std=std.astype(np.float32),
             fraction=fractions(pool[train_rows]).mean(0), train_rows=train_rows, val_rows=val_rows)
    print("pool stats", out, flush=True)


def step_context(cfg, pool, train_rows, val_rows, model, depth):
    spec = cfg["adapter"]["context"]
    head = cfg["adapter"]["head"]
    generator = rng(cfg, 82)
    target = depth * float(head["depth_multiplier"])
    sampler = WindowSampler(pool, train_rows, int(cfg["adapter"]["pool"]["window_size"]), cfg["adapter"]["pool"]["window_composition"], generator)
    grid = [float(b) for b in spec["beta_grid"]]
    scores = {b: [] for b in grid}
    pool_px = []
    floor = float(cfg["benchmark"]["min_cpm_for_metric"])
    for _ in range(int(spec["n_windows"])):
        counts = pool[sampler()]
        mask = hidden_mask(pool.shape[1], cfg["adapter"]["head"], generator)
        x = torch.from_numpy(counts).unsqueeze(0)
        with torch.no_grad():
            log_x = harmonized_log(x, target).masked_fill(torch.from_numpy(mask)[None, None, :], 0.0)
            emb, _ = backbone(model, log_x)
            px = F.softmax(frozen_logits(model, emb), dim=-1)[0].numpy()
        pool_px.append(px.mean(0))
        expressed = (1e6 * fractions(counts)).mean(0) >= floor
        genes = np.flatnonzero(mask & expressed)
        truth = masked_log_cpm(counts, genes)
        eps = 1.0 / counts.sum(1, keepdims=True)
        for b in grid:
            pred = context_decode(px, loo(fractions(counts)), loo(px), b, eps)
            scores[b].append(np.nanmean(row_pearson(masked_log_cpm(pred, genes), truth)))
    summary = {str(b): float(np.mean(v)) for b, v in scores.items()}
    best = max(grid, key=lambda b: np.mean(scores[b]))
    record = {"beta": best, "scores": summary, "depth_multiplier": float(head["depth_multiplier"])}
    (path(cfg, "results") / "context.json").write_text(json.dumps(record, indent=2) + "\n")
    stats = dict(np.load(path(cfg, "results") / "pool_stats.npz"))
    stats["px"] = np.mean(pool_px, axis=0)
    np.savez(path(cfg, "results") / "pool_stats.npz", **stats)
    print("context", json.dumps(record), flush=True)


def build_cache(cfg, pool, rows, model, depth, n_windows, generator):
    spec = cfg["adapter"]["head"]
    target = depth * float(spec["depth_multiplier"])
    sampler = WindowSampler(pool, rows, int(cfg["adapter"]["pool"]["window_size"]), cfg["adapter"]["pool"]["window_composition"], generator)
    cache = []
    for _ in range(n_windows):
        native = pool[sampler()]
        for view in range(1 + int(spec["n_depth_views"])):
            counts = depth_view(native, depth, cfg["adapter"]["pool"], generator, native=view == 0)
            for _ in range(int(spec["n_mask_draws"])):
                mask = hidden_mask(pool.shape[1], spec, generator)
                x = torch.from_numpy(counts).unsqueeze(0)
                with torch.no_grad():
                    log_x = harmonized_log(x, target).masked_fill(torch.from_numpy(mask)[None, None, :], 0.0)
                    emb, _ = backbone(model, log_x)
                    logits = frozen_logits(model, emb)
                cache.append((x, torch.from_numpy(mask), emb, logits))
    return cache


def head_step(head, item, spec):
    x, mask, emb, logits = item
    c_obs, c_model, eps = head_inputs(x, F.softmax(logits, dim=-1))
    return bulk_loss(head(logits, emb, c_obs, c_model, eps), head.theta(), x, mask, spec)


def make_head(cfg, model, pool, train_rows) -> BulkHead:
    spec = cfg["adapter"]["head"]
    return BulkHead(model.n_genes, model.n_hidden * model.token_dim, int(spec["rank"]), init_dispersion(cfg, pool, train_rows))


def step_head(cfg, pool, train_rows, val_rows, model, depth):
    spec = cfg["adapter"]["head"]
    generator = rng(cfg, 83)
    start = time.time()
    train_cache = build_cache(cfg, pool, train_rows, model, depth, int(spec["n_train_windows"]), generator)
    val_cache = build_cache(cfg, pool, val_rows, model, depth, int(spec["n_val_windows"]), generator)
    print(f"cached {len(train_cache)} train / {len(val_cache)} val windows in {time.time() - start:.0f}s", flush=True)
    head = make_head(cfg, model, pool, train_rows)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(spec["learning_rate"]), weight_decay=float(spec["weight_decay"]))
    history = []
    for epoch in range(1, int(spec["epochs"]) + 1):
        order = generator.permutation(len(train_cache))
        losses = []
        for index in order:
            terms = head_step(head, train_cache[index], spec)
            optimizer.zero_grad()
            terms["loss"].backward()
            optimizer.step()
            losses.append({k: float(v) for k, v in terms.items()})
        with torch.no_grad():
            val = [{k: float(v) for k, v in head_step(head, item, spec).items()} for item in val_cache]
        record = {"epoch": epoch, **{f"train_{k}": float(np.mean([l[k] for l in losses])) for k in losses[0]},
                  **{f"val_{k}": float(np.mean([v[k] for v in val])) for k in val[0]}}
        history.append(record)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    torch.save({"state": head.state_dict(), "reference_depth": depth, "spec": spec}, path(cfg, "results") / "bulk_head.pt")
    (path(cfg, "results") / "bulk_head_history.json").write_text(json.dumps(history) + "\n")


def make_side_channel(cfg, model, head, depth) -> SideChannel:
    spec = cfg["adapter"]["side_channel"]
    stats = np.load(path(cfg, "results") / "pool_stats.npz")
    return SideChannel(
        model, head,
        target_depth=depth * float(cfg["adapter"]["head"]["depth_multiplier"]),
        gene_mean=torch.from_numpy(stats["gene_mean"]),
        gene_std=torch.from_numpy(stats["gene_std"]),
        rank=int(spec["rank"]),
        gate_slope=float(spec["gate_slope"]),
        gate_center=float(np.log(float(spec["gate_center_multiplier"]))),
    )


def step_side_channel(cfg, pool, train_rows, val_rows, model, depth):
    spec = cfg["adapter"]["side_channel"]
    head_spec = cfg["adapter"]["head"]
    generator = rng(cfg, 84)
    head = make_head(cfg, model, pool, train_rows)
    head.load_state_dict(torch.load(path(cfg, "results") / "bulk_head.pt")["state"])
    adapter = make_side_channel(cfg, model, head, depth)
    head_params = list(head.parameters())
    own = [p for name, p in adapter.named_parameters() if not name.startswith(("model.", "head."))]
    optimizer = torch.optim.AdamW(
        [{"params": own, "lr": float(spec["learning_rate"])}, {"params": head_params, "lr": float(spec["head_learning_rate"])}],
        weight_decay=float(spec["weight_decay"]),
    )
    steps = int(spec["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    size = int(cfg["adapter"]["pool"]["window_size"])
    composition = cfg["adapter"]["pool"]["window_composition"]
    sampler = WindowSampler(pool, train_rows, size, composition, generator)
    val_generator = rng(cfg, 85)
    val_sampler = WindowSampler(pool, val_rows, size, composition, val_generator)
    val_items = []
    for _ in range(int(spec["n_val_windows"])):
        native = pool[val_sampler()]
        counts = depth_view(native, depth, cfg["adapter"]["pool"], val_generator, native=bool(val_generator.integers(2)))
        val_items.append((torch.from_numpy(counts).unsqueeze(0), torch.from_numpy(hidden_mask(pool.shape[1], head_spec, val_generator))))
    history = []
    start = time.time()
    for step in range(1, steps + 1):
        adapter.train()
        native = pool[sampler()]
        counts = depth_view(native, depth, cfg["adapter"]["pool"], generator, native=bool(generator.integers(2)))
        x = torch.from_numpy(counts).unsqueeze(0)
        mask = torch.from_numpy(hidden_mask(pool.shape[1], head_spec, generator))
        terms = bulk_loss(adapter(x, mask)["logits"], head.theta(), x, mask, head_spec)
        optimizer.zero_grad()
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(adapter.trainable(), float(spec["grad_clip"]))
        optimizer.step()
        scheduler.step()
        record = {"step": step, **{k: float(v.detach()) for k, v in terms.items()}, "seconds": round(time.time() - start, 1)}
        if step % int(spec["validate_every"]) == 0 or step == 1 or step == steps:
            adapter.eval()
            with torch.no_grad():
                val = [bulk_loss(adapter(c, m)["logits"], head.theta(), c, m, head_spec) for c, m in val_items]
            record.update({f"val_{k}": float(np.mean([float(v[k]) for v in val])) for k in val[0]})
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
        history.append(record)
    state = {k: v for k, v in adapter.state_dict().items() if not k.startswith("model.")}
    torch.save({"state": state, "reference_depth": depth}, path(cfg, "results") / "side_channel.pt")
    (path(cfg, "results") / "side_channel_history.json").write_text(json.dumps(history) + "\n")


def step_pca(cfg, pool, train_rows, val_rows, model, depth):
    spec = cfg["adapter"]["pca_imputer"]
    lc = log_cpm(pool[train_rows])
    mean = lc.mean(0)
    _, _, vt = np.linalg.svd(lc - mean, full_matrices=False)
    k = min(int(spec["n_components"]), vt.shape[0])
    np.savez(path(cfg, "results") / "pca_imputer.npz", mean=mean, loadings=vt[:k], ridge=float(spec["ridge"]))
    print("pca imputer components", k, flush=True)


def main(steps: list[str]) -> None:
    set_threads()
    cfg = load_config()
    torch.manual_seed(int(cfg["stack"]["random_seed"]))
    depth = reference_depth(cfg)
    pool = load_pool(cfg)
    train_rows, val_rows = split_pool(cfg, pool)
    model = load_stack(cfg)
    for name in steps or STEPS:
        print(f"== {name}", flush=True)
        globals()[f"step_{name}"](cfg, pool, train_rows, val_rows, model, depth)


if __name__ == "__main__":
    main(sys.argv[1:])
