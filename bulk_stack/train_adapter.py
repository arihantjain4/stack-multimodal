"""Train the first adapter attempt (BulkAdapter: FiLM + LoRA) on real ARCHS4 bulk samples with the STACK backbone frozen.

Objective: the pretraining task (predict genes hidden in log space) on bulk
profiles thinned to random depths, so the adapter has to work from single-cell
depth up to native bulk depth.
"""

from __future__ import annotations

import json
import time

import anndata as ad
import numpy as np
import torch
from scipy import sparse

from adapters import BulkAdapter, thin_counts
from common import load_config, load_gene_list, load_stack, path, rng, set_threads, symbol_matrix


def reference_depth(cfg: dict) -> float:
    setting = cfg["adapter"]["reference_depth"]
    if setting != "auto":
        return float(setting)
    obs = ad.read_h5ad(path(cfg, "data") / "profiles.h5ad", backed="r").obs
    return float(np.median(obs.loc[obs["kind"].eq("cell"), "library_size"]))


def load_pool(cfg: dict) -> np.ndarray:
    pool = ad.read_h5ad(path(cfg, "data") / cfg["adapter"]["pool"]["file"])
    mapper = symbol_matrix(pool.var["symbol"].tolist(), load_gene_list(cfg))
    return np.asarray((sparse.csr_matrix(pool.X) @ mapper).todense(), dtype=np.float32)


def build_adapter(cfg: dict, model, depth: float) -> BulkAdapter:
    spec = cfg["adapter"]["film_lora"]
    return BulkAdapter(
        model,
        n_genes=model.n_genes,
        target_depth=depth * float(spec["depth_multiplier"]),
        film_hidden=int(spec["film_hidden"]),
        lora_rank=int(spec["lora_rank"]),
        lora_alpha=float(spec["lora_alpha"]),
    )


def losses(adapter: BulkAdapter, counts: torch.Tensor, hidden: torch.Tensor, train: dict) -> dict:
    out = adapter(counts, hidden)
    logits = out["logits"]
    lib = counts.sum(-1, keepdim=True).clamp(min=1.0)
    truth_frac = counts / lib
    floor = float(train["min_cpm_for_loss"]) / 1e6
    expressed = truth_frac.mean(dim=(0, 1)) >= floor

    hidden_genes = hidden & expressed
    masked_logits = logits.masked_fill(~hidden_genes[None, None, :], float("-inf"))
    log_p_hidden = torch.log_softmax(masked_logits, dim=-1)
    hidden_counts = counts * hidden_genes
    multinomial = -(hidden_counts * torch.nan_to_num(log_p_hidden, neginf=0.0)).sum(-1) / hidden_counts.sum(-1).clamp(min=1.0)

    def logcpm_mse(genes: torch.Tensor) -> torch.Tensor:
        pred = torch.softmax(logits.masked_fill(~genes[None, None, :], float("-inf")), dim=-1)
        target = counts * genes
        target = target / target.sum(-1, keepdim=True).clamp(min=1.0)
        diff = torch.log1p(1e6 * pred) - torch.log1p(1e6 * target)
        return (diff.pow(2) * genes).sum(-1) / genes.sum().clamp(min=1)

    visible_genes = ~hidden & expressed
    terms = {
        "multinomial": multinomial.mean(),
        "logcpm": logcpm_mse(hidden_genes).mean(),
        "visible": logcpm_mse(visible_genes).mean(),
    }
    terms["loss"] = (
        float(train["multinomial_weight"]) * terms["multinomial"]
        + float(train["logcpm_weight"]) * terms["logcpm"]
        + float(train["visible_weight"]) * terms["visible"]
    )
    return terms


def sample_batch(pool: np.ndarray, rows: np.ndarray, depth: float, train: dict, generator, min_depth_multiplier: float) -> tuple[torch.Tensor, torch.Tensor]:
    chosen = generator.choice(rows, int(train["window_size"]), replace=rows.size < int(train["window_size"]))
    native = pool[chosen]
    lib = native.sum(1)
    low = np.log(depth * float(min_depth_multiplier))
    targets = np.exp(generator.uniform(low, np.log(np.maximum(lib, np.exp(low) + 1))))
    thinned = np.vstack([thin_counts(native[i : i + 1], targets[i], generator) for i in range(len(native))])
    n_genes = pool.shape[1]
    rate = generator.uniform(float(train["mask_fraction_min"]), float(train["mask_fraction_max"]))
    hidden = np.zeros(n_genes, dtype=bool)
    hidden[generator.choice(n_genes, int(rate * n_genes), replace=False)] = True
    return torch.from_numpy(thinned).unsqueeze(0), torch.from_numpy(hidden)


def main() -> None:
    set_threads()
    cfg = load_config()
    train = cfg["adapter"]["film_lora"]
    generator = rng(cfg, 61)
    torch.manual_seed(int(cfg["stack"]["random_seed"]))
    depth = reference_depth(cfg)
    pool = load_pool(cfg)
    order = generator.permutation(pool.shape[0])
    n_val = max(1, int(round(pool.shape[0] * float(cfg["adapter"]["pool"]["validation_fraction"]))))
    val_rows, train_rows = order[:n_val], order[n_val:]
    model = load_stack(cfg)
    adapter = build_adapter(cfg, model, depth)
    params = adapter.trainable()
    print(json.dumps({"reference_depth": depth, "pool": list(pool.shape), "trainable_parameters": int(sum(p.numel() for p in params))}), flush=True)
    optimizer = torch.optim.AdamW(params, lr=float(train["learning_rate"]), weight_decay=float(train["weight_decay"]))
    steps = int(train["steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
    val_generator = rng(cfg, 62)
    val_batches = [sample_batch(pool, val_rows, depth, train, val_generator, cfg["adapter"]["pool"]["min_depth_multiplier"]) for _ in range(4)]
    history = []
    start = time.time()
    for step in range(1, steps + 1):
        adapter.train()
        counts, hidden = sample_batch(pool, train_rows, depth, train, generator, cfg["adapter"]["pool"]["min_depth_multiplier"])
        terms = losses(adapter, counts, hidden, train)
        optimizer.zero_grad()
        terms["loss"].backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        record = {"step": step, **{k: float(v) for k, v in terms.items()}, "seconds": round(time.time() - start, 1)}
        if step % int(train["validate_every"]) == 0 or step == 1 or step == steps:
            adapter.eval()
            with torch.no_grad():
                val = [losses(adapter, c, h, train) for c, h in val_batches]
            record.update({f"val_{k}": float(np.mean([float(v[k]) for v in val])) for k in val[0]})
        history.append(record)
        if step % int(train["log_every"]) == 0 or "val_loss" in record:
            print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}), flush=True)
    state = {name: tensor for name, tensor in adapter.state_dict().items() if not name.startswith("model.") and not name.startswith("output_lora.base.")}
    torch.save({"state": state, "reference_depth": depth, "config": cfg["adapter"]}, path(cfg, "results") / train["weights"])
    (path(cfg, "results") / "film_lora_adapter_history.json").write_text(json.dumps(history) + "\n")


if __name__ == "__main__":
    main()
