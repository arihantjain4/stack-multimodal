"""Score adapter variants from config.yaml with the same benchmark as the baseline.

Usage: python run_variants.py [variant ...]   (default: every configured variant)
"""

from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np
import torch

from common import load_config, load_stack, path, rng, set_threads


def make_runner(cfg: dict, name: str, model, depth: float):
    from adapters import (BulkHead, PCAImputer, bulk_head_runner, context_decoder_runner, depth_scale_runner,
                          pseudocell_runner, thin_ensemble_runner)
    from train_adapter import build_adapter
    from train_bulk import make_side_channel

    spec = cfg["adapter"]["variants"][name]
    kind = spec["type"]
    results = path(cfg, "results")
    generator = rng(cfg, 71)
    if kind == "depth_scale":
        return depth_scale_runner(model, depth * float(spec["depth_multiplier"]))
    if kind == "thin_ensemble":
        return thin_ensemble_runner(model, depth * float(spec["depth_multiplier"]), int(spec["draws"]), generator)
    if kind == "pseudocell":
        return pseudocell_runner(model, depth * float(spec["depth_multiplier"]), int(spec["draws"]), generator)
    if kind == "film_lora":
        saved = torch.load(results / cfg["adapter"]["film_lora"]["weights"], map_location="cpu")
        adapter = build_adapter(cfg, model, float(saved["reference_depth"]))
        missing, unexpected = adapter.load_state_dict(saved["state"], strict=False)
        if unexpected or any(not k.startswith(("model.", "output_lora.base.")) for k in missing):
            raise RuntimeError(f"Adapter weights do not match: missing={missing} unexpected={unexpected}")
        adapter.eval()
        return adapter.run_window
    if kind == "context_decoder":
        context = json.loads((results / "context.json").read_text())
        stats = np.load(results / "pool_stats.npz")
        prior = {"fraction": stats["fraction"], "px": stats["px"]}
        return context_decoder_runner(make_runner(cfg, spec["base"], model, depth), float(context["beta"]), spec["prior"], prior)
    if kind in ("bulk_head", "side_channel"):
        head_spec = cfg["adapter"]["head"]
        saved_head = torch.load(results / "bulk_head.pt", map_location="cpu")
        head = BulkHead(model.n_genes, model.n_hidden * model.token_dim, int(head_spec["rank"]), torch.ones(model.n_genes))
        head.load_state_dict(saved_head["state"])
        head.eval()
        if kind == "bulk_head":
            return bulk_head_runner(model, head, float(saved_head["reference_depth"]) * float(head_spec["depth_multiplier"]))
        saved = torch.load(results / "side_channel.pt", map_location="cpu")
        adapter = make_side_channel(cfg, model, head, float(saved["reference_depth"]))
        missing, unexpected = adapter.load_state_dict(saved["state"], strict=False)
        if unexpected or any(not k.startswith("model.") for k in missing):
            raise RuntimeError(f"Side-channel weights do not match: missing={missing} unexpected={unexpected}")
        adapter.eval()
        return adapter.run_window
    if kind == "pca_imputer":
        saved = np.load(results / "pca_imputer.npz")
        return PCAImputer(saved["mean"], saved["loadings"], float(saved["ridge"])).run_window
    raise ValueError(f"Unknown variant type {kind!r}")


def main(names: list[str]) -> None:
    set_threads()
    from evaluate import cross_modal, evaluate
    from train_adapter import reference_depth

    cfg = load_config()
    results = path(cfg, "results")
    profiles = ad.read_h5ad(path(cfg, "data") / "profiles.h5ad")
    model = load_stack(cfg)
    depth = reference_depth(cfg)
    for name in names or list(cfg["adapter"]["variants"]):
        print(f"== {name} (reference depth {depth})", flush=True)
        runner = make_runner(cfg, name, model, depth)
        table, embeddings = evaluate(runner, profiles, cfg, rng(cfg, 51))
        table.to_csv(results / f"{name}_sets.csv", index=False)
        matches = cross_modal(profiles, embeddings, cfg["benchmark"])
        (results / f"{name}_cross_modal.json").write_text(json.dumps(matches, indent=2) + "\n")
        np.savez_compressed(path(cfg, "data") / f"{name}_embeddings.npz", **{k.replace("/", "__"): v for k, v in embeddings.items()})


if __name__ == "__main__":
    main(sys.argv[1:])
