"""Run the benchmark on the published STACK checkpoint (no changes) and draw figures."""

from __future__ import annotations

import json
import sys

import anndata as ad
import numpy as np

from common import load_config, load_stack, path, rng, set_threads


def main(tag: str = "baseline") -> None:
    set_threads()
    from evaluate import cross_modal, evaluate
    from stack_ops import run_window

    cfg = load_config()
    results = path(cfg, "results")
    profiles = ad.read_h5ad(path(cfg, "data") / "profiles.h5ad")
    model = load_stack(cfg)
    table, embeddings = evaluate(lambda c, m: run_window(model, c, m), profiles, cfg, rng(cfg, 51))
    table.to_csv(results / f"{tag}_sets.csv", index=False)
    matches = cross_modal(profiles, embeddings, cfg["benchmark"])
    (results / f"{tag}_cross_modal.json").write_text(json.dumps(matches, indent=2) + "\n")
    np.savez_compressed(path(cfg, "data") / f"{tag}_embeddings.npz", **{k.replace("/", "__"): v for k, v in embeddings.items()})
    for row in matches:
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
