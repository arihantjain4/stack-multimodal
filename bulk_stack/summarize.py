"""Print a compact table of results/<tag>_sets.csv and the cross-modal matches."""

import json
import sys

import pandas as pd

from common import load_config, path

COLUMNS = ["set", "median_library", "zero_fraction", "recon_r", "baseline_r", "recon_gain", "visible_r",
           "knn_stack", "knn_pca", "emb_cosine", "preact_std", "gelu_active"]

if __name__ == "__main__":
    cfg = load_config()
    results = path(cfg, "results")
    for tag in sys.argv[1:] or ["baseline"]:
        table = pd.read_csv(results / f"{tag}_sets.csv")
        print(f"== {tag}")
        print(table[[c for c in COLUMNS if c in table]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        cross = results / f"{tag}_cross_modal.json"
        if cross.exists():
            for row in json.loads(cross.read_text()):
                print(json.dumps(row))
