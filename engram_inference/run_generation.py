"""Run stack-generation on the prepared Engram inputs. Does not train."""

from __future__ import annotations

import os

# BLAS libraries read these at import, before torch or numpy.
_THREADS = str(os.cpu_count() or 1)
for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = _THREADS

import sys

import numpy as np

# Pandas string indexes from the checkpoint gene list are object arrays of
# numpy.str_, and this NumPy rejects np.char.upper on them. The generation
# CLI hits that path. Adapt the call without changing the model package.
_NP_CHAR_UPPER = np.char.upper


def _char_upper(values, *args, **kwargs):
    try:
        return _NP_CHAR_UPPER(values, *args, **kwargs)
    except TypeError:
        array = np.asarray(values)
        upper = np.array([str(item).upper() for item in array.ravel()])
        return upper.reshape(array.shape)


np.char.upper = _char_upper

from common import REPO, ROOT, load_config

DATASETS = {
    "cortical_organoid": {
        "base": ROOT / "data" / "cortical_organoid" / "stack_inputs" / "base.h5ad",
        "test": ROOT / "data" / "cortical_organoid" / "stack_inputs" / "test.h5ad",
        "split_key": "cortical_organoid",
        "output": ROOT / "outputs" / "cortical_organoid",
    },
    "sciplex": {
        "base": ROOT / "data" / "sciplex" / "stack_inputs" / "base.h5ad",
        "test": ROOT / "data" / "sciplex" / "stack_inputs" / "test.h5ad",
        "split_key": "sciplex",
        "output": ROOT / "outputs" / "sciplex",
    },
}


def run_one(name: str) -> None:
    cfg = load_config()
    stack = cfg["stack"]
    spec = DATASETS[name]
    split_column = cfg[spec["split_key"]]["split_column"]
    for path in (spec["base"], spec["test"]):
        if not path.exists():
            raise FileNotFoundError(path)
    spec["output"].mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "models" / stack["checkpoint_name"]
    genelist = ROOT / "models" / stack["genelist_name"]
    command = [
        sys.executable,
        "-m",
        "stack.cli.generation",
        "--checkpoint",
        str(checkpoint),
        "--base-adata",
        str(spec["base"]),
        "--test-adata",
        str(spec["test"]),
        "--genelist",
        str(genelist),
        "--output-dir",
        str(spec["output"]),
        "--split-column",
        split_column,
        "--prompt-ratio",
        str(stack["prompt_ratio"]),
        "--context-ratio",
        str(stack["context_ratio"]),
        "--context-ratio-min",
        str(stack["context_ratio_min"]),
        "--num-steps",
        str(stack["num_steps"]),
        "--mode",
        str(stack["mode"]),
        "--batch-size",
        str(stack["batch_size"]),
        "--num-workers",
        str(stack["num_workers"]),
        "--random-seed",
        str(stack["random_seed"]),
        "--show-progress",
    ]
    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    print("running", " ".join(command[3:]), flush=True)
    from stack.cli.generation import main as generation_main

    generation_main(command[3:])
    written = sorted(spec["output"].glob("*.h5ad"))
    if not written:
        raise RuntimeError(f"{name} generation wrote no h5ad files")
    print(f"{name} outputs:", flush=True)
    for path in written:
        print(f"  {path} ({path.stat().st_size} bytes)", flush=True)


def main() -> None:
    names = sys.argv[1:] or list(DATASETS)
    for name in names:
        if name not in DATASETS:
            raise SystemExit(f"Unknown dataset {name}")
        run_one(name)


if __name__ == "__main__":
    main()
