"""Run every model under every input configuration and cache the quantile forecasts.

Usage: python -m bench.run [--models ...] [--configs ...] [--datasets ...]

Non-raw configurations only re-run the tasks whose model input actually changes; all other
tasks reuse the raw forecasts, so results are identical where a technique is a no-op.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch

from bench.data import Dataset, Task, dataset_names, load_dataset
from bench.models import MODELS
from bench.transforms import can_log, hampel_clip, impute_linear

PRED_DIR = Path(__file__).resolve().parent.parent / "results" / "preds"
RAW_CTX = 1024
CONFIGS = {
    "raw": dict(ctx=RAW_CTX),
    "impute": dict(ctx=RAW_CTX, pre="impute"),
    "log1p": dict(ctx=RAW_CTX, pre="log1p"),
    "hampel": dict(ctx=RAW_CTX, pre="hampel"),
    "ctx256": dict(ctx=256),
    "ctx2048": dict(ctx=2048),
    "negate": dict(ctx=RAW_CTX, pre="negate"),
}


def prepare(ds: Dataset, task: Task, cfg: dict) -> tuple[np.ndarray, str | None]:
    """Return the model context and the inverse transform to apply to its quantiles."""
    ctx = ds.history(task)[-cfg["ctx"] :]
    pre = cfg.get("pre")
    if pre == "impute":
        return impute_linear(ctx), None
    if pre == "hampel":
        return hampel_clip(ctx, ds.season), None
    if pre == "log1p" and can_log(ctx):
        return np.log1p(ctx), "expm1"
    if pre == "negate":
        return -ctx, None
    return ctx, None


def changed(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape != b.shape or not np.array_equal(a, b, equal_nan=True)


def forecast(model, ds: Dataset, tasks: list[Task], cfg: dict, raw_q: np.ndarray | None) -> tuple[np.ndarray, int]:
    contexts, inverses, todo = [], [], []
    for k, task in enumerate(tasks):
        ctx, inv = prepare(ds, task, cfg)
        if raw_q is None or changed(ctx, ds.history(task)[-RAW_CTX:]):
            contexts.append(ctx)
            inverses.append(inv)
            todo.append(k)
    out = np.empty((len(tasks), ds.horizon, 9)) if raw_q is None else raw_q.copy()
    if todo:
        q = model.predict(contexts, ds.horizon)
        for row, (k, inv) in enumerate(zip(todo, inverses)):
            out[k] = np.expm1(q[row]) if inv == "expm1" else q[row]
    return out, len(todo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--datasets", nargs="+", default=dataset_names())
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    torch.set_num_threads(args.threads)

    datasets = {name: load_dataset(name) for name in args.datasets}
    timing_path = PRED_DIR / "timing.jsonl"
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        model = None
        for cfg_name in args.configs:
            cfg = CONFIGS[cfg_name]
            for name, ds in datasets.items():
                path = PRED_DIR / cfg_name / model_name / f"{name}.npz"
                if path.exists():
                    continue
                model = model or MODELS[model_name]()
                raw_q = None
                if cfg_name != "raw":
                    raw_q = np.load(PRED_DIR / "raw" / model_name / f"{name}.npz")["test"]
                t0 = time.perf_counter()
                test_q, n_run = forecast(model, ds, ds.tasks, cfg, raw_q)
                arrays = {"test": test_q}
                if cfg_name == "raw" and ds.val_tasks:
                    arrays["val"], n_val = forecast(model, ds, ds.val_tasks, cfg, None)
                    n_run += n_val
                seconds = time.perf_counter() - t0
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                record = dict(model=model_name, config=cfg_name, dataset=name, n_run=n_run, seconds=seconds)
                with timing_path.open("a") as f:
                    f.write(json.dumps(record) + "\n")
                print(f"{model_name:12s} {cfg_name:8s} {name:28s} ran={n_run:4d} {seconds:7.1f}s", flush=True)


if __name__ == "__main__":
    main()
