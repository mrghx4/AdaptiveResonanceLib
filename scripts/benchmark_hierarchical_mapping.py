#!/usr/bin/env python3
"""Benchmark DeepARTMAP map_deep chain mapping (Python fallback vs C++ helper)."""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime, timezone
from statistics import median
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.datasets import make_blobs

# Keep matplotlib cache local/writable when ART imports visualization modules.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import artlib.hierarchical.DeepARTMAP as deep_mod
from artlib.elementary.FuzzyART import FuzzyART
from artlib.hierarchical.DeepARTMAP import DeepARTMAP


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="benchmarks/hierarchical_mapping_benchmark.csv")
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--n-features", type=int, default=24)
    p.add_argument("--n-classes", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def make_model() -> DeepARTMAP:
    modules = [
        FuzzyART(rho=0.4, alpha=1e-3, beta=1.0),
        FuzzyART(rho=0.6, alpha=1e-3, beta=1.0),
        FuzzyART(rho=0.8, alpha=1e-3, beta=1.0),
    ]
    return DeepARTMAP(modules)


def build_data(
    n_samples: int,
    n_features: int,
    n_classes: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    x, y = make_blobs(
        n_samples=n_samples,
        centers=n_classes,
        n_features=n_features,
        cluster_std=0.9,
        random_state=seed,
        shuffle=True,
    )
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.int32, copy=False)
    # Use three channels with slight deterministic perturbations.
    x0 = x
    x1 = (0.95 * x + 0.05).astype(np.float64, copy=False)
    x2 = (0.90 * x + 0.10).astype(np.float64, copy=False)
    return [x0, x1, x2], y


def run_map_deep_benchmark(
    backend_label: str,
    configure: Callable[[], None],
    model: DeepARTMAP,
    labels_last: np.ndarray,
    repeats: int,
    warmup: int,
) -> tuple[list[dict[str, str]], list[float]]:
    rows: list[dict[str, str]] = []
    timings: list[float] = []

    for run_idx in range(warmup + repeats):
        configure()
        t0 = perf_counter()
        _ = model.map_deep(-1, labels_last)
        dt = perf_counter() - t0
        if run_idx >= warmup:
            rep_idx = run_idx - warmup
            timings.append(dt)
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": "DeepARTMAP_map_deep_chain",
                    "backend_requested": backend_label,
                    "backend_actual": backend_label,
                    "status": "ok",
                    "repeat_idx": str(rep_idx),
                    "prepare_s": "",
                    "fit_s": "",
                    "predict_s": f"{dt:.6f}",
                    "total_s": f"{dt:.6f}",
                    "prepare_median_s": "",
                    "fit_median_s": "",
                    "predict_median_s": "",
                    "total_median_s": "",
                    "n_train": str(labels_last.shape[0]),
                    "n_test": "0",
                    "n_features": "",
                    "n_classes": "",
                    "seed": "",
                    "notes": "phase=map_deep_chain",
                }
            )
    return rows, timings


def main() -> int:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    X, y = build_data(args.n_samples, args.n_features, args.n_classes, args.seed)
    model = make_model()
    X_prep, _ = model.prepare_data(X)
    model.fit(X_prep, y, max_iter=1)
    labels_last = model.layers[-1].labels_a

    cpp_helper = deep_mod.MapSimpleARTMAPLabelsChain

    def force_python() -> None:
        deep_mod.MapSimpleARTMAPLabelsChain = None

    def force_cpp() -> None:
        deep_mod.MapSimpleARTMAPLabelsChain = cpp_helper

    rows_py, ts_py = run_map_deep_benchmark(
        "python", force_python, model, labels_last, args.repeats, args.warmup
    )
    rows_cpp, ts_cpp = run_map_deep_benchmark(
        "c++", force_cpp, model, labels_last, args.repeats, args.warmup
    )

    # Restore module-level helper.
    deep_mod.MapSimpleARTMAPLabelsChain = cpp_helper

    summary_rows = []
    for label, vals in (("python", ts_py), ("c++", ts_cpp)):
        summary_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": "DeepARTMAP_map_deep_chain",
                "backend_requested": label,
                "backend_actual": label,
                "status": "ok",
                "repeat_idx": "summary",
                "prepare_s": "",
                "fit_s": "",
                "predict_s": "",
                "total_s": "",
                "prepare_median_s": "",
                "fit_median_s": "",
                "predict_median_s": f"{median(vals):.6f}",
                "total_median_s": f"{median(vals):.6f}",
                "n_train": str(labels_last.shape[0]),
                "n_test": "0",
                "n_features": str(args.n_features),
                "n_classes": str(args.n_classes),
                "seed": str(args.seed),
                "notes": "phase=map_deep_chain",
            }
        )

    fieldnames = [
        "timestamp_utc",
        "model",
        "backend_requested",
        "backend_actual",
        "status",
        "repeat_idx",
        "prepare_s",
        "fit_s",
        "predict_s",
        "total_s",
        "prepare_median_s",
        "fit_median_s",
        "predict_median_s",
        "total_median_s",
        "n_train",
        "n_test",
        "n_features",
        "n_classes",
        "seed",
        "notes",
    ]
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_py + rows_cpp + summary_rows)

    print(f"Wrote benchmark rows to {args.out}")
    print(f"python map_deep median: {median(ts_py):.6f}s")
    print(f"c++ map_deep median:    {median(ts_cpp):.6f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
