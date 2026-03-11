#!/usr/bin/env python3
"""Benchmark FusionART python vs c++ dispatch (predict + predict_regression)."""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from datetime import datetime, timezone
from statistics import median
from time import perf_counter

import numpy as np
from sklearn.datasets import make_blobs

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

from artlib.elementary.FuzzyART import FuzzyART
from artlib.optimized.FusionARTFactory import FusionARTFactory


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="benchmarks/fusionart_benchmark.csv")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=16)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def build_data(
    n_train: int, n_test: int, n_features: int, n_classes: int, seed: int
) -> tuple[list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    x, y = make_blobs(
        n_samples=n_train + n_test,
        centers=n_classes,
        n_features=n_features,
        cluster_std=0.9,
        random_state=seed,
        shuffle=True,
    )
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.int32, copy=False)
    # 2-channel setup with deterministic channel transforms.
    c0 = x
    c1 = (0.92 * x + 0.08).astype(np.float64, copy=False)
    return [c0[:n_train], c1[:n_train]], y[:n_train], [c0[n_train:], c1[n_train:]], y[n_train:]


def make_model(backend: str, n_features: int):
    modules = [
        FuzzyART(rho=0.55, alpha=1e-3, beta=1.0),
        FuzzyART(rho=0.65, alpha=1e-3, beta=1.0),
    ]
    gamma = np.array([0.5, 0.5], dtype=float)
    dims = [2 * n_features, 2 * n_features]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return FusionARTFactory(
            modules=modules,
            gamma_values=gamma,
            channel_dims=dims,
            backend=backend,
        )


def base_row(model: str, backend: str, n_train: int, n_test: int, n_features: int, n_classes: int, seed: int):
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "backend_requested": backend,
        "backend_actual": backend,
        "status": "ok",
        "repeat_idx": "",
        "prepare_s": "",
        "fit_s": "",
        "predict_s": "",
        "total_s": "",
        "prepare_median_s": "",
        "fit_median_s": "",
        "predict_median_s": "",
        "total_median_s": "",
        "n_train": n_train,
        "n_test": n_test,
        "n_features": n_features,
        "n_classes": n_classes,
        "seed": seed,
        "notes": "",
    }


def main() -> int:
    args = parse_args()
    n_train = args.n_train
    n_test = args.n_test
    n_features = args.n_features
    n_classes = args.n_classes
    if args.quick:
        n_train = 800
        n_test = 300
        n_features = 10
        n_classes = 3

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rows: list[dict[str, object]] = []

    for backend in ("python", "c++"):
        rep_pred: list[float] = []
        rep_reg: list[float] = []
        rep_total: list[float] = []

        for run_idx in range(args.warmup + args.repeats):
            seed = args.seed + run_idx
            x_train_ch, y_train, x_test_ch, _ = build_data(
                n_train, n_test, n_features, n_classes, seed
            )
            model = make_model(backend, n_features)

            t0 = perf_counter()
            x_full_ch = [
                np.vstack([x_train_ch[k], x_test_ch[k]]) for k in range(len(x_train_ch))
            ]
            x_full_p = model.prepare_data(x_full_ch)
            x_train_p = x_full_p[:n_train]
            x_test_p = x_full_p[n_train:]
            p_s = perf_counter() - t0

            t1 = perf_counter()
            model.fit(x_train_p, max_iter=1)
            f_s = perf_counter() - t1

            t2 = perf_counter()
            _ = model.predict(x_test_p)
            pred_s = perf_counter() - t2

            t3 = perf_counter()
            _ = model.predict_regression(x_test_p, target_channels=[-1])
            reg_s = perf_counter() - t3

            total_s = p_s + f_s + pred_s + reg_s

            if run_idx >= args.warmup:
                rep_idx = run_idx - args.warmup
                rep_pred.append(pred_s)
                rep_reg.append(reg_s)
                rep_total.append(total_s)

                row_pred = base_row(
                    "FusionART_predict",
                    backend,
                    n_train,
                    n_test,
                    n_features,
                    n_classes,
                    seed,
                )
                row_pred.update(
                    {
                        "repeat_idx": str(rep_idx),
                        "prepare_s": f"{p_s:.6f}",
                        "fit_s": f"{f_s:.6f}",
                        "predict_s": f"{pred_s:.6f}",
                        "total_s": f"{total_s:.6f}",
                    }
                )
                rows.append(row_pred)

                row_reg = base_row(
                    "FusionART_predict_regression",
                    backend,
                    n_train,
                    n_test,
                    n_features,
                    n_classes,
                    seed,
                )
                row_reg.update(
                    {
                        "repeat_idx": str(rep_idx),
                        "prepare_s": "",
                        "fit_s": "",
                        "predict_s": f"{reg_s:.6f}",
                        "total_s": f"{reg_s:.6f}",
                        "notes": "phase=predict_regression",
                    }
                )
                rows.append(row_reg)

        summary_pred = base_row(
            "FusionART_predict",
            backend,
            n_train,
            n_test,
            n_features,
            n_classes,
            args.seed,
        )
        summary_pred.update(
            {
                "repeat_idx": "summary",
                "predict_median_s": f"{median(rep_pred):.6f}",
                "total_median_s": f"{median(rep_total):.6f}",
            }
        )
        rows.append(summary_pred)

        summary_reg = base_row(
            "FusionART_predict_regression",
            backend,
            n_train,
            n_test,
            n_features,
            n_classes,
            args.seed,
        )
        summary_reg.update(
            {
                "repeat_idx": "summary",
                "predict_median_s": f"{median(rep_reg):.6f}",
                "total_median_s": f"{median(rep_reg):.6f}",
                "notes": "phase=predict_regression",
            }
        )
        rows.append(summary_reg)

    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote FusionART benchmark rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
