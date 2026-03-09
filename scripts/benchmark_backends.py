#!/usr/bin/env python3
"""Benchmark Python/Torch/C++ ART backends and write CSV timing baselines."""

from __future__ import annotations

import argparse
import csv
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from time import perf_counter
from typing import Any, Callable

import numpy as np
from sklearn.datasets import make_blobs

# Keep matplotlib cache local/writable when ART imports visualization modules.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

from artlib.optimized.BinaryFuzzyARTMAPFactory import BinaryFuzzyARTMAPFactory
from artlib.optimized.ART2Factory import ART2Factory
from artlib.optimized.BayesianARTFactory import BayesianARTFactory
from artlib.optimized.BayesianARTMAPFactory import BayesianARTMAPFactory
from artlib.optimized.BinaryFuzzyARTFactory import BinaryFuzzyARTFactory
from artlib.optimized.EllipsoidARTFactory import EllipsoidARTFactory
from artlib.optimized.FuzzyARTFactory import FuzzyARTFactory
from artlib.optimized.FuzzyARTMAPFactory import FuzzyARTMAPFactory
from artlib.optimized.GaussianARTFactory import GaussianARTFactory
from artlib.optimized.GaussianARTMAPFactory import GaussianARTMAPFactory
from artlib.optimized.HypersphereARTFactory import HypersphereARTFactory
from artlib.optimized.HypersphereARTMAPFactory import HypersphereARTMAPFactory
from artlib.optimized.QuadraticNeuronARTFactory import QuadraticNeuronARTFactory


@dataclass
class Case:
    name: str
    kind: str  # "artmap" or "art"
    backends: list[str]
    factory: Callable[..., Any]
    kwargs_builder: Callable[[int], dict[str, Any]]
    binary_input: bool = False


def detect_backend(model: Any) -> str:
    mod = type(model).__module__.lower()
    if ".torch." in mod:
        return "torch"
    if ".cpp." in mod:
        return "c++"
    return "python"


def build_continuous_data(
    n_train: int, n_test: int, n_features: int, n_classes: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def build_binary_data(
    n_train: int, n_test: int, n_features: int, n_classes: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n_train + n_test, n_features), dtype=np.int8)
    # Deterministic synthetic labels from block-wise population counts.
    block = max(1, n_features // max(1, n_classes))
    y = np.zeros((n_train + n_test,), dtype=np.int32)
    for c in range(n_classes):
        start = c * block
        end = min(n_features, start + block)
        if start >= n_features:
            break
        y += (x[:, start:end].sum(axis=1) > ((end - start) // 2)).astype(np.int32)
    y %= max(1, n_classes)
    return x[:n_train], y[:n_train], x[n_train:], y[n_train:]


def benchmark_case(
    case: Case,
    n_train: int,
    n_test: int,
    n_features: int,
    n_classes: int,
    seed: int,
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    if case.binary_input:
        x_train, y_train, x_test, y_test = build_binary_data(
            n_train, n_test, n_features, n_classes, seed
        )
    else:
        x_train, y_train, x_test, y_test = build_continuous_data(
            n_train, n_test, n_features, n_classes, seed
        )

    x_full = np.vstack([x_train, x_test])
    rows: list[dict[str, Any]] = []

    for backend in case.backends:
        notes = ""
        try:
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                model = case.factory(backend=backend, **case.kwargs_builder(n_features))
            backend_actual = detect_backend(model)
            if rec:
                notes = " | ".join(str(w.message) for w in rec)
        except Exception as exc:
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": case.name,
                    "backend_requested": backend,
                    "backend_actual": "unknown",
                    "status": "error",
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
                    "notes": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        rep_prepare: list[float] = []
        rep_fit: list[float] = []
        rep_predict: list[float] = []
        rep_total: list[float] = []

        total_runs = warmup + repeats
        for run_idx in range(total_runs):
            row: dict[str, Any] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model": case.name,
                "backend_requested": backend,
                "backend_actual": backend_actual,
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
                "seed": seed + run_idx,
                "notes": notes,
            }
            t0 = perf_counter()
            try:
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("ignore")
                    run_model = case.factory(
                        backend=backend,
                        **case.kwargs_builder(n_features),
                    )

                p0 = perf_counter()
                x_prepared = run_model.prepare_data(x_full)
                p1 = perf_counter()
                prepare_s = p1 - p0

                x_train_p = x_prepared[:n_train]
                x_test_p = x_prepared[n_train:]

                f0 = perf_counter()
                if case.kind == "artmap":
                    run_model.fit(x_train_p, y_train)
                else:
                    run_model.fit(x_train_p)
                f1 = perf_counter()
                fit_s = f1 - f0

                y0 = perf_counter()
                _ = run_model.predict(x_test_p)
                y1 = perf_counter()
                predict_s = y1 - y0
                total_s = perf_counter() - t0
            except Exception as exc:
                row["status"] = "error"
                row["notes"] = (
                    (row["notes"] + " | ") if row["notes"] else ""
                ) + f"run[{run_idx}] {type(exc).__name__}: {exc}"
                rows.append(row)
                continue

            if run_idx >= warmup:
                rep_idx = run_idx - warmup
                rep_prepare.append(prepare_s)
                rep_fit.append(fit_s)
                rep_predict.append(predict_s)
                rep_total.append(total_s)
                row["repeat_idx"] = str(rep_idx)
                row["prepare_s"] = f"{prepare_s:.6f}"
                row["fit_s"] = f"{fit_s:.6f}"
                row["predict_s"] = f"{predict_s:.6f}"
                row["total_s"] = f"{total_s:.6f}"
                rows.append(row)

        if rep_total:
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": case.name,
                    "backend_requested": backend,
                    "backend_actual": backend_actual,
                    "status": "ok",
                    "repeat_idx": "summary",
                    "prepare_s": "",
                    "fit_s": "",
                    "predict_s": "",
                    "total_s": "",
                    "prepare_median_s": f"{median(rep_prepare):.6f}",
                    "fit_median_s": f"{median(rep_fit):.6f}",
                    "predict_median_s": f"{median(rep_predict):.6f}",
                    "total_median_s": f"{median(rep_total):.6f}",
                    "n_train": n_train,
                    "n_test": n_test,
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "seed": seed,
                    "notes": notes,
                }
            )

    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="benchmarks/backend_runtime_baseline.csv")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-features", type=int, default=32)
    p.add_argument("--n-classes", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append", action="store_true")
    p.add_argument("--quick", action="store_true", help="Small/fast benchmark sizes.")
    p.add_argument("--repeats", type=int, default=1, help="Measured repeats per case/backend.")
    p.add_argument("--warmup", type=int, default=0, help="Warmup runs per case/backend.")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Convenience flag for stable results (equivalent to --warmup 1 --repeats 5).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    n_train = args.n_train
    n_test = args.n_test
    n_features = args.n_features
    n_classes = args.n_classes
    repeats = max(1, args.repeats)
    warmup = max(0, args.warmup)
    if args.quick:
        n_train = 600
        n_test = 200
        n_features = 16
        n_classes = 3
    if args.strict:
        warmup = max(warmup, 1)
        repeats = max(repeats, 5)

    cases = [
        Case(
            name="FuzzyARTMAP",
            kind="artmap",
            backends=["python", "torch", "c++"],
            factory=FuzzyARTMAPFactory,
            kwargs_builder=lambda _: {"rho": 0.8, "alpha": 1e-10, "beta": 1.0},
        ),
        Case(
            name="GaussianARTMAP",
            kind="artmap",
            backends=["python", "torch", "c++"],
            factory=GaussianARTMAPFactory,
            kwargs_builder=lambda f: {
                "rho": 0.05,
                "alpha": 1e-10,
                "sigma_init": np.full((f,), 0.33, dtype=np.float64),
            },
        ),
        Case(
            name="HypersphereARTMAP",
            kind="artmap",
            backends=["python", "torch", "c++"],
            factory=HypersphereARTMAPFactory,
            kwargs_builder=lambda _: {"rho": 0.8, "alpha": 1e-10, "beta": 1.0, "r_hat": 8.0},
        ),
        Case(
            name="BayesianARTMAP",
            kind="artmap",
            backends=["python", "torch", "c++"],
            factory=BayesianARTMAPFactory,
            kwargs_builder=lambda f: {
                "rho": 0.7,
                "cov_init": np.eye(f, dtype=np.float64),
            },
        ),
        Case(
            name="BinaryFuzzyARTMAP",
            kind="artmap",
            backends=["python", "torch", "c++"],
            factory=BinaryFuzzyARTMAPFactory,
            kwargs_builder=lambda _: {"rho": 0.8},
            binary_input=True,
        ),
        Case(
            name="GaussianART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=GaussianARTFactory,
            kwargs_builder=lambda f: {
                "rho": 0.05,
                "alpha": 1e-10,
                "sigma_init": np.full((f,), 0.33, dtype=np.float64),
            },
        ),
        Case(
            name="FuzzyART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=FuzzyARTFactory,
            kwargs_builder=lambda _: {
                "rho": 0.8,
                "alpha": 1e-10,
                "beta": 1.0,
            },
        ),
        Case(
            name="BinaryFuzzyART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=BinaryFuzzyARTFactory,
            kwargs_builder=lambda _: {"rho": 0.8},
            binary_input=True,
        ),
        Case(
            name="ART2A",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=ART2Factory,
            kwargs_builder=lambda _: {
                "rho": 0.7,
                "alpha": 0.1,
                "beta": 0.5,
            },
        ),
        Case(
            name="HypersphereART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=HypersphereARTFactory,
            kwargs_builder=lambda _: {
                "rho": 0.8,
                "alpha": 1e-10,
                "beta": 1.0,
                "r_hat": 8.0,
            },
        ),
        Case(
            name="EllipsoidART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=EllipsoidARTFactory,
            kwargs_builder=lambda _: {
                "rho": 0.7,
                "alpha": 1e-5,
                "beta": 0.1,
                "mu": 0.5,
                "r_hat": 1.0,
            },
        ),
        Case(
            name="BayesianART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=BayesianARTFactory,
            kwargs_builder=lambda f: {
                "rho": 0.7,
                "cov_init": np.eye(f, dtype=np.float64),
            },
        ),
        Case(
            name="QuadraticNeuronART",
            kind="art",
            backends=["python", "torch", "c++"],
            factory=QuadraticNeuronARTFactory,
            kwargs_builder=lambda _: {
                "rho": 0.7,
                "s_init": 0.5,
                "lr_b": 0.1,
                "lr_w": 0.1,
                "lr_s": 0.05,
            },
        ),
    ]

    all_rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        all_rows.extend(
            benchmark_case(
                case=case,
                n_train=n_train,
                n_test=n_test,
                n_features=n_features,
                n_classes=n_classes,
                seed=args.seed + i,
                repeats=repeats,
                warmup=warmup,
            )
        )

    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fields = [
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

    write_header = True
    mode = "w"
    if args.append and os.path.exists(out_path):
        mode = "a"
        write_header = False

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerows(all_rows)

    ok = sum(1 for r in all_rows if r["status"] == "ok")
    err = len(all_rows) - ok
    print(f"Wrote {len(all_rows)} rows to {out_path} (ok={ok}, error={err})")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
