#!/usr/bin/env python3
"""Summarize backend benchmark CSVs with per-model speedup tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean
from typing import Any

PHASES = ("prepare_s", "fit_s", "predict_s", "total_s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, help="Input benchmark CSV path")
    p.add_argument(
        "--phase",
        choices=["prepare_s", "fit_s", "predict_s", "total_s", "all"],
        default="all",
        help="Report one phase or all phases",
    )
    p.add_argument(
        "--format",
        choices=["markdown", "plain"],
        default="markdown",
        help="Output table format",
    )
    return p.parse_args()


def safe_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    agg: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        if r.get("status") != "ok":
            continue
        model = r.get("model", "")
        backend = r.get("backend_actual", "")
        if not model or not backend:
            continue
        for phase in PHASES:
            val = safe_float(r.get(phase, ""))
            if val is not None:
                agg[model][backend][phase].append(val)

    out: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for model, bmap in agg.items():
        for backend, pmap in bmap.items():
            for phase, vals in pmap.items():
                out[model][backend][phase] = mean(vals)
    return out


def speedup(base: float | None, comp: float | None) -> str:
    if base is None or comp is None or comp <= 0:
        return ""
    return f"{base / comp:.2f}x"


def row_for_phase(
    model: str,
    phase: str,
    stats: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    b = stats.get(model, {})
    py_t = b.get("python", {}).get(phase)
    torch_t = b.get("torch", {}).get(phase)
    cpp_t = b.get("c++", {}).get(phase)

    def fmt(v: float | None) -> str:
        return "" if v is None else f"{v:.6f}"

    return [
        model,
        phase,
        fmt(py_t),
        fmt(torch_t),
        fmt(cpp_t),
        speedup(py_t, cpp_t),
        speedup(torch_t, cpp_t),
    ]


def print_markdown(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(r) + " |")


def print_plain(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(r: list[str]) -> str:
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(r))

    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def main() -> int:
    args = parse_args()
    rows = load_rows(args.in_path)
    stats = aggregate(rows)

    if not stats:
        print("No successful benchmark rows found.")
        return 1

    phases = PHASES if args.phase == "all" else (args.phase,)
    headers = [
        "model",
        "phase",
        "python_s",
        "torch_s",
        "cpp_s",
        "py_over_cpp",
        "torch_over_cpp",
    ]

    out_rows: list[list[str]] = []
    for model in sorted(stats.keys()):
        for phase in phases:
            out_rows.append(row_for_phase(model, phase, stats))

    if args.format == "markdown":
        print_markdown(headers, out_rows)
    else:
        print_plain(headers, out_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
