#!/usr/bin/env python3
"""Compare two backend benchmark CSV baselines and report regressions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean

PHASES = ("prepare_s", "fit_s", "predict_s", "total_s")


def safe_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_avg(path: str) -> dict[tuple[str, str], dict[str, float]]:
    summaries: dict[tuple[str, str], dict[str, float]] = {}
    buckets: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            key = (r.get("model", ""), r.get("backend_actual", ""))
            if not key[0] or not key[1]:
                continue
            if r.get("repeat_idx") == "summary":
                pmap: dict[str, float] = {}
                for p in PHASES:
                    med_v = safe_float(r.get(p.replace("_s", "_median_s"), ""))
                    if med_v is not None:
                        pmap[p] = med_v
                if pmap:
                    summaries[key] = pmap
                continue
            for p in PHASES:
                v = safe_float(r.get(p, ""))
                if v is not None:
                    buckets[key][p].append(v)

    out: dict[tuple[str, str], dict[str, float]] = {}
    for key, pmap in summaries.items():
        out[key] = pmap
    for key, pmap in buckets.items():
        if key in out:
            continue
        out[key] = {p: mean(vals) for p, vals in pmap.items() if vals}
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, help="Baseline CSV (older)")
    p.add_argument("--new", required=True, help="New CSV (candidate)")
    p.add_argument(
        "--phase",
        choices=["prepare_s", "fit_s", "predict_s", "total_s", "all"],
        default="total_s",
    )
    p.add_argument(
        "--regress-threshold-pct",
        type=float,
        default=5.0,
        help="Flag if new time is slower than base by at least this percent",
    )
    p.add_argument(
        "--min-base-s",
        type=float,
        default=0.0,
        help="Ignore regression checks when base timing is below this absolute threshold.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base = load_avg(args.base)
    new = load_avg(args.new)
    phases = PHASES if args.phase == "all" else (args.phase,)

    keys = sorted(set(base.keys()) & set(new.keys()))
    if not keys:
        print("No overlapping (model, backend_actual) rows found.")
        return 1

    threshold = args.regress_threshold_pct / 100.0
    min_base = max(0.0, args.min_base_s)
    regressions = 0

    print("model,backend,phase,base_s,new_s,delta_pct,status")
    for key in keys:
        model, backend = key
        for phase in phases:
            b = base[key].get(phase)
            n = new[key].get(phase)
            if b is None or n is None or b <= 0:
                continue
            delta = (n - b) / b
            status = "ok"
            if b < min_base:
                status = "skip_fast"
            elif delta >= threshold:
                status = "REGRESSION"
                regressions += 1
            print(f"{model},{backend},{phase},{b:.6f},{n:.6f},{delta*100:.2f},{status}")

    if regressions > 0:
        print(f"Regressions detected: {regressions}")
        return 2

    print("No regressions detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
