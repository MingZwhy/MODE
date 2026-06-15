#!/usr/bin/env python3
"""Aggregate prune_verify JSON files under log/results into summary_prune_sweep.csv."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def fmt_metric(v):
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        if abs(v - round(v)) < 1e-9 and abs(v) <= 1e6:
            return str(int(round(v)))
        s = f"{v:.6f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Directory containing *.json (default: <this>/../log/results)",
    )
    args = ap.parse_args()
    script_dir = Path(__file__).resolve().parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / "log" / "results"
    out_csv = results_dir / "summary_prune_sweep.csv"

    rows: dict[float, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name.startswith("summary"):
            continue
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        cfg = data["config"]
        target = cfg["prune_target"]
        ratio = float(cfg["dominant_ratio"])
        tr = data.get("task_results") or {}
        cq = tr.get("chartqa") or {}
        mb = tr.get("mmbench_en_dev") or {}
        if ratio not in rows:
            rows[ratio] = {"ratio_pct": round(ratio * 100, 2)}
        r = rows[ratio]
        prefix = "prune_dom" if target == "dom" else "prune_red"
        r[f"{prefix}_chartqa_relaxed_overall"] = cq.get("relaxed_overall,none")
        r[f"{prefix}_chartqa_relaxed_human"] = cq.get("relaxed_human_split,none")
        r[f"{prefix}_mmbench_gpt_eval"] = mb.get("gpt_eval_score,none")

    cols = ["ratio_pct"]
    for pfx in ("prune_dom", "prune_red"):
        cols += [
            f"{pfx}_chartqa_relaxed_overall",
            f"{pfx}_chartqa_relaxed_human",
            f"{pfx}_mmbench_gpt_eval",
        ]

    with open(out_csv, "w", encoding="utf-8", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for ratio in sorted(rows.keys()):
            r = rows[ratio]
            w.writerow([fmt_metric(r.get(c, "")) for c in cols])

    print(f"Wrote {out_csv} ({len(rows)} ratios)")


if __name__ == "__main__":
    main()
