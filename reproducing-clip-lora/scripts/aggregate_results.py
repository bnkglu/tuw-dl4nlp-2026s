#!/usr/bin/env python3
"""
Aggregate per-run results into seed-averaged numbers for the report.

Reads the CSV produced by run_table_3.sh / run_table_4.sh / run_fig3.sh
(default: results/clip_lora_results.csv) and groups every run by its full
configuration *except* the seed, then reports mean +/- std accuracy over seeds.

Because the paper reports top-1 accuracy averaged over seeds 1, 2, 3, this is
the view you compare against the paper's tables/figure.

Usage:
    python scripts/aggregate_results.py
    python scripts/aggregate_results.py --csv results/clip_lora_results.csv
    python scripts/aggregate_results.py --out results/clip_lora_summary.csv

Notes:
- FAILED rows (and any non-numeric accuracy) are skipped.
- If a (config, seed) appears multiple times (e.g. a failed run that was later
  re-run successfully), the LAST OK row for that exact run is kept.
- Uses only the Python standard library (csv, statistics) -- no pandas needed.
"""

import argparse
import csv
import os
import statistics
import sys

# Columns that together identify a single run (everything except the seed and
# the measured outcome). Grouping on these collapses the 3 seeds together.
GROUP_KEYS = ["table", "dataset", "backbone", "shots", "rank", "params", "encoder", "position"]


def load_runs(csv_path):
    """Return a dict: run_key (incl. seed) -> (accuracy, seconds_or_None), last OK row wins."""
    runs = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in GROUP_KEYS + ["seed", "accuracy", "status"] if c not in fields]
        if missing:
            sys.exit(f"ERROR: CSV is missing expected columns: {missing}\nFound: {fields}")
        has_seconds = "seconds" in fields  # older CSVs may not have it
        for row in reader:
            if (row.get("status") or "").strip().upper() != "OK":
                continue
            try:
                acc = float(row["accuracy"])
            except (TypeError, ValueError):
                continue
            secs = None
            if has_seconds:
                try:
                    secs = float(row["seconds"])
                except (TypeError, ValueError):
                    secs = None
            run_key = tuple(row[k] for k in GROUP_KEYS) + (row["seed"],)
            runs[run_key] = (acc, secs)  # last OK wins
    return runs


def aggregate(runs):
    """Group runs by GROUP_KEYS and compute mean/std over seeds."""
    groups = {}
    for run_key, (acc, secs) in runs.items():
        gkey = run_key[:-1]          # drop the seed
        seed = run_key[-1]
        groups.setdefault(gkey, []).append((seed, acc, secs))

    summary = []
    for gkey, rows in groups.items():
        accs = [a for _, a, _ in rows]
        secs = [s for _, _, s in rows if s is not None]
        seeds = ",".join(sorted(s for s, _, _ in rows))
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        summary.append(dict(zip(GROUP_KEYS, gkey)) | {
            "n_seeds": len(accs),
            "mean_acc": round(mean, 2),
            "std_acc": round(std, 2),
            "mean_sec": round(statistics.mean(secs)) if secs else "",
            "total_sec": round(sum(secs)) if secs else "",
            "seeds": seeds,
        })

    # Sort for stable, readable output.
    def sort_key(r):
        return (r["table"], r["dataset"], r["backbone"], int(r["shots"]),
                int(r["rank"]), r["params"], r["encoder"], r["position"])

    summary.sort(key=sort_key)
    return summary


def print_summary(summary):
    if not summary:
        print("No OK runs found to aggregate.")
        return
    by_table = {}
    for r in summary:
        by_table.setdefault(r["table"], []).append(r)

    for table, rows in by_table.items():
        print(f"\n=== {table} ===")
        # Tables (table3/table4) have fixed rank/params/encoder/position, so show a
        # compact view; fig3 varies them, so show the full config.
        is_fig3 = table == "fig3"
        if is_fig3:
            header = f"{'dataset':<16}{'shots':>6}{'rank':>5}{'params':>10}{'encoder':>8}{'position':>9}{'n':>3}{'mean':>8}{'std':>7}{'sec':>7}"
        else:
            header = f"{'dataset':<16}{'backbone':>10}{'shots':>6}{'n':>3}{'mean':>8}{'std':>7}{'sec':>7}"
        print(header)
        print("-" * len(header))
        for r in rows:
            if is_fig3:
                print(f"{r['dataset']:<16}{r['shots']:>6}{r['rank']:>5}{r['params']:>10}"
                      f"{r['encoder']:>8}{r['position']:>9}{r['n_seeds']:>3}"
                      f"{r['mean_acc']:>8.2f}{r['std_acc']:>7.2f}{str(r['mean_sec']):>7}")
            else:
                print(f"{r['dataset']:<16}{r['backbone']:>10}{r['shots']:>6}{r['n_seeds']:>3}"
                      f"{r['mean_acc']:>8.2f}{r['std_acc']:>7.2f}{str(r['mean_sec']):>7}")


def write_summary(summary, out_path):
    fields = GROUP_KEYS + ["n_seeds", "mean_acc", "std_acc", "mean_sec", "total_sec", "seeds"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nWrote {len(summary)} aggregated rows to {out_path}")


def main():
    # Resolve paths relative to the repo root (this file lives in scripts/).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default=os.path.join(repo_root, "results", "clip_lora_results.csv"),
                        help="path to the per-run results CSV")
    parser.add_argument("--out", default=None,
                        help="optional path to write the aggregated summary CSV "
                             "(default: <results dir>/clip_lora_summary.csv)")
    parser.add_argument("--no-save", action="store_true", help="print only, do not write a summary CSV")
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"ERROR: results CSV not found: {args.csv}\n"
                 f"Run the experiments first (e.g. bash scripts/run_table_3.sh).")

    runs = load_runs(args.csv)
    summary = aggregate(runs)
    print_summary(summary)

    if not args.no_save:
        out_path = args.out or os.path.join(os.path.dirname(args.csv), "clip_lora_summary.csv")
        write_summary(summary, out_path)


if __name__ == "__main__":
    main()
