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

# Extra config columns present only in some CSVs (e.g. the KL-extension ablation).
# They are appended to the grouping key only when the CSV actually has them, so the
# table3/table4/table5/fig3 behavior is unchanged.
OPTIONAL_GROUP_KEYS = ["kl_weight", "kl_temp"]


def _active_group_keys(fields):
    """Required keys plus any optional config columns the CSV happens to carry."""
    return GROUP_KEYS + [k for k in OPTIONAL_GROUP_KEYS if k in fields]


def load_runs(csv_path):
    """Return (runs, group_keys). runs: run_key (incl. seed) -> (acc, secs), last OK wins."""
    runs = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        missing = [c for c in GROUP_KEYS + ["seed", "accuracy", "status"] if c not in fields]
        if missing:
            sys.exit(f"ERROR: CSV is missing expected columns: {missing}\nFound: {fields}")
        group_keys = _active_group_keys(fields)
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
            run_key = tuple(row[k] for k in group_keys) + (row["seed"],)
            runs[run_key] = (acc, secs)  # last OK wins
    return runs, group_keys


def _sortable(v):
    """Numeric-aware sort: numbers before strings, each in natural order."""
    try:
        return (0, float(v))
    except (TypeError, ValueError):
        return (1, str(v))


def aggregate(runs, group_keys):
    """Group runs by group_keys and compute mean/std over seeds."""
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
        summary.append(dict(zip(group_keys, gkey)) | {
            "n_seeds": len(accs),
            "mean_acc": round(mean, 2),
            "std_acc": round(std, 2),
            "mean_sec": round(statistics.mean(secs)) if secs else "",
            "total_sec": round(sum(secs)) if secs else "",
            "seeds": seeds,
        })

    summary.sort(key=lambda r: tuple(_sortable(r[k]) for k in group_keys))
    return summary


def print_summary(summary, group_keys):
    if not summary:
        print("No OK runs found to aggregate.")
        return
    has_kl = "kl_weight" in group_keys
    by_table = {}
    for r in summary:
        by_table.setdefault(r["table"], []).append(r)

    for table, rows in by_table.items():
        print(f"\n=== {table} ===")
        # Tables (table3/4/5) have fixed rank/params/encoder/position -> compact view.
        # fig3 varies them; the KL ablation varies kl_weight/kl_temp -> show those.
        is_fig3 = table == "fig3"
        is_kl = table == "kl" and has_kl
        if is_fig3:
            header = f"{'dataset':<16}{'shots':>6}{'rank':>5}{'params':>10}{'encoder':>8}{'position':>9}{'n':>3}{'mean':>8}{'std':>7}{'sec':>7}"
        elif is_kl:
            header = f"{'dataset':<16}{'shots':>6}{'kl_w':>7}{'kl_T':>6}{'n':>3}{'mean':>8}{'std':>7}{'sec':>7}"
        else:
            header = f"{'dataset':<16}{'backbone':>10}{'shots':>6}{'n':>3}{'mean':>8}{'std':>7}{'sec':>7}"
        print(header)
        print("-" * len(header))
        for r in rows:
            if is_fig3:
                print(f"{r['dataset']:<16}{r['shots']:>6}{r['rank']:>5}{r['params']:>10}"
                      f"{r['encoder']:>8}{r['position']:>9}{r['n_seeds']:>3}"
                      f"{r['mean_acc']:>8.2f}{r['std_acc']:>7.2f}{str(r['mean_sec']):>7}")
            elif is_kl:
                print(f"{r['dataset']:<16}{r['shots']:>6}{r['kl_weight']:>7}{r['kl_temp']:>6}"
                      f"{r['n_seeds']:>3}{r['mean_acc']:>8.2f}{r['std_acc']:>7.2f}{str(r['mean_sec']):>7}")
            else:
                print(f"{r['dataset']:<16}{r['backbone']:>10}{r['shots']:>6}{r['n_seeds']:>3}"
                      f"{r['mean_acc']:>8.2f}{r['std_acc']:>7.2f}{str(r['mean_sec']):>7}")


def write_summary(summary, out_path, group_keys):
    fields = group_keys + ["n_seeds", "mean_acc", "std_acc", "mean_sec", "total_sec", "seeds"]
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

    runs, group_keys = load_runs(args.csv)
    summary = aggregate(runs, group_keys)
    print_summary(summary, group_keys)

    if not args.no_save:
        out_path = args.out or os.path.join(os.path.dirname(args.csv), "clip_lora_summary.csv")
        write_summary(summary, out_path, group_keys)


if __name__ == "__main__":
    main()
