#!/usr/bin/env python3
"""
Audit the results BOOKKEEPING (not the experiments). Checks four things and
prints a short pass/fail report:

  1. Completeness   - expected (table,dataset,shots,seed) cross-product vs CSV;
                      flag missing / duplicated cells and wrong seed count.
  2. Parse integrity- every accuracy casts to float and lands in 0..100.
  3. FAILED<->log   - each CSV row's accuracy must agree with its run log's
                      "Final test accuracy" line (the exact bug class that
                      already bit us: CSV FAILED while log had a real number).
  4. Paper sanity   - seed-averaged means vs a committed paper reference
                      (table4 has one); flag deviations > 0.3.

Run from anywhere; paths are resolved relative to the repo root.
"""
import csv, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(ROOT, "results")
RESULTS_CSV = os.path.join(RES, "clip_lora_results.csv")
FIG3_CSV    = os.path.join(RES, "clip_lora_fig3.csv")
PAPER_T4    = os.path.join(RES, "table4_vitb32_10datasets_summary.csv")

DATASETS = ["caltech101", "dtd", "eurosat", "fgvc", "food101", "imagenet",
            "oxford_flowers", "oxford_pets", "stanford_cars", "ucf101"]
SHOTS = [1, 2, 4, 8, 16]
SEEDS = [1, 2, 3]
TABLE_LOGDIR = {"table3": "table3", "table4": "table4", "table5": "table5"}

ACC_RE = re.compile(r"Final test accuracy:\s*([0-9]+\.[0-9]+)")
problems = {1: [], 2: [], 3: [], 4: []}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def log_acc(path):
    """Last 'Final test accuracy' number in a log, or None."""
    if not os.path.isfile(path):
        return "NO_LOG"
    with open(path, encoding="utf-8", errors="ignore") as f:
        hits = ACC_RE.findall(f.read())
    return float(hits[-1]) if hits else None


def table_logpath(row):
    d = TABLE_LOGDIR[row["table"]]
    return os.path.join(RES, d, row["dataset"], f"s{row['shots']}_seed{row['seed']}.log")


def fig3_logpath(row):
    name = f"r{row['rank']}_{row['params']}_{row['encoder']}_{row['position']}_seed{row['seed']}.log"
    return os.path.join(RES, "fig3", row["dataset"], name)


# ---------------------------------------------------------------- load
rows = read_csv(RESULTS_CSV)
fig3 = read_csv(FIG3_CSV)

# ============================================================ CHECK 1
# Completeness for the table cross-products.
seen = {}   # (table,dataset,shots,seed) -> count
for r in rows:
    key = (r["table"], r["dataset"], r["shots"], r["seed"])
    seen[key] = seen.get(key, 0) + 1

for table in ["table3", "table4", "table5"]:
    expected = {(table, d, str(s), str(sd)) for d in DATASETS for s in SHOTS for sd in SEEDS}
    present = {k for k in seen if k[0] == table}
    missing = expected - present
    extra = present - expected
    dups = {k: c for k, c in seen.items() if k[0] == table and c > 1}
    for k in sorted(missing):
        problems[1].append(f"MISSING  {k[0]} {k[1]} s{k[2]} seed{k[3]}")
    for k in sorted(extra):
        problems[1].append(f"UNEXPECTED {k}")
    for k, c in sorted(dups.items()):
        problems[1].append(f"DUPLICATE x{c}  {k}")

# seed-count per (table,dataset,shots) should be exactly 3
cells = {}
for r in rows:
    cells.setdefault((r["table"], r["dataset"], r["shots"]), set()).add(r["seed"])
for (t, d, s), sds in sorted(cells.items()):
    if len(sds) != len(SEEDS):
        problems[1].append(f"SEED COUNT {t} {d} s{s}: n={len(sds)} (seeds={sorted(sds)})")

# fig3: each of its datasets should have 129 unique configs, no dups
fig3_cfg = {}
for r in fig3:
    k = (r["dataset"], r["rank"], r["params"], r["encoder"], r["position"], r["seed"])
    fig3_cfg[k] = fig3_cfg.get(k, 0) + 1
for ds in sorted({r["dataset"] for r in fig3}):
    n = len({k for k in fig3_cfg if k[0] == ds})
    dups = sum(1 for k, c in fig3_cfg.items() if k[0] == ds and c > 1)
    if n != 129:
        problems[1].append(f"FIG3 {ds}: {n} unique configs (expected 129)")
    if dups:
        problems[1].append(f"FIG3 {ds}: {dups} duplicated configs")

# ============================================================ CHECK 2
for tag, data, path in [("results", rows, RESULTS_CSV), ("fig3", fig3, FIG3_CSV)]:
    for i, r in enumerate(data, start=2):
        raw = (r.get("accuracy") or "").strip()
        try:
            v = float(raw)
        except (TypeError, ValueError):
            problems[2].append(f"{tag} line~{i}: accuracy={raw!r} not a float "
                               f"({r['table']} {r['dataset']} s{r.get('shots')} seed{r.get('seed')})")
            continue
        if not (0.0 <= v <= 100.0):
            problems[2].append(f"{tag} line~{i}: accuracy {v} out of 0..100")

# ============================================================ CHECK 3
def check_log_match(data, logpath_fn, tag):
    for r in data:
        status = (r.get("status") or "").upper()
        raw = (r.get("accuracy") or "").strip()
        try:
            csv_acc = float(raw)
        except (TypeError, ValueError):
            csv_acc = None
        lp = logpath_fn(r)
        la = log_acc(lp)
        ident = f"{r['table']} {r['dataset']} s{r.get('shots')} seed{r.get('seed')}"
        if status == "FAILED":
            if isinstance(la, float):
                problems[3].append(f"{tag} {ident}: CSV=FAILED but log has {la}  <-- the known bug")
            continue
        # status OK / numeric
        if la == "NO_LOG":
            problems[3].append(f"{tag} {ident}: CSV OK ({raw}) but log missing ({os.path.relpath(lp, RES)})")
        elif la is None:
            problems[3].append(f"{tag} {ident}: CSV OK ({raw}) but log has NO 'Final test accuracy'")
        elif csv_acc is None:
            problems[3].append(f"{tag} {ident}: CSV accuracy empty/unparseable ({raw!r}) but log has {la} (recoverable)")
        elif abs(csv_acc - la) > 0.01:
            problems[3].append(f"{tag} {ident}: CSV={csv_acc} != log={la}")

check_log_match(rows, table_logpath, "tbl")
check_log_match(fig3, fig3_logpath, "fig3")

# reverse direction: logs with a real accuracy but NO OK csv row
csv_keys = {(TABLE_LOGDIR[r["table"]], r["dataset"], r["shots"], r["seed"])
            for r in rows if (r.get("status") or "").upper() == "OK"}
for table, d in TABLE_LOGDIR.items():
    base = os.path.join(RES, d)
    if not os.path.isdir(base):
        continue
    for ds in os.listdir(base):
        dd = os.path.join(base, ds)
        if not os.path.isdir(dd):
            continue
        for fn in os.listdir(dd):
            m = re.match(r"s(\d+)_seed(\d+)\.log", fn)
            if not m:
                continue
            shots, seed = m.group(1), m.group(2)
            la = log_acc(os.path.join(dd, fn))
            if isinstance(la, float) and (d, ds, shots, seed) not in csv_keys:
                problems[3].append(f"ORPHAN LOG {d}/{ds}/{fn}: log={la} but no OK CSV row")

# ============================================================ CHECK 4
# means
means = {}
for r in rows:
    if (r.get("status") or "").upper() != "OK":
        continue
    try:
        means.setdefault((r["table"], r["dataset"], r["shots"]), []).append(float(r["accuracy"]))
    except ValueError:
        pass

# table4 strict vs committed paper reference
if os.path.isfile(PAPER_T4):
    ref = {}
    for r in read_csv(PAPER_T4):
        try:
            ref[(r["dataset"], r["shots"])] = float(r["paper_table4"])
        except (KeyError, ValueError):
            pass
    checked = 0
    for (t, d, s), accs in means.items():
        if t != "table4":
            continue
        if (d, s) in ref:
            checked += 1
            diff = statistics.mean(accs) - ref[(d, s)]
            if abs(diff) > 0.3:
                problems[4].append(f"table4 {d} s{s}: ours={statistics.mean(accs):.2f} "
                                   f"paper={ref[(d, s)]:.2f} (d={diff:+.2f})")
    print(f"[check4] table4 compared {checked} cells vs committed paper reference.")
else:
    problems[4].append("table4 paper reference file missing")

# table3 / table5: no committed reference -> plausibility audit (std outliers)
for t in ["table3", "table5"]:
    for (tt, d, s), accs in sorted(means.items()):
        if tt != t:
            continue
        if len(accs) > 1 and statistics.stdev(accs) > 5.0:
            problems[4].append(f"PLAUSIBILITY {t} {d} s{s}: seed spread std={statistics.stdev(accs):.2f} "
                               f"(accs={[round(a,2) for a in accs]})")
print("[check4] table3/table5: no committed paper reference -> plausibility (std) audit only.")

# ---------------------------------------------------------------- report
names = {1: "Completeness", 2: "Parse integrity", 3: "FAILED<->log match", 4: "Paper / plausibility"}
print("\n" + "=" * 60)
print("RESULTS AUDIT REPORT")
print("=" * 60)
allclean = True
for i in [1, 2, 3, 4]:
    p = problems[i]
    status = "PASS" if not p else f"FAIL ({len(p)})"
    print(f"\n[{i}] {names[i]}: {status}")
    for line in p[:40]:
        print(f"    - {line}")
    if len(p) > 40:
        print(f"    ... and {len(p) - 40} more")
    allclean = allclean and not p
print("\n" + "=" * 60)
print("CLEAN BILL OF HEALTH" if allclean else "PROBLEMS FOUND - see above")
print("=" * 60)
sys.exit(0 if allclean else 1)
