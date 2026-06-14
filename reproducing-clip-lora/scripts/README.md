# Scripts

Helper scripts for reproducing the CLIP-LoRA experiments. All shell scripts `cd` to the
repo root (`reproducing-clip-lora/`) on startup, so you can run them from anywhere, e.g.
`bash scripts/run_table_3.sh` or `cd scripts && bash run_table_3.sh` — both work.

For the *why* behind dataset quirks, fixes, and known gotchas, see
[`docs/reproduction_notes.md`](../../docs/reproduction_notes.md). For the experiment plan,
see [`docs/reproduction_details.md`](../../docs/reproduction_details.md).

## Overview

| Script | Purpose |
|:--|:--|
| `download_datasets.sh` | Download, extract, and **restructure** all datasets into the exact folder names the loaders expect. |
| `test_dataset_urls.sh` | Quick reachability check (no downloads) — see which dataset URLs/mirrors are still alive. |
| `run_sanity_check.sh` | One fast run to verify the environment, data paths, and training loop work. |
| `run_table_3.sh` | Phase 1.A — full few-shot grid on **ViT-B/16** (Table 3). |
| `run_table_4.sh` | Phase 1.B — full few-shot grid on **ViT-B/32** (Table 4). |
| `run_fig3.sh` | Phase 2 — Figure 3 ablations (rank × matrices × encoder + placement) for one dataset. |
| `aggregate_results.py` | Collapse per-run results into seed-averaged mean ± std for the report. |

## Typical workflow

```bash
# 0. (once) get the data — place manual archives (ImageNet tars, stanford_cars.zip)
#    in datasets/ first; the script handles the rest.
bash scripts/download_datasets.sh

# 1. sanity check (each teammate runs their backbone)
bash scripts/run_sanity_check.sh ViT-B/16 datasets   # teammate 1
bash scripts/run_sanity_check.sh ViT-B/32 datasets   # teammate 2

# (nohup master logs go in logs/, which is git-ignored)
mkdir -p logs

# 2. main grids (background, resume-safe)
nohup bash scripts/run_table_3.sh datasets > logs/table3_run.log 2>&1 &   # teammate 1
nohup bash scripts/run_table_4.sh datasets > logs/table4_run.log 2>&1 &   # teammate 2

# 3. Figure 3 ablations (per dataset)
nohup bash scripts/run_fig3.sh eurosat       datasets > logs/fig3_eurosat.log 2>&1 &
nohup bash scripts/run_fig3.sh imagenet      datasets > logs/fig3_imagenet.log 2>&1 &
nohup bash scripts/run_fig3.sh stanford_cars datasets > logs/fig3_cars.log 2>&1 &

# 4. summarize for the report
python scripts/aggregate_results.py
```

## Script details

### `download_datasets.sh`
```bash
bash scripts/download_datasets.sh [DATA_DIR]   # DATA_DIR defaults to ./datasets
```
- Downloads each dataset (using working mirrors for the expired originals), extracts it, and
  renames/restructures into the loader's expected layout (e.g. `Caltech101/101_ObjectCategories`,
  `eurosat/2750`, `imagenet/train` + `imagenet/val`).
- Uses `gdown` for Google-Drive files (auto-installs if missing).
- **Manual archives:** if you place `stanford_cars.zip` or the ImageNet tars
  (`ILSVRC2012_img_train.tar`, `ILSVRC2012_img_val.tar`) in `DATA_DIR`, it auto-extracts and
  organizes them (ImageNet val is sorted into class folders via `valprep.sh`).
- Prints a colored PASS / FAIL / MANUAL-ACTIONS summary at the end.
- Datasets needing manual download (dead URLs / ToS): **ImageNet, StanfordCars, SUN397**.

### `test_dataset_urls.sh`
```bash
bash scripts/test_dataset_urls.sh
```
Probes each dataset URL with `curl` and prints reachable (✓) vs unreachable (✗). Handy before
a big download since these URLs expire over time. Read-only, downloads nothing.

### `run_sanity_check.sh`
```bash
bash scripts/run_sanity_check.sh [BACKBONE] [DATA_DIR]   # defaults: ViT-B/16  ./datasets
```
Runs EuroSAT 4-shot, seed 1 — a quick end-to-end check before launching the full grids.

### `run_table_3.sh` / `run_table_4.sh`
```bash
bash scripts/run_table_3.sh [DATA_DIR] [LOG_DIR]   # ViT-B/16, LOG_DIR=results/table3
bash scripts/run_table_4.sh [DATA_DIR] [LOG_DIR]   # ViT-B/32, LOG_DIR=results/table4
```
- Grid: 10 datasets × shots {1,2,4,8,16} × seeds {1,2,3} = **150 runs** each.
- **Resume-safe:** a run whose `.log` already contains `Final test accuracy` is skipped.
- Appends one row per run to `results/clip_lora_results.csv`.

### `run_fig3.sh`
```bash
bash scripts/run_fig3.sh <dataset> [DATA_DIR]
```
- ViT-B/16, 4-shot. Main grid = rank {1,2,4,8,16,32} × matrix set {k, q, v, o, q v, q v k, q v k o}
  × encoder {vision, text, both}, plus a placement set {bottom, up, all} at rank 2 / `q k v` / both.
- Defaults to seed 1; edit `SEEDS=(1 2 3)` in the script for error bars.
- Same resume-safe + CSV-append behavior as the table scripts.

### `aggregate_results.py`
```bash
python scripts/aggregate_results.py [--csv PATH] [--out PATH] [--no-save]
```
- Reads `results/clip_lora_results.csv`, groups runs by config (ignoring seed), and reports
  **mean ± std accuracy over seeds** (plus **mean / total wall-clock seconds** per config) —
  the view to compare against the paper.
- Skips `FAILED` rows; if a `(config, seed)` was re-run, keeps the last `OK` row (so duplicate
  rows from retries don't inflate the average).
- Prints compact tables for `table3`/`table4` and the full config for `fig3`; writes
  `results/clip_lora_summary.csv` unless `--no-save`. Pure standard library (no pandas).

## Output layout

```
results/
├── clip_lora_results.csv      # per-run rows (table,dataset,backbone,shots,seed,rank,
│                              #   params,encoder,position,dropout,accuracy,status,
│                              #   start_time,seconds)  <- seconds = per-run wall-clock
├── clip_lora_summary.csv      # seed-averaged (written by aggregate_results.py)
├── table3/   *.log            # per-run logs (ViT-B/16)
├── table4/   *.log            # per-run logs (ViT-B/32)
└── fig3/     *.log            # per-run logs (ablations)
```
The per-run log directories are git-ignored; `clip_lora_results.csv` / `clip_lora_summary.csv`
can be committed for the report.
