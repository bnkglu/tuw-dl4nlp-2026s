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
| `run_table_5.sh` | Phase 1.C — full few-shot grid on **ViT-L/14** (Table 5). |
| `run_fig3.sh` | Phase 2 — Figure 3 ablations (rank × matrices × encoder + placement) for one dataset. |
| `run_kl_ablation.sh` | Extension — Knowledge-Preserving CLIP-LoRA: KL-to-zero-shot distillation sweep. |
| `run_kl_table3.sh` | Extension — the full Table 3 grid (all 10 datasets) with KL distillation on. |
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
nohup bash scripts/run_table_3.sh datasets > logs/table3_run.log 2>&1 &   # teammate 1 (ViT-B/16)
nohup bash scripts/run_table_4.sh datasets > logs/table4_run.log 2>&1 &   # teammate 2 (ViT-B/32)
nohup bash scripts/run_table_5.sh datasets > logs/table5_run.log 2>&1 &   # ViT-L/14 (heaviest — most GPU mem + slowest)

# 3. Figure 3 ablations (per dataset)
nohup bash scripts/run_fig3.sh eurosat       datasets > logs/fig3_eurosat.log 2>&1 &
nohup bash scripts/run_fig3.sh imagenet      datasets > logs/fig3_imagenet.log 2>&1 &
nohup bash scripts/run_fig3.sh stanford_cars datasets > logs/fig3_cars.log 2>&1 &

# 4. summarize for the report (outputs go to results/aggregated/)
python scripts/aggregate_results.py

# 5. (extension) KL-distillation ablation — own CSV, reuses Table 3 as the kl_weight=0 baseline
nohup bash scripts/run_kl_ablation.sh datasets > logs/kl_ablation.log 2>&1 &
python scripts/aggregate_results.py --csv results/clip_lora_kl.csv --out results/aggregated/clip_lora_kl_summary.csv

# 6. (extension) full Table 3 grid WITH KL — pick best kl_weight/kl_temp from step 5 (defaults 1.0 / 4)
nohup bash scripts/run_kl_table3.sh datasets 1.0 4 > logs/kl_table3.log 2>&1 &
python scripts/aggregate_results.py --csv results/clip_lora_kl.csv --out results/aggregated/clip_lora_kl_summary.csv
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

### `run_table_3.sh` / `run_table_4.sh` / `run_table_5.sh`
```bash
bash scripts/run_table_3.sh [DATA_DIR] [LOG_DIR]   # ViT-B/16, LOG_DIR=results/table3
bash scripts/run_table_4.sh [DATA_DIR] [LOG_DIR]   # ViT-B/32, LOG_DIR=results/table4
bash scripts/run_table_5.sh [DATA_DIR] [LOG_DIR]   # ViT-L/14, LOG_DIR=results/table5
```
- Grid: 10 datasets × shots {1,2,4,8,16} × seeds {1,2,3} = **150 runs** each.
- Identical except for the backbone (and the `table3`/`table4`/`table5` tag written to the CSV).
  **ViT-L/14 is the heaviest** — bigger model + checkpoint, so expect longer per-run times and
  more GPU memory.
- **Resume-safe:** a run whose `.log` already contains `Final test accuracy` is skipped.
- Appends one row per run to `results/clip_lora_results.csv`.

### `run_fig3.sh`
```bash
bash scripts/run_fig3.sh <dataset> [DATA_DIR]
```
- ViT-B/16, 4-shot. Main grid = rank {1,2,4,8,16,32} × matrix set {k, q, v, o, q v, q v k, q v k o}
  × encoder {vision, text, both}, plus a placement set {bottom, up, all} at rank 2 / `q k v` / both.
- Defaults to seed 1; edit `SEEDS=(1 2 3)` in the script for error bars.
- Same resume-safe behavior as the table scripts, but appends to its **own** CSV
  `results/clip_lora_fig3.csv` (kept separate from the table runs so Figure 3 can run
  concurrently on a second server without clashing on one shared file).

### `run_kl_ablation.sh`
```bash
bash scripts/run_kl_ablation.sh [DATA_DIR] [LOG_DIR]   # ViT-B/16, LOG_DIR=results/kl
```
- **Extension, not part of the paper reproduction.** Adds a knowledge-preserving KL term that
  distills frozen zero-shot CLIP (teacher) into the LoRA-adapted predictions (student):
  `loss = CE + kl_weight · T² · KL(teacher ‖ student)`. Motivated by the paper's note that
  CE-only LoRA underperforms on Food101 / OxfordPets for lack of regularization.
- Enabled via two new args in `main.py`: `--kl_weight` (0 = off, the CE-only baseline) and
  `--kl_temp` (softmax temperature). With `--kl_weight 0` the run is identical to the baseline.
- Grid: datasets {food101, oxford_pets, eurosat} × shots {1,4,16} × `kl_weight` {0.1,0.3,1.0}
  × `kl_temp` {4,8}, seed 1 = **54 runs**. The `kl_weight=0` baseline is **not** re-run here —
  reuse it from Table 3 (same backbone/config) in `results/clip_lora_results.csv`.
- Same resume-safe behavior, but appends to its **own** CSV `results/clip_lora_kl.csv`, which has
  two extra columns (`kl_weight`, `kl_temp`) before `accuracy`.

### `run_kl_table3.sh`
```bash
bash scripts/run_kl_table3.sh [DATA_DIR] [KL_WEIGHT] [KL_TEMP]   # ViT-B/16; defaults 1.0 / 4
```
- **Extension, broad view.** The *full* Table 3 grid (10 datasets × shots {1,2,4,8,16} × seeds
  {1,2,3} = **150 runs**) but with the KL term enabled, to see how distillation affects **every**
  dataset — not just the Food101 / OxfordPets failures the ablation targets.
- `KL_WEIGHT` / `KL_TEMP` are passed straight to `main.py`; set them to the **best values from
  `run_kl_ablation.sh`** (defaults `1.0` / `4`). The `kl_weight=0` baseline is the existing
  Table 3 in `results/clip_lora_results.csv` — **not** re-run here.
- Appends to `results/clip_lora_kl.csv` with the `kl_table3` tag (so it sits next to the ablation
  rows but stays distinguishable). Logs go to `results/kl_table3/w<KL_WEIGHT>_t<KL_TEMP>/<dataset>/`,
  so different `(weight, temp)` settings don't collide and each is independently resume-safe.
- Heavy: it's a full 150-run grid, so only run it when you have spare GPU time.

### `aggregate_results.py`
```bash
python scripts/aggregate_results.py [--csv PATH] [--out PATH] [--no-save]
```
- Reads `results/clip_lora_results.csv`, groups runs by config (ignoring seed), and reports
  **mean ± std accuracy over seeds** (plus **mean / total wall-clock seconds** per config) —
  the view to compare against the paper.
- Skips `FAILED` rows; if a `(config, seed)` was re-run, keeps the last `OK` row (so duplicate
  rows from retries don't inflate the average).
- Prints compact tables for `table3`/`table4`/`table5` and the full config for `fig3`. All written
  outputs go under **`results/aggregated/`** (created automatically); the summary defaults to
  `results/aggregated/clip_lora_summary.csv` unless `--no-save`. Pure standard library (no pandas).
- Also writes **paper-shaped** CSVs (`paper_<table>_<backbone>.csv`, e.g. `paper_table3_ViT-B-16.csv`):
  **shots as rows, datasets as columns**, mean accuracy per cell, plus an **Average column** — matching
  the paper's orientation, pasteable straight into the report. fig3 and the KL ablation are skipped
  (not a clean shots×datasets grid); `table3/4/5` and `kl_table3` are pivoted.
- **Tables vs Figure 3 use separate CSVs.** Default reads the table CSV. For Figure 3, point
  it at the fig3 CSV and a distinct output (otherwise the summary name collides):
  ```bash
  python scripts/aggregate_results.py --csv results/clip_lora_fig3.csv --out results/aggregated/clip_lora_fig3_summary.csv
  ```

## Output layout

```
results/
├── clip_lora_results.csv      # table3/table4/table5 per-run rows (table,dataset,backbone,shots,
│                              #   seed,rank,params,encoder,position,dropout,accuracy,status,
│                              #   start_time,seconds)  <- seconds = per-run wall-clock
├── clip_lora_fig3.csv         # figure-3 per-run rows (same columns)
├── clip_lora_kl.csv           # KL-extension per-run rows (+ kl_weight, kl_temp columns)
├── aggregated/                # everything aggregate_results.py writes lives here
│   ├── clip_lora_summary.csv          # seed-averaged tables (long format)
│   ├── clip_lora_fig3_summary.csv     # seed-averaged figure-3 (--csv/--out)
│   ├── clip_lora_kl_summary.csv       # seed-averaged KL ablation
│   └── paper_table3_ViT-B-16.csv      # paper-shaped pivot (shots × datasets, + Average); one per table/backbone
├── table3/<dataset>/*.log     # per-run logs (ViT-B/16), grouped by dataset
├── table4/<dataset>/*.log     # per-run logs (ViT-B/32), grouped by dataset
├── table5/<dataset>/*.log     # per-run logs (ViT-L/14), grouped by dataset
├── fig3/<dataset>/*.log       # per-run logs (ablations), grouped by dataset
├── kl/<dataset>/*.log         # per-run logs (KL ablation), grouped by dataset
└── kl_table3/w<W>_t<T>/<dataset>/*.log   # per-run logs (Table 3 + KL), grouped by setting & dataset
```
The per-run log directories are git-ignored; the raw CSVs and `aggregated/` outputs can be
committed for the report.
