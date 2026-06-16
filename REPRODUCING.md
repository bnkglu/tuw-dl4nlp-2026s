# Reproducing CLIP-LoRA — Step-by-Step Guide

A practical runbook for setting up the environment, getting all datasets (including the
manual ones), and running the reproduction experiments.

- **Plan / which tables & figures we target:** [`docs/reproduction_details.md`](docs/reproduction_details.md)
- **Bugs, gotchas, and fixes:** [`docs/reproduction_notes.md`](docs/reproduction_notes.md)
- **Script reference:** [`reproducing-clip-lora/scripts/README.md`](reproducing-clip-lora/scripts/README.md)

All commands below assume you are in the code directory:

```bash
cd reproducing-clip-lora
```

---

## 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate

# PyTorch — pick the build that matches your GPU:
#   - Modern GPUs (A40/A100/RTX 30xx+):  the default CUDA build is fine
pip install torch torchvision
#   - OLDER GPUs (Tesla V100, compute capability 7.0): use a CUDA 12.1/11.8 build,
#     otherwise you get "no kernel image is available" (see reproduction_notes.md §2):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

GPU note: ImageNet aside, one few-shot run takes ~1–9 minutes depending on backbone/shots.

---

## 2. Datasets

All data lives under `reproducing-clip-lora/datasets/`. The downloader handles most of it;
three datasets need a manual step first.

### 2a. Automatic datasets (no action needed)

These are downloaded, extracted, and restructured automatically by the script:
**Caltech101, OxfordPets, Flowers102, Food101, FGVCAircraft, DTD, EuroSAT, UCF101**
(plus all the split JSON files). Working mirrors are used for URLs that have since died — see
`docs/reproduction_notes.md` §3.

### 2b. Manual datasets — place the archive first, then the script extracts it

#### ImageNet (ILSVRC2012) — the big one (~144 GB, ~2.5 h to download)
Requires a (free) account. First **sign up / log in** at image-net.org, then go to the
downloads page:
<https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php>

You need these three files:

| File on the site | Size | MD5 | Saves as (the link's own filename) |
|:--|:--|:--|:--|
| Training images (Task 1 & 2) | 138 GB | `1d675b47d978889d74fa0da5fadfb00e` | `ILSVRC2012_img_train.tar` |
| Validation images (all tasks) | 6.3 GB | `29b22e2961454d5413ddabcf34fc5622` | `ILSVRC2012_img_val.tar` |
| Development kit (Task 1 & 2) | 2.5 MB | — | `ILSVRC2012_devkit_t12.tar.gz` |

Easiest way to pull them straight onto the server (no browser download + re-upload):

1. On the downloads page, **do not left-click** the links ("Training images (Task 1 & 2)",
   "Validation images (all tasks)", "Development kit (Task 1 & 2)"). Instead **right-click each
   link → "Copy link address"** (Chrome) / "Copy Link" (Firefox/Safari). The copied URL already
   ends in the correct filename (e.g. `.../ILSVRC2012_img_train.tar`), so `wget` saves it with
   the right name — no renaming needed.
2. In the terminal, go to the datasets folder and `wget` each copied URL:
   ```bash
   cd reproducing-clip-lora/datasets
   wget "<paste copied Training images URL>"      # ILSVRC2012_img_train.tar  (138 GB, the long one)
   wget "<paste copied Validation images URL>"     # ILSVRC2012_img_val.tar    (6.3 GB)
   wget "<paste copied Development kit URL>"        # ILSVRC2012_devkit_t12.tar.gz
   ```
   (Keep the quotes — the URLs contain `&` and other characters the shell would otherwise split on.)
3. Verify the downloads (optional but recommended for 144 GB):
   ```bash
   md5sum ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar
   ```
   The sums must match the table above.

That's the whole manual part — just leave the three tarballs in `reproducing-clip-lora/datasets/`.
Everything else (extracting the train tar + its 1,000 synset sub-tars, extracting the val tar,
and running `valprep.sh` to sort the 50,000 flat val images into class folders) is done
**automatically** by `download_datasets.sh` in step 2c. (Why `valprep.sh` is needed: see
`docs/reproduction_notes.md` §4.)

#### StanfordCars (~2 GB) — original servers are dead, use Kaggle
This one has to be **downloaded manually** from Kaggle (it needs a Kaggle login, so `wget`
won't work on the link directly). On your own machine, open:
<https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars>
and click the **"Download"** button (or `kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars`).

- **Rename the downloaded zip to `stanford_cars.zip`** and put it into
  `reproducing-clip-lora/datasets/` (upload it to the server there if you downloaded on a laptop).
- The downloader will unzip it and restructure it into `datasets/StanfordCars/` automatically.

#### SUN397 — skipped
The original Princeton URLs are dead and the HuggingFace mirror ships parquet files (not raw
JPEGs). We **skip SUN397** in all runs (the run scripts exclude it). If you ever need it, see
`docs/reproduction_notes.md` §3.

### 2c. Run the downloader

```bash
bash scripts/download_datasets.sh
```
- Safe to re-run: it skips what's already in place.
- It extracts/organizes the manual archives you dropped in `datasets/` (ImageNet tars,
  `stanford_cars.zip`).
- At the end it prints **MANUAL ACTIONS REQUIRED** listing only datasets still missing — a
  fully successful run shows none (except SUN397, which we skip).

### 2d. Verify the layout

```bash
ls datasets/*.py | wc -l        # 13  (loader modules present)
ls datasets                     # Caltech101 OxfordPets Flower102 Food101 fgvc_aircraft
                                #   DTD eurosat UCF101 StanfordCars imagenet ...
ls datasets/imagenet            # train/  val/  classnames.txt
ls datasets/eurosat             # 2750/   split_zhou_EuroSAT.json
```

---

## 3. Sanity check (do this before the full runs)

```bash
bash scripts/run_sanity_check.sh ViT-B/16 datasets
```
It should print a **zero-shot** accuracy and a **final test** accuracy on EuroSAT 4-shot
(~1–2 min on an A40). If that works, the whole pipeline (data + GPU + training + logging) is
good.

---

## 4. Run the experiments

> **Reproduction progress (as of this writing):**
> - ✅ **Table 3 (ViT-B/16)** — complete (150/150 runs, 0 failed)
> - ✅ **Figure 3 — EuroSAT** — complete (129/129 runs, 0 failed)
> - ⏳ **Figure 3 — ImageNet** — in progress
> - ⬜ **Table 4 (ViT-B/32)** and **Figure 3 — StanfordCars** — not started yet

Run from the repo root (`reproducing-clip-lora/`) so `logs/` lands here. Master logs go to
`logs/` (git-ignored); per-run logs and result CSVs go under `results/` (committed).

```bash
mkdir -p logs

# Phase 1 — Table 3 (ViT-B/16) and Table 4 (ViT-B/32): 10 datasets × {1,2,4,8,16} shots × seeds {1,2,3}
nohup bash scripts/run_table_3.sh datasets > logs/table3_run.log 2>&1 &   # teammate 1 (ViT-B/16)
nohup bash scripts/run_table_4.sh datasets > logs/table4_run.log 2>&1 &   # teammate 2 (ViT-B/32)

# Phase 2 — Figure 3 ablations (ViT-B/16, 4-shot), one dataset at a time
nohup bash scripts/run_fig3.sh eurosat       datasets > logs/fig3_eurosat.log 2>&1 &
nohup bash scripts/run_fig3.sh imagenet      datasets > logs/fig3_imagenet.log 2>&1 &
nohup bash scripts/run_fig3.sh stanford_cars datasets > logs/fig3_cars.log 2>&1 &
```

Monitor:
```bash
tail -f logs/table3_run.log
cat results/clip_lora_results.csv
```
The runs are **resume-safe** — re-running skips any run whose log already contains
`Final test accuracy`.

---

## 5. Collect results

```bash
# Tables (Table 3 / Table 4)
python scripts/aggregate_results.py

# Figure 3 uses its own CSV (so it can run on a second server without clashing):
python scripts/aggregate_results.py --csv results/clip_lora_fig3.csv --out results/clip_lora_fig3_summary.csv
```
Groups runs by config (ignoring seed) and reports **mean ± std accuracy over seeds** plus
**mean / total wall-clock time** per config. Writes the seed-averaged summary CSV — that's the
table to compare against the paper.

---

## 6. Output layout

```
reproducing-clip-lora/
├── logs/                              # nohup master logs (git-ignored)
└── results/
    ├── clip_lora_results.csv          # table3/table4 per-run rows (committed)
    ├── clip_lora_summary.csv          # seed-averaged tables (committed)
    ├── clip_lora_fig3.csv             # figure-3 per-run rows (committed)
    ├── clip_lora_fig3_summary.csv     # seed-averaged figure-3 (committed)
    ├── table3/<dataset>/*.log         # per-run logs, ViT-B/16
    ├── table4/<dataset>/*.log         # per-run logs, ViT-B/32
    └── fig3/<dataset>/*.log           # per-run logs, ablations
```

Stuck? Almost every error we hit (and its fix) is documented in
[`docs/reproduction_notes.md`](docs/reproduction_notes.md).
