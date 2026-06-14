# Reproduction Notes and Bug Fixes

This file tracks the changes and bug fixes made to the original `reproducing-clip-lora` codebase to ensure it runs successfully on modern Python environments.

## 1. NumPy Scalar Conversion Error

**File**: `reproducing-clip-lora/utils.py`
**Function**: `cls_acc(output, target, topk=1)`

**Issue**:
Newer versions of `numpy` (>= 1.24) no longer allow calling `float()` directly on a 1-dimensional array returned by `tensor.cpu().numpy()`. This caused the evaluation loop to crash with:
`TypeError: only 0-dimensional arrays can be converted to Python scalars`

**Fix**:
We commented out the original line and replaced it with a direct call to PyTorch's `.item()`, which avoids the NumPy version incompatibility entirely.

```python
# Old code:
# acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())

# New code:
acc = correct[: topk].reshape(-1).float().sum(0).item()
```

## 2. PyTorch CUDA Compute Capability Incompatibility (V100 GPU)

**Environment**: NVIDIA Tesla V100 GPU (Compute Capability 7.0)

**Issue**:
Running the code on a Tesla V100 with newer pre-compiled PyTorch binaries (which often drop support for older architectures) results in a CUDA error:
`torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device`
This happens because the installed PyTorch version lacks the specific kernel code (`sm_70`) for the Volta architecture.

**Fix**:
Uninstall the existing PyTorch installation and reinstall it specifying a CUDA version that retains support for Compute Capability 7.0 (such as CUDA 12.1 or 11.8).

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Expired Dataset Download URLs

Many of the original dataset URLs referenced in `DATASETS.md` have expired or become unreachable. We tested all URLs on 2026-06-13 and found the following issues:

### Working URLs (no changes needed)
- ✅ OxfordPets (images + annotations)
- ✅ Flowers102 (images + labels)
- ✅ Food101
- ✅ FGVCAircraft
- ✅ DTD

### Expired URLs and Alternatives

#### Caltech101 — `404 Not Found`
- **Original (dead):** `http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz`
- **Alternative:** `https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1`
- **Nested-archive gotcha:** this zip does **not** contain an extracted image folder. It
  unzips to `caltech-101/101_ObjectCategories.tar.gz` (a 131 MB tarball) + `Annotations.tar`.
  You must extract the inner `101_ObjectCategories.tar.gz` and place the resulting folder at
  `Caltech101/101_ObjectCategories` (where `caltech101.py` reads images). `download_datasets.sh`
  does this automatically; if extracting by hand, don't forget this extra step.

#### StanfordCars — `500 Internal Server Error` / `404`
All 4 Stanford AI server URLs are completely down:
- `http://ai.stanford.edu/~jkrause/car196/cars_train.tgz`
- `http://ai.stanford.edu/~jkrause/car196/cars_test.tgz`
- `https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz`
- `http://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat`

- **Alternative (Kaggle):** `https://www.kaggle.com/datasets/rickyyyyyyy/torchvision-stanford-cars`
  - Requires Kaggle API key configuration: `kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars`

#### SUN397 — `404 Not Found`
Both Princeton server URLs are dead:
- `http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz`
- `https://vision.princeton.edu/projects/2010/SUN/download/Partitions.zip`

- **Alternative (HuggingFace):** `https://huggingface.co/datasets/1aurent/SUN397` (~39.5 GB)
  - **Critical Warning:** The HuggingFace dataset stores the images inside 93 massive `.parquet` database files instead of raw JPEGs. PyTorch's `torchvision.datasets.SUN397` expects raw JPEGs. Downloading this will require you to write a custom Python script to extract over 100,000 images from the parquets before PyTorch can read them.

#### EuroSAT — `403 Forbidden`
- **Original (blocked):** `http://madm.dfki.de/files/sentinel/EuroSAT.zip`
- **Alternative (Zenodo):** `https://zenodo.org/records/7711810`
  - RGB version (94.7 MB): `https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1`
  - Multispectral version (2.1 GB): `https://zenodo.org/records/7711810/files/EuroSAT_MS.zip?download=1`
  - **Note:** Use the RGB version for CLIP-LoRA reproduction.

### Google Drive Files
The split JSON files and UCF101 midframes are hosted on Google Drive. Standard `wget`/`curl` downloads an HTML preview page instead of the actual file. Use either:
- **`gdown`**: `pip install gdown && gdown "https://drive.google.com/uc?id=FILE_ID"`
- **Direct URL format with `wget`**: `wget "https://drive.google.com/uc?export=download&id=FILE_ID" -O output_file`

**Note:** For larger Google Drive files (>100MB), Google shows a virus scan confirmation page. In that case, `gdown` handles it automatically, while `wget`/`curl` may require additional cookie handling.

**How this shows up:** if `wget`/`curl` saved the HTML preview page instead of the real file,
the split JSON is not valid JSON and the loader fails with
`json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)`. A common giveaway is
a split JSON that is around 70 KB. Delete the bad file and re-download it with `gdown` (or the
`uc?export=download` URL form).

## 4. ImageNet Setup

The standard ImageNet (ILSVRC2012) dataset requires specific filenames for PyTorch evaluation. You must log in to `image-net.org` and manually download the following exact files:
- `ILSVRC2012_img_train.tar`
- `ILSVRC2012_img_val.tar`
- `ILSVRC2012_devkit_t12.tar.gz`

**Challenge:** The total size of these files is over **144 GB** (`train.tar` is ~138 GB, `val.tar` is ~6 GB, and the devkit is very small), making it the largest and most computationally heavy dataset to download and extract for this reproduction.

**Discrepancy between `DATASETS.md` and the code:** `DATASETS.md` says to place the
train/val folders under `imagenet/images/` (i.e. `imagenet/images/train`,
`imagenet/images/val`). However, the actual loader `datasets/imagenet.py` (lines 210–212)
reads them from `imagenet/train` and `imagenet/val` — it uses `self.dataset_dir`, **not**
`self.image_dir`. Following the `DATASETS.md` layout therefore produces
`FileNotFoundError: 'datasets/imagenet/train'`.

**Our choice:** we leave both the upstream code and `DATASETS.md` untouched and instead
organize the extracted data to match what the code reads. `scripts/download_datasets.sh`
extracts the tars directly into `imagenet/train` and `imagenet/val`. The final structure is:

```
$DATA/
  imagenet/
    train/
      n01440764/
      n01443537/
      ...
    val/
      n01440764/
      n01443537/
      ...
    classnames.txt
```
*(Note: `classnames.txt` is downloaded automatically by the dataset download script from Google Drive. It is not actually read by `imagenet.py`, which uses a hardcoded class list, but is kept for completeness.)*

**The validation set needs reorganizing (`valprep.sh`).** `DATASETS.md` shows a `val/` folder
as if it were already class-organized like `train/`, but it is not. `ILSVRC2012_img_train.tar`
contains 1,000 per-class sub-tars (`n01440764.tar`, …) that extract into class folders, so
`train/` ends up correct. `ILSVRC2012_img_val.tar`, however, extracts to **50,000 flat
files** (`ILSVRC2012_val_00000001.JPEG`, …) with no class folders. The loader uses
`datasets.ImageFolder` on `val/` (`imagenet.py:212`), which requires one subfolder per class,
so the flat val dump will not load. `scripts/download_datasets.sh` therefore runs the standard
[`valprep.sh`](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
after extracting `val/`, which moves each of the 50,000 images into its correct `n*/` class
folder using the fixed validation→class mapping. After it runs, `val/` matches `train/`'s
structure and `ImageFolder` works.

## 5. Dataset Folder-Name Mismatch (code vs. `DATASETS.md`)

The dataset loaders in `datasets/*.py` expect folder names that differ from what
`DATASETS.md` and the raw downloads provide. This is the same kind of code-vs-doc
discrepancy as the ImageNet one (§4), but it affects almost every dataset. Downloading a
dataset by hand into the `DATASETS.md` name produces a "folder not found" error.

| Raw download / `DATASETS.md` name | Name the loader code expects |
|:--|:--|
| `caltech-101/` | `Caltech101/` (images under `101_ObjectCategories/`) |
| `oxford_pets/` | `OxfordPets/` |
| `oxford_flowers/` | `Flower102/` |
| `food-101/` | `Food101/` |
| `dtd/` | `DTD/` |
| `ucf101/` | `UCF101/` |
| `EuroSAT_RGB/` (Zenodo) | `eurosat/2750/` |

**Our choice:** `scripts/download_datasets.sh` downloads each dataset and then renames /
restructures it into the exact name the loader expects, so no code or `DATASETS.md` change
is needed. If you download a dataset manually, rename its folder to match the right-hand
column above.

## 6. `--eval_only` Requires Trained Weights (expected crash)

Running with `--eval_only` first prints the **zero-shot CLIP accuracy**, then tries to
**load a previously trained LoRA weights file** to evaluate it. If you have not trained and
saved weights (i.e. no `--save_path` was used), it crashes with:

```
FileNotFoundError: File None/vitb16/<dataset>/16shots/seed1/lora_weights.pt does not exist.
```

This is expected, not a bug — the zero-shot accuracy printed **above** the traceback is
valid. Use `--eval_only` only to read the zero-shot baseline, or after a real training run
that saved weights via `--save_path`.

## 7. `ModuleNotFoundError: No module named 'datasets.oxford_pets'`

**Symptom:** `python main.py ...` fails immediately at
`from datasets import build_dataset` → `datasets/__init__.py` →
`from .oxford_pets import OxfordPets` with
`ModuleNotFoundError: No module named 'datasets.oxford_pets'`.

**Cause:** the loader module files are missing from the local `datasets/` package on that
machine. The traceback points at the **local** `datasets/__init__.py` (so the package itself
is found), but the sibling module file (`datasets/oxford_pets.py`) isn't present, so the
relative import fails. This is **not** a clash with the HuggingFace `datasets` library — that
would instead produce an error pointing at a `site-packages/datasets/...` path.

**Fix:** restore the loader files (they are tracked in git):
```bash
ls datasets/*.py        # should list 13 files
git checkout -- datasets/   # if any are missing, restore them (or re-pull the repo)
```

## 8. Missing or Manual Datasets Summary

Due to expired server URLs and ToS restrictions, the `download_datasets.sh` script cannot completely download all 11 datasets automatically. The following datasets require manual download/intervention:
- **ImageNet:** Requires manual download (140GB+) and extraction.
- **StanfordCars:** Original URLs are dead. Requires manual Kaggle download and potential directory restructuring.
- **SUN397:** Original URLs are dead. Requires manual download and extraction.
- **UCF101:** The midframes zip file is hosted on Google Drive and frequently hits download limits or requires `gdown`/browser download.
- **Food101:** Occasional timeout/block from the original ETH Zurich server.
