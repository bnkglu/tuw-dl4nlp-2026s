#!/bin/bash
# =============================================================================
# Download all 11 datasets for CLIP-LoRA reproduction
# Usage: bash download_datasets.sh [DATA_DIR]
#   DATA_DIR defaults to ./datasets
#
# Updated 2026-06-13: Uses alternative mirrors for expired URLs.
# Uses gdown for Google Drive files to avoid large-file virus scan warnings.
# =============================================================================

set -e

# Move to repo root so relative paths work
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="${1:-./datasets}"
mkdir -p "$DATA_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[1;34m'
NC='\033[0m'

PASS=()
FAIL=()

if ! command -v gdown &> /dev/null; then
    echo -e "${BLUE}gdown not found. Installing gdown via pip...${NC}"
    pip install gdown -q
fi

# Helper: download with wget (preferred) or curl
download() {
    local url="$1"
    local output="$2"
    echo "  Downloading: $(basename "$output")"
    if command -v wget &> /dev/null; then
        wget -q --show-progress -O "$output" "$url" && return 0
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "$url" && return 0
    fi
    return 1
}

gdrive_download() {
    local file_id="$1"
    local output="$2"
    local url="https://drive.google.com/uc?id=${file_id}"
    echo "  Downloading from Google Drive: $(basename "$output")"
    
    if command -v gdown &> /dev/null; then
        gdown -q "$url" -O "$output" && return 0
    fi
    
    # Fallback to wget/curl if gdown is not installed
    local export_url="https://drive.google.com/uc?export=download&id=${file_id}"
    if command -v wget &> /dev/null; then
        wget -q --show-progress --no-check-certificate "$export_url" -O "$output" 2>/dev/null
        if file "$output" 2>/dev/null | grep -q "HTML"; then
            echo "  Large file detected, handling confirmation..."
            local confirm_url="https://drive.google.com/uc?export=download&id=${file_id}&confirm=t"
            wget -q --show-progress --no-check-certificate "$confirm_url" -O "$output"
        fi
    elif command -v curl &> /dev/null; then
        curl -L -o "$output" "${export_url}&confirm=t"
    fi
    
    if file "$output" 2>/dev/null | grep -q "HTML"; then
        echo -e "  ${RED}WARNING: Downloaded file appears to be HTML, not the actual data.${NC}"
        echo -e "  ${RED}Please run: pip install gdown${NC}"
        return 1
    fi
    return 0
}

echo "=============================================="
echo " CLIP-LoRA Dataset Downloader (Updated)"
echo " Target directory: $DATA_DIR"
echo "=============================================="
echo ""

# ==============================
# 1. Caltech101 (ALTERNATIVE URL)
# ==============================
echo -e "${BLUE}[1/11] Caltech101${NC}"
CALTECH_DIR="$DATA_DIR/Caltech101"
mkdir -p "$CALTECH_DIR"

# Original URL is dead (404). Using new Caltech data portal.
if download "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1" "$CALTECH_DIR/caltech-101.zip"; then
    unzip -qo "$CALTECH_DIR/caltech-101.zip" -d "$CALTECH_DIR"
    rm "$CALTECH_DIR/caltech-101.zip"
    rm -rf "$CALTECH_DIR/__MACOSX" 2>/dev/null

    # The data.caltech.edu zip contains caltech-101/101_ObjectCategories.tar.gz (a TARBALL,
    # not an extracted folder) plus Annotations.tar. Extract the inner tarball, then lift the
    # resulting class folders up to Caltech101/101_ObjectCategories (what caltech101.py reads).
    if [ -f "$CALTECH_DIR/caltech-101/101_ObjectCategories.tar.gz" ]; then
        tar -xzf "$CALTECH_DIR/caltech-101/101_ObjectCategories.tar.gz" -C "$CALTECH_DIR/caltech-101"
    fi
    if [ -d "$CALTECH_DIR/caltech-101/101_ObjectCategories" ] && [ ! -d "$CALTECH_DIR/101_ObjectCategories" ]; then
        mv "$CALTECH_DIR/caltech-101/101_ObjectCategories" "$CALTECH_DIR/101_ObjectCategories"
    fi
    rm -rf "$CALTECH_DIR/caltech-101" 2>/dev/null
    echo -e "  ${GREEN}✓ Images downloaded and restructured${NC}"
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("Caltech101 - images")
fi

# Split JSON
if gdrive_download "1hyarUivQE36mY6jSomru6Fjd-JzwcCzN" "$CALTECH_DIR/split_zhou_Caltech101.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    PASS+=("Caltech101")
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("Caltech101 - split JSON")
fi
echo ""

# ==============================
# 2. OxfordPets
# ==============================
echo -e "${BLUE}[2/11] OxfordPets${NC}"
PETS_DIR="$DATA_DIR/OxfordPets"
mkdir -p "$PETS_DIR"

if download "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz" "$PETS_DIR/images.tar.gz"; then
    tar -xzf "$PETS_DIR/images.tar.gz" -C "$PETS_DIR"
    rm "$PETS_DIR/images.tar.gz"
    echo -e "  ${GREEN}✓ Images downloaded${NC}"
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("OxfordPets - images")
fi

if download "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz" "$PETS_DIR/annotations.tar.gz"; then
    tar -xzf "$PETS_DIR/annotations.tar.gz" -C "$PETS_DIR"
    rm "$PETS_DIR/annotations.tar.gz"
    echo -e "  ${GREEN}✓ Annotations downloaded${NC}"
else
    echo -e "  ${RED}✗ Annotations download failed${NC}"
    FAIL+=("OxfordPets - annotations")
fi

if gdrive_download "1501r8Ber4nNKvmlFVQZ8SeUHTcdTTEqs" "$PETS_DIR/split_zhou_OxfordPets.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    PASS+=("OxfordPets")
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("OxfordPets - split JSON")
fi
echo ""

# ==============================
# 3. StanfordCars (EXPIRED — Kaggle alternative)
# ==============================
echo -e "${BLUE}[3/11] StanfordCars${NC}"
CARS_DIR="$DATA_DIR/StanfordCars"
mkdir -p "$CARS_DIR"

echo -e "  ${BLUE}⚠ All original Stanford AI server URLs are dead (500/404).${NC}"

if [ -f "$DATA_DIR/stanford_cars.zip" ]; then
    if [ ! -d "$CARS_DIR/cars_train" ]; then
        echo -e "  ${BLUE}Found stanford_cars.zip, extracting...${NC}"
        unzip -qo "$DATA_DIR/stanford_cars.zip" -d "$DATA_DIR"
        if [ -d "$DATA_DIR/stanford_cars" ]; then
            # Copy instead of move to avoid "Directory not empty" errors if files exist
            cp -R "$DATA_DIR/stanford_cars/"* "$CARS_DIR/"
            rm -rf "$DATA_DIR/stanford_cars"
        fi
        echo -e "  ${GREEN}✓ StanfordCars manually extracted${NC}"
    else
        echo -e "  ${GREEN}✓ StanfordCars already extracted${NC}"
    fi
    
    if gdrive_download "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT" "$CARS_DIR/split_zhou_StanfordCars.json"; then
        echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
        PASS+=("StanfordCars")
    else
        echo -e "  ${RED}✗ Split JSON failed${NC}"
        FAIL+=("StanfordCars - split JSON")
    fi
else
    echo -e "  ${BLUE}  Alternative: kaggle datasets download -d rickyyyyyyy/torchvision-stanford-cars -p $DATA_DIR${NC}"
    echo -e "  ${BLUE}  (Download the zip to datasets/stanford_cars.zip and re-run this script)${NC}"
    
    if gdrive_download "1ObCFbaAgVu0I-k_Au-gIUcefirdAuizT" "$CARS_DIR/split_zhou_StanfordCars.json"; then
        echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    fi
    FAIL+=("StanfordCars - images (manual Kaggle download required)")
fi
echo ""

# ==============================
# 4. Flowers102
# ==============================
echo -e "${BLUE}[4/11] Flowers102${NC}"
FLOWERS_DIR="$DATA_DIR/Flower102"
mkdir -p "$FLOWERS_DIR"

if download "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz" "$FLOWERS_DIR/102flowers.tgz"; then
    tar -xzf "$FLOWERS_DIR/102flowers.tgz" -C "$FLOWERS_DIR"
    rm "$FLOWERS_DIR/102flowers.tgz"
    echo -e "  ${GREEN}✓ Images downloaded${NC}"
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("Flowers102 - images")
fi

if download "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat" "$FLOWERS_DIR/imagelabels.mat"; then
    echo -e "  ${GREEN}✓ Labels downloaded${NC}"
else
    echo -e "  ${RED}✗ Labels download failed${NC}"
    FAIL+=("Flowers102 - labels")
fi

gdrive_download "1AkcxCXeK_RCGCEC_GvmWxjcjaNhu-at0" "$FLOWERS_DIR/cat_to_name.json" || FAIL+=("Flowers102 - cat_to_name.json")

if gdrive_download "1Pp0sRXzZFZq15zVOzKjKBu4A9i01nozT" "$FLOWERS_DIR/split_zhou_OxfordFlowers.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    PASS+=("Flowers102")
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("Flowers102 - split JSON")
fi
echo ""

# ==============================
# 5. Food101
# ==============================
echo -e "${BLUE}[5/11] Food101${NC}"
FOOD_DIR="$DATA_DIR/Food101"

if download "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz" "$DATA_DIR/food-101.tar.gz"; then
    tar -xzf "$DATA_DIR/food-101.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/food-101.tar.gz"
    # tar extracts to food-101/, rename to Food101
    if [ -d "$DATA_DIR/food-101" ] && [ ! -d "$DATA_DIR/Food101" ]; then
        mv "$DATA_DIR/food-101" "$DATA_DIR/Food101"
    fi
    echo -e "  ${GREEN}✓ Images downloaded and restructured${NC}"
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("Food101 - images")
fi

if gdrive_download "1QK0tGi096I0Ba6kggatX1ee6dJFIcEJl" "$FOOD_DIR/split_zhou_Food101.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    PASS+=("Food101")
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("Food101 - split JSON")
fi
echo ""

# ==============================
# 6. FGVCAircraft
# ==============================
echo -e "${BLUE}[6/11] FGVCAircraft${NC}"
FGVC_DIR="$DATA_DIR/fgvc_aircraft"
mkdir -p "$FGVC_DIR"

if download "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz" "$DATA_DIR/fgvc-aircraft-2013b.tar.gz"; then
    tar -xzf "$DATA_DIR/fgvc-aircraft-2013b.tar.gz" -C "$DATA_DIR"
    if [ -d "$DATA_DIR/fgvc-aircraft-2013b/data" ]; then
        cp -r "$DATA_DIR/fgvc-aircraft-2013b/data/"* "$FGVC_DIR/"
        rm -rf "$DATA_DIR/fgvc-aircraft-2013b"
    fi
    rm -f "$DATA_DIR/fgvc-aircraft-2013b.tar.gz"
    echo -e "  ${GREEN}✓ Images downloaded${NC}"
    PASS+=("FGVCAircraft")
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("FGVCAircraft - images")
fi
echo ""

# ==============================
# 7. SUN397 (EXPIRED — HuggingFace alternative)
# ==============================
echo -e "${BLUE}[7/11] SUN397${NC}"
SUN_DIR="$DATA_DIR/SUN397"
mkdir -p "$SUN_DIR"

echo -e "  ${BLUE}⚠ Both Princeton server URLs are dead (404).${NC}"
echo -e "  ${BLUE}  Alternative (HuggingFace, ~39.5 GB): https://huggingface.co/datasets/1aurent/SUN397${NC}"
echo -e "  ${BLUE}  Download manually:${NC}"
echo -e "  ${BLUE}    pip install huggingface_hub${NC}"
echo -e "  ${BLUE}    huggingface-cli download 1aurent/SUN397 --repo-type dataset --local-dir $SUN_DIR${NC}"

if gdrive_download "1y2RD81BYuiyvebdN-JymPfyWYcd8_MUq" "$SUN_DIR/split_zhou_SUN397.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
    PASS+=("SUN397 (split only)")
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("SUN397 - split JSON")
fi
FAIL+=("SUN397 - images (manual HuggingFace download required)")
echo ""

# ==============================
# 8. DTD
# ==============================
echo -e "${BLUE}[8/11] DTD${NC}"
DTD_DIR="$DATA_DIR/DTD"

if download "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz" "$DATA_DIR/dtd-r1.0.1.tar.gz"; then
    tar -xzf "$DATA_DIR/dtd-r1.0.1.tar.gz" -C "$DATA_DIR"
    rm "$DATA_DIR/dtd-r1.0.1.tar.gz"
    # tar extracts to dtd/, rename to DTD
    if [ -d "$DATA_DIR/dtd" ] && [ ! -d "$DATA_DIR/DTD" ]; then
        mv "$DATA_DIR/dtd" "$DATA_DIR/DTD"
    fi
    echo -e "  ${GREEN}✓ Images downloaded and restructured${NC}"
    PASS+=("DTD")
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("DTD - images")
fi

if gdrive_download "1u3_QfB467jqHgNXC00UIzbLZRQCg2S7x" "$DTD_DIR/split_zhou_DescribableTextures.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("DTD - split JSON")
fi
echo ""

# ==============================
# 9. EuroSAT (ALTERNATIVE URL — Zenodo)
# ==============================
echo -e "${BLUE}[9/11] EuroSAT${NC}"
EUROSAT_DIR="$DATA_DIR/eurosat"
mkdir -p "$EUROSAT_DIR"

# Original URL is blocked (403). Using Zenodo RGB version.
if download "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1" "$EUROSAT_DIR/EuroSAT_RGB.zip"; then
    unzip -qo "$EUROSAT_DIR/EuroSAT_RGB.zip" -d "$EUROSAT_DIR"
    rm "$EUROSAT_DIR/EuroSAT_RGB.zip"
    # Code expects eurosat/2750/, Zenodo extracts to EuroSAT_RGB/
    if [ -d "$EUROSAT_DIR/EuroSAT_RGB" ] && [ ! -d "$EUROSAT_DIR/2750" ]; then
        mv "$EUROSAT_DIR/EuroSAT_RGB" "$EUROSAT_DIR/2750"
    fi
    echo -e "  ${GREEN}✓ Images downloaded and restructured (RGB → 2750/)${NC}"
    PASS+=("EuroSAT")
else
    echo -e "  ${RED}✗ Images download failed${NC}"
    FAIL+=("EuroSAT - images")
fi

if gdrive_download "1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o" "$EUROSAT_DIR/split_zhou_EuroSAT.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("EuroSAT - split JSON")
fi
echo ""

# ==============================
# 10. UCF101 (Google Drive)
# ==============================
echo -e "${BLUE}[10/11] UCF101${NC}"
UCF_DIR="$DATA_DIR/UCF101"
mkdir -p "$UCF_DIR"

if gdrive_download "10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O" "$UCF_DIR/UCF-101-midframes.zip"; then
    unzip -qo "$UCF_DIR/UCF-101-midframes.zip" -d "$UCF_DIR"
    rm "$UCF_DIR/UCF-101-midframes.zip"
    echo -e "  ${GREEN}✓ Midframes downloaded${NC}"
    PASS+=("UCF101")
else
    echo -e "  ${RED}✗ Midframes download failed${NC}"
    FAIL+=("UCF101 - midframes")
fi

if gdrive_download "1I0S0q91hJfsV9Gf4xDIjgDq4AqBNJb1y" "$UCF_DIR/split_zhou_UCF101.json"; then
    echo -e "  ${GREEN}✓ Split JSON downloaded${NC}"
else
    echo -e "  ${RED}✗ Split JSON failed${NC}"
    FAIL+=("UCF101 - split JSON")
fi
echo ""

# ==============================
# 11. ImageNet (manual)
# ==============================
echo -e "${BLUE}[11/11] ImageNet${NC}"
IMAGENET_DIR="$DATA_DIR/imagenet"
mkdir -p "$IMAGENET_DIR"

# datasets/imagenet.py loads from imagenet/train and imagenet/val, so we organize the
# extracted data there. (See docs/reproduction_notes.md for the DATASETS.md discrepancy.)
if [ -f "$DATA_DIR/ILSVRC2012_img_train.tar" ] && [ -f "$DATA_DIR/ILSVRC2012_img_val.tar" ]; then
    echo -e "  ${BLUE}Found ImageNet tar files, extracting...${NC}"

    mkdir -p "$IMAGENET_DIR/train"
    mkdir -p "$IMAGENET_DIR/val"

    # Train
    if [ ! -d "$IMAGENET_DIR/train/n01440764" ]; then
        echo -e "  ${BLUE}Extracting train tar...${NC}"
        tar -xf "$DATA_DIR/ILSVRC2012_img_train.tar" -C "$IMAGENET_DIR/train"
        echo -e "  ${BLUE}Extracting synset sub-tars...${NC}"
        find "$IMAGENET_DIR/train" -name "*.tar" | while read -r f; do
            synset_dir="${f%.tar}"
            mkdir -p "$synset_dir"
            tar -xf "$f" -C "$synset_dir"
            rm "$f"
        done
    fi

    # Val
    if [ ! -d "$IMAGENET_DIR/val/n01440764" ]; then
        echo -e "  ${BLUE}Extracting val tar...${NC}"
        tar -xf "$DATA_DIR/ILSVRC2012_img_val.tar" -C "$IMAGENET_DIR/val"
        echo -e "  ${BLUE}Organizing val images using valprep.sh...${NC}"
        curl -qL https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh -o "$IMAGENET_DIR/val/valprep.sh"
        (cd "$IMAGENET_DIR/val" && bash valprep.sh && rm valprep.sh)
    fi
    echo -e "  ${GREEN}✓ ImageNet manually extracted and organized${NC}"
    
    if gdrive_download "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF" "$IMAGENET_DIR/classnames.txt"; then
        echo -e "  ${GREEN}✓ classnames.txt downloaded${NC}"
        PASS+=("ImageNet")
    else
        echo -e "  ${RED}✗ classnames.txt failed${NC}"
        FAIL+=("ImageNet - classnames.txt")
    fi
else
    echo -e "  ${BLUE}⚠ ImageNet requires manual download from https://image-net.org${NC}"
    echo -e "  ${BLUE}  Place ILSVRC2012_img_train.tar and ILSVRC2012_img_val.tar in $DATA_DIR and re-run this script${NC}"
    
    if gdrive_download "1-61f_ol79pViBFDG_IDlUQSwoLcn2XXF" "$IMAGENET_DIR/classnames.txt"; then
        echo -e "  ${GREEN}✓ classnames.txt downloaded${NC}"
    fi
    FAIL+=("ImageNet - missing tar files")
fi
echo ""

# ==============================
# SUMMARY
# ==============================
echo "=============================================="
echo " DOWNLOAD SUMMARY"
echo "=============================================="

if [ ${#PASS[@]} -gt 0 ]; then
    echo -e "${GREEN}PASSED (${#PASS[@]}):${NC}"
    for item in "${PASS[@]}"; do
        echo -e "  ${GREEN}✓ $item${NC}"
    done
fi
echo ""
if [ ${#FAIL[@]} -gt 0 ]; then
    echo -e "${RED}FAILED / MANUAL (${#FAIL[@]}):${NC}"
    for item in "${FAIL[@]}"; do
        echo -e "  ${RED}✗ $item${NC}"
    done
    echo ""
    echo -e "${BLUE}See docs/reproduction_notes.md for alternative download links.${NC}"
fi

echo ""
echo -e "${RED}==============================================${NC}"
echo -e "${RED} MANUAL ACTIONS REQUIRED${NC}"
echo -e "${RED}==============================================${NC}"
echo -e "${RED}ImageNet${NC}: Requires manual download of 144GB from image-net.org"
echo -e "${RED}StanfordCars${NC}: Requires manual Kaggle download (URLs are dead)"
echo -e "${RED}SUN397${NC}: Requires manual HuggingFace/Kaggle download (URLs are dead)"
echo -e "${RED}UCF101 / Food101${NC}: May require manual download if Google Drive/ETH servers time out."
echo ""
echo -e "See ${BLUE}docs/reproduction_notes.md${NC} for full manual instructions."

echo ""
echo "Done."
