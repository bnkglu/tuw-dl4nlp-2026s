#!/bin/bash
# Test all URLs including the NEW alternative mirrors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=()
FAIL=()

test_url() {
    local name="$1"
    local url="$2"
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" -L --max-time 15 "$url" 2>/dev/null)
    if [[ "$http_code" == "200" || "$http_code" == "302" || "$http_code" == "301" ]]; then
        echo -e "  ${GREEN}✓ [$http_code] $name${NC}"
        PASS+=("$name")
    else
        echo -e "  ${RED}✗ [$http_code] $name${NC}"
        FAIL+=("$name")
    fi
}

echo "=============================================="
echo " Testing ALL URLs (originals + alternatives)"
echo "=============================================="
echo ""

echo -e "${YELLOW}[1] Caltech101${NC}"
test_url "Caltech101 ORIGINAL (expected dead)" "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz"
test_url "Caltech101 ALTERNATIVE" "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1"
echo ""

echo -e "${YELLOW}[2] OxfordPets${NC}"
test_url "OxfordPets - images" "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
test_url "OxfordPets - annotations" "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"
echo ""

echo -e "${YELLOW}[3] Flowers102${NC}"
test_url "Flowers102 - images" "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
test_url "Flowers102 - labels" "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
echo ""

echo -e "${YELLOW}[4] Food101${NC}"
test_url "Food101 - images" "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
echo ""

echo -e "${YELLOW}[5] FGVCAircraft${NC}"
test_url "FGVCAircraft - images" "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz"
echo ""

echo -e "${YELLOW}[6] DTD${NC}"
test_url "DTD - images" "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
echo ""

echo -e "${YELLOW}[7] EuroSAT${NC}"
test_url "EuroSAT ORIGINAL (expected dead)" "http://madm.dfki.de/files/sentinel/EuroSAT.zip"
test_url "EuroSAT ALTERNATIVE (Zenodo RGB)" "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"
echo ""

echo -e "${YELLOW}[8] SUN397${NC}"
test_url "SUN397 ORIGINAL (expected dead)" "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
echo -e "  ${YELLOW}⚠ SUN397 alternative is HuggingFace (requires HF CLI, cannot test via HTTP)${NC}"
echo ""

echo -e "${YELLOW}[9] StanfordCars${NC}"
test_url "StanfordCars ORIGINAL (expected dead)" "http://ai.stanford.edu/~jkrause/car196/cars_train.tgz"
echo -e "  ${YELLOW}⚠ StanfordCars alternative is Kaggle (requires Kaggle API, cannot test via HTTP)${NC}"
echo ""

echo -e "${YELLOW}[10] Google Drive split JSONs${NC}"
test_url "GDrive direct DL (Caltech101 split)" "https://drive.google.com/uc?export=download&id=1hyarUivQE36mY6jSomru6Fjd-JzwcCzN"
test_url "GDrive direct DL (EuroSAT split)" "https://drive.google.com/uc?export=download&id=1Ip7yaCWFi0eaOFUGga0lUdVi_DDQth1o"
test_url "GDrive direct DL (UCF101 midframes)" "https://drive.google.com/uc?export=download&id=10Jqome3vtUA2keJkNanAiFpgbyC9Hc2O"
echo ""

echo "=============================================="
echo " RESULTS"
echo "=============================================="
if [ ${#PASS[@]} -gt 0 ]; then
    echo -e "${GREEN}REACHABLE (${#PASS[@]}):${NC}"
    for item in "${PASS[@]}"; do echo -e "  ${GREEN}✓ $item${NC}"; done
fi
echo ""
if [ ${#FAIL[@]} -gt 0 ]; then
    echo -e "${RED}UNREACHABLE (${#FAIL[@]}):${NC}"
    for item in "${FAIL[@]}"; do echo -e "  ${RED}✗ $item${NC}"; done
fi
