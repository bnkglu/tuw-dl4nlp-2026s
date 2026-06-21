# Zarek StanfordCars Reproduction Results

This folder contains Zarek Asif's StanfordCars reproduction outputs for the CLIP-LoRA project.

## Included results

### ViT-B/16 StanfordCars
Completed runs:
- 1-shot, seeds 1, 2, 3
- 2-shot, seeds 1, 2, 3
- 4-shot, seeds 1, 2, 3
- 8-shot, seeds 1, 2, 3

### ViT-B/32 StanfordCars
Completed runs:
- 1-shot, seeds 1, 2, 3

## Excluded result

The 16-shot seed 1 StanfordCars run is intentionally excluded because the JupyterLab/server session crashed or was interrupted before a final test accuracy was printed.

## Files

- `zarek_stanford_cars_summary.csv`: clean per-run summary extracted from completed logs.
- `logs/`: terminal logs for completed runs.
- `checkpoints/`: saved LoRA checkpoint files.
- `notes/`: experiment notes collected during reproduction.

## Default CLIP-LoRA settings used

- Dataset: StanfordCars
- Rank: r = 2
- Target matrices: q, k, v
- Encoder: both vision and text
- Placement: all layers
- Dropout: 0.25
- Learning rate: 0.0002
- Batch size: 32
