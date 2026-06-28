# Reproducibility and Extension Challenge

**Course:** Deep Learning for Natural Language Processing (192.039-2026S)  
**Group Members:** Hasan Berke Bankoglu (12432802), Zarek Asif (12432949)  
**Contact:** e12432802@student.tuwien.ac.at

## Selected Paper

**Title:** Low-Rank Few-Shot Adaptation of Vision-Language Models  
**ArXiv ID:** [2405.18541](https://arxiv.org/abs/2405.18541)

## Project Objective

The goal of this project is to reproduce key experiments from the selected research paper and extend it.

## Deliverables

1. **GitHub Repository (This Repo):** Contains the full implementation of the paper's reproduction and any extensions.
2. **Presentation:** A 20-minute presentation demonstrating the understanding, reproduction, and extension of the chosen paper.
3. **Individual Session:** A follow-up discussion regarding personal contributions and questions related to the reproduction and extension tasks.

## Methodology

### 1. Reproduction

- **Objective:** Present the reproduction of the paper’s key experiments.
- **[How to Reproduce (step-by-step guide)](REPRODUCING.md):** Hands-on runbook — environment setup, downloading every dataset (including the manual ones: ImageNet and StanfordCars), running the experiments, and collecting results.
- **[Reproduction Plan & Details](docs/reproduction_details.md):** A concise breakdown of our step-by-step reproduction plan, including the specific tables, figures, datasets we intend to target, and experiment details in the paper.
- **[Reproduction Notes & Fixes](docs/reproduction_notes.md):** Tracks environment bugs encountered during implementation (e.g., NumPy and PyTorch/CUDA versioning) and how we fixed them to ensure the original codebase runs successfully today.

### 2. Extension — Knowledge-Preserving CLIP-LoRA (KL anchor to zero-shot)

- **Objective:** Reduce the few-shot drift/overfitting of CE-only CLIP-LoRA by anchoring the adapted model to the **frozen zero-shot CLIP**.
- **Method:** add a KL-distillation term to the loss — `L = CE + lambda * T^2 * KL(zero-shot || LoRA)`. A small ablation (datasets x shots x lambda{0.1,0.3,1.0} x T{4,8}) selected **lambda = 0.1, T = 8**.
- **Result (ViT-B/16, 10 datasets x 5 shots x 3 seeds):** mean **+0.47** accuracy over the CE-only baseline (improves 35/50 cells), with the largest gains exactly where CE-only overfits — **Food101 +1.9, EuroSAT +1.2, DTD +0.9** (averaged over shots); only FGVC dips slightly. KL also improves on the paper's original CLIP-LoRA numbers (~+0.3 avg). Inference cost is unchanged; training adds one frozen-CLIP teacher forward per batch.
- **Code/results:** KL loss in `reproducing-clip-lora/lora.py`; scripts `run_kl_ablation.sh` and `run_kl_table3.sh`; per-run results in `reproducing-clip-lora/results/clip_lora_kl.csv`.

## Acknowledgements and External Code

This project builds upon the official PyTorch implementation of CLIP-LoRA, which can be found at: [MaxZanella/CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA).

If you find this work or the original repository useful, please consider citing their paper:

```bibtex
@inproceedings{zanella2024low,
  title={Low-Rank Few-Shot Adaptation of Vision-Language Models},
  author={Zanella, Maxime and Ben Ayed, Ismail},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={1593--1603},
  year={2024}
}
```