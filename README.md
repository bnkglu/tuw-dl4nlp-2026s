# Reproducibility and Extension Challenge

**Course:** Deep Learning for Natural Language Processing (192.039-2026S)  
**Group Members:** Hasan Berke Bankoglu, Zarek Asif
**Student Number:** 12432802, <student_number>

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
- **[Reproduction Plan & Details](docs/reproduction_details.md):** A concise breakdown of our step-by-step reproduction plan, including the specific tables, figures, datasets we intend to target, and experiment details in the paper.
- **[Reproduction Notes & Fixes](docs/reproduction_notes.md):** Tracks environment bugs encountered during implementation (e.g., NumPy and PyTorch/CUDA versioning) and how we fixed them to ensure the original codebase runs successfully today.

### 2. Extension
*(To be updated during implementation)*
- Objective: Present novel extensions or additional experiments.
- The motivation for these extensions and the outcomes will be discussed here.

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