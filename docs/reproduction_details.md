# Reproducibility Plan

First, reproduce the proposed method's runs which is CLIP-LoRA.
Use the standard seeds 1, 2, and 3 for all experiments. Then we can try to extend the paper such as changing regularization.

# CLIP-LoRA Reproducibility Plan

**Paper:** Low-Rank Few-Shot Adaptation of Vision-Language Models (CLIP-LoRA)
**Base code:** Official repo [MaxZanella/CLIP-LoRA](https://github.com/MaxZanella/CLIP-LoRA)

Reproduce CLIP-LoRA using the paper's default configuration. Use seeds 1, 2, 3
for all experiments. After the tables are stable, run the ablations and the
regularization extension.

## Default CLIP-LoRA Setup

Unless an ablation changes it, every run uses:

| Parameter       | Value                         |
|-----------------|-------------------------------|
| rank `r`        | 2                             |
| target matrices | `{Wq, Wk, Wv}`                |
| encoder         | vision + text (both)          |
| placement       | all encoder layers            |
| dropout `p`     | 0.25                          |
| learning rate   | 2e-4                          |
| scheduler       | cosine                        |
| batch size      | 32                            |
| iterations      | 500 × shots                   |
| prompt          | `"a photo of a [class name]"` |

## Phase 0: Sanity Check

Each person verifies environment, dataset paths, and logging:
- dataset = EuroSAT, shots = 4, seed = 1
- Teammate 1: ViT-B/16 — Teammate 2: ViT-B/32

## Phase 1 — Main Table Reproduction (1.A and 1.B run concurrently)

Both sub-phases are the same experiment with different backbones, run in parallel
(one backbone per person). ViT-B/16 (Table 3) is the priority; if 1.B breaks,
Teammate 2 helps finish 1.A first.

Shared datasets (all 11): ImageNet, SUN397, Aircraft, EuroSAT, StanfordCars,
Food101, OxfordPets, Flowers102, Caltech101, DTD, UCF101.

### Phase 1.A — Table 3 (ViT-B/16) — *Teammate 1*
- `shots=(1 2 4 8 16)`, `seeds=(1 2 3)`, `backbone=ViT-B/16`
- Run count: 11 × 5 × 3 = **165 runs**
- Script: `run_table_3.sh`

### Phase 1.B — Table 4 (ViT-B/32) — *Teammate 2*
- `shots=(1 2 4 8 16)`, `seeds=(1 2 3)`, `backbone=ViT-B/32`
- Run count: **165 runs**
- Script: `run_table_4.sh`

## Phase 2 — Figure 3 (full grid)

Figure 3's two bar-chart axes are **rank** (x) and **attention matrix set**
(bars), so these are crossed. **Encoder choice** (vision / text / both) is the
separate panels, also crossed. **Placement** is fixed to `All` for the main
grid; the placement comparison (bottom/up/all) is a separate small set of bars.

- Datasets: ImageNet, StanfordCars, EuroSAT
- Fixed: `backbone=ViT-B/16`, `shots=4`

**Main grid** (placement = all):
- ranks: `1, 2, 4, 8, 16, 32`  (6)
- matrix sets: `{Wk}, {Wq}, {Wv}, {Wo}, {Wq,Wv}, {Wq,Wv,Wk}, {Wq,Wv,Wk,Wo}`  (7)
- encoders: vision only, text only, both  (3)
- → 6 × 7 × 3 = **126 combinations per dataset**

**Placement set** (at default rank 2, `{Wq,Wk,Wv}`, both):
- placement: bottom, up, all  (3 per dataset)

Per dataset: 126 + 3 = **129**.

**Run count by seed budget:**
| Seeds | Runs (3 datasets) |
|-------|-------------------|
| 1 seed | 129 × 3 = **387** |
| 3 seeds | 129 × 3 × 3 = **1161** |

**Strategy:** generate one job list (one row per run), split it round-robin
between both people (`awk 'NR%2'`) so rank/dataset/matrix load is balanced.
Start with **1 seed**. If the divided load finishes with time to spare, re-run
with seeds 2 and 3 for error bars. Do not split by rank or by placement — the
job-list split balances automatically.

## Phase 3: Extension — Regularization of the Loss (the main "extension" deliverable)

**Motivation.** CLIP-LoRA trains the low-rank updates with a plain
cross-entropy (CE) loss. CE is straightforward but provides no constraint
keeping the adapted model close to the original zero-shot CLIP. With only a few
shots, this can let the model overfit and drift away from CLIP's strong
pretrained priors — a plausible reason it underperforms on harder / fine-grained
datasets such as Food101. The extension adds a regularization term that anchors
the adapted model to frozen zero-shot CLIP and measures whether this recovers
performance on those datasets.

**Proposed loss.**
```
L = CE(lora_logits, labels) + λ · KL( zero_shot_CLIP_probs ‖ lora_probs )
```
- `zero_shot_CLIP_probs`: softmax over class similarities from the **frozen**
  CLIP (no LoRA), acting as a fixed teacher.
- `lora_probs`: softmax over class similarities from the **adapted** (LoRA)
  model.
- The KL term keeps the adapted output distribution from drifting too far from
  zero-shot CLIP — a knowledge-distillation / anchoring regularizer.
- `λ` controls the strength. `λ = 0` recovers the original CE-only CLIP-LoRA
  baseline, giving a clean controlled comparison.

> Note: KL is between the two models' **output probability distributions over
> classes**, not between their weight tensors.

**Experiment.**
- Vary: `λ = 0 (baseline), 0.5, 1.0, 2.0` (tune range after a first look)
- Datasets: Food101, StanfordCars, EuroSAT (+ ImageNet if time) — include the
  datasets where CLIP-LoRA is weakest so the effect is visible
- Parameters: `shots=4`, `seeds=(1 2 3)`, `backbone=ViT-B/16`, all else default
- Report: accuracy vs. λ, compared against the λ=0 CE-only baseline; highlight
  any recovery on Food101.

> A simpler variant (dropout sweep: `0.0, 0.1, 0.25, 0.4`) can be run as a
> cheap secondary check if the KL term needs more implementation time.

## Logging (minimum requirement)

Every run must record at least: `dataset, backbone, shots, seed, rank, targets,
encoder, placement, dropout, accuracy, status`. All results merge into one file:
`results/clip_lora_results.csv`.

---

*Execution details — script arguments, the `--dataset` / run-part options,
nohup invocations, job-list generation and splitting, and monitoring commands —
are documented separately in `experiment_running.md`, written after the shell
scripts are built.*

Scripts (in `reproducing-clip-lora/scripts/`): `download_datasets.sh`,
`test_dataset_urls.sh`, `run_sanity_check.sh`, `run_table_3.sh`, `run_table_4.sh`,
`run_fig3.sh`, `aggregate_results.py`. See `scripts/README.md` for usage.


# Details of Experiments in the Paper
## Training Time Details
### Table 1
Table 1. Training time on 16-shots ImageNet task. Experiments were conducted on a single A100 80Gb with the original code provided by the authors. For PLOT++ the time reported includes the 2 training stages.

| Method    | Training Time |
| :-------- | :------------ |
|CoOp       | 2h            |
|PLOT++     | 15h30         |
|ProGrad    | 3h20          |
|CLIP-LoRA  | 50 min        |


## Table2 Runs
Models:
  - CLIP (only 0 shot)
  - CoOp (4 tokens)
  - CoOp (16 tokens)
  - CoCoOp
  - Tip-Adapter-F
  - CLIP-Adapter
  - PLOT++
  - KgCoOp
  - TaskRes
  - MaPLe
  - ProGrad
  - CLIP-LoRA

Shots:
  - 1, 4, 16
  - 2 and 8 (given in appendix separately)

Datasets:
  - ImageNet
  - SUN397
  - Aircraft
  - EuroSAT
  - StanfordCars
  - Food101
  - OxfordPets
  - Flower102
  - Caltech101
  - DTD
  - UCF101

Backbones:
  - ViT-B/16

Avg is reported over 3 random seeds.

## Figure2 Runs
Figure 2. Detailed few-shot learning results on the 10 fine-grained datasets and ImageNet with the ViT-B/16 visual backbone. Average performance for the ViT-B/16, ViT-B/32 and ViT-L/14 on the same 11 datasets is reported in the last three plots, respectively.

- Used Shots are 1,2,4,8,16
- Models are PLOT++, TaskRes, Tip-Adapter-F, CoOp, ProGrad, CLIP-LoRA
- Average(ViT-B/16), avg(ViT-B/32), avg(ViT-L/14) across datasets for each method

## Figure 3 Runs

Figure 3. Top-1 accuracy with 4-shots for different matrices of the attention bloc and increasing rank, when the low-rank matrices are positioned at every level of the encoders (All). The fourth bar plot study the impact of positioning the low-rank matrices only on the half last levels (Up), the first half levels (Bottom), or at every level (All). Reported top-1 accuracy is averaged over 3 random seeds.

**Datasets:** ImageNet, StanfordCars, EuroSAT
**What is done:**
- To which encoder: 
    - Vision Encoder
    - Text Encoder
    - Both
- Which Attention Matrices: 
    1.  $W_k$
    2.  $W_q$
    3.  $W_v$
    4.  $W_o$
    5.  $W_qW_v$ (Query and Value)
    6.  $W_qW_vW_k$ (Query, Value, and Key)
    7.  $W_qW_vW_kW_o$ (All four).
    
- Rank: 1, 2, 4, 8, 16, 32 (64 can be run as extension)
- Placement: 
    - Bottom
    - Up
    - All

## Table 3 Runs
Which is same as table2 except for the shots.

Models:
  - CLIP (only 0 shot)
  - CoOp (4 tokens)
  - CoOp (16 tokens)
  - CoCoOp
  - Tip-Adapter-F
  - CLIP-Adapter
  - PLOT++
  - KgCoOp
  - TaskRes
  - MaPLe
  - ProGrad
  - CLIP-LoRA

Shots:
  - 1, 2, 4, 8, 16

Datasets:
  - ImageNet
  - SUN397
  - Aircraft
  - EuroSAT
  - StanfordCars
  - Food101
  - OxfordPets
  - Flower102
  - Caltech101
  - DTD
  - UCF101

Backbones:
  - ViT-B/16

Avg is reported over 3 random seeds.

## Table 4 Runs
Which is same as Table3 except for the backbone. This uses ViT-B/32 backbone.

Models:
  - CLIP (only 0 shot)
  - CoOp (4 tokens)
  - CoOp (16 tokens)
  - CoCoOp
  - Tip-Adapter-F
  - CLIP-Adapter
  - PLOT++
  - KgCoOp
  - TaskRes
  - MaPLe
  - ProGrad
  - CLIP-LoRA

Shots:
  - 1, 2, 4, 8, 16

Datasets:
  - ImageNet
  - SUN397
  - Aircraft
  - EuroSAT
  - StanfordCars
  - Food101
  - OxfordPets
  - Flower102
  - Caltech101
  - DTD
  - UCF101

Backbones:
  - ViT-B/32

Avg is reported over 3 random seeds.

## Table 5 Runs
Which is same as tables 3 and 4 except for the backbone. This uses ViT-L/14 backbone.

Models:
  - CLIP (only 0 shot)
  - CoOp (4 tokens)
  - CoOp (16 tokens)
  - CoCoOp
  - Tip-Adapter-F
  - CLIP-Adapter
  - PLOT++
  - KgCoOp
  - TaskRes
  - MaPLe
  - ProGrad
  - CLIP-LoRA

Shots:
  - 1, 2, 4, 8, 16

Datasets:
  - ImageNet
  - SUN397
  - Aircraft
  - EuroSAT
  - StanfordCars
  - Food101
  - OxfordPets
  - Flower102
  - Caltech101
  - DTD
  - UCF101

Backbones:
  - ViT-L/14

Avg is reported over 3 random seeds.


## Datasets and Methods

We evaluate on **11 datasets** to benchmark few-shot visual classification:
*   **Fine-grained classification of scenes**: SUN397, Aircraft, EuroSAT, Stanford-Cars, Food101, OxfordPets, Flower102
*   **General objects**: Caltech101, DTD
*   **Human actions & ImageNet**: UCF101, ImageNet

## Comparative Methods

We compare CLIP-LoRA against existing state-of-the-art methods:

**Prompt-Based Methods:**
*   **CoOp** (with 4 learnable tokens and 16 learnable tokens)
*   **CoCoOp**
*   **PLOT++**
*   **KgCoOp**
*   **MaPLe** (following their training procedure "base-to-new" setting)
*   **ProGrad** (with 16 tokens)

**Adapter-Based Methods:**
*   **Tip-Adapter-F** (validation set: min(n shots, 4))
*   **TaskRes** (not enhanced base performance due to its unavailability for all datasets/shots/backbones studied in this paper)

*Note: Their specific hyperparameters are kept while CLIP-LoRA uses the same hyperparameters for every task.*

## Hyperparameters and Implementation

*   **LoRA Application**: Applied on the Query (Q), Key (K), and Value (V) matrices.
*   **LoRA Position**: Modules are positioned at every level of both the vision and language encoders. Adapting both modalities can be necessary for certain tasks (impact studied in Section 6).
*   **Hardware Setup**: All training could be performed on a single 24GB GPU.
*   **Consistency**: These hyperparameters are kept fixed across all experiments.

*(Reference: Section 4 CLIP-LoRA - LoRA for VLMs part)*

|Hyperparameter                | Value                         |
| :--------------------------- | :---------------------------- |
| Rank                         | r=2                           |
| Target Modules               | $W_q, W_k, W_v$               |
| Dropout                      | p = 0.25                      |
| Number of iterations         | 500 * N/K                     |
| Learning rate                | 2 * 10^-4 (2e-4)              |
| Scheduler                    | cosine                        |
| Batch size                   | 32                            |
| Input prompt                 | "a photo of a [class name]"   |
