# Paper Summary

There are 11 datasets.

## Their stated contribution
Contributions. In this work, we investigate the deployment of Low-Rank Adaptation (LoRA) in the context of few-shot VLMs, an emergent, already quite abundant literature dominated by prompt-learning and adapter-based strategies. We thoroughly examine different design choices for deploying LoRA in this context, namely, the choices of the encoders (vision, language or both), of the specific weight matrices to adapt, and of the rank of the matrices. We conduct comprehensive empirical ablations and comparisons, over 11 datasets, emphasizing the best design choices for our baseline and juxtaposing it to the existing state-of-the-art prompt- and adapter-based methods.

## Other notes
CLIP-Adapter [14] learns visual adapters to combine adapted and original features.

## 3. Overview of Few-shot fine-tuning for VLMs

When dealing with a classification task based on a vision-language model, and given a set of K candidate classes, one creates textual descriptions, the so-called prompts [35], each corresponding to a class, e.g., $c_k$ is the tokenized version of "a photo of a [kth class name]", $k= 1,...,K$. Let $t_k = \theta_t(c_k)$ denotes the corresponding normalized (unit-hypersphere) textual embedding representation, with $\theta_t$ representing the parameters of the language encoder. Similarly, each image $x_i, i = 1,...,N$, is projected onto a normalized embedding space of the same dimension, using the visual encoder $\theta_v$: $f_i = \theta_v(x_i)$.

- Prompt Tuning (Context Optimization aka CoOp)
- Adapters 


### Table 1
Table 1. Training time on 16-shots ImageNet task. Experiments were conducted on a single A100 80Gb with the original code provided by the authors. For PLOT++ the time reported includes the 2 training stages.

| Method    | Training Time |
| :-------- | :------------ |
|CoOp       | 2h            |
|PLOT++     | 15h30         |
|ProGrad    | 3h20          |
|CLIP-LoRA  | 50 min        |

## Used Regularization
Cross-Entropy Loss
$$L_{\text{CE}} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_{i,k} ln p_{i,k}$$
where $p_{i,k} = \frac{exp(l_{i,k})}{\sum_{j}^{K} exp(l_{i,j}/\tau)}$, $l_{i,k}$ is the prediction logit for class $k$ on image $i$, $y_{i,k}$ is the correct class label, $\tau$ is the temperature parameter, and $N$ is the number of samples.

This is done either (i) by fine-tuning the input prompts, ck , k = 1,...,K, as in prompt-tuning methods following on from the pioneering work of CoOp [5, 26, 53, 62–64]; or (ii) by updating a set of additional parameters, as in adapters.
1Note that each CLIP version comes with a temperature scaling τ , which is optimized along with the learnable parameters during pre-training.


# CLIP-LoRA
ow-Rank Adaptation (LoRA) [21] models the incremen-
tal update of the pre-trained weights as the product of two
small matrices, A and B, based on the idea of ‘’intrinsic
rank” of a downstream task. For an input x, a hidden state
h, and a weight matrix $W \in \mathbb{R}^{d_1 \times d_2}$, the modified forward
pass, following the application of a LoRA module, is: 

$$h = Wx + \gamma \Delta Wx = Wx + \gamma BAx$$

where $A \in \mathbb{R}^{r \times d_2}$, $B \in \mathbb{R}^{d_1 \times r}$, $\Delta W \in \mathbb{R}^{d_1 \times d_2}$ of rank r, with r typically ≪{d1,d2}, and γ a scaling factor. Values in A are randomly initialized via K aiming initialization while B is filled with zeros. This implies that there is no incremental update before training, and therefore, the output remains unchanged

LoRA for VLMs. A straightforward way to apply LoRA in vision-language is to apply it to all the matrices of the vision and text encoders. However, due to the relatively
small supervision inherent to the few-shot setting, we only apply low-rank matrices on the query, key and value matrices with r = 2. We regularize the input of the LoRA
module by a dropout layer with p = 0.25 [21]. The number of iterations is set equal to 500 times N/K (the number of labeled samples per class). We used a learning rate of
2 ∗10−4, with a cosine scheduler and a batch size of 32, so that all training could be performed on a single GPU of 24Gb. These hyper-parameters are kept fixed across all the experiments. The input prompt is simply set to a photo of a [kth class name], k = 1,...,K, for every dataset, to emphasize the applicability of CLIP-LoRA without re-
sorting to complex initial manual prompting. Note that the LoRA modules are positioned at every levels of both encoders. The impact of the location of the LoRA modules is
studied in Section 6, putting in evidence that adapting both modalities can be necessary for certain tasks.

## Hyperparameters and Implementation

*   **LoRA Application**: Applied on the Query (Q), Key (K), and Value (V) matrices.
*   **LoRA Position**: Modules are positioned at every level of both the vision and language encoders. Adapting both modalities can be necessary for certain tasks (impact studied in Section 6).
*   **Hardware Setup**: All training could be performed on a single 24GB GPU.
*   **Consistency**: These hyperparameters are kept fixed across all experiments.

*(Reference: Section 4 CLIP-LoRA - LoRA for VLMs part)*

|Hyperparameter             | Value                    |
| :-------------------------- | :----------------------- |
| Rank                         | r=2                      |
| Dropout                      | p = 0.25                 |
| Number of iterations         | 500 times N/K            |
| Learning rate                | 2 ∗10−4                  |
| Scheduler                    | cosine                 |
| Batch size                   | 32                       |
| Input prompt                 | "a photo of a [class name]" |

# 5. Few-Shot Learning

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

## Claim:
**CLIP-LoRA outperforms, on average, adapter- and prompt-based few-shot methods.** 
The strongest adapter-based method in Table 2 is Tip-Adapter-F, which is not competitive with CLIP-LoRA despite relying heavily on arbitrary hyper-parameters for each dataset (namely the starting value of their α,β as well as the search range during validation). We can conclude the same for TaskRes, which also relies on arbitrary choices for a given dataset, i.e., a specific learning rate for ImageNet and a specific scaling factor for the Flowers dataset.

**Regarding prompt-based approaches:** 
Table 2 shows that CoOp and ProGrad are outperformed by a large margin. The strongest competitor is, without a doubt, PLOT++. PLOT++ necessitates a two-stage training (each of 50 epochs for ImageNet) as well as several dataset-specific textual templates for their optimal transport formulation, reducing its portability to other downstream tasks.

## **Overall**
Overall, CLIP-LoRA performs better, especially on ImageNet, UCF101 and Aircraft, while being more practical. 
- However, it **underperforms on two datasets, Food101 and OxfordPet, where few-shot learning offers minimal improvement. This may be attributed to the lack of regularization, considering we use straightforward cross-entropy loss.** 
- We observe a similar trend with CoOp, whereas approaches that incorporate explicit regularization, such as ProGrad, do not exhibit this issue. Note that more detailed results, including for 2 and 8 shots, are available in the Appendix.

- CLIP-LoRA performances are consistent across various vision encoders. As depicted in Figure 2, CLIP-LoRA surpasses, on average, the other few-shot methods with both the ViT-B/32 architecture and the larger ViT-L/14. This fur-
ther supports the versatility of our approach. Detailed results for the three backbones are available in the Appendix

- CLIP-LoRA is computationally and memory efficient. Table 1 compares the training time of the leading prompt-learning methods; CLIP-LoRA achieves better performance
with shorter training. Moreover, the best performing adapter method, namely Tip-Adapter-F, depends on a large cache model that stores embeddings for all instances across every class. In contrast, LoRA merges its adapted matrices at the inference stage, thereby eliminating the need for extra memory beyond what is required by the original model.

## Table2 Runs
Models:
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
- Backbone is ViT-B/16

# 6. How to apply LoRA for VLMs?

In this section, we delve into the utilization of LoRA mod-
ules, identifying three principal design considerations: 
    1. the choice between tuning the vision encoder, the text en-
coder, or both, including the specific layers to adjust; 
    2. the selection of attention matrices for tuning; 
    3. the determination of the appropriate rank for these matrices.

We explore these aspects across three datasets: 
- ImageNet (selected for its broad diversity)
- Stanford Cars
- EuroSAT

Results are depicted in Figure 3 for
- seven different groups of adapted attention matrices
- increasing rank value.

Adapting both encoders leads to the best results on average. 
With the exception of EuroSAT, where adapting solely the vision encoder shows marginally better stability, tuning both encoders concurrently is the most effective
strategy, leading to significant enhancements. This aligns with recent approaches that incorporate additional vision tokens [5, 26] to augment performance beyond what is achievable with text-only prompt tuning, as seen in CoOp [63].

Tuning more attention matrices can lead to better re-
sults but... Among the four attention matrices studied,
adapting value or output matrices (Wv and Wo) appears to
be the best strategy, showing quite consistent differences in
performance. Moreover, as discussed in the original LoRA
paper and subsequent works [21, 60], adapting a larger
number of weight matrices can lead to better results. How-
ever, it can also decrease performance, as demonstrated on
ImageNet and StanfordCars with high rank. This is in line
with recent methods that aim to dynamically adjust the rank
of the matrices [48, 60].

Choosing the location of LoRA modules requires care-
ful consideration. The impact of LoRA module place-
ment—whether on the lower half (bottom), or the upper half
(up)—is illustrated in the bar plots of Figure 3 with varying
performance and no clear winner. We found it more effec-
tive to add LoRA modules across all layers. In comparison,
in the context of LLMs, AdaLoRA [60] suggests that al-
locating a larger rank to the middle and last layers rather
than the first ones yields better results. Similar strategies
applied for VLMs could reveal promising avenues for future research.

## Figure 3 Runs

Figure 3. Top-1 accuracy with 4-shots for different matrices of the attention bloc and increasing rank, when the low-rank matrices are positioned at every level of the encoders (All). The fourth bar plot study the impact of positioning the low-rank matrices only on the half last levels (Up), the first half levels (Bottom), or at every level (All). Reported top-1 accuracy is averaged over 3 random seeds.

**Datasets:** ImageNet, StanfordCars, EuroSAT
**What is done:**
- To which encoder: 
    - Vision Encoder
    - Text Encoder
    - Both
- Which Attention Matrices: 
    - Query (Wq)
    - Key (Wk)
    - Value (Wv)
    - Output (Wo)
    - All
- Rank: 1, 2, 4, 8, 16, 32, 64
- Placement: 
    - Bottom
    - Top
    - All

# Conclusion

- We established a strong baseline by consistently outperforming prompt- and adapter-based methods in few-shot adaptation of Vision-Language Models (VLMs) using fixed hyper-parameters. 
- We hope our work inspires future efforts to design methods that either uphold this simplicity and efficiency with fixed hyper-parameters or offer clear guidelines for adaptable hyper-parameter settings. 
- Additionally, we demonstrated that selecting the matrices to adapt and determining the corresponding rank to maximize performance using LoRA modules is not trivial. We believe these aspects of our work suggest a promising area for future research.