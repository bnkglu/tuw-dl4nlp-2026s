CLIP-LoRA Reproduction Notes
=============================

Environment
-----------
Repo path: /home/jovyan/scratch_projects/CLIP-LoRA
Repo commit: ce3701ba370576f082e1295fbd90760961ead724
Python env: /home/jovyan/scratch_projects/envs/clip-lora
Python version: 3.10.20
GPU: Tesla V100-SXM2-32GB

Dataset status
--------------
StanfordCars dataset root: /home/jovyan/scratch_projects/CLIP-LoRA/datasets/StanfordCars
StanfordCars train images: 8144
StanfordCars test images: 8041
Split file present: split_zhou_StanfordCars.json
ImageNet: not started yet

Run 1
-----
Purpose: first sanity run with official repo on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 1
Seed: 1
Learning rate: 0.0002
Iterations: 500
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 1 --backbone ViT-B/16 --lr 0.0002 --n_iters 500 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot1 --filename lora_seed1_shot1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed1_shot1.txt
Final test accuracy: 69.62
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot1/vitb16/stanford_cars/1shots/seed1/lora_seed1_shot1.pt
Checkpoint size: 763K
Notes:
First attempt failed because --alpha was passed as 1.0 instead of integer 1.
A second attempt failed with "optimizer got an empty parameter list" when manual LoRA placement arguments were passed.
Successful run used the official repo defaults for LoRA placement and only set the core paper hyperparameters explicitly.

Run 2
-----
Purpose: second sanity run (different seed) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 1
Seed: 2
Learning rate: 0.0002
Iterations: 500
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 1 --backbone ViT-B/16 --lr 0.0002 --n_iters 500 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot1 --filename lora_seed2_shot1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed2_shot1.txt
Final test accuracy: 70.51
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot1/vitb16/stanford_cars/1shots/seed2/lora_seed2_shot1.pt
Checkpoint size: 763K

Run 4
-----
Purpose: first 2-shot run (seed 1) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 2
Seed: 1
Learning rate: 0.0002
Iterations: 1000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.51
Final test accuracy (eval_only): 70.95
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 2 --backbone ViT-B/16 --lr 0.0002 --n_iters 1000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot2 --filename lora_seed1_shot2 > logs_stanford_cars_seed1_shot2.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed1_shot2.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot2/vitb16/stanford_cars/2shots/seed1/lora_seed1_shot2.pt
Checkpoint size: 763K
Notes:
Same hyperparameters as the 1-shot runs, but with 2 shots and 1000 iterations. Zero-shot baseline is around 65.5%, CLIP-LoRA improves it to about 71%.

Run 5
-----
Purpose: second 2-shot run (seed 2) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 2
Seed: 2
Learning rate: 0.0002
Iterations: 1000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.55
Final test accuracy (eval_only): 71.86
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 2 --backbone ViT-B/16 --lr 0.0002 --n_iters 1000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot2 --filename lora_seed2_shot2 > logs_stanford_cars_seed2_shot2.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed2_shot2.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot2/vitb16/stanford_cars/2shots/seed2/lora_seed2_shot2.pt
Checkpoint size: 763K
Notes:
Same hyperparameters as the 2-shot seed 1 run; performance slightly higher, around 71.9% vs. 71.0%.

Run 4
-----
Purpose: first 2-shot run (seed 1) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 2
Seed: 1
Learning rate: 0.0002
Iterations: 1000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.51
Final test accuracy (eval_only): 70.95
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 2 --backbone ViT-B/16 --lr 0.0002 --n_iters 1000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot2 --filename lora_seed1_shot2 > logs_stanford_cars_seed1_shot2.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed1_shot2.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot2/vitb16/stanford_cars/2shots/seed1/lora_seed1_shot2.pt
Checkpoint size: 763K
Notes:
Same hyperparameters as the 1-shot runs, but with 2 shots and 1000 iterations. Zero-shot baseline is around 65.5%, CLIP-LoRA improves it to about 71%.

Run 5
-----
Purpose: second 2-shot run (seed 2) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 2
Seed: 2
Learning rate: 0.0002
Iterations: 1000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.55
Final test accuracy (eval_only): 71.86
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 2 --backbone ViT-B/16 --lr 0.0002 --n_iters 1000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot2 --filename lora_seed2_shot2 > logs_stanford_cars_seed2_shot2.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed2_shot2.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot2/vitb16/stanford_cars/2shots/seed2/lora_seed2_shot2.pt
Checkpoint size: 763K
Notes:
Same hyperparameters as the 2-shot seed 1 run; performance slightly higher, around 71.9% vs. 71.0%.

Run 6
-----
Purpose: third 2-shot run (seed 3) on StanfordCars
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 2
Seed: 3
Learning rate: 0.0002
Iterations: 1000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.54
Final test accuracy (eval_only): 70.45
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 3 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 2 --backbone ViT-B/16 --lr 0.0002 --n_iters 1000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot2 --filename lora_seed3_shot2 > logs_stanford_cars_seed3_shot2.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed3_shot2.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot2/vitb16/stanford_cars/2shots/seed3/lora_seed3_shot2.pt
Checkpoint size: 763K
Notes:
Same hyperparameters as the other 2-shot runs; accuracy is around 70.5%, slightly lower than seeds 1–2 but consistent with expected variance.

Summary (StanfordCars, ViT-B/16, 2 shots)
----------------------------------------
Test accuracies over 3 seeds: [70.95, 71.86, 70.45]
Mean test accuracy: 71.09

Run 7
-----
Purpose: first 1-shot run on StanfordCars with ViT-B/32 (Table 4)
Dataset: stanford_cars
Backbone: ViT-B/32
Shots: 1
Seed: 1
Learning rate: 0.0002
Iterations: 500
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 60.15
Final test accuracy (eval_only): 63.11
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 1 --backbone ViT-B/32 --lr 0.0002 --n_iters 500 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed1_shot1 --filename lora_seed1_shot1_vitb32 > logs_stanford_cars_vitb32_seed1_shot1.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_vitb32_seed1_shot1.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed1_shot1/vitb32/stanford_cars/1shots/seed1/lora_seed1_shot1_vitb32.pt
Checkpoint size: 763K
Notes:
First ViT-B/32 run for Table 4; CLIP-LoRA improves zero-shot performance from about 60.2% to about 63.1% on StanfordCars 1-shot.

Run 8
-----
Purpose: second 1-shot run on StanfordCars with ViT-B/32 (Table 4)
Dataset: stanford_cars
Backbone: ViT-B/32
Shots: 1
Seed: 2
Learning rate: 0.0002
Iterations: 500
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 60.15
Final test accuracy (eval_only): 64.33
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 1 --backbone ViT-B/32 --lr 0.0002 --n_iters 500 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed2_shot1 --filename lora_seed2_shot1_vitb32 > logs_stanford_cars_vitb32_seed2_shot1.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_vitb32_seed2_shot1.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed2_shot1/vitb32/stanford_cars/1shots/seed2/lora_seed2_shot1_vitb32.pt
Checkpoint size: 763K
Notes:
Same settings as Run 7 but with seed 2; CLIP-LoRA improves zero-shot performance from about 60.2% to about 64.3%.

Run 9
-----
Purpose: third 1-shot run on StanfordCars with ViT-B/32 (Table 4)
Dataset: stanford_cars
Backbone: ViT-B/32
Shots: 1
Seed: 3
Learning rate: 0.0002
Iterations: 500
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 60.15
Final test accuracy (eval_only): 63.14
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 3 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 1 --backbone ViT-B/32 --lr 0.0002 --n_iters 500 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed3_shot1 --filename lora_seed3_shot1_vitb32 > logs_stanford_cars_vitb32_seed3_shot1.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_vitb32_seed3_shot1.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars_vitb32/seed3_shot1/vitb32/stanford_cars/1shots/seed3/lora_seed3_shot1_vitb32.pt
Checkpoint size: 763K
Notes:
Same settings as Runs 7–8 but with seed 3; CLIP-LoRA improves zero-shot performance from about 60.2% to about 63.1–64.3% across seeds.

Run 10
------
Purpose: first 4-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 4
Seed: 1
Learning rate: 0.0002
Iterations: 2000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.59
Final test accuracy (eval_only): 72.69
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 4 --backbone ViT-B/16 --lr 0.0002 --n_iters 2000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot4 --filename lora_seed1_shot4 > logs_stanford_cars_seed1_shot4.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed1_shot4.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot4/vitb16/stanford_cars/4shots/seed1/lora_seed1_shot4.pt
Checkpoint size: 763K
Notes:
Baseline CLIP-LoRA 4-shot configuration for StanfordCars with ViT-B/16 (Figure 3 curve).

Run 11
------
Purpose: second 4-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 4
Seed: 2
Learning rate: 0.0002
Iterations: 2000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.54
Final test accuracy (eval_only): 72.88
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 4 --backbone ViT-B/16 --lr 0.0002 --n_iters 2000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot4 --filename lora_seed2_shot4 > logs_stanford_cars_seed2_shot4.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed2_shot4.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot4/vitb16/stanford_cars/4shots/seed2/lora_seed2_shot4.pt
Checkpoint size: 763K
Notes:
Same 4-shot CLIP-LoRA setting as Run 10 but with seed 2; accuracy is very similar.

Run 12
------
Purpose: third 4-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 4
Seed: 3
Learning rate: 0.0002
Iterations: 2000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.60
Final test accuracy (eval_only): 73.29
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 3 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 4 --backbone ViT-B/16 --lr 0.0002 --n_iters 2000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot4 --filename lora_seed3_shot4 > logs_stanford_cars_seed3_shot4.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed3_shot4.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot4/vitb16/stanford_cars/4shots/seed3/lora_seed3_shot4.pt
Checkpoint size: 763K
Notes:
Same 4-shot CLIP-LoRA setting as Runs 10–11 but with seed 3; all three seeds show consistent performance.

Run 13
------
Purpose: first 8-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 8
Seed: 1
Learning rate: 0.0002
Iterations: 4000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.55
Final test accuracy (eval_only): 78.25
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 1 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 8 --backbone ViT-B/16 --lr 0.0002 --n_iters 4000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot8 --filename lora_seed1_shot8 > logs_stanford_cars_seed1_shot8.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed1_shot8.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed1_shot8/vitb16/stanford_cars/8shots/seed1/lora_seed1_shot8.pt
Checkpoint size: 763K
Notes:
Baseline CLIP-LoRA 8-shot configuration for StanfordCars with ViT-B/16 (Figure 3 curve).

Run 14
------
Purpose: second 8-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 8
Seed: 2
Learning rate: 0.0002
Iterations: 4000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.49
Final test accuracy (eval_only): 78.39
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 2 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 8 --backbone ViT-B/16 --lr 0.0002 --n_iters 4000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot8 --filename lora_seed2_shot8 > logs_stanford_cars_seed2_shot8.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed2_shot8.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed2_shot8/vitb16/stanford_cars/8shots/seed2/lora_seed2_shot8.pt
Checkpoint size: 763K
Notes:
Same 8-shot CLIP-LoRA setting as Run 13 but with seed 2; performance is consistent.

Run 15
------
Purpose: third 8-shot run on StanfordCars with ViT-B/16 (Figure 3)
Dataset: stanford_cars
Backbone: ViT-B/16
Shots: 8
Seed: 3
Learning rate: 0.0002
Iterations: 4000
Batch size: 32
Rank r: 2
Alpha: 1
Dropout rate: 0.25
Zero-shot baseline (eval_only): 65.60
Final test accuracy (eval_only): 78.76
Command: micromamba run -p /home/jovyan/scratch_projects/envs/clip-lora python main.py --seed 3 --root_path /home/jovyan/scratch_projects/CLIP-LoRA/datasets --dataset stanford_cars --shots 8 --backbone ViT-B/16 --lr 0.0002 --n_iters 4000 --batch_size 32 --r 2 --alpha 1 --dropout_rate 0.25 --save_path /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot8 --filename lora_seed3_shot8 > logs_stanford_cars_seed3_shot8.txt 2>&1
Log file: /home/jovyan/scratch_projects/CLIP-LoRA/logs_stanford_cars_seed3_shot8.txt
Checkpoint path: /home/jovyan/scratch_projects/CLIP-LoRA/results/stanford_cars/seed3_shot8/vitb16/stanford_cars/8shots/seed3/lora_seed3_shot8.pt
Checkpoint size: 763K
Notes:
Third seed for 8-shot CLIP-LoRA on StanfordCars; highest accuracy among the three seeds.
