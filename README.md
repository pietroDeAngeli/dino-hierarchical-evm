# Hierarchical Object Classification
---

A hierarchical concept-based classification framework built on top of frozen **DINOv3** embeddings.
Instead of assigning each image to a single leaf label, the model traverses a fixed semantic hierarchy
top-down and returns the most specific *reliable* prediction, stopping at an internal ancestor node
when the visual evidence is insufficient. Each node of the hierarchy is modeled as an uncertainty-aware
concept region using **Extreme Value Machines (EVM)**, which provide open-set rejection without any
fine-tuning of the vision backbone.

See [`report/`](report/) for the full write-up (method, ablations, results on COCO and iNat21-Mini).

# Setup

Run [`installation.sh`](installation.sh) (or follow it step by step):

```bash
# 1-2. create + activate the conda environment (environment.yml)
conda env create -f environment.yml
conda activate hol

# 3. install the remaining pip dependencies (requirements.txt)
pip install -r requirements.txt --no-build-isolation
pip install click huggingface_hub --upgrade python-dotenv lz4

# HF login, then download the DINOv3 weights used by scripts/embed_dataset.py
HF_TOKEN="your_hf_token_here" hf auth login
hf download facebook/dinov3-vit7b16-pretrain-lvd1689m --local-dir ./models/dinov3-vit7b16-pretrain-lvd1689m
```

All commands below assume they are run from the repo root with `PYTHONPATH=.`.

# 1. Build a dataset

You can either use COCO (custom) as a dataset or iNaturalist. Note that the validation set of COCO (custom) is ideal for quick tests. 

## COCO

`dataset_preprocessing.sh` runs the whole pipeline end-to-end: downloads COCO annotations/images,
maps COCO categories onto the WordNet hierarchy, crops objects out of the annotated bounding boxes,
and computes DINOv3 embeddings for every crop.

```bash
./dataset_preprocessing.sh
```

Edit `selected_partition` at the top of the script to switch between `val` and `train`.

## iNaturalist 2021

```bash
./iNat_preprocessing.sh
```

## Custom dataset

If you already have a `dataset/<class>/<image>` folder tree, generate a plain descriptor with:

```bash
PYTHONPATH=. python scripts/fs2desc.py dataset/ descriptor.json
```

# 2. Run the static hierarchical classifier

`scripts/thr_sweep.py` builds the static hierarchy memory once (fitting one EVM per node from the
training split) and then evaluates it at multiple acceptance-threshold ($\lambda$) values, producing a
JSON file with one risk/coverage point per threshold.

```bash
PYTHONPATH=. python scripts/thr_sweep.py \
    --descriptor descriptor.json \
    --obj-mem-args input_static/obj_mem_args_faiss.json \
    --test-size 0.1 \
    --seed 0 \
    --thr-a-linspace 0.0 1.0 41 \
    --test-output outputs/flat_test_samples.json \
    --output outputs/flat_init_summary.json \
    --evm-batch-size 999999 \
```

- `--obj-mem-args` points to a JSON config controlling the EVM fitting and negative-selection policy;
  two ready-to-use configs are provided:
  - `input_static/obj_mem_args_faiss.json` — FAISS nearest-neighbor sibling negatives
  - `input_static/obj_mem_args_random.json` — random sibling negatives
- `--flat-hierarchy` collapses the tree to a single root + one leaf per class (the "flat hierarchy" ablation).
- `--jl-dim` applies a Johnson–Lindenstrauss random projection to that many dimensions before fitting/inference.

- `evm-batch-size` is higher than the number of the samples so that the EVM will be updated only at the end of the training after seeing all the samples. 

Use `recsiam/no_hierarchy.py` for the non-hierarchical EVM baseline, and `recsiam/knn.py` /
`recsiam/logistic_head.py` for the flat KNN / logistic-regression baselines described in the report.

