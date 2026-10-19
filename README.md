# JEPA block HSIC on PushT

Research code for **prediction + SIGReg + block-versus-rest HSIC** in LeWorldModel. The motivation is Hugues Van Assel's [JEPA feature-selection analysis and stable-token stress test](https://huguesva.github.io/blog/2026/world-models/).

**This is an exploratory, partly negative result.** With one world-model training seed, λ200 slightly improves the stable-color condition but substantially degrades grey and unpredictable-color planning. An extra 5000 updates did not recover planning. The four-head/fusion variants in the note are proposals and are not implemented here.

- [中文：完整思路、方法与实验反馈](docs/idea-and-method.zh-CN.md)
- [Sources and attribution](THIRD_PARTY.md)
- [Model card](docs/model-card.md)
- [Result summary](results/summary.json)

## Installation

Use Linux, Python 3.10, and a CUDA-capable PyTorch installation for full training and planning. CPU works for unit tests, model loading and the small interface check; full planning is substantially slower.

```bash
git clone https://github.com/yousamar/jepa-block-hsic-pusht.git
cd jepa-block-hsic-pusht
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The environment framework is pinned to a Git revision in requirements.txt. `requirements-tested.txt` lists the core versions used for checkpoint export and verification. The dependency installation may require system build tools and EGL/OpenGL support.

## Data

```bash
export STABLEWM_HOME=/path/to/lewm-data
bash scripts/download_data.sh
```

This downloads the official [PushT expert dataset](https://huggingface.co/datasets/quentinll/lewm-pusht). Install `zstd` for decompression. The dataset is not included in this repository or the model Release.

## Train a baseline or HSIC model

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py train \
  --condition stable --lambda-hsic 200 --seed 4711 \
  --data "$STABLEWM_HOME" --output outputs/stable_lambda200
```

Set `--lambda-hsic 0` for the original prediction+SIGReg objective. Conditions are `stable`, `grey`, and `unpredictable`. Default training is 2 epochs, batch128, BF16, 192-dimensional embeddings split into four48-dimensional blocks, SIGReg weight0.09, AdamW lr5e-5, weight decay0.001, global gradient clipping1. Sixteen loader workers and prefetch3 match the completed experiments; use `--workers` for your machine's memory capacity.

HSIC is computed separately at each time index and averaged over block-versus-rest pairs. The sample-median RBF bandwidth is differentiable. Weight0 skips HSIC completely. No auxiliary prefix objectives or label supervision are used.

The final checkpoint is saved as `*_epoch_2_object.ckpt`; the optimizer checkpoint `*_weights.ckpt` is for local training recovery. A short wiring check can use `--max-steps 20`; it is not an efficacy experiment and does not follow the full-run learning-rate schedule exactly.

## Download the released models

**Model attachments are pending transfer.** The three FP32 bundles have been exported and verified on the experiment server; they are not yet downloadable from GitHub. The planned private Release `models-lambda200-step32866-v1` will contain three inference bundles, about72 MB each, and checksums. Once available, you need access to this private repository and an authenticated [GitHub CLI](https://cli.github.com/).

```bash
python scripts/download_models.py
python infer.py --checkpoint models/stable_lambda200_step32866.pt \
  --self-test --output outputs/interface_check.npz
```

The interface check uses synthetic normalized inputs and tests shapes only. For your own `.npz`, supply `pixels` shaped `[B,T,3,252,252]` after the training ImageNet preprocessing, and `action` shaped `[B,T,10]` after the dataset action normalization/grouping. `infer.py --input ...` writes embeddings and predicted embeddings. Use the evaluation command below for raw environment images and proper dataset normalization.

## Run closed-loop PushT planning

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run.py eval \
  --condition stable --checkpoint models/stable_lambda200_step32866.pt \
  --seed 42 --episodes 50 --data "$STABLEWM_HOME" --output outputs/eval_stable_42
```

Repeat with seed4242 and the matching condition/model. The wrapper fixes the one-token nuisance geometry, matched observation/goal nuisance, model-frame timing and nuisance stream1. CEM retains 300 samples,30 iterations,30 elites, horizon5 and action block5. Evaluation keeps per-episode successes, starting states, goals and planning configuration.

Original trusted `*_object.ckpt` files remain supported by the evaluation wrapper; portable Release `.pt` bundles load only state dictionaries and architecture configuration. No optimizer is required for inference. The model has **18,034,478 parameters**, including6,294,144 representation parameters and11,740,334 action/dynamics parameters. HSIC adds no parameters.

## What the models achieved

| Condition | Before continuation | Released model | Nonlinear physical R², before → after |
|---|---:|---:|---:|
| Stable color | 9/100 | 8/100 | 0.495 → 0.488 |
| Grey | 20/100 | 19/100 | 0.842 → 0.850 |
| Unpredictable color | 10/100 | 11/100 | 0.835 → 0.847 |

The before/after campaign freshly evaluated all600 episodes. These are one world-model training trajectory, not multiple training seeds. Models were extended from27,866 to32,866 updates with retained AdamW moments **and a restarted learning-rate schedule** (50-step warmup, cosine to zero, peak5e-5). Original RNG state was unavailable, so this is not exact uninterrupted training. See the note for the separate λ0/200/600 comparisons and probe interpretation.

## Provenance and license

The model, SIGReg and original intervention code are derived from the upstream repositories listed in SOURCE_PROVENANCE.json and THIRD_PARTY.md. Their MIT copyright notice is retained in LICENSE. The upstream blog's ten-seed results must not be attributed to this project. No datasets, server credentials, machine-specific deployment scripts or complete experiment archives are included.
