---
license: apache-2.0
pipeline_tag: tabular-regression
tags:
  - tabular-regression
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-regressor
---

# Mitra regressor — model card (for the DIMER pipeline)

**This is not a model this project trained.** Mitra was pretrained by the AutoGluon team at
AWS and is used here unmodified under Apache-2.0. This card records what the pipeline uses,
where it comes from, and how it behaves; see the [mitra-regressor-pipeline](https://github.com/kurtvalcorza/mitra-regressor-pipeline)
project for the full pipeline docs, or this repo's [README](README.md) for the fine-tuner.

## Summary

| | |
|---|---|
| Base model | [`autogluon/mitra-regressor`](https://huggingface.co/autogluon/mitra-regressor) |
| Pinned revision | `5f277aa8f69042d39d6ac3612aed18bb9279bd95` |
| Architecture | 12-layer Transformer, 512 embedding, 4 heads, row + column (2D) attention |
| Parameters | ~72M (upstream card) / 75.7M (HF metadata) |
| Pretraining | 45M synthetic datasets on 8×A100 (~60 h); no real data seen |
| Task | tabular regression (numeric target) |
| Licence | Apache-2.0 — redistribution and hosted serving permitted, including commercial |

`model.safetensors` is 302,683,140 bytes, SHA-256
`d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642`; `config.json` is 81 bytes,
SHA-256 `2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1`.

## Providing the weights to DIMER

The weights are **not committed to this repository** (289 MB; and DIMER's build may not fetch
Git LFS). Provide them to the pipeline in one of three ways:

1. **HuggingFace Model ID (default).** Enter `autogluon/mitra-regressor` as the Base Model;
   AutoGluon downloads it at runtime. Requires egress to `huggingface.co`.
2. **Upload weights (no egress).** Use the wizard's *upload your own model weights* box with
   `model.safetensors`; DIMER stores it in S3 and the fine-tuner reads it at runtime.
3. **Bake into the image.** Uncomment Option A in the fine-tuner `Dockerfile` to download the
   pinned revision at build time (needs egress at build), or `COPY` a local copy in.

Fetch the pinned bytes with the Hugging Face CLI:

```bash
hf download autogluon/mitra-regressor model.safetensors config.json \
  --revision 5f277aa8f69042d39d6ac3612aed18bb9279bd95 --local-dir .
```

Every fine-tuning run records the revision it actually used in `result.json`
(`provenance.baseModelRevision`) and compares it to the pinned value.

## How the pipeline uses it

- **Fine-tune (GPU)** — adapts the pretrained weights to the uploaded table. Requires a GPU.
- **Zero-shot (CPU)** — in-context inference with no weight update; used automatically when no
  GPU is available (Mitra's fine-tuning backward pass is unsupported on many CPUs).

The fine-tuner selects the mode from GPU availability at runtime and records it in
`result.json` (`metrics.mode`, `metrics.device`). See the [README](README.md).

## Applicability and limits

Mitra is strongest on **small tabular data** (below ~5,000 samples and ~100 features). Hard
limits: **10,000 training rows, 500 features, 10 classes** (classification). It needs about
**8.7 GB of memory** (measured on 6,400 rows); request a profile that clears that with
headroom (see the [pipeline README](https://github.com/kurtvalcorza/mitra-regressor-pipeline#resource-profile)).

## Measured behaviour in this pipeline

Performance depends on whether the target carries signal — Mitra is an in-context learner, not
a universal improvement:

| Dataset | Zeros | Mode | Mitra MAE | Best naive baseline |
|---|---|---|---|---|
| FreshRetailNet-derived (dense) | 3.6% | fine-tune (GPU) | **0.35** | 0.38 (roll-7 mean) |
| FreshRetailNet-derived (dense) | 3.6% | zero-shot (CPU) | 0.47 | 0.38 |
| Intermittent demand panel | 84% | fine-tune (GPU) | 5.56 | **5.29 (predict-zero)** |

On a dense target Mitra beats the naive baselines; on a highly intermittent one it does not
beat predicting zero. These are small smoke-test runs — a few thousand rows, one seed, a random
split — not a benchmark; read them as direction, not scores. Treat Mitra's published benchmark
results as evidence of strong performance where signal exists, not a guarantee on any table.

## Licence

Apache-2.0. Redistribution and hosted serving are both permitted, including commercial use, on
the conditions of retaining the `LICENSE` and stating modifications (there are none). The
Apache-2.0 licence text ships with the weights upstream.

## Citation

Cite the original work, not this repository:

> Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu,
> L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025).
> *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025.
> arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204
