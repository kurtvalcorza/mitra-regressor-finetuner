# mitra-regressor-finetuner

DIMER fine-tuner for the Mitra regressor pipeline. It fine-tunes AutoGluon's Mitra
([`autogluon/mitra-regressor`](https://huggingface.co/autogluon/mitra-regressor)) on a
validated tabular-regression dataset, then writes the model artifact and a `result.json` with
metrics and provenance.

- Runs as a GPU Kubernetes Job, and also on a CPU-only node (see below).
- DIMER builds the root `Dockerfile` into an ECR image and runs `train.py`.
- `dimer-pipeline.json` at the repo root defines the workbench preprocessing and fine-tuning
  fields. The finetuner build re-reads it on every build.
- Pairs with `mitra-regressor-dataset-validator`.

## Fine-tune (GPU) versus zero-shot (CPU)

Fine-tuning Mitra requires a GPU; on CPU its backward pass hits an unsupported low-precision
path. `train.py` detects the GPU at runtime:

- **GPU present** → fine-tunes Mitra's weights (`fine_tune=True`).
- **No GPU** (CPU node, or the GPU image run without a GPU) → runs Mitra **zero-shot**, in-context
  inference with no weight update (`fine_tune=False`), automatically.

Each run records the effective `mode` (`fine-tune`/`zero-shot`) and `device` in `result.json`.

Two images are provided: `Dockerfile` (GPU, CUDA base, auto-falls back to CPU zero-shot) and
`Dockerfile.cpu` (a lean CPU-only image). DIMER builds the root `Dockerfile`.

The complete pipeline documentation, dataset specification, and the validator are in the
[mitra-regressor-pipeline](https://github.com/kurtvalcorza/mitra-regressor-pipeline) project.
