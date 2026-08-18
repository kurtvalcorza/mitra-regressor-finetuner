# dimer-finetuner-mitra-regressor

DIMER fine-tuner for the Mitra regressor pipeline. It fine-tunes AutoGluon's Mitra
([`autogluon/mitra-regressor`](https://huggingface.co/autogluon/mitra-regressor)) on a
validated tabular-regression dataset, then writes the model artifact and a `result.json` with
metrics and provenance.

- Runs as a GPU Kubernetes Job. Mitra also runs CPU-only — see the project docs.
- DIMER builds the root `Dockerfile` into an ECR image and runs `train.py`.
- `dimer-pipeline.json` at the repo root defines the workbench preprocessing and fine-tuning
  fields. The finetuner build re-reads it on every build.
- Pairs with `dimer-dataset-validator-mitra-regressor`.

The complete pipeline documentation, dataset specification, and the validator are in the
[mitra-regressor-pipeline](https://github.com/kurtvalcorza/mitra-regressor-pipeline) project.
