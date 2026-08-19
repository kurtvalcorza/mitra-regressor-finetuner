# Issue #1 — acceptance record (regression finetuner)

Traceability for *Harden DIMER regression finetuning contract, evaluation, and provenance*.
Each acceptance criterion maps to the code and test that satisfy it on `main`. The runtime
path was verified end-to-end on a GPU (5070 Ti) 2026-08-19: real fine-tune → save → reload →
predict, with the values below observed in `result.json`.

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Generic regression no longer clips negative predictions | ✅ | `_regression_scores` has no `np.clip`; `_score_holdout` predicts as-is; `tests/test_train.py::test_regression_scores_no_clipping` |
| Runtime config failures write structured DIMER results | ✅ | `load_config()` parses inside `main()`'s protected path; failure writes `result.json` and attempts the callback |
| Selected Base Model is the model actually loaded | ✅ | `resolve_and_verify_weights` resolves the exact `model.safetensors` the AutoGluon Mitra loader will use; uploaded weights installed into the loader cache via `_install_uploaded_weights` |
| Provenance identifies the exact loaded checkpoint/revision | ✅ | `provenance` records `baseModelRevision`, `weightsSha256`, **and `configSha256`**, both checksum-enforced against pinned values; no lexicographic snapshot selection |
| `test.csv` behavior matches the public contract | ✅ | `test.csv` scored only after fitting, reported under `metrics.test` separate from validation |
| Result messages accurately report execution mode | ✅ | `metrics.mode` = `fine-tune`/`zero-shot` from GPU availability; surfaced in the top-level message. Observed `mode=fine-tune` on GPU |
| Evaluation is resource-bounded/chunked where needed | ✅ | `max_eval_rows` cap on holdouts + chunked-read row ceiling in the shared block |
| CI covers regression correctness and DIMER contract tests | ✅ | `.github/workflows/ci.yml` (unit + `check_shared`); `.github/workflows/integration.yml` (real-stack GPU smoke: build → offline load → fit → save → reload → predict) |

Also fixed this round: `valEvaluation` error-metric signs normalized to conventional positive
values (`_normalize_regression_eval`; `test_regression_eval_sign_normalized`), and the run
`seed` + native `metric` propagated into Mitra (`metrics.mitraSeed`/`metrics.mitraMetric`).

No open items in-repo. `Closes #1`.
