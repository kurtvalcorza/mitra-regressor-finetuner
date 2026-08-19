"""Unit tests for the Mitra regressor finetuner (no model fitting)."""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as T  # noqa: E402


def _zip(tmp: Path, members: dict[str, pd.DataFrame]) -> Path:
    p = tmp / "d.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for arc, df in members.items():
            buf = io.StringIO()
            df.to_csv(buf, index=False)
            zf.writestr(arc, buf.getvalue())
    return p


def _cfg(tmp: Path, **over):
    base = dict(
        dataset_dir=tmp, output_dir=tmp / "out", result_path=tmp / "r.json", done_callback="",
        callback_timeout=1.0, train_device="cpu", default_task_type="tabular_regression",
        pipeline_metadata={}, target_column="target", drop_columns=[], max_train_rows=10000,
        validation_split=0.2, time_limit=60, seed=0, eval_metric="mean_absolute_error",
        fine_tune=True, fine_tune_steps=0, model_dir=None,
        required_revision=T.PINNED_MITRA_REVISION, max_eval_rows=50000,
    )
    base.update(over)
    return T.Config(**base)


def _frame(n=60):
    return pd.DataFrame({"f1": range(n), "f2": np.arange(n) * 1.5, "target": np.arange(n) - 30.0})


def test_ambiguous_train_rejected(tmp_path):
    _zip(tmp_path, {"train.csv": _frame(), "dataset/train.csv": _frame()})
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.resolve_single("train")
    finally:
        src.close()


def test_prepare_frames_drops_nonfinite_and_keeps_negatives(tmp_path):
    df = _frame(60)
    df.loc[df.index[:10], "target"] = np.nan
    _zip(tmp_path, {"train.csv": df})
    src = T.DatasetSource(tmp_path)
    try:
        train, val, test = T._prepare_frames(_cfg(tmp_path), src)
    finally:
        src.close()
    assert len(train) + len(val) == 50           # 10 non-finite dropped
    assert (train["target"] < 0).any()           # negatives preserved (no clipping upstream)
    assert test is None


def test_regression_scores_no_clipping():
    # Predictions below zero must count as-is; the served artifact returns raw predictions.
    y_true = np.array([0.0, 1.0, -5.0])
    y_pred = np.array([-2.0, 1.0, -5.0])
    s = T._regression_scores(y_true, y_pred)
    assert s["mae"] == pytest.approx(2.0 / 3.0)   # only first row errs, by 2.0


def test_uploaded_weights_installed_and_resolvable(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    model_dir = tmp_path / "weights"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"FAKE_REG_WEIGHTS")
    (model_dir / "config.json").write_text('{"dim": 512}')

    prov = T.resolve_and_verify_weights(_cfg(tmp_path, model_dir=model_dir))
    assert prov["source"] == "uploaded"

    from huggingface_hub import hf_hub_download
    path = hf_hub_download(T.BASE_MODEL, "model.safetensors")
    assert Path(path).read_bytes() == b"FAKE_REG_WEIGHTS"


def test_finetuner_never_drops_target(tmp_path):
    _zip(tmp_path, {"train.csv": _frame(60)})
    src = T.DatasetSource(tmp_path)
    try:
        # target listed in drop_columns must not cause a KeyError; the target survives.
        train, val, test = T._prepare_frames(_cfg(tmp_path, drop_columns=["target", "f1"]), src)
    finally:
        src.close()
    assert "target" in train.columns and len(train) > 0


def test_safe_parse_bad_env():
    assert T._safe_int("nope", 42) == 42
    assert T._safe_float(None, 2.5) == 2.5
