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
    assert prov.get("configSha256")  # finding 3: config.json is hashed too

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


def test_mitra_metric_map():
    assert T._mitra_metric("root_mean_squared_error") == "rmse"
    assert T._mitra_metric("MEAN_ABSOLUTE_ERROR") == "mae"
    assert T._mitra_metric("r2") == "r2"
    assert T._mitra_metric("explained_variance") is None  # unmapped -> Mitra default


def test_regression_eval_sign_normalized():
    raw = {"root_mean_squared_error": -3.5, "mean_absolute_error": -2.0, "r2": 0.9}
    out = T._normalize_regression_eval(raw)
    assert out["root_mean_squared_error"] == 3.5
    assert out["mean_absolute_error"] == 2.0
    assert out["r2"] == 0.9  # correlation-style metric unchanged


def test_member_byte_cap_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_MEMBER_UNCOMPRESSED_BYTES", 100)  # smaller than the CSV
    _zip(tmp_path, {"train.csv": _frame(60)})
    with pytest.raises(ValueError):
        T.DatasetSource(tmp_path)


def test_row_ceiling_rejected_not_truncated(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_CSV_ROWS", 10)
    monkeypatch.setattr(T, "CSV_READ_CHUNK_ROWS", 4)
    _zip(tmp_path, {"train.csv": _frame(60)})  # 60 rows > ceiling of 10
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_directory_mode_byte_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "MAX_MEMBER_UNCOMPRESSED_BYTES", 50)
    _frame(60).to_csv(tmp_path / "train.csv", index=False)  # no zip -> directory mode
    src = T.DatasetSource(tmp_path)
    try:
        with pytest.raises(ValueError):
            src.read_csv("train.csv")
    finally:
        src.close()


def test_safe_parse_bad_env():
    assert T._safe_int("nope", 42) == 42
    assert T._safe_float(None, 2.5) == 2.5


def test_normalize_device():
    # cpu passes through; the DIMER-documented bare-integer form becomes cuda:<n>.
    assert T._normalize_device("cpu") == "cpu"
    assert T._normalize_device("CPU") == "cpu"
    assert T._normalize_device("cuda:0") == "cuda:0"
    assert T._normalize_device("cuda:1") == "cuda:1"
    assert T._normalize_device("CUDA:2") == "cuda:2"
    assert T._normalize_device("0") == "cuda:0"      # bare integer -> cuda:0 (docs call-out)
    assert T._normalize_device("3") == "cuda:3"      # explicit index honored
    assert T._normalize_device("gpu") == "cuda:0"
    assert T._normalize_device("cuda") == "cuda:0"
    assert T._normalize_device("") == "cuda:0"
    assert T._normalize_device(None) == "cuda:0"
    assert T._normalize_device("mps") == "cpu"       # unknown accelerator -> safe CPU fallback
    assert T._normalize_device("cuda:x") == "cpu"    # malformed index -> CPU fallback


class _FakePost:
    """Records callback POST URLs and returns a minimal ok response (stands in for requests.post)."""

    def __init__(self):
        self.urls: list[str] = []

    def __call__(self, url, timeout=None):
        self.urls.append(url)
        return type("_Resp", (), {"ok": True, "status_code": 200})()


def test_config_error_still_notifies_callback(tmp_path, monkeypatch):
    """A config-parse failure must still POST the done callback, or the UI hangs at the train stage."""
    fake = _FakePost()
    monkeypatch.setattr(T.requests, "post", fake)
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "http://backend/done")
    monkeypatch.setenv("DIMER_HYPERPARAMETERS_JSON", "{not valid json")   # load_config raises
    assert T.main() == 1
    assert fake.urls == ["http://backend/done"]


def test_write_failure_still_notifies_callback(tmp_path, monkeypatch):
    """A crash plus a result-write failure must still POST the callback (decoupled from the write)."""
    fake = _FakePost()
    monkeypatch.setattr(T.requests, "post", fake)

    def _run_boom(cfg):
        raise RuntimeError("fit crashed")

    def _write_boom(cfg, payload):
        raise OSError("disk full")

    monkeypatch.setattr(T, "run", _run_boom)
    monkeypatch.setattr(T, "write_result", _write_boom)
    monkeypatch.setenv("DIMER_DATASET_DIR", str(tmp_path))
    monkeypatch.setenv("DIMER_RESULT_PATH", str(tmp_path / "result.json"))
    monkeypatch.setenv("DIMER_DONE_CALLBACK", "http://backend/done")
    assert T.main() == 1
    assert fake.urls == ["http://backend/done"]
