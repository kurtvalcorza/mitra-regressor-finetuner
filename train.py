"""
DIMER Fine-tuner — Mitra tabular regression (Custom / Other task type)
======================================================================
Fits AutoGluon's Mitra on the validated tabular dataset, evaluates on a holdout
(and on test.csv when present), saves the predictor, and writes result.json.

Design points:
  - single model, no ensembling (fit_weighted_ensemble=False, hyperparameters={"MITRA": {}})
  - resolve and checksum-verify the base weights before fitting; refuse unexpected weights
  - assert the requested model actually trained (AutoGluon can skip a model and still
    report generic success)
  - seed every RNG (Mitra's fit is stochastic; a fixed seed makes runs reproducible)
  - 10,000-row training ceiling (an upstream limit of Mitra)

DIMER contract:
  - dataset at DIMER_DATASET_DIR (raw zip or dir); write model under DIMER_OUTPUT_DIR;
    write result.json to DIMER_RESULT_PATH; POST DIMER_DONE_CALLBACK when done
  - DIMER_TRAIN_DEVICE = "cuda:0" | "cpu"
  - DIMER_HYPERPARAMETERS_JSON / DIMER_PREPROCESSING_ARGS_JSON = the dimer-pipeline.json fields
  - DIMER_MODEL_DIR (optional) = a directory holding uploaded model.safetensors + config.json
  - DIMER_MITRA_REVISION (optional) = required base-model revision (defaults to the pinned one)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

TEMPLATE_NAME = "mitra-regressor-finetuner"
PROBLEM_TYPE = "regression"
BASE_MODEL = "autogluon/mitra-regressor"
# Pinned weights revision and its model.safetensors SHA-256. The loaded weights are verified
# against this checksum before fitting; a mismatch fails the run.
PINNED_MITRA_REVISION = "5f277aa8f69042d39d6ac3612aed18bb9279bd95"
EXPECTED_WEIGHTS_SHA256 = "d8e75c62af0bec2fd404b0ad20a442d951d43ca6d331315cfcc0509b54f2c642"
# config.json is checksum-enforced too: it carries the architecture Mitra builds before loading
# the weights, so a drifted config with matching weights would still change the model.
EXPECTED_CONFIG_SHA256 = "2bc1ed5047f7c25368245e8ad32540a5fa28940b1ec05d3f1f454a09ff5384c1"

MITRA_MODEL_KEY = "MITRA"
MITRA_ROW_LIMIT = 10_000  # hard upstream ceiling
MIN_ROWS_FOR_SPLIT = 20   # below this, don't carve a holdout out of train

# ============================================================================
# CANONICAL DATASET RESOLUTION + ARCHIVE SAFETY
# Keep this block byte-identical across the validator and finetuner containers.
# The CI parity check (scripts/check_shared.py) enforces it.
# ============================================================================
def _safe_int(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


MAX_TOTAL_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_UNCOMPRESSED_BYTES"), 4 * 1024**3)
MAX_COMPRESSION_RATIO = _safe_float(os.getenv("DIMER_MAX_COMPRESSION_RATIO"), 200.0)
# Per-file uncompressed-byte ceiling and a full-read row ceiling. These bound memory BEFORE
# pandas materializes a table, and — unlike the zip-bomb guard — also apply to directory-mode
# inputs. An over-limit table is rejected, never silently truncated (truncation would corrupt
# the validator's row/usable counts and the finetuner's class-preservation guarantees).
MAX_MEMBER_UNCOMPRESSED_BYTES = _safe_int(os.getenv("DIMER_MAX_MEMBER_BYTES"), 1 * 1024**3)
MAX_CSV_ROWS = _safe_int(os.getenv("DIMER_MAX_CSV_ROWS"), 5_000_000)
CSV_READ_CHUNK_ROWS = _safe_int(os.getenv("DIMER_CSV_CHUNK_ROWS"), 200_000)
_DATASET_DIR_ALIASES = {"dataset", "datasets"}


def _normalize_member(name: str) -> str | None:
    """Normalize an archive member or relative path to a canonical form, or None
    for directories/junk. Strips a single leading dataset/ or datasets/ wrapper."""
    if not name or name.endswith("/"):
        return None
    cleaned = name.replace("\\", "/").lstrip("./")
    parts = [p for p in cleaned.split("/") if p not in ("", ".")]
    if not parts:
        return None
    if len(parts) > 1 and parts[0].lower() in _DATASET_DIR_ALIASES:
        parts = parts[1:]
    return "/".join(parts)


def _assert_zip_safe(zf: zipfile.ZipFile) -> None:
    """Reject pathological archives (zip bombs, oversized expansion) before any read."""
    total = 0
    for info in zf.infolist():
        if info.is_dir():
            continue
        total += info.file_size
        if info.file_size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive member {info.filename!r} is {info.file_size:,} uncompressed bytes "
                f"(> {MAX_MEMBER_UNCOMPRESSED_BYTES:,}); refusing (per-file guard)"
            )
        if info.compress_size > 0:
            ratio = info.file_size / info.compress_size
            if ratio > MAX_COMPRESSION_RATIO:
                raise ValueError(
                    f"archive member {info.filename!r} expands {ratio:.0f}x "
                    f"(> {MAX_COMPRESSION_RATIO:.0f}); refusing (zip-bomb guard)"
                )
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"archive expands to {total:,} bytes (> {MAX_TOTAL_UNCOMPRESSED_BYTES:,}); "
            f"refusing to load"
        )


class DatasetSource:
    """Canonical dataset reader: a zip (preferred) or an unzipped directory.

    Resolves train/val/test.csv deterministically and rejects ambiguous archives
    (two members that resolve to the same table). Streams CSV members straight into
    pandas rather than materializing whole members in memory.
    """

    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.archive_name: str | None = None
        self.source_type = "directory"
        self._zip: zipfile.ZipFile | None = None
        self._members: dict[str, list[str]] = {}  # normalized -> [raw member/path, ...]

        zips = sorted(dataset_dir.glob("*.zip"))
        if zips:
            self.archive_name = zips[0].name
            self.source_type = "zip"
            self._zip = zipfile.ZipFile(zips[0])
            _assert_zip_safe(self._zip)
            for raw in self._zip.namelist():
                nm = _normalize_member(raw)
                if nm:
                    self._members.setdefault(nm, []).append(raw)
        else:
            for p in sorted(dataset_dir.rglob("*")):
                if p.is_file():
                    key = str(p.relative_to(dataset_dir)).replace("\\", "/")
                    self._members.setdefault(key, []).append(str(p))

    @property
    def files(self) -> list[str]:
        return sorted(self._members)

    def has_nested_zip(self) -> bool:
        return any(f.lower().endswith(".zip") for f in self._members)

    def candidates(self, stem: str) -> list[str]:
        return [
            f for f in self.files
            if f.lower().endswith(".csv") and Path(f).stem.lower() == stem
        ]

    def duplicate_raw_count(self, stem: str) -> int:
        """Total raw members that resolve to <stem>.csv (across normalized names)."""
        return sum(len(self._members[nm]) for nm in self.candidates(stem))

    def resolve_single(self, stem: str) -> str | None:
        """Return the one normalized <stem>.csv, or None. Raise if ambiguous."""
        cands = self.candidates(stem)
        raw_total = self.duplicate_raw_count(stem)
        if len(cands) > 1 or raw_total > 1:
            raise ValueError(
                f"ambiguous dataset: multiple {stem}.csv candidates "
                f"({cands or raw_total} raw members). Put exactly one {stem}.csv at the "
                f"archive root."
            )
        return cands[0] if cands else None

    def open(self, normalized: str):
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.open(raw)  # ZipExtFile — streams, no full-member read
        return open(raw, "rb")

    def _member_bytes(self, normalized: str) -> int:
        raw = self._members[normalized][0]
        if self._zip is not None:
            return self._zip.getinfo(raw).file_size
        return os.path.getsize(raw)

    def _guard_size(self, normalized: str) -> None:
        """Reject an oversized member/file before pandas materializes it. Covers directory-mode
        inputs too (the zip-bomb guard only sees archives)."""
        size = self._member_bytes(normalized)
        if size > MAX_MEMBER_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"{normalized}: {size:,} uncompressed bytes exceeds the per-file limit "
                f"{MAX_MEMBER_UNCOMPRESSED_BYTES:,} (DIMER_MAX_MEMBER_BYTES); refusing to load."
            )

    def read_csv(self, normalized: str, nrows: int | None = None) -> pd.DataFrame:
        """Read a CSV member with memory bounded before materialization: a raw-byte ceiling per
        file, and — for a full read — a chunked parse that refuses to build a frame past
        MAX_CSV_ROWS rather than OOM on a hostile or accidental giant table. Rows are never
        silently dropped: an over-limit table is rejected, not truncated."""
        self._guard_size(normalized)
        if nrows is not None:
            with self.open(normalized) as handle:
                return pd.read_csv(handle, nrows=nrows)
        with self.open(normalized) as handle:
            chunks: list[pd.DataFrame] = []
            rows = 0
            for chunk in pd.read_csv(handle, chunksize=CSV_READ_CHUNK_ROWS):
                rows += len(chunk)
                if rows > MAX_CSV_ROWS:
                    raise ValueError(
                        f"{normalized}: exceeds the {MAX_CSV_ROWS:,}-row read ceiling "
                        f"(DIMER_MAX_CSV_ROWS); refusing to load the whole table into memory."
                    )
                chunks.append(chunk)
        return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
# ============================================================================
# END shared block
# ============================================================================


@dataclass
class Config:
    dataset_dir: Path
    output_dir: Path
    result_path: Path
    done_callback: str
    callback_timeout: float
    train_device: str
    default_task_type: str
    pipeline_metadata: dict[str, Any]
    target_column: str
    drop_columns: list[str]
    max_train_rows: int
    validation_split: float
    time_limit: int
    seed: int
    eval_metric: str
    fine_tune: bool
    fine_tune_steps: int
    model_dir: Path | None
    required_revision: str
    max_eval_rows: int


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in ("true", "1", "yes")


def load_config() -> Config:
    """Parse the DIMER_* environment. Called inside main()'s try so malformed input
    produces a structured failure result rather than an import-time crash."""
    hp = json.loads(os.getenv("DIMER_HYPERPARAMETERS_JSON", "{}") or "{}")
    pre = json.loads(os.getenv("DIMER_PREPROCESSING_ARGS_JSON", "{}") or "{}")
    model_dir = os.getenv("DIMER_MODEL_DIR", "").strip()
    return Config(
        dataset_dir=Path(os.getenv("DIMER_DATASET_DIR", "/data/dataset")),
        output_dir=Path(os.getenv("DIMER_OUTPUT_DIR", "/data/output")),
        result_path=Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json")),
        done_callback=os.getenv("DIMER_DONE_CALLBACK", "").strip(),
        callback_timeout=float(os.getenv("DIMER_CALLBACK_TIMEOUT_SECONDS", "10")),
        train_device=os.getenv("DIMER_TRAIN_DEVICE", "cuda:0").strip(),
        default_task_type=os.getenv("DIMER_TASK_TYPE", "tabular_regression"),
        pipeline_metadata=json.loads(os.getenv("DIMER_PIPELINE_METADATA_JSON", "{}") or "{}"),
        target_column=str(pre.get("target_column") or "target").strip(),
        drop_columns=[c.strip() for c in str(pre.get("drop_columns") or "").split(",") if c.strip()],
        max_train_rows=int(pre.get("max_train_rows") or MITRA_ROW_LIMIT),
        validation_split=float(pre.get("validation_split") if pre.get("validation_split") is not None else 0.2),
        time_limit=int(hp.get("time_limit_seconds") or 600),
        seed=int(hp.get("seed") or 0),
        eval_metric=str(hp.get("eval_metric") or "mean_absolute_error").strip(),
        fine_tune=_as_bool(hp.get("fine_tune", True), True),
        fine_tune_steps=int(hp.get("fine_tune_steps") or 0),
        model_dir=Path(model_dir) if model_dir else None,
        required_revision=os.getenv("DIMER_MITRA_REVISION", "").strip() or PINNED_MITRA_REVISION,
        max_eval_rows=int(os.getenv("DIMER_MAX_EVAL_ROWS", "50000")),
    )


def log(message: str) -> None:
    print(f"[{TEMPLATE_NAME}] {message}", flush=True)


def write_result(cfg: Config, payload: dict[str, Any]) -> None:
    cfg.result_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    cfg.result_path.write_text(content, encoding="utf-8")


def notify_done_callback(cfg: Config) -> dict[str, Any]:
    if not cfg.done_callback:
        return {"attempted": False, "message": "DIMER_DONE_CALLBACK not set; skipping."}
    parsed = urlparse(cfg.done_callback)
    if parsed.scheme not in {"http", "https"}:
        return {"attempted": False, "message": f"Unsupported scheme: {parsed.scheme}"}
    try:
        response = requests.post(cfg.done_callback, timeout=cfg.callback_timeout)
        return {"attempted": True, "ok": response.ok, "statusCode": response.status_code}
    except requests.RequestException as exc:
        return {"attempted": True, "ok": False, "error": str(exc)}


def _seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# --- Weights resolution + provenance (findings: revision enforcement, uploaded weights) ---

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _snapshot_commit_from_path(path: str) -> str | None:
    m = re.search(r"/snapshots/([0-9a-fA-F]{7,64})/", path.replace("\\", "/"))
    return m.group(1) if m else None


def _hf_hub_dir() -> Path:
    home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    return Path(home) / "hub"


def _install_uploaded_weights(model_dir: Path) -> tuple[str, str]:
    """Materialize an uploaded checkpoint into the HF cache for BASE_MODEL so AutoGluon's
    Mitra loader (which only accepts a repo id) serves these exact bytes offline.
    Returns (synthetic_commit, sha256)."""
    msf, cfgf = model_dir / "model.safetensors", model_dir / "config.json"
    if not msf.exists() or not cfgf.exists():
        raise FileNotFoundError(
            f"DIMER_MODEL_DIR {model_dir} must contain model.safetensors and config.json"
        )
    sha = _sha256_file(str(msf))
    commit = sha[:40]  # deterministic synthetic revision from content
    repo = _hf_hub_dir() / ("models--" + BASE_MODEL.replace("/", "--"))
    snap = repo / "snapshots" / commit
    snap.mkdir(parents=True, exist_ok=True)
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    for name in ("model.safetensors", "config.json"):
        dst = snap / name
        if not dst.exists():
            shutil.copy(model_dir / name, dst)
    (repo / "refs" / "main").write_text(commit)
    os.environ["HF_HUB_OFFLINE"] = "1"
    return commit, sha


def resolve_and_verify_weights(cfg: Config) -> dict[str, Any]:
    """Resolve the weights AutoGluon's Mitra loader will actually use, record provenance,
    and refuse to proceed on unexpected weights. AutoGluon 1.5.0's Mitra loads from a repo id
    via hf_hub_download with no revision arg, so the enforceable guarantee is a SHA-256 check
    of the resolved model.safetensors, not a revision pin."""
    prov: dict[str, Any] = {
        "baseModel": BASE_MODEL,
        "baseModelRevisionExpected": cfg.required_revision,
        "expectedSha256": EXPECTED_WEIGHTS_SHA256,
        "expectedConfigSha256": EXPECTED_CONFIG_SHA256,
    }
    if cfg.model_dir is not None:
        # _install sets HF_HUB_OFFLINE=1 before huggingface_hub is first imported (it reads the
        # flag at import time), so AutoGluon's loader serves the uploaded bytes from the cache.
        commit, sha = _install_uploaded_weights(cfg.model_dir)
        config_sha = _sha256_file(str(cfg.model_dir / "config.json"))
        prov.update({
            "source": "uploaded", "baseModelRevision": commit, "weightsSha256": sha,
            "configSha256": config_sha, "enforced": False,
            "note": "Uploaded weights used verbatim; not checked against the public pinned checksum.",
        })
        return prov

    from huggingface_hub import hf_hub_download

    # Resolve exactly as Mitra's from_pretrained does: hf_hub_download(repo, filename) on main.
    loaded = hf_hub_download(BASE_MODEL, "model.safetensors")
    config_path = hf_hub_download(BASE_MODEL, "config.json")
    commit = _snapshot_commit_from_path(loaded)
    sha = _sha256_file(loaded)
    config_sha = _sha256_file(config_path)
    prov.update({
        "source": "huggingface", "baseModelRevision": commit, "weightsSha256": sha,
        "configSha256": config_sha, "enforced": True,
    })
    if cfg.required_revision == PINNED_MITRA_REVISION and sha != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(
            f"Mitra weights to load have SHA-256 {sha}, expected {EXPECTED_WEIGHTS_SHA256} for "
            f"pinned revision {PINNED_MITRA_REVISION}. The hub 'main' may have drifted. Bake the "
            f"pinned revision into the image (Dockerfile Option A) or set DIMER_MITRA_REVISION."
        )
    if cfg.required_revision == PINNED_MITRA_REVISION and config_sha != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            f"Mitra config.json has SHA-256 {config_sha}, expected {EXPECTED_CONFIG_SHA256} for "
            f"pinned revision {PINNED_MITRA_REVISION}. The hub 'main' may have drifted; bake the "
            f"pinned revision (Dockerfile Option A) or set DIMER_MITRA_REVISION."
        )
    if commit and cfg.required_revision and commit != cfg.required_revision:
        raise RuntimeError(
            f"Mitra weights resolved to revision {commit}, required {cfg.required_revision}."
        )
    return prov


# --- Data preparation ---

def _prepare_frames(cfg: Config, source: DatasetSource) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Return (train, val, test) with target present, drop_columns removed, capped to the
    ceiling. Rows with a non-finite target are dropped (they are unusable for regression)."""
    train_path = source.resolve_single("train")
    if train_path is None:
        raise FileNotFoundError("no train.csv in the dataset (validator should have caught this)")
    train = source.read_csv(train_path)
    if cfg.target_column not in train.columns:
        raise KeyError(f"target column '{cfg.target_column}' not in {list(train.columns)}")

    # Never drop the target, even if a malformed config lists it in drop_columns.
    drop = [c for c in cfg.drop_columns if c in train.columns and c != cfg.target_column]
    train = train.drop(columns=drop)
    train[cfg.target_column] = pd.to_numeric(train[cfg.target_column], errors="coerce")
    train = train[np.isfinite(train[cfg.target_column])]

    def _prep_holdout(df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in cfg.drop_columns if c in df.columns and c != cfg.target_column]
        df = df.drop(columns=cols)
        df[cfg.target_column] = pd.to_numeric(df[cfg.target_column], errors="coerce")
        return df[np.isfinite(df[cfg.target_column])]

    val_path = source.resolve_single("val")
    if val_path is not None:
        val = _prep_holdout(source.read_csv(val_path))
    else:
        val_frac = min(max(cfg.validation_split, 0.0), 0.4)
        if val_frac > 0 and len(train) > MIN_ROWS_FOR_SPLIT:
            val = train.sample(frac=val_frac, random_state=cfg.seed)
            train = train.drop(index=val.index)
        else:
            val = pd.DataFrame(columns=train.columns)

    ceiling = min(cfg.max_train_rows, MITRA_ROW_LIMIT)
    if len(train) > ceiling:
        log(f"Sampling train {len(train)} -> {ceiling} rows (seed={cfg.seed}).")
        train = train.sample(n=ceiling, random_state=cfg.seed)

    test_path = source.resolve_single("test")
    test = _prep_holdout(source.read_csv(test_path)) if test_path is not None else None

    return (train.reset_index(drop=True), val.reset_index(drop=True),
            test.reset_index(drop=True) if test is not None else None)


# AutoGluon stores regression metrics higher-is-better, so error metrics come out negative;
# negate these keys to report conventional positive values (finding: valEvaluation signs).
_REG_LOWER_IS_BETTER = {
    "root_mean_squared_error", "mean_squared_error", "mean_absolute_error",
    "median_absolute_error", "mean_absolute_percentage_error",
    "symmetric_mean_absolute_percentage_error", "root_mean_squared_logarithmic_error",
}


def _normalize_regression_eval(raw: dict[str, Any]) -> dict[str, float]:
    """Present AutoGluon's regression evaluate() in conventional form: error metrics come back
    sign-flipped negative (higher-is-better), so negate them; r2/correlation stay as-is. This
    makes valEvaluation agree with the positive mae/rmse computed directly above."""
    return {k: float(-v if k in _REG_LOWER_IS_BETTER else v) for k, v in raw.items()}


def _regression_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_true - y_pred
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def _score_holdout(cfg: Config, predictor, frame: pd.DataFrame) -> dict[str, Any]:
    """Score a labelled holdout. Predictions are used as-is (no clipping): the served
    artifact returns raw predictions, so the reported error must reflect that."""
    if cfg.max_eval_rows and len(frame) > cfg.max_eval_rows:
        log(f"Capping evaluation set {len(frame)} -> {cfg.max_eval_rows} rows (seed={cfg.seed}).")
        frame = frame.sample(n=cfg.max_eval_rows, random_state=cfg.seed)
    y_true = frame[cfg.target_column].to_numpy(dtype=float)
    y_pred = np.asarray(predictor.predict(frame.drop(columns=[cfg.target_column])), dtype=float)
    scores = _regression_scores(y_true, y_pred)
    scores["rows"] = int(len(frame))
    try:
        scores["evaluation"] = _normalize_regression_eval(predictor.evaluate(frame, silent=True))
    except Exception as exc:  # noqa: BLE001 - full evaluation is best-effort
        scores["evaluationError"] = str(exc)
    return scores


# DIMER/AutoGluon eval-metric name -> Mitra's native early-stopping metric. Unmapped names
# fall back to Mitra's default and only drive AutoGluon's reported metric.
_MITRA_METRIC_MAP = {
    "mean_absolute_error": "mae", "mae": "mae",
    "root_mean_squared_error": "rmse", "rmse": "rmse",
    "mean_squared_error": "mse", "mse": "mse",
    "r2": "r2",
}


def _mitra_metric(name: str) -> str | None:
    return _MITRA_METRIC_MAP.get(name.strip().lower())


def _fit_and_evaluate(cfg: Config, train: pd.DataFrame, val: pd.DataFrame,
                      test: pd.DataFrame | None) -> dict[str, Any]:
    requested_cpu = cfg.train_device.lower() == "cpu"
    if requested_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _seed_everything(cfg.seed)
    try:
        import torch

        gpu_available = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        gpu_available = False
    use_gpu = gpu_available and not requested_cpu

    fine_tune = cfg.fine_tune
    if not use_gpu and fine_tune:
        why = "no GPU is available" if not gpu_available else "CPU was requested"
        log(f"Running zero-shot (fine_tune=False): {why}; Mitra fine-tuning requires a GPU.")
        fine_tune = False
    # Propagate the run seed and (when mappable) the eval metric into Mitra itself, not just
    # AutoGluon's reporting: "seed" seeds Mitra's val-split/augmentation RNG (ConfigRun.seed),
    # and "metric" drives its fine-tune early-stopping. NOTE: AutoGluon 1.5.0's Mitra disables
    # its global set_seed (an upstream FIXME), so a fixed seed makes the internal split
    # reproducible but not the full fit — a known upstream limit, not a bug here.
    mitra_hp: dict[str, Any] = {"fine_tune": fine_tune, "seed": cfg.seed}
    if fine_tune and cfg.fine_tune_steps:
        mitra_hp["fine_tune_steps"] = cfg.fine_tune_steps
    mitra_metric = _mitra_metric(cfg.eval_metric)
    if mitra_metric is not None:
        mitra_hp["metric"] = mitra_metric
    else:
        log(f"eval_metric '{cfg.eval_metric}' has no Mitra-native early-stopping equivalent; "
            f"Mitra keeps its default metric (AutoGluon still reports '{cfg.eval_metric}').")

    from autogluon.tabular import TabularPredictor

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    predictor_path = cfg.output_dir / "mitra_predictor"

    predictor = TabularPredictor(
        label=cfg.target_column,
        problem_type=PROBLEM_TYPE,
        eval_metric=cfg.eval_metric,
        path=str(predictor_path),
        verbosity=2,
    )
    predictor.fit(
        train,
        hyperparameters={MITRA_MODEL_KEY: mitra_hp},
        fit_weighted_ensemble=False,
        time_limit=cfg.time_limit,
    )

    trained = list(predictor.model_names())
    if not trained:
        raise RuntimeError(
            f"{MITRA_MODEL_KEY} did not train. A common cause is AutoGluon's memory guard "
            f"(it needs the projected footprint under the available-RAM threshold). Request "
            f"a larger GPU/memory profile for this pipeline, then re-run. Check the fit log "
            f"for 'Not enough memory to safely train model'."
        )
    if not any("mitra" in m.lower() for m in trained):
        raise RuntimeError(
            f"expected Mitra but AutoGluon trained {trained} — refusing to report a result "
            f"for a model that was not the one requested"
        )

    metrics: dict[str, Any] = {
        "trainedModels": trained,
        "trainRows": int(len(train)),
        "mode": "fine-tune" if fine_tune else "zero-shot",
        "device": "cuda" if use_gpu else "cpu",
        "evalMetric": cfg.eval_metric,
        "mitraMetric": mitra_metric or "<mitra-default>",
        "mitraSeed": cfg.seed,
    }
    if len(val) > 0:
        val_scores = _score_holdout(cfg, predictor, val)
        metrics["valRows"] = val_scores.pop("rows")
        metrics.update({k: v for k, v in val_scores.items() if k in ("mae", "rmse")})
        if "evaluation" in val_scores:
            metrics["valEvaluation"] = val_scores["evaluation"]
        headline = metrics["rmse"] if cfg.eval_metric == "root_mean_squared_error" else metrics["mae"]
        metrics["headlineMetric"] = cfg.eval_metric
        metrics["headlineScore"] = float(headline)
        log(f"Holdout {cfg.eval_metric}={headline:.4f} on {metrics['valRows']} rows.")
    else:
        metrics["valRows"] = 0
        metrics["note"] = "No validation rows available; trained on all rows without holdout."

    if test is not None and len(test) > 0:
        test_scores = _score_holdout(cfg, predictor, test)
        metrics["test"] = {
            "rows": test_scores["rows"], "mae": test_scores.get("mae"),
            "rmse": test_scores.get("rmse"),
        }
        log(f"Test MAE={test_scores.get('mae'):.4f} on {test_scores['rows']} rows.")

    metrics["artifactPath"] = str(predictor_path)
    return metrics


def run(cfg: Config) -> int:
    provenance = resolve_and_verify_weights(cfg)
    source = DatasetSource(cfg.dataset_dir)
    try:
        train, val, test = _prepare_frames(cfg, source)
    finally:
        source.close()
    metrics = _fit_and_evaluate(cfg, train, val, test)
    provenance["dataset"] = _dataset_sha256(cfg)
    provenance["autogluonVersion"] = getattr(sys.modules.get("autogluon.tabular"), "__version__", None)
    headline = metrics.get("headlineScore")
    payload = {
        "successful": True,
        "message": (
            f"Mitra {metrics['mode']} succeeded on {metrics['trainRows']} rows"
            + (f"; holdout {cfg.eval_metric} {headline:.4f}." if headline is not None else ".")
        ),
        "metrics": metrics,
        "artifacts": {"modelDir": str(cfg.output_dir / "mitra_predictor")},
        "provenance": provenance,
        "metadata": {
            "template": TEMPLATE_NAME,
            "taskType": cfg.default_task_type,
            "baseModel": BASE_MODEL,
            "targetColumn": cfg.target_column,
            "dropColumns": cfg.drop_columns,
            "seed": cfg.seed,
            "timeLimitSeconds": cfg.time_limit,
            "evalMetric": cfg.eval_metric,
            "trainDevice": cfg.train_device,
        },
    }
    write_result(cfg, payload)
    log(f"Callback: {json.dumps(notify_done_callback(cfg), sort_keys=True)}")
    return 0


def _dataset_sha256(cfg: Config) -> dict[str, Any] | None:
    h = hashlib.sha256()
    zips = sorted(cfg.dataset_dir.glob("*.zip"))
    if zips:
        with open(zips[0], "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return {"file": zips[0].name, "sha256": h.hexdigest()}
    csvs = sorted(cfg.dataset_dir.rglob("*.csv"))
    if not csvs:
        return None
    for p in csvs:
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return {"files": [p.name for p in csvs], "sha256": h.hexdigest()}


def _failure_provenance(cfg: Config | None) -> dict[str, Any]:
    prov: dict[str, Any] = {"baseModel": BASE_MODEL, "baseModelRevisionExpected": PINNED_MITRA_REVISION}
    if cfg is not None:
        try:
            prov["dataset"] = _dataset_sha256(cfg)
        except Exception:  # noqa: BLE001
            pass
    prov["autogluonVersion"] = getattr(sys.modules.get("autogluon.tabular"), "__version__", None)
    return prov


def main() -> int:
    cfg: Config | None = None
    try:
        cfg = load_config()
        return run(cfg)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "successful": False,
            "message": f"Mitra fine-tuning failed: {exc}",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
            "provenance": _failure_provenance(cfg),
            "metadata": {
                "template": TEMPLATE_NAME,
                "taskType": (cfg.default_task_type if cfg else "tabular_regression"),
            },
        }
        try:
            if cfg is not None:
                write_result(cfg, payload)
                notify_done_callback(cfg)
            else:
                fallback = Path(os.getenv("DIMER_RESULT_PATH", "/data/results/result.json"))
                fallback.parent.mkdir(parents=True, exist_ok=True)
                fallback.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as write_exc:  # noqa: BLE001
            log(f"Failed to persist crash result: {write_exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
