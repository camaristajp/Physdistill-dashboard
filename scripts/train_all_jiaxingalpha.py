"""
train_all_jiaxing.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Unified Training Script — Teacher TFT + Student Baseline + Student KD
EV Charging Demand Forecasting · Jiaxing Dataset

Runs all three models sequentially in a single script.

Usage:
    # Train all three models
    python train_all_jiaxing.py

    # Train specific models only
    python train_all_jiaxing.py --models teacher
    python train_all_jiaxing.py --models teacher baseline
    python train_all_jiaxing.py --models teacher kd
    python train_all_jiaxing.py --models baseline kd

Outputs per model:
    models/teacher_tft_jiaxingv3/
        checkpoints/best_teacher.ckpt
        metrics.json
        teacher_soft_targets.npz       ← required for KD student
        training_curve.png
        predictions.png                ← individual 4-panel forecast
        pred_vs_actual.png             ← scatter: predicted vs actual

    models/student_baseline_jiaxing/
        checkpoints/best_student.ckpt
        metrics.json
        training_curve.png
        predictions.png
        pred_vs_actual.png / .pdf

    models/student_kd_tft111_jiaxing1/
        checkpoints/best_student.ckpt
        metrics.json
        training_curve.png
        predictions.png
        pred_vs_actual.png / .pdf

    comparison/
        predictions_3model_panel.png   ← combined Pred vs Actual for all 3

Config files:
    data/tft_artifacts_jiaxing/tft_config.json          ← Teacher  (max_encoder_length=336)
    data/tft_artifacts_jiaxing/student_tft_config.json  ← Students (max_encoder_length=168)

    If student_tft_config.json does not exist, it is auto-generated at startup
    by copying tft_config.json and patching max_encoder_length to 168.
    You can then edit it manually for any other student-specific overrides.

Notes:
    • Teacher must run (or have soft targets) before KD student.
    • Student Baseline and KD student share student_tft_config.json (168).
    • Teacher uses tft_config.json (336).  The two configs are intentionally
      separate — do NOT merge them or pass one to the wrong pipeline.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse, gc, json, math, os, pickle, platform, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Numeric / ML ──────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, RobustScaler

# ── PyTorch Lightning + PyTorch Forecasting ───────────────────────────────────
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED PATHS
# ─────────────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR        = Path("data/tft_artifacts_jiaxing")
CONFIG_PATH          = ARTIFACTS_DIR / "tft_config.json"           # Teacher  (max_encoder_length=336)
STUDENT_CONFIG_PATH  = ARTIFACTS_DIR / "student_tft_config.json"  # Students (max_encoder_length=168)
TRAIN_PATH           = ARTIFACTS_DIR / "train_data.csv"
VAL_PATH             = ARTIFACTS_DIR / "val_data.csv"
FEATURE_SCALER_PKL   = ARTIFACTS_DIR / "feature_scaler.pkl"
LABEL_ENCODERS_PKL   = ARTIFACTS_DIR / "label_encoders.pkl"

# Per-model output directories
TEACHER_OUTPUT_DIR   = Path("model/teacher_tft_jiaxingv3")
BASELINE_OUTPUT_DIR  = Path("model/student_baseline_jiaxing")
KD_OUTPUT_DIR        = Path("model/student_kd_tft111_jiaxing1")
COMPARISON_DIR       = Path("comparison")

SOFT_TARGETS_PATH    = TEACHER_OUTPUT_DIR / "teacher_soft_targets.npz"

# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEED         = 42
BATCH_SIZE   = 128
QUANTILES    = [0.02, 0.10, 0.25, 0.50, 0.75, 0.90, 0.98]
MEDIAN_IDX   = QUANTILES.index(0.50)
Q_IDX        = {q: i for i, q in enumerate(QUANTILES)}
ZERO_THRESH  = 0.01
EPS          = 1e-8
WARMUP_EPOCHS = 2

SCALED_FEATURE_COLS: List[str] = [
    "Fee", "Lagged_Energy_168h", "Rolling_Mean_Energy_24h",
    "Lagged_Energy_24h", "Rolling_Mean_Energy_168h",
    "Temperature_C", "Relative_Humidity", "Precipitation_mm",
]
STATIC_CAT_COLS: List[str] = ["Location_Information", "District_Name"]
KNOWN_CAT_COLS: List[str] = [
    "Month", "Hour_of_Day", "Day_of_Week",
    "Is_Holiday", "Is_Weekend", "ToU_Period", "Hour_of_Week",
]
KNOWN_REAL_ALLOWLIST = {"Fee"}

KD_ALPHA       = 0.3                  # weight of task loss in student KD; (1-alpha) = distillation weight
KD_ALPHA_SWEEP = [0.40, 0.5, 0.60]  # alpha values to sweep in the KD pipeline

STUDENT_BASE_LR      = 3e-4
STUDENT_WEIGHT_DECAY = 1e-4

PREDICT_TRAINER_KWARGS = {"accelerator": "auto", "devices": 1}
NUM_WORKERS = 0 if platform.system() == "Windows" else min(4, os.cpu_count() or 1)

PALETTE = {
    "train": "#2563EB", "val": "#DC2626",
    "gap": "#7C3AED", "grid": "#E5E7EB",
    "median": "#2563EB",
}

# Per-model colours for the combined comparison panel
MODEL_COLORS = {
    "Teacher":          {"pred": "#E07B2A", "band": "#F0A85C", "actual": "#5B3A29"},
    "Student Baseline": {"pred": "#1A7FBD", "band": "#6BBDE0", "actual": "#0D3B55"},
    "Student KD":       {"pred": "#27AE60", "band": "#7DCEA0", "actual": "#145A32"},
}

# ─────────────────────────────────────────────────────────────────────────────
# LOSS CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class SparseQuantileLoss(QuantileLoss):
    def __init__(self, quantiles: list, nonzero_weight: float = 5.0,
                 overpredict_penalty: float = 2.5, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.nonzero_weight      = nonzero_weight
        self.overpredict_penalty = overpredict_penalty

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 3:
            target = target[..., 0]
        pinball_terms = []
        for i, q in enumerate(self.quantiles):
            err = target - y_pred[..., i]
            pinball_terms.append(torch.max((q - 1.0) * err, q * err))
        base         = torch.stack(pinball_terms, dim=-1).mean(dim=-1)
        is_nonzero   = (target > ZERO_THRESH).float()
        weight       = 1.0 + (self.nonzero_weight - 1.0) * is_nonzero
        median_pred  = y_pred[..., MEDIAN_IDX]
        over_penalty = torch.relu(median_pred) * (1.0 - is_nonzero) * self.overpredict_penalty
        p90_undershoot = torch.relu(target - y_pred[..., Q_IDX[0.90]]) * is_nonzero * 0.5
        return (base * weight + over_penalty + p90_undershoot).mean()


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CLASSES
# ─────────────────────────────────────────────────────────────────────────────
class RawLossTFT(TemporalFusionTransformer):
    """Base TFT subclass that overrides PTF's rescaling and computes loss on raw log1p target."""

    def _compute_raw_loss(self, batch, stage: str) -> torch.Tensor:
        x, y_raw = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch, None)
        with torch.set_grad_enabled(stage == "train"):
            out = self(x)
        if isinstance(out, dict):
            ypred = out.get("prediction", out.get("output", next(iter(out.values()))))
        elif hasattr(out, "prediction"):
            ypred = out.prediction
        else:
            ypred = out[0] if isinstance(out, (tuple, list)) else out
        if y_raw is not None:
            raw_target = y_raw[0] if isinstance(y_raw, (tuple, list)) else y_raw
        else:
            raw_target = x.get("decoder_target")
            if raw_target is None:
                raise KeyError(f"Cannot find raw target. x keys: {list(x.keys())}")
        if isinstance(raw_target, (tuple, list)):
            raw_target = raw_target[0]
        loss_val = self.loss.loss(ypred.float(), raw_target.float())
        log_key  = "train_loss" if stage == "train" else "val_loss"
        self.log(log_key, loss_val, on_step=(stage == "train"),
                 on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_val

    def training_step(self, batch, batch_idx):
        return self._compute_raw_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_raw_loss(batch, "val")

    def training_epoch_end(self, outputs):
        if not outputs:
            return
        losses = [o[0] if isinstance(o, (tuple, list)) else o for o in outputs]
        losses = [v.detach().float() for v in losses if isinstance(v, torch.Tensor)]
        if losses:
            self.log("train_loss_epoch", torch.stack(losses).mean(),
                     on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        losses = [o[0] if isinstance(o, (tuple, list)) else o for o in outputs]
        losses = [v.detach().float() for v in losses if isinstance(v, torch.Tensor)]
        if losses:
            self.log("val_loss", torch.stack(losses).mean(),
                     on_epoch=True, prog_bar=True, sync_dist=True)


class StudentBaseline(RawLossTFT):
    """Standalone Student TFT — no Knowledge Distillation (ablation lower bound)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._task_losses: List[float] = []
        self._val_losses:  List[float] = []
        self._batch_task:  List[float] = []

    def training_step(self, batch, batch_idx):
        return self._compute_raw_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_raw_loss(batch, "val")


class SoftTargetRegistry:
    """Wraps teacher_soft_targets.npz for O(1) lookup per (station, time_idx) pair."""
    def __init__(self, npz_path: Path):
        data     = np.load(npz_path, allow_pickle=True)
        soft     = data["quantiles_log1p"].astype(np.float32)
        stations = data["station_names"].astype(str)
        time_idx = data["time_index"].astype(np.int64)
        self.warmup = int(data["warmup_excluded_per_station"]) if "warmup_excluded_per_station" in data else 336
        self._map: Dict[Tuple[str, int], np.ndarray] = {
            (str(s), int(t)): soft[i]
            for i, (s, t) in enumerate(zip(stations, time_idx))
        }
        print(f"  [KD] Soft target registry: {len(self._map):,} sequences loaded")
        print(f"  [KD] Warmup exclusion    : Time_Index 0 → {self.warmup - 1} per station")

    def lookup_batch(self, station_names, time_indices, device):
        B = len(station_names)
        soft_batch = np.full((B, 24, len(QUANTILES)), np.nan, dtype=np.float32)
        for i, (stn, tidx) in enumerate(zip(station_names, time_indices)):
            if int(tidx) < self.warmup:   # skip warm-up sequences (unreliable teacher preds)
                continue
            val = self._map.get((stn, int(tidx)))
            if val is not None:
                soft_batch[i] = val
        t = torch.from_numpy(soft_batch).to(device)
        valid_mask = ~torch.isnan(t).any(dim=-1).any(dim=-1)
        return t, valid_mask


class StudentKDTFT(TemporalFusionTransformer):
    """Student TFT with knowledge distillation loss (task + MSE vs teacher soft targets)."""
    _soft_reg:          Optional["SoftTargetRegistry"] = None
    _group_decode:      Optional[Dict[int, str]]       = None
    _soft_targets_path: Optional[str]                  = None
    _group_decode_map:  Optional[Dict[int, str]]       = None

    def on_train_start(self):
        if self._soft_reg is None and self._soft_targets_path is not None:
            _path = Path(self._soft_targets_path)
            if _path.exists():
                self._soft_reg = SoftTargetRegistry(_path)
            else:
                warnings.warn(f"[KD] on_train_start: soft targets path not found: {_path}.",
                              RuntimeWarning, stacklevel=2)
        if self._group_decode is None and self._group_decode_map is not None:
            self._group_decode = self._group_decode_map
        if self._soft_reg is not None:
            n = len(self._soft_reg._map)
            print(f"  [KD] on_train_start: registry active — {n:,} sequences on "
                  f"{next(self.parameters()).device}")
        else:
            warnings.warn("[KD] on_train_start: _soft_reg is None — task loss only.",
                          RuntimeWarning, stacklevel=2)

    def _forward_and_extract(self, batch):
        x, y_raw = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch, None)
        out = self(x)
        if isinstance(out, dict):
            ypred = out.get("prediction", out.get("output", next(iter(out.values()))))
        elif hasattr(out, "prediction"):
            ypred = out.prediction
        else:
            ypred = out[0] if isinstance(out, (tuple, list)) else out
        if y_raw is not None:
            raw_target = y_raw[0] if isinstance(y_raw, (tuple, list)) else y_raw
        else:
            raw_target = x.get("decoder_target")
            if raw_target is None:
                raise KeyError(f"Cannot find raw target. x keys: {list(x.keys())}")
        if isinstance(raw_target, (tuple, list)):
            raw_target = raw_target[0]
        return x, ypred.float(), raw_target.float()

    def training_step(self, batch, batch_idx):
        x, ypred, raw_target = self._forward_and_extract(batch)
        task_loss    = self.loss.loss(ypred, raw_target)
        distill_loss = torch.tensor(0.0, device=ypred.device)
        if self._soft_reg is not None:
            groups     = x.get("groups")
            _t1 = x.get("decoder_time_idx")
            _t2 = x.get("decoder_time_idx_start")
            target_idx = _t1 if _t1 is not None else _t2
            if target_idx is None:
                enc_idx = x.get("encoder_time_idx")
                enc_len = x.get("encoder_lengths")
                if enc_idx is not None and enc_len is not None:
                    first_dec  = enc_idx[torch.arange(enc_idx.shape[0]), enc_len.long() - 1] + 1
                    target_idx = first_dec.unsqueeze(1).expand(-1, self.hparams.get("max_prediction_length", 24))
            if batch_idx == 0:
                print(f"  [KD DEBUG] batch_idx=0 | groups: {groups is not None} | "
                      f"target_idx: {target_idx is not None}"
                      + (f" shape={target_idx.shape}" if target_idx is not None else ""))
            if groups is not None and target_idx is not None:
                station_names  = ([self._group_decode.get(int(g[0]), str(int(g[0]))) for g in groups]
                                  if self._group_decode else [str(int(g[0])) for g in groups])
                first_pred_idx = [int(target_idx[i, 0]) for i in range(target_idx.shape[0])]
                soft_t, valid  = self._soft_reg.lookup_batch(station_names, first_pred_idx, ypred.device)
                if batch_idx == 0:
                    print(f"  [KD DEBUG] registry hit rate on batch 0: "
                          f"{valid.float().mean().item()*100:.1f}% ({valid.sum()}/{len(valid)})")
                if valid.any():
                    distill_loss = F.mse_loss(ypred[valid], soft_t[valid])
        alpha      = getattr(self, "_kd_alpha", KD_ALPHA)
        total_loss = alpha * task_loss + (1.0 - alpha) * distill_loss
        self.log("train_task_loss",    task_loss,    on_step=True,  on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_distill_loss", distill_loss, on_step=True,  on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train_loss",         total_loss,   on_step=True,  on_epoch=True, prog_bar=True,  sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, ypred, raw_target = self._forward_and_extract(batch)
        loss_val = self.loss.loss(ypred, raw_target)
        self.log("val_loss", loss_val, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss_val

    def training_epoch_end(self, outputs):
        if not outputs:
            return
        losses = [o[0] if isinstance(o, (tuple, list)) else o for o in outputs]
        losses = [v.detach().float() for v in losses if isinstance(v, torch.Tensor)]
        if losses:
            self.log("train_loss_epoch", torch.stack(losses).mean(),
                     on_epoch=True, prog_bar=False, sync_dist=True)

    def validation_epoch_end(self, outputs):
        if not outputs:
            return
        losses = [o[0] if isinstance(o, (tuple, list)) else o for o in outputs]
        losses = [v.detach().float() for v in losses if isinstance(v, torch.Tensor)]
        if losses:
            self.log("val_loss", torch.stack(losses).mean(),
                     on_epoch=True, prog_bar=True, sync_dist=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────────────────────
class FlushSafeTensorBoardLogger(TensorBoardLogger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        try:
            super().save()
        except (FileNotFoundError, OSError) as exc:
            print(f"[WARN] TensorBoardLogger.save() caught {type(exc).__name__}: {exc}. Continuing.")
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
class EpochMetricsCallback(pl.Callback):
    def __init__(self):
        self.train_losses: List[float] = []
        self.val_losses:   List[float] = []

    def on_train_epoch_end(self, trainer, pl_module):
        logged = trainer.callback_metrics
        for key in ("train_loss_epoch", "train_loss", "loss"):
            v = logged.get(key)
            if v is not None:
                self.train_losses.append(float(v)); break

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        logged = trainer.callback_metrics
        for key in ("val_loss", "val_QuantileLoss", "val_loss_epoch", "val/loss"):
            v = logged.get(key)
            if v is not None:
                if key != "val_loss":
                    trainer.callback_metrics["val_loss"] = v
                self.val_losses.append(float(v)); break

    def aligned_losses(self):
        n = min(len(self.train_losses), len(self.val_losses))
        return self.train_losses[:n], self.val_losses[:n]


class LinearWarmupCallback(pl.Callback):
    def __init__(self, warmup_epochs: int, base_lr: float):
        assert warmup_epochs >= 2
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        if epoch < self.warmup_epochs:
            factor = (epoch + 1) / (self.warmup_epochs + 1)
            opt = pl_module.optimizers()
            if isinstance(opt, list): opt = opt[0]
            for pg in opt.param_groups: pg["lr"] = self.base_lr * factor

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch != self.warmup_epochs - 1: return
        opt = pl_module.optimizers()
        if isinstance(opt, list): opt = opt[0]
        for pg in opt.param_groups: pg["lr"] = self.base_lr
        for sched_cfg in trainer.lr_scheduler_configs:
            sched = getattr(sched_cfg, "scheduler", sched_cfg)
            if hasattr(sched, "num_bad_epochs"): sched.num_bad_epochs = 0
            if hasattr(sched, "_last_lr"):        sched._last_lr = [self.base_lr]


class WarmupAwareEarlyStopping(EarlyStopping):
    def __init__(self, warmup_epochs: int, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking or trainer.current_epoch < self.warmup_epochs:
            return
        super().on_validation_end(trainer, pl_module)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_preprocessing_artifacts(scaler_path, encoders_path):
    with open(scaler_path, "rb") as f:
        bundle = pickle.load(f)
    fee_scaler  = bundle["fee_scaler"]
    lag_scaler  = bundle["lag_scaler"]
    scaler_cols = list(bundle["columns"])
    if scaler_cols != SCALED_FEATURE_COLS:
        missing = [c for c in SCALED_FEATURE_COLS if c not in scaler_cols]
        extra   = [c for c in scaler_cols if c not in SCALED_FEATURE_COLS]
        raise ValueError(f"feature_scaler.pkl columns mismatch. Missing: {missing}  Extra: {extra}.")
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return fee_scaler, lag_scaler, encoders


def validate_encoders(encoders, df, context=""):
    for col, le in encoders.items():
        if col not in df.columns or col not in STATIC_CAT_COLS:
            continue
        nan_count = df[col].isna().sum()
        if nan_count:
            raise ValueError(f"[{context}] '{col}' has {nan_count} NaN(s).")
        unseen = set(df[col].astype(str).dropna().unique()) - set(pd.Series(le.classes_).astype(str).tolist())
        if unseen:
            raise ValueError(f"[{context}] '{col}' has unseen values: {sorted(unseen)[:10]}")


def load_dataframes(train_path, val_path):
    train_df = pd.read_csv(train_path, parse_dates=["Timestamp"])
    val_df   = pd.read_csv(val_path,   parse_dates=["Timestamp"])
    for df in (train_df, val_df):
        for col in STATIC_CAT_COLS + KNOWN_CAT_COLS:
            if col in df.columns:
                df[col] = df[col].astype(str)
        df["Time_Index"] = df["Time_Index"].astype(int)
    for df in (train_df, val_df):
        if "Is_Weekend" not in df.columns:
            df["Is_Weekend"] = (pd.to_datetime(df["Timestamp"]).dt.dayofweek >= 5).astype(int).astype(str)
        else:
            df["Is_Weekend"] = df["Is_Weekend"].astype(str)
    _required_precomputed = ["IsCharging_Lag1h_f", "IsCharging_Lag24h_f", "Was_Idle_LastHour"]
    for _df_name, _df in (("train_df", train_df), ("val_df", val_df)):
        _missing = [c for c in _required_precomputed if c not in _df.columns]
        if _missing:
            raise ValueError(f"[BUG-TRAIN-1 FIX] {_df_name} missing precomputed columns: {_missing}.")
    for _df in (train_df, val_df):
        for _col in _required_precomputed:
            _df[_col] = _df[_col].astype(float)
    return train_df, val_df


def build_combined_df(train_df, val_df):
    train_df = train_df.copy(); train_df["_src"] = 0
    val_df   = val_df.copy();   val_df["_src"]   = 1
    combined = (
        pd.concat([train_df, val_df], ignore_index=True)
        .sort_values(["Location_Information", "Time_Index", "_src"])
        .drop_duplicates(subset=["Location_Information", "Time_Index"], keep="last")
        .drop(columns=["_src"])
        .sort_values(["Location_Information", "Time_Index"])
        .reset_index(drop=True)
    )
    return combined


def make_target_normalizer():
    return GroupNormalizer(transformation=None, center=False, scale_by_group=False)


def build_timeseries_datasets(train_df, val_df, cfg, normalizer):
    known_reals = [c for c in cfg["time_varying_known_reals"] if c != cfg["time_idx"]]
    _new_unknown    = ["Lagged_Energy_24h", "Rolling_Mean_Energy_168h",
                       "IsCharging_Lag1h_f", "IsCharging_Lag24h_f", "Was_Idle_LastHour"]
    base_unknown      = list(cfg["time_varying_unknown_reals"])
    augmented_unknown = base_unknown + [c for c in _new_unknown
                                        if c in train_df.columns and c not in base_unknown]
    _missing = [c for c in _new_unknown if c not in train_df.columns]
    if _missing:
        print(f"  [WARN] Augmented unknown_reals not found: {_missing}")
    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx                        = cfg["time_idx"],
        target                          = cfg["target"],
        group_ids                       = cfg["group_ids"],
        max_encoder_length              = cfg["max_encoder_length"],
        max_prediction_length           = cfg["max_prediction_length"],
        min_encoder_length              = cfg["max_encoder_length"] * 3 // 4,
        static_categoricals             = cfg["static_categoricals"],
        static_reals                    = cfg["static_reals"],
        time_varying_known_categoricals = cfg["time_varying_known_categoricals"],
        time_varying_known_reals        = known_reals,
        time_varying_unknown_reals      = augmented_unknown,
        target_normalizer               = normalizer,
        add_relative_time_idx           = True,
        add_target_scales               = False,
        add_encoder_length              = True,
        allow_missing_timesteps         = False,
    )
    val_loss_start_str = cfg["preprocessing"]["val_loss_start"]
    mask = pd.to_datetime(val_df["Timestamp"]) >= pd.Timestamp(val_loss_start_str)
    if not mask.any():
        raise ValueError(f"No val rows with Timestamp >= '{val_loss_start_str}'.")
    val_loss_start_idx = int(val_df.loc[mask, "Time_Index"].min())
    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df,
        predict=False, stop_randomization=True,
        min_prediction_idx=val_loss_start_idx,
    )
    return train_dataset, val_dataset


def make_dataloaders(train_dataset, val_dataset):
    persistent = NUM_WORKERS > 0
    train_dl = train_dataset.to_dataloader(train=True,  batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS, persistent_workers=persistent)
    val_dl   = val_dataset.to_dataloader(  train=False, batch_size=BATCH_SIZE * 2,
                                           num_workers=NUM_WORKERS, persistent_workers=persistent)
    return train_dl, val_dl


def filter_vocab(train_df, val_df, encoders):
    """Filter rows to known station/district vocab (required for student pipeline)."""
    known_stations = set(encoders["Location_Information"].classes_)
    extra = set(train_df["Location_Information"].unique()) - known_stations
    if extra:
        print(f"  [VOCAB-GUARD] Removing {len(extra)} unknown station(s): {sorted(extra)}")
        train_df = train_df[train_df["Location_Information"].isin(known_stations)].copy()
        val_df   = val_df[val_df["Location_Information"].isin(known_stations)].copy()
    if "District_Name" in encoders:
        known_districts = set(encoders["District_Name"].classes_)
        extra_train = set(train_df["District_Name"].unique()) - known_districts
        if extra_train:
            train_df = train_df[train_df["District_Name"].isin(known_districts)].copy()
        extra_val = set(val_df["District_Name"].unique()) - known_districts
        if extra_val:
            val_df = val_df[val_df["District_Name"].isin(known_districts)].copy()
    return train_df, val_df

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────────────────────────────────────
def build_teacher(train_dataset):
    return RawLossTFT.from_dataset(
        train_dataset,
        hidden_size              = 128,
        hidden_continuous_size   = 64,
        attention_head_size      = 4,
        dropout                  = 0.10,
        loss                     = SparseQuantileLoss(quantiles=QUANTILES, nonzero_weight=5.0,
                                                      overpredict_penalty=2.5),
        optimizer                = "adamw",
        weight_decay             = 1e-4,
        learning_rate            = 3e-4,
        reduce_on_plateau_patience = 3,
        log_interval             = 50,
        log_val_interval         = 1,
    )


def build_student_baseline(train_dataset):
    return StudentBaseline.from_dataset(
        train_dataset,
        hidden_size               = 4,
        hidden_continuous_size    = 2,
        attention_head_size       = 1,
        dropout                   = 0.30,
        loss = SparseQuantileLoss(quantiles=QUANTILES, nonzero_weight=5.0, overpredict_penalty=2.5),
        optimizer                 = "adamw",
        weight_decay              = STUDENT_WEIGHT_DECAY,
        learning_rate             = STUDENT_BASE_LR,
        reduce_on_plateau_patience= 1,
        log_interval              = 50,
        log_val_interval          = 1,
    )


def build_student_kd(train_dataset):
    return StudentKDTFT.from_dataset(
        train_dataset,
        hidden_size               = 4,
        hidden_continuous_size    = 2,
        attention_head_size       = 1,
        dropout                   = 0.30,
        loss = SparseQuantileLoss(quantiles=QUANTILES, nonzero_weight=5.0, overpredict_penalty=2.5),
        optimizer                 = "adamw",
        weight_decay              = STUDENT_WEIGHT_DECAY,
        learning_rate             = STUDENT_BASE_LR,
        reduce_on_plateau_patience= 1,
        log_interval              = 50,
        log_val_interval          = 1,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _unwrap_batch(b):
    if isinstance(b, dict):
        b = b.get("prediction", b.get("output", next(iter(b.values()))))
    elif hasattr(b, "output"):
        b = b.output
    for _ in range(2):
        if isinstance(b, (tuple, list)): b = b[0]
        else: break
    if not isinstance(b, torch.Tensor):
        raise TypeError(f"_unwrap_batch: cannot extract tensor from {type(b)}.")
    return b


def run_predict(model, dataloader, supports_trainer_kwargs):
    kwargs = {"trainer_kwargs": PREDICT_TRAINER_KWARGS} if supports_trainer_kwargs else {}
    with torch.no_grad():
        raw = model.predict(dataloader, mode="quantiles", **kwargs)
    if isinstance(raw, (list, tuple)):
        return torch.cat([_unwrap_batch(b) for b in raw], dim=0)
    return _unwrap_batch(raw)


def extract_actuals_from_index(index_df, source_df, target_col, horizon):
    lookup  = source_df.set_index(["Location_Information", "Time_Index"])[target_col]
    actuals = []
    for _, row in index_df.iterrows():
        station, start = row["Location_Information"], int(row["Time_Index"])
        actuals.append([float(lookup.get((station, start + h), np.nan)) for h in range(horizon)])
    result = np.array(actuals, dtype=np.float32)
    nan_count = np.isnan(result).sum()
    if nan_count:
        print(f"  [WARN] extract_actuals_from_index: {nan_count} NaN(s) → 0.0")
        result = np.nan_to_num(result, nan=0.0)
    return result


def get_prediction_index(dataset):
    return (dataset.decoded_index
            .rename(columns={"time_idx_first_prediction": "Time_Index"})
            .reset_index(drop=True))


def supports_trainer_kwarg(model):
    import inspect
    return "trainer_kwargs" in inspect.signature(model.predict).parameters


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_business_metrics(pred_log, actuals_log):
    pred_kwh    = np.expm1(pred_log)
    actuals_kwh = np.expm1(actuals_log)
    p_flat, a_flat = pred_kwh.flatten(), actuals_kwh.flatten()
    wmape        = np.sum(np.abs(a_flat - p_flat)) / (np.sum(np.abs(a_flat)) + EPS) * 100.0
    active_mask  = a_flat > np.expm1(ZERO_THRESH)
    wmape_active = (np.sum(np.abs(a_flat[active_mask] - p_flat[active_mask])) /
                    (np.sum(a_flat[active_mask]) + EPS) * 100.0
                    if active_mask.sum() >= 1 else float("nan"))
    mape_active  = (float(np.mean(np.abs(a_flat[active_mask] - p_flat[active_mask]) /
                                  (a_flat[active_mask] + EPS)) * 100.0)
                    if active_mask.sum() >= 1 else float("nan"))
    mape_all     = float(np.mean(np.abs(a_flat - p_flat) / (a_flat + EPS)) * 100.0)
    r2_per_h = []
    for h in range(pred_kwh.shape[1]):
        a_h = actuals_kwh[:, h]
        r2_per_h.append(float("nan") if np.var(a_h) < 1e-10 else float(r2_score(a_h, pred_kwh[:, h])))
    r2_mean = float(np.nanmean(r2_per_h)) if not all(math.isnan(v) for v in r2_per_h) else 0.0
    nz_mask = actuals_kwh.sum(axis=1) > 0
    if nz_mask.sum() >= 2:
        r2_nz_per_h = []
        for h in range(pred_kwh.shape[1]):
            a_h, p_h = actuals_kwh[nz_mask, h], pred_kwh[nz_mask, h]
            r2_nz_per_h.append(float("nan") if np.var(a_h) < 1e-10 else float(r2_score(a_h, p_h)))
        r2_nz_mean = float(np.nanmean(r2_nz_per_h))
    else:
        r2_nz_per_h = [float("nan")] * pred_kwh.shape[1]
        r2_nz_mean  = float("nan")
    def _fmt(lst): return [None if math.isnan(v) else round(v, 4) for v in lst]
    return {
        "wmape_pct":              round(float(wmape), 4),
        "wmape_active_pct":       round(float(wmape_active), 4) if not math.isnan(wmape_active) else None,
        "mape_active_pct":        round(mape_active, 4) if not math.isnan(mape_active) else None,
        "mape_pct":               round(mape_all, 4),
        "mae_kwh":                round(float(mean_absolute_error(a_flat, p_flat)), 4),
        "rmse_kwh":               round(float(math.sqrt(mean_squared_error(a_flat, p_flat))), 4),
        "r2_score":               round(r2_mean, 4),
        "r2_score_nonzero":       round(r2_nz_mean, 4) if not math.isnan(r2_nz_mean) else None,
        "nonzero_seq_pct":        round(float(nz_mask.mean()) * 100, 2),
        "r2_per_horizon":         _fmt(r2_per_h),
        "r2_per_horizon_nonzero": _fmt(r2_nz_per_h),
    }


def compute_quantile_loss(quant_preds_log, actuals_log):
    loss_fn = QuantileLoss(quantiles=QUANTILES)
    with torch.no_grad():
        raw = loss_fn.loss(
            torch.tensor(quant_preds_log, dtype=torch.float32),
            torch.tensor(actuals_log,     dtype=torch.float32),
        )
    return float(raw.mean())


# ─────────────────────────────────────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────────────────────────────────────
def plot_training_curve(train_losses, val_losses, save_path, title="Training Curve"):
    if not train_losses or not val_losses:
        print(f"  [WARN] No epoch metrics — skipping {save_path.name}"); return
    epochs_x = list(range(1, len(train_losses) + 1))
    gap      = [v - t for t, v in zip(train_losses, val_losses)]
    best_ep  = int(np.argmin(val_losses)) + 1
    best_val = min(val_losses)
    fig, (ax_loss, ax_gap) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
    fig.patch.set_facecolor("white")
    ax_loss.plot(epochs_x, train_losses, color=PALETTE["train"], lw=2, label="Train loss")
    ax_loss.plot(epochs_x, val_losses,   color=PALETTE["val"],   lw=2, ls="--", label="Val loss")
    ax_loss.fill_between(epochs_x, train_losses, val_losses, color=PALETTE["val"], alpha=0.08)
    ax_loss.axvline(best_ep, color=PALETTE["val"], lw=1.2, ls=":", alpha=0.7)
    ax_loss.annotate(f"Best val {best_val:.4f}\n(epoch {best_ep})",
                     xy=(best_ep, best_val), fontsize=8.5, color=PALETTE["val"],
                     xytext=(best_ep + max(1, len(epochs_x)*0.05), best_val + (max(val_losses)-best_val)*0.15),
                     arrowprops=dict(arrowstyle="->", color=PALETTE["val"], lw=1))
    ax_loss.set_ylabel("Quantile Loss (log1p space)", fontsize=10)
    ax_loss.legend(fontsize=9, framealpha=0.9)
    ax_loss.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax_loss.yaxis.grid(True, color=PALETTE["grid"], lw=0.8)
    ax_loss.spines[["top","right"]].set_visible(False)
    ax_gap.bar(epochs_x, gap, color=PALETTE["gap"], alpha=0.7, width=0.7)
    ax_gap.axhline(0, color="#6B7280", lw=0.8)
    ax_gap.set_ylabel("Gap\n(val−train)", fontsize=8.5)
    ax_gap.set_xlabel("Epoch", fontsize=10)
    ax_gap.yaxis.grid(True, color=PALETTE["grid"], lw=0.8)
    ax_gap.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  training_curve.png → {save_path}")


def _pick_panels(quant_preds_log, actuals_log, station_arr, n_panels=4, random_seed=SEED):
    """Select n_panels stations: 2 best-R² + 2 random. Returns list of (stn, q_kwh, a_kwh, r2)."""
    from sklearn.metrics import r2_score as _r2

    all_stations    = list(np.unique(station_arr))
    rng             = np.random.default_rng(random_seed)
    station_scores  = {}
    for stn in all_stations:
        rows = np.where(station_arr == stn)[0]
        r2_list = []
        for r in rows:
            a = np.expm1(actuals_log[r])
            p = np.expm1(quant_preds_log[r, :, MEDIAN_IDX])
            if np.var(a) >= 1e-10:
                try:   r2_list.append(_r2(a, p))
                except Exception: pass
        if r2_list:
            station_scores[stn] = float(np.mean(r2_list))

    sorted_stns  = sorted(station_scores.items(), key=lambda x: x[1], reverse=True)
    best_stns    = [s for s, _ in sorted_stns[:2]]
    remaining    = [s for s in all_stations if s not in best_stns]
    random_stns  = rng.choice(remaining, size=min(2, len(remaining)), replace=False).tolist()
    chosen       = best_stns + random_stns

    panel_data = []
    for stn in chosen:
        rows = np.where(station_arr == stn)[0]
        if len(rows) == 0:
            continue
        best_r2, best_row = -np.inf, rows[0]
        for r in rows:
            a = np.expm1(actuals_log[r])
            p = np.expm1(quant_preds_log[r, :, MEDIAN_IDX])
            if np.var(a) < 1e-10 or a.sum() <= np.expm1(ZERO_THRESH):
                continue
            try:
                rv = float(_r2(a, p))
            except Exception:
                rv = -np.inf
            if rv > best_r2:
                best_r2, best_row = rv, r
        if best_r2 == -np.inf:
            best_row = rows[0]; best_r2 = float("nan")
        panel_data.append((
            stn,
            np.expm1(quant_preds_log[best_row]),
            np.expm1(actuals_log[best_row]),
            round(best_r2, 3) if not math.isnan(best_r2) else float("nan"),
        ))
    return panel_data


def plot_predictions(quant_preds_log, actuals_log, station_arr, save_path,
                     title="24h Forecast vs Actual", n_panels=4, random_seed=SEED,
                     colors=None):
    """Individual 4-panel prediction plot for one model."""
    if colors is None:
        colors = {"pred": "#E07B2A", "band": "#F0A85C", "actual": "#5B3A29"}
    C_PRED, C_BAND, C_ACTUAL = colors["pred"], colors["band"], colors["actual"]
    C_GRID   = "#F0EDE8"
    horizon  = np.arange(1, 25)

    panel_data = _pick_panels(quant_preds_log, actuals_log, station_arr,
                               n_panels=n_panels, random_seed=random_seed)
    if not panel_data:
        print(f"  [WARN] No non-idle panel data — skipping {save_path.name}"); return

    ncols = 2
    nrows = math.ceil(len(panel_data) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 4.5 * nrows))
    axes = np.array(axes).flatten()
    fig.patch.set_facecolor("white")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)

    for ax, (label, q_kwh, a_kwh, r2_val) in zip(axes, panel_data):
        ax.fill_between(horizon, q_kwh[:, Q_IDX[0.10]], q_kwh[:, Q_IDX[0.90]],
                        color=C_BAND, alpha=0.30, linewidth=0, label="P10–P90")
        ax.fill_between(horizon, q_kwh[:, Q_IDX[0.25]], q_kwh[:, Q_IDX[0.75]],
                        color=C_BAND, alpha=0.55, linewidth=0, label="P25–P75")
        ax.plot(horizon, q_kwh[:, MEDIAN_IDX], color=C_PRED, lw=2.2, label="Prediction (P50)", zorder=4)
        ax.plot(horizon, a_kwh, color=C_ACTUAL, lw=1.5, ls="--",
                marker="o", markersize=3.5, label="Actual", zorder=5)
        t_str  = (str(label)[:24] + "…") if len(str(label)) > 26 else str(label)
        r2_str = f"{r2_val:.3f}" if not math.isnan(r2_val) else "N/A"
        ax.set_title(f"{t_str}   R²={r2_str}", fontsize=9, pad=5, fontweight="bold")
        ax.set_xlabel("Forecast horizon (h)", fontsize=8.5)
        ax.set_ylabel("Energy (kWh)", fontsize=8.5)
        ax.set_xlim(0.5, 24.5)
        ax.yaxis.grid(True, color=C_GRID, lw=0.9); ax.xaxis.grid(True, color=C_GRID, lw=0.9)
        ax.set_facecolor("white")
        ax.spines[["top","right"]].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC"); ax.spines["bottom"].set_color("#CCCCCC")
        ax.tick_params(labelsize=8)

    for ax in axes[len(panel_data):]:
        ax.set_visible(False)
    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  predictions.png → {save_path}")


def plot_pred_vs_actual_scatter(actual_flat, pred_flat, save_path, model_label, r2_val,
                                color="#2563EB"):
    """Individual scatter plot: Predicted vs Actual (kWh). Saves .png and .pdf."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10, color=color, rasterized=True)
    lims = [min(actual_flat.min(), pred_flat.min()), max(actual_flat.max(), pred_flat.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual Energy (kWh)", fontsize=11)
    ax.set_ylabel("Predicted Energy (kWh)", fontsize=11)
    ax.set_title(f"Prediction vs Actual — {model_label}", fontsize=12, fontweight="bold")
    ax.text(0.05, 0.95, f"$R^2$ = {r2_val:.3f}", transform=ax.transAxes,
            verticalalignment="top", fontsize=11)
    ax.grid(True, linewidth=0.5, alpha=0.5, color=PALETTE["grid"])
    ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  pred_vs_actual → {save_path.parent}/pred_vs_actual.png / .pdf")


def _station_r2_map(quant_preds_log, actuals_log, station_arr):
    """Return {station: mean_r2} computed over all prediction windows for that station."""
    from sklearn.metrics import r2_score as _r2
    scores = {}
    for stn in np.unique(station_arr):
        rows = np.where(station_arr == stn)[0]
        vals = []
        for r in rows:
            a = np.expm1(actuals_log[r])
            p = np.expm1(quant_preds_log[r, :, MEDIAN_IDX])
            if np.var(a) < 1e-10 or a.sum() <= np.expm1(ZERO_THRESH):
                continue
            try:
                vals.append(float(_r2(a, p)))
            except Exception:
                pass
        if vals:
            scores[stn] = float(np.mean(vals))
    return scores


def _pick_representative_stations(results_dict, n_stations=2):
    """
    Select stations whose per-station R² (averaged across all prediction windows)
    is closest to each model's own overall R² score.

    Strategy:
      1. For every model, compute per-station mean R² and the model's global R².
      2. Find stations present in ALL models.
      3. For each model, rank shared stations by |station_r2 - model_global_r2|.
      4. Pick the n_stations stations that minimise the *maximum* rank across all
         models — i.e. stations that are representative for every model at once.

    Returns a list of station names (length ≤ n_stations).
    """
    from sklearn.metrics import r2_score as _r2

    model_station_r2   = {}   # {model: {stn: r2}}
    model_global_r2    = {}   # {model: float}

    for model_label, data in results_dict.items():
        global_r2 = data.get("metrics", {}).get("r2_score")
        if global_r2 is None:
            # Fallback: compute from flattened arrays
            pred_flat   = np.expm1(data["quant_preds_log"][:, :, MEDIAN_IDX]).flatten()
            actual_flat = np.expm1(data["actuals_log"]).flatten()
            try:
                global_r2 = float(_r2(actual_flat, pred_flat))
            except Exception:
                global_r2 = 0.0
        model_global_r2[model_label]  = global_r2
        model_station_r2[model_label] = _station_r2_map(
            data["quant_preds_log"], data["actuals_log"], data["station_arr"]
        )

    # Stations present in all models
    shared = set.intersection(*[set(sr.keys()) for sr in model_station_r2.values()])
    if not shared:
        # Fall back: union, missing entries get worst score
        shared = set.union(*[set(sr.keys()) for sr in model_station_r2.values()])

    shared = list(shared)
    if len(shared) <= n_stations:
        return shared

    # For each candidate station, compute worst-case |station_r2 - global_r2|
    # across models (so it's representative for every model, not just one).
    def worst_gap(stn):
        gaps = []
        for model_label, sr in model_station_r2.items():
            sr2 = sr.get(stn)
            if sr2 is None:
                gaps.append(1.0)   # penalise missing station heavily
            else:
                gaps.append(abs(sr2 - model_global_r2[model_label]))
        return max(gaps)

    shared_sorted = sorted(shared, key=worst_gap)
    chosen = shared_sorted[:n_stations]

    # Log the choices for transparency
    print(f"\n  [Combined plot] Representative stations (closest to model R²):")
    for stn in chosen:
        parts = []
        for ml in results_dict:
            sr2 = model_station_r2[ml].get(stn)
            gr2 = model_global_r2[ml]
            if sr2 is not None:
                parts.append(f"{ml}: stn_R²={sr2:.3f} vs model_R²={gr2:.3f} (Δ={abs(sr2-gr2):.3f})")
        print(f"    {stn}: {' | '.join(parts)}")

    return chosen


def _median_r2_window(qpreds_log, actuals_log, rows):
    """
    Among all prediction windows (rows) for a station, return the row whose
    single-window R² is closest to the station's own mean R² — i.e. the most
    typical window, not the best or worst.
    """
    from sklearn.metrics import r2_score as _r2
    window_r2 = []
    valid_rows = []
    for r in rows:
        a = np.expm1(actuals_log[r])
        p = np.expm1(qpreds_log[r, :, MEDIAN_IDX])
        if np.var(a) < 1e-10 or a.sum() <= np.expm1(ZERO_THRESH):
            continue
        try:
            window_r2.append(float(_r2(a, p)))
            valid_rows.append(r)
        except Exception:
            pass
    if not valid_rows:
        return rows[0], float("nan")
    mean_r2  = float(np.mean(window_r2))
    best_idx = int(np.argmin(np.abs(np.array(window_r2) - mean_r2)))
    return valid_rows[best_idx], window_r2[best_idx]


def plot_combined_3model(results_dict, save_path):
    """
    Combined 3-model prediction vs actual panel.

    results_dict = {
        "Teacher":          {"quant_preds_log": ..., "actuals_log": ...,
                             "station_arr": ..., "metrics": {...}},
        "Student Baseline": {...},
        "Student KD":       {...},
    }

    Layout  : n_models rows × 2 columns.
    Stations: chosen so that each station's per-station R² is as close as
              possible to the model's *own* overall R² — making cross-model
              differences plainly visible rather than hiding them behind
              cherry-picked best windows.
    Window  : for each (model, station) the prediction window whose single-
              window R² is closest to that station's mean R² is shown (again,
              the typical window, not the best one).
    """
    if not results_dict:
        print("  [WARN] No results for combined panel — skipping."); return

    horizon = np.arange(1, 25)

    # ── Select representative stations ───────────────────────────────────────
    shared_stations = _pick_representative_stations(results_dict, n_stations=2)
    if not shared_stations:
        print("  [WARN] Could not find shared stations — skipping combined panel."); return

    n_models = len(results_dict)
    n_cols   = len(shared_stations)
    fig, axes = plt.subplots(n_models, n_cols,
                             figsize=(7 * n_cols, 4.5 * n_models),
                             sharey=False)
    # Normalise axes to always be 2-D (n_models × n_cols)
    axes = np.array(axes)
    if axes.ndim == 0:
        axes = axes.reshape(1, 1)
    elif axes.ndim == 1:
        axes = axes.reshape(1, -1) if n_models == 1 else axes.reshape(-1, 1)

    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Prediction vs Actual — Teacher vs Student Baseline vs Student KD\n"
        "(Jiaxing Dataset · Validation Set · stations representative of each model's R²)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    for row_idx, (model_label, data) in enumerate(results_dict.items()):
        qpreds_log  = data["quant_preds_log"]
        actuals_log = data["actuals_log"]
        station_arr = data["station_arr"]
        global_r2   = data.get("metrics", {}).get("r2_score", float("nan"))
        colors      = MODEL_COLORS.get(model_label, {"pred": "#333333", "band": "#888888", "actual": "#000000"})

        for col_idx, stn in enumerate(shared_stations):
            ax   = axes[row_idx, col_idx]
            rows = np.where(station_arr == stn)[0]

            if len(rows) == 0:
                ax.text(0.5, 0.5, f"{stn}\n(no data)", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9)
                ax.set_visible(True)
                continue

            # Typical (median-R²) window rather than best window
            chosen_row, window_r2 = _median_r2_window(qpreds_log, actuals_log, rows)

            q_kwh = np.expm1(qpreds_log[chosen_row])
            a_kwh = np.expm1(actuals_log[chosen_row])

            ax.fill_between(horizon, q_kwh[:, Q_IDX[0.10]], q_kwh[:, Q_IDX[0.90]],
                            color=colors["band"], alpha=0.25, linewidth=0, label="P10–P90")
            ax.fill_between(horizon, q_kwh[:, Q_IDX[0.25]], q_kwh[:, Q_IDX[0.75]],
                            color=colors["band"], alpha=0.50, linewidth=0, label="P25–P75")
            ax.plot(horizon, q_kwh[:, MEDIAN_IDX], color=colors["pred"],
                    lw=2.2, label="Pred (P50)", zorder=4)
            ax.plot(horizon, a_kwh, color=colors["actual"], lw=1.5, ls="--",
                    marker="o", markersize=3, label="Actual", zorder=5)

            t_str     = (str(stn)[:20] + "…") if len(str(stn)) > 22 else str(stn)
            r2_str    = f"{window_r2:.3f}" if not math.isnan(window_r2) else "N/A"
            gr2_str   = f"{global_r2:.3f}" if not math.isnan(global_r2) else "N/A"

            # Left column carries the model row label
            if col_idx == 0:
                ax.set_ylabel(f"[{model_label}]\nEnergy (kWh)", fontsize=8.5)
            else:
                ax.set_ylabel("Energy (kWh)", fontsize=8.5)

            ax.set_title(
                f"{t_str}   window R²={r2_str}  (model R²={gr2_str})",
                fontsize=8.5, pad=4, fontweight="bold",
            )
            ax.set_xlabel("Forecast horizon (h)", fontsize=8.5)
            ax.set_xlim(0.5, 24.5)
            ax.yaxis.grid(True, color="#F0EDE8", lw=0.9)
            ax.xaxis.grid(True, color="#F0EDE8", lw=0.9)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["left"].set_color("#CCCCCC")
            ax.spines["bottom"].set_color("#CCCCCC")
            ax.tick_params(labelsize=7.5)

    # ── Legends ──────────────────────────────────────────────────────────────
    handles, labels_leg = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="lower center", ncol=4, fontsize=9,
               framealpha=0.9, bbox_to_anchor=(0.5, -0.02))
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=MODEL_COLORS[m]["pred"], label=m) for m in results_dict]
    fig.legend(handles=legend_patches, loc="upper right", fontsize=8.5,
               framealpha=0.9, title="Model", title_fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  ✅ Combined 3-model panel → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# TRAINER FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def make_trainer(output_dir, checkpoint_filename, gradient_clip=0.5, max_epochs=50,
                 early_stop_patience=5, warmup_epochs=2, base_lr=3e-4):
    """Build a standard PL Trainer with shared callbacks."""
    ckpt_dir     = output_dir / "checkpoints"
    tb_log_dir   = str(output_dir / "logs")
    csv_log_dir  = str(output_dir / "logs_csv")
    for d in (ckpt_dir, Path(tb_log_dir), Path(csv_log_dir)):
        d.mkdir(parents=True, exist_ok=True)

    _n_gpus   = torch.cuda.device_count()
    _strategy = "ddp_find_unused_parameters_false" if _n_gpus > 1 else "auto"
    print(f"  GPUs detected: {_n_gpus} → strategy='{_strategy}'")

    epoch_metrics_cb = EpochMetricsCallback()
    callbacks = [
        LinearWarmupCallback(warmup_epochs=warmup_epochs, base_lr=base_lr),
        WarmupAwareEarlyStopping(warmup_epochs=warmup_epochs + 1, monitor="val_loss",
                                  mode="min", patience=early_stop_patience,
                                  verbose=True, strict=False),
        ModelCheckpoint(dirpath=str(ckpt_dir), filename=checkpoint_filename,
                        monitor="val_loss", mode="min", save_top_k=1,
                        save_weights_only=True, verbose=True, auto_insert_metric_name=False),
        LearningRateMonitor(logging_interval="epoch"),
        epoch_metrics_cb,
    ]
    trainer = pl.Trainer(
        max_epochs          = max_epochs,
        accelerator         = "auto",
        devices             = "auto",
        strategy            = _strategy,
        sync_batchnorm      = True,
        gradient_clip_val   = gradient_clip,
        logger              = [FlushSafeTensorBoardLogger(save_dir=tb_log_dir, name=""),
                               CSVLogger(save_dir=csv_log_dir, name="")],
        callbacks           = callbacks,
        enable_progress_bar = True,
        enable_model_summary= True,
        log_every_n_steps   = 50,
        deterministic       = False,
    )
    return trainer, epoch_metrics_cb


# ─────────────────────────────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  PIPELINE A — TEACHER TFT                                                │
# └──────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────
def run_teacher(cfg_in, train_df, val_df, encoders):
    """Train Teacher TFT. Returns dict with quant_preds_log, actuals_log, station_arr."""
    print("\n" + "="*64 + "\n  TEACHER TFT PIPELINE\n" + "="*64)
    OUTPUT_DIR = TEACHER_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = {**cfg_in}
    assert cfg["max_encoder_length"] == 336, "Stale config: max_encoder_length must be 336."

    if set(cfg["static_categoricals"]) != set(STATIC_CAT_COLS):
        cfg["static_categoricals"] = STATIC_CAT_COLS
    unexpected = [c for c in cfg.get("time_varying_known_reals", [])
                  if c != cfg["time_idx"] and c not in KNOWN_REAL_ALLOWLIST]
    if unexpected:
        raise ValueError(f"time_varying_known_reals has unexpected columns: {unexpected}")

    normalizer                 = make_target_normalizer()
    train_dataset, val_dataset = build_timeseries_datasets(train_df, val_df, cfg, normalizer)
    train_dl, val_dl           = make_dataloaders(train_dataset, val_dataset)

    teacher  = build_teacher(train_dataset)
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher params : {n_params:,}")

    trainer, epoch_metrics_cb = make_trainer(
        OUTPUT_DIR, checkpoint_filename="best_teacher-{epoch:02d}",
        gradient_clip=0.7, max_epochs=50, early_stop_patience=6,
        warmup_epochs=WARMUP_EPOCHS, base_lr=3e-4,
    )
    trainer.fit(teacher, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        raise FileNotFoundError(f"No valid checkpoint at '{best_ckpt}'.")
    print(f"  Best checkpoint : {best_ckpt}")

    result = {}
    if trainer.is_global_zero:
        best_teacher = RawLossTFT.load_from_checkpoint(
            best_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
        best_teacher.eval()
        teacher.cpu(); del teacher; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        _stk = supports_trainer_kwarg(best_teacher)
        quant_preds_log = run_predict(best_teacher, val_dl, _stk).cpu().numpy()
        pred_log        = quant_preds_log[:, :, MEDIAN_IDX]
        val_index       = get_prediction_index(val_dataset)
        actuals_log     = extract_actuals_from_index(
            val_index, val_df, cfg["target"], cfg["max_prediction_length"])
        val_station_arr = val_index["Location_Information"].values

        biz     = compute_business_metrics(pred_log, actuals_log)
        ql_val  = compute_quantile_loss(quant_preds_log, actuals_log)
        akt_kwh = np.expm1(actuals_log)
        qpk     = np.expm1(quant_preds_log)
        p90_cov = float(np.mean(akt_kwh <= qpk[:, :, Q_IDX[0.90]]))
        mce     = float(np.mean([abs(np.mean(actuals_log <= quant_preds_log[:,:,i]) - q)
                                  for i, q in enumerate(QUANTILES)]))

        metrics = {
            **{k: v for k, v in biz.items() if k != "r2_per_horizon"},
            "r2_per_horizon":         biz["r2_per_horizon"],
            "quantile_loss_log1p":    round(ql_val, 6),
            "p90_coverage":           round(p90_cov, 4),
            "mean_calibration_error": round(mce, 6),
            "quantiles":              QUANTILES,
            "best_val_loss":          round(float(trainer.checkpoint_callback.best_model_score), 6)
                                      if trainer.checkpoint_callback.best_model_score else None,
            "best_ckpt":              best_ckpt,
        }
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")
        _r2h = biz["r2_per_horizon"]
        _g = lambda lst, i: float("nan") if i >= len(lst) or lst[i] is None else lst[i]
        print(f"  WMAPE={biz['wmape_pct']:.2f}%  R²={biz['r2_score']:.4f}  "
              f"QL={ql_val:.6f}  P90cov={p90_cov:.4f}  MCE={mce:.6f}")
        print(f"  MAE={biz['mae_kwh']:.4f} kWh  RMSE={biz['rmse_kwh']:.4f} kWh")
        print(f"  h1={_g(_r2h,0):.3f}  h6={_g(_r2h,5):.3f}  h12={_g(_r2h,11):.3f}  h24={_g(_r2h,23):.3f}")

        # KD soft target extraction
        print("\n  Extracting KD soft targets (combined train+val)...")
        combined_df      = build_combined_df(train_df, val_df)
        combined_dataset = TimeSeriesDataSet.from_dataset(
            train_dataset, combined_df, predict=False, stop_randomization=True,
            min_encoder_length=cfg["max_encoder_length"],
        )
        combined_dl = combined_dataset.to_dataloader(
            train=False, batch_size=BATCH_SIZE * 2,
            num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
        soft_log = run_predict(best_teacher, combined_dl, _stk).cpu().numpy()
        soft_kwh = np.expm1(soft_log)
        comb_idx = get_prediction_index(combined_dataset)
        np.savez_compressed(
            SOFT_TARGETS_PATH,
            quantiles_log1p             = soft_log,
            quantiles_kwh               = soft_kwh,
            station_names               = comb_idx["Location_Information"].values,
            time_index                  = comb_idx["Time_Index"].values,
            quantile_levels             = np.array(QUANTILES),
            warmup_excluded_per_station = np.int64(cfg["max_encoder_length"]),
        )
        print(f"  Soft targets shape : {soft_log.shape}  → {SOFT_TARGETS_PATH}")
        best_teacher.cpu(); del best_teacher; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Figures
        tr_l, vl_l = epoch_metrics_cb.aligned_losses()
        plot_training_curve(tr_l, vl_l, OUTPUT_DIR / "training_curve.png",
                            title="Teacher TFT — Training Curve")
        plot_predictions(quant_preds_log, actuals_log, val_station_arr,
                         OUTPUT_DIR / "predictions.png",
                         title="Teacher TFT — 24h Forecast vs Actual - Jiaxing Dataset",
                         colors=MODEL_COLORS["Teacher"])
        pred_flat   = np.expm1(pred_log).flatten()
        actual_flat = np.expm1(actuals_log).flatten()
        r2_scatter  = float(r2_score(actual_flat, pred_flat))
        plot_pred_vs_actual_scatter(actual_flat, pred_flat,
                                    OUTPUT_DIR / "pred_vs_actual",
                                    model_label="Teacher TFT", r2_val=r2_scatter,
                                    color=MODEL_COLORS["Teacher"]["pred"])
        # Build per-sequence records with station label and R²
        from sklearn.metrics import r2_score as _r2

        rows = []
        for i, stn in enumerate(val_station_arr):
            a = np.expm1(actuals_log[i])          # shape (24,)
            p = np.expm1(pred_log[i])             # shape (24,)
            r2_val = float(_r2(a, p)) if np.var(a) >= 1e-10 else float("nan")
            for h in range(len(a)):
                rows.append({
                    "station":    stn,
                    "horizon_h":  h + 1,
                    "actual":     float(a[h]),
                    "prediction": float(p[h]),
                    "seq_r2":     round(r2_val, 4),   # same R² for all hours in this sequence
                })

        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "pred_vs_actual.csv", index=False)

        result = {
            "quant_preds_log": quant_preds_log,
            "actuals_log":     actuals_log,
            "station_arr":     val_station_arr,
            "metrics":         metrics,
        }
        print(f"\n  Teacher training complete. Outputs → {OUTPUT_DIR}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  PIPELINE B — STUDENT BASELINE                                           │
# └──────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────
def run_student_baseline(cfg_in, train_df, val_df, encoders):
    """Train Student Baseline TFT (no KD)."""
    print("\n" + "="*64 + "\n  STUDENT BASELINE PIPELINE\n" + "="*64)
    OUTPUT_DIR = BASELINE_OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = {**cfg_in}
    assert cfg["max_encoder_length"] == 168, "Student config: max_encoder_length must be 168."
    if set(cfg["static_categoricals"]) != set(STATIC_CAT_COLS):
        cfg["static_categoricals"] = STATIC_CAT_COLS
    unexpected = [c for c in cfg.get("time_varying_known_reals", [])
                  if c != cfg["time_idx"] and c not in KNOWN_REAL_ALLOWLIST]
    if unexpected:
        raise ValueError(f"time_varying_known_reals has unexpected columns: {unexpected}")

    train_df_s, val_df_s = filter_vocab(train_df.copy(), val_df.copy(), encoders)
    normalizer                 = make_target_normalizer()
    train_dataset, val_dataset = build_timeseries_datasets(train_df_s, val_df_s, cfg, normalizer)
    train_dl, val_dl           = make_dataloaders(train_dataset, val_dataset)

    student  = build_student_baseline(train_dataset)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student Baseline params : {n_params:,}")

    trainer, epoch_metrics_cb = make_trainer(
        OUTPUT_DIR, checkpoint_filename="best_student-{epoch:02d}",
        gradient_clip=0.5, max_epochs=50, early_stop_patience=5,
        warmup_epochs=WARMUP_EPOCHS, base_lr=STUDENT_BASE_LR,
    )
    trainer.fit(student, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        raise FileNotFoundError(f"No valid checkpoint at '{best_ckpt}'.")
    print(f"  Best checkpoint : {best_ckpt}")

    # Free training model on all ranks before rank-0 eval
    student.cpu(); del student; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    result = {}
    if trainer.is_global_zero:
        best_student = StudentBaseline.load_from_checkpoint(
            best_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
        best_student.eval()

        _stk = supports_trainer_kwarg(best_student)
        quant_preds_log = run_predict(best_student, val_dl, _stk).cpu().numpy()
        pred_log        = quant_preds_log[:, :, MEDIAN_IDX]
        val_index       = get_prediction_index(val_dataset)
        actuals_log     = extract_actuals_from_index(
            val_index, val_df_s, cfg["target"], cfg["max_prediction_length"])
        val_station_arr = val_index["Location_Information"].values

        biz     = compute_business_metrics(pred_log, actuals_log)
        ql_val  = compute_quantile_loss(quant_preds_log, actuals_log)
        akt_kwh = np.expm1(actuals_log)
        qpk     = np.expm1(quant_preds_log)
        p90_cov = float(np.mean(akt_kwh <= qpk[:, :, Q_IDX[0.90]]))
        mce     = float(np.mean([abs(np.mean(actuals_log <= quant_preds_log[:,:,i]) - q)
                                  for i, q in enumerate(QUANTILES)]))

        metrics = {
            **{k: v for k, v in biz.items() if k != "r2_per_horizon"},
            "r2_per_horizon":         biz["r2_per_horizon"],
            "quantile_loss_log1p":    round(ql_val, 6),
            "p90_coverage":           round(p90_cov, 4),
            "mean_calibration_error": round(mce, 6),
            "quantiles":              QUANTILES,
            "best_val_loss":          round(float(trainer.checkpoint_callback.best_model_score), 6)
                                      if trainer.checkpoint_callback.best_model_score else None,
            "best_ckpt":              best_ckpt,
            "mode":                   "baseline_no_kd",
        }
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")
        _r2h = biz["r2_per_horizon"]
        _g   = lambda lst, i: float("nan") if i >= len(lst) or lst[i] is None else lst[i]
        print(f"  WMAPE={biz['wmape_pct']:.2f}%  R²={biz['r2_score']:.4f}  "
              f"QL={ql_val:.6f}  P90cov={p90_cov:.4f}  MCE={mce:.6f}")
        print(f"  MAE={biz['mae_kwh']:.4f} kWh  RMSE={biz['rmse_kwh']:.4f} kWh")
        print(f"  h1={_g(_r2h,0):.3f}  h6={_g(_r2h,5):.3f}  h12={_g(_r2h,11):.3f}  h24={_g(_r2h,23):.3f}")

        best_student.cpu(); del best_student; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Figures
        tr_l, vl_l = epoch_metrics_cb.aligned_losses()
        plot_training_curve(tr_l, vl_l, OUTPUT_DIR / "training_curve.png",
                            title="Student TFT (Baseline) — Training Curve")
        plot_predictions(quant_preds_log, actuals_log, val_station_arr,
                         OUTPUT_DIR / "predictions.png",
                         title="Student TFT (Baseline) — 24h Forecast vs Actual - Jiaxing Dataset",
                         colors=MODEL_COLORS["Student Baseline"])
        pred_flat   = np.expm1(pred_log).flatten()
        actual_flat = np.expm1(actuals_log).flatten()
        r2_scatter  = float(r2_score(actual_flat, pred_flat))
        plot_pred_vs_actual_scatter(actual_flat, pred_flat,
                                    OUTPUT_DIR / "pred_vs_actual",
                                    model_label="Student TFT (Baseline)", r2_val=r2_scatter,
                                    color=MODEL_COLORS["Student Baseline"]["pred"])
        # Build per-sequence records with station label and R²
        from sklearn.metrics import r2_score as _r2

        rows = []
        for i, stn in enumerate(val_station_arr):
            a = np.expm1(actuals_log[i])          # shape (24,)
            p = np.expm1(pred_log[i])             # shape (24,)
            r2_val = float(_r2(a, p)) if np.var(a) >= 1e-10 else float("nan")
            for h in range(len(a)):
                rows.append({
                    "station":    stn,
                    "horizon_h":  h + 1,
                    "actual":     float(a[h]),
                    "prediction": float(p[h]),
                    "seq_r2":     round(r2_val, 4),   # same R² for all hours in this sequence
                })

        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "pred_vs_actual.csv", index=False)

        result = {
            "quant_preds_log": quant_preds_log,
            "actuals_log":     actuals_log,
            "station_arr":     val_station_arr,
            "metrics":         metrics,
        }
        print(f"\n  Student Baseline training complete. Outputs → {OUTPUT_DIR}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ┌──────────────────────────────────────────────────────────────────────────┐
# │  PIPELINE C — STUDENT KD                                                 │
# └──────────────────────────────────────────────────────────────────────────┘
# ─────────────────────────────────────────────────────────────────────────────
def run_student_kd(cfg_in, train_df, val_df, encoders, alpha: float = KD_ALPHA):
    """Train Student KD TFT. Requires SOFT_TARGETS_PATH to exist."""
    alpha_tag  = f"alpha{int(round(alpha * 100)):03d}"   # e.g. 0.70 → "alpha070"
    alpha_label = f"α={alpha:.2f}"
    print("\n" + "="*64 + f"\n  STUDENT KD PIPELINE  ({alpha_label})\n" + "="*64)
    OUTPUT_DIR = Path(f"models/student_kd_tft111_jiaxing1_{alpha_tag}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not SOFT_TARGETS_PATH.exists():
        raise FileNotFoundError(
            f"Soft targets not found at '{SOFT_TARGETS_PATH}'. "
            "Run the Teacher pipeline first (--models teacher kd or --models teacher)."
        )
    print(f"  Soft targets : {SOFT_TARGETS_PATH}")

    cfg = {**cfg_in}
    assert cfg["max_encoder_length"] == 168, "Student config: max_encoder_length must be 168."
    if set(cfg["static_categoricals"]) != set(STATIC_CAT_COLS):
        cfg["static_categoricals"] = STATIC_CAT_COLS
    unexpected = [c for c in cfg.get("time_varying_known_reals", [])
                  if c != cfg["time_idx"] and c not in KNOWN_REAL_ALLOWLIST]
    if unexpected:
        raise ValueError(f"time_varying_known_reals has unexpected columns: {unexpected}")

    train_df_s, val_df_s = filter_vocab(train_df.copy(), val_df.copy(), encoders)
    normalizer                 = make_target_normalizer()
    train_dataset, val_dataset = build_timeseries_datasets(train_df_s, val_df_s, cfg, normalizer)
    train_dl, val_dl           = make_dataloaders(train_dataset, val_dataset)

    student  = build_student_kd(train_dataset)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"  Student KD params : {n_params:,}")

    # Attach soft target registry and group-decode map before training
    soft_reg    = SoftTargetRegistry(SOFT_TARGETS_PATH)
    group_vocab = train_dataset.categorical_encoders[cfg["group_ids"][0]]
    if hasattr(group_vocab, "classes_"):
        group_decode_map = {i: str(c) for i, c in enumerate(group_vocab.classes_)}
    elif hasattr(group_vocab, "vocab"):
        group_decode_map = {i: str(v) for i, v in enumerate(group_vocab.vocab)}
    else:
        group_decode_map = {}

    student._soft_reg          = soft_reg
    student._group_decode      = group_decode_map
    student._soft_targets_path = str(SOFT_TARGETS_PATH)
    student._group_decode_map  = group_decode_map
    student._kd_alpha          = alpha

    trainer, epoch_metrics_cb = make_trainer(
        OUTPUT_DIR, checkpoint_filename=f"best_student_kd_{alpha_tag}" + "-{epoch:02d}",
        gradient_clip=0.5, max_epochs=50, early_stop_patience=5,
        warmup_epochs=WARMUP_EPOCHS, base_lr=STUDENT_BASE_LR,
    )
    trainer.fit(student, train_dataloaders=train_dl, val_dataloaders=val_dl)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    if not best_ckpt or not Path(best_ckpt).exists():
        raise FileNotFoundError(f"No valid checkpoint at '{best_ckpt}'.")
    print(f"  Best checkpoint : {best_ckpt}")

    # Free training model on all ranks before rank-0 eval
    student.cpu(); del student; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    result = {}
    if trainer.is_global_zero:
        best_student = StudentKDTFT.load_from_checkpoint(
            best_ckpt, map_location="cuda" if torch.cuda.is_available() else "cpu")
        best_student.eval()

        _stk = supports_trainer_kwarg(best_student)
        quant_preds_log = run_predict(best_student, val_dl, _stk).cpu().numpy()
        pred_log        = quant_preds_log[:, :, MEDIAN_IDX]
        val_index       = get_prediction_index(val_dataset)
        actuals_log     = extract_actuals_from_index(
            val_index, val_df_s, cfg["target"], cfg["max_prediction_length"])
        val_station_arr = val_index["Location_Information"].values

        biz     = compute_business_metrics(pred_log, actuals_log)
        ql_val  = compute_quantile_loss(quant_preds_log, actuals_log)
        akt_kwh = np.expm1(actuals_log)
        qpk     = np.expm1(quant_preds_log)
        p90_cov = float(np.mean(akt_kwh <= qpk[:, :, Q_IDX[0.90]]))
        mce     = float(np.mean([abs(np.mean(actuals_log <= quant_preds_log[:,:,i]) - q)
                                  for i, q in enumerate(QUANTILES)]))

        metrics = {
            **{k: v for k, v in biz.items() if k != "r2_per_horizon"},
            "r2_per_horizon":         biz["r2_per_horizon"],
            "quantile_loss_log1p":    round(ql_val, 6),
            "p90_coverage":           round(p90_cov, 4),
            "mean_calibration_error": round(mce, 6),
            "quantiles":              QUANTILES,
            "best_val_loss":          round(float(trainer.checkpoint_callback.best_model_score), 6)
                                      if trainer.checkpoint_callback.best_model_score else None,
            "best_ckpt":              best_ckpt,
            "mode":                   "kd_student",
            "kd_alpha":               alpha,
        }
        metrics_path = OUTPUT_DIR / "metrics.json"
        with open(metrics_path, "w") as f: json.dump(metrics, f, indent=2)
        print(f"  Metrics → {metrics_path}")
        _r2h = biz["r2_per_horizon"]
        _g   = lambda lst, i: float("nan") if i >= len(lst) or lst[i] is None else lst[i]
        print(f"  WMAPE={biz['wmape_pct']:.2f}%  R²={biz['r2_score']:.4f}  "
              f"QL={ql_val:.6f}  P90cov={p90_cov:.4f}  MCE={mce:.6f}")
        print(f"  MAE={biz['mae_kwh']:.4f} kWh  RMSE={biz['rmse_kwh']:.4f} kWh")
        print(f"  h1={_g(_r2h,0):.3f}  h6={_g(_r2h,5):.3f}  h12={_g(_r2h,11):.3f}  h24={_g(_r2h,23):.3f}")

        best_student.cpu(); del best_student; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Figures
        tr_l, vl_l = epoch_metrics_cb.aligned_losses()
        plot_training_curve(tr_l, vl_l, OUTPUT_DIR / "training_curve.png",
                            title=f"Student TFT (KD {alpha_label}) — Training Curve")
        plot_predictions(quant_preds_log, actuals_log, val_station_arr,
                         OUTPUT_DIR / "predictions.png",
                         title=f"Student TFT (KD {alpha_label}) — 24h Forecast vs Actual - Jiaxing Dataset",
                         colors=MODEL_COLORS["Student KD"])
        pred_flat   = np.expm1(pred_log).flatten()
        actual_flat = np.expm1(actuals_log).flatten()
        r2_scatter  = float(r2_score(actual_flat, pred_flat))
        plot_pred_vs_actual_scatter(actual_flat, pred_flat,
                                    OUTPUT_DIR / "pred_vs_actual",
                                    model_label=f"Student TFT (KD {alpha_label})", r2_val=r2_scatter,
                                    color=MODEL_COLORS["Student KD"]["pred"])
        # Build per-sequence records with station label and R²
        from sklearn.metrics import r2_score as _r2

        rows = []
        for i, stn in enumerate(val_station_arr):
            a = np.expm1(actuals_log[i])          # shape (24,)
            p = np.expm1(pred_log[i])             # shape (24,)
            r2_val = float(_r2(a, p)) if np.var(a) >= 1e-10 else float("nan")
            for h in range(len(a)):
                rows.append({
                    "station":    stn,
                    "horizon_h":  h + 1,
                    "actual":     float(a[h]),
                    "prediction": float(p[h]),
                    "seq_r2":     round(r2_val, 4),   # same R² for all hours in this sequence
                })

        pd.DataFrame(rows).to_csv(OUTPUT_DIR / "pred_vs_actual.csv", index=False)

        result = {
            "quant_preds_log": quant_preds_log,
            "actuals_log":     actuals_log,
            "station_arr":     val_station_arr,
            "metrics":         metrics,
        }
        print(f"\n  Student KD ({alpha_label}) training complete. Outputs → {OUTPUT_DIR}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def generate_student_config(teacher_config_path: Path, student_config_path: Path,
                             student_encoder_length: int = 168) -> dict:
    """
    Auto-generate student_tft_config.json from the teacher config if it does not
    already exist. Only patches max_encoder_length — all other fields are preserved.

    If student_tft_config.json already exists it is loaded as-is (no overwrite),
    so you can safely hand-edit it for any student-specific tweaks.

    Returns the loaded student config dict.
    """
    if student_config_path.exists():
        with open(student_config_path) as f:
            cfg = json.load(f)
        actual_len = cfg.get("max_encoder_length")
        if actual_len != student_encoder_length:
            raise ValueError(
                f"{student_config_path.name} has max_encoder_length={actual_len}, "
                f"expected {student_encoder_length}. "
                f"Edit the file manually or delete it to regenerate."
            )
        print(f"  [Config] Loaded existing {student_config_path.name} "
              f"(max_encoder_length={actual_len})")
        return cfg

    # Generate from teacher config
    with open(teacher_config_path) as f:
        teacher_cfg = json.load(f)

    student_cfg = {**teacher_cfg, "max_encoder_length": student_encoder_length}

    # Also patch val_loss_start in preprocessing block if present, since a shorter
    # encoder window means PTF needs fewer warm-up rows in the val split.
    # We leave val_loss_start unchanged — it refers to a calendar date, not encoder rows.

    student_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(student_config_path, "w") as f:
        json.dump(student_cfg, f, indent=2)

    print(f"  [Config] Generated {student_config_path.name} "
          f"(max_encoder_length={student_encoder_length}) from {teacher_config_path.name}")
    print(f"           → {student_config_path}")
    print(f"           You may edit this file for additional student-specific overrides.")
    return student_cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train Teacher + Student Baseline + Student KD")
    parser.add_argument(
        "--models", nargs="+",
        choices=["teacher", "baseline", "kd", "all"],
        default=["all"],
        help=(
            "Which model(s) to train. "
            "'all' (default) trains all three. "
            "Example: --models teacher kd"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    models = set(args.models)
    if "all" in models:
        models = {"teacher", "baseline", "kd"}

    pl.seed_everything(SEED, workers=True)
    _is_rank0 = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if _is_rank0:
        print("\n" + "="*64)
        print(f"  Unified Jiaxing Training Script")
        print(f"  Models to train : {sorted(models)}")
        print(f"  Platform        : {platform.system()} · num_workers={NUM_WORKERS}")
        print("="*64)

    # ── Load teacher config (max_encoder_length=336) ─────────────────────────
    with open(CONFIG_PATH) as f:
        cfg_teacher = json.load(f)

    if cfg_teacher.get("max_encoder_length") != 336:
        raise ValueError(
            f"tft_config.json has max_encoder_length={cfg_teacher.get('max_encoder_length')}, "
            f"expected 336 (teacher). Check your config file."
        )

    # ── Load / generate student config (max_encoder_length=168) ──────────────
    # student_tft_config.json is auto-created from tft_config.json on first run,
    # patching only max_encoder_length → 168. Edit the file for further tweaks.
    cfg_student = generate_student_config(CONFIG_PATH, STUDENT_CONFIG_PATH,
                                           student_encoder_length=168)

    # ── Load shared preprocessing artifacts & data ────────────────────────────
    fee_scaler, lag_scaler, encoders = load_preprocessing_artifacts(
        FEATURE_SCALER_PKL, LABEL_ENCODERS_PKL)
    train_df, val_df = load_dataframes(TRAIN_PATH, VAL_PATH)
    validate_encoders(encoders, train_df, context="train_df")
    validate_encoders(encoders, val_df,   context="val_df")

    if _is_rank0:
        print(f"  Train rows    : {len(train_df):,}  |  Val rows: {len(val_df):,}")
        print(f"  Stations      : {train_df['Location_Information'].nunique()}")
        print(f"  Target        : {cfg_teacher['target']}")
        print(f"  Teacher encoder length  : {cfg_teacher['max_encoder_length']}h")
        print(f"  Student encoder length  : {cfg_student['max_encoder_length']}h")

    # Collect results for combined panel
    all_results: Dict[str, dict] = {}

    # ── TEACHER ───────────────────────────────────────────────────────────────
    if "teacher" in models:
        teacher_result = run_teacher(cfg_teacher, train_df, val_df, encoders)
        if teacher_result:
            all_results["Teacher"] = teacher_result

    # ── STUDENT BASELINE ──────────────────────────────────────────────────────
    if "baseline" in models:
        baseline_result = run_student_baseline(cfg_student, train_df, val_df, encoders)
        if baseline_result:
            all_results["Student Baseline"] = baseline_result

    # ── STUDENT KD — alpha sweep ──────────────────────────────────────────────
    if "kd" in models:
        for _alpha in KD_ALPHA_SWEEP:
            _alpha_tag   = f"alpha{int(round(_alpha * 100)):03d}"
            _alpha_label = f"α={_alpha:.2f}"
            _kd_result   = run_student_kd(cfg_student, train_df, val_df, encoders, alpha=_alpha)
            if _kd_result:
                all_results[f"Student KD {_alpha_label}"] = _kd_result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── COMBINED 3-MODEL PANEL ────────────────────────────────────────────────
    if _is_rank0 and len(all_results) >= 2:
        print("\n" + "="*64 + "\n  COMBINED 3-MODEL PREDICTION PANEL\n" + "="*64)
        plot_combined_3model(all_results, COMPARISON_DIR / "predictions_3model_panel.png")

    # ── FINAL SUMMARY ─────────────────────────────────────────────────────────
    if _is_rank0:
        print("\n" + "="*64)
        print("  ALL TRAINING COMPLETE — Summary")
        print("="*64)
        for name, res in all_results.items():
            m = res.get("metrics", {})
            print(f"\n  [{name}]")
            print(f"    WMAPE       : {m.get('wmape_pct', 'N/A'):.2f} %")
            print(f"    R²          : {m.get('r2_score', 'N/A'):.4f}")
            print(f"    MAE         : {m.get('mae_kwh', 'N/A'):.4f} kWh")
            print(f"    RMSE        : {m.get('rmse_kwh', 'N/A'):.4f} kWh")
            print(f"    QL (log1p)  : {m.get('quantile_loss_log1p', 'N/A'):.6f}")
            print(f"    P90 Cov     : {m.get('p90_coverage', 'N/A'):.4f}")
            print(f"    MCE         : {m.get('mean_calibration_error', 'N/A'):.6f}")
        print("\n  Output directories:")
        if "teacher" in models:
            print(f"    Teacher    → {TEACHER_OUTPUT_DIR}")
        if "baseline" in models:
            print(f"    Baseline   → {BASELINE_OUTPUT_DIR}")
        if "kd" in models:
            for _alpha in KD_ALPHA_SWEEP:
                _alpha_tag = f"alpha{int(round(_alpha * 100)):03d}"
                print(f"    Student KD α={_alpha:.2f} → models/student_kd_tft111_jiaxing1_{_alpha_tag}")
        if len(all_results) >= 2:
            print(f"    Combined   → {COMPARISON_DIR}/predictions_3model_panel.png")
        print("="*64 + "\n")