"""
Chronological train / validation / test split.

NO random splitting — matches are time-ordered so the model never sees
future results during training (avoids temporal leakage).

Split boundaries:
  Train : 2000-01-01 – 2021-12-31   (full historical base)
  Val   : 2022-01-01 – 2023-12-31   (hyperparameter tuning / early stopping)
  Test  : 2024-01-01 – present       (held-out final evaluation)
"""
from __future__ import annotations

import pandas as pd

from src.models.config import ML_FEATURES, TARGET, TRAIN_END, VAL_END
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split(df: pd.DataFrame) -> tuple[
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
    pd.DataFrame, pd.Series,
]:
    """
    Returns (X_train, y_train, X_val, y_val, X_test, y_test).
    Feature columns are taken from ML_FEATURES; any missing cols raise KeyError.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Resolve actual feature list (handle 'neutral' vs 'is_neutral' naming)
    feat_cols = []
    for f in ML_FEATURES:
        if f in df.columns:
            feat_cols.append(f)
        elif f == "neutral" and "neutral" not in df.columns:
            logger.warning(f"Feature '{f}' not found — skipping")
        else:
            raise KeyError(f"Feature '{f}' not found in dataset columns: {df.columns.tolist()}")

    train_mask = df["date"] <= TRAIN_END
    val_mask   = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    test_mask  = df["date"] > VAL_END

    def _xy(mask: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
        sub = df[mask]
        return sub[feat_cols].reset_index(drop=True), sub[TARGET].reset_index(drop=True)

    X_tr, y_tr = _xy(train_mask)
    X_va, y_va = _xy(val_mask)
    X_te, y_te = _xy(test_mask)

    logger.info(
        f"Split sizes — Train: {len(X_tr):,}  Val: {len(X_va):,}  Test: {len(X_te):,}"
    )
    logger.info(f"Features used: {len(feat_cols)}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, feat_cols
