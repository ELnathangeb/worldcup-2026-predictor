"""
Model training — five classifiers, each evaluated on val + test sets.

Models
------
1. Logistic Regression  (baseline, interpretable)
2. Random Forest        (non-linear ensemble, handles missing patterns)
3. Gradient Boosting    (sklearn, slower but well-calibrated)
4. XGBoost              (fast gradient boosting, gold standard)
5. LightGBM             (fastest, often best on tabular data)

Each model is wrapped in a sklearn Pipeline so that scaling only ever
sees training data (StandardScaler fit on train, transform on val/test).
Tree models don't benefit from scaling but it doesn't hurt either and
keeps the interface uniform.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    logger.warning("LightGBM not available")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def _build_models(class_weight: dict | None) -> dict[str, Pipeline]:
    """Return ordered dict of name → sklearn Pipeline."""
    cw = class_weight  # {0: w0, 1: w1, 2: w2}

    models: dict[str, Pipeline] = {}

    models["Logistic Regression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=2000,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight=cw,
            C=1.0,
            random_state=42,
        )),
    ])

    models["Random Forest"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=10,
            class_weight=cw,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    models["Gradient Boosting"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    if HAS_XGB:
        # Compute sample_weight separately (XGB doesn't accept class_weight dict)
        models["XGBoost"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=400,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=2,
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
            )),
        ])

    if HAS_LGBM:
        models["LightGBM"] = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LGBMClassifier(
                feature_name="auto",
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=2,
                class_weight=cw,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )),
        ])

    return models


def _class_weight_dict(y_train: pd.Series) -> dict[int, float]:
    """Balanced class weights: w_c = n_samples / (n_classes * n_c)."""
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.array(sorted(y_train.unique()))
    weights = compute_class_weight("balanced", classes=classes, y=y_train.values)
    return dict(zip(classes.tolist(), weights.tolist()))


def _sample_weights(y: pd.Series, cw: dict[int, float]) -> np.ndarray:
    """Map class weights to per-sample weights (for XGBoost)."""
    return np.array([cw[c] for c in y])


def _evaluate(name: str, pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
               split_label: str) -> dict[str, Any]:
    y_pred = pipe.predict(X)
    y_prob = pipe.predict_proba(X)
    acc    = accuracy_score(y, y_pred)
    prec   = precision_score(y, y_pred, average="macro", zero_division=0)
    rec    = recall_score(y, y_pred, average="macro", zero_division=0)
    f1     = f1_score(y, y_pred, average="macro", zero_division=0)
    ll     = log_loss(y, y_prob)
    cm     = confusion_matrix(y, y_pred, labels=[0, 1, 2])
    return {
        "model": name,
        "split": split_label,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec, 4),
        "recall_macro": round(rec, 4),
        "f1_macro": round(f1, 4),
        "log_loss": round(ll, 4),
        "confusion_matrix": cm,
    }


def train_all(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame,   y_val: pd.Series,
    X_test: pd.DataFrame,  y_test: pd.Series,
) -> tuple[dict[str, Pipeline], list[dict]]:
    """
    Train every model, evaluate on val and test.

    Returns
    -------
    trained  : dict[model_name → fitted Pipeline]
    results  : list of metric dicts (one per model × split)
    """
    cw = _class_weight_dict(y_train)
    logger.info(f"Class weights: {cw}")

    models  = _build_models(cw)
    trained: dict[str, Pipeline] = {}
    results: list[dict] = []

    for name, pipe in models.items():
        logger.info(f"Training {name} ...")
        t0 = time.time()

        # XGBoost needs sample_weight passed via fit params
        if name == "XGBoost":
            sw = _sample_weights(y_train, cw)
            pipe.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            pipe.fit(X_train, y_train)

        elapsed = time.time() - t0
        logger.info(f"  {name} trained in {elapsed:.1f}s")

        trained[name] = pipe

        for X_ev, y_ev, label in [(X_val, y_val, "val"), (X_test, y_test, "test")]:
            row = _evaluate(name, pipe, X_ev, y_ev, label)
            results.append(row)
            logger.info(
                f"  [{label}] acc={row['accuracy']:.4f}  "
                f"f1={row['f1_macro']:.4f}  "
                f"logloss={row['log_loss']:.4f}"
            )

    return trained, results


def results_to_df(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Model": r["model"],
            "Split": r["split"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision_macro"],
            "Recall": r["recall_macro"],
            "F1 (macro)": r["f1_macro"],
            "Log Loss": r["log_loss"],
        })
    return pd.DataFrame(rows)
