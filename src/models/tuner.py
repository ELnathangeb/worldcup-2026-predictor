"""
Hyperparameter tuning for the best-performing model.

Strategy
--------
- Use RandomizedSearchCV with TimeSeriesSplit cross-validation so folds
  respect temporal order and avoid future leakage within the training set.
- Optimise for macro F1 (balances all three classes; better than accuracy
  when classes are imbalanced).
- After tuning, refit on full train+val combined and evaluate on test.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score, log_loss

from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# ---------------------------------------------------------------------------
# Search spaces for each candidate model
# ---------------------------------------------------------------------------

_XGB_SPACE = {
    "clf__n_estimators":     [200, 300, 400, 500, 600],
    "clf__max_depth":        [3, 4, 5, 6],
    "clf__learning_rate":    [0.02, 0.03, 0.05, 0.08, 0.1],
    "clf__subsample":        [0.65, 0.75, 0.85, 1.0],
    "clf__colsample_bytree": [0.65, 0.75, 0.85, 1.0],
    "clf__min_child_weight": [1, 3, 5, 7],
    "clf__reg_alpha":        [0, 0.05, 0.1, 0.3, 0.5],
    "clf__reg_lambda":       [0.5, 1, 2, 3, 5],
    "clf__gamma":            [0, 0.1, 0.2, 0.3],
}

_LGBM_SPACE = {
    "clf__n_estimators":     [200, 300, 400, 500, 600],
    "clf__max_depth":        [4, 6, 8, -1],
    "clf__learning_rate":    [0.02, 0.03, 0.05, 0.08, 0.1],
    "clf__num_leaves":       [31, 47, 63, 95, 127],
    "clf__subsample":        [0.65, 0.75, 0.85, 1.0],
    "clf__colsample_bytree": [0.65, 0.75, 0.85, 1.0],
    "clf__reg_alpha":        [0, 0.05, 0.1, 0.3, 0.5],
    "clf__reg_lambda":       [0.5, 1, 2, 3, 5],
    "clf__min_child_samples":[10, 20, 30, 50],
}

_SEARCH_SPACES = {
    "XGBoost":  _XGB_SPACE  if HAS_XGB  else None,
    "LightGBM": _LGBM_SPACE if HAS_LGBM else None,
}


def tune(
    best_model_name: str,
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_iter: int = 60,
    cv_folds: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, dict[str, Any]]:
    """
    Run RandomizedSearchCV on the best model, then refit on train+val.

    Returns
    -------
    tuned_pipe   : fitted Pipeline with best hyperparameters
    tune_results : dict with best_params and test metrics
    """
    search_space = _SEARCH_SPACES.get(best_model_name)
    if search_space is None:
        logger.warning(f"No tuning space defined for {best_model_name} — returning base model")
        base_pipeline.fit(X_train, y_train)
        return base_pipeline, {}

    logger.info(f"Tuning {best_model_name} with {n_iter} random iterations × {cv_folds} CV folds ...")

    # TimeSeriesSplit: respects temporal ordering within the training window
    tscv = TimeSeriesSplit(n_splits=cv_folds)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=tscv,
        refit=True,
        random_state=random_state,
        n_jobs=-1,
        verbose=1,
    )

    t0 = time.time()
    # XGBoost class weights via sample_weight
    if best_model_name == "XGBoost":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array(sorted(y_train.unique()))
        cw = compute_class_weight("balanced", classes=classes, y=y_train.values)
        cw_dict = dict(zip(classes.tolist(), cw.tolist()))
        sw = np.array([cw_dict[c] for c in y_train])
        search.fit(X_train, y_train, clf__sample_weight=sw)
    else:
        search.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(f"Search complete in {elapsed/60:.1f} min")
    logger.info(f"Best CV F1: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    best_pipe = search.best_estimator_

    # Refit on train + val combined with best params
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    logger.info("Refitting on train+val with best params ...")
    if best_model_name == "XGBoost":
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.array(sorted(y_trainval.unique()))
        cw = compute_class_weight("balanced", classes=classes, y=y_trainval.values)
        cw_dict = dict(zip(classes.tolist(), cw.tolist()))
        sw_tv = np.array([cw_dict[c] for c in y_trainval])
        best_pipe.fit(X_trainval, y_trainval, clf__sample_weight=sw_tv)
    else:
        best_pipe.fit(X_trainval, y_trainval)

    # Final test evaluation
    y_pred = best_pipe.predict(X_test)
    y_prob = best_pipe.predict_proba(X_test)

    tune_results = {
        "model": best_model_name,
        "best_cv_f1": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
        "tuning_time_min": round(elapsed / 60, 2),
        "test_accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "test_f1_macro":  round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "test_log_loss":  round(float(log_loss(y_test, y_prob)), 4),
        "cv_results_top5": _top5_cv(search),
    }

    logger.info(
        f"Tuned model — Test acc={tune_results['test_accuracy']:.4f}  "
        f"f1={tune_results['test_f1_macro']:.4f}  "
        f"logloss={tune_results['test_log_loss']:.4f}"
    )
    return best_pipe, tune_results


def _top5_cv(search: RandomizedSearchCV) -> list[dict]:
    cv_df = pd.DataFrame(search.cv_results_)
    top = cv_df.nlargest(5, "mean_test_score")[["params", "mean_test_score", "std_test_score"]]
    rows = []
    for _, r in top.iterrows():
        rows.append({
            "params": r["params"],
            "mean_f1": round(float(r["mean_test_score"]), 4),
            "std_f1":  round(float(r["std_test_score"]),  4),
        })
    return rows
