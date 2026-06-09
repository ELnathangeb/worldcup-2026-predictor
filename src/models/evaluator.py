"""
Model evaluation utilities:
  - Per-class metrics
  - Confusion matrix plots
  - Feature importance extraction and plotting
  - Model comparison table
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import Pipeline

from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASS_NAMES = ["Home Win", "Draw", "Away Win"]
PLOT_DIR = Path("outputs/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    trained: dict[str, Pipeline],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_path: Path = PLOT_DIR / "confusion_matrices.png",
) -> Path:
    n = len(trained)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    for ax, (name, pipe) in zip(axes, trained.items()):
        y_pred = pipe.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        # Normalise rows for readability
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(
            cm_norm, annot=cm, fmt="d", ax=ax,
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cmap="Blues", cbar=False,
            annot_kws={"size": 9},
        )
        acc = (y_pred == y_test.values).mean()
        ax.set_title(f"{name}\n(acc={acc:.3f})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

    # Hide unused subplots
    for ax in axes[len(trained):]:
        ax.set_visible(False)

    fig.suptitle("Confusion Matrices — Test Set", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrices saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _extract_importance(pipe: Pipeline, feat_cols: list[str]) -> pd.DataFrame | None:
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_).mean(axis=0)
    else:
        return None
    return pd.DataFrame({"feature": feat_cols, "importance": imp}).sort_values(
        "importance", ascending=False
    )


def plot_feature_importance(
    best_name: str,
    best_pipe: Pipeline,
    feat_cols: list[str],
    top_n: int = 25,
    save_path: Path = PLOT_DIR / "feature_importance.png",
) -> Path:
    imp_df = _extract_importance(best_pipe, feat_cols)
    if imp_df is None:
        logger.warning(f"Cannot extract feature importance for {best_name}")
        return save_path

    top = imp_df.head(top_n).copy()
    top = top.sort_values("importance")  # ascending so bar chart reads top-to-bottom

    # Color by feature group
    def _group_color(feat: str) -> str:
        if "elo" in feat:             return "#2196F3"  # blue
        if "rank" in feat or "fifa" in feat: return "#FF9800"  # orange
        if "form" in feat or "win_rate" in feat or "draw_rate" in feat: return "#4CAF50"  # green
        if "gf" in feat or "ga" in feat or "gd" in feat: return "#9C27B0"  # purple
        if "h2h" in feat:             return "#F44336"  # red
        return "#607D8B"              # grey for context

    colors = [_group_color(f) for f in top["feature"]]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.32)))
    bars = ax.barh(top["feature"], top["importance"], color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Feature Importance (mean |gain|)", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — {best_name}", fontsize=13, fontweight="bold")
    ax.tick_params(axis="y", labelsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="Elo Ratings"),
        Patch(facecolor="#FF9800", label="FIFA Rankings"),
        Patch(facecolor="#4CAF50", label="Rolling Form"),
        Patch(facecolor="#9C27B0", label="Goals (Rolling)"),
        Patch(facecolor="#F44336", label="Head-to-Head"),
        Patch(facecolor="#607D8B", label="Context"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    # Add value labels
    for bar, val in zip(bars, top["importance"]):
        ax.text(
            bar.get_width() + max(top["importance"]) * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Feature importance plot saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Model comparison bar chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Path = PLOT_DIR / "model_comparison.png",
) -> Path:
    test_df = results_df[results_df["Split"] == "test"].copy()
    metrics = ["Accuracy", "F1 (macro)", "Precision", "Recall"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    palette = sns.color_palette("Set2", n_colors=len(test_df))

    for ax, metric in zip(axes, metrics):
        sorted_df = test_df.sort_values(metric, ascending=False)
        bars = ax.bar(sorted_df["Model"], sorted_df[metric], color=palette, edgecolor="white")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(max(0, sorted_df[metric].min() - 0.05), min(1, sorted_df[metric].max() + 0.05))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.set_ylabel("Score")
        for bar, val in zip(bars, sorted_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    fig.suptitle("Model Comparison — Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Model comparison plot saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# Per-class breakdown
# ---------------------------------------------------------------------------

def per_class_report(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    y_pred = pipe.predict(X_test)
    return classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4)


# ---------------------------------------------------------------------------
# Full importance table
# ---------------------------------------------------------------------------

def importance_table(pipe: Pipeline, feat_cols: list[str]) -> pd.DataFrame | None:
    return _extract_importance(pipe, feat_cols)
