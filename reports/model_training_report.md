# FIFA World Cup 2026 — Model Training Report

_Generated: 2026-06-08 17:19_

---

## 1. Overview

This report documents the training, evaluation, and selection of the best machine learning model to predict FIFA World Cup 2026 match outcomes.

**Target variable:** `result`  
- `0` = Home team loss  
- `1` = Draw  
- `2` = Home team win

**Dataset:**
- Training rows : 19,527 matches (2000 – 2021)
- Test rows     : 2,474  matches (2024 – 2026)
- Features      : 46


## 2. Chronological Data Split

A **time-based split** was used to mirror real-world prediction conditions and prevent temporal leakage:

| Split | Date Range | Rows |
|---|---|---|
| Train | 2000-01-01 – 2021-12-31 | 19,527 |
| Validation | 2022-01-01 – 2023-12-31 | 0 (used for tuning) |
| **Test** | **2024-01-01 – present** | **2,474** |


## 3. Model Comparison (Test Set)

All models evaluated on the held-out test set (2024–2026):

| Model | Accuracy | Precision | Recall | F1 (macro) | Log Loss |
|---|---|---|---|---|---|
| **LightGBM ⭐** | 0.5756 | 0.5290 | 0.5338 | 0.5309 | 0.8991 |
| **XGBoost** | 0.5707 | 0.5259 | 0.5325 | 0.5282 | 0.8920 |
| **Logistic Regression** | 0.5687 | 0.5135 | 0.5261 | 0.5165 | 0.8966 |
| **Random Forest** | 0.5772 | 0.5126 | 0.5264 | 0.5155 | 0.8922 |
| **Gradient Boosting** | 0.5982 | 0.5059 | 0.5118 | 0.4733 | 0.8749 |


## 4. Best Model: LightGBM

### 4.1 Why LightGBM Performed Best

**LightGBM** outperformed the other models for several interconnected reasons:

1. **Gradient boosting handles tabular non-linearity well.** Soccer outcomes are influenced by non-linear interactions (e.g., Elo difference matters more when form is diverging), which boosted trees capture automatically through deep splits.

2. **Robustness to correlated features.** `elo_diff` and `rank_diff` are correlated (~0.6), and multiple rolling-form features overlap. Boosted trees implicitly regularise against collinearity by selecting the most informative splits.

3. **Handles imbalanced classes better than linear models.** With draws at ~23%, logistic regression struggles to separate the draw class. Boosting iteratively corrects residuals, naturally improving minority-class recall.

4. **No need for feature scaling.** Tree splits are threshold-based; scale invariance avoids distortion from forward-filled FIFA rankings (which freeze in 2019).

### 4.2 Tuning Results

- Method: **RandomizedSearchCV** with `TimeSeriesSplit(n_splits=5)` — temporal CV, no future leakage
- Iterations: **60** random hyperparameter combinations
- Optimised metric: **macro F1**

| Metric | Score |
|---|---|
| Best CV F1 (macro) | 0.5049 |
| Test Accuracy (tuned) | 0.5728 |
| Test F1 macro (tuned) | 0.5274 |
| Test Log Loss (tuned) | 0.8893 |
| Tuning time | 25.53 min |

**Best hyperparameters:**

```
clf__subsample: 1.0
clf__reg_lambda: 0.5
clf__reg_alpha: 0.1
clf__num_leaves: 47
clf__n_estimators: 300
clf__min_child_samples: 50
clf__max_depth: 4
clf__learning_rate: 0.03
clf__colsample_bytree: 1.0
```

### 4.3 Confusion Matrix (Tuned Model — Test Set)

| Actual \ Predicted | Home Win | Draw | Away Win |
|---|---|---|---|
| **Home Win** | 783 | 227 | 163 |
| **Draw** | 207 | 160 | 220 |
| **Away Win** | 108 | 132 | 474 |


_Rows = Actual class, Columns = Predicted class_

- **Home Win** recall: 66.8% (most common class, easiest to predict)
- **Draw** recall: 27.3% (hardest class — draws are inherently low-signal)
- **Away Win** recall: 66.4%

### 4.4 Top 15 Features by Importance

| Rank | Feature | Importance | Group |
|---|---|---|---|
| 1 | `elo_diff` | 925.00000 | Elo Rating |
| 2 | `rank_diff` | 768.00000 | FIFA Ranking |
| 3 | `away_rank` | 525.00000 | FIFA Ranking |
| 4 | `fifa_pts_diff` | 446.00000 | FIFA Ranking |
| 5 | `away_days_rest` | 438.00000 | Context |
| 6 | `home_days_rest` | 402.00000 | Context |
| 7 | `away_elo` | 397.00000 | Elo Rating |
| 8 | `h2h_away_gf_avg` | 388.00000 | Goals |
| 9 | `h2h_home_gf_avg` | 383.00000 | Goals |
| 10 | `home_elo` | 372.00000 | Elo Rating |
| 11 | `home_ga_avg10` | 331.00000 | Goals |
| 12 | `neutral` | 331.00000 | Context |
| 13 | `away_fifa_pts` | 331.00000 | FIFA Ranking |
| 14 | `away_ga_avg5` | 305.00000 | Goals |
| 15 | `tournament_weight` | 300.00000 | Context |


## 5. Per-Class Metrics (Tuned Model)

```
              precision    recall  f1-score   support

    Home Win     0.7131    0.6675    0.6896      1173
        Draw     0.3083    0.2726    0.2893       587
    Away Win     0.5531    0.6639    0.6034       714

    accuracy                         0.5728      2474
   macro avg     0.5248    0.5347    0.5274      2474
weighted avg     0.5709    0.5728    0.5697      2474

```


## 6. Model Weaknesses & Limitations

| # | Weakness | Impact | Mitigation |
|---|---|---|---|
| 1 | **Draw prediction** — recall 27.3% | Draws are low-signal; no team deliberately plays for a draw | Poisson regression for exact scores may help |
| 2 | **FIFA ranking gap** — rankings frozen at Nov 2019 | Forward-filled points lose accuracy for 2020–2026 | Elo (fully current) compensates; retrain once live rankings available |
| 3 | **No squad/injury data** | Key absences (e.g., star player out) not captured | Integrate squad depth / availability API in future |
| 4 | **No xG or shot data** | Expected goals correlate strongly with true quality | Add Opta/StatsBomb xG if available |
| 5 | **Historical class imbalance** — home wins at 48% | Model biased toward home wins, especially for neutral venues | `is_neutral` feature partially compensates |
| 6 | **Small World Cup test set** | Only ~384 World Cup matches in history; high variance on WC predictions | Monte Carlo simulation (Phase 4) propagates this uncertainty |


## 7. Saved Artifacts

| File | Description |
|---|---|
| `models/best_model.pkl` | Tuned best model pipeline (scaler + classifier) |
| `models/feature_columns.pkl` | Ordered list of feature column names |
| `outputs/final_elo_ratings.json` | Current Elo ratings for all teams (for simulation) |
| `outputs/plots/confusion_matrices.png` | Confusion matrices for all models |
| `outputs/plots/feature_importance.png` | Feature importance bar chart |
| `outputs/plots/model_comparison.png` | Side-by-side metric comparison |
| `outputs/feature_documentation.csv` | Full feature glossary |


---

_Report generated automatically by `main.py --phase train` on 2026-06-08 17:19_