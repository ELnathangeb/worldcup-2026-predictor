"""
FIFA World Cup 2026 Prediction Model — entry point.

Usage:
    python main.py --phase data        # Phase 1: download & preprocess
    python main.py --phase features    # Phase 2: feature engineering
    python main.py --phase train       # Phase 3: model training
    python main.py --phase simulate    # Phase 4: tournament simulation
    python main.py --phase all         # Run everything end-to-end
"""

import argparse

from src.utils.logger import get_logger

logger = get_logger("main", log_file="outputs/run.log")


def run_data_phase() -> None:
    from src.data.downloader import download_all
    from src.data.loader import load_results, load_fifa_rankings
    from src.data.validator import validate_results, validate_rankings
    from src.data.preprocessor import preprocess_results, merge_rankings, save_processed

    logger.info("=== Phase 1: Data Collection & Preprocessing ===")

    download_all()

    results = load_results()
    rankings = load_fifa_rankings()

    validate_results(results)
    validate_rankings(rankings)

    processed = preprocess_results(results)
    processed = merge_rankings(processed, rankings)
    save_processed(processed, "matches_processed.csv")

    logger.info("Phase 1 complete. Processed data saved to data/processed/")


def run_features_phase() -> None:
    from pathlib import Path
    import pandas as pd

    from src.data.quality_report import run as quality_report
    from src.data.cleaner import clean_results, clean_rankings, merge_rankings_safe
    from src.data.loader import load_results, load_fifa_rankings
    from src.features.engineer import (
        compute_elo, compute_rolling_form, compute_h2h,
        assemble_features, impute_missing, validate_no_leakage,
    )
    from src.features.documentation import print_feature_docs, save_feature_docs

    logger.info("=== Phase 2: Data Quality, Cleaning & Feature Engineering ===")

    # 1. Quality report
    quality_report()

    # 2. Load & clean
    results_raw = load_results()
    rankings_raw = load_fifa_rankings()
    results_clean = clean_results(results_raw)
    rankings_clean = clean_rankings(rankings_raw)

    # 3. Merge rankings (no leakage — backward merge_asof)
    df = merge_rankings_safe(results_clean, rankings_clean)

    # 4. Elo ratings (chronological)
    df, final_elo = compute_elo(df)

    # 5. Rolling form
    df = compute_rolling_form(df)

    # 6. Head-to-head
    df = compute_h2h(df)

    # 7. Assemble derived features
    df = assemble_features(df)

    # 8. Impute missing values
    df = impute_missing(df)

    # 9. Leakage validation
    validate_no_leakage(df)

    # 10. Save final ML dataset
    PROC_DIR = Path("data/processed")
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROC_DIR / "model_dataset.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"ML dataset saved → {out_path}  shape={df.shape}")

    # 11. Save final Elo ratings for simulation
    import json
    elo_path = Path("outputs") / "final_elo_ratings.json"
    elo_path.parent.mkdir(exist_ok=True)
    with open(elo_path, "w") as f:
        json.dump({k: round(v, 2) for k, v in sorted(final_elo.items(), key=lambda x: -x[1])}, f, indent=2)
    logger.info(f"Final Elo ratings saved → {elo_path}")

    # 12. Documentation + summary
    print_feature_docs(df)
    save_feature_docs(df)
    _phase2_summary(df)

    logger.info("Phase 2 complete.")


def _phase2_summary(df) -> None:
    import pandas as pd
    import numpy as np

    ML_FEATURES = [
        "rank_diff", "fifa_pts_diff", "home_elo", "away_elo", "elo_diff",
        "home_form5", "away_form5", "home_form10", "away_form10",
        "home_win_rate5", "away_win_rate5", "home_draw_rate5", "away_draw_rate5",
        "home_win_rate10", "away_win_rate10", "home_draw_rate10", "away_draw_rate10",
        "home_gf_avg5", "away_gf_avg5", "home_ga_avg5", "away_ga_avg5", "home_gd_avg5", "away_gd_avg5",
        "home_gf_avg10", "away_gf_avg10", "home_ga_avg10", "away_ga_avg10", "home_gd_avg10", "away_gd_avg10",
        "h2h_home_wins", "h2h_draws", "h2h_away_wins", "h2h_home_gf_avg", "h2h_away_gf_avg", "h2h_total_games",
        "home_advantage", "is_neutral", "tournament_weight", "tournament_tier", "is_world_cup",
        "home_days_rest", "away_days_rest",
        "home_rank", "away_rank", "home_fifa_pts", "away_fifa_pts",
    ]

    sep = "=" * 72
    print(f"\n{sep}")
    print("  PHASE 2 SUMMARY — DATASET READY FOR TRAINING")
    print(sep)
    print(f"\n  Dataset shape      : {df.shape[0]:,} rows × {df.shape[1]} columns")
    feat_cols = [c for c in ML_FEATURES if c in df.columns]
    print(f"  ML features        : {len(feat_cols)}")
    print(f"  Date range         : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"\n  Target distribution (result):")
    vc = df["result"].value_counts().sort_index()
    labels = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    for k, v in vc.items():
        print(f"    {labels[k]:10s} ({k})  : {v:6,}  ({v/len(df)*100:.1f}%)")

    print(f"\n  Sample rows (5 random):")
    sample_cols = ["date","home_team","away_team","result","elo_diff","rank_diff","home_form5","away_form5","h2h_total_games"]
    print(df[sample_cols].sample(5, random_state=7).to_string(index=False))

    null_counts = df[feat_cols].isnull().sum()
    remaining_nulls = null_counts[null_counts > 0]
    print(f"\n  Missing values in ML features : {remaining_nulls.sum()}")

    print(f"\n  RECOMMENDATIONS BEFORE TRAINING")
    print("  ─────────────────────────────────────────────────────────────")
    recs = [
        ("1", "Feature scaling",
         "Tree models (XGBoost/LightGBM) don't need scaling. Logistic Regression does.\n"
         "     Apply StandardScaler inside the pipeline, not before splitting."),
        ("2", "Train/val/test split",
         "Use a TIME-BASED split, not random.\n"
         "     Suggested: train on 2000–2021, validate on 2022–2023, test on 2024–2026.\n"
         "     This mirrors real prediction conditions and avoids temporal leakage."),
        ("3", "Class imbalance",
         f"Home win: {vc.get(0,0)/len(df)*100:.0f}%  Draw: {vc.get(1,0)/len(df)*100:.0f}%  Away win: {vc.get(2,0)/len(df)*100:.0f}%.\n"
         "     Draws are underrepresented. Use class_weight='balanced' or sample weighting."),
        ("4", "Friendly matches",
         "Friendlies (tournament_tier=0) are low-signal. Consider training on\n"
         "     tier>=1 (competitive matches) only, or upweighting competitive matches."),
        ("5", "Elo vs FIFA ranking",
         "elo_diff and rank_diff are correlated (~0.6). Both add signal but watch multicollinearity\n"
         "     if using linear models. XGBoost handles this natively."),
        ("6", "H2H data sparsity",
         f"  {(df['h2h_total_games']==0).sum():,} matches ({(df['h2h_total_games']==0).mean()*100:.0f}%) have no H2H history.\n"
         "     These used the 1/3 uniform prior. h2h_total_games is included as a confidence weight."),
        ("7", "FIFA points gap (2020–2026)",
         "Rankings data ends Nov 2019. Matches 2020–2026 use forward-filled last-known points.\n"
         "     Elo ratings are fully up-to-date as they are computed from all matches.\n"
         "     Consider dropping home_rank/away_rank for post-2019 rows if model is sensitive to this."),
    ]
    for num, title, body in recs:
        print(f"\n  [{num}] {title}")
        print(f"     {body}")
    print(f"\n{sep}\n")


def run_train_phase() -> None:
    import joblib
    import json
    import pandas as pd
    from pathlib import Path

    from src.models.splitter  import split
    from src.models.trainer   import train_all, results_to_df
    from src.models.tuner     import tune
    from src.models.evaluator import (
        plot_confusion_matrices,
        plot_feature_importance,
        plot_model_comparison,
        per_class_report,
        importance_table,
    )
    from src.models.reporter  import generate as generate_report

    logger.info("=== Phase 3: Model Training & Evaluation ===")

    # 1. Load dataset
    df = pd.read_csv("data/processed/model_dataset.csv", parse_dates=["date"])
    logger.info(f"Dataset loaded: {df.shape}")

    # 2. Chronological split
    X_train, y_train, X_val, y_val, X_test, y_test, feat_cols = split(df)
    logger.info(
        f"Splits — Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}"
    )

    # 3. Train all models
    trained, results = train_all(X_train, y_train, X_val, y_val, X_test, y_test)

    # 4. Results table
    results_df = results_to_df(results)
    test_df = results_df[results_df["Split"] == "test"].sort_values("F1 (macro)", ascending=False)

    print("\n" + "=" * 72)
    print("  MODEL COMPARISON — TEST SET (2024–2026)")
    print("=" * 72)
    print(test_df.drop(columns=["Split"]).to_string(index=False))
    print("=" * 72 + "\n")

    # 5. Pick best model by test F1
    best_name = test_df.iloc[0]["Model"]
    logger.info(f"Best model: {best_name}  (F1={test_df.iloc[0]['F1 (macro)']:.4f})")

    # 6. Hyperparameter tuning
    base_pipe = trained[best_name]
    tuned_pipe, tune_results = tune(
        best_model_name=best_name,
        base_pipeline=base_pipe,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        n_iter=60,
        cv_folds=5,
    )

    # 7. Per-class report
    print("\nPer-class classification report (tuned model, test set):")
    print(per_class_report(tuned_pipe, X_test, y_test))

    # 8. Save model artefacts
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(tuned_pipe, models_dir / "best_model.pkl")
    joblib.dump(feat_cols,  models_dir / "feature_columns.pkl")
    # Save metadata alongside
    meta = {
        "model_name": best_name,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "train_range": "2000-01-04 – 2021-12-31",
        "test_range":  "2024-01-01 – present",
        "tune_results": {k: v for k, v in tune_results.items() if k != "cv_results_top5"},
    }
    with open(models_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Model artefacts saved → {models_dir}/")

    # 9. Plots
    plot_confusion_matrices(trained, X_test, y_test)
    plot_feature_importance(best_name, tuned_pipe, feat_cols)
    plot_model_comparison(results_df)

    # 10. Markdown report
    generate_report(
        results_df=results_df,
        trained=trained,
        best_name=best_name,
        tuned_pipe=tuned_pipe,
        tune_results=tune_results,
        feat_cols=feat_cols,
        X_test=X_test,
        y_test=y_test,
        X_train=X_train,
        y_train=y_train,
    )

    logger.info("Phase 3 complete.")
    logger.info(f"Best model:   models/best_model.pkl  ({best_name})")
    logger.info(f"Features:     models/feature_columns.pkl")
    logger.info(f"Report:       reports/model_training_report.md")


def run_simulate_phase() -> None:
    import argparse as _ap
    from src.monte_carlo import main as mc_main

    logger.info("=== Phase 4: Tournament Simulation ===")

    # Allow --n / --seed / --quick to be passed through from CLI
    _p = _ap.ArgumentParser(add_help=False)
    _p.add_argument("--n",     type=int, default=10_000)
    _p.add_argument("--seed",  type=int, default=None)
    _p.add_argument("--quick", action="store_true")
    _args, _ = _p.parse_known_args()

    mc_main(n=_args.n, seed=_args.seed, quick=_args.quick)
    logger.info("Phase 4 complete.")


PHASES = {
    "data": run_data_phase,
    "features": run_features_phase,
    "train": run_train_phase,
    "simulate": run_simulate_phase,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="WC2026 Prediction Pipeline")
    parser.add_argument("--phase", choices=[*PHASES, "all"], default="data")
    args = parser.parse_args()

    if args.phase == "all":
        for fn in PHASES.values():
            fn()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
