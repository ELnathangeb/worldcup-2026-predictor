# ⚽ FIFA World Cup 2026 — Prediction & Simulation System

A production-quality machine learning system for predicting FIFA World Cup 2026 match outcomes and simulating the full 48-team tournament. Built end-to-end: raw data ingestion → feature engineering → model training → tournament simulation → interactive Streamlit dashboard.

---

## Features

- **Match Prediction** — LightGBM classifier with 46 features (Elo, FIFA rankings, rolling form, H2H records) predicts home win / draw / away win probabilities for any international fixture
- **Elo Engine** — Custom Elo rating system computed across 49,000+ historical matches; 259 national teams rated
- **Full Tournament Simulator** — Simulates the complete WC 2026 bracket: 12 groups × round-robin → best-8 third-place rule → R32 → R16 → QF → SF → Final (with extra time and penalty shootouts)
- **Monte Carlo Engine** — 10,000 independent full-tournament simulations in ~7 seconds; outputs championship, runner-up, SF, QF, R32 probabilities for all 48 teams
- **Interactive Dashboard** — 6-page Streamlit app: team explorer, match predictor, live bracket simulator, championship odds charts, data insights
- **Reproducible Pipeline** — Single `main.py` CLI runs the full pipeline from raw data download through report generation

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| ML Model | LightGBM (multi-class classifier) |
| Feature Engineering | pandas, NumPy, scikit-learn |
| Simulation | NumPy (Poisson distribution), Elo rating system |
| Dashboard | Streamlit 1.35+, Plotly Express + Graph Objects |
| Score Matrix | SciPy (Poisson PMF) |
| Stats | statsmodels (OLS trendline) |
| Config | PyYAML |
| Serialisation | joblib (model), JSON (Elo ratings) |
| Testing | pytest |

---

## Project Structure

```
Worldcup/
├── configs/
│   ├── config.yaml                  # Pipeline configuration
│   └── wc2026_groups.yaml           # Official WC 2026 group draw (Dec 2024)
│
├── dashboard/
│   ├── streamlit_app.py             # Entry point — run this
│   ├── utils.py                     # Shared loaders, CSS, constants
│   └── pages/
│       ├── home.py                  # Overview, KPIs, top favourites
│       ├── team_explorer.py         # Form, W/D/L trends, H2H
│       ├── match_predictor.py       # ML + Elo prediction, Poisson heatmap
│       ├── tournament_simulator.py  # Live bracket simulation
│       ├── championship_odds.py     # Monte Carlo results & charts
│       └── data_insights.py        # Feature importance, correlations
│
├── data/
│   ├── raw/                         # Source CSV files (gitignored)
│   └── processed/
│       ├── matches_processed.csv    # Cleaned matches with rankings
│       └── model_dataset.csv        # 23,958-row ML-ready dataset
│
├── models/
│   ├── best_model.pkl               # Trained LightGBM pipeline (gitignored)
│   ├── feature_columns.pkl          # Ordered 46-feature list (gitignored)
│   └── model_metadata.json          # Training metrics & config
│
├── outputs/
│   ├── final_elo_ratings.json       # Elo ratings for 259 teams
│   ├── monte_carlo_results.csv      # 10,000-sim results (all 48 WC teams)
│   └── single_simulation_bracket.json
│
├── reports/
│   ├── model_training_report.md     # Full model evaluation report
│   ├── tournament_simulation_report.md
│   └── figures/                     # 5 chart PNGs from Monte Carlo
│
├── src/
│   ├── predict_match.py             # ML + Elo prediction API
│   ├── simulate_tournament.py       # CLI: single tournament run
│   ├── monte_carlo.py               # Monte Carlo engine + chart generation
│   ├── simulation/
│   │   └── tournament.py            # Core bracket simulator
│   ├── data/                        # Loaders, cleaners, validators
│   ├── features/                    # Feature engineering, Elo computation
│   └── models/                      # Trainers, tuners, evaluators
│
├── tests/
│   ├── test_data.py                 # Data preprocessing tests
│   └── test_predict_match.py        # Prediction & simulation tests
│
├── main.py                          # Full pipeline CLI
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python 3.9+
- ~500 MB disk (raw data + model)

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/worldcup-2026-predictor.git
cd worldcup-2026-predictor
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

Opens at **http://localhost:8501** — all pre-computed results (Elo ratings, Monte Carlo odds, trained model) are loaded from `outputs/` and `models/`.

### 3. Run a single tournament simulation (CLI)

```bash
python src/simulate_tournament.py --seed 42 --verbose
```

Prints the full bracket to stdout and saves JSON to `outputs/single_simulation_bracket.json`.

### 4. Re-run Monte Carlo (10,000 simulations)

```bash
python src/monte_carlo.py
```

Runs in ~7 seconds. Saves `outputs/monte_carlo_results.csv` and 5 charts to `reports/figures/`.

### 5. Run the full ML pipeline from scratch

> Requires internet access for data download. Regenerates everything.

```bash
python main.py --phase data        # Download & preprocess (~2 min)
python main.py --phase features    # Feature engineering (~1 min)
python main.py --phase train       # Train & evaluate models (~5 min)
python main.py --phase simulate    # Monte Carlo + report (~1 min)
# or
python main.py --phase all         # End-to-end
```

### 6. Run tests

```bash
pytest tests/ -v
```

All 17 tests pass.

---

## Dashboard Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Dataset stats, model performance metrics, WC 2026 groups with Elo, top 5 favourites bar chart |
| 🌍 **Team Explorer** | 12-match rolling goal trend, W/D/L breakdown, annual win rate, full match history, head-to-head stats |
| ⚡ **Match Predictor** | Pick any two teams → ML probabilities + Elo comparison, confidence gauge, Poisson score matrix |
| 🏆 **Tournament Simulator** | Click "Simulate World Cup 2026" → full bracket with group standings, knockout match cards, champion banner |
| 📊 **Championship Odds** | Monte Carlo results: top-20 bar, all-stages grouped bar, confederation breakdown, Elo vs odds scatter |
| 🔍 **Data Insights** | Feature importance (LightGBM gain), goal trends over time, feature correlation heatmap, score distribution |

---

## Model

### Architecture

A **LightGBM** multi-class classifier trained on 23,958 international matches (2000–2026):

- **Target**: match result — `0` = Away Win, `1` = Draw, `2` = Home Win
- **Split**: time-based — Train 2000–2021 → Validate 2022–2023 → Test 2024–2026
- **Features (46 total)**:
  - Elo ratings and Elo difference
  - FIFA World Rankings (rank number + points) for both teams
  - 5-game and 10-game rolling form (weighted win rate)
  - 5-game and 10-game rolling goals for/against/differential averages
  - Head-to-head record (wins, draws, losses, goal averages, game count)
  - Tournament weight/tier, neutral venue flag, home advantage indicator
  - Days rest for each team

### Results

| Metric | Score |
|---|---|
| Test Accuracy | **57.28%** (random baseline: 33.3%) |
| F1 Macro | **0.5274** |
| Log Loss | 0.8893 |
| Best CV F1 | 0.5049 |

Test set covers Jan 2024 – Jun 2026 (most recent, highest-quality data).

### Two-tier prediction system

| Use case | Method |
|---|---|
| Match Predictor page | LightGBM ML model (accounts for home/away context) |
| Tournament simulation | Elo-based neutral-venue formula |

The Elo formula is used for simulation because the WC training data has a systematic label artifact: the "home team" field in historical WC data anti-correlates with team strength at neutral venues — the ML model learns this and inverts predictions. The Elo system is mathematically symmetric and works correctly at neutral venues.

---

## Monte Carlo Results (10,000 simulations)

> All 48 WC 2026 teams ranked by championship probability.

| Rank | Team | Champion % | Semifinal % |
|---|---|---|---|
| 1 | 🇪🇸 Spain | 8.4% | 20.9% |
| 2 | 🇦🇷 Argentina | 6.5% | 17.3% |
| 3 | 🇫🇷 France | 5.2% | 15.6% |
| 4 | 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England | 4.8% | 16.4% |
| 5 | 🇲🇦 Morocco | 4.6% | 15.7% |
| 6 | 🇧🇷 Brazil | 4.1% | 14.7% |
| 7 | 🇩🇪 Germany | 3.9% | 12.6% |
| 8 | 🇵🇹 Portugal | 3.7% | 13.0% |
| 9 | 🇪🇨 Ecuador | 3.5% | 12.8% |
| 10 | 🇳🇱 Netherlands | 3.4% | 12.2% |

Full results in `outputs/monte_carlo_results.csv`.

---

## Model Limitations

This is a research/portfolio project. Several important caveats apply before drawing real-world conclusions:

1. **Historical data ≠ future performance.** The model trains on 26 years of results. Player rosters, coaching staff, and team form change continuously; a team's 2025–2026 Elo may not reflect their true current strength.

2. **Elo neutral-venue artifact.** The ML model cannot reliably predict neutral-venue WC matches because the training data's home/away labels anti-correlate with team strength at WC finals. The simulation switches to an Elo model, which is theoretically correct but trades feature richness for consistency.

3. **FIFA ranking staleness.** FIFA ranking data was last downloaded through November 2019. Matches 2020–2026 use forward-filled last-known ranking points. Elo ratings are fully up-to-date (recomputed from all 49,000 matches) but FIFA rankings are not.

4. **Low draw classification.** The model has lower F1 on draws (~0.30) than wins and losses (~0.60). Draws are inherently unpredictable and are underrepresented in high-stakes matches. The Elo model uses a calibrated draw probability that decreases with Elo difference.

5. **10,000 simulations ≠ deterministic.** Monte Carlo results have sampling variance (~±0.3pp on champion probabilities at N=10,000). Running 100,000 simulations would reduce this further.

6. **No injury / suspension data.** The model has no awareness of player availability, which can dramatically affect real match outcomes.

7. **Group-stage format change.** WC 2026 introduces 12 groups of 4 (new format). The advancement rule (top 2 + best 8 third-place) is implemented correctly but has no historical training precedent.

---

## Resume Bullets

> Copy-paste ready for a CV or LinkedIn.

- Built an end-to-end FIFA World Cup 2026 prediction system in Python: ingested 49,000+ historical matches, engineered 46 features (Elo, rolling form, FIFA rankings, H2H), trained a LightGBM classifier achieving 57% accuracy (vs. 33% random baseline) on a held-out 2024–2026 test set
- Designed a full 48-team tournament simulator using Elo-based Poisson goal generation and FIFA tiebreaker rules; ran 10,000 Monte Carlo simulations in 7 seconds via matchup precomputation caching
- Deployed a 6-page Streamlit dashboard with interactive match prediction, live bracket simulation, championship odds charts, and Poisson score probability heatmaps
- Identified and resolved a critical ML model artifact (neutral-venue label inversion in WC training data) by building a two-tier prediction system separating individual match prediction (ML) from simulation (Elo)
- Implemented a robust Elo rating system across 259 national teams, handling 26+ years of match history with configurable K-factors by tournament weight

---

## Data Sources

| Dataset | Source |
|---|---|
| International results (49,000+ matches, 1872–2026) | [martj42/international-football-results](https://github.com/martj42/international_results) |
| FIFA World Rankings (2006–2019) | [samuraitruong/fifa-ranking-data](https://github.com/samuraitruong/fifa-ranking-data) |
| WC 2026 group draw | Official FIFA announcement, December 2024 |

---

## Requirements

Full list in `requirements.txt`. Key dependencies:

```
lightgbm>=4.3.0
scikit-learn>=1.4.2
pandas>=2.2.0
numpy>=1.26.0
scipy>=1.13.0
streamlit>=1.35.0
plotly>=5.21.0
statsmodels>=0.14.0
pyyaml>=6.0.0
joblib>=1.4.0
```

---

## License

MIT — free to use, modify, and distribute. See `LICENSE`.
