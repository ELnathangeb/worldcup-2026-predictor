# FIFA World Cup 2026 — Tournament Simulation Report

_Generated: 2026-06-08 23:06 · 10,000 Monte Carlo simulations_

---

## 1. Simulation Methodology

### 1.1 Match Prediction Engine

Match win/draw/loss probabilities are computed using a **two-tier system**:

**For the WC simulation (neutral venues):** Elo ratings drive the prediction.
The Elo-based formula is `P(A wins) = E_A × (1 − P(draw))` where
`E_A = 1 / (1 + 10^((Elo_B − Elo_A)/400))` is the standard Elo expected score and
`P(draw) = max(0.10, 0.27 − 0.0002 × |ΔElo|)` calibrated to WC history (27% base draw rate, decreasing with mismatch size).

*Why Elo instead of the ML model for WC neutral venues:*
In the raw historical dataset the "home team" label at WC neutral matches anti-correlates with Elo strength (the weaker team was listed as home 59% of the time when Elo gap ≥50 pts). The ML model learned this artifact and produced inverted predictions at neutral venues. Elo is symmetric, calibrated from all historical results, and the correct tool for neutral-venue tournament simulation.

**For the individual match API (`predict_match`):** The **tuned LightGBM pipeline** (Phase 3, test F1 = 0.527, accuracy = 57.3%) is used directly. It provides feature-rich predictions (form, H2H, rankings) for labelled home/away non-WC matches.

### 1.2 Score Generation (Poisson Model)

Exact scores are generated using a **calibrated Poisson model**:

1. **Baseline rates**: World Cup neutral-venue averages (1.32 and 1.10 goals/90 min).
2. **Strength scaling**: rates are scaled by the ratio of the team's predicted win probability to the historical baseline win rate, with a damping exponent of 0.35 to prevent extreme scorelines.
3. **Regression to mean**: 25% blend toward the baseline prevents 5–0 outcomes from near-certain predictions.
4. **Poisson sampling**: `goals ~ Poisson(λ)`, capped at 7 per team.

Example: Brazil (p_win=0.52) vs Ecuador (p_win=0.22) → λ_Brazil=1.45, λ_Ecuador=1.05

### 1.3 Knockout Tiebreaker (Extra Time + Penalties)

When a knockout match is level after 90 min:

- **Extra time**: additional 30 min simulated with scaled Poisson rates (×30/90).
- **Penalty shootout**: if still level, winner drawn with probability biased by Elo strength difference (clamped to 30%–70%).

### 1.4 Group Stage Tiebreaks (FIFA rules)

1. Points  2. Goal difference  3. Goals scored  4. H2H points  5. H2H goal difference  6. H2H goals scored  7. Random draw

### 1.5 Monte Carlo Setup

- **10,000 independent simulations** — each uses a fresh random seed.
- Each simulation is fully independent: group stage → R32 → R16 → QF → SF → Final.
- Probabilities are the fraction of simulations where each outcome occurred.

---

## 2. Championship Probabilities

| Rank | Team | Group | Champion% | Runner-Up% | Semifinal% | QF% | R16% | R32% |
|------|------|-------|-----------|------------|------------|-----|------|------|
| 1 | **Spain** | G | 8.4% | 13.0% | 20.9% | 33.3% | 52.1% | 82.0% |
| 2 | **Argentina** | E | 6.5% | 10.7% | 17.3% | 30.8% | 50.2% | 82.0% |
| 3 | **France** | H | 5.2% | 9.2% | 15.6% | 27.4% | 48.5% | 79.3% |
| 4 | **England** | I | 4.8% | 9.1% | 16.4% | 27.6% | 47.4% | 79.9% |
| 5 | **Morocco** | C | 4.6% | 8.3% | 15.7% | 26.4% | 44.8% | 75.4% |
| 6 | **Brazil** | D | 4.1% | 7.7% | 14.7% | 25.8% | 47.5% | 81.1% |
| 7 | **Germany** | F | 3.9% | 7.3% | 12.6% | 23.4% | 42.9% | 74.6% |
| 8 | **Portugal** | J | 3.7% | 6.9% | 13.0% | 24.1% | 44.4% | 79.2% |
| 9 | **Ecuador** | H | 3.5% | 6.8% | 12.8% | 23.7% | 44.6% | 76.8% |
| 10 | **Netherlands** | K | 3.4% | 6.3% | 12.2% | 22.8% | 41.5% | 81.2% |
| 11 | **Colombia** | F | 3.4% | 6.6% | 12.2% | 23.2% | 42.7% | 75.0% |
| 12 | **Mexico** | B | 3.3% | 6.9% | 13.7% | 25.8% | 45.1% | 81.8% |
| 13 | **Japan** | G | 3.2% | 6.6% | 12.3% | 22.6% | 40.1% | 68.2% |
| 14 | **Switzerland** | L | 3.1% | 6.4% | 11.7% | 22.0% | 42.3% | 78.7% |
| 15 | **Croatia** | C | 2.6% | 4.8% | 9.9% | 18.8% | 34.8% | 64.4% |
| 16 | **Senegal** | J | 2.2% | 4.7% | 9.5% | 19.7% | 38.9% | 74.8% |
| 17 | **Canada** | C | 2.1% | 4.0% | 8.8% | 16.7% | 32.5% | 62.7% |
| 18 | **Uruguay** | F | 2.1% | 4.5% | 8.7% | 17.8% | 35.7% | 67.9% |
| 19 | **Belgium** | C | 2.1% | 4.5% | 9.4% | 17.3% | 33.8% | 64.3% |
| 20 | **United States** | A | 2.0% | 4.2% | 9.4% | 19.1% | 37.8% | 79.0% |
| 21 | **Nigeria** | E | 1.9% | 4.2% | 8.3% | 17.2% | 34.5% | 66.7% |
| 22 | **South Korea** | G | 1.8% | 3.7% | 7.5% | 14.7% | 29.1% | 56.6% |
| 23 | **Australia** | I | 1.6% | 3.7% | 8.2% | 17.5% | 35.1% | 68.0% |
| 24 | **Algeria** | I | 1.6% | 3.7% | 8.6% | 17.7% | 35.1% | 67.4% |
| 25 | **Paraguay** | G | 1.6% | 3.3% | 7.3% | 15.7% | 31.7% | 59.8% |
| 26 | **Iran** | E | 1.4% | 3.4% | 7.1% | 15.5% | 32.1% | 65.3% |
| 27 | **Egypt** | D | 1.3% | 2.8% | 6.7% | 14.6% | 31.9% | 65.8% |
| 28 | **Ivory Coast** | K | 1.3% | 3.1% | 7.0% | 15.4% | 32.0% | 73.6% |
| 29 | **Panama** | A | 1.2% | 2.8% | 6.8% | 14.9% | 32.9% | 72.8% |
| 30 | **Uzbekistan** | L | 1.2% | 3.0% | 6.9% | 15.1% | 33.5% | 69.4% |
| 31 | **Greece** | L | 1.0% | 2.2% | 5.0% | 11.9% | 28.2% | 63.8% |
| 32 | **Venezuela** | B | 1.0% | 2.3% | 5.7% | 13.7% | 30.6% | 69.2% |


### Top 5 Favourites

```
  Spain                   ████████████████  8.4%
  Argentina               ████████████  6.5%
  France                  ██████████  5.2%
  England                 █████████  4.8%
  Morocco                 █████████  4.6%
```

---

## 3. Example Tournament Run (Seed 42)

Below is the result of one deterministic simulation to illustrate the output:

### 3.1 Group Stage Standings

**Group A**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | United States | 9 | 3 | 0 | 0 | 9 | 1 | +8 |
| 2 ✓ | Albania | 3 | 1 | 0 | 2 | 4 | 6 | -2 |
| 3 | Panama | 3 | 1 | 0 | 2 | 4 | 7 | -3 |
| 4 | Honduras | 3 | 1 | 0 | 2 | 4 | 7 | -3 |

**Group B**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Mexico | 6 | 2 | 0 | 1 | 4 | 1 | +3 |
| 2 ✓ | Venezuela | 4 | 1 | 1 | 1 | 4 | 3 | +1 |
| 3 * | Jamaica | 4 | 1 | 1 | 1 | 2 | 2 | +0 |
| 4 | New Zealand | 3 | 1 | 0 | 2 | 2 | 6 | -4 |

**Group C**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Canada | 7 | 2 | 1 | 0 | 4 | 2 | +2 |
| 2 ✓ | Croatia | 4 | 1 | 1 | 1 | 5 | 5 | +0 |
| 3 * | Belgium | 4 | 1 | 1 | 1 | 2 | 2 | +0 |
| 4 | Morocco | 1 | 0 | 1 | 2 | 1 | 3 | -2 |

**Group D**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | DR Congo | 6 | 2 | 0 | 1 | 3 | 3 | +0 |
| 2 ✓ | Egypt | 4 | 1 | 1 | 1 | 5 | 5 | +0 |
| 3 * | Serbia | 4 | 1 | 1 | 1 | 4 | 4 | +0 |
| 4 | Brazil | 3 | 1 | 0 | 2 | 2 | 2 | +0 |

**Group E**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Argentina | 6 | 2 | 0 | 1 | 5 | 4 | +1 |
| 2 ✓ | Nigeria | 6 | 2 | 0 | 1 | 3 | 2 | +1 |
| 3 * | Iran | 3 | 1 | 0 | 2 | 3 | 4 | -1 |
| 4 | Chile | 3 | 1 | 0 | 2 | 1 | 2 | -1 |

**Group F**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Germany | 9 | 3 | 0 | 0 | 8 | 2 | +6 |
| 2 ✓ | Colombia | 6 | 2 | 0 | 1 | 4 | 3 | +1 |
| 3 * | Cameroon | 3 | 1 | 0 | 2 | 4 | 6 | -2 |
| 4 | Uruguay | 0 | 0 | 0 | 3 | 2 | 7 | -5 |

**Group G**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Japan | 6 | 2 | 0 | 1 | 6 | 2 | +4 |
| 2 ✓ | Spain | 6 | 2 | 0 | 1 | 4 | 2 | +2 |
| 3 * | Paraguay | 6 | 2 | 0 | 1 | 4 | 2 | +2 |
| 4 | South Korea | 0 | 0 | 0 | 3 | 1 | 9 | -8 |

**Group H**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Saudi Arabia | 7 | 2 | 1 | 0 | 3 | 0 | +3 |
| 2 ✓ | Ecuador | 6 | 2 | 0 | 1 | 4 | 1 | +3 |
| 3 | South Africa | 2 | 0 | 2 | 1 | 1 | 4 | -3 |
| 4 | France | 1 | 0 | 1 | 2 | 1 | 4 | -3 |

**Group I**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | England | 9 | 3 | 0 | 0 | 7 | 1 | +6 |
| 2 ✓ | Algeria | 4 | 1 | 1 | 1 | 3 | 3 | +0 |
| 3 | Slovakia | 3 | 1 | 0 | 2 | 4 | 6 | -2 |
| 4 | Australia | 1 | 0 | 1 | 2 | 3 | 7 | -4 |

**Group J**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Portugal | 7 | 2 | 1 | 0 | 4 | 2 | +2 |
| 2 ✓ | Senegal | 6 | 2 | 0 | 1 | 4 | 2 | +2 |
| 3 * | Bolivia | 4 | 1 | 1 | 1 | 2 | 2 | +0 |
| 4 | Costa Rica | 0 | 0 | 0 | 3 | 1 | 5 | -4 |

**Group K**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Netherlands | 7 | 2 | 1 | 0 | 3 | 1 | +2 |
| 2 ✓ | Ivory Coast | 4 | 1 | 1 | 1 | 2 | 1 | +1 |
| 3 * | Qatar | 4 | 1 | 1 | 1 | 3 | 3 | +0 |
| 4 | Guatemala | 1 | 0 | 1 | 2 | 1 | 4 | -3 |

**Group L**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Greece | 7 | 2 | 1 | 0 | 8 | 2 | +6 |
| 2 ✓ | Uzbekistan | 7 | 2 | 1 | 0 | 4 | 1 | +3 |
| 3 | Switzerland | 1 | 0 | 1 | 2 | 1 | 3 | -2 |
| 4 | Peru | 1 | 0 | 1 | 2 | 2 | 9 | -7 |

### 3.2 Round of 32

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| United States | 1–1 (pens) | Paraguay | **Paraguay** *(pens)* |
| Mexico | 3–0 | Serbia | **Mexico** |
| Canada | 4–1 | Qatar | **Canada** |
| DR Congo | 1–2 | Jamaica | **Jamaica** |
| Argentina | 2–3 (aet) | Belgium | **Belgium** *(aet)* |
| Germany | 3–1 | Bolivia | **Germany** |
| Japan | 1–1 (pens) | Iran | **Iran** *(pens)* |
| Saudi Arabia | 2–1 | Cameroon | **Saudi Arabia** |
| England | 1–1 (pens) | Albania | **England** *(pens)* |
| Portugal | 2–1 | Venezuela | **Portugal** |
| Netherlands | 1–0 (aet) | Croatia | **Netherlands** *(aet)* |
| Greece | 3–1 | Egypt | **Greece** |
| Nigeria | 3–2 | Algeria | **Nigeria** |
| Colombia | 2–2 (pens) | Senegal | **Colombia** *(pens)* |
| Spain | 3–3 (pens) | Ivory Coast | **Ivory Coast** *(pens)* |
| Ecuador | 0–1 | Uzbekistan | **Uzbekistan** |

### 3.3 Round of 16

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Paraguay | 1–1 (pens) | Mexico | **Paraguay** *(pens)* |
| Canada | 0–1 (aet) | Jamaica | **Jamaica** *(aet)* |
| Belgium | 1–5 | Germany | **Germany** |
| Iran | 0–1 (aet) | Saudi Arabia | **Saudi Arabia** *(aet)* |
| England | 1–3 (aet) | Portugal | **Portugal** *(aet)* |
| Netherlands | 0–1 | Greece | **Greece** |
| Nigeria | 2–0 | Colombia | **Nigeria** |
| Ivory Coast | 0–3 | Uzbekistan | **Uzbekistan** |

### 3.4 Quarterfinals

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Paraguay | 1–1 (pens) | Jamaica | **Jamaica** *(pens)* |
| Germany | 1–1 (pens) | Saudi Arabia | **Germany** *(pens)* |
| Portugal | 2–1 (aet) | Greece | **Portugal** *(aet)* |
| Nigeria | 0–1 | Uzbekistan | **Uzbekistan** |

### 3.5 Semifinals

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Jamaica | 0–1 | Germany | **Germany** |
| Portugal | 0–1 | Uzbekistan | **Uzbekistan** |

### 3.6 Final

| Germany | **3–2** | Uzbekistan |
|---|---|---|
| 🏆 **CHAMPION** | |  |

**🏆 Champion: Germany**  
**🥈 Runner-up: Uzbekistan**

---

## 4. Visualisations

Charts saved to `reports/figures/`:

| File | Description |
|------|-------------|
| `championship_probabilities.png` | Top 20 teams by championship % |
| `semifinal_probabilities.png`    | Top 20 teams by semifinal % |
| `advancement_odds.png`           | Stacked advancement bar chart |
| `team_strength_map.png`          | Bubble chart: R32% vs SF%, size=champion% |
| `group_advancement_heatmap.png`  | Heatmap of group advancement by team |

---

## 5. Model Limitations & Caveats

| # | Limitation | Impact |
|---|-----------|--------|
| 1 | **Model accuracy 57.3%** — soccer outcomes are inherently uncertain | Probabilities are calibrated estimates, not certainties |
| 2 | **No squad/injury data** — key absences (e.g., star player suspended) are invisible | Upsets may be under/over-estimated |
| 3 | **FIFA rankings frozen at Nov 2019** — rank features are forward-filled | Elo (fully current) is the primary strength signal |
| 4 | **Draw prediction recall ≈ 34%** — draws are inherently hard to predict | Score outcomes have higher variance than outcome probabilities |
| 5 | **Poisson independence assumption** — goals for each team are drawn independently | Under-represents 0-0 and 1-1 draws (Dixon-Coles correction not applied) |
| 6 | **WC format novelty** — 48-team format with 12 groups is new in 2026 | No historical WC data in this exact format |

---

_Simulation report generated automatically by `src/monte_carlo.py` on 2026-06-08 23:06_