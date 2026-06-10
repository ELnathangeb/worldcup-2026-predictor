# FIFA World Cup 2026 — Tournament Simulation Report

_Generated: 2026-06-10 14:16 · 10,000 Monte Carlo simulations_

---

## 1. Simulation Methodology

### 1.1 Match Prediction Engine

Each match is predicted using the **tuned LightGBM pipeline** trained in Phase 3 (test F1-macro = 0.527, accuracy = 57.3%). The model outputs three probabilities:

- `p(Home Win)` → remapped as `p(Team A win)` at neutral venue
- `p(Draw)`
- `p(Away Win)` → remapped as `p(Team B win)`

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
| 1 | **Spain** | G | 26.9% | 35.3% | 46.0% | 63.0% | 83.3% | 96.3% |
| 2 | **Argentina** | E | 13.8% | 21.8% | 31.3% | 54.7% | 77.2% | 97.1% |
| 3 | **France** | H | 8.3% | 15.4% | 24.1% | 40.4% | 75.9% | 97.0% |
| 4 | **England** | I | 6.8% | 15.4% | 27.4% | 45.3% | 68.9% | 93.7% |
| 5 | **Morocco** | C | 5.2% | 10.5% | 22.0% | 35.5% | 56.3% | 84.6% |
| 6 | **Brazil** | D | 5.1% | 10.2% | 24.4% | 40.0% | 66.6% | 95.1% |
| 7 | **Portugal** | J | 4.1% | 9.9% | 19.6% | 35.7% | 62.7% | 94.5% |
| 8 | **Germany** | F | 3.6% | 7.4% | 13.8% | 31.1% | 58.3% | 87.0% |
| 9 | **Colombia** | F | 3.6% | 8.0% | 15.1% | 32.3% | 59.9% | 87.9% |
| 10 | **Ecuador** | H | 3.4% | 8.4% | 16.3% | 31.3% | 65.0% | 93.5% |
| 11 | **Netherlands** | K | 2.6% | 7.3% | 15.4% | 29.3% | 50.4% | 97.3% |
| 12 | **Japan** | G | 2.6% | 7.1% | 13.8% | 27.2% | 50.3% | 74.8% |
| 13 | **Mexico** | B | 2.4% | 6.4% | 18.6% | 35.7% | 58.9% | 96.4% |
| 14 | **Switzerland** | L | 2.1% | 5.6% | 13.1% | 25.9% | 51.9% | 91.1% |
| 15 | **Croatia** | C | 1.0% | 2.9% | 8.5% | 17.7% | 34.4% | 65.8% |
| 16 | **Belgium** | C | 1.0% | 2.7% | 7.6% | 16.6% | 32.6% | 62.7% |
| 17 | **Uruguay** | F | 0.8% | 2.6% | 6.6% | 16.9% | 38.2% | 71.6% |
| 18 | **Senegal** | J | 0.8% | 2.9% | 7.7% | 19.4% | 43.2% | 86.2% |
| 19 | **United States** | A | 0.8% | 2.1% | 7.9% | 20.1% | 42.8% | 91.5% |
| 20 | **Canada** | C | 0.6% | 2.1% | 6.5% | 14.7% | 29.7% | 60.0% |
| 21 | **Australia** | I | 0.6% | 2.0% | 6.0% | 16.0% | 36.2% | 73.0% |
| 22 | **Nigeria** | E | 0.6% | 2.0% | 5.4% | 14.7% | 33.7% | 69.8% |
| 23 | **Iran** | E | 0.5% | 1.5% | 4.2% | 12.1% | 29.6% | 64.4% |
| 24 | **Algeria** | I | 0.5% | 1.8% | 5.8% | 14.9% | 34.5% | 72.3% |
| 25 | **Paraguay** | G | 0.5% | 1.3% | 3.7% | 9.5% | 23.1% | 47.3% |
| 26 | **South Korea** | G | 0.4% | 1.3% | 3.2% | 8.7% | 22.6% | 45.7% |
| 27 | **Ivory Coast** | K | 0.4% | 1.3% | 4.2% | 11.9% | 30.1% | 88.3% |
| 28 | **Egypt** | D | 0.2% | 0.9% | 3.5% | 10.1% | 29.6% | 69.2% |
| 29 | **Uzbekistan** | L | 0.2% | 1.0% | 3.6% | 10.6% | 29.1% | 75.0% |
| 30 | **Venezuela** | B | 0.1% | 0.5% | 2.3% | 8.1% | 25.2% | 76.5% |
| 31 | **Panama** | A | 0.1% | 0.8% | 3.3% | 10.7% | 29.3% | 81.7% |
| 32 | **Greece** | L | 0.1% | 0.2% | 1.6% | 5.5% | 19.4% | 61.6% |


### Top 5 Favourites

```
  Spain                   █████████████████████████████████████████████████████  26.9%
  Argentina               ███████████████████████████  13.8%
  France                  ████████████████  8.3%
  England                 █████████████  6.8%
  Morocco                 ██████████  5.2%
```

---

## 3. Example Tournament Run (Seed 42)

Below is the result of one deterministic simulation to illustrate the output:

### 3.1 Group Stage Standings

**Group A**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Panama | 9 | 3 | 0 | 0 | 6 | 1 | +5 |
| 2 ✓ | United States | 6 | 2 | 0 | 1 | 8 | 4 | +4 |
| 3 | Honduras | 3 | 1 | 0 | 2 | 2 | 5 | -3 |
| 4 | Albania | 0 | 0 | 0 | 3 | 0 | 6 | -6 |

**Group B**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Mexico | 7 | 2 | 1 | 0 | 6 | 4 | +2 |
| 2 ✓ | Venezuela | 5 | 1 | 2 | 0 | 6 | 5 | +1 |
| 3 * | Jamaica | 3 | 1 | 0 | 2 | 3 | 4 | -1 |
| 4 | New Zealand | 1 | 0 | 1 | 2 | 1 | 3 | -2 |

**Group C**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Morocco | 6 | 2 | 0 | 1 | 4 | 1 | +3 |
| 2 ✓ | Belgium | 6 | 2 | 0 | 1 | 3 | 3 | +0 |
| 3 * | Croatia | 4 | 1 | 1 | 1 | 3 | 3 | +0 |
| 4 | Canada | 1 | 0 | 1 | 2 | 1 | 4 | -3 |

**Group D**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Brazil | 6 | 2 | 0 | 1 | 5 | 1 | +4 |
| 2 ✓ | Egypt | 5 | 1 | 2 | 0 | 3 | 2 | +1 |
| 3 * | DR Congo | 4 | 1 | 1 | 1 | 7 | 4 | +3 |
| 4 | Serbia | 1 | 0 | 1 | 2 | 1 | 9 | -8 |

**Group E**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Argentina | 9 | 3 | 0 | 0 | 9 | 0 | +9 |
| 2 ✓ | Iran | 6 | 2 | 0 | 1 | 5 | 4 | +1 |
| 3 | Nigeria | 1 | 0 | 1 | 2 | 1 | 6 | -5 |
| 4 | Chile | 1 | 0 | 1 | 2 | 1 | 6 | -5 |

**Group F**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Germany | 6 | 2 | 0 | 1 | 5 | 3 | +2 |
| 2 ✓ | Colombia | 6 | 2 | 0 | 1 | 3 | 3 | +0 |
| 3 * | Cameroon | 4 | 1 | 1 | 1 | 3 | 3 | +0 |
| 4 | Uruguay | 1 | 0 | 1 | 2 | 1 | 3 | -2 |

**Group G**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Japan | 7 | 2 | 1 | 0 | 7 | 4 | +3 |
| 2 ✓ | Spain | 6 | 2 | 0 | 1 | 5 | 3 | +2 |
| 3 * | South Korea | 3 | 1 | 0 | 2 | 3 | 4 | -1 |
| 4 | Paraguay | 1 | 0 | 1 | 2 | 1 | 5 | -4 |

**Group H**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Ecuador | 7 | 2 | 1 | 0 | 5 | 1 | +4 |
| 2 ✓ | France | 6 | 2 | 0 | 1 | 2 | 2 | +0 |
| 3 | South Africa | 2 | 0 | 2 | 1 | 3 | 4 | -1 |
| 4 | Saudi Arabia | 1 | 0 | 1 | 2 | 2 | 5 | -3 |

**Group I**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | England | 7 | 2 | 1 | 0 | 6 | 2 | +4 |
| 2 ✓ | Algeria | 4 | 1 | 1 | 1 | 2 | 2 | +0 |
| 3 * | Slovakia | 4 | 1 | 1 | 1 | 5 | 6 | -1 |
| 4 | Australia | 1 | 0 | 1 | 2 | 2 | 5 | -3 |

**Group J**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Portugal | 7 | 2 | 1 | 0 | 5 | 2 | +3 |
| 2 ✓ | Senegal | 7 | 2 | 1 | 0 | 4 | 2 | +2 |
| 3 | Costa Rica | 1 | 0 | 1 | 2 | 0 | 2 | -2 |
| 4 | Bolivia | 1 | 0 | 1 | 2 | 0 | 3 | -3 |

**Group K**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Netherlands | 9 | 3 | 0 | 0 | 4 | 0 | +4 |
| 2 ✓ | Guatemala | 3 | 1 | 0 | 2 | 2 | 2 | +0 |
| 3 * | Qatar | 3 | 1 | 0 | 2 | 2 | 3 | -1 |
| 4 | Ivory Coast | 3 | 1 | 0 | 2 | 1 | 4 | -3 |

**Group L**

| Pos | Team | Pts | W | D | L | GF | GA | GD |
|-----|------|-----|---|---|---|----|----|-----|
| 1 ✓ | Uzbekistan | 7 | 2 | 1 | 0 | 6 | 2 | +4 |
| 2 ✓ | Switzerland | 4 | 1 | 1 | 1 | 4 | 4 | +0 |
| 3 * | Greece | 3 | 0 | 3 | 0 | 5 | 5 | +0 |
| 4 | Peru | 1 | 0 | 1 | 2 | 3 | 7 | -4 |

### 3.2 Round of 32

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Panama | 0–1 | DR Congo | **DR Congo** |
| Mexico | 1–0 | Cameroon | **Mexico** |
| Morocco | 6–0 | Croatia | **Morocco** |
| Brazil | 1–0 | Slovakia | **Brazil** |
| Argentina | 1–1 (pens) | Greece | **Greece** *(pens)* |
| Germany | 1–0 | South Korea | **Germany** |
| Japan | 3–0 | Jamaica | **Japan** |
| Ecuador | 2–0 | Qatar | **Ecuador** |
| England | 1–0 | United States | **England** |
| Portugal | 2–1 | Venezuela | **Portugal** |
| Netherlands | 0–1 | Belgium | **Belgium** |
| Uzbekistan | 4–1 | Egypt | **Uzbekistan** |
| Iran | 0–3 | Algeria | **Algeria** |
| Colombia | 1–0 | Senegal | **Colombia** |
| Spain | 3–1 | Guatemala | **Spain** |
| France | 2–4 | Switzerland | **Switzerland** |

### 3.3 Round of 16

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| DR Congo | 0–2 | Mexico | **Mexico** |
| Morocco | 1–2 | Brazil | **Brazil** |
| Greece | 1–1 (pens) | Germany | **Germany** *(pens)* |
| Japan | 1–3 | Ecuador | **Ecuador** |
| England | 4–1 | Portugal | **England** |
| Belgium | 1–0 | Uzbekistan | **Belgium** |
| Algeria | 1–0 | Colombia | **Algeria** |
| Spain | 1–0 | Switzerland | **Spain** |

### 3.4 Quarterfinals

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Mexico | 1–2 | Brazil | **Brazil** |
| Germany | 1–2 | Ecuador | **Ecuador** |
| England | 2–1 | Belgium | **England** |
| Algeria | 1–1 (pens) | Spain | **Algeria** *(pens)* |

### 3.5 Semifinals

| Team A | Score | Team B | Winner |
|--------|-------|--------|--------|
| Brazil | 2–0 | Ecuador | **Brazil** |
| England | 1–0 | Algeria | **England** |

### 3.6 Final

| Brazil | **0–1** | England |
|---|---|---|
|  | | 🏆 **CHAMPION** |

**🏆 Champion: England**  
**🥈 Runner-up: Brazil**

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

_Simulation report generated automatically by `src/monte_carlo.py` on 2026-06-10 14:16_