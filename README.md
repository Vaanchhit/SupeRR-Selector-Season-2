# Pressure Intelligence: SupeRR Selector S2, Round 1

Ranks the batters and bowlers who perform most reliably under pressure, using only the competition data pack.
Built by Vaanchhit Agarwal.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m rrpi.stage1   # ~90 s: match state, win probability, leverage, expected runs
.venv/bin/python -m rrpi.stage2   # ~30 s: ratings, backtest, bootstrap, sensitivity, dashboard data
python3 -m http.server 5190 --directory dashboard   # open http://localhost:5190
```

The data pack CSVs must sit one folder up (`config.json → data_dir`). Every threshold and the random seed are set in `config.json`.

## Layout

| Path | Purpose |
|---|---|
| `rrpi/data.py` | Ingest, quality report, pre-delivery match-state reconstruction |
| `rrpi/winprob.py` | Monotone GBM win probability + Platt calibration (leave-one-season-out), leverage index, WPA attribution |
| `rrpi/context.py` | Expected-runs / expected-wicket benchmarks, run-expectancy wicket price |
| `rrpi/pressure.py` | Competing pressure definitions and their comparison |
| `rrpi/ratings.py` | Player × match table, six rating models, empirical-Bayes shrinkage, ensemble |
| `rrpi/validate.py` | Temporal backtest, family weights, ablation, split-half persistence, negative control, bootstrap |
| `rrpi/stage2.py` | Orchestration, archetypes, sensitivity grid, export |
| `outputs/bat_rankings_full.csv`, `outputs/bowl_rankings_full.csv` | Every qualified player with all component scores |
| `outputs/results.json` | Everything the dashboard shows |
| `dashboard/` | `index.html` + `data.js` (aggregated player-level outputs only, no raw deliveries) |

## Method in one paragraph

Pressure is defined as **leverage**: for each delivery, the expected absolute change in win probability over the possible outcomes, normalised so the average ball = 1. The top 25% of balls count as pressure. Players are scored on those balls by six models: conventional, contextual runs above expected, win-probability added, Bayesian WPA with a baseline-informed prior, consistency, and role-adjusted. Every rate is shrunk with empirical Bayes, using standard errors clustered by match. Near-duplicate models are grouped into families, each family gets an equal vote, and the final rank is the average over 300 match-level bootstrap resamples (bagging). "Clutch uplift" (pressure minus normal) is computed and shown, but it gets zero weight because it fails the shuffled-label negative control.

## Key honest results

- The win-probability model is calibrated within about 2–3 points per decile on unseen seasons (chase AUC 0.89).
- Clutch uplift is indistinguishable from noise for both batters and bowlers.
- Bowling pressure ratings predict future pressure performance modestly (ρ ≈ 0.15–0.20). Batting ratings barely do.
- A naive pressure-economy model forecasts bowlers about as well as the ensemble.
- Top-10 membership is uncertain. P(top-10) for each player is shown rather than hidden.

## Data-use note

The proprietary data pack is not redistributed. `dashboard/data.js` contains only player-level aggregates and the per-ball win-probability path for six featured matches.
