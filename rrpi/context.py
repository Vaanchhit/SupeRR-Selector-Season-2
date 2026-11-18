"""Contextual expected-performance benchmarks (the xG analogue for a delivery).

For every delivery we estimate what an *average* batter / bowler would produce in
exactly that situation (innings, over, wickets, scoring rate, chase equation,
competition tier, pace vs spin, handedness). The player is then credited with
runs-above-expected, with dismissals converted into runs via a data-derived
run-expectancy table (cricket's RE24): a wicket costs the expected runs the
batting side loses by having one fewer wicket in hand at that point.

Two benchmarks: `ctx` (situation only) and `role` (situation + batting position),
so the value of a role adjustment can be tested rather than assumed.
All fits are leave-one-season-out.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

CTX = ["innings", "over", "balls_left", "wkts_before", "crr", "rrr_f", "tier_c", "spin", "lhb"]


def _features(d: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=d.index)
    X["innings"] = d.innings
    X["over"] = d.over
    X["balls_left"] = d.balls_left
    X["wkts_before"] = d.wkts_before
    X["crr"] = d.crr
    X["rrr_f"] = d.rrr.fillna(-1).clip(upper=36)
    X["tier_c"] = d.tier_c
    X["spin"] = (d.pace_or_spin == "Spin").astype(int)
    X["lhb"] = (d.batting_type == "LHB").astype(int)
    X["bat_pos"] = d.batting_order.clip(upper=11)
    return X


def _loso(X, y, years, make):
    pred = np.empty(len(X))
    for yr in np.unique(years):
        te = years == yr
        mdl = make().fit(X[~te], y[~te])
        pred[te] = mdl.predict_proba(X[te])[:, 1] if hasattr(mdl, "predict_proba") else mdl.predict(X[te])
    return pred


def run_expectancy(d: pd.DataFrame, seed: int):
    s = d[(d.innings == 1) & d.legal & (d.total_balls == 120)]
    final = s.groupby("match_id").total_runs.transform("sum")
    # runs still to come from this state = innings total - runs already scored
    s = s.assign(rem=final - s.runs_before)
    mdl = HistGradientBoostingRegressor(max_iter=200, monotonic_cst=[1, 1], random_state=seed)
    mdl.fit(np.c_[s.balls_left, s.wkts_left], s.rem)
    return mdl


def annotate(d: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, dict]:
    d = d.copy()
    X = _features(d)
    yrs = d.year.to_numpy()
    reg = lambda: HistGradientBoostingRegressor(loss="poisson", max_iter=200, learning_rate=0.05,
                                                min_samples_leaf=300, random_state=seed)
    clf = lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, min_samples_leaf=300, random_state=seed)

    bf = d.bat_faced.to_numpy()
    for tag, cols in [("ctx", CTX), ("role", CTX + ["bat_pos"])]:
        d.loc[bf, f"xr_{tag}"] = _loso(X.loc[bf, cols], d.loc[bf, "runs"].to_numpy(), yrs[bf], reg)
        d.loc[bf, f"xo_{tag}"] = _loso(X.loc[bf, cols], d.loc[bf, "striker_out"].astype(int).to_numpy(), yrs[bf], clf)
    d["xc"] = _loso(X[CTX], d.bowler_runs.to_numpy(), yrs, reg)
    d["xw"] = _loso(X[CTX], d.bowler_wkt.astype(int).to_numpy(), yrs, clf)

    re = run_expectancy(d, seed)
    bl, wl = d.balls_left.to_numpy(), d.wkts_left.to_numpy()
    d["wkt_cost"] = np.clip(re.predict(np.c_[bl, wl]) - re.predict(np.c_[bl, np.maximum(wl - 1, 0)]), 0, None)

    out, so = d.striker_out.astype(int), d.bowler_wkt.astype(int)
    for tag in ["ctx", "role"]:
        d[f"rae_bat_{tag}"] = np.where(bf, (d.runs - d[f"xr_{tag}"]) - d.wkt_cost * (out - d[f"xo_{tag}"]), np.nan)
    d["rae_bowl"] = (d.xc - d.bowler_runs) + d.wkt_cost * (so - d.xw)

    grid = [{"balls_left": b, "wkts_left": w, "re": float(re.predict([[b, w]])[0])}
            for b in [120, 90, 60, 36, 24, 12, 6] for w in range(10, 0, -1)]
    diag = {
        "xr_bias": float((d.loc[bf, "runs"] - d.loc[bf, "xr_ctx"]).mean()),
        "xo_bias": float((out[bf] - d.loc[bf, "xo_ctx"]).mean()),
        "xc_bias": float((d.bowler_runs - d.xc).mean()),
        "re_grid": grid,
        "mean_wkt_cost": float(d.wkt_cost.mean()),
    }
    return d, diag
