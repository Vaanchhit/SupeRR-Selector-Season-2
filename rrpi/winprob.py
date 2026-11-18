"""Win-probability engine, leverage index and win-probability-added attribution.

Design (borrowed from baseball's WPA / Leverage Index, Tango et al. *The Book*, 2007):
  * WP(s): probability the batting side wins from pre-delivery state s.
  * Leverage of a delivery = expected absolute WP swing over the possible outcomes
    of that delivery, normalised so the average delivery has LI = 1.
  * WPA of a delivery = WP(after) - WP(before), split between batter and bowler by
    applying only the events each is responsible for.

Models are gradient-boosted classifiers with monotonic constraints (more runs
needed can never raise a chasing side's WP, etc). They are fitted leave-one-season-out:
the WP used for a delivery in season Y comes from a model that never saw season Y,
so a match's own result cannot leak into its own state values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

TIER_CODE = {"IPL": 0, "T20I-FM": 1, "T20I-Assoc": 2, "A-team": 3}
# NB: raw `target` was tried as a feature and rejected — unconstrained, it let the model learn
# that low targets were "unwinnable" (116 off 110 with 9 wkts -> 13%). Required rate is monotone.
F2 = ["runs_needed", "balls_left", "wkts_left", "rrr_c", "tier_c"]
F1 = ["runs_before", "balls_left", "wkts_left", "tier_c"]
OUTCOMES = [(0, 0), (1, 0), (2, 0), (4, 0), (6, 0), (0, 1)]          # (runs, wicket)


def _clf(mono, seed):
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          monotonic_cst=mono, random_state=seed)


class WPEngine:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.m1: dict[int, HistGradientBoostingClassifier] = {}
        self.m2: dict[int, HistGradientBoostingClassifier] = {}

    def fit(self, d: pd.DataFrame) -> "WPEngine":
        d = d[d.bat_won != 0.5].assign(tier_c=d.tier.map(TIER_CODE),
                                       rrr_c=(6 * d.runs_needed / d.balls_left).clip(-6, 36))
        for y in sorted(d.year.unique()):
            tr = d[d.year != y]
            t1, t2 = tr[tr.innings == 1], tr[tr.innings == 2]
            self.m1[y] = _clf([1, 1, 1, 0], self.seed).fit(t1[F1], t1.bat_won.astype(int))
            self.m2[y] = _clf([-1, 1, 1, -1, 0], self.seed).fit(t2[F2], t2.bat_won.astype(int))
        # Platt calibration, also leave-one-season-out: season Y's calibrator is fitted on the
        # out-of-season predictions of every *other* season, so Y's outcomes never touch Y's WP.
        self.cal = {}
        for inn, models, F in [(1, self.m1, F1), (2, self.m2, F2)]:
            s = d[d.innings == inn]
            raw = self._p(models, s[F], s.year.to_numpy(), calibrate=False)
            lg = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4)))
            for y in models:
                tr = (s.year != y).to_numpy()
                self.cal[(inn, y)] = LogisticRegression().fit(lg[tr, None], s.bat_won.astype(int)[tr])
        return self

    def _p(self, models, X: pd.DataFrame, years: np.ndarray, calibrate: bool = True) -> np.ndarray:
        inn = 1 if models is self.m1 else 2
        out = np.empty(len(X))
        for y in np.unique(years):
            idx = years == y
            p = models[y].predict_proba(X[idx])[:, 1]
            if calibrate:
                p = np.clip(p, 1e-4, 1 - 1e-4)
                p = self.cal[(inn, y)].predict_proba(np.log(p / (1 - p))[:, None])[:, 1]
            out[idx] = p
        return out

    def wp(self, inn, runs, balls_used, wkts, target, total_balls, tier_c, year) -> np.ndarray:
        """Batting-side WP for arbitrary (possibly hypothetical/terminal) states. All args are arrays."""
        n = len(inn)
        out = np.full(n, np.nan)
        i2 = inn == 2
        won = i2 & (runs >= target)
        dead2 = i2 & ~won & ((wkts >= 10) | (balls_used >= total_balls))
        out[won] = 1.0
        out[dead2] = np.where(runs[dead2] == target[dead2] - 1, 0.5, 0.0)
        live2 = i2 & ~won & ~dead2
        if live2.any():
            rn, bl = target[live2] - runs[live2], total_balls[live2] - balls_used[live2]
            X = pd.DataFrame({"runs_needed": rn, "balls_left": bl, "wkts_left": 10 - wkts[live2],
                              "rrr_c": np.clip(6 * rn / bl, -6, 36), "tier_c": tier_c[live2]})
            out[live2] = self._p(self.m2, X, year[live2])
        i1 = inn == 1
        end1 = i1 & ((wkts >= 10) | (balls_used >= total_balls))
        if end1.any():   # innings over: convert to the chase starting state, from the batting-first side's view
            t = runs[end1] + 1
            X = pd.DataFrame({"runs_needed": t, "balls_left": np.full(end1.sum(), 120), "wkts_left": np.full(end1.sum(), 10),
                              "rrr_c": np.clip(6 * t / 120, -6, 36), "tier_c": tier_c[end1]})
            out[end1] = 1 - self._p(self.m2, X, year[end1])
        live1 = i1 & ~end1
        if live1.any():
            X = pd.DataFrame({"runs_before": runs[live1], "balls_left": total_balls[live1] - balls_used[live1],
                              "wkts_left": 10 - wkts[live1], "tier_c": tier_c[live1]})
            out[live1] = self._p(self.m1, X, year[live1])
        return out


def annotate(d: pd.DataFrame, eng: WPEngine) -> tuple[pd.DataFrame, dict]:
    d = d.copy()
    d["tier_c"] = d.tier.map(TIER_CODE).values
    A = {k: d[k].to_numpy() for k in ["innings", "runs_before", "balls_before", "wkts_before", "target", "total_balls", "tier_c", "year"]}
    A["target"] = np.nan_to_num(A["target"], nan=0)
    base = lambda dr, db, dw: eng.wp(A["innings"], A["runs_before"] + dr, A["balls_before"] + db, A["wkts_before"] + dw,
                                     A["target"], A["total_balls"], A["tier_c"], A["year"])
    d["wp"] = base(0, 0, 0)
    legal = d.legal.to_numpy().astype(int)
    d["wp_after"] = base(d.total_runs.to_numpy(), legal, d.is_wicket.to_numpy().astype(int))
    d["wpa_bat"] = base(d.runs.to_numpy(), legal, d.striker_out.to_numpy().astype(int)) - d.wp
    d["wpa_bowl"] = -(base(d.bowler_runs.to_numpy(), legal, d.bowler_wkt.to_numpy().astype(int)) - d.wp)
    d["wpa_total"] = d.wp_after - d.wp

    # Leverage: expected |dWP| over empirical outcome mix for this innings & phase (legal balls only).
    lg = d[d.legal]
    probs = {}
    for (inn, ph), grp in lg.groupby(["innings", "phase"]):
        cats = np.select([grp.is_wicket, grp.runs >= 6, grp.runs >= 4, grp.runs >= 2, grp.runs == 1], [5, 4, 3, 2, 1], 0)
        probs[(inn, ph)] = np.bincount(cats, minlength=6) / len(cats)
    P = np.vstack([probs.get(k, np.full(6, 1 / 6)) for k in zip(d.innings, d.phase)])
    swing = np.zeros(len(d))
    for j, (r, w) in enumerate(OUTCOMES):
        swing += P[:, j] * np.abs(base(r, 1, w) - d.wp.to_numpy())
    d["li"] = swing / swing.mean()

    # Calibration diagnostics (out-of-season predictions vs final result).
    v = d[(d.bat_won != 0.5)]
    bins = pd.cut(v.wp, np.linspace(0, 1, 11), include_lowest=True)
    cal = v.groupby(bins, observed=True).agg(pred=("wp", "mean"), obs=("bat_won", "mean"), n=("wp", "size")).reset_index(drop=True)
    diag = {
        "brier": float(brier_score_loss(v.bat_won, v.wp)),
        "brier_naive": float(brier_score_loss(v.bat_won, np.full(len(v), v.bat_won.mean()))),
        "logloss": float(log_loss(v.bat_won, v.wp.clip(1e-4, 1 - 1e-4))),
        "auc": float(roc_auc_score(v.bat_won, v.wp)),
        "auc_by_innings": {int(k): float(roc_auc_score(g.bat_won, g.wp)) for k, g in v.groupby("innings")},
        "calibration": cal.round(4).to_dict("records"),
    }
    return d, diag
