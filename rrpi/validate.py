"""Backtesting, ensemble-weight fitting, ablation, sensitivity, persistence and negative controls."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import spearmanr

from .ratings import BAT_MODELS, BOWL_MODELS, score


def _target(pm, years, min_balls, col="wpa"):
    t = pm[pm.P & pm.year.isin(years)].groupby("player")[[col, "n"]].sum()
    t = t[t.n >= min_balls]
    return t[col] / t.n


def _naive(pm, years, kind):
    s = pm[pm.P & pm.year.isin(years)].groupby("player")[["runs", "n"]].sum()
    return s.runs / s.n * (1 if kind == "bat" else -1)          # pressure SR or (minus) pressure economy


def _overall(pm, years):
    s = pm[pm.year.isin(years)].groupby("player")[["wpa", "n"]].sum()
    return s.wpa / s.n


def family_weights(S, models, exclude=("R",), thr=0.85):
    """Give each *family* of near-duplicate models one vote (union-find on Spearman >= thr).
    Stops five variants of one formula outvoting one genuinely different method."""
    keep = [m for m in models if m not in exclude]
    C = S[keep].corr(method="spearman")
    parent = {m: m for m in keep}
    find = lambda m: m if parent[m] == m else find(parent[m])
    for i, a in enumerate(keep):
        for b in keep[i + 1:]:
            if C.loc[a, b] >= thr:
                parent[find(b)] = find(a)
    fam = {}
    for m in keep:
        fam.setdefault(find(m), []).append(m)
    w = {m: 0.0 for m in models}
    for members in fam.values():
        for m in members:
            w[m] = 1 / (len(fam) * len(members))
    return w, [sorted(v) for v in fam.values()]


def backtest(pm, kind, cfg):
    v, el = cfg["validation"], dict(cfg["eligibility"])
    small = {k: (30 if "balls" in k else 5) for k in el}            # fewer seasons in-window -> lower bar
    models = BAT_MODELS if kind == "bat" else BOWL_MODELS
    res = {"models": models}

    def window(rate_years, tgt_years, weights=None):
        S = score(pm[pm.year.isin(rate_years)], kind, small, weights)
        tgt = _target(pm, tgt_years, v["min_target_pressure_balls"])
        tgt_b = _target(pm, tgt_years, v["min_target_pressure_balls"], "rae")
        common = S.index.intersection(tgt.index)
        return S.loc[common], tgt.loc[common], tgt_b.loc[common], common

    # 1) fit ensemble weights: ratings 2022-23 -> target 2024 (non-negative stacking on z-scores)
    S, y, yb, _ = window(v["weight_fit"]["ratings"], v["weight_fit"]["target"])
    Z = S[[f"z_{m}" for m in models]].to_numpy()
    yz = ((y - y.mean()) / y.std()).to_numpy()
    w, _ = nnls(Z, yz)
    w = w / w.sum() if w.sum() > 0 else np.ones(len(models)) / len(models)
    res["fit_window"] = {"n_players": int(len(y)),
                         "spearman": {m: float(spearmanr(S[m], y).statistic) for m in models}}
    res["nnls_weights"] = dict(zip(models, w.round(4).tolist()))

    # 2) clean holdout: ratings 2022-24 -> target 2025-26, weights frozen from step 1
    eq = {m: 1.0 for m in models}
    wd = dict(zip(models, w))
    S_eq, y2, yb2, common = window(v["holdout"]["ratings"], v["holdout"]["target"], eq)
    S_w = score(pm[pm.year.isin(v["holdout"]["ratings"])], kind, small, wd).loc[common]
    hy = v["holdout"]["ratings"]
    wf, fams = family_weights(score(pm[pm.year.isin(hy)], kind, small), models)
    S_f = score(pm[pm.year.isin(hy)], kind, small, wf).loc[common]
    naive = _naive(pm, hy, kind).reindex(common)
    overall = _overall(pm, hy).reindex(common)
    sp = lambda a, b: float(spearmanr(a, b, nan_policy="omit").statistic)
    rows = [{"model": m, "target_wpa": sp(S_eq[m], y2), "target_xruns": sp(S_eq[m], yb2)} for m in models]
    rows += [
        {"model": "Ensemble (family-balanced, final)", "target_wpa": sp(S_f.ens, y2), "target_xruns": sp(S_f.ens, yb2)},
        {"model": "Ensemble (stacked weights)", "target_wpa": sp(S_w.ens, y2), "target_xruns": sp(S_w.ens, yb2)},
        {"model": "Ensemble (equal weights)", "target_wpa": sp(S_eq.ens, y2), "target_xruns": sp(S_eq.ens, yb2)},
        {"model": "Naive pressure " + ("strike rate" if kind == "bat" else "economy"), "target_wpa": sp(naive, y2), "target_xruns": sp(naive, yb2)},
        {"model": "Overall WPA rate (no pressure split)", "target_wpa": sp(overall, y2), "target_xruns": sp(overall, yb2)},
    ]
    # bootstrap CI over players for the two ensembles
    rng = np.random.default_rng(cfg["seed"])
    for r in rows[-5:-2]:
        e = S_w.ens if "stacked" in r["model"] else (S_f.ens if "family" in r["model"] else S_eq.ens)
        bs = []
        for _ in range(500):
            i = rng.integers(0, len(common), len(common))
            bs.append(spearmanr(e.iloc[i], y2.iloc[i]).statistic)
        r["ci90"] = [float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95))]
    res["holdout"] = {"n_players": int(len(common)), "rows": rows}

    # 3) ablation on the holdout (final family weights, drop one model at a time)
    wfull, fams_full = family_weights(score(pm, kind, cfg["eligibility"]), models)
    res["families"], res["final_weights"] = fams_full, wfull
    abl = []
    full_top = set(score(pm, kind, cfg["eligibility"], wfull).head(10).index)
    for m in models:
        if wfull.get(m, 0) == 0:
            continue
        w2 = {k: (0 if k == m else val) for k, val in wfull.items()}
        if sum(w2.values()) == 0:
            continue
        hs = score(pm[pm.year.isin(hy)], kind, small, w2).loc[common]
        ft = set(score(pm, kind, cfg["eligibility"], w2).head(10).index)
        abl.append({"dropped": m, "holdout_spearman": sp(hs.ens, y2), "holdout_spearman_xruns": sp(hs.ens, yb2),
                    "top10_overlap": len(ft & full_top)})
    res["ablation"] = abl
    return wfull, res


def persistence(pm, kind, elig, P_label="pressure"):
    """Odd/even-match split-half reliability of absolute pressure skill vs clutch uplift."""
    odd = pm.match_id % 2 == 1
    half = {k: score(pm[m], kind, {kk: v // 2 for kk, v in elig.items()}, detail=True) for k, m in [("odd", odd), ("even", ~odd)]}
    c = half["odd"].index.intersection(half["even"].index)
    out = {"n_players": int(len(c))}
    for lbl, col in [("abs_pressure_xruns", "raw_B"), ("abs_pressure_wpa", "raw_C"), ("clutch_uplift", "raw_R"),
                     ("baseline_wpa", "base_wpa_rate")]:
        r = float(spearmanr(half["odd"].loc[c, col], half["even"].loc[c, col]).statistic)
        out[lbl] = {"split_half_r": r, "spearman_brown": float(2 * r / (1 + r)) if r > -1 else None}
    return out


def negative_control(d, P, kind, elig, reps=40, seed=0):
    """Shuffle the pressure label *within each player* (keeping their pressure-ball count).
    If real 'clutch uplift' spread is no wider than shuffled spread, uplift is noise."""
    from .ratings import player_match
    rng = np.random.default_rng(seed)
    key = d.striker_id if kind == "bat" else d.bowler_id
    mask = d.bat_faced if kind == "bat" else pd.Series(True, index=d.index)
    real = score(player_match(d, P, kind), kind, elig, detail=True)
    pool = real.index
    sd_real = float(real.raw_R.std())
    sds, tops = [], []
    idx = d.index[mask & key.isin(pool)]
    k = key[idx].to_numpy()
    p = P[idx].to_numpy()
    order = np.argsort(k, kind="stable")
    for _ in range(reps):
        p2 = p.copy()
        ks = k[order]
        bounds = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1], True])
        sh = p2[order]
        for a, b in zip(bounds[:-1], bounds[1:]):
            sh[a:b] = rng.permutation(sh[a:b])
        p2[order] = sh
        Pp = P.copy()
        Pp[idx] = p2
        s = score(player_match(d, Pp, kind), kind, elig, detail=True, fixed_pool=pool)
        sds.append(float(s.raw_R.std()))
        tops.append(float(s.raw_R.sort_values(ascending=False).head(10).mean()))
    return {"sd_uplift_real": sd_real, "sd_uplift_shuffled_mean": float(np.mean(sds)),
            "sd_uplift_shuffled_p95": float(np.quantile(sds, 0.95)),
            "top10_uplift_real": float(real.raw_R.sort_values(ascending=False).head(10).mean()),
            "top10_uplift_shuffled_mean": float(np.mean(tops)),
            "top10_uplift_shuffled_p95": float(np.quantile(tops, 0.95)), "reps": reps}


def bootstrap(pm, kind, elig, weights, pool, reps, seed):
    rng = np.random.default_rng(seed)
    matches = pm.match_id.unique()
    ranks, ens = [], []
    for _ in range(reps):
        wm = pd.Series(rng.poisson(1.0, len(matches)), index=matches)
        s = score(pm, kind, elig, weights, w_match=wm, fixed_pool=pool)
        ranks.append(s["rank"].reindex(pool))
        ens.append(s["ens"].reindex(pool))
    R, E = pd.concat(ranks, axis=1), pd.concat(ens, axis=1)
    # Bagged score (Breiman 1996): averaging the ranking procedure over match resamples
    # reduces the variance of an unstable top-k selection.
    return pd.DataFrame({"bagged": E.mean(axis=1), "bagged_sd": E.std(axis=1),
                         "rank_med": R.median(axis=1), "rank_lo": R.quantile(0.1, axis=1), "rank_hi": R.quantile(0.9, axis=1),
                         "p_top10": (R <= 10).mean(axis=1), "p_top3": (R <= 3).mean(axis=1)})
