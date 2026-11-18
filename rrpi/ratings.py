"""Player rating models and the ensemble.

The atomic table is player x match x pressure-flag sums, which makes the match
the unit of statistical independence (standard errors are cluster-robust by
match) and makes match-level bootstrap a cheap re-weighting.

Models (each a different *idea* of what "delivering under pressure" means):
  A  conventional  pressure runs with dismissals priced at the pressure-league runs-per-wicket
  B  contextual    runs above an average player's expectation in the same situation (per ball)
  C  leverage      win-probability added per pressure ball
  D  bayesian      C, shrunk toward a prior predicted from the player's own non-pressure WPA
  E  consistency   share of pressure innings / spells that beat expectation (beta-binomial)
  G  role          B but the benchmark also knows the batting position (batters only)
  R  relative      pressure minus non-pressure contextual rate ("clutch uplift")
Every rate (A,B,C,G,R) is empirical-Bayes shrunk (normal-normal, method of moments).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BAT_MODELS = ["A", "B", "C", "D", "E", "G", "R"]
BOWL_MODELS = ["A", "B", "C", "D", "E", "R"]
MODEL_NAMES = {"A": "Conventional", "B": "Contextual xRuns", "C": "Win-prob added", "D": "Bayesian WPA",
               "E": "Consistency", "G": "Role-adjusted", "R": "Clutch uplift"}


def player_match(d: pd.DataFrame, P: pd.Series, kind: str) -> pd.DataFrame:
    if kind == "bat":
        s = d[d.bat_faced]
        t = pd.DataFrame({
            "player": s.striker_id, "match_id": s.match_id, "year": s.year, "tier": s.tier, "P": P[s.index],
            "n": 1, "runs": s.runs, "out": s.striker_out.astype(int), "rae": s.rae_bat_ctx, "rae_role": s.rae_bat_role,
            "wpa": s.wpa_bat * 100, "four": s.is_four.astype(int), "six": s.is_six.astype(int),
            "dot": (s.runs == 0).astype(int), "li": s.li, "pos": s.batting_order,
            "death": (s.phase == "Death").astype(int), "spin": (s.pace_or_spin == "Spin").astype(int),
            "spin_runs": s.runs * (s.pace_or_spin == "Spin"),
        })
    else:
        s = d
        t = pd.DataFrame({
            "player": s.bowler_id, "match_id": s.match_id, "year": s.year, "tier": s.tier, "P": P[s.index],
            "n": s.legal.astype(int), "deliv": 1, "runs": s.bowler_runs, "out": s.bowler_wkt.astype(int),
            "rae": s.rae_bowl, "wpa": s.wpa_bowl * 100, "dot": ((s.total_runs == 0) & s.legal).astype(int),
            "bnd": (s.runs >= 4).astype(int), "li": s.li, "death": (s.phase == "Death") & s.legal,
            "pp": (s.phase == "Powerplay") & s.legal, "spin": (s.pace_or_spin == "Spin").astype(int),
        })
    return t.groupby(["player", "match_id", "year", "tier", "P"], as_index=False).sum(numeric_only=True)


def _rate_se(df: pd.DataFrame, x: str, w: np.ndarray | None = None) -> pd.DataFrame:
    """Ratio estimator sum(x)/sum(n) per player with match-clustered standard error."""
    w = np.ones(len(df)) if w is None else w
    X = (df[x] * w).groupby(df.player).sum()
    N = (df.n * w).groupby(df.player).sum()
    M = pd.Series(w, index=df.index).groupby(df.player).sum()
    r = X / N.replace(0, np.nan)
    e = df[x] - df.player.map(r) * df.n
    v = (w * e ** 2).groupby(df.player).sum() / N ** 2 * (M / (M - 1).clip(lower=0.5))
    return pd.DataFrame({"r": r, "se2": v.clip(lower=1e-9)})


def eb_shrink(r: pd.Series, se2: pd.Series, prior_mean: pd.Series | float | None = None) -> tuple[pd.Series, pd.Series, dict]:
    ok = r.notna() & se2.notna()
    r, se2 = r[ok], se2[ok]
    if prior_mean is None:
        mu = np.average(r, weights=1 / se2)
        resid = r - mu
    else:
        mu = prior_mean.reindex(r.index) if isinstance(prior_mean, pd.Series) else prior_mean
        resid = r - mu
    tau2 = max(float(resid.var() - se2.mean()), float(resid.var()) * 0.05)
    B = se2 / (se2 + tau2)
    post = (1 - B) * r + B * mu
    return post, np.sqrt((1 - B) * se2), {"tau2": tau2, "mean_shrink": float(B.mean())}


def _beta_binom(k: pd.Series, m: pd.Series) -> pd.Series:
    p = (k / m).dropna()
    mu, var = p.mean(), p.var()
    common = max(mu * (1 - mu) / max(var, 1e-6) - 1, 1.0)
    a, b = mu * common, (1 - mu) * common
    return (k + a) / (m + a + b)


def score(pm: pd.DataFrame, kind: str, elig: dict, weights: dict[str, float] | None = None,
          w_match: pd.Series | None = None, fixed_pool: pd.Index | None = None, detail: bool = False) -> pd.DataFrame:
    models = BAT_MODELS if kind == "bat" else BOWL_MODELS
    w = None if w_match is None else pm.match_id.map(w_match).fillna(0).to_numpy()
    wa = np.ones(len(pm)) if w is None else w
    P, B = pm[pm.P], pm[~pm.P]
    wP, wB = (None, None) if w is None else (w[pm.P.to_numpy()], w[~pm.P.to_numpy()])
    wPa = np.ones(len(P)) if wP is None else wP

    nP = (P.n * wPa).groupby(P.player).sum()
    mP = pd.Series(wPa * (P.n > 0), index=P.index).groupby(P.player).sum()
    if fixed_pool is None:
        k = "bat" if kind == "bat" else "bowl"
        pool = nP[(nP >= elig[f"{k}_min_pressure_balls"]) & (mP >= elig[f"{k}_min_pressure_matches"])].index
    else:
        pool = fixed_pool.intersection(nP[nP > 0].index)
    P, B = P[P.player.isin(pool)], B[B.player.isin(pool)]
    wP = None if w is None else w[pm.P.to_numpy() & pm.player.isin(pool).to_numpy()]
    wB = None if w is None else w[(~pm.P).to_numpy() & pm.player.isin(pool).to_numpy()]

    out = pd.DataFrame(index=pool)
    allP = pm[pm.P]
    rpw = (allP.runs * (np.ones(len(allP)) if w is None else w[pm.P.to_numpy()])).sum() / \
          (allP.out * (np.ones(len(allP)) if w is None else w[pm.P.to_numpy()])).sum()
    P = P.assign(conv=(P.runs - P.out * rpw) if kind == "bat" else (P.out * rpw - P.runs))

    raw = {}
    for m, col in [("A", "conv"), ("B", "rae"), ("C", "wpa")] + ([("G", "rae_role")] if kind == "bat" else []):
        rs = _rate_se(P, col, wP)
        raw[m] = rs
        out[m], out[m + "_sd"], _ = eb_shrink(rs.r, rs.se2)
    # D: hierarchical prior from the player's own non-pressure WPA rate
    base_wpa = _rate_se(B, "wpa", wB).reindex(pool)
    cr = raw["C"].reindex(pool)
    ok = (base_wpa.r.notna() & cr.r.notna()).to_numpy()
    X = np.c_[np.ones(ok.sum()), base_wpa.r.to_numpy()[ok]]
    sw = 1 / np.sqrt(cr.se2.to_numpy()[ok])
    beta = np.linalg.lstsq(X * sw[:, None], cr.r.to_numpy()[ok] * sw, rcond=None)[0]
    prior = pd.Series(beta[0] + beta[1] * base_wpa.r.fillna(base_wpa.r.mean()), index=pool)
    out["D"], out["D_sd"], _ = eb_shrink(cr.r, cr.se2, prior)
    # R: pressure minus baseline contextual rate
    rb = _rate_se(B, "rae", wB).reindex(pool)
    diff = raw["B"].r - rb.r.fillna(0)
    out["R"], out["R_sd"], _ = eb_shrink(diff, raw["B"].se2 + rb.se2.fillna(raw["B"].se2))
    # E: consistency across pressure innings / spells
    min_n = 4 if kind == "bat" else 6
    inn = P[P.n >= min_n]
    wi = np.ones(len(inn)) if wP is None else wP[(P.n >= min_n).to_numpy()]
    kk = pd.Series(wi * (inn.rae > 0), index=inn.index).groupby(inn.player).sum().reindex(pool).fillna(0)
    mm = pd.Series(wi, index=inn.index).groupby(inn.player).sum().reindex(pool).fillna(0)
    out["E"] = _beta_binom(kk, mm)

    z = (out[models] - out[models].mean()) / out[models].std()
    out[[f"z_{m}" for m in models]] = z.values
    wts = weights or {m: 1.0 for m in models}
    tot = sum(wts.get(m, 0) for m in models)
    out["ens"] = sum(z[m] * wts.get(m, 0) for m in models) / tot
    out["rank"] = out.ens.rank(ascending=False, method="first").astype(int)

    if detail:
        out = out.join(_describe(pm[pm.player.isin(pool)], kind))
        out["raw_C"], out["raw_B"], out["raw_R"] = raw["C"].r, raw["B"].r, diff
        out["base_wpa_rate"] = base_wpa.r
        out["prior_D"] = prior
    return out.sort_values("rank")


def _describe(pm: pd.DataFrame, kind: str) -> pd.DataFrame:
    def agg(df):
        g = df.groupby("player")
        s = g.sum(numeric_only=True)
        res = pd.DataFrame(index=s.index)
        res["balls"] = s.n
        res["matches"] = g.match_id.nunique()
        res["runs"] = s.runs
        res["outs"] = s.out
        if kind == "bat":
            res["sr"] = 100 * s.runs / s.n
            res["avg"] = s.runs / s.out.replace(0, np.nan)
            res["bnd_pct"] = 100 * (s.four + s.six) / s.n
            res["dot_pct"] = 100 * s["dot"] / s.n
            res["xrae_per100"] = 100 * s.rae / s.n
        else:
            res["econ"] = 6 * s.runs / s.n
            res["bsr"] = s.n / s.out.replace(0, np.nan)
            res["avg"] = s.runs / s.out.replace(0, np.nan)
            res["dot_pct"] = 100 * s["dot"] / s.n
            res["bnd_pct"] = 100 * s.bnd / s.deliv
            res["xrae_per100"] = 100 * s.rae / s.n
            res["death_share"] = s.death / s.n
            res["pp_share"] = s.pp / s.n
        res["wpa_total"] = s.wpa
        res["spin_share"] = s.spin / (s.n if kind == "bat" else s.deliv)
        return res
    p = agg(pm[pm.P]).add_prefix("p_")
    b = agg(pm[~pm.P]).add_prefix("b_")
    a = agg(pm).add_prefix("all_")
    extra = pd.DataFrame(index=p.index)
    if kind == "bat":
        g = pm.groupby("player")
        extra["avg_pos"] = (g.pos.sum() / g.n.sum())
    return p.join(b).join(a).join(extra)
