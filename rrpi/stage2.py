"""Stage 2: annotated deliveries -> pressure flags, ratings, validation, uncertainty, dashboard data."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from . import pressure, validate
from .ratings import BAT_MODELS, BOWL_MODELS, MODEL_NAMES, player_match, score

ROOT = Path(__file__).resolve().parents[1]
log = logging.getLogger("stage2")


def names_and_teams(cfg):
    D = (ROOT / cfg["data_dir"]).resolve()
    ctp = pd.read_csv(D / "competition_team_player_mapping.csv")
    comps = pd.read_csv(D / "competitions.csv")
    ctp = ctp.merge(comps[["comp_id", "match_format", "season"]], on="comp_id")
    name = ctp.groupby("player_id").player_name.last()
    ipl = ctp[ctp.comp_name.str.startswith("IPL")].copy()
    ipl["yr"] = ipl.comp_name.str.extract(r"(\d{4})")[0].astype(int)
    last_ipl = ipl.sort_values("yr").groupby("player_id").agg(last_ipl_team=("team_name", "last"), last_ipl_year=("yr", "last"))
    t20i = ctp[ctp.match_format == "T20I"]
    nat = t20i.groupby("player_id").team_name.agg(lambda s: s.value_counts().index[0]).rename("t20i_team")
    info = pd.DataFrame({"name": name}).join(last_ipl).join(nat)
    # Nationality is NOT in the data pack; this is only "which T20I side did they play for in 2022+".
    info["origin"] = np.where(info.t20i_team.isna(), "No T20I in data",
                              np.where(info.t20i_team.str.startswith("India"), "India (T20I)", "Overseas (T20I)"))
    return info


def archetypes(S, kind, seed):
    if kind == "bat":
        cols = ["all_sr", "all_bnd_pct", "all_dot_pct", "out_rate", "avg_pos"]
        X = S.assign(out_rate=S.all_outs / S.all_balls)[cols]
    else:
        cols = ["all_econ", "wkt_rate", "all_dot_pct", "all_bnd_pct", "all_death_share", "all_pp_share", "all_spin_share"]
        X = S.assign(wkt_rate=S.all_outs / S.all_balls)[cols]
    Z = ((X - X.mean()) / X.std()).fillna(0)
    best = max(range(3, 7), key=lambda k: silhouette_score(Z, KMeans(k, n_init=20, random_state=seed).fit_predict(Z)))
    km = KMeans(best, n_init=20, random_state=seed).fit(Z)
    cent = pd.DataFrame(km.cluster_centers_ * X.std().values + X.mean().values, columns=cols)
    labels = {}
    for i, c in cent.iterrows():
        if kind == "bat":
            pos, sr = c.avg_pos, c.all_sr
            hi = sr >= X.all_sr.median()
            lbl = ("Finisher" if hi else "Lower-order") if pos >= 5.2 else (("Aggressive opener" if hi else "Anchor opener") if pos <= 2.6
                                                                           else ("Middle-order enforcer" if hi else "Middle-order stabiliser"))
        else:
            wk = c.wkt_rate >= X.wkt_rate.median()
            if c.all_spin_share > 0.5:
                lbl = "Wicket-taking spinner" if wk else "Control spinner"
            elif c.all_death_share > 0.3:
                lbl = "Death specialist"
            elif c.all_pp_share > 0.4:
                lbl = "New-ball swing/seam"
            else:
                lbl = "Middle-overs enforcer" if wk else "Holding seamer"
        while lbl in labels.values():
            lbl += " II"
        labels[i] = lbl
    return pd.Series(km.labels_, index=S.index).map(labels), {"k": best, "features": cols,
                                                              "centroids": cent.round(3).assign(label=[labels[i] for i in cent.index]).to_dict("records")}


def splits(d, P, kind, players):
    if kind == "bat":
        s = d[d.bat_faced & P & d.striker_id.isin(players)]
        g = s.groupby(["striker_id", "phase"]).agg(runs=("runs", "sum"), balls=("runs", "size"), outs=("striker_out", "sum"))
        g["val"] = 100 * g.runs / g.balls
        g2 = s.groupby(["striker_id", "pace_or_spin"]).agg(runs=("runs", "sum"), balls=("runs", "size"))
        g2["val"] = 100 * g2.runs / g2.balls
        g3 = s.groupby(["striker_id", "innings"]).agg(runs=("runs", "sum"), balls=("runs", "size"))
        g3["val"] = 100 * g3.runs / g3.balls
    else:
        s = d[P & d.bowler_id.isin(players)]
        agg = dict(runs=("bowler_runs", "sum"), balls=("legal", "sum"), wk=("bowler_wkt", "sum"))
        g, g2, g3 = (s.groupby(["bowler_id", k]).agg(**agg) for k in ["phase", "batting_type", "innings"])
        for x in (g, g2, g3):
            x["val"] = 6 * x.runs / x.balls.clip(lower=1)
    out = {}
    for pid in players:
        out[int(pid)] = {nm: {str(k): {"val": round(float(r.val), 1), "balls": int(r.balls)} for k, r in t.loc[pid].iterrows()}
                         if pid in t.index.get_level_values(0) else {} for nm, t in [("phase", g), ("matchup", g2), ("innings", g3)]}
    return out


def strengths(row, kind, pct):
    s, w = [], []
    mp = {m: pct.get(f"z_{m}", 50) for m in (BAT_MODELS if kind == "bat" else BOWL_MODELS)}
    top = sorted(mp, key=mp.get, reverse=True)
    for m in top[:2]:
        if mp[m] >= 70:
            s.append(f"{MODEL_NAMES[m]} ({mp[m]:.0f}th pct)")
    for m in top[::-1][:2]:
        if mp[m] <= 45:
            w.append(f"{MODEL_NAMES[m]} ({mp[m]:.0f}th pct)")
    if kind == "bat":
        if row.p_sr - row.b_sr >= 15: s.append(f"Strike rate rises under pressure (+{row.p_sr - row.b_sr:.0f})")
        if row.p_sr - row.b_sr <= -15: w.append(f"Strike rate drops under pressure ({row.p_sr - row.b_sr:.0f})")
        if row.p_dot_pct >= 40: w.append(f"High pressure dot-ball % ({row.p_dot_pct:.0f}%)")
        if row.p_bnd_pct >= 20: s.append(f"Boundary every {100 / row.p_bnd_pct:.1f} balls under pressure")
    else:
        if row.b_econ - row.p_econ >= 0.75: s.append(f"Economy improves under pressure ({row.p_econ:.2f} vs {row.b_econ:.2f})")
        if row.p_econ - row.b_econ >= 1.0: w.append(f"Economy leaks under pressure ({row.p_econ:.2f} vs {row.b_econ:.2f})")
        if row.p_dot_pct >= 40: s.append(f"{row.p_dot_pct:.0f}% dot balls under pressure")
        if pd.notna(row.p_bsr) and row.p_bsr <= 15: s.append(f"Wicket every {row.p_bsr:.0f} pressure balls")
    if row.p_matches < 15: w.append(f"Thin evidence: {int(row.p_matches)} pressure matches")
    return s[:4], w[:3]


def run_kind(d, Pflags, kind, cfg, info, out):
    elig = {k: v for k, v in cfg["eligibility"].items() if isinstance(v, int)}
    cfg = {**cfg, "eligibility": elig, "eligibility_full": cfg["eligibility"]}
    P = Pflags["LI"]
    main_tiers = cfg["main_tiers"]
    sub = lambda x: x[x.tier.isin(main_tiers)]
    pm_all = player_match(d, P, kind)
    pm = sub(pm_all)
    d_main = sub(d)
    log.info("%s: backtest", kind)
    weights, bt = validate.backtest(pm, kind, cfg)
    S = score(pm, kind, elig, weights, detail=True)
    pool = S.index
    log.info("%s: %d eligible; bootstrap", kind, len(pool))
    boot = validate.bootstrap(pm, kind, elig, weights, pool, cfg["bootstrap_reps"], cfg["seed"])
    S = S.join(boot)
    S["point_rank"] = S["rank"]
    S["rank"] = S.bagged.rank(ascending=False, method="first").astype(int)
    S = S.sort_values("rank")
    models = BAT_MODELS if kind == "bat" else BOWL_MODELS
    pct = S[[f"z_{m}" for m in models] + ["bagged"]].rank(pct=True) * 100
    S["clutch_index"] = pct.bagged.round(1)
    S["disagreement"] = S[[f"z_{m}" for m in models]].rank(ascending=False).std(axis=1)
    S["model_ranks"] = S[models].rank(ascending=False).astype(int).to_dict("records")
    arche, arche_meta = archetypes(S, kind, cfg["seed"])
    S["archetype"] = arche
    S = S.join(info, how="left")

    # sensitivity
    log.info("%s: sensitivity", kind)
    base_top = list(S.head(10).index)
    variants = {}
    for q in cfg["pressure"]["sensitivity_quantiles"]:
        variants[f"LI top {int(round((1 - q) * 100))}%"] = (sub(player_match(d, pressure.flags(d, cfg["pressure"], q)["LI"], kind)), elig)
    variants["Close & late (WP band)"] = (sub(player_match(d, Pflags["CLOSE"], kind)), elig)
    for mb in cfg["eligibility_full"]["sensitivity_min_balls"]:
        e2 = {k: (mb if "balls" in k else v) for k, v in elig.items()}
        variants[f"Min {mb} pressure balls"] = (pm, e2)
    variants["IPL only"] = (pm[pm.tier == "IPL"], {k: (v // 2 if "matches" in k else v) for k, v in elig.items()})
    variants["All tiers (incl. associate & A-team)"] = (pm_all, elig)
    variants["Point estimate (no bagging)"] = (pm, elig)
    variants["Recent (2024-26)"] = (pm[pm.year >= 2024], {k: (v // 2 if "matches" in k else int(v * 0.6)) for k, v in elig.items()})
    variants["Equal model weights"] = (pm, elig)
    sens = []
    rank_grid = {}
    for nm, (pmv, ev) in variants.items():
        Sv = score(pmv, kind, ev, None if nm == "Equal model weights" else weights)
        top = list(Sv.head(10).index)
        common = S.index.intersection(Sv.index)
        sens.append({"variant": nm, "top10_overlap": len(set(top) & set(base_top)), "n_eligible": int(len(Sv)),
                     "spearman_vs_main": float(spearmanr(S.loc[common, "ens"], Sv.loc[common, "ens"]).statistic)})
        rank_grid[nm] = [int(Sv["rank"].get(p)) if p in Sv.index else None for p in base_top]

    # pressure-definition head-to-head for ranking (how different are the top-10s?)
    persist = validate.persistence(pm, kind, elig)
    log.info("%s: negative control", kind)
    neg = validate.negative_control(d_main, P[d_main.index], kind, elig, reps=40, seed=cfg["seed"])

    # correlation between models (redundancy)
    corr = S[models].corr(method="spearman").round(3)

    # match timelines for top 30
    top30 = list(S.head(30).index)
    mi = d.drop_duplicates("match_id").set_index("match_id")[["match_date", "comp_name", "tier"]]
    tl = {}
    for p in top30:
        r = pm[(pm.player == p) & pm.P].merge(mi, left_on="match_id", right_index=True).sort_values("match_date")
        tl[int(p)] = [{"d": x.match_date, "c": x.comp_name, "wpa": round(float(x.wpa), 2), "n": int(x.n),
                       "rae": round(float(x.rae), 2), "runs": int(x.runs), "out": int(x.out)} for x in r.itertuples()]
    spl = splits(d_main, P[d_main.index], kind, top30)

    players = []
    for pid, r in S.iterrows():
        st, wk = strengths(r, kind, pct.loc[pid].to_dict())
        rec = {"id": int(pid), "name": r["name"], "rank": int(r["rank"]), "point_rank": int(r.point_rank),
               "ens": round(float(r.ens), 3), "bagged": round(float(r.bagged), 3), "bagged_sd": round(float(r.bagged_sd), 3),
               "clutch_index": float(r.clutch_index), "archetype": r.archetype,
               "rank_med": float(r.rank_med), "rank_lo": float(r.rank_lo), "rank_hi": float(r.rank_hi),
               "p_top10": round(float(r.p_top10), 3), "p_top3": round(float(r.p_top3), 3),
               "disagreement": round(float(r.disagreement), 2), "model_ranks": r.model_ranks,
               "z": {m: round(float(r[f"z_{m}"]), 3) for m in models},
               "pct": {m: round(float(pct.loc[pid, f"z_{m}"]), 1) for m in models},
               "est": {m: round(float(r[m]), 4) for m in models},
               "sd": {m: round(float(r[m + "_sd"]), 4) for m in models if m + "_sd" in r},
               "last_ipl_team": None if pd.isna(r.last_ipl_team) else r.last_ipl_team,
               "last_ipl_year": None if pd.isna(r.last_ipl_year) else int(r.last_ipl_year),
               "t20i_team": None if pd.isna(r.t20i_team) else r.t20i_team, "origin": r.origin,
               "strengths": st, "weaknesses": wk}
        for c in S.columns:
            if c.startswith(("p_", "b_", "all_")) and c not in ("p_top10", "p_top3"):
                v = r[c]
                rec[c] = None if pd.isna(v) else round(float(v), 3)
        if kind == "bat":
            rec["avg_pos"] = round(float(r.avg_pos), 2)
        rec["raw_R"], rec["raw_C"], rec["base_wpa_rate"] = (round(float(r[k]), 4) for k in ["raw_R", "raw_C", "base_wpa_rate"])
        players.append(rec)

    S.drop(columns=["model_ranks"]).to_csv(out / f"{kind}_rankings_full.csv")
    return {"players": players, "weights": weights, "backtest": bt, "sensitivity": sens, "rank_grid": rank_grid,
            "base_top": [int(p) for p in base_top], "persistence": persist, "negative_control": neg,
            "model_corr": {"models": models, "z": corr.values.tolist()}, "timelines": tl, "splits": spl,
            "archetypes": arche_meta, "n_eligible": int(len(S))}


def thrillers(d, n=6):
    m = d.groupby("match_id").agg(li=("li", "sum"), date=("match_date", "first"), comp=("comp_name", "first"))
    final = d[d.innings == 2].groupby("match_id").wp_after.last()
    m = m.join(final).dropna()
    teams = pd.read_csv((ROOT / ".." / "teams.csv").resolve()).drop_duplicates("team_id").set_index("team_id").team_name
    out = []
    for mid in m.sort_values("li", ascending=False).head(n).index:
        s = d[d.match_id == mid]
        t1 = teams.get(s[s.innings == 1].batting_team_id.iloc[0], "Team 1")
        t2 = teams.get(s[s.innings == 2].batting_team_id.iloc[0], "Team 2")
        balls = []
        for i, x in enumerate(s.itertuples()):
            wp1 = x.wp_after if x.innings == 1 else 1 - x.wp_after
            ev = "W" if x.is_wicket else ("6" if x.is_six else ("4" if x.is_four else ""))
            balls.append([i, x.innings, round(float(wp1), 4), round(float(x.li), 2), ev,
                          f"{x.over - 1}.{x.over_ball_num}", x.striker_name, x.bowler_name, int(x.runs_before + x.total_runs),
                          int(x.wkts_before + x.is_wicket)])
        res = pd.read_csv((ROOT / ".." / "matches.csv").resolve()).set_index("match_id").result.get(mid, "")
        out.append({"match_id": int(mid), "title": f"{t1} v {t2}", "date": m.loc[mid, "date"], "comp": m.loc[mid, "comp"],
                    "result": res, "t1": t1, "t2": t2, "balls": balls})
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = json.loads((ROOT / "config.json").read_text())
    out = ROOT / cfg["out_dir"]
    d = pd.read_parquet(out / "intermediate" / "deliveries.parquet")
    diag = json.loads((out / "intermediate" / "stage1_diag.json").read_text())
    F = pressure.flags(d, cfg["pressure"])
    info = names_and_teams(cfg)
    result = {
        "meta": {"deliveries": int(len(d)), "matches": int(d.match_id.nunique()),
                 "pressure_balls": int(F.LI.sum()), "li_threshold": float(d.li.quantile(cfg["pressure"]["li_quantile"])),
                 "seasons": sorted(int(y) for y in d.year.unique()),
                 "by_tier": d.drop_duplicates("match_id").tier.value_counts().to_dict(),
                 "config": cfg},
        "quality": diag["quality"], "wp": diag["wp"], "ctx": diag["ctx"],
        "pressure_compare": pressure.compare(d, F), "li_maps": pressure.li_maps(d),
        "thrillers": thrillers(d[d.tier.isin(cfg["main_tiers"])]),
    }
    for kind in ("bat", "bowl"):
        result[kind] = run_kind(d, F, kind, cfg, info, out)
        print(kind, "weights", result[kind]["weights"])
        print(pd.DataFrame(result[kind]["backtest"]["holdout"]["rows"]).round(3).to_string())
        print([(p["rank"], p["name"], p["p_top10"]) for p in result[kind]["players"][:15]])
    dash = ROOT / "dashboard"
    dash.mkdir(exist_ok=True)
    payload = json.dumps(result, default=lambda o: o.item() if hasattr(o, "item") else str(o), separators=(",", ":"))
    (dash / "data.js").write_text("window.RRPI=" + payload + ";")
    (out / "results.json").write_text(payload)
    log.info("wrote dashboard/data.js (%.1f KB)", len(payload) / 1024)


if __name__ == "__main__":
    sys.exit(main())
