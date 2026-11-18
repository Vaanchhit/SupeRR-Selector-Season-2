"""Competing operational definitions of pressure, and how they compare.

LI      leverage index above the q-th quantile (primary; continuous, outcome-based)
CLOSE   match still in the balance (|WP-0.5| <= band) from over `min_over` on
DEATH   naive strawman: overs 17-20 regardless of match state
RRR     naive strawman: chasing with required rate >= 10
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def flags(d: pd.DataFrame, cfg: dict, q: float | None = None) -> pd.DataFrame:
    q = cfg["li_quantile"] if q is None else q
    return pd.DataFrame({
        "LI": d.li >= d.li.quantile(q),
        "CLOSE": ((d.wp - 0.5).abs() <= cfg["close_late_wp_band"]) & (d.over > cfg["close_late_min_over"]),
        "DEATH": d.over >= 17,
        "RRR": (d.innings == 2) & (d.rrr >= 10),
    }, index=d.index)


def compare(d: pd.DataFrame, F: pd.DataFrame) -> list[dict]:
    decided = (d.wp < 0.1) | (d.wp > 0.9)
    rows = []
    for k in F.columns:
        f = F[k]
        rows.append({
            "definition": k,
            "share_of_balls": float(f.mean()),
            "mean_li": float(d.li[f].mean()),
            "mean_abs_wpa": float(d.wpa_total[f].abs().mean()),
            "share_decided": float(decided[f].mean()),
            "jaccard_with_LI": float((f & F.LI).sum() / (f | F.LI).sum()),
            "share_inn2": float((d.innings[f] == 2).mean()),
            "share_of_total_abs_wpa": float(d.wpa_total[f].abs().sum() / d.wpa_total.abs().sum()),
        })
    return rows


def li_maps(d: pd.DataFrame) -> dict:
    out = {}
    for inn in (1, 2):
        s = d[d.innings == inn]
        t = s.groupby([s.over, s.wkts_before.clip(upper=8)]).li.mean().unstack()
        out[f"inn{inn}_over_x_wkts"] = {"overs": t.index.tolist(), "wkts": t.columns.tolist(), "z": t.round(3).fillna(0).values.tolist()}
    s = d[d.innings == 2]
    rb = pd.cut(s.rrr.clip(0, 24), [0, 6, 8, 10, 12, 15, 24], include_lowest=True)
    t = s.groupby([rb, pd.cut(s.balls_left, [0, 12, 24, 36, 60, 90, 120])], observed=True).li.mean().unstack()
    out["chase_rrr_x_balls_left"] = {"rrr": [str(i) for i in t.index], "balls_left": [str(c) for c in t.columns],
                                     "z": t.round(3).fillna(0).values.tolist()}
    wpb = pd.cut(d.wp, np.linspace(0, 1, 21), include_lowest=True)
    t = d.groupby(wpb, observed=True).li.mean()
    out["li_by_wp"] = {"wp": [round(i.mid, 3) for i in t.index], "li": t.round(3).tolist()}
    out["li_hist"] = np.histogram(d.li.clip(upper=6), bins=60)[0].tolist()
    return out
