"""Ingestion, validation and pre-delivery match-state reconstruction.

Every state variable here is computed from deliveries strictly *before* the
current one (cumsum minus current), so nothing downstream can leak the outcome
of the ball being evaluated.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ILLEGAL = {"wide", "Overstep", "other"}          # 'other' carries bat runs + 1 extra => a no-ball variant
NOT_BOWLER_WICKET = {"RunOut", "Retire Out", "Obstracting the field", "Obstructing the field"}


def load_raw(data_dir: Path) -> dict[str, pd.DataFrame]:
    f = lambda n: pd.read_csv(data_dir / n, low_memory=False)
    return {
        "balls": f("ball_by_ball.csv"),
        "matches": f("matches.csv"),
        "comps": f("competitions.csv"),
        "teams": f("teams.csv"),
        "xi": f("playing_eleven.csv"),
        "ctp": f("competition_team_player_mapping.csv"),
    }


def quality_report(raw: dict[str, pd.DataFrame]) -> dict:
    b, m = raw["balls"], raw["matches"]
    b2 = b[b.innings <= 2].copy()
    b2["tot"] = b2.runs + b2.extras1_runs + b2.extras_2_runs
    tot = b2.groupby(["match_id", "innings"]).tot.sum().unstack()
    mm = m.set_index("match_id").join(tot)
    mm["margin"] = mm.result.str.extract(r"by (\d+) [Rr]un")[0].astype(float)
    chk = mm[mm.margin.notna() & mm.revised_target.isna() & ~mm.result.str.contains("D/?L", na=False)]
    return {
        "deliveries": int(len(b)),
        "matches_in_matches_csv": int(m.match_id.nunique()),
        "matches_with_ball_data": int(b.match_id.nunique()),
        "duplicate_delivery_keys": int(b.duplicated(["match_id", "innings", "innings_ball_num"]).sum()),
        "super_over_deliveries": int((b.innings > 2).sum()),
        "super_over_matches": int(b.loc[b.innings > 2, "match_id"].nunique()),
        "rain_revised_matches": int(m.revised_target.notna().sum()),
        "no_result_or_abandoned": int(m.result.str.contains("No Result|Abandon", case=False, na=False).sum()),
        "bowl_speed_zero_share": round(float((b.bowl_speed == 0).mean()), 3),
        "bowl_speed_implausible": int((b.bowl_speed > 170).sum()),
        "stroke_missing_share": round(float(b.stroke.isna().mean()), 3),
        "run_margin_reconciliation": round(float(((chk[1] - chk[2]) == chk.margin).mean()), 4),
        "run_margin_matches_checked": int(len(chk)),
        "wickets_nonstriker_share": round(float((b.loc[b.is_wicket, "dismissed_player_id"] == b.loc[b.is_wicket, "non_striker_id"]).mean()), 4),
        "extras_types": b.extras_type.fillna("none").value_counts().to_dict(),
        "dismissal_types": b.dismissal_type.fillna("none").value_counts().to_dict(),
    }


def tier_of(row, full: set[str]) -> str:
    if str(row.comp_name).startswith("IPL"):
        return "IPL"
    if row.match_format == "T20":
        return "A-team"
    if row.home_team in full and row.away_team in full:
        return "T20I-FM"
    return "T20I-Assoc"


def build_deliveries(raw: dict[str, pd.DataFrame], full_members: list[str]) -> pd.DataFrame:
    b, m, c = raw["balls"].copy(), raw["matches"].copy(), raw["comps"]
    m = m.merge(c[["comp_id", "comp_name", "match_format", "season"]], on="comp_id", how="left")
    m["tier"] = m.apply(tier_of, axis=1, full=set(full_members))
    m["year"] = pd.to_datetime(m.match_date).dt.year
    m["no_result"] = m.result.str.contains("No Result|Abandon", case=False, na=False)
    m["tie"] = m.result.str.contains("Tie", case=False, na=False) & ~m.result.str.contains("won", case=False, na=False)

    b = b[b.innings <= 2].merge(
        m[["match_id", "tier", "year", "comp_name", "no_result", "tie", "win_team_id", "knock_out",
           "revised_target", "first_inn_revised_overs", "second_inn_revised_overs", "stage", "match_date"]],
        on="match_id", how="left")
    b = b[~b.no_result].sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

    b["legal"] = ~b.extras_type.isin(ILLEGAL)
    b["total_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
    b["bowler_runs"] = b.runs + np.where(b.extras_type.isin(ILLEGAL), b.extras1_runs, 0)
    b["bat_faced"] = b.extras_type.ne("wide")
    b["striker_out"] = b.is_wicket & (b.dismissed_player_id == b.striker_id)
    b["bowler_wkt"] = b.is_wicket & ~b.dismissal_type.isin(NOT_BOWLER_WICKET)

    g = b.groupby(["match_id", "innings"], sort=False)
    b["runs_before"] = g.total_runs.cumsum() - b.total_runs
    b["wkts_before"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
    b["balls_before"] = g.legal.cumsum() - b.legal.astype(int)

    inn1 = b[b.innings == 1].groupby("match_id").total_runs.sum().rename("inn1_total")
    b = b.join(inn1, on="match_id")
    b["target"] = np.where(b.innings == 2, b.revised_target.fillna(b.inn1_total + 1), np.nan)
    b["total_balls"] = np.where(b.innings == 1, b.first_inn_revised_overs.fillna(120),
                                b.second_inn_revised_overs.fillna(b.first_inn_revised_overs).fillna(120))
    b["total_balls"] = b.total_balls.clip(lower=b.groupby(["match_id", "innings"]).legal.transform("sum"))
    b["balls_left"] = (b.total_balls - b.balls_before).clip(lower=1)
    b["wkts_left"] = 10 - b.wkts_before
    b["runs_needed"] = b.target - b.runs_before
    b["crr"] = np.where(b.balls_before > 0, 6 * b.runs_before / b.balls_before.clip(lower=1), 0.0)
    b["rrr"] = np.where(b.innings == 2, 6 * b.runs_needed / b.balls_left, np.nan)
    b["over"] = b.overid
    b["phase"] = pd.cut(b.overid, [0, 6, 15, 20], labels=["Powerplay", "Middle", "Death"]).astype(str)
    b["bat_won"] = np.where(b.tie, 0.5, (b.win_team_id == b.batting_team_id).astype(float))
    log.info("deliveries after cleaning: %d", len(b))
    return b
