"""Stage 1: raw CSVs -> annotated deliveries parquet (state, WP, leverage, WPA, expected-performance)."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from . import context, data, winprob

ROOT = Path(__file__).resolve().parents[1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = json.loads((ROOT / "config.json").read_text())
    out = ROOT / cfg["out_dir"]
    (out / "intermediate").mkdir(parents=True, exist_ok=True)
    raw = data.load_raw((ROOT / cfg["data_dir"]).resolve())
    qr = data.quality_report(raw)
    d = data.build_deliveries(raw, cfg["full_members"])
    eng = winprob.WPEngine(cfg["seed"]).fit(d)
    d, wp_diag = winprob.annotate(d, eng)
    d, ctx_diag = context.annotate(d, cfg["seed"])
    d.to_parquet(out / "intermediate" / "deliveries.parquet")
    (out / "intermediate" / "stage1_diag.json").write_text(json.dumps({"quality": qr, "wp": wp_diag, "ctx": ctx_diag}, indent=1, default=str))
    print(json.dumps({k: v for k, v in wp_diag.items() if k != "calibration"}, indent=1))
    print(json.dumps({k: v for k, v in ctx_diag.items() if k != "re_grid"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())
