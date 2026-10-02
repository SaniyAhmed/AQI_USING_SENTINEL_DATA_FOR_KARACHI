"""
Exports the latest live forecast + model performance stats for the webapp/ dashboard.

Reads:
    data/forecast/prediction_<issue>.csv      (phase5_models/predict_today.py output)
    outputs/phase5_validation_report.json     (phase5_models/step4_evaluate_and_gate.py output)
Writes:
    webapp/server/seedData.json               the backend re-seeds MongoDB from this on next boot
                                               (seed.js compares issue_date, so a stale server picks
                                               up a new day automatically - just restart `npm run dev`)

Run:  python phase5_models/export_webapp_data.py [--issue YYYY-MM-DD]   (default: today, Karachi time)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "phase3_features"))
import pandas as pd

import config
from step5_gfs_forecast import local_today

DISTRICT_ORDER = ["Karachi Central", "Karachi East", "Karachi South", "Karachi West", "Korangi", "Malir"]
HORIZON_LABEL = {0: "Today", 1: "Tomorrow", 2: "Day After"}
WEBAPP_SEED_FILE = config.PROJECT_ROOT / "webapp" / "server" / "seedData.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", default=None, help="issue date YYYY-MM-DD (default: today, Karachi time)")
    args = ap.parse_args()
    issue = pd.Timestamp(args.issue) if args.issue else local_today()

    pred_path = config.FORECAST_DIR / f"prediction_{issue:%Y-%m-%d}.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found - run: python phase5_models/predict_today.py --issue {issue:%Y-%m-%d}")
    pred = pd.read_csv(pred_path, parse_dates=["issue_date", "target_date"])
    report = json.loads(config.PHASE5_REPORT.read_text(encoding="utf-8"))

    data = {"issue_date": str(pred["issue_date"].iloc[0].date()), "districts": {}}
    for name in DISTRICT_ORDER:
        rows = pred[pred["district"] == name].sort_values("horizon")
        data["districts"][name] = [
            {"horizon": int(r.horizon), "label": HORIZON_LABEL[int(r.horizon)], "date": str(r.target_date.date()),
             "pm25": round(float(r.pred_final), 1), "aqi": int(r.aqi_final), "category": r.category_final, "model": r.best_model}
            for r in rows.itertuples()
        ]
    m = report["metrics"]
    data["model_stats"] = {k: {kk: (round(vv, 2) if isinstance(vv, (int, float)) else vv) for kk, vv in m[k].items()}
                           for k in ("overall|pred_lightgbm", "overall|pred_xgboost", "overall|pred_persistence")}
    data["best_model_per_horizon"] = report["best_model_per_horizon"]

    WEBAPP_SEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEBAPP_SEED_FILE.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"Saved {WEBAPP_SEED_FILE} (issue date {data['issue_date']}). "
          f"Restart the webapp server (or let it reboot) to pick it up.")


if __name__ == "__main__":
    main()
