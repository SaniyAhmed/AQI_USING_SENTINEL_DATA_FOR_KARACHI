import { aqiColor, aqiShort } from "../aqi.js";

function mode(arr) {
  const counts = {};
  let best = arr[0], bestCount = 0;
  for (const v of arr) {
    counts[v] = (counts[v] ?? 0) + 1;
    if (counts[v] > bestCount) { best = v; bestCount = counts[v]; }
  }
  return best;
}

export default function InsightsPanel({ districts, districtOrder, selectedDay, dayLabel, modelStats }) {
  const todays = districtOrder.map((name) => districts[name].find((e) => e.horizon === selectedDay));
  const avgAqi = Math.round(todays.reduce((s, e) => s + e.aqi, 0) / todays.length);
  const dominant = mode(todays.map((e) => e.category));

  const lgb = modelStats["overall|pred_lightgbm"];
  const xgb = modelStats["overall|pred_xgboost"];
  const base = modelStats["overall|pred_persistence"];
  const bestRmse = Math.min(lgb.rmse, xgb.rmse);
  const improvement = Math.round(((base.rmse - bestRmse) / base.rmse) * 100);

  const ranked = districtOrder
    .map((name) => ({ name, ...districts[name].find((e) => e.horizon === selectedDay) }))
    .sort((a, b) => b.aqi - a.aqi);
  const maxAqi = Math.max(...ranked.map((r) => r.aqi), 100);

  return (
    <aside className="sidebar-right">
      <div className="section-label">Model Insights</div>

      <div className="section">
        <div className="section-label">{dayLabel} · City-Wide</div>
        <div className="metric-grid">
          <div className="metric-tile">
            <span className="m-label">Average AQI</span>
            <span className="m-value tabular" style={{ color: aqiColor(dominant) }}>{avgAqi}</span>
            <span className="m-sub">across 6 districts</span>
          </div>
          <div className="metric-tile">
            <span className="m-label">Dominant Band</span>
            <span className="m-value" style={{ fontSize: 19, color: aqiColor(dominant) }}>{aqiShort(dominant)}</span>
            <span className="m-sub" title={dominant}>most common · {dayLabel.toLowerCase()}</span>
          </div>
        </div>
      </div>

      <div className="callout">
        <span className="c-big">-{improvement}% RMSE</span>
        <span className="c-sub">vs. a naive "tomorrow = today" forecast, measured on 7,401 held-out real days</span>
      </div>

      <div className="section">
        <div className="section-label">District Ranking · {dayLabel}</div>
        <div className="bd-list">
          {ranked.map((r) => (
            <div className="bd-row" key={r.name}>
              <span className="bd-name">{r.name.replace("Karachi ", "")}</span>
              <span className="bd-track">
                <span
                  className="bd-fill"
                  style={{ width: `${Math.min(100, (r.aqi / maxAqi) * 100)}%`, background: aqiColor(r.category) }}
                />
              </span>
              <span className="bd-val tabular" style={{ color: aqiColor(r.category) }}>{r.aqi}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-label">Out-Of-Fold Accuracy</div>
        <table className="perf-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>RMSE</th>
              <th>R²</th>
              <th>AQI Acc.</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Persistence</td>
              <td>{base.rmse}</td>
              <td>{base.r2}</td>
              <td>{Math.round(base.aqi_category_accuracy * 100)}%</td>
            </tr>
            <tr>
              <td>LightGBM</td>
              <td className={lgb.rmse <= xgb.rmse ? "best" : ""}>{lgb.rmse}</td>
              <td>{lgb.r2}</td>
              <td>{Math.round(lgb.aqi_category_accuracy * 100)}%</td>
            </tr>
            <tr>
              <td>XGBoost</td>
              <td className={xgb.rmse < lgb.rmse ? "best" : ""}>{xgb.rmse}</td>
              <td>{xgb.r2}</td>
              <td>{Math.round(xgb.aqi_category_accuracy * 100)}%</td>
            </tr>
          </tbody>
        </table>
      </div>
    </aside>
  );
}
