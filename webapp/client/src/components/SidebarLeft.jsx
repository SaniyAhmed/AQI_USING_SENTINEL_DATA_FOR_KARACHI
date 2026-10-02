import { AQI_BANDS } from "../aqi.js";

export default function SidebarLeft({ days, selectedDay, onSelectDay, bestModelPerHorizon }) {
  return (
    <aside className="sidebar-left">
      <div>
        <div className="brand-name">
          <span className="urdu">ہوا</span>
          <span className="lat">Hawa</span>
        </div>
        <div className="brand-sub">Satellite AQI Forecast · Karachi</div>
      </div>

      <div className="section">
        <div className="section-label">Forecast Day</div>
        <div className="pill-group">
          {days.map((d) => (
            <button
              key={d.horizon}
              className={"pill" + (d.horizon === selectedDay ? " active" : "")}
              onClick={() => onSelectDay(d.horizon)}
            >
              <span>{d.label}</span>
              <span className="pill-date">{d.date}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="section">
        <div className="section-label">Model</div>
        <div className="model-note">
          Each horizon is served by whichever of <b>LightGBM</b> or <b>XGBoost</b> scored lower error on
          held-out historical days: T+0 <b>{bestModelPerHorizon?.["0"]}</b>, T+1{" "}
          <b>{bestModelPerHorizon?.["1"]}</b>.
        </div>
      </div>

      <div className="section">
        <div className="section-label">Legend · AQI Category</div>
        <div className="legend-list">
          {AQI_BANDS.map((b) => (
            <div className="legend-item" key={b.key}>
              <span className="legend-dot" style={{ background: b.color }} />
              <span>{b.short ?? b.key}</span>
              <span className="legend-range">{b.range}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="foot-note">
        Ground truth: OpenAQ Karachi network. Inputs: Sentinel-5P, MODIS MAIAC, ERA5 / GFS, CAMS.
        District boundaries: HDX admin data, simplified.
      </div>
    </aside>
  );
}
