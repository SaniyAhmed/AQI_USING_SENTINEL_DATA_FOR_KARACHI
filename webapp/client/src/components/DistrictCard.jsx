import { aqiColor, aqiShort, DISTRICT_CODE, MODEL_CODE } from "../aqi.js";

export default function DistrictCard({ name, entries, selected, onSelect }) {
  const today = entries.find((e) => e.horizon === 0);
  const dotColor = today ? aqiColor(today.category) : "var(--text-faint)";

  return (
    <div className={"dcard" + (selected ? " selected" : "")} onClick={onSelect}>
      <div className="dcard-head">
        <span className="dot" style={{ background: dotColor }} />
        <h3>{name}</h3>
        <span className="code-badge">{DISTRICT_CODE[name]}</span>
      </div>
      <div className="day-row">
        {entries.map((e) => {
          const color = aqiColor(e.category);
          return (
            <div
              key={e.horizon}
              className="day-chip"
              style={{ "--chip-color": color, "--chip-bg": `${color}1f`, "--chip-border": `${color}55` }}
            >
              <span className="d-label">{e.label}</span>
              <span className="d-date">{e.date.slice(5)}</span>
              <span className="d-aqi tabular">{e.aqi}</span>
              <span className="d-cat">{aqiShort(e.category)}</span>
              <span className="d-model">{MODEL_CODE[e.model] ?? e.model}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
