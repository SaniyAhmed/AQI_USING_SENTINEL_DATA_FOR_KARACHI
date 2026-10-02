import { aqiColor, DISTRICT_CODE } from "../aqi.js";
import mapData from "../data/mapData.json";

export default function KarachiMap({ districts, selectedDay, selectedDistrict, onSelectDistrict }) {
  const entries = Object.entries(mapData.districts);

  return (
    <div className="map-wrap">
      <svg viewBox={mapData.viewbox} xmlns="http://www.w3.org/2000/svg">
        {entries.map(([name, geo]) => {
          const day = districts[name]?.find((d) => d.horizon === selectedDay);
          const color = day ? aqiColor(day.category) : "var(--text-faint)";
          return (
            <path
              key={name}
              d={geo.path}
              style={{ fill: color, fillOpacity: selectedDistrict === name ? 0.55 : 0.34 }}
              className={"district-shape" + (selectedDistrict === name ? " selected" : "")}
              onClick={() => onSelectDistrict(name)}
            >
              <title>
                {name}: AQI {day?.aqi} · {day?.category}
              </title>
            </path>
          );
        })}
        {entries.map(([name, geo]) => {
          const day = districts[name]?.find((d) => d.horizon === selectedDay);
          const color = day ? aqiColor(day.category) : "var(--text-faint)";
          return (
            <g
              key={name + "-marker"}
              className="district-marker"
              transform={`translate(${geo.cx} ${geo.cy})`}
              onClick={() => onSelectDistrict(name)}
              style={{ cursor: "pointer" }}
            >
              <circle r="46" />
              <circle r="46" style={{ fill: "none", stroke: color, strokeWidth: 4 }} />
              <text y="-6" fontSize="34" style={{ fill: color }}>
                {day?.aqi ?? "–"}
              </text>
              <text className="code" y="30" fontSize="20">
                {DISTRICT_CODE[name]}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
