import { useEffect, useState } from "react";
import SidebarLeft from "./components/SidebarLeft.jsx";
import KarachiMap from "./components/KarachiMap.jsx";
import DistrictCard from "./components/DistrictCard.jsx";
import InsightsPanel from "./components/InsightsPanel.jsx";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api";

export default function App() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [selectedDay, setSelectedDay] = useState(0);
  const [selectedDistrict, setSelectedDistrict] = useState(null);

  useEffect(() => {
    fetch(`${API_BASE}/forecast`)
      .then((res) => {
        if (!res.ok) throw new Error(`Server responded ${res.status}`);
        return res.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="center-state">
        <span className="err">Could not reach the forecast API ({error}).</span>
      </div>
    );
  }
  if (!data) {
    return (
      <div className="center-state">
        <span className="spin" /> Loading today&rsquo;s forecast…
      </div>
    );
  }

  const { districts, districtOrder, modelStats, bestModelPerHorizon, issueDate } = data;
  const days = districts[districtOrder[0]].map((e) => ({ horizon: e.horizon, label: e.label, date: e.date }));
  const dayLabel = days.find((d) => d.horizon === selectedDay)?.label ?? "Today";

  return (
    <div className="shell">
      <SidebarLeft
        days={days}
        selectedDay={selectedDay}
        onSelectDay={setSelectedDay}
        bestModelPerHorizon={bestModelPerHorizon}
      />

      <main className="main">
        <div className="topbar">
          <h1>Karachi District Forecast</h1>
          <span className="issue">
            issued <b>{issueDate}</b>
          </span>
        </div>

        <div className="map-card">
          <div className="map-card-head">
            <span className="label">District AQI · {dayLabel}</span>
            <span className="label">{selectedDistrict ?? "click a district"}</span>
          </div>
          <KarachiMap
            districts={districts}
            selectedDay={selectedDay}
            selectedDistrict={selectedDistrict}
            onSelectDistrict={(name) => setSelectedDistrict(name === selectedDistrict ? null : name)}
          />
          <div className="map-caption">
            District boundaries simplified from HDX administrative data · Malir extends further north
            and south beyond this frame
          </div>
        </div>

        <div className="grid-label">
          <div className="section-label">2-Day Outlook · All Districts</div>
        </div>
        <div className="district-grid">
          {districtOrder.map((name) => (
            <DistrictCard
              key={name}
              name={name}
              entries={districts[name]}
              selected={selectedDistrict === name}
              onSelect={() => setSelectedDistrict(name === selectedDistrict ? null : name)}
            />
          ))}
        </div>
      </main>

      <InsightsPanel
        districts={districts}
        districtOrder={districtOrder}
        selectedDay={selectedDay}
        dayLabel={dayLabel}
        modelStats={modelStats}
      />
    </div>
  );
}
