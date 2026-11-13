/**
 * What a Flamescapes run can colour the graph by (Simulate, "Show on the
 * graph"), and each one's legend. Four groups, so the panel stays light:
 *
 *   heat        gas temperature at head height (the run's own ramp);
 *   smoke       visibility, soot mass concentration, extinction coefficient,
 *               obscuration per metre or optical density per metre: all one
 *               function of the soot (core/tenability.js), so they share the
 *               smoke shading and differ only in the legend's labels;
 *   gases       CO, CO2 or O2, each on the heat ramp to a mark from ISO 13571;
 *   tenability  the tenability index (core/tenability.js), red at 1.
 */
import { SIGN_CONSTANT, SOOT_EXTINCTION_G } from "./tenability.js";

/** The smoke ramp's ticks are visibilities (viewer setSmoke: 30 m clear to 1 m). */
const TICKS = [30, 5, 1];
const K = (S) => SIGN_CONSTANT / S; // 1/m at visibility S
const fmt = (x) => (x >= 100 ? Math.round(x).toString() : x >= 10 ? x.toFixed(0) : x >= 1 ? x.toFixed(1) : x.toFixed(2));

export const SMOKE_MEASURES = {
  visibility: { label: "Visibility", title: "Visibility at head height", scale: TICKS.map((s) => `${s} m`) },
  soot: { label: "Soot concentration", title: "Soot mass concentration (mg/m³)", scale: TICKS.map((s) => fmt((1000 * K(s)) / SOOT_EXTINCTION_G)) },
  extinction: { label: "Extinction coefficient", title: "Extinction coefficient (1/m)", scale: TICKS.map((s) => fmt(K(s))) },
  obscuration: { label: "Obscuration per metre", title: "Light lost over 1 m of smoke (%)", scale: TICKS.map((s) => fmt(100 * (1 - Math.exp(-K(s))))) },
  density: { label: "Optical density per metre", title: "Optical density per metre", scale: TICKS.map((s) => fmt(K(s) / Math.LN10)) },
};

export const GAS_MEASURES = {
  co: {
    label: "CO",
    title: "Carbon monoxide (ppm)",
    scale: ["0", "1750", "3500"],
    note: "3,500 ppm for 10 minutes is a full incapacitating dose (ISO 13571); the dose, not the concentration, closes a place: see Tenability.",
    level: (f, v) => f.co[v] / 3500,
    extras: false,
  },
  co2: {
    label: "CO₂",
    title: "Carbon dioxide (% by volume)",
    scale: ["0", "2.5", "5"],
    note: "Above 2 % it makes people breathe faster and take in CO sooner (ISO 13571's factor).",
    level: (f, v) => f.co2[v] / 5,
    extras: true,
  },
  o2: {
    label: "O₂",
    title: "Oxygen (% by volume)",
    scale: ["20.9", "17", "13"],
    note: "Below 13 % ISO 13571 asks for the low-oxygen dose too; it is in the tenability index throughout.",
    level: (f, v) => (20.9 - f.o2[v]) / (20.9 - 13),
    extras: true,
  },
};

export const SHOW_GROUPS = [
  { value: "heat", label: "Heat" },
  { value: "smoke", label: "Smoke" },
  { value: "gases", label: "Gases" },
  { value: "tenability", label: "Tenability" },
];

/**
 * What to draw for the settings: {kind: "heat" | "smoke" | "levels", key,
 * legend, levels?(frame, ti)}. Asking for a measure the run did not record
 * (CO2 or O2 without the extra measures) falls back to CO.
 */
export function measureView(settings, hasExtras) {
  const show = settings.flamescapesShow || "heat";
  if (show === "smoke") {
    const m = SMOKE_MEASURES[settings.flamescapesSmokeMeasure] || SMOKE_MEASURES.visibility;
    return {
      kind: "smoke",
      key: `smoke:${settings.flamescapesSmokeMeasure || "visibility"}`,
      legend: {
        title: m.title,
        ramp: "smoke",
        scale: m.scale,
        note: "Flamescapes: soot at head height (1.8 m above each node's floor), read through Bouguer's law with 8.7 m²/g (Mulholland and Croarkin 2000); clear air keeps the graph's colour.",
      },
    };
  }
  if (show === "gases") {
    let id = settings.flamescapesGasMeasure || "co";
    if (GAS_MEASURES[id]?.extras && !hasExtras) id = "co";
    const m = GAS_MEASURES[id] || GAS_MEASURES.co;
    return {
      kind: "levels",
      key: `gases:${id}`,
      legend: { title: m.title, scale: m.scale, note: m.note },
      levels: (frame) => {
        const n = frame.T.length;
        const out = new Float32Array(n);
        for (let v = 0; v < n; v += 1) out[v] = m.level(frame, v);
        return out;
      },
    };
  }
  if (show === "tenability") {
    return {
      kind: "levels",
      key: "tenability",
      legend: {
        title: "Tenability index",
        scale: ["0", "0.5", "1: untenable"],
        note: "The largest share of a limit reached: gas temperature, visibility, and the asphyxiant and heat doses against FED 0.3 (ISO 13571). Under an uncalibrated model.",
      },
      levels: (frame, ti) => ti,
    };
  }
  // Heat on the fixed 20 to 600 °C ramp, or fitted to the run (fittedHeatPeak).
  const fit = settings.flamescapesHeatScale === "run";
  return { kind: "heat", key: fit ? "heat:fit" : "heat", legend: null, fit };
}

/**
 * The top of the heat ramp fitted to a run (settings.flamescapesHeatScale
 * "run"): its hottest gas at head height so far, rounded up to 10 °C and at
 * least 10 °C above the 20 °C ambient, so a small fire still shades the graph.
 * Drawing only: the run, its card and its limits are the same either way.
 */
export function fittedHeatPeak(hottest, ambient = 20) {
  const top = Number.isFinite(hottest) ? hottest : ambient;
  return Math.max(ambient + 10, Math.ceil(top / 10) * 10);
}
