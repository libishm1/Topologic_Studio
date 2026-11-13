/**
 * What the Flamescapes solver is validated for, quantity by quantity, from
 * held-out measured fire tests (wiki/research/fire-calibration-claim-protocol-2026-11-12.md,
 * written before the comparisons; results in fire-calibration-plan-2026-11-11.md,
 * section 13.27, and benchmark/results/2026-11-13-validation-13b-*.json). One
 * table, so the run card, the panel, Methods and the describer say the same thing.
 *
 * "calibrated": within this project's calibrated band (bias 0.8 to 1.25,
 * scatter at most 0.35; 2026-11-08 plan, section 8), computed with the FDS
 * Validation Guide's statistics, in at least three independent datasets and
 * outside the indicative band (0.67 to 1.5, at most 0.5) in none;
 * "indicative": within the indicative band; otherwise "not calibrated".
 * The guide defines the statistics, not the bands.
 */

/** The cell sizes the held-out tests ran on, m. */
export const TESTED_CELLS = [0.2, 0.27, 0.34];

/**
 * The cell size the app's tested grid builds, m: the middle of the three the
 * held-out tests ran on. Named CALIBRATED_CELL until 13.B, whose strictest
 * reading of the smoke layer's depth on it is "indicative".
 */
export const TESTED_GRID_CELL = 0.27;

/** A run's cells count as a tested size within this, m; within NEAR_CELL they borrow its verdict, at most "indicative". */
export const EXACT_CELL = 0.005;
export const NEAR_CELL = 0.035;

/**
 * Per quantity: its status on each tested cell, from the held-out run 13.B
 * (solver 550fce5, each test's walls in layers and its stated leaks, 178 runs
 * a cell; plan 13.27). Each word is the strictest of 13.B's readings: plan
 * 12.6 and protocol section 4, unweighted and weighted, every run and the
 * 2026-11-12 rule, default leak and sealed (13.16), eight series and nine
 * (15.9). Held-out: Steckler (NBS 1982), NBS multi-room (1991), NIST/NRC
 * (2006), WTC (2005), ATF corridors (2008), UL FSRI compartments (2018),
 * UL/NIST vents and NIST FSE (2008); NBS and NIST count as one laboratory.
 *
 * `figures`: the pooled bias and scatter behind a "calibrated", per cell.
 * `oneLab`: the cells on which every series in band is one laboratory's (13.1).
 * `before`: the word of 13.A (solver d3c5751, plan 13.14) where it differs (13.5).
 */
export const VALIDATION = [
  {
    id: "layerDepth",
    quantity: "smoke layer depth",
    status: { 0.2: "calibrated", 0.27: "indicative", 0.34: "calibrated" },
    figures: { 0.2: "bias 1.00, scatter 0.14, in band in six of seven series", 0.34: "bias 1.11, scatter 0.14, in band in five of seven series" },
    before: { 0.27: "calibrated" },
    note: "on 0.27 m cells in band in all seven series as first run (bias 1.05, scatter 0.12); indicative once the three tests whose 0.24 m door the grid closes are sealed, as the plan asked (scatter 0.23)",
  },
  {
    id: "layerTemperature",
    quantity: "smoke layer temperature",
    status: { 0.2: "indicative", 0.27: "not calibrated", 0.34: "not calibrated" },
    oneLab: [0.2],
    note: "reads warm (pooled bias 1.35 to 1.48): in band in NIST's two series, 1.4 to 2.2 times the measured rise in UL's small lined rooms",
  },
  {
    id: "ceilingJet",
    quantity: "gas temperature under the ceiling",
    status: { 0.2: "indicative", 0.27: "not calibrated", 0.34: "not calibrated" },
    oneLab: [0.2],
    note: "in band in NIST's two series only (pooled bias 1.37 to 1.45)",
  },
  {
    id: "headTemperature",
    quantity: "gas temperature at head height",
    status: { 0.2: "not calibrated", 0.27: "not calibrated", 0.34: "not calibrated" },
    before: { 0.2: "indicative" },
    note: "reads warm: pooled bias 1.33 to 1.56, and 1.53 to 1.77 with each series weighted alike",
  },
  {
    id: "co2",
    quantity: "carbon dioxide",
    status: { 0.2: "indicative", 0.27: "indicative", 0.34: "not calibrated" },
    oneLab: [0.2, 0.27],
    note: "in band in NIST/NRC only, of three series",
  },
  { id: "o2", quantity: "oxygen", status: { 0.2: "not calibrated", 0.27: "not calibrated", 0.34: "not calibrated" }, note: "in band in NIST/NRC only, on 0.20 and 0.27 m cells; elsewhere the drop reads well above measured" },
  { id: "doorFlow", quantity: "flow through a door", status: { 0.2: "not calibrated", 0.27: "not calibrated", 0.34: "not calibrated" }, note: "door and vent velocities are in band in none of five series (pooled scatter 0.59 to 1.19); a door is a few cells wide" },
  {
    id: "neutralPlane",
    quantity: "neutral plane in a door",
    status: { 0.2: "calibrated", 0.27: "indicative", 0.34: "not calibrated" },
    figures: { 0.2: "bias 0.90, scatter 0.12, in band in three of four series" },
    before: { 0.2: "indicative", 0.34: "indicative" },
    note: "the height where a door's flow turns, not its speed: the velocities through the same doors are not calibrated",
  },
  { id: "visibility", quantity: "smoke visibility", status: { 0.2: "not calibrated", 0.27: "not calibrated", 0.34: "not calibrated" }, note: "smoke obscuration reads 1.6 times measured in the one series compared (NIST/NRC)" },
];

/**
 * What the held-out tests covered: prescribed (measured) heat release, so
 * they judge the smoke's transport, not the fire's growth; each test's own
 * walls, conducted in layers, where the app gives every wall the solver's
 * stone. One test (UL/NIST 1) ran to 1,215 s; the others to 900 s at most.
 */
export const COVERED = {
  peakKW: 3200,
  horizon: 900,
  fires: "their measured heat release (32 kW to 3.2 MW)",
  rooms: "rooms of 17 to 584 m³, open or closed, and corridors",
  times: "up to 900 s",
  walls: "each test's own walls in layers (insulating board, gypsum board, fire brick or concrete)",
};

/** The tested cell size nearest a run's, or null when none is within NEAR_CELL. */
export function nearestTestedCell(cell) {
  if (!(cell > 0)) return null;
  let best = null;
  for (const c of TESTED_CELLS) if (Math.abs(c - cell) <= NEAR_CELL && (best === null || Math.abs(c - cell) < Math.abs(best - cell))) best = c;
  return best;
}

/**
 * Each quantity's status for a run on `cell` m cells. On a tested size (within
 * EXACT_CELL) its verdict; near one, the same at most "indicative" (the
 * verdict holds on the cells it was found on); else not tested.
 */
export function statusFor(cell) {
  const near = nearestTestedCell(cell);
  const exact = near !== null && Math.abs(near - cell) <= EXACT_CELL;
  return VALIDATION.map((v) => {
    if (near === null) return { ...v, on: null, said: "not tested on cells this size" };
    const s = v.status[near] || "not calibrated";
    return { ...v, on: near, said: !exact && s === "calibrated" ? "indicative" : s };
  });
}

const join = (a) => (a.length > 1 ? `${a.slice(0, -1).join(", ")} and ${a.at(-1)}` : a[0]);

/**
 * The run card's one line, first on the card (validationLine follows under
 * More): what the held-out tests support on these cells for what the run
 * draws, its visibility and gas temperature (the weakest of head height's, the
 * layer's and the ceiling's), and the quantities calibrated here that it does
 * not draw.
 */
export function validationVerdict(cell, { peakKW = null, horizon = null } = {}) {
  const outside = (peakKW !== null && peakKW > COVERED.peakKW) || (horizon !== null && horizon > COVERED.horizon);
  const tail = outside ? " Outside the tested range." : "";
  const size = `${cell.toFixed(2)} m`;
  if (nearestTestedCell(cell) === null) return `Not compared with measured fires on ${size} cells: read it as a ranking, not a prediction.${tail}`;
  const rows = statusFor(cell);
  const said = (id) => rows.find((r) => r.id === id).said;
  const rank = { "not calibrated": 0, indicative: 1, calibrated: 2 };
  const temperature = [said("headTemperature"), said("layerTemperature"), said("ceilingJet")].sort((a, b) => rank[a] - rank[b])[0];
  const visibility = said("visibility");
  const drawn = visibility === temperature ? `the visibility and temperature drawn are ${visibility}` : `the visibility drawn is ${visibility}, the temperature ${temperature}`;
  const calibrated = rows.filter((r) => r.said === "calibrated").map((r) => r.quantity);
  const line = calibrated.length ? `${join(calibrated)} calibrated on these ${size} cells, not drawn; ${drawn}` : `On these ${size} cells ${drawn}`;
  return `${line[0].toUpperCase()}${line.slice(1)}.${tail}`;
}

/**
 * The run card's and the describer's sentence for a run on `cell` m cells,
 * with its peak heat release (kW) and length (s): what decides its untenable
 * places, then what the held-out tests support on these cells, and under what.
 */
export function validationLine(cell, { peakKW = null, horizon = null } = {}) {
  const lead = "This run's untenable places come from smoke visibility and gas temperature at head height; it reports no layer depth.";
  const near = nearestTestedCell(cell);
  if (near === null) return `${lead} Not compared with measured fires on ${cell.toFixed(2)} m cells (tested on ${TESTED_CELLS.join(", ")} m): read it as not calibrated.`;
  const rows = statusFor(cell);
  const of = (s) => rows.filter((r) => r.said === s);
  const parts = [];
  // A calibrated quantity carries its own figures.
  if (of("calibrated").length) parts.push(`${join(of("calibrated").map((r) => `${r.quantity} (${r.figures[near]})`))} calibrated`);
  if (of("indicative").length) {
    // Where every series in band is one laboratory's, the verdict says so (plan 13.1).
    const one = (r) => r.oneLab?.includes(near);
    const all = of("indicative").every(one);
    const names = of("indicative").map((r) => (one(r) && !all ? `${r.quantity} (from one laboratory's tests)` : r.quantity));
    parts.push(`${join(names)} indicative${all ? `, ${names.length > 1 ? "each " : ""}from one laboratory's tests` : ""}`);
  }
  if (of("not calibrated").length) parts.push(`${join(of("not calibrated").map((r) => r.quantity))} not calibrated`);
  const exact = Math.abs(near - cell) <= EXACT_CELL;
  const where = exact ? "on cells of this size" : `near cells of this size (tested on ${near} m; a verdict holds only on the cells it was found on)`;
  const outside = (peakKW !== null && peakKW > COVERED.peakKW) || (horizon !== null && horizon > COVERED.horizon);
  return (
    `${lead} Against held-out measured fires ${where}: ${parts.join("; ")}. ` +
    `Those tests prescribed ${COVERED.fires} in ${COVERED.rooms}, for ${COVERED.times}, with ${COVERED.walls}; this run predicts its own heat release, which is not calibrated, and its walls are stone` +
    (outside ? ", and it lies outside the tested range." : ".")
  );
}

/**
 * The whole table in one sentence, for the panel's notes and Methods: each
 * quantity with a verdict and the cells it holds on, then the ones with none.
 */
export function validationSummary() {
  const cells = (list) => `${join(list.map((c) => c.toFixed(2)))} m`;
  const some = [];
  const none = [];
  for (const v of VALIDATION) {
    const on = (s) => TESTED_CELLS.filter((c) => v.status[c] === s);
    const parts = [];
    if (on("calibrated").length) parts.push(`calibrated on ${cells(on("calibrated"))}`);
    if (on("indicative").length) parts.push(`indicative on ${cells(on("indicative"))}${on("indicative").every((c) => v.oneLab?.includes(c)) ? " (from one laboratory's tests)" : ""}`);
    if (parts.length) some.push(`${v.quantity} ${parts.join(" and ")}`);
    else none.push(v.quantity);
  }
  return `Checked against held-out measured room fires (eight test series, 2026-11-13) on cells of ${cells(TESTED_CELLS)}: ${some.join("; ")}; ${join(none)} not calibrated on any.`;
}

/** The words of the comparison before it (13.A, an earlier solver) where they differ, in one sentence (plan 13.5). */
export function validationBefore() {
  const was = VALIDATION.flatMap((v) => Object.entries(v.before ?? {}).map(([cell, word]) => `${v.quantity} on ${Number(cell).toFixed(2)} m read ${word}`));
  return `The comparison before it, on the solver of 2026-11-12, read otherwise in places: ${join(was)}.`;
}
