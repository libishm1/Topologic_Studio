/**
 * What the Flamescapes solver is validated for, quantity by quantity, from
 * held-out measured fire tests (wiki/research/fire-calibration-claim-protocol-2026-11-12.md,
 * written before the comparisons; results in fire-calibration-plan-2026-11-11.md,
 * section 12, and benchmark/results/2026-11-12-validation-*.json). One table, so
 * the run card, the panel, Methods and the describer say the same thing.
 *
 * "calibrated": within the FDS Validation Guide's calibrated band (bias
 * 0.8 to 1.25, scatter at most 0.35) in at least three independent datasets
 * and outside the indicative band in none; "indicative": within 0.67 to 1.5,
 * scatter at most 0.5; otherwise "not calibrated".
 */

/** The cell sizes the held-out tests ran on, m. */
export const TESTED_CELLS = [0.2, 0.27, 0.34];

/**
 * Per quantity: its status on each tested cell, and a short note. Held-out:
 * Steckler (NBS 1982), NBS multi-room (1991), NIST/NRC (2006) and WTC (2005).
 */
export const VALIDATION = [
  {
    id: "layerDepth",
    quantity: "smoke layer depth",
    status: { 0.2: "indicative", 0.27: "calibrated", 0.34: "indicative" },
    note: "bias 1.05, scatter 0.15 on 0.27 m cells, in band in all four held-out test series",
  },
  { id: "layerTemperature", quantity: "smoke layer temperature", status: { 0.2: "not calibrated", 0.27: "indicative", 0.34: "indicative" }, note: "bias 1.06 on 0.27 m cells; NBS's multi-room tests scatter too much" },
  { id: "ceilingJet", quantity: "gas temperature under the ceiling", status: { 0.27: "indicative", 0.34: "indicative" }, note: "bias 1.18 on 0.27 m cells; measured in two series from one laboratory" },
  { id: "gases", quantity: "oxygen and carbon dioxide", status: { 0.27: "not calibrated", 0.34: "not calibrated" }, note: "the WTC tests read 1.6 to 1.8 times their measured change" },
  { id: "doorFlow", quantity: "flow through a door", status: { 0.2: "not calibrated", 0.27: "not calibrated", 0.34: "not calibrated" }, note: "the peak outflow reads two thirds of measured; a door is a few cells wide" },
  { id: "visibility", quantity: "smoke visibility", status: {}, note: "not yet compared with measurements" },
];

/**
 * What the held-out tests covered. Walls: lightweight insulating boards in
 * every test; the app gives every wall the solver's stone, which they did not.
 */
export const COVERED = {
  fires: "prescribed fires of 32 kW to 3.2 MW",
  rooms: "rooms of 17 to 584 m3, open or closed, one with a corridor",
  walls: "insulating-board walls (the app's walls are stone, which no test had)",
};

/**
 * Sprinkler activation, reported apart: the walls' convection coefficient
 * was set on these tests (2026-11-08), so they are not fully held out.
 */
export const SPRINKLERS = "the first head's activation, by the app's estimate, within the calibrated band against NIST, University of Canterbury and UL tests with their measured fires (bias 0.95 to 1.18)";

/** The tested cell size nearest a run's, or null when none is within 0.035 m. */
export function nearestTestedCell(cell) {
  if (!(cell > 0)) return null;
  let best = null;
  for (const c of TESTED_CELLS) if (Math.abs(c - cell) <= 0.035 && (best === null || Math.abs(c - cell) < Math.abs(best - cell))) best = c;
  return best;
}

/** Each quantity's status for a run on `cell` m cells ("not tested on cells this size" where none is near). */
export function statusFor(cell) {
  const near = nearestTestedCell(cell);
  return VALIDATION.map((v) => ({ ...v, on: near, said: near === null ? "not tested on cells this size" : v.status[near] || "not calibrated" }));
}

/** One sentence for a run on `cell` m cells: what holds, what is indicative, what is not. */
export function validationLine(cell) {
  const near = nearestTestedCell(cell);
  if (near === null) return `Not compared with measured fires on ${cell.toFixed(2)} m cells (tested on ${TESTED_CELLS.join(", ")} m): read it as not calibrated.`;
  const rows = statusFor(cell);
  const list = (s) => rows.filter((r) => r.said === s).map((r) => r.quantity);
  const join = (a) => (a.length > 1 ? `${a.slice(0, -1).join(", ")} and ${a.at(-1)}` : a[0]);
  const parts = [];
  if (list("calibrated").length) parts.push(`${join(list("calibrated"))} calibrated`);
  if (list("indicative").length) parts.push(`${join(list("indicative"))} indicative`);
  if (list("not calibrated").length) parts.push(`${join(list("not calibrated"))} not calibrated`);
  const nearNote = Math.abs(near - cell) > 1e-6 ? `, the nearest tested (${near} m)` : "";
  return `Against held-out measured fires on cells of this size${nearNote}: ${parts.join("; ")}. Those tests had ${COVERED.walls}.`;
}
