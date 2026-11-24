/**
 * What --store-species keeps of a run (validation-suite.mjs; plan 16.54, the
 * scope the user set on 2026-11-24): each second, at the cells the egress
 * would read and at the cells of the test's own gas analysers and smoke
 * meters, the model's gas temperature, soot density, CO, CO2 and O2.
 * Storing judges nothing: no pair, no statistic and no verdict is made here,
 * and nothing stored is compared until plan 16.55 fixes how.
 *
 * The sites:
 *   - each tree's head-height cells, the ones series.head keeps (centres
 *     1.5 - h to 2.0 + h over the tree's model floor), and among the tree's
 *     kept cells the one the app would read: the cell that holds 1.8 m over
 *     the floor, or the nearest kept cell within two above or below
 *     (flamescapesRun.js headCells: 0, -1, +1, -2, +2);
 *   - each gas analyser (o2, co2, co, soot) at its own cell, as the suite
 *     reads it for its comparisons, and each smoke meter at the cells along
 *     its path (the suite's nine samples).
 *
 * The units are the solver's: T in K (sim.T); soot in kg/m3 (sim.rho times
 * the soot mass fraction, what the app's visibility 3 / (8700 x soot) takes);
 * co, co2, o2 and h2o (water vapour, kept for a dry reading) as mass
 * fractions (sim.Y); and mol, the gas's moles per kg
 * (the sum of Y / M over the solver's species, the rest as nitrogen at
 * 0.028014 kg/mol, as the suite's own moleFraction), so that a wet volume
 * fraction is Y / M / mol with M the species' molar mass in kg/mol.
 *
 * Each series is one string of comma-separated numbers, one a second from
 * `from` in steps of `step` (an empty field is a reading that was not finite),
 * so that a results file stays readable in size; decode() gives the numbers.
 */

/** The reading height of the app (flamescapesRun.js headCells, headHeight). */
export const APP_HEAD_HEIGHT = 1.8;

/** The fields kept and their units. */
export const UNITS = Object.freeze({
  T: "K, gas temperature (sim.T)",
  soot: "kg/m3, soot density (sim.rho x the soot mass fraction)",
  co: "kg/kg, CO mass fraction (sim.Y)",
  co2: "kg/kg, CO2 mass fraction (sim.Y)",
  o2: "kg/kg, O2 mass fraction (sim.Y)",
  h2o: "kg/kg, water vapour mass fraction (sim.Y); for a dry reading, as an analyser behind a water trap takes it: the wet volume fraction over 1 less water's",
  mol: "mol/kg, the gas's moles per kg (sum of Y / M, the rest as N2 at 0.028014 kg/mol); a wet volume fraction is Y / M / mol",
});

/** Molar masses for the conversion above, kg/mol: the solver's own (flamescapes.js MOLAR). */
export const MOLAR_KG = Object.freeze({ co: 0.028, co2: 0.044, o2: 0.032, h2o: 0.018 });

/** The quantities of a test's points that are gas analysers or smoke meters. */
export const SPECIES_QUANTITIES = Object.freeze(["o2", "co2", "co", "soot", "smoke"]);

/**
 * Among a tree's kept cells (centre heights z over its model floor, rising),
 * the index of the cell the app would read at `height`: the cell that holds
 * it, else the nearest kept cell one or two cells below or above, below
 * first (the app's order 0, -1, +1, -2, +2); -1 when none.
 */
export function appCellIndex(z, h, height = APP_HEAD_HEIGHT) {
  const want = Math.floor(height / h + 1e-9);
  const level = z.map((x) => Math.round(x / h - 0.5));
  for (const d of [0, -1, 1, -2, 2]) {
    const i = level.indexOf(want + d);
    if (i >= 0) return i;
  }
  return -1;
}

/** The indices of a tree's head-height cells, as series.head picks them. */
export function headIndices(z, h) {
  return z.map((x, i) => (x >= 1.5 - h && x <= 2.0 + h ? i : -1)).filter((i) => i >= 0);
}

/**
 * The sites of a run and the distinct cells they read.
 *   trees:  [{ id, z: [heights over the model floor], cells: [cell index] }]
 *   points: [{ id, quantity, x, y, z, cell: { c, moved } | null, pathCells?: [cell index] }]
 * Returns { cells: [distinct cell indices], sites: { trees, points } } where
 * every `at` is an index into cells.
 */
export function speciesSites(trees, points, h) {
  const cells = [];
  const seen = new Map();
  const slot = (c) => {
    if (!seen.has(c)) {
      seen.set(c, cells.length);
      cells.push(c);
    }
    return seen.get(c);
  };
  const r3 = (x) => +x.toFixed(3);
  const outTrees = {};
  for (const tr of trees || []) {
    if (!tr.z || !tr.cells || !tr.z.length) continue;
    const pick = headIndices(tr.z, h);
    const app = appCellIndex(tr.z, h);
    if (!pick.length && app < 0) continue;
    outTrees[tr.id] = {
      z: pick.map((i) => r3(tr.z[i])),
      at: pick.map((i) => slot(tr.cells[i])),
      app: app < 0 ? null : { height: APP_HEAD_HEIGHT, z: r3(tr.z[app]), at: slot(tr.cells[app]), moved: Math.round(tr.z[app] / h - 0.5) - Math.floor(APP_HEAD_HEIGHT / h + 1e-9) },
    };
  }
  const outPoints = {};
  for (const p of points || []) {
    if (!SPECIES_QUANTITIES.includes(p.quantity) || !p.cell) continue;
    const list = p.pathCells?.length ? p.pathCells : [p.cell.c];
    outPoints[p.id] = { quantity: p.quantity, x: p.x, y: p.y, z: p.z, ...(p.path ? { path: p.path } : {}), moved: p.cell.moved ?? null, ...(p.stored ? { stored: p.stored } : {}), at: list.map(slot) };
  }
  return { cells, sites: { trees: outTrees, points: outPoints } };
}

/**
 * A recorder over the solver's state: read() takes one reading at every cell.
 *   state: { T, rho, Y } typed arrays; NS species a cell; SP: { SOOT, CO, CO2, O2 };
 *   MOLAR: kg/mol by species index. SP also gives H2O.
 */
export function recorder(state, cells, { NS, SP, MOLAR, MN2 = 0.028014 }) {
  const rows = { T: cells.map(() => []), soot: cells.map(() => []), co: cells.map(() => []), co2: cells.map(() => []), o2: cells.map(() => []), h2o: cells.map(() => []), mol: cells.map(() => []) };
  const read = () => {
    cells.forEach((c, q) => {
      let inv = 0, rest = 1;
      for (let s = 0; s < NS; s += 1) {
        const y = state.Y[c * NS + s];
        inv += y / MOLAR[s];
        rest -= y;
      }
      inv += Math.max(0, rest) / MN2;
      rows.T[q].push(state.T[c]);
      rows.soot[q].push(state.rho[c] * state.Y[c * NS + SP.SOOT]);
      rows.co[q].push(state.Y[c * NS + SP.CO]);
      rows.co2[q].push(state.Y[c * NS + SP.CO2]);
      rows.o2[q].push(state.Y[c * NS + SP.O2]);
      rows.h2o[q].push(state.Y[c * NS + SP.H2O]);
      rows.mol[q].push(inv);
    });
  };
  return { rows, read };
}

/** A series as one string: T to 0.01 K, the others to five significant figures. */
export function encode(values, field) {
  const one = field === "T" ? (x) => String(+x.toFixed(2)) : (x) => String(+x.toPrecision(5));
  return values.map((x) => (Number.isFinite(x) ? one(x) : "")).join(",");
}

/** The numbers of an encoded series (NaN where a reading was not finite). */
export function decode(text) {
  if (text === "") return [];
  return text.split(",").map((s) => (s === "" ? NaN : Number(s)));
}

/** What a run keeps under series.species. */
export function speciesSeries(rec, cells, sites) {
  const out = { units: UNITS, format: "each series is a string of comma-separated numbers, one a second (lib/species.mjs decode); T to 0.01 K, the others to five significant figures", cells, sites };
  for (const f of Object.keys(UNITS)) out[f] = rec.rows[f].map((v) => encode(v, f));
  return out;
}

/** The dry volume fraction from a wet one and water's wet volume fraction (decoded arrays). */
export function dryFraction(x, xWater) {
  return x.map((v, i) => v / (1 - xWater[i]));
}

/** The wet volume fraction of a species from stored mass fractions and mol (decoded arrays). */
export function volumeFraction(y, mol, molarKg) {
  return y.map((v, i) => v / molarKg / mol[i]);
}
