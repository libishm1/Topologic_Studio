/**
 * Sprinkler activation and suppression for the Flamescapes runs, shared by
 * the CPU solver (flamescapes.js, flamescapesRun.js) and the GPU port
 * (flamescapesGpu.js): where each head sits on the fire grid, when its link
 * opens, and what an open head does to the fire
 * (wiki/research/sprinkler-physics-2026-11-07.md sections 3 to 7;
 * wiki/research/fire-suppression-sprinklers-2026-11-07.md section 5).
 *
 * Activation. Each head's link follows the RTI equation with conduction to
 * its mount, dT/dt = (sqrt(u) (Tg - T) - C (T - Tm)) / RTI (Heskestad and
 * Bill 1988; the FDS and CFAST form without the wet term), stepped exactly
 * for gas held over each frame at the mean of the frame's two readings
 * (Sprinklers.lean exactStep, exactStep_add), so no step size bounds it.
 * The gas is read two ways (section 4.4):
 *   - "late": the top open cell of the head's column and the speed there,
 *     floored at 0.1 m/s as CFAST does; a 0.3 to 0.4 m cell averages the
 *     whole ceiling jet and reads it low, so this is the late bound;
 *   - "estimate", CFAST's hybrid: for a head under the fire's own ceiling,
 *     the layer temperature from the grid plus Alpert's ceiling-jet excess
 *     for the run's heat release, the ceiling's height above the fire and
 *     the head's distance from it, and the larger of Alpert's jet speed and
 *     the cell's; elsewhere the cell, as the late bound.
 * A head opens when its link reaches its activation temperature, at the
 * exact crossing time inside the frame (Sprinklers.lean
 * linkExact_activation). The fire is changed from the end of that frame.
 *
 * Suppression, after a head opens:
 *   - Evans (1993, NISTIR 5254) as a cap on every fuel cell under the
 *     head's footprint: its reaction is at most its rate in the step after
 *     the head opened times exp(-(t - t_act)/tau), tau = 3.0 w''^-1.85 s for
 *     the local spray density w'' (mm/s, all open heads over the cell
 *     added; Sprinklers.lean evansTau; Suppression.lean evansCap_antitone,
 *     evansCap_sum: the cap never rises and its sum is at most r (tau + dt);
 *     evans_cells: the covered cells' total stays under one envelope). Below 0.026 mm/s,
 *     the lowest density in Evans's fit, the cell is left alone: Walton's
 *     runs at 0.013 and 0.014 mm/s showed no measurable effect and Evans
 *     left them out (NISTIR 5254, with one at 0.039 mm/s on the 610 mm crib). Evans
 *     is calibrated on full-scale furnishings and is conservative: the
 *     least reduction to expect.
 *   - The spray (section 7.2, Fire-X's energy-limited evaporation on the
 *     grid): each open head's water falls through the visible columns of
 *     its footprint, evaporating in each cell a fraction set by the gas's
 *     excess over boiling (1 mm droplets: 1.7 % per 0.35 m cell at 300 K
 *     over, the report's estimate from the FDS correlations), at most what
 *     cools the cell to boiling. Steam joins the gas (the oxygen is
 *     diluted: vapour_dilutes_O2) and the gas cools, never below boiling
 *     (evaporate_step); the falling water drags the gas down; what is left
 *     reaches the floor, so the water emitted is the water evaporated plus
 *     the water at the floor (spray_budget_merged: heads sharing a column
 *     are one entry).
 *     The fuel is not wetted in this mode.
 *   - Wetting ("wetting" mode, in place of Evans's cap): the spray as above,
 *     and in each fuel cell the water passes, a share WETTED_FRACTION of
 *     what is left soaks into the fuel's surface layer (its `moisture`),
 *     mixed in at the water's temperature so the layer keeps its energy.
 *     The solver's own drying then holds the surface near 100 C while the
 *     water lasts, which slows its pyrolysis and oxidation: the water, not an
 *     envelope, slows the fire. On a 0.35 m grid a fully wetted cell would
 *     stop at once, far faster than any of Evans's furnishings, so the share
 *     is calibrated (tools/wetting-calibrate.mjs) to Evans's decay at the
 *     light hazard design density on the test room; at other densities it
 *     follows from the water, not from Evans's fit. Emitted water =
 *     evaporated + wetted + at the floor.
 */
const GAS = 0;
const POROUS = 1;
const SOLID = 2;

/** Water's boiling point, K. */
export const T_BOIL = 373.15;
/** The sprinkler water's temperature, K. */
export const WATER_TEMPERATURE = 293.15;
/** Liquid water's and vapour's heat capacities, J/(kg K) (the solver's). */
export const WATER_CP = 4186;
export const VAPOUR_CP = 1864;
/**
 * Evaporation per cell: 1 - exp(-k (T - T_boil) h), k per kelvin per metre,
 * for 1 mm droplets (the report's section 7.1: 1.7 % across a 0.35 m cell
 * at 300 K over boiling). 0.5 mm droplets would be about five times faster.
 */
export const DROPLET_EVAPORATION = -Math.log(1 - 0.017) / (300 * 0.35);
/** Droplets fall at about this speed (m/s), for their drag on the gas. */
export const TERMINAL_SPEED = 5;
/** CFAST's least gas speed at a link (m/s): a link in still hot gas still heats. */
export const SPEED_FLOOR = 0.1;
/** The lowest spray density (mm/s) in Evans's fit; below it the cap is off (outside the data). */
/**
 * The share of the water reaching a fuel cell's top surface that soaks in,
 * in "wetting" mode (the rest runs off to the floor). Calibrated on the test
 * room (tools/wetting-calibrate.mjs, benchmark/results/2026-11-07-wetting-calibration.txt):
 * at 0.068 mm/s (NFPA 13 light hazard) wetting burns as much of the covered
 * fuel in the 120 s after the head opens as Evans's cap does (7.4 kg, against
 * 21.3 kg with no water). At other densities it follows the water: 9.0 kg at
 * 0.026 mm/s and 4.5 kg at 0.133 mm/s, where the cap gives 8.6 and 7.4 kg.
 */
export const WETTED_FRACTION = 0.52;
export const EVANS_MIN_DENSITY = 0.026;

/** Evans's decay time (s) at spray density w (mm/s): 3.0 w^-1.85 (Sprinklers.lean evansTau). */
export const evansTau = (w) => 3.0 * w ** -1.85;

/**
 * Alpert's ceiling jet (SI, Q in kW, as quoted by Zeng et al. 2023): the
 * excess temperature (K) and speed (m/s) at distance r (m) from the plume
 * under a ceiling H (m) above the fire.
 */
export function alpert(Q, H, r) {
  if (!(Q > 0) || !(H > 0)) return { dT: 0, U: 0 };
  const dT = r / H <= 0.18 ? (16.9 * Q ** (2 / 3)) / H ** (5 / 3) : (5.38 * (Q / r) ** (2 / 3)) / H;
  const U = r / H <= 0.15 ? 0.947 * (Q / H) ** (1 / 3) : (0.197 * Q ** (1 / 3) * H ** 0.5) / r ** (5 / 6);
  return { dT, U };
}

/**
 * The exact link step for gas held at Tg and speed u over dt
 * (Sprinklers.lean exactStep with a = sqrt(u)/RTI, b = C/RTI): the link
 * relaxes toward T_inf = (a Tg + b Tm)/(a + b) at the rate a + b.
 */
export function linkStep(T, Tg, Tm, u, rti, c, dt) {
  const a = Math.sqrt(Math.max(u, 0)) / rti;
  const b = c / rti;
  const k = a + b;
  if (!(k > 0)) return { T, Tinf: T, k: 0 };
  const Tinf = (a * Tg + b * Tm) / k;
  return { T: Tinf + (T - Tinf) * Math.exp(-k * dt), Tinf, k };
}

/** Time (s) for a link at T0 relaxing toward Tinf at rate k to reach Ta (Sprinklers.lean linkExact_activation). */
export function crossing(T0, Tinf, k, Ta) {
  return Math.log((Tinf - T0) / (Tinf - Ta)) / k;
}

const planAxes = (upAxis) => (upAxis === "z" ? [0, 1] : upAxis === "x" ? [1, 2] : [0, 2]);

/**
 * Each head on the fire grid, from its record (sprinklers.js): the sensor
 * cell (the top open cell of its column: climb from the link point to the
 * first solid cell; down to the first open one if the point is in a slab),
 * the visible columns of its footprint with their open cells down to the
 * fuel or floor, the fuel cells under them, and whether it is under the
 * fire's own ceiling (a clear line from the plume's ceiling cell at the
 * ceiling, so walls and lintels part rooms) with its distance from the fire.
 *
 * @param {object} scene  buildScene's (grid, solid, ignition, contents)
 * @param {Array} heads   sprinkler records: position (viewer frame), footprint {centre, axis, half} in plan
 * @param {object} opts   upAxis; fuel: Float32Array or Float64Array of each cell's fuel mass (optional)
 */
export function sprinklerSetup(scene, heads, { upAxis = "y", fuel = null } = {}) {
  const { grid: g, solid } = scene;
  const at = (i, j, k) => i + g.nx * (j + g.ny * k);
  const ijk = (c) => [c % g.nx, Math.floor(c / g.nx) % g.ny, Math.floor(c / (g.nx * g.ny))];
  const inside = (i, j, k) => i >= 0 && j >= 0 && k >= 0 && i < g.nx && j < g.ny && k < g.nz;
  const fuelCells = new Uint8Array(g.cells ?? g.nx * g.ny * g.nz);
  for (const c of scene.contents?.cells || []) fuelCells[c] = 1;
  const isFuel = (c) => (fuel ? fuel[c] > 0 : fuelCells[c] === 1);
  // Local grid coordinates of a viewer point (the grid's own axes; b is up).
  const local = (p) => {
    const q = [p[g.axes[0]], p[g.axes[1]], p[g.axes[2]]];
    return [(q[0] - g.origin[0]) / g.h, (q[1] - g.origin[1]) / g.h, (q[2] - g.origin[2]) / g.h];
  };
  // Climb a column from row j to the cell under the first solid one.
  const topFrom = (i, j, k) => {
    if (!inside(i, j, k)) return -1;
    let jj = j;
    for (let d = 0; d < 3 && solid[at(i, jj, k)]; d += 1) jj -= 1;
    if (jj < 0 || solid[at(i, jj, k)]) return -1;
    while (jj + 1 < g.ny && !solid[at(i, jj + 1, k)]) jj += 1;
    return jj;
  };
  // A straight line between two cell centres crosses no solid cell.
  const clear = (a, b) => {
    const [ai, aj, ak] = ijk(a), [bi, bj, bk] = ijk(b);
    const steps = Math.ceil(2 * Math.max(Math.abs(bi - ai), Math.abs(bj - aj), Math.abs(bk - ak))) || 1;
    for (let s = 1; s < steps; s += 1) {
      const f = s / steps;
      const c = at(Math.floor(ai + 0.5 + f * (bi - ai)), Math.floor(aj + 0.5 + f * (bj - aj)), Math.floor(ak + 0.5 + f * (bk - ak)));
      if (solid[c]) return false;
    }
    return true;
  };

  // The plume: the ignition cell, and its ceiling.
  let plume = null;
  if (scene.ignition) {
    const [pi, pj, pk] = ijk(scene.ignition.cell);
    const top = topFrom(pi, pj, pk);
    if (top >= 0) plume = { i: pi, j: pj, k: pk, top: at(pi, top, pk), H: Math.max(0.5, (top - pj + 0.5) * g.h) };
  }

  // Plan axes of the head records: the grid's horizontal axes are local 0 and 2.
  const [pa, pb] = planAxes(upAxis);
  const out = heads.map((head) => {
    // The storey fallback's heads stand at the floor: climb from 1.5 m above it.
    const [x, y0, z] = local(head.position);
    const y = head.provenance?.ceiling === "unknown" ? y0 + 1.5 / g.h : y0;
    const i = Math.floor(x), k = Math.floor(z);
    const jTop = topFrom(i, Math.floor(y), k);
    const base = { ...head, sensor: -1, columns: [], covered: [], inSight: false, r: null };
    if (jTop < 0) return base;
    const sensor = at(i, jTop, k);
    // The footprint's columns, each from its own ceiling down to the fuel or the floor.
    const { centre, axis, half } = head.footprint || { centre: [head.position[pa], head.position[pb]], axis: [1, 0], half: [2.3, 2.3] };
    const reach = Math.ceil(Math.hypot(half[0], half[1]) / g.h) + 1;
    const columns = [], covered = [];
    for (let kk = k - reach; kk <= k + reach; kk += 1) {
      for (let ii = i - reach; ii <= i + reach; ii += 1) {
        if (!inside(ii, 0, kk)) continue;
        // The column's centre in the viewer frame, then in the footprint's axes.
        const world = [0, 0, 0];
        world[g.axes[0]] = g.origin[0] + (ii + 0.5) * g.h;
        world[g.axes[1]] = g.origin[1] + (jTop + 0.5) * g.h;
        world[g.axes[2]] = g.origin[2] + (kk + 0.5) * g.h;
        const da = world[pa] - centre[0], db = world[pb] - centre[1];
        const along = da * axis[0] + db * axis[1], across = -da * axis[1] + db * axis[0];
        if (Math.abs(along) > half[0] || Math.abs(across) > half[1]) continue;
        const top = topFrom(ii, jTop, kk);
        if (top < 0 || Math.abs(top - jTop) > 1) continue;
        const cTop = at(ii, top, kk);
        if (!clear(sensor, cTop)) continue;
        let count = 0;
        let j = top;
        while (j >= 0 && !solid[at(ii, j, kk)] && !isFuel(at(ii, j, kk))) {
          count += 1;
          j -= 1;
        }
        if (count) columns.push(cTop, count);
        // The fuel under the open run: consecutive fuel cells.
        while (j >= 0 && isFuel(at(ii, j, kk))) {
          covered.push(at(ii, j, kk));
          j -= 1;
        }
      }
    }
    const r = plume ? Math.hypot(i - plume.i, k - plume.k) * g.h : null;
    const inSight = Boolean(plume) && (sensor === plume.top || clear(plume.top, sensor));
    return { ...base, sensor, columns: Int32Array.from(columns), covered: Int32Array.from(covered), inSight, r };
  });
  return { heads: out, plume, h: g.h, nx: g.nx };
}

/**
 * Activation state: per head and reading, the link temperature and the
 * time it opened (NaN while shut), and the frame's previous readings.
 */
export function createActivation(setup, { ambient = 293.15 } = {}) {
  const n = setup.heads.length;
  const fill = (v) => new Float64Array(n).fill(v);
  return {
    ambient,
    link: { late: fill(ambient), estimate: fill(ambient) },
    opened: { late: fill(NaN), estimate: fill(NaN) },
    hrrAt: { late: fill(NaN), estimate: fill(NaN) },
    prevT: fill(ambient),
    prevU: fill(0),
    prevQ: 0,
  };
}

/**
 * One frame of activation: gas temperatures (K) and speeds (m/s) at the
 * sensor cells at the frame's end, its mean heat release (kW), and the
 * frame's start and end times. Each reading's links step exactly for the
 * gas held at the mean of the frame's two readings. Returns the heads that
 * opened in this frame under `reading`, with their times.
 */
export function activationFrame(setup, state, { t0, t1, T, u, hrr, reading = "estimate" }) {
  const dt = t1 - t0;
  const n = setup.heads.length;
  const plume = setup.plume;
  const Q = 0.5 * (state.prevQ + Math.max(0, hrr));
  // The fire's layer: the cells of the heads under its ceiling, outside the plume's jet core.
  let sum = 0, count = 0;
  for (let h = 0; h < n; h += 1) {
    const head = setup.heads[h];
    if (head.sensor < 0 || !head.inSight || !(head.r >= 0.2 * (plume?.H ?? 0))) continue;
    sum += 0.5 * (state.prevT[h] + T[h]);
    count += 1;
  }
  const layer = count ? sum / count : state.ambient;
  const opened = [];
  for (let h = 0; h < n; h += 1) {
    const head = setup.heads[h];
    if (head.sensor < 0) continue;
    const Tg = 0.5 * (state.prevT[h] + T[h]);
    const ug = Math.max(SPEED_FLOOR, 0.5 * (state.prevU[h] + u[h]));
    const Ta = head.activationC + 273.15;
    const gases = { late: [Tg, ug] };
    if (head.inSight && plume) {
      const jet = alpert(Q, plume.H, Math.max(head.r, 0.01));
      gases.estimate = [Math.max(Tg, layer + jet.dT), Math.max(ug, jet.U)];
    } else gases.estimate = gases.late;
    for (const name of ["late", "estimate"]) {
      const T0 = state.link[name][h];
      const [gas, speed] = gases[name];
      const step = linkStep(T0, gas, state.ambient, speed, head.rti, head.c, dt);
      state.link[name][h] = step.T;
      if (Number.isNaN(state.opened[name][h]) && step.T >= Ta && T0 < Ta) {
        const t = t0 + Math.min(dt, Math.max(0, crossing(T0, step.Tinf, step.k, Ta)));
        state.opened[name][h] = t;
        state.hrrAt[name][h] = Q;
        if (name === reading) opened.push({ head: h, time: t });
      }
    }
    state.prevT[h] = T[h];
    state.prevU[h] = u[h];
  }
  state.prevQ = Math.max(0, hrr);
  return opened;
}

/**
 * Suppression state for N cells: Evans's decay time (0: not covered), the
 * time each cell's head opened, and its reaction rate when it opened (-1:
 * armed, to be read in the next step), the spray density of the open heads
 * over each covered cell, and the spray's columns (top cell, open cells,
 * water kg/s), one entry per column however many heads cover it.
 */
export function createSuppression(N, { mode = "spray", eta = DROPLET_EVAPORATION, wet = mode === "wetting" ? WETTED_FRACTION : 0 } = {}) {
  return {
    mode,
    eta,
    wet,
    tau: new Float32Array(N),
    tAct: new Float32Array(N),
    rate: new Float32Array(N).fill(-1),
    density: new Float32Array(N),
    columns: [],
    columnAt: new Map(),
    open: [],
    stats: { emitted: 0, evaporated: 0, wetted: 0, floor: 0 },
  };
}

/** A head's water, kg/s (1 L a kg), and its density over its coverage, mm/s. */
export const headFlow = (head) => head.flowLpm / 60;
export const headDensity = (head) => head.flowLpm / 60 / head.coverage;

/**
 * Open heads: the covered fuel cells take Evans's cap (density added over
 * every open head, tau from it, armed when first covered) and, in "spray"
 * mode, each head's water is shared over its visible columns. Returns the
 * cells newly armed and those whose tau changed, for a backend to copy.
 */
export function openHeads(sup, setup, opened) {
  const touched = new Set(), armed = [];
  for (const { head: h, time } of opened) {
    const head = setup.heads[h];
    if (head.sensor < 0 || sup.open.includes(h)) continue;
    sup.open.push(h);
    if (sup.mode === "activation") continue;
    const w = headDensity(head);
    // Evans's cap in "evans" and "spray" modes; "wetting" leaves it to the water.
    for (const c of sup.mode === "wetting" ? [] : head.covered) {
      sup.density[c] += w;
      const tau = sup.density[c] >= EVANS_MIN_DENSITY ? evansTau(sup.density[c]) : 0;
      if (tau > 0 && sup.tau[c] === 0) {
        sup.tAct[c] = time;
        sup.rate[c] = -1;
        armed.push(c);
      }
      sup.tau[c] = tau;
      touched.add(c);
    }
    if ((sup.mode !== "spray" && sup.mode !== "wetting") || !head.columns.length) continue;
    const share = headFlow(head) / (head.columns.length / 2);
    for (let q = 0; q < head.columns.length; q += 2) {
      const top = head.columns[q];
      let col = sup.columnAt.get(top);
      if (!col) {
        col = { top, count: head.columns[q + 1], flux: 0 };
        sup.columnAt.set(top, col);
        sup.columns.push(col);
      }
      col.flux += share;
    }
  }
  return { armed, touched: [...touched] };
}

/**
 * Evans's cap on one fuel cell this step: the factor (at most 1) to scale
 * its reactions by, given their total `loss` (kg) uncapped. The step after
 * the head opens reads the cell's rate and leaves it alone.
 */
export function evansFactor(sup, c, loss, time, dt) {
  const tau = sup.tau[c];
  if (!(tau > 0)) return 1;
  // Armed: this step's rate is the cell's rate when its head opened, zero
  // for a cell not yet reacting, which then never ignites (Evans's premise
  // that a sprinklered fire does not grow).
  if (sup.rate[c] < 0) {
    sup.rate[c] = Math.max(0, loss) / dt;
    return 1;
  }
  if (!(loss > 0)) return 1;
  const allowed = sup.rate[c] * Math.exp(-(time - sup.tAct[c]) / tau) * dt;
  return loss > allowed ? allowed / loss : 1;
}

/** The heads, and when each opened by both readings, for the run's card. */
export function activationSummary(setup, state) {
  return setup.heads.map((head, h) => ({
    id: head.id,
    onGrid: head.sensor >= 0,
    inSight: head.inSight,
    r: head.r,
    estimate: Number.isNaN(state.opened.estimate[h]) ? null : state.opened.estimate[h],
    late: Number.isNaN(state.opened.late[h]) ? null : state.opened.late[h],
    hrrEstimate: Number.isNaN(state.hrrAt.estimate[h]) ? null : state.hrrAt.estimate[h],
    hrrLate: Number.isNaN(state.hrrAt.late[h]) ? null : state.hrrAt.late[h],
  }));
}

export { GAS, POROUS, SOLID };
