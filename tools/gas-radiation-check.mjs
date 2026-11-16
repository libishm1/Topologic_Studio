#!/usr/bin/env node
/**
 * The checks of physics item 8, second attempt: the gas's CO2 and H2O radiate
 * by the rank correlated spectral line weighted sum of gray gases
 * (wiki/research/fire-calibration-plan-2026-11-11.md, 16.43, "The checks",
 * with the coordinator's decisions 16.44; src/core/gasRadiation.js and
 * flamescapes.js radiationBands, marchBands, burnerBands), on the CPU solver.
 * Rewritten from the first attempt's tool (plan 13.19). No measured datum
 * enters any check.
 *
 *   node tools/gas-radiation-check.mjs --tables FILE [--source DIR]
 *        [--only a,b,c,d,e,h,l,m,burners,i,k2,speed] [--radcal values.json]
 *        [--b-json out.json] [--drafted values.json] [--planck values.json]
 *        [--base DIR] [--steps 200] [--seconds 120] [--json out.json]
 *        [--with-flame-rays] [--suite-base FILE] [--suite-test REGEX]
 *        [--suite-cells 0.34] [--suite-out DIR]
 *
 * --tables (or the environment variable FLAMESCAPES_GAS_TABLES) is the
 * tables' file that tools/gas-radiation-tables.mjs builds; by default the
 * repository's own, public/gas-radiation/gas-radiation-tables.f32 (plan
 * 16.44, the user's decision on decision 7). --source (or FLAMESCAPES_RADLIB_DATA)
 * is the folder of RadLib's two text files, outside this repository, which
 * (a) needs.
 *
 * (a) Tables and functions.
 *   1. The two source files have the SHA-256 the plan lists, and the tables'
 *      file equals them value for value after rounding to float32, CO2's
 *      [T_g][T_b][C] then H2O's [Y][T_g][T_b][C].
 *   2. The solver's F (albdf), its inversion (albdfInverse), kappa_j
 *      (gasCellBands times the molar density) and a_j (gasWeights) against
 *      this tool's own transcription, written apart from the solver's code
 *      (class Transcription below: the text files' values rounded to
 *      float32, searched and interpolated axis by axis), within 1e-5
 *      relative or 1e-12 absolute, at T in {293.15, 300, 345, 400, 600, 900,
 *      1200, 1800, 2400} K, T_b in {T, 293.15, 1173.15} K and the plan's ten
 *      compositions. F is compared at 15 cross-sections from 1e-7 to 1e3
 *      m2/mol and at 0.
 *   3. At each of those states, on the solver's values: the nine a_j sum to
 *      1 within 1e-12; every a_j is at least -1e-12; kappa_j is not negative
 *      and does not fall as j rises; with x = (0, 0) every kappa_j is 0 and
 *      a_0 is 1.
 *   4. The solver's 17 levels equal the plan's (point 12) within 5e-7.
 *   5. A lookup at each of 1000 table points drawn with a fixed seed returns
 *      the stored value exactly.
 * (b) Emissivity and absorptivity against RadCal: 13.19 (b)'s 72 cases. The
 *     solver's side is sum over j of a_j(T, T_b) (1 - exp(-kappa_j L)) from
 *     the solver's own functions, T_b = T for the 48 emissivities and
 *     1173.15 K for the 24 absorptivities. The 72 values are written to
 *     --b-json before --radcal is opened. Judged only with --radcal: every
 *     case within 25 % of RadCal's value or 0.01, whichever is larger; a
 *     case with no RadCal value fails. Reported: the sum over eight gray
 *     gases beside the tables' own limit (4000 cross-sections, the tool's
 *     transcription); with --drafted, the draft's values (its eight gray
 *     gases and its line-by-line values at 400 and 600 K) beside the
 *     solver's; the thin limit, sum of a_j kappa_j, with --planck beside a
 *     line-by-line Planck mean.
 * (c) Gray transport in a cube: 13.19 (c), soot only, within 2 %.
 * (d) Band transport in a cube: 13.19 (d)'s scenes; the reference is sum
 *     over j of 4 kappa_j sigma B_j V P(kappa_j L) with this method's bands.
 *     Within 2 %.
 * (e) A soot-free layer heats a surface it does not touch: 13.19 (e)'s box,
 *     three cell sizes, lumped and one-dimensional walls; the reference is
 *     sum over j of sigma B_j times the integral over the surface and its
 *     hemisphere of (1 - exp(-kappa_j l)) cos / pi. Within 3 %.
 * (f) Energy: in (c), (d), (e), (h), (l) and the two burner runs, in every
 *     step, what was emitted equals what the gas, the solids and the walls'
 *     ends took plus what left the grid, within 1e-9 of the emission.
 * (g) The radiant fraction counted once, in both burner runs, in every step:
 *     a cell the burner burnt in emits in no band and takes nothing of the
 *     burner's rays, and the burner's rays carry its radiant fraction of the
 *     heat burnt within 1e-12 relative.
 * (h) An isothermal box stays isothermal: (d)'s 3.0 m cube, soot 1e-3.
 *   1. Gas and black walls at 600 K: the gas's net radiative gain within 2 %
 *      of its emission, on 0.2 and 0.3 m cells.
 *   2. The same at 400 K with surfaceEmitFloor 1.
 *   3. Reported: 2 with surfaceEmitFloor 0.
 * (i) Off is off (the parts this tool runs): the two fingerprints
 *     (tools/flamescapes-fingerprint.mjs, run here) are d62b581d and
 *     29793d1a; the tables' reader is never called over runs with the switch
 *     off; and with --base DIR (a src/core folder of the base), runs with
 *     the switch off give the base's state bit for bit. The unit suite is
 *     run apart. The GPU check (--only ig, with --gpu-base a.json,b.json,...
 *     and --gpu-branch c.json, each a tools/flamescapes-gpu-check.mjs --json
 *     report: two or more of the base, one of this branch with the switch
 *     off): every leaf of the branch's report equals every base run's bit
 *     for bit, but the timings (set apart) and the leaves of a GPU run that
 *     differs between two runs of the base (plan 13.32's rule, by run as
 *     16.37 records it), which are reported beside the base's range.
 * (k) The ledger.
 *   1. In both burner runs, over every call of the radiation stage and every
 *      cast of the burner's rays, the stored fields before and after: the
 *      gas's heat at each cell's own c_p, the solids' at the capacity the
 *      march divides by, the one-dimensional walls' nodes, and what left the
 *      grid. Their sum changes by no more than 1e-9 of the emission beyond
 *      the float32 round-off the stage counts (stats.radRoundoff); over a
 *      cast of the burner's rays it rises by what was cast.
 *   2. (--only k2, with --suite-base FILE, a copy of the base's
 *      validation-suite.mjs placed beside this branch's.) On one test of the
 *      suite (--suite-test, default Steckler's test 14, on --suite-cells,
 *      default 0.34 m, its full horizon): the suite's ledger with
 *      --gas-radiation carries radiatedOut and the item's tallies, and
 *      without it equals the base's field for field.
 *   3. Reported: the first design (13.19 (k)), both burner runs on and off
 *      at the ledger's times.
 * (l) A black ray is split by the gas it meets: (e)'s box on 0.2 and 0.27 m
 *     cells, the layer at ambient temperature, the floor black at 900 K and
 *     at 1173.15 K. What the layer's gas takes against sum over j of sigma
 *     [a_j(T_a, T_s) T_s^4 - a_j(T_a, T_a) T_a^4] times (e)'s integral over
 *     the floor. Within 3 %.
 * (m) The number of rays changes no mean: (d)'s 3.0 m cube at 600 K with
 *     gasRays 2, 8 and 32; the walls' power agrees between each pair within
 *     4 standard deviations, from the spread of the steps' own powers.
 *     Outside what ends the line.
 * Speed (reported): the radiation stage alone in (d) and (e), ms a step, on
 * and off.
 *
 * The radiation-stage checks build each scene, set its temperatures and
 * gases, and call the stage alone --steps times (the plan's 200) of 0.05 s,
 * seed 1, the step's number advanced each time, every temperature set back
 * after each. surfaceRayStart is 1 throughout. The burner runs are 13.19
 * (f)'s: a 100 kW burner 0.3 m square at the centre of (e)'s box's floor on
 * 0.2 m cells, a door 0.8 x 2.0 m to 1.0 m of open cells, the solver's stone
 * walls, 120 s with the flow on; methane and the class A fuel at a radiant
 * fraction of 0.2, flameRadiantOnce 1 (burnerFlameRays 0). With
 * --with-flame-rays each is run again with burnerFlameRays 1, reported and
 * not judged: (f), (g) and (k) 1 with item 12's rays through marchBands.
 */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const CORE = path.resolve(HERE, "../topologicpy-web-frontend/src/core");
const imp = (dir, f) => import(pathToFileURL(path.join(dir, f)).href);
const F = await imp(CORE, "flamescapes.js");
const G = await imp(CORE, "gasRadiation.js");
const V = await imp(CORE, "voxelize.js");
const R = await imp(CORE, "flamescapesRun.js");
const FC = await imp(CORE, "fireClasses.js");

const ONLY = new Set((flag("--only", "a,b,c,d,e,h,l,m,burners,i,speed") || "").split(","));
const STEPS = Number(flag("--steps", 200));
const SECONDS = Number(flag("--seconds", 120));
const DT = 0.05;
const TA = F.DEFAULTS.ambient;
const SIGMA = F.SIGMA;
const NS = F.NS, S = F.SPECIES;
const MOLAR = F.MOLAR;
const MOLAR_N2 = 0.028;
const NB = G.GAS_BANDS;
const RADTMP = 1173.15;
const TABLES_FILE = flag("--tables", process.env.FLAMESCAPES_GAS_TABLES || path.resolve(HERE, "../topologicpy-web-frontend/public/gas-radiation/gas-radiation-tables.f32"));
const SOURCE_DIR = flag("--source", process.env.FLAMESCAPES_RADLIB_DATA || null);
const SOURCES = [
  { file: "co2_p1_0.txt", values: 28 * 28 * 71, sha256: "5bd9c9ecdd3449f5c145a9e6bf07a181486752cec22920ef1928e9422a17508b" },
  { file: "h2o_p1_0.txt", values: 9 * 28 * 28 * 71, sha256: "b3dd6c860866d36c443719c3cadb2c1c1e2b83c3198040ee9d3b0847a2972e53" },
];

const verdicts = {};
const counts = {};
const report = { tool: "tools/gas-radiation-check.mjs", plan: "wiki/research/fire-calibration-plan-2026-11-11.md 16.43, 16.44", steps: STEPS, checks: {} };
const pct = (x) => `${(100 * x).toFixed(3)} %`;
const sum = (a) => {
  let s = 0;
  for (let q = 0; q < a.length; q += 1) s += a[q];
  return s;
};
/** A judged line. */
function verdict(check, ok, line) {
  verdicts[check] = (verdicts[check] ?? true) && ok;
  const n = (counts[check] ||= { judged: 0, failed: 0 });
  n.judged += 1;
  if (!ok) n.failed += 1;
  console.log(`  ${ok ? "pass" : "FAIL"}  ${line}`);
}
const reported = (line) => console.log(`        reported: ${line}`);

// The tables are read once, here, and only for the checks that set gasRadiation 1.
const NEED_TABLES = ["a", "b", "c", "d", "e", "h", "l", "m", "burners", "speed"].some((k) => ONLY.has(k));
let MODEL = null;
if (NEED_TABLES) {
  if (!TABLES_FILE || !existsSync(TABLES_FILE)) {
    console.error("The tables' file is needed: --tables FILE or FLAMESCAPES_GAS_TABLES (tools/gas-radiation-tables.mjs builds it; plan 16.44, decision 7).");
    process.exit(2);
  }
  MODEL = G.gasTablesFrom(readFileSync(TABLES_FILE));
}
const ON = () => ({ gasRadiation: 1, surfaceRayStart: 1, gasTables: MODEL });

// ------------------------------------------------------------- (a) the tool's own transcription

/**
 * The method's functions (plan 16.43, points 11 and 12), written for this
 * tool from the plan's text and apart from the solver's code: the two text
 * files' values, each rounded to float32 as the tables' file holds them, in
 * nested arrays; an axis searched by bisection; interpolation one axis at a
 * time, recursively.
 */
class Transcription {
  constructor(dir) {
    this.Tgrid = Array.from({ length: 28 }, (_, i) => 300 + 100 * i);
    this.Cgrid = Array.from({ length: 71 }, (_, k) => 1e-4 * Math.pow(10, k / 10));
    this.Ygrid = [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1];
    this.flat = SOURCES.map((s) => {
      const bytes = readFileSync(path.join(dir, s.file));
      const sha = createHash("sha256").update(bytes).digest("hex");
      const values = bytes.toString("latin1").split(/\s+/).filter((t) => t !== "").map((t) => Math.fround(Number(t)));
      return { file: s.file, sha256: s.sha256, values_expected: s.values, sha, values };
    });
    const nest = (flat, dims) => {
      if (dims.length === 1) return flat;
      const size = flat.length / dims[0];
      return Array.from({ length: dims[0] }, (_, i) => nest(flat.slice(i * size, (i + 1) * size), dims.slice(1)));
    };
    this.co2 = nest(this.flat[0].values, [28, 28, 71]);
    this.h2o = nest(this.flat[1].values, [9, 28, 28, 71]);
    // The levels (point 12).
    const lnA = Math.log(3e-4), lnB = Math.log(60);
    const edges = Array.from({ length: 9 }, (_, k) => Math.exp(lnA + ((lnB - lnA) * k) / 8));
    this.Ftilde = edges.map((C, k) => (k === 8 ? 1 : this.F(C, 600, 600, 0.095, 0.19)));
    this.Fband = Array.from({ length: 8 }, (_, j) => this.F(Math.sqrt(edges[j] * edges[j + 1]), 600, 600, 0.095, 0.19));
  }
  /** Linear interpolation of a nested table over its axes at the coordinates, each held at its axis's ends. */
  static interp(table, axes, coords) {
    if (!axes.length) return table;
    const axis = axes[0];
    const x = Math.min(Math.max(coords[0], axis[0]), axis[axis.length - 1]);
    let lo = 0, hi = axis.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (axis[mid] <= x) lo = mid;
      else hi = mid;
    }
    const t = (x - axis[lo]) / (axis[hi] - axis[lo]);
    const a = Transcription.interp(table[lo], axes.slice(1), coords.slice(1));
    if (t === 0) return a;
    const b = Transcription.interp(table[hi], axes.slice(1), coords.slice(1));
    return a + t * (b - a);
  }
  F(C, Tg, Tb, xc, xw) {
    let f = 1;
    if (xc > 0) f *= Transcription.interp(this.co2, [this.Tgrid, this.Tgrid, this.Cgrid], [Tg, Tb, C / xc]);
    if (xw > 0) f *= Transcription.interp(this.h2o, [this.Ygrid, this.Tgrid, this.Tgrid, this.Cgrid], [xw, Tg, Tb, C / xw]);
    return f;
  }
  /** The cross-section at which F(C; Tg, Tb, x) is `level`: bisection in ln C from 1e-7 to 1e3, to 1e-6; 0 where F at 1e-7 is at or above the level. */
  inverse(level, Tg, Tb, xc, xw) {
    if (this.F(1e-7, Tg, Tb, xc, xw) >= level) return 0;
    let lo = Math.log(1e-7), hi = Math.log(1e3);
    while (hi - lo > 1e-6) {
      const mid = (lo + hi) / 2;
      if (this.F(Math.exp(mid), Tg, Tb, xc, xw) < level) lo = mid;
      else hi = mid;
    }
    return Math.exp((lo + hi) / 2);
  }
  /** A state's gray gases: kappa_j (1/m) for j = 1 to 8, the levels' cross-sections, and the nine weights for black radiation at Tb. */
  state(T, Tb, xc, xw) {
    const molar = 101325 / (8.314462618 * T);
    const C = this.Fband.map((f) => this.inverse(f, T, 600, xc, xw));
    const Ct = this.Ftilde.slice(0, 8).map((f) => this.inverse(f, T, 600, xc, xw));
    const Fb = Ct.map((c) => this.F(c, T, Tb, xc, xw));
    const a = [Fb[0], ...Fb.slice(1).map((f, k) => f - Fb[k]), 1 - Fb[7]];
    return { kappa: C.map((c) => molar * c), C, Ct, a };
  }
  /** The tables' own limit for a uniform column: 4000 cross-sections evenly spaced in ln C. */
  column(T, Tb, xc, xw, L, n = 4000) {
    const molar = 101325 / (8.314462618 * T);
    const lo = Math.log(1e-4 * Math.min(...[xc, xw].filter((x) => x > 0))), hi = Math.log(1e3);
    let e = 0, prevF = 0, prevC = 0;
    for (let q = 0; q < n; q += 1) {
      const C = Math.exp(lo + ((hi - lo) * q) / (n - 1));
      const f = this.F(C, T, Tb, xc, xw);
      if (q) e += (f - prevF) * -Math.expm1(-molar * Math.sqrt(C * prevC) * L);
      prevF = f;
      prevC = C;
    }
    return e + (1 - prevF) * -Math.expm1(-molar * prevC * L);
  }
}

const PLAN_LEVELS = {
  Ftilde: [0.174284, 0.331192, 0.485571, 0.662434, 0.827895, 0.92998, 0.981241, 0.996933, 1],
  Fband: [0.249345, 0.410365, 0.567447, 0.751544, 0.888712, 0.959218, 0.992104, 0.998888],
};
const within = (x, ref) => Math.abs(x - ref) / Math.max(1e-5 * Math.abs(ref), 1e-12);

/** The solver's gray gases at a state, from its own functions. */
function solverState(T, Tb, xc, xw) {
  const C = new Float64Array(NB), Ct = new Float64Array(NB), a = new Float64Array(NB + 1);
  G.gasCellBands(MODEL, T, xc, xw, C, 0, Ct, 0);
  G.gasWeights(MODEL, Ct, 0, T, Tb, xc, xw, a);
  const molar = F.P_AMBIENT / (F.R_GAS * T);
  return { kappa: Array.from(C, (c) => molar * c), C: Array.from(C), Ct: Array.from(Ct), a: Array.from(a) };
}

function checkA() {
  console.log("\n(a) tables and functions");
  if (!SOURCE_DIR) {
    console.log("  not run: --source DIR (RadLib's two text files) not given");
    report.checks.a = { run: false };
    return;
  }
  const X = new Transcription(SOURCE_DIR);
  // 1. The sources' SHA-256, and the file against them.
  const bytes = readFileSync(TABLES_FILE);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  verdict("a", bytes.length === 4 * (SOURCES[0].values + SOURCES[1].values), `1: the tables' file has ${bytes.length} bytes (${4 * (SOURCES[0].values + SOURCES[1].values)}: float32, CO2 then H2O)`);
  let offset = 0;
  for (const s of X.flat) {
    verdict("a", s.sha === s.sha256, `1: ${s.file} SHA-256 ${s.sha.slice(0, 8)}...${s.sha.slice(-8)} (the plan's ${s.sha256.slice(0, 8)}...${s.sha256.slice(-8)})`);
    let unequal = 0;
    for (let q = 0; q < s.values.length && offset + 4 * q + 4 <= bytes.length; q += 1) if (!Object.is(view.getFloat32(offset + 4 * q, true), s.values[q])) unequal += 1;
    verdict("a", s.values.length === s.values_expected && unequal === 0, `1: ${s.file}: ${s.values.length} values; ${unequal} differ from the file's after rounding to float32, in the stated order`);
    offset += 4 * s.values.length;
  }
  // 2 and 3. The functions at the plan's states.
  const temps = [293.15, 300, 345, 400, 600, 900, 1200, 1800, 2400];
  const mixes = [[0, 0], [0.0004, 0.012], [0.0133, 0.0167], [0.0103, 0.0197], [0.0667, 0.0833], [0.0517, 0.0983], [0.05, 0.1], [0.12, 0], [0, 0.2], [0.095, 0.19]];
  const crossSections = [0, ...Array.from({ length: 15 }, (_, q) => 1e-7 * Math.pow(1e10, q / 14))];
  // The sixteen levels the method inverts (the ninth, set to 1, is never inverted).
  const levels = [...X.Fband, ...X.Ftilde.slice(0, 8)];
  let worst2 = { v: 0, at: "" }, states = 0, worst3 = { sum: 0, least: Infinity };
  const rows = [];
  for (const T of temps) {
    for (const [xc, xw] of mixes) {
      for (const Tb of [T, 293.15, 1173.15]) {
        states += 1;
        const ref = X.state(T, Tb, xc, xw), sol = solverState(T, Tb, xc, xw);
        let wF = 0, wI = 0, wK = 0, wA = 0;
        for (const C of crossSections) wF = Math.max(wF, within(G.albdf(MODEL, C, T, Tb, xc, xw), X.F(C, T, Tb, xc, xw)));
        for (const level of levels) wI = Math.max(wI, within(G.albdfInverse(MODEL, level, T, Tb, xc, xw), X.inverse(level, T, Tb, xc, xw)));
        for (let j = 0; j < NB; j += 1) {
          wK = Math.max(wK, within(sol.kappa[j], ref.kappa[j]));
          wI = Math.max(wI, within(sol.Ct[j], ref.Ct[j]));
        }
        for (let j = 0; j <= NB; j += 1) wA = Math.max(wA, within(sol.a[j], ref.a[j]));
        const w = Math.max(wF, wI, wK, wA);
        if (w > worst2.v) worst2 = { v: w, at: `T ${T}, T_b ${Tb}, x (${xc}, ${xw})` };
        verdict("a", w <= 1, `2: T ${T} K, T_b ${Tb} K, x (${xc}, ${xw}): of the tolerance (1e-5 relative or 1e-12), F ${wF.toExponential(1)}, inversion ${wI.toExponential(1)}, kappa_j ${wK.toExponential(1)}, a_j ${wA.toExponential(1)}`);
        // 3, on the solver's values.
        const total = sum(sol.a), least = Math.min(...sol.a);
        let ordered = true;
        for (let j = 0; j < NB; j += 1) if (!(sol.kappa[j] >= 0) || (j && sol.kappa[j] < sol.kappa[j - 1])) ordered = false;
        const empty = xc + xw === 0 ? sol.kappa.every((k) => k === 0) && sol.a[0] === 1 : true;
        worst3 = { sum: Math.max(worst3.sum, Math.abs(total - 1)), least: Math.min(worst3.least, least) };
        verdict("a", Math.abs(total - 1) <= 1e-12 && least >= -1e-12 && ordered && empty, `3: the same state: sum of a_j - 1 = ${(total - 1).toExponential(1)}, least a_j ${least.toExponential(2)}, kappa_j ${ordered ? "not negative, not falling" : "OUT OF ORDER"}${xc + xw === 0 ? `, no gas: ${empty ? "every kappa_j 0 and a_0 1" : "NOT clear"}` : ""}`);
        if (Tb === T) rows.push({ T, xc, xw, kappa: sol.kappa, a: sol.a });
      }
    }
  }
  console.log(`  ${states} states; the worst of 2 is ${worst2.v.toExponential(2)} of its tolerance (${worst2.at}); of 3, |sum - 1| at most ${worst3.sum.toExponential(1)}, least a_j ${worst3.least.toExponential(2)}`);
  // 4. The levels.
  for (let k = 0; k < 9; k += 1) verdict("a", Math.abs(MODEL.levels[k] - PLAN_LEVELS.Ftilde[k]) <= 5e-7, `4: level Ft_${k} ${MODEL.levels[k].toFixed(7)} (the plan's ${PLAN_LEVELS.Ftilde[k]}; the tool's transcription ${X.Ftilde[k].toFixed(7)}), within 5e-7`);
  for (let j = 0; j < 8; j += 1) verdict("a", Math.abs(MODEL.bandLevels[j] - PLAN_LEVELS.Fband[j]) <= 5e-7, `4: level F_${j + 1} ${MODEL.bandLevels[j].toFixed(7)} (the plan's ${PLAN_LEVELS.Fband[j]}; the tool's transcription ${X.Fband[j].toFixed(7)}), within 5e-7`);
  // 5. A thousand table points.
  let state = 20261116 >>> 0, exact = 0, co2Points = 0;
  const pick = (n) => {
    state = F.pcgHash(state);
    return state % n;
  };
  for (let q = 0; q < 1000; q += 1) {
    if (pick(10) === 0) {
      const i = pick(28), j = pick(28), k = pick(71);
      co2Points += 1;
      const got = G.albdfCO2(MODEL, X.Cgrid[k], X.Tgrid[i], X.Tgrid[j]);
      if (Object.is(got, X.co2[i][j][k]) && Object.is(got, MODEL.co2[(i * 28 + j) * 71 + k])) exact += 1;
    } else {
      const y = pick(9), i = pick(28), j = pick(28), k = pick(71);
      const got = G.albdfH2O(MODEL, X.Cgrid[k], X.Tgrid[i], X.Tgrid[j], X.Ygrid[y]);
      if (Object.is(got, X.h2o[y][i][j][k]) && Object.is(got, MODEL.h2o[((y * 28 + i) * 28 + j) * 71 + k])) exact += 1;
    }
  }
  verdict("a", exact === 1000, `5: ${exact} of 1000 lookups at table points (${co2Points} of CO2, ${1000 - co2Points} of H2O; seed 20261116) return the stored value exactly`);
  report.checks.a = { run: true, states, worst2, worst3, levels: { Ftilde: Array.from(MODEL.levels), Fband: Array.from(MODEL.bandLevels) }, exact, sample: rows.filter((r) => r.xc === 0.05 && [600, 1200].includes(r.T)), pass: verdicts.a };
}

// ------------------------------------------------------------- scenes

const BLACK = { name: "black", rho: 2700, c: 800, k: 2.5, eps: 1 };
const BOARD = { thickness: 0.016, rho: 1440, c: 840, k: 0.48 };
const gypsum = (h) => ({ name: "gypsum", rho: (BOARD.rho * BOARD.thickness) / h, c: BOARD.c, k: BOARD.k, eps: 1, layers: [BOARD] });
const INERT_FUEL = { heatOfCombustion: 17.5e6, yields: { co2: 1.27, co: 0.004, soot: 0.015 }, embers: false, ignitionTemperature: 300 };

/**
 * A box of inner[0] x inner[1] x inner[2] open cells (y up) in walls one cell
 * thick at the grid's edge, of `material` (null: the solver's stone); with
 * `beyond`, that many open cells past its x+ wall, and a door through that
 * wall of `door` = [width, height] cells from the floor, centred in z.
 */
function boxScene({ inner, h, material = null, beyond = 0, door = null }) {
  const [ni, nj, nk] = inner;
  const nx = ni + 2 + beyond, ny = nj + 2, nz = nk + 2;
  const N = nx * ny * nz;
  const grid = { nx, ny, nz, h, origin: [0, 0, 0], axes: [0, 1, 2], upAxis: "y", cells: N };
  const fraction = new Float32Array(N * V.MATERIALS);
  const solid = new Uint8Array(N);
  const k0 = door ? 1 + Math.floor((nk - door[0]) / 2) : 0;
  for (let c = 0; c < N; c += 1) {
    const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
    if (i > ni + 1) continue;
    if (!(i === 0 || j === 0 || k === 0 || i === ni + 1 || j === ny - 1 || k === nz - 1)) continue;
    if (door && i === ni + 1 && j >= 1 && j <= door[1] && k >= k0 && k < k0 + door[0]) continue;
    solid[c] = 1;
    fraction[c * V.MATERIALS] = 1;
  }
  return { grid, fraction, solid, material, contents: { cells: [], masses: [] }, ignition: null, pilot: null };
}

function solver(scene, options, Fm = F) {
  const inert = scene.material ? { inert: [scene.material, scene.material, scene.material] } : {};
  return Fm.createSolver(scene, { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL, seed: 1, ...inert, ...options });
}

const openCells = (sim) => {
  const out = [];
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] !== F.SOLID) out.push(c);
  return out;
};

/** Mass fractions for mole fractions of CO2 and H2O, soot's mass fraction ys, and air (0.232 O2 by mass) the rest. */
function mixture({ xc = 0, xw = 0, ys = 0 }) {
  const airO2 = 0.232 / 0.032 / (0.232 / 0.032 + 0.768 / MOLAR_N2);
  const rest = 1 - xc - xw;
  const xo = rest * airO2, xn = rest - xo;
  const m = xc * MOLAR[S.CO2] + xw * MOLAR[S.H2O] + xo * MOLAR[S.O2] + xn * MOLAR_N2;
  const Y = new Float64Array(NS);
  Y[S.CO2] = ((xc * MOLAR[S.CO2]) / m) * (1 - ys);
  Y[S.H2O] = ((xw * MOLAR[S.H2O]) / m) * (1 - ys);
  Y[S.O2] = ((xo * MOLAR[S.O2]) / m) * (1 - ys);
  Y[S.SOOT] = ys;
  return Y;
}

function setGas(sim, c, T, Y) {
  for (let s = 0; s < NS; s += 1) sim.Y[c * NS + s] = Y[s];
  sim.T[c] = T;
  sim.rho[c] = F.gasDensity(T, sim.Y, c * NS);
}

/** A cell's mole fractions of CO2 and H2O as the solver takes them (gasDensity's sums). */
function fractionsOf(sim, c) {
  let inv = 0, rest = 1;
  for (let s = 0; s < NS; s += 1) {
    const y = sim.Y[c * NS + s];
    inv += y / MOLAR[s];
    rest -= y;
  }
  inv += Math.max(0, rest) / MOLAR_N2;
  return { xc: sim.Y[c * NS + S.CO2] / MOLAR[S.CO2] / inv, xw: sim.Y[c * NS + S.H2O] / MOLAR[S.H2O] / inv };
}

/** The solid cell's group in a box: floor, ceiling, x-, x+, z-, z+ (floor and ceiling first). */
function groupOf(sim, c) {
  const { nx, ny } = sim.grid;
  const i = c % nx, j = Math.floor(c / nx) % ny, k = Math.floor(c / (nx * ny));
  if (j === 0) return "floor";
  if (j === ny - 1) return "ceiling";
  if (i === 0) return "x-";
  if (k === 0) return "z-";
  if (k === sim.grid.nz - 1) return "z+";
  return "x+";
}

/** One call's tallies from the solver's work arrays of `name` ("band": the radiation stage; "burnerBand": the burner's rays), J. */
function readCall(sim, name, groups = false) {
  const w = sim.work;
  const gasArr = w[`${name}Gas`], solidArr = w[`${name}Solid`], lo = w[`${name}WallLo`], hi = w[`${name}WallHi`];
  let solid = 0;
  const byGroup = {};
  for (let c = 0; c < sim.N; c += 1) {
    if (!solidArr[c]) continue;
    solid += solidArr[c];
    if (groups) {
      const g = groupOf(sim, c);
      byGroup[g] = (byGroup[g] || 0) + solidArr[c];
    }
  }
  if (sim.wall && lo) {
    for (let q = 0; q < sim.wall.cells.length; q += 1) {
      const e = lo[q] + hi[q];
      if (!e) continue;
      solid += e;
      if (groups) {
        const g = groupOf(sim, sim.wall.cells[q]);
        byGroup[g] = (byGroup[g] || 0) + e;
      }
    }
  }
  const gas = sum(gasArr), out = w[`${name}Out`][0];
  const gasEmit = name === "band" ? w.bandEmit[0] : 0, surfEmit = name === "band" ? w.bandEmit[1] : 0;
  const emitted = name === "band" ? gasEmit + surfEmit : w.burnerBandEmit[0];
  return { gasEmit, surfEmit, emitted, gas, solid, out, byGroup, imbalance: emitted - (gas + solid + out) };
}

const TALLIES = ["radFloor", "radCapped", "radNegative", "radShareNegative", "surfaceFloor", "surfaceCapped", "gasRays", "gasCrossings", "gasEmitters", "surfaceRays", "surfaceCrossings"];

/**
 * The radiation stage alone, `steps` times from the same temperatures (each
 * set back after), the step's number advanced each time: the mean tallies a
 * step (J), each step's solids' take, the worst energy imbalance relative to
 * the step's emission, the item's tallies over the steps, and the ms a step.
 * `gasOf` (a set of cells, or null) limits `gasIn`, the gas's take in them.
 */
function runStage(sim, steps = STEPS, gasOf = null) {
  const T0 = sim.T.slice(), Ts0 = sim.Ts.slice();
  const W0 = sim.wall ? sim.wall.T.slice() : null, L0 = sim.wall ? sim.wall.last.slice() : null;
  const on = sim.P.gasRadiation > 0;
  const acc = { gasEmit: 0, surfEmit: 0, gas: 0, gasIn: 0, solid: 0, out: 0, byGroup: {} };
  const solids = new Float64Array(steps);
  let worst = 0, ms = 0;
  const st0 = { ...sim.stats };
  for (let s = 0; s < steps; s += 1) {
    const t0 = performance.now();
    F.radiationStage(sim, DT);
    ms += performance.now() - t0;
    if (on) {
      const r = readCall(sim, "band", true);
      for (const k of ["gasEmit", "surfEmit", "gas", "solid", "out"]) acc[k] += r[k];
      for (const [g, e] of Object.entries(r.byGroup)) acc.byGroup[g] = (acc.byGroup[g] || 0) + e;
      if (gasOf) for (const c of gasOf) acc.gasIn += sim.work.bandGas[c];
      solids[s] = r.solid;
      if (r.emitted > 0) worst = Math.max(worst, Math.abs(r.imbalance) / r.emitted);
    } else {
      // The item off: the solids' take from their temperatures' rise (lumped walls).
      for (let c = 0; c < sim.N; c += 1) {
        if (sim.kind[c] === F.GAS || sim.Ts[c] === Ts0[c]) continue;
        const e = (sim.Ts[c] - Ts0[c]) * Math.max(F.solidCapacity(sim, c), 1e-6);
        acc.solid += e;
        const g = groupOf(sim, c);
        acc.byGroup[g] = (acc.byGroup[g] || 0) + e;
      }
    }
    sim.T.set(T0);
    sim.Ts.set(Ts0);
    if (W0) {
      sim.wall.T.set(W0);
      sim.wall.last.set(L0);
    }
    sim.steps += 1;
    sim.time += DT;
  }
  const mean = { gasEmit: acc.gasEmit / steps, surfEmit: acc.surfEmit / steps, gas: acc.gas / steps, gasIn: acc.gasIn / steps, solid: acc.solid / steps, out: acc.out / steps, byGroup: Object.fromEntries(Object.entries(acc.byGroup).map(([g, e]) => [g, e / steps])) };
  const tallies = {};
  for (const k of TALLIES) tallies[k] = (sim.stats[k] ?? 0) - (st0[k] ?? 0);
  return { mean, worst, tallies, solids, ms: ms / steps };
}

const talliesLine = (t) =>
  `radFloor ${(t.radFloor / STEPS).toExponential(2)} J, radCapped ${(t.radCapped / STEPS).toExponential(2)} J, radNegative ${(t.radNegative / STEPS).toExponential(2)} J, radShareNegative ${(t.radShareNegative / STEPS).toExponential(2)} J, surfaceFloor ${(t.surfaceFloor / STEPS).toExponential(2)} J a step; ${t.gasRays + t.surfaceRays ? ((t.gasCrossings + t.surfaceCrossings) / (t.gasRays + t.surfaceRays)).toFixed(2) : "-"} crossings a ray; ${(t.gasEmitters / STEPS).toFixed(0)} emitting gas cells`;

// ------------------------------------------------------------- quadrature

/**
 * For rays leaving a rectangle's points (uniform over it) in one direction,
 * the mean of 1 - exp(-c d), d the part of the height between lo and hi that
 * the ray climbs inside the box before it leaves by a side: the height Y at
 * which a ray leaves by a side has P(Y > y) = (1 - A y)+ (1 - B y)+ (A, B the
 * direction's horizontal run per unit height over the box's widths), so the
 * mean over the rectangle is a one-dimensional integral, worked in closed
 * form: of (1 - exp(-c s)) times the density (A + B) - 2 A B y from lo to
 * min(hi, 1 / A, 1 / B), plus 1 - exp(-c (hi - lo)) for the rays that reach hi.
 */
function slabMean(c, A, B, lo, hi) {
  const yStar = Math.min(A > 0 ? 1 / A : Infinity, B > 0 ? 1 / B : Infinity);
  const top = Math.min(hi, yStar);
  let mean = 0;
  const Sl = top - lo;
  if (Sl > 0) {
    const alpha = A + B - 2 * A * B * lo, beta = 2 * A * B;
    const e = Math.exp(-c * Sl);
    // The integral from 0 to S of (1 - exp(-c s)) (alpha - beta s) ds.
    mean += alpha * (Sl + Math.expm1(-c * Sl) / c) - beta * ((Sl * Sl) / 2 - (-Math.expm1(-c * Sl) - c * Sl * e) / (c * c));
  }
  if (hi > lo) mean += -Math.expm1(-c * (hi - lo)) * Math.max(0, 1 - A * hi) * Math.max(0, 1 - B * hi);
  return mean;
}

/**
 * The cosine-weighted mean over a hemisphere of f(dy, a, b) (dy the normal's
 * component, a and b the horizontal components' magnitudes over dy), a
 * midpoint rule in u = dy^2 and the azimuth over a quarter (the boxes being
 * symmetric), doubled until doubling moves no value by 0.05 % (the plan's
 * criterion). Returns the values, the last change and the grid.
 */
function hemisphere(f, count, start = 64) {
  const at = (n) => {
    const out = new Float64Array(count);
    for (let p = 0; p < n; p += 1) {
      const mu = Math.sqrt((p + 0.5) / n), st = Math.sqrt(1 - mu * mu);
      for (let q = 0; q < n; q += 1) {
        const phi = (Math.PI / 2) * ((q + 0.5) / n);
        f(mu, (st * Math.cos(phi)) / mu, (st * Math.sin(phi)) / mu, out);
      }
    }
    for (let t = 0; t < count; t += 1) out[t] /= n * n;
    return out;
  };
  let n = start, prev = at(n);
  for (let r = 0; r < 8; r += 1) {
    n *= 2;
    const next = at(n);
    let change = 0;
    for (let t = 0; t < count; t += 1) if (next[t] !== 0) change = Math.max(change, Math.abs(next[t] - prev[t]) / Math.abs(next[t]));
    if (change < 5e-4) return { values: next, change, grid: n };
    prev = next;
  }
  throw new Error("the quadrature did not converge");
}

/** The exact escape fractions of a uniform unit cube at optical sides `taus`: (6 / 4 tau) <1 - exp(-tau l)> over a face (reciprocity). A tau of 0 gives 1. */
function cubeEscape(taus) {
  const r = hemisphere((dy, a, b, out) => {
    for (let t = 0; t < taus.length; t += 1) if (taus[t] > 0) out[t] += slabMean(taus[t] / dy, a, b, 0, 1);
  }, taus.length);
  return { P: Array.from(r.values, (v, t) => (taus[t] > 0 ? (6 / (4 * taus[t])) * v : 1)), change: r.change, grid: `${r.grid} x ${r.grid} directions` };
}

/**
 * Over a box Lx x H x Lz's floor (from below) or ceiling (from above), the
 * integral over the surface of the cosine-weighted mean of (1 - exp(-kappa
 * l)) for each kappa, l the chord through the layer from y0 to H: m^2.
 */
function layerIntegrals(kappas, Lx, H, Lz, y0, ceiling) {
  const r = hemisphere((dy, a, b, out) => {
    for (let t = 0; t < kappas.length; t += 1) if (kappas[t] > 0) out[t] += ceiling ? slabMean(kappas[t] / dy, a / Lx, b / Lz, 0, H - y0) : slabMean(kappas[t] / dy, a / Lx, b / Lz, y0, H);
  }, kappas.length);
  return { values: Array.from(r.values, (v) => v * Lx * Lz), change: r.change, grid: `${r.grid} x ${r.grid} directions` };
}

/**
 * This method's bands for a cell's gas at its stored temperature (plan 16.43,
 * points 12 to 14), from the solver's own functions on the mole fractions the
 * solver takes: for j = 0 to 8, kappa_j + kappa_s (1/m; kappa_0 = 0), the
 * source sigma B_j = sigma [a_j(T, T) T^4 - a_j(T, Ta) Ta^4] (W/m2; a
 * negative one 0, as the method sets it), and, for black radiation from Tb,
 * `share`: sigma [a_j(T, Tb) Tb^4 - a_j(T, Ta) Ta^4] (W/m2).
 */
function bandsOf(sim, c, Tb = null) {
  const { xc, xw } = fractionsOf(sim, c);
  const T = sim.T[c];
  const C = new Float64Array(NB), Ct = new Float64Array(NB), a = new Float64Array(NB + 1), a0 = new Float64Array(NB + 1), ab = new Float64Array(NB + 1);
  G.gasCellBands(MODEL, T, xc, xw, C, 0, Ct, 0);
  G.gasWeights(MODEL, Ct, 0, T, T, xc, xw, a);
  G.gasWeights(MODEL, Ct, 0, T, TA, xc, xw, a0);
  if (Tb !== null) G.gasWeights(MODEL, Ct, 0, T, Tb, xc, xw, ab);
  const molar = F.P_AMBIENT / (F.R_GAS * T);
  const ks = sim.P.sootAbsorption * sim.Y[c * NS + S.SOOT];
  return Array.from({ length: NB + 1 }, (_, j) => ({
    kappa: (j ? molar * C[j - 1] : 0) + ks,
    source: Math.max(0, SIGMA * (a[j] * T ** 4 - a0[j] * TA ** 4)),
    share: Tb !== null ? SIGMA * (ab[j] * Tb ** 4 - a0[j] * TA ** 4) : 0,
  }));
}

// ------------------------------------------------------------- (b)

function checkB() {
  console.log("\n(b) emissivity and absorptivity: the solver's 72 values, then RadCal's");
  const sums = [0.03, 0.15], ratiosB = [1.25, 1.9], lengths = [0.3, 1, 3];
  const airO2 = 0.232 / 0.032 / (0.232 / 0.032 + 0.768 / MOLAR_N2);
  const cases = [];
  const column = (X, Mr) => {
    const xc = X / (1 + Mr), xw = X - xc;
    return { xc, xw, xo: (1 - X) * airO2, xn: (1 - X) * (1 - airO2) };
  };
  const value = (T, Tb, col, L) => {
    const s = solverState(T, Tb, col.xc, col.xw);
    let e = 0;
    for (let j = 0; j < NB; j += 1) e += s.a[j + 1] * -Math.expm1(-s.kappa[j] * L);
    return e;
  };
  for (const kind of ["eps", "abs"]) {
    for (const T of kind === "eps" ? [400, 600, 900, 1200] : [400, 600]) {
      for (const X of sums) {
        for (const Mr of ratiosB) {
          for (const L of lengths) {
            const col = column(X, Mr);
            const key = `${kind}-T${T}-X${X}-Mr${Mr}-L${L}`;
            cases.push({ key, kind: kind === "eps" ? "emissivity" : "absorptivity", T_K: T, ...(kind === "abs" ? { source_K: RADTMP } : {}), pressure_atm: 1, X_CO2: col.xc, X_H2O: col.xw, X_O2: col.xo, X_N2: col.xn, Mr, L_m: L, solver: value(T, kind === "eps" ? T : RADTMP, col, L) });
          }
        }
      }
    }
  }
  const thin = [];
  for (const T of [400, 600, 900, 1200]) {
    for (const X of sums) {
      for (const Mr of ratiosB) {
        const col = column(X, Mr);
        const s = solverState(T, T, col.xc, col.xw);
        let k = 0;
        for (let j = 0; j < NB; j += 1) k += s.a[j + 1] * s.kappa[j];
        thin.push({ key: `planck-T${T}-X${X}-Mr${Mr}`, T_K: T, X, Mr, X_CO2: col.xc, X_H2O: col.xw, solverThin_perM: k });
      }
    }
  }
  const out = flag("--b-json", path.join(os.tmpdir(), "gas-radiation-b-solver.json"));
  writeFileSync(out, JSON.stringify({ tool: report.tool, check: "16.43 (b)", method: "rank correlated SLW, eight gray gases, the tables of Pearson et al. (2014) at 1 atm", count: cases.length, cases, thinLimit: thin }, null, 1));
  console.log(`  ${cases.filter((c) => c.kind === "emissivity").length} emissivities and ${cases.filter((c) => c.kind === "absorptivity").length} absorptivities written to ${out}`);
  report.checks.b = { cases: cases.length, file: out, values: cases, thin };
  // Reported: the eight gray gases beside the tables' own limit.
  if (SOURCE_DIR) {
    const X = new Transcription(SOURCE_DIR);
    let lo = Infinity, hi = 0;
    for (const c of cases) {
      c.limit = X.column(c.T_K, c.kind === "emissivity" ? c.T_K : RADTMP, c.X_CO2, c.X_H2O, c.L_m);
      lo = Math.min(lo, c.solver / c.limit);
      hi = Math.max(hi, c.solver / c.limit);
    }
    reported(`eight gray gases over the tables' own limit (4000 cross-sections) at the 72 conditions: ${lo.toFixed(3)} to ${hi.toFixed(3)} (the draft: 0.983 to 1.036)`);
  }
  // Reported: the draft's values.
  const drafted = flag("--drafted", null);
  if (drafted) {
    const rows = JSON.parse(readFileSync(drafted, "utf8")).rows;
    const range = { j8: [Infinity, 0], lblE: [Infinity, 0], lblA: [Infinity, 0] };
    const note = (r, v) => {
      r[0] = Math.min(r[0], v);
      r[1] = Math.max(r[1], v);
    };
    for (const row of rows) {
      const e = cases.find((c) => c.key === `eps-T${row.T}-X${row.X}-Mr${row.Mr}-L${row.L}`), a = cases.find((c) => c.key === `abs-T${row.T}-X${row.X}-Mr${row.Mr}-L${row.L}`);
      if (e && Number.isFinite(row.eps_J8)) note(range.j8, e.solver / row.eps_J8);
      if (a && Number.isFinite(row.alpha_J8)) note(range.j8, a.solver / row.alpha_J8);
      if (e && Number.isFinite(row.eps_lbl)) {
        e.lineByLine = row.eps_lbl;
        note(range.lblE, e.solver / row.eps_lbl);
      }
      if (a && Number.isFinite(row.alpha_lbl)) {
        a.lineByLine = row.alpha_lbl;
        note(range.lblA, a.solver / row.alpha_lbl);
      }
    }
    reported(`the solver over the draft's own eight-gas values, where the draft has them: ${range.j8[0].toFixed(5)} to ${range.j8[1].toFixed(5)}`);
    reported(`the solver over the draft's line-by-line values at 400 and 600 K: emissivity ${range.lblE[0].toFixed(3)} to ${range.lblE[1].toFixed(3)}, absorptivity ${range.lblA[0].toFixed(3)} to ${range.lblA[1].toFixed(3)}`);
  }
  console.log("  the thin limit, sum of a_j kappa_j (1/m), reported:");
  const planck = flag("--planck", null);
  const planckRows = planck ? (JSON.parse(readFileSync(planck, "utf8")).results || []).filter((r) => Number.isFinite(r.planck_mean_per_m)) : [];
  for (const p of thin) {
    const lbl = planckRows.find((r) => r.T === p.T_K && r.X === p.X && r.Mr === p.Mr);
    console.log(`    T ${p.T_K} K, X ${p.X}, Mr ${p.Mr}: ${p.solverThin_perM.toFixed(4)}${lbl ? `; line-by-line Planck mean ${lbl.planck_mean_per_m.toFixed(4)} (ratio ${(p.solverThin_perM / lbl.planck_mean_per_m).toFixed(3)})` : ""}`);
  }
  const file = flag("--radcal", null);
  if (!file) {
    console.log("  not judged: no --radcal file (the plan's reference is RadCal; no other is taken)");
    for (const c of cases) console.log(`    ${c.key.padEnd(30)} ${c.solver.toFixed(5)}`);
    return;
  }
  const raw = JSON.parse(readFileSync(file, "utf8"));
  const ref = raw.values ?? raw;
  let worst = 0, failed = 0;
  const ratios = {};
  for (const c of cases) {
    const r = ref[c.key];
    const ok = Number.isFinite(r) && Math.abs(c.solver - r) <= Math.max(0.25 * Math.abs(r), 0.01);
    if (!ok) failed += 1;
    c.radcal = Number.isFinite(r) ? r : null;
    if (Number.isFinite(r)) {
      c.ofTolerance = Math.abs(c.solver - r) / Math.max(0.25 * Math.abs(r), 0.01);
      worst = Math.max(worst, c.ofTolerance);
      const g = `${c.kind} at ${c.T_K} K`;
      ratios[g] ||= [Infinity, 0];
      ratios[g][0] = Math.min(ratios[g][0], c.solver / r);
      ratios[g][1] = Math.max(ratios[g][1], c.solver / r);
    }
    verdict("b", ok, `${c.key.padEnd(30)} solver ${c.solver.toFixed(5)}, RadCal ${Number.isFinite(r) ? r.toFixed(5) : "missing"}, ratio ${Number.isFinite(r) ? (c.solver / r).toFixed(3) : "-"}${c.lineByLine ? `, line by line ${c.lineByLine.toFixed(5)}` : ""}${c.limit ? `, the tables' limit ${c.limit.toFixed(5)}` : ""} (within 25 % or 0.01)`);
  }
  for (const [g, [lo, hi]] of Object.entries(ratios)) reported(`${g}: the solver over RadCal ${lo.toFixed(3)} to ${hi.toFixed(3)}`);
  for (const p of thin) {
    const r = ref[p.key];
    if (Number.isFinite(r)) reported(`${p.key}: solver ${p.solverThin_perM.toFixed(4)}, RadCal's Planck mean ${r.toFixed(4)} 1/m`);
  }
  report.checks.b = { ...report.checks.b, radcal: file, failed, worstOfTolerance: worst, ratios, pass: verdicts.b };
}

// ------------------------------------------------------------- (c), (d), (h), (m)

function cube(L, h, T, Y, wallsAt = TA, options = {}) {
  const n = Math.round(L / h);
  const sim = solver(boxScene({ inner: [n, n, n], h, material: BLACK }), { ...ON(), ...options });
  for (const c of openCells(sim)) setGas(sim, c, T, Y);
  for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === F.SOLID) sim.Ts[c] = wallsAt;
  return { sim, n, gasCell: openCells(sim)[0] };
}

function checkC() {
  console.log("\n(c) gray transport in a cube: soot only, 1000 K, a 3.0 m cube, black walls at ambient");
  const taus = [0.1, 0.5, 1, 2, 5];
  const esc = cubeEscape(taus);
  console.log(`  P(tau) by quadrature (${esc.grid}; last doubling moved it ${pct(esc.change)}): ${esc.P.map((p, t) => `${taus[t]}: ${p.toFixed(5)}`).join(", ")}`);
  const L = 3.0, T = 1000;
  const rows = [];
  for (const h of [0.2, 0.3]) {
    for (let t = 0; t < taus.length; t += 1) {
      const kappa = taus[t] / L;
      const { sim, gasCell } = cube(L, h, T, mixture({ ys: kappa / F.DEFAULTS.sootAbsorption }));
      const Tg = sim.T[gasCell];
      const r = runStage(sim);
      const power = r.mean.solid / DT;
      const ref = 4 * kappa * SIGMA * (Tg ** 4 - TA ** 4) * L ** 3 * esc.P[t];
      const ratio = power / ref;
      verdict("c", Math.abs(ratio - 1) <= 0.02, `h ${h} m, kappa L ${taus[t]}: walls ${(power / 1e3).toFixed(3)} kW, reference ${(ref / 1e3).toFixed(3)} kW, ratio ${ratio.toFixed(4)} (within 2 %)`);
      verdict("f", r.worst <= 1e-9, `(c) h ${h}, kappa L ${taus[t]}: worst step imbalance ${r.worst.toExponential(2)} of its emission (at most 1e-9)`);
      reported(talliesLine(r.tallies));
      rows.push({ h, tau: taus[t], power, ref, ratio, imbalance: r.worst, tallies: r.tallies });
    }
  }
  report.checks.c = { P: esc, rows, pass: verdicts.c };
}

/** (d)'s reference for a cube of side L whose gas is cell c's: sum over j of 4 kappa_j sigma B_j V P(kappa_j L), W; and its bands. */
function cubeReference(sim, c, L) {
  const bands = bandsOf(sim, c);
  const esc = cubeEscape(bands.map((b) => b.kappa * L));
  let ref = 0;
  for (let j = 0; j <= NB; j += 1) ref += 4 * bands[j].kappa * bands[j].source * L ** 3 * esc.P[j];
  return { ref, bands, esc };
}

function checkD() {
  console.log("\n(d) band transport in a cube: X_CO2 0.05, X_H2O 0.10, no soot, black walls at ambient");
  const Y = mixture({ xc: 0.05, xw: 0.1 });
  const rows = [];
  for (const L of [1.2, 3.0]) {
    for (const T of [600, 1000]) {
      let shown = false;
      for (const h of [0.2, 0.3]) {
        const { sim, gasCell } = cube(L, h, T, Y);
        const { ref, bands, esc } = cubeReference(sim, gasCell, L);
        if (!shown) console.log(`  L ${L} m, ${T} K: kappa_j ${bands.slice(1).map((b) => b.kappa.toPrecision(4)).join(", ")} 1/m; sigma B_j ${bands.map((b) => b.source.toPrecision(4)).join(", ")} W/m2; P ${esc.P.slice(1).map((p) => p.toPrecision(4)).join(", ")} (last doubling ${pct(esc.change)})`);
        shown = true;
        const r = runStage(sim);
        const power = r.mean.solid / DT;
        const ratio = power / ref;
        verdict("d", Math.abs(ratio - 1) <= 0.02, `L ${L} m, ${T} K, h ${h} m: walls ${(power / 1e3).toFixed(3)} kW, reference ${(ref / 1e3).toFixed(3)} kW, ratio ${ratio.toFixed(4)} (within 2 %)`);
        verdict("f", r.worst <= 1e-9, `(d) L ${L}, ${T} K, h ${h}: worst step imbalance ${r.worst.toExponential(2)} of its emission (at most 1e-9)`);
        reported(`${talliesLine(r.tallies)}; the stage ${r.ms.toFixed(2)} ms a step`);
        rows.push({ L, T, h, power, ref, ratio, imbalance: r.worst, tallies: r.tallies, ms: r.ms });
      }
    }
  }
  report.checks.d = { rows, pass: verdicts.d };
}

function checkH() {
  console.log("\n(h) an isothermal box stays isothermal: (d)'s 3.0 m cube, soot 1e-3, gas and black walls at one temperature");
  const Y = mixture({ xc: 0.05, xw: 0.1, ys: 1e-3 });
  const rows = [];
  const cases = [
    { part: 1, T: 600, floor: 0, judged: true },
    { part: 2, T: 400, floor: 1, judged: true },
    { part: 3, T: 400, floor: 0, judged: false },
  ];
  for (const k of cases) {
    for (const h of [0.2, 0.3]) {
      const { sim } = cube(3.0, h, k.T, Y, k.T, { surfaceEmitFloor: k.floor });
      const r = runStage(sim);
      const net = r.mean.gas - r.mean.gasEmit;
      const rel = net / r.mean.gasEmit;
      const line = `${k.part}: ${k.T} K, surfaceEmitFloor ${k.floor}, h ${h} m: gas emits ${(r.mean.gasEmit / DT / 1e3).toFixed(3)} kW, absorbs ${(r.mean.gas / DT / 1e3).toFixed(3)} kW, net ${pct(rel)} of its emission`;
      if (k.judged) {
        verdict("h", Math.abs(rel) <= 0.02, `${line} (within 2 %)`);
        verdict("f", r.worst <= 1e-9, `(h) ${k.part}, h ${h}: worst step imbalance ${r.worst.toExponential(2)} of its emission (at most 1e-9)`);
      } else reported(`${line} (not judged: the walls return nothing below 500 K)`);
      reported(`walls emit ${(r.mean.surfEmit / DT / 1e3).toFixed(3)} kW, solids take ${(r.mean.solid / DT / 1e3).toFixed(3)} kW, ${(r.mean.out / DT / 1e3).toFixed(3)} kW leaves the grid; ${talliesLine(r.tallies)}`);
      rows.push({ ...k, h, gasEmit: r.mean.gasEmit / DT, gasAbsorbed: r.mean.gas / DT, wallEmit: r.mean.surfEmit / DT, solid: r.mean.solid / DT, out: r.mean.out / DT, rel, imbalance: r.worst, tallies: r.tallies });
    }
  }
  report.checks.h = { rows, pass: verdicts.h };
}

function checkM() {
  console.log("\n(m) the number of rays changes no mean: (d)'s 3.0 m cube at 600 K, gasRays 2, 8 and 32 (outside what ends the line)");
  const Y = mixture({ xc: 0.05, xw: 0.1 });
  const rows = [];
  for (const h of [0.2, 0.3]) {
    const runs = [2, 8, 32].map((gasRays) => {
      const { sim } = cube(3.0, h, 600, Y, TA, { gasRays });
      const r = runStage(sim);
      const mean = sum(r.solids) / r.solids.length;
      let v = 0;
      for (const s of r.solids) v += (s - mean) ** 2;
      const se = Math.sqrt(v / (r.solids.length - 1) / r.solids.length);
      return { gasRays, power: mean / DT, se: se / DT, ms: r.ms };
    });
    for (let p = 0; p < runs.length; p += 1) {
      for (let q = p + 1; q < runs.length; q += 1) {
        const a = runs[p], b = runs[q];
        const z = Math.abs(a.power - b.power) / Math.hypot(a.se, b.se);
        verdict("m", z <= 4, `h ${h} m: gasRays ${a.gasRays} ${a.power.toFixed(1)} W (standard error ${a.se.toFixed(1)}), gasRays ${b.gasRays} ${b.power.toFixed(1)} W (${b.se.toFixed(1)}): ${z.toFixed(2)} standard deviations apart (at most 4)`);
      }
    }
    reported(`h ${h} m: the stage ${runs.map((r) => `${r.ms.toFixed(2)} ms a step with ${r.gasRays} rays`).join(", ")}`);
    rows.push({ h, runs });
  }
  report.checks.m = { rows, pass: verdicts.m };
}

// ------------------------------------------------------------- (e), (l)

/** (e)'s box on cells of h: its counts, its sizes, and the layer's base. */
function layerBox(h) {
  const ni = Math.round(3.0 / h), nj = Math.round(2.4 / h), nk = Math.round(3.0 / h), nl = Math.round(1.2 / h);
  return { ni, nj, nk, nl, Lx: ni * h, H: nj * h, Lz: nk * h, y0: (nj - nl) * h };
}

/** (e)'s scene: the box, its upper rows gas at `T` of `Y`, air at ambient below. Returns the solver, a layer cell and the layer's cells. */
function layerScene(h, walls, options, T, Y) {
  const b = layerBox(h);
  const sim = solver(boxScene({ inner: [b.ni, b.nj, b.nk], h, material: walls === "lumped" ? BLACK : gypsum(h) }), options);
  if (walls === "gypsum" && !(sim.wall && sim.wall.cells.length)) throw new Error("the gypsum box has no one-dimensional walls");
  const air = mixture({});
  const layer = [];
  for (const c of openCells(sim)) {
    const j = Math.floor(c / sim.grid.nx) % sim.grid.ny;
    if (j > b.nj - b.nl) {
      setGas(sim, c, T, Y);
      layer.push(c);
    } else setGas(sim, c, TA, air);
  }
  return { sim, layer, box: b };
}

function checkE() {
  console.log("\n(e) a soot-free layer at 600 K heats the floor it does not touch: 3.0 x 2.4 x 3.0 m, black walls at ambient");
  const Y = mixture({ xc: 0.05, xw: 0.1 });
  const rows = [];
  for (const h of [0.2, 0.27, 0.34]) {
    let refs = null;
    for (const walls of ["lumped", "gypsum"]) {
      for (const on of [1, 0]) {
        if (walls === "gypsum" && !on) continue;
        const { sim, layer, box } = layerScene(h, walls, on ? ON() : { surfaceRayStart: 1 }, 600, Y);
        if (!refs) {
          const bands = bandsOf(sim, layer[0]);
          const kap = bands.map((b) => b.kappa);
          const fl = layerIntegrals(kap, box.Lx, box.H, box.Lz, box.y0, false);
          const ce = layerIntegrals(kap, box.Lx, box.H, box.Lz, box.y0, true);
          refs = {
            floor: bands.reduce((s, b, j) => s + b.source * fl.values[j], 0),
            ceiling: bands.reduce((s, b, j) => s + b.source * ce.values[j], 0),
            change: Math.max(fl.change, ce.change),
          };
          console.log(`  h ${h} m: box ${box.Lx.toFixed(2)} x ${box.H.toFixed(2)} x ${box.Lz.toFixed(2)} m, layer ${box.nl} rows (${(box.nl * h).toFixed(2)} m); reference floor ${(refs.floor / 1e3).toFixed(3)} kW, ceiling ${(refs.ceiling / 1e3).toFixed(3)} kW (last doubling ${pct(refs.change)})`);
        }
        const r = runStage(sim);
        const floor = (r.mean.byGroup.floor || 0) / DT, ceiling = (r.mean.byGroup.ceiling || 0) / DT;
        if (!on) {
          reported(`item off (surfaceRayStart 1): floor ${floor.toFixed(1)} W, ceiling ${ceiling.toFixed(1)} W (the gas emits nothing); the stage ${r.ms.toFixed(3)} ms a step`);
          rows.push({ h, walls, on: false, floor, ceiling, ms: r.ms });
          continue;
        }
        const rf = floor / refs.floor, rc = ceiling / refs.ceiling;
        verdict("e", Math.abs(rf - 1) <= 0.03, `h ${h} m, ${walls} walls: floor ${(floor / 1e3).toFixed(3)} kW, ratio ${rf.toFixed(4)} (within 3 %)`);
        verdict("e", Math.abs(rc - 1) <= 0.03, `h ${h} m, ${walls} walls: ceiling ${(ceiling / 1e3).toFixed(3)} kW, ratio ${rc.toFixed(4)} (within 3 %)`);
        verdict("f", r.worst <= 1e-9, `(e) h ${h}, ${walls}: worst step imbalance ${r.worst.toExponential(2)} of its emission (at most 1e-9)`);
        const convective = sim.P.wallConvection * (600 - TA) * box.Lx * box.Lz;
        reported(`the ceiling's radiative gain ${(ceiling / 1e3).toFixed(2)} kW beside its convective ${(convective / 1e3).toFixed(2)} kW at ${sim.P.wallConvection} W/m2K; ${talliesLine(r.tallies)}; the stage ${r.ms.toFixed(2)} ms a step`);
        rows.push({ h, walls, on: true, floor, ceiling, refFloor: refs.floor, refCeiling: refs.ceiling, rf, rc, convective, imbalance: r.worst, tallies: r.tallies, ms: r.ms });
      }
    }
  }
  report.checks.e = { rows, pass: verdicts.e };
}

function checkL() {
  console.log("\n(l) a black ray is split by the gas it meets: (e)'s box, the layer at ambient temperature, the floor black and hot");
  const Y = mixture({ xc: 0.05, xw: 0.1 });
  const rows = [];
  for (const h of [0.2, 0.27]) {
    for (const Tfloor of [900, 1173.15]) {
      const { sim, layer, box } = layerScene(h, "lumped", ON(), TA, Y);
      for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === F.SOLID && Math.floor(c / sim.grid.nx) % sim.grid.ny === 0) sim.Ts[c] = Tfloor;
      // The floor's stored temperature (float32), which the solver's ray carries.
      let Ts = 0;
      for (let c = 0; c < sim.N; c += 1) if (sim.kind[c] === F.SOLID && Math.floor(c / sim.grid.nx) % sim.grid.ny === 0) Ts = sim.Ts[c];
      const bands = bandsOf(sim, layer[0], Ts);
      const fl = layerIntegrals(bands.map((b) => b.kappa), box.Lx, box.H, box.Lz, box.y0, false);
      const ref = bands.reduce((s, b, j) => s + b.share * fl.values[j], 0);
      const r = runStage(sim, STEPS, layer);
      const took = r.mean.gasIn / DT;
      const ratio = took / ref;
      verdict("l", Math.abs(ratio - 1) <= 0.03, `h ${h} m, floor at ${Tfloor} K: the layer's gas takes ${(took / 1e3).toFixed(3)} kW, reference ${(ref / 1e3).toFixed(3)} kW, ratio ${ratio.toFixed(4)} (within 3 %)`);
      verdict("f", r.worst <= 1e-9, `(l) h ${h}, ${Tfloor} K: worst step imbalance ${r.worst.toExponential(2)} of its emission (at most 1e-9)`);
      reported(`the floor emits ${(r.mean.surfEmit / DT / 1e3).toFixed(2)} kW, the gas of the whole box takes ${(r.mean.gas / DT / 1e3).toFixed(3)} kW and emits ${(r.mean.gasEmit / DT).toFixed(1)} W; the shares w_j ${bands.map((b) => (b.share / (SIGMA * (Ts ** 4 - TA ** 4))).toFixed(4)).join(", ")}; ${talliesLine(r.tallies)}`);
      rows.push({ h, Tfloor, took, ref, ratio, imbalance: r.worst, tallies: r.tallies });
    }
  }
  report.checks.l = { rows, pass: verdicts.l };
}

// ------------------------------------------------------------- burner runs: (f), (g), (k)

/** (e)'s box on 0.2 m cells with a door 0.8 x 2.0 m to 1.0 m of open cells, a 100 kW burner of 0.09 m2 at the floor's centre (the suite's: lib/fds.mjs burnerBase, itemBurner). */
function burnerScene(fuel, walls = null, Rm = R) {
  const h = 0.2;
  const sc = boxScene({ inner: [15, 12, 15], h, material: walls === "gypsum" ? gypsum(h) : null, beyond: 5, door: [4, 10] });
  const { nx, ny } = sc.grid;
  const cell0 = 8 + nx * (1 + ny * 8);
  const spec = Rm.itemBurner(sc.grid, sc.solid, cell0, { alpha: 0.0117, item: { area: 0.09, mass: 1 }, fuel });
  spec.hrr = [[0, 100e3]];
  spec.design = null;
  spec.radiantFraction = 0.2;
  spec.rays = 16;
  return { ...sc, burner: spec, fuels: { contentsFuel: INERT_FUEL, ignitionFuel: INERT_FUEL } };
}

const FUELS = {
  methane: { ...F.METHANE, embers: false, ignitionTemperature: 300 },
  "class A": FC.fuelsFor("A").ignitionFuel,
};

/** The stored fields' heat that the radiation moves (J, relative to a snapshot): see (k) 1. */
function fieldsSnapshot(sim) {
  return { T: sim.T.slice(), Ts: sim.Ts.slice(), W: sim.wall ? sim.wall.T.slice() : null, last: sim.wall ? sim.wall.last.slice() : null, out: (sim.stats.radiatedOut ?? 0) + (sim.stats.burnerRadiatedOut ?? 0), round: sim.stats.radRoundoff ? sim.stats.radRoundoff.own + sim.stats.radRoundoff.deposits + sim.stats.radRoundoff.burner : 0 };
}
function fieldsChange(sim, before) {
  let gas = 0, solids = 0, wallsJ = 0;
  const W = sim.wall;
  for (let c = 0; c < sim.N; c += 1) {
    if (sim.kind[c] !== F.SOLID && sim.T[c] !== before.T[c]) gas += sim.rho[c] * F.mixtureCp(sim.Y, c * NS) * sim.V * (sim.T[c] - before.T[c]);
    if (sim.kind[c] !== F.GAS && !(W && W.index[c] >= 0) && sim.Ts[c] !== before.Ts[c]) solids += Math.max(F.solidCapacity(sim, c), 1e-6) * (sim.Ts[c] - before.Ts[c]);
  }
  if (W) {
    const A = sim.grid.h ** 2, n = W.n;
    for (let q = 0; q < W.cells.length; q += 1) {
      const s = W.meta[q] >> 2, c = W.cells[q];
      for (let p = 0; p < n; p += 1) wallsJ += W.C[s * n + p] * A * (W.T[q * n + p] - before.W[q * n + p]);
      // What the cell's mean has taken that its nodes have not yet (the walls stage spreads it).
      wallsJ += W.heat[s] * A * (sim.Ts[c] - W.last[q] - (before.Ts[c] - before.last[q]));
    }
  }
  const now = fieldsSnapshot(sim);
  return { gas, solids, walls: wallsJ, out: now.out - before.out, round: now.round - before.round, total: gas + solids + wallsJ + (now.out - before.out) };
}

/** A burner run of `seconds` with the flow on, the validation suite's substeps; `each(sim)` after every step; the ledger at every 12 s (J). */
function burnerRun(fuel, options, seconds, each = null, probe = null, Fm = F, Rm = R) {
  const scene = burnerScene(fuel, null, Rm);
  const sim = solver(scene, options, Fm);
  if (probe) sim.probe = (tag, phase) => probe(sim, tag, phase);
  const h = scene.grid.h;
  const ledger = [];
  let speed = 0;
  const dt = sim.P.dt;
  for (let k = 0; k < seconds; k += 1) {
    const end = k + 1;
    const substeps = Fm.substepsFor(sim.P, h, speed, dt);
    while (sim.time < end - 1e-9) {
      Fm.step(sim, Math.min(dt / substeps, end - sim.time));
      if (each) each(sim);
    }
    const reading = Fm.healthReadings(sim, sim.grid, sim.P, sim.V);
    speed = reading.maxSpeed;
    if (end % 12 === 0 || end === seconds) {
      let wallHeat = 0;
      for (let c = 0; c < sim.N; c += 1) if (sim.inertC[c] > 0) wallHeat += sim.inertC[c] * (sim.Ts[c] - TA);
      const st = sim.stats;
      const out = (st.radiatedOut ?? 0) + (st.burnerRadiatedOut ?? 0);
      const remainder = st.released + st.pilotHeat - (reading.gasHeat + wallHeat + st.shellHeat + (st.leakEnthalpy || 0) + out);
      ledger.push({ t: end, released: st.released, pilot: st.pilotHeat, gas: reading.gasHeat, walls: wallHeat, shell: st.shellHeat, leak: st.leakEnthalpy || 0, radiatedOut: st.radiatedOut ?? null, burnerRadiatedOut: st.burnerRadiatedOut ?? null, radiated: st.radiated, remainder });
    }
  }
  return { sim, ledger };
}

function checkBurners() {
  console.log(`\n(f), (g), (k) the burner runs: 100 kW, 0.3 m square, radiant fraction 0.2, ${SECONDS} s with the flow on, flameRadiantOnce 1`);
  const out = {};
  const variants = [["", { burnerFlameRays: 0 }, true], [", burnerFlameRays 1 (reported)", { burnerFlameRays: 1 }, false]];
  for (const [name, fuel] of Object.entries(FUELS)) {
    for (const [suffix, extra, judged] of variants) {
      if (!judged && !args.includes("--with-flame-rays")) continue;
      const label = `${name}${suffix}`;
      const say = (check, ok, line) => (judged ? verdict(check, ok, line) : reported(`${ok ? "pass" : "FAIL"}: ${line}`));
      let worstF = 0, worstFburner = 0, worstG = 0, silencedEmit = 0, silencedTook = 0, steps = 0, casts = 0, stages = 0, silencedCells = 0;
      let released0 = 0, radiant0 = 0, worstK = 0, worstKburner = 0, largestRound = 0;
      let snap = null;
      const probe = (sim, tag, phase) => {
        if (phase === 0) {
          snap = fieldsSnapshot(sim);
          return;
        }
        const d = fieldsChange(sim, snap);
        largestRound = Math.max(largestRound, Math.abs(d.round));
        if (tag === "radiation") {
          stages += 1;
          const emitted = sim.work.bandEmit[0] + sim.work.bandEmit[1];
          if (emitted > 0) worstK = Math.max(worstK, Math.abs(d.total - d.round) / emitted);
          else if (d.total !== 0) worstK = Infinity;
        } else {
          casts += 1;
          const cast = sim.work.burnerBandEmit[0];
          worstKburner = Math.max(worstKburner, Math.abs(d.total - cast - d.round) / cast);
        }
      };
      const each = (sim) => {
        const b = sim.burner;
        steps += 1;
        const r = readCall(sim, "band");
        if (r.emitted > 0) worstF = Math.max(worstF, Math.abs(r.imbalance) / r.emitted);
        else if (r.imbalance !== 0) worstF = Infinity;
        const burnt = b.released - released0, radiant = b.radiant - radiant0;
        released0 = b.released;
        radiant0 = b.radiant;
        const want = b.radiantFraction * burnt;
        if (radiant > 0) {
          const rb = readCall(sim, "burnerBand");
          worstFburner = Math.max(worstFburner, Math.abs(rb.imbalance) / rb.emitted);
          worstG = Math.max(worstG, Math.abs(rb.emitted - want) / want);
          const rec = b.flame;
          // The cells the burner burnt in this step (item 14's stamp, set in the step just made).
          if (rec && rec.step === sim.steps - 1) {
            for (let c = 0; c < sim.N; c += 1) {
              if (rec.stamp[c] !== rec.mark) continue;
              silencedCells += 1;
              if (sim.work.bandCellEmit[c] !== 0) silencedEmit += 1;
              if (sim.work.burnerBandGas[c] !== 0) silencedTook += 1;
            }
          }
        } else if (want > 0) worstG = Infinity;
      };
      const t0 = performance.now();
      const on = burnerRun(fuel, { ...ON(), flameRadiantOnce: 1, ...extra }, SECONDS, each, probe);
      const secondsOn = (performance.now() - t0) / 1e3;
      const st = on.sim.stats;
      console.log(`  ${label}: ${steps} steps in ${secondsOn.toFixed(0)} s (${((secondsOn * 1e3) / steps).toFixed(1)} ms a step); released ${(st.released / 1e6).toFixed(3)} MJ`);
      say("f", worstF <= 1e-9, `${label}: the radiation stage's worst step imbalance ${worstF.toExponential(2)} of the step's emission (at most 1e-9)`);
      say("f", worstFburner <= 1e-9, `${label}: the burner's rays' worst step imbalance ${worstFburner.toExponential(2)} of what was cast (at most 1e-9)`);
      say("g", silencedEmit === 0 && silencedTook === 0 && silencedCells > 0, `${label}: ${silencedCells} burning cell-steps; ${silencedEmit} emitted in a band and ${silencedTook} took of the burner's rays (none of either)`);
      say("g", worstG <= 1e-12, `${label}: the burner's rays carry its radiant fraction of the heat burnt within ${worstG.toExponential(2)} relative (at most 1e-12)`);
      say("k", worstK <= 1e-9 && stages > 0, `1: ${label}: over ${stages} calls of the radiation stage, the stored fields and the outflow change by at most ${worstK.toExponential(2)} of the emission beyond the counted round-off (at most 1e-9)`);
      say("k", worstKburner <= 1e-9 && casts > 0, `1: ${label}: over ${casts} casts of the burner's rays, they rise by what was cast within ${worstKburner.toExponential(2)} of it beyond the counted round-off (at most 1e-9)`);
      const rr = st.radRoundoff;
      reported(`flameSilenced ${(st.flameSilenced / 1e6).toFixed(5)} MJ, the burner's radiant part ${(on.sim.burner.radiant / 1e6).toFixed(4)} MJ, radiated ${(st.radiated / 1e6).toFixed(4)} MJ, radiatedOut ${(st.radiatedOut / 1e6).toFixed(4)} MJ, burnerRadiatedOut ${(st.burnerRadiatedOut / 1e6).toFixed(4)} MJ`);
      reported(`radFloor ${(st.radFloor / 1e6).toFixed(5)} MJ, radCapped ${(st.radCapped / 1e6).toExponential(2)} MJ, radNegative ${(st.radNegative / 1e6).toExponential(2)} MJ, radShareNegative ${(st.radShareNegative / 1e6).toExponential(2)} MJ, surfaceFloor ${(st.surfaceFloor / 1e6).toFixed(4)} MJ, surfaceCapped ${(st.surfaceCapped / 1e6).toExponential(2)} MJ; round-off counted ${rr.own.toFixed(3)} J own, ${rr.deposits.toFixed(3)} J deposits, ${rr.burner.toFixed(3)} J burner (largest in one call ${largestRound.toExponential(2)} J)`);
      reported(`${((st.gasCrossings + st.surfaceCrossings) / Math.max(1, st.gasRays + st.surfaceRays)).toFixed(2)} crossings a ray, ${(st.gasEmitters / steps).toFixed(0)} emitting gas cells a step, ${(st.gasRays / steps).toFixed(0)} gas rays and ${(st.surfaceRays / steps).toFixed(0)} surface rays a step`);
      const entry = { steps, seconds: secondsOn, worstF, worstFburner, worstG, worstK, worstKburner, stages, casts, silencedCells, silencedEmit, silencedTook, stats: { ...st }, burnerRadiant: on.sim.burner.radiant, ledgerOn: on.ledger };
      if (judged) {
        // (k) 3, reported: the first design, the remainder on and off at the ledger's times.
        const t1 = performance.now();
        const off = burnerRun(fuel, { surfaceRayStart: 1, flameRadiantOnce: 1 }, SECONDS);
        const base = burnerRun(fuel, {}, SECONDS);
        const secondsOff = (performance.now() - t1) / 2e3;
        console.log(`        (k) 3, reported, not judged: the first design (13.19 (k)). Remainder, MJ: released + pilot - (gas + walls + shell + leak + what left the grid). An off run took ${secondsOff.toFixed(0)} s (${((secondsOff * 1e3) / steps).toFixed(1)} ms a step at the on run's steps).`);
        console.log(`           t    released   on        off (items 10 and 14)   off (the base's defaults)   |on| - |off| - 1 % of released, against each`);
        for (let q = 0; q < on.ledger.length; q += 1) {
          const a = on.ledger[q], b = off.ledger[q], c = base.ledger[q];
          const m1 = Math.abs(a.remainder) - Math.abs(b.remainder) - 0.01 * a.released, m2 = Math.abs(a.remainder) - Math.abs(c.remainder) - 0.01 * a.released;
          console.log(`        ${String(a.t).padStart(4)} ${(a.released / 1e6).toFixed(3).padStart(9)} ${(a.remainder / 1e6).toFixed(4).padStart(9)} ${(b.remainder / 1e6).toFixed(4).padStart(12)} ${(c.remainder / 1e6).toFixed(4).padStart(22)} ${(m1 / 1e6).toFixed(4).padStart(22)} ${(m2 / 1e6).toFixed(4).padStart(9)}`);
        }
        const last = on.ledger.at(-1), lastOff = off.ledger.at(-1), lastBase = base.ledger.at(-1);
        reported(`at ${last.t} s: gas ${(last.gas / 1e6).toFixed(3)} MJ on, ${(lastOff.gas / 1e6).toFixed(3)} off, ${(lastBase.gas / 1e6).toFixed(3)} base; walls ${(last.walls / 1e6).toFixed(3)}, ${(lastOff.walls / 1e6).toFixed(3)}, ${(lastBase.walls / 1e6).toFixed(3)}; shell ${(last.shell / 1e6).toFixed(3)}, ${(lastOff.shell / 1e6).toFixed(3)}, ${(lastBase.shell / 1e6).toFixed(3)}; left the grid ${(((last.radiatedOut ?? 0) + (last.burnerRadiatedOut ?? 0)) / 1e6).toFixed(3)}, ${(((lastOff.radiatedOut ?? 0) + (lastOff.burnerRadiatedOut ?? 0)) / 1e6).toFixed(3)}, ${(((lastBase.radiatedOut ?? 0) + (lastBase.burnerRadiatedOut ?? 0)) / 1e6).toFixed(3)} MJ`);
        entry.ledgerOff = off.ledger;
        entry.ledgerBase = base.ledger;
        entry.secondsOff = secondsOff;
      }
      out[label] = entry;
    }
  }
  report.checks.burners = out;
}

// ------------------------------------------------------------- (i)

async function checkI() {
  console.log("\n(i) off is off (the parts this tool runs)");
  const loads0 = G.gasTableLoads.count;
  // The fingerprints, by the fingerprint tool itself.
  const text = execFileSync(process.execPath, [path.join(HERE, "flamescapes-fingerprint.mjs")], { encoding: "utf8" });
  const prints = [...text.matchAll(/^(.+?): \d+ cells, fingerprint ([0-9a-f]{8})/gm)].map((m) => [m[1], m[2]]);
  const want = ["d62b581d", "29793d1a"];
  prints.forEach(([name, fp], q) => verdict("i", fp === want[q], `tools/flamescapes-fingerprint.mjs, ${name}: ${fp} (the base's ${want[q]})`));
  if (prints.length !== 2) verdict("i", false, `the fingerprint tool printed ${prints.length} fingerprints, not 2`);
  const out = { fingerprints: prints };
  const base = flag("--base", null);
  const hashOf = (sim) => {
    const parts = [sim.T, sim.Ts, sim.Y, sim.u, sim.v, sim.w, sim.rho, sim.p, sim.fuelMass, sim.moisture, sim.parts.x.subarray(0, sim.parts.n * 3)];
    if (sim.wall) parts.push(sim.wall.T, sim.wall.last);
    let h = 2166136261 >>> 0;
    for (const a of parts) {
      const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
      for (let q = 0; q < bytes.length; q += 1) h = Math.imul(h ^ bytes[q], 16777619) >>> 0;
    }
    return h.toString(16).padStart(8, "0");
  };
  const runOn = (Fm, scene, options, steps) => {
    const sim = Fm.createSolver(scene, { contentsFuel: scene.fuels.contentsFuel, ignitionFuel: scene.fuels.ignitionFuel, seed: 1, ...options });
    for (let s = 0; s < steps; s += 1) Fm.step(sim, 0.05);
    return sim;
  };
  const box = (x0, y0, z0, x1, y1, z1) => ({ positions: Float32Array.from([x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1]), indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3]), material: V.MATERIAL.STONE });
  const Wd = 5, D = 4, H = 2.7, t = 0.2;
  const geometry = { floors: [box(-t, -t, -t, Wd + t, 0, D + t)], roofs: [box(-t, H, -t, Wd + t, H + t, D + t)], walls: [box(-t, 0, -t, Wd + t, H, 0), box(-t, 0, D, Wd + t, H, D + t), box(-t, 0, 0, 0, H, D), box(Wd, 0, 0, Wd + t, H, 1.5), box(Wd, 0, 2.4, Wd + t, H, D), box(Wd, 2.0, 1.5, Wd + t, H, 2.4)], stairs: [], obstacles: [] };
  const { contentsFuel, ignitionFuel, item } = FC.fuelsFor("A");
  const design = (Rm) => Rm.buildScene(geometry, { upAxis: "y", fire: [2.5, 0, 2.0], maxCells: 8000, contentsFuel, ignitionFuel, item, designFire: FC.designFireFor("A", "class"), fireLoad: 780e6, seed: 1 });
  const switches = [
    ["every switch off", {}],
    ["surfaceRayStart 1", { surfaceRayStart: 1 }],
    ["surfaceRayStart 1, surfaceEmitFloor 1, burnerFlameRays 1, flameRadiantOnce 1", { surfaceRayStart: 1, surfaceEmitFloor: 1, burnerFlameRays: 1, flameRadiantOnce: 1 }],
  ];
  const cases = [
    ["methane burner, stone walls", (Rm) => burnerScene(FUELS.methane, null, Rm)],
    ["class A burner, stone walls", (Rm) => burnerScene(FUELS["class A"], null, Rm)],
    ["class A burner, gypsum boards", (Rm) => burnerScene(FUELS["class A"], "gypsum", Rm)],
  ];
  if (base) {
    const Bf = await imp(path.resolve(base), "flamescapes.js");
    const Br = await imp(path.resolve(base), "flamescapesRun.js");
    out.cases = [];
    for (const [name, make] of cases) {
      for (const [sw, options] of switches) {
        const sc = make(R);
        const inert = sc.material ? { inert: [sc.material, sc.material, sc.material] } : {};
        const a = runOn(F, sc, { ...inert, ...options, gasRadiation: 0 }, 600);
        const b = runOn(Bf, make(Br), { ...inert, ...options }, 600);
        const ha = hashOf(a), hb = hashOf(b);
        const sa = JSON.stringify(a.stats), sb = JSON.stringify(b.stats);
        verdict("i", ha === hb && sa === sb, `${name}, ${sw}, gasRadiation 0, 600 steps of 0.05 s: state ${ha} here, ${hb} on the base; the stats ${sa === sb ? "equal, field for field" : "DIFFER"}`);
        out.cases.push({ name, switches: sw, here: ha, base: hb, statsEqual: sa === sb });
      }
    }
    const a = runOn(F, design(R), { gasRadiation: 0 }, 400), b = runOn(Bf, design(Br), {}, 400);
    const ha = hashOf(a), hb = hashOf(b);
    verdict("i", ha === hb && JSON.stringify(a.stats) === JSON.stringify(b.stats), `the app's design fire (buildScene, 400 steps of 0.05 s), gasRadiation 0: state ${ha} here, ${hb} on the base; the stats ${JSON.stringify(a.stats) === JSON.stringify(b.stats) ? "equal" : "DIFFER"}`);
    out.cases.push({ name: "design fire", here: ha, base: hb });
  } else {
    console.log("  not run: the comparison of states with the base (--base DIR, the base's src/core)");
    // Still a run with the switch off, for the reader's count.
    runOn(F, burnerScene(FUELS["class A"]), { gasRadiation: 0 }, 100);
    runOn(F, design(R), {}, 100);
  }
  // runScene, the app's run layer, with the switch off.
  const scene = design(R);
  R.runScene(scene, { nodes: Float32Array.from([2.5, 0, 2.0]), horizon: 10, frameSeconds: 5, particles: 0 });
  const loads = G.gasTableLoads.count - loads0;
  verdict("i", loads === 0 && F.DEFAULTS.gasRadiation === 0, `over these runs with gasRadiation 0 (its default: ${F.DEFAULTS.gasRadiation}), and a runScene of the app's design fire, the tables' reader was called ${loads} times (0)`);
  // The solver refuses to start with the switch on and what it needs missing.
  const refuses = (label, make) => {
    let message = null;
    try {
      make();
    } catch (e) {
      message = e.message;
    }
    verdict("i", message !== null, `gasRadiation 1 ${label}: the solver ${message !== null ? `refuses to start ("${message.slice(0, 70)}...")` : "STARTS"}`);
  };
  refuses("without the tables", () => solver(boxScene({ inner: [3, 3, 3], h: 0.2 }), { gasRadiation: 1, surfaceRayStart: 1 }));
  refuses("without surfaceRayStart", () => solver(boxScene({ inner: [3, 3, 3], h: 0.2 }), { gasRadiation: 1, gasTables: MODEL || { co2: 1, h2o: 1, levels: 1, bandLevels: 1 } }));
  refuses("with a radiating burner and flameRadiantOnce 0", () => solver(burnerScene(FUELS.methane), { gasRadiation: 1, surfaceRayStart: 1, gasTables: MODEL || { co2: 1, h2o: 1, levels: 1, bandLevels: 1 } }));
  out.loads = loads;
  report.checks.i = { ...out, pass: verdicts.i };
}

// ------------------------------------------------------------- (i): the GPU check's reports

/** Leaves of a JSON value: path -> value. */
function leaves(x, p = "", out = new Map()) {
  if (x && typeof x === "object") {
    if (Array.isArray(x)) x.forEach((v, q) => leaves(v, `${p}[${q}]`, out));
    else for (const [k, v] of Object.entries(x)) leaves(v, p ? `${p}.${k}` : k, out);
  } else out.set(p, x);
  return out;
}
/** A timing leaf of the GPU check (plan 16.18's set): wall times, anything under an `ms` object, and every leaf of a `timing` list. */
const TIMING = /(^|\.)(compileMs|ms|cpuMs|gpuMs|encodeMsPerStep|gpuGridOnlyMs)$|(^|\.)ms\.|(^|\.)timing(\[|\.|$)/;
/**
 * The GPU runScene result a leaf belongs to (plan 16.27, decision 4, as
 * 16.37 records the rule built: a run of one phase), or null for a leaf of
 * no GPU run: the CPU side of every phase, and every leaf of the
 * deterministic phases, which are judged bit for bit. The runScene phases'
 * GPU runs: each sprinklers and design run's `gpu`, each systems run's,
 * embers' and the worker's `gpu`, the ensemble's GPU members together, and
 * phase full's GPU run, which holds every leaf of that phase that is not the
 * CPU's or the scene's size.
 */
function gpuRunOf(k) {
  const m = k.match(/^((?:sprinklers|design)\[\d+\]|systems\.runs\[\d+\]|embers|worker|ensemble\.runs)\.gpu(?:\.|\[|$)/);
  if (m) return `${m[1]}.gpu`;
  if (/^full\./.test(k) && !/(^|\.)cpu(\.|\[|$)|Cpu$/.test(k) && !/^full\.(cells|nodes|frames|horizon|detectors\.count)$/.test(k)) return "full.gpu";
  return null;
}

/**
 * (i), the GPU check's reports, leaf by leaf, by the rule item 17's tool
 * built (plan 16.37, "Departures"; tools/porous-exchange-check.mjs on its
 * branch): a GPU run any of whose leaves differ between two runs of the base
 * is set apart, its leaves reported beside the base's range; every other
 * leaf, timings apart, is judged bit for bit against every base run.
 */
function checkIGpu() {
  console.log("\n(i) off is off: tools/flamescapes-gpu-check.mjs, this branch with the switch off against the base");
  const baseFiles = (flag("--gpu-base", "") || "").split(",").filter(Boolean), branchFile = flag("--gpu-branch", null);
  if (baseFiles.length < 2 || !branchFile) {
    console.log("  not run: --gpu-base a.json,b.json (two or more runs of the base) and --gpu-branch c.json are needed");
    return;
  }
  const bases = baseFiles.map((f) => leaves(JSON.parse(readFileSync(f, "utf8"))));
  const branch = leaves(JSON.parse(readFileSync(branchFile, "utf8")));
  const timing = (k) => TIMING.test(k);
  const keys = new Set([...bases.flatMap((b) => [...b.keys()]), ...branch.keys()]);
  // The GPU runs that vary: a run any of whose leaves differ between two base runs.
  const nondet = new Set();
  let timings = 0;
  for (const k of keys) {
    if (timing(k)) {
      timings += 1;
      continue;
    }
    const run = gpuRunOf(k);
    if (!run) continue;
    const vals = bases.map((b) => b.get(k));
    if (vals.some((v) => !Object.is(v, vals[0]))) nondet.add(run);
  }
  const judged = [], differ = [], baseOwnElse = [], reportedRuns = new Map();
  for (const k of keys) {
    if (timing(k)) continue;
    const run = gpuRunOf(k);
    const vals = bases.map((b) => b.get(k));
    const v = branch.get(k);
    if (run && nondet.has(run)) {
      const nums = vals.filter((x) => typeof x === "number");
      const r = reportedRuns.get(run) || { leaves: 0, outside: [] };
      r.leaves += 1;
      if (typeof v === "number" && nums.length === vals.length && (v < Math.min(...nums) || v > Math.max(...nums))) r.outside.push(`${k}: ${v} against ${Math.min(...nums)} to ${Math.max(...nums)}`);
      else if (typeof v !== "number" && !vals.some((x) => Object.is(x, v))) r.outside.push(`${k}: ${JSON.stringify(v)} against ${JSON.stringify([...new Set(vals)])}`);
      reportedRuns.set(run, r);
      continue;
    }
    judged.push(k);
    if (vals.some((x) => !Object.is(x, vals[0]))) baseOwnElse.push(k);
    if (!vals.every((x) => Object.is(x, v))) differ.push(`${k}: branch ${JSON.stringify(v)}, base ${JSON.stringify(vals)}`);
  }
  verdict("i", differ.length === 0 && judged.length > 0, `tools/flamescapes-gpu-check.mjs, gasRadiation 0: of ${keys.size} leaves, ${timings} timings set apart; ${judged.length} judged bit for bit against the base's ${bases.length} runs (every leaf of the deterministic phases and the CPU side of every phase), ${differ.length} differ${differ.length ? `: ${differ.slice(0, 8).join("; ")}` : ""}`);
  if (baseOwnElse.length) reported(`${baseOwnElse.length} judged leaves differ between the base's own runs: ${baseOwnElse.slice(0, 8).join(", ")}`);
  for (const [run, r] of reportedRuns) reported(`the GPU run ${run} varies between the ${bases.length} base runs: its ${r.leaves} leaves reported, not judged; the branch's outside the base's range in ${r.outside.length}${r.outside.length ? `: ${r.outside.slice(0, 4).join("; ")}` : ""}`);
  report.checks.iGpu = { base: baseFiles, branch: branchFile, leaves: keys.size, timings, judged: judged.length, differ, nondet: [...nondet], baseOwnElse, reported: Object.fromEntries(reportedRuns) };
}

// ------------------------------------------------------------- (k) 2: the suite's ledger

function checkK2() {
  console.log("\n(k) 2: the suite's ledger carries the new term");
  const suite = path.join(HERE, "calibration", "validation-suite.mjs");
  const suiteBase = flag("--suite-base", null);
  const test = flag("--suite-test", "steckler/14$");
  const cells = flag("--suite-cells", "0.34");
  const tmp = flag("--suite-out", os.tmpdir());
  const run = (file, extra, name) => {
    const out = path.join(tmp, `gas-radiation-k2-${name}.json`);
    execFileSync(process.execPath, [file, "--datasets", "steckler", "--tests", test, "--cells", cells, "--jobs", "1", ...(flag("--suite-horizon", null) ? ["--horizon-cap", flag("--suite-horizon", null)] : []), "--json", out, ...extra], { encoding: "utf8", env: { ...process.env, FLAMESCAPES_GAS_TABLES: TABLES_FILE || "" }, stdio: ["ignore", "pipe", "inherit"] });
    return JSON.parse(readFileSync(out, "utf8"));
  };
  const common = ["--surface-ray-start", "--flame-radiant-once"];
  const on = run(suite, [...common, "--gas-radiation"], "on");
  const off = run(suite, common, "off");
  const ledgerOn = on.runs[0].ledger, ledgerOff = off.runs[0].ledger;
  if (!ledgerOn || !ledgerOff) throw new Error(`the suite's run kept no ledger: ${JSON.stringify(on.runs[0].health)}`);
  const wanted = ["radiated", "radiatedOut", "burnerRadiatedOut", "flameSilenced", "radFloor", "radCapped", "radNegative", "radShareNegative", "gasRays", "gasCrossings", "gasEmitters", "surfaceFloor", "surfaceCapped", "radRoundoff"];
  const missing = wanted.filter((k) => !ledgerOn.every((row) => k in row && row[k] !== null));
  verdict("k", ledgerOn.length > 0 && missing.length === 0, `2: with --gas-radiation, ${on.runs[0].id}: each of the ledger's ${ledgerOn.length} rows records radiatedOut and the item's tallies beside radiated${missing.length ? ` (MISSING: ${missing.join(", ")})` : ""}`);
  reported(`its last row: ${JSON.stringify(ledgerOn.at(-1))}`);
  if (suiteBase) {
    const base = run(path.resolve(suiteBase), common, "base");
    const same = JSON.stringify(base.runs[0].ledger) === JSON.stringify(ledgerOff);
    verdict("k", same, `2: without it, the ledger equals the base suite's (${path.basename(suiteBase)}) field for field over its ${ledgerOff.length} rows`);
    const bare = run(suite, [], "bare"), bareBase = run(path.resolve(suiteBase), [], "bare-base");
    verdict("k", JSON.stringify(bare.runs[0].ledger) === JSON.stringify(bareBase.runs[0].ledger), `2: with no switch at all, the ledger equals the base suite's field for field over its ${bare.runs[0].ledger.length} rows`);
  } else console.log("  not run: the comparison with the base's suite (--suite-base FILE)");
  report.checks.k2 = { on: ledgerOn, off: ledgerOff, pass: verdicts.k };
}

// ------------------------------------------------------------- speed

function checkSpeed() {
  console.log("\nspeed (reported): the radiation stage alone, ms a step, on and off (off: surfaceRayStart 1, gasRadiation 0)");
  const Y = mixture({ xc: 0.05, xw: 0.1 });
  const rows = [];
  const time = (label, make) => {
    const a = runStage(make(ON()), Math.min(STEPS, 50)), b = runStage(make({ surfaceRayStart: 1 }), Math.min(STEPS, 50));
    console.log(`    ${label}: on ${a.ms.toFixed(2)} ms a step (${(a.tallies.gasEmitters / Math.min(STEPS, 50)).toFixed(0)} emitting gas cells, ${((a.tallies.gasCrossings + a.tallies.surfaceCrossings) / Math.max(1, a.tallies.gasRays + a.tallies.surfaceRays)).toFixed(2)} crossings a ray), off ${b.ms.toFixed(3)} ms a step`);
    rows.push({ label, on: a.ms, off: b.ms, emitters: a.tallies.gasEmitters / Math.min(STEPS, 50) });
  };
  for (const L of [1.2, 3.0]) {
    for (const h of [0.2, 0.3]) {
      time(`(d) L ${L} m, 600 K, h ${h} m`, (options) => {
        const n = Math.round(L / h);
        const sim = solver(boxScene({ inner: [n, n, n], h, material: BLACK }), options);
        for (const c of openCells(sim)) setGas(sim, c, 600, Y);
        return sim;
      });
    }
  }
  for (const h of [0.2, 0.27, 0.34]) time(`(e) h ${h} m, lumped walls`, (options) => layerScene(h, "lumped", options, 600, Y).sim);
  report.checks.speed = { rows };
}

// ------------------------------------------------------------- main

if (ONLY.has("a")) checkA();
if (ONLY.has("b")) checkB();
if (ONLY.has("c")) checkC();
if (ONLY.has("d")) checkD();
if (ONLY.has("e")) checkE();
if (ONLY.has("h")) checkH();
if (ONLY.has("l")) checkL();
if (ONLY.has("m")) checkM();
if (ONLY.has("burners")) checkBurners();
if (ONLY.has("i")) await checkI();
if (ONLY.has("ig")) checkIGpu();
if (ONLY.has("k2")) checkK2();
if (ONLY.has("speed")) checkSpeed();
if (STEPS !== 200) console.log(`\nnote: ${STEPS} steps, not the plan's 200`);
if (SECONDS !== 120 && ONLY.has("burners")) console.log(`note: burner runs of ${SECONDS} s, not the plan's 120`);
console.log("\njudged lines:", Object.entries(counts).map(([k, n]) => `(${k}) ${n.judged}, ${n.failed} failed`).join("; ") || "none");
console.log("verdicts:", Object.entries(verdicts).map(([k, v]) => `(${k}) ${v ? "pass" : "FAIL"}`).join(", ") || "none");
report.verdicts = verdicts;
report.counts = counts;
const jsonOut = flag("--json", null);
if (jsonOut) writeFileSync(jsonOut, JSON.stringify(report, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v) : k === "gasTables" ? undefined : v), 1));
