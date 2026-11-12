/**
 * The measured data of the held-out validation suite
 * (tools/calibration/validation-suite.mjs; wiki/research/fire-calibration-plan-2026-11-11.md,
 * section 12): fetched from NIST's public repositories at fixed commits,
 * checked against their SHA-256 digests, and reduced to the numbers the suite
 * compares, in tools/calibration/data/<dataset>.json.
 *
 *   - measurements: github.com/firemodels/exp (NIST Fire Research Division
 *     Experimental Database);
 *   - test conditions (geometry, burner, heat release ramp, instrument
 *     positions) and the FDS Validation Guide's own choice of compared
 *     quantities and time windows: github.com/firemodels/fds
 *     (Validation/<name>/FDS_Input_Files, Utilities/Python/FDS_validation_dataplot_inputs.csv).
 *
 * Both repositories carry NIST's notice (kept in each reduced file): NIST
 * works, not subject to copyright in the United States, to be acknowledged
 * as NIST's, with changes marked. The raw files go to
 * tools/calibration/data/cache/ (git-ignored); only the reduced numbers are
 * kept beside the suite. Downloaded files are data: they are parsed, never run.
 *
 * The reduction (the same code as the model's, lib/stats.mjs):
 *   - each thermocouple tree's hot gas layer by Janssens and Tran (1992), as
 *     the FDS Validation Guide reduces it, at each measured time;
 *   - point readings (ceiling-jet thermocouples, O2 and CO2) as measured;
 *   - Steckler's steady profiles as measured (room tree, door centreline);
 *   - each tree's thermocouples at head height, 1.5 to 2.0 m over its floor,
 *     as measured (plan 13.2 item 1; added 2026-11-12).
 *
 *   node tools/calibration/validation-data.mjs [--datasets steckler,nbs,nistnrc,wtc] [--offline]
 *
 * --offline reduces from the cache only (no network); a digest that differs
 * from the one recorded in the reduced file stops the reduction.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { leakageOf, readGeometry, surfaceMaterial } from "./lib/fds.mjs";
import { janssensTran } from "./lib/stats.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA = path.join(HERE, "data");
const CACHE = path.join(DATA, "cache");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const OFFLINE = args.includes("--offline");

/** The commits read (2026-11-11). */
export const COMMITS = Object.freeze({
  exp: "704e0d91f15088ae41c6a30a662a7524bf5ff4e8",
  fds: "193faffb1a7607e62f1c28ee98a4a66b6daa8c39",
  // UL FSRI's 2018 compartments (read 2026-11-12, plan 13.2 item 3).
  fsri: "b7d3929f80131e3e99c62fbf5baf7d17048ad94b",
});
/** Each repository's GitHub owner and name. */
const REPOS = Object.freeze({ exp: "firemodels/exp", fds: "firemodels/fds", fsri: "ulfsri/fsri-compartments-2018" });

export const NOTICE =
  "Measurements from NIST's Fire Research Division Experimental Database (github.com/firemodels/exp) and test conditions from NIST's FDS validation inputs (github.com/firemodels/fds), both carrying NIST's notice: NIST-developed works, not subject to copyright protection within the United States; acknowledge the National Institute of Standards and Technology as the source; modified works should note the date and nature of any change. Changed here (2026-11-11): reduced to the numbers compared by tools/calibration/validation-suite.mjs (thermocouple trees reduced by Janssens and Tran's method at each measured time; point readings within the compared windows; geometry and burner read from the FDS inputs).";

// ------------------------------------------------------------ fetching

const known = new Map(); // repo/path -> sha256 recorded in an earlier reduced file

async function fetchFile(repo, file) {
  const local = path.join(CACHE, repo, ...file.split("/"));
  let buf;
  if (existsSync(local)) buf = readFileSync(local);
  else {
    if (OFFLINE) throw new Error(`not in the cache (offline): ${repo}/${file}`);
    const url = `https://raw.githubusercontent.com/${REPOS[repo]}/${COMMITS[repo]}/${file.split("/").map(encodeURIComponent).join("/")}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} for ${url}`);
    buf = Buffer.from(await r.arrayBuffer());
    mkdirSync(path.dirname(local), { recursive: true });
    writeFileSync(local, buf);
  }
  const sha256 = createHash("sha256").update(buf).digest("hex");
  const key = `${repo}/${file}`;
  if (known.has(key) && known.get(key) !== sha256) throw new Error(`digest mismatch for ${key}: ${sha256} against ${known.get(key)} recorded`);
  return { text: buf.toString("utf8"), source: { repo: `github.com/${REPOS[repo]}`, commit: COMMITS[repo], path: file, sha256, bytes: buf.length } };
}

// ---------------------------------------------------------------- CSV

/** A CSV with a header row: {head: [names trimmed], rows: [[numbers]]} (NaN for blanks and words). */
function csv(text, headerRow = 0) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  const split = (l) => l.split(",").map((s) => s.trim());
  const head = split(lines[headerRow]);
  const rows = lines.slice(headerRow + 1).map((l) => split(l).map((s) => (s === "" ? NaN : Number(s))));
  return { head, rows, col: (name) => head.findIndex((h) => h === name) };
}

/** RFC-4180-ish split (the dataplot file quotes some fields). */
function splitQuoted(line) {
  const out = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    if (q) {
      if (ch === '"') {
        if (line[i + 1] === '"') (cur += '"'), (i += 1);
        else q = false;
      } else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") out.push(cur), (cur = "");
    else cur += ch;
  }
  out.push(cur);
  return out;
}

/** The FDS Validation Guide's dataplot rows (switch 'd') for an exp file: [{quantity, columns, start, end, initial, metric}]. */
function dataplotRows(text, expFile) {
  const lines = text.split(/\r?\n/);
  const head = splitQuoted(lines[0]);
  const ix = (n) => head.indexOf(n);
  const out = [];
  for (const l of lines.slice(1)) {
    const r = splitQuoted(l);
    if (r[0] !== "d" || r[ix("d1_Filename")] !== expFile) continue;
    out.push({
      quantity: r[ix("Quantity")],
      columns: r[ix("d1_Dep_Col_Name")].split("|").map((s) => s.trim()),
      modelColumns: r[ix("d2_Dep_Col_Name")].split("|").map((s) => s.trim()),
      start: Number(r[ix("d1_Comp_Start")]),
      end: Number(r[ix("d1_Comp_End")]),
      initial: Number(r[ix("d1_Initial_Value")]),
      metric: r[ix("Metric")],
    });
  }
  return out;
}

// --------------------------------------------------------- reductions

const K = 273.15;
const r3 = (x) => (Number.isFinite(x) ? +x.toFixed(3) : null);
const r4 = (x) => (Number.isFinite(x) ? +x.toFixed(5) : null);
const valid = (t) => Number.isFinite(t) && t > -50 && t < 2000;

/** Each time's Janssens and Tran reduction of a tree given its columns' heights (°C in, °C out). */
function reduceTree(table, columns, heights, H) {
  const idx = columns.map((c) => table.col(c));
  if (idx.some((i) => i < 0)) throw new Error(`missing tree columns ${columns.filter((c, q) => idx[q] < 0)}`);
  const Tu = [], zi = [], Tl = [];
  for (const row of table.rows) {
    const zs = [], Ts = [];
    idx.forEach((i, q) => {
      if (valid(row[i])) (zs.push(heights[q]), Ts.push(row[i] + K));
    });
    const r = zs.length >= 3 ? janssensTran(zs, Ts, H) : { height: null, upper: null, lower: null };
    Tu.push(r.upper == null ? null : r3(r.upper - K));
    zi.push(r.height == null ? null : r3(r.height));
    Tl.push(r.lower == null ? null : r3(r.lower - K));
  }
  return { Tu, zi, Tl };
}

/**
 * A tree's head-height thermocouples (plan 13.2 item 1, added 2026-11-12):
 * those 1.5 to 2.0 m over the tree's floor, each with its measured series
 * (°C, null where invalid). The plume rule is the suite's (lib/head.mjs).
 */
const HEAD_BAND = [1.5, 2.0];
function headOf(table, columns, heights) {
  const devices = [], series = {};
  columns.forEach((c, q) => {
    const z = heights[q];
    if (!(z >= HEAD_BAND[0] - 1e-9 && z <= HEAD_BAND[1] + 1e-9)) return;
    const i = table.col(c);
    if (i < 0) throw new Error(`missing head column ${c}`);
    devices.push({ id: c, z });
    series[c] = table.rows.map((r) => (valid(r[i]) ? r3(r[i]) : null));
  });
  return { devices, series };
}

/**
 * The door-probe rule (2026-11-12, before any model comparison of door
 * velocity): a single column of door probes is compared only when it carries
 * about as much mass out as in. Each probe stands for its band of the door
 * (halfway to its neighbours, the sill to the soffit) across the door's width,
 * the gas's density 353 / T from its paired thermocouple; out over in, over
 * the compared window, must lie within 1/3 to 3. A column outside it misreads
 * the flow (a probe not reading, or the inflow passing beside the column),
 * and its velocities and neutral plane are left out of that test.
 */
function doorBalance(probes, M, width, sill, soffit, window, outSign) {
  const col = probes.slice().sort((a, b) => a.z - b.z);
  let mIn = 0, mOut = 0;
  M.times.forEach((t, iq) => {
    if (t < window[0] || t > window[1]) return;
    col.forEach((p, q) => {
      const lo = q === 0 ? sill : (col[q - 1].z + p.z) / 2;
      const hi = q === col.length - 1 ? soffit : (p.z + col[q + 1].z) / 2;
      const v = M.series[p.id][iq], T = M.temperatures?.[p.id]?.[iq];
      if (v == null || T == null) return;
      const m = (353 / (T + 273.15)) * Math.abs(v) * width * (hi - lo);
      if (v * outSign > 0) mOut += m;
      else mIn += m;
    });
  });
  const ratio = mIn > 0 ? mOut / mIn : Infinity;
  return { outOverIn: Number.isFinite(ratio) ? r3(ratio) : null, window, usable: ratio >= 1 / 3 && ratio <= 3 };
}

/** The hold window of an FDS ramp [[t, f]]: from where it first reaches its top to where it leaves it. */
function holdWindow(ramp) {
  const top = Math.max(...ramp.map((r) => r[1]));
  const at = ramp.filter((r) => r[1] >= 0.95 * top).map((r) => r[0]);
  return [Math.min(...at), Math.max(...at)];
}

/** The plain-data geometry the suite needs from an FDS input. */
const geometryOf = (g) => ({ meshes: g.meshes, obsts: g.obsts.map(({ xb, surf, surfs }) => ({ xb, ...(surf ? { surf } : {}), ...(surfs ? { surfs } : {}) })), holes: g.holes, open: g.open, tmpa: g.tmpa });

/**
 * The walls' materials (plan section 12): the FDS input's own surfaces, at most
 * three (the solver's three inert slots), each with the cells it covers:
 * `rules` in order, a cell taking the first whose box (grown by a cell)
 * holds its centre and, for a floor rule, whose centre is under the floor;
 * the rest take slot 0.
 */
const materialsOf = (geo, surfaces, rules = []) => ({ slots: surfaces.map((id) => surfaceMaterial(geo, id)), rules });
const defaultSurface = (geo) => Object.entries(geo.surfs).find(([, s]) => s.default)?.[0];

const deviceXyz = (geo, id) => {
  const d = geo.devices.find((x) => x.id === id);
  if (!d) throw new Error(`no device ${id}`);
  return d.xyz || [(d.xb[0] + d.xb[1]) / 2, (d.xb[2] + d.xb[3]) / 2, (d.xb[4] + d.xb[5]) / 2];
};

/** The top of the mesh over a plan point (the local ceiling, FDS z). */
const ceilingAt = (geo, x, y) => Math.max(...geo.meshes.filter((m) => x >= m[0] && x <= m[1] && y >= m[2] && y <= m[3]).map((m) => m[5]));

// ------------------------------------------------------------ datasets

/**
 * CO2 per joule of each test fuel (kg/J), for the suite's burner (the app's
 * design-fire burner with its CO2 yield set to the test fuel's CO2 per unit
 * heat; validation-suite.mjs): methane, 44.01/16.04 kg per kg over 50.0 MJ/kg;
 * heptane with FDS's soot (0.015) and CO (0.008) yields, (0.838 - 0.015 -
 * 0.008 x 12/28) x 44.01/12.01 = 3.006 kg per kg over 44.6 MJ/kg. Toluene's
 * soot yield (0.195 +- 0.052) leaves its CO2 per unit heat too uncertain: its
 * CO2 is not compared.
 */
const CO2_PER_J = { methane: 2.744 / 50.0e6, heptane: 3.006 / 44.6e6, toluene: null };

/**
 * The burner's fuel as the test's own (plan section 12.9, input fidelity, one
 * rule for every test): its heat of combustion and its yields of CO2, CO and
 * soot (kg per kg), so that the fuel mass, the oxygen, the products and the
 * soot per joule are the test fuel's. The heats and CO2 yields as above;
 * CO and soot as the test series documents them (its FDS validation input's
 * REAC, or the test report as the FDS Validation Guide quotes it).
 */
const TEST_FUELS = {
  methane: { heatOfCombustion: 50.0e6, carbon: 12.011 / 16.043, co: 0 },
  heptane: { heatOfCombustion: 44.6e6, carbon: (7 * 12.011) / 100.205, co: 0.008 },
};
/**
 * The burner's radiant fraction (plan section 12.9, amendment 8): the FDS
 * input's RADIATIVE_FRACTION, or where the input leaves it out FDS's own
 * default for its fuel (FDS User's Guide, table "Default radiative fraction
 * for some common fuels": METHANE 0.20, N-HEPTANE 0.40, TOLUENE 0.40; 0.35
 * for species it does not list).
 */
const FDS_DEFAULT_RF = { METHANE: 0.2, "N-HEPTANE": 0.4, TOLUENE: 0.4 };
const radiantOf = (geo) => {
  const given = geo.reac?.radiativeFraction;
  const fuel = String(geo.reac?.fuel || "").toUpperCase();
  return given != null
    ? { radiantFraction: given, radiantSource: `the FDS input's RADIATIVE_FRACTION ${given}` }
    : { radiantFraction: FDS_DEFAULT_RF[fuel] ?? 0.35, radiantSource: `FDS's default for ${fuel || "an unlisted fuel"} (${FDS_DEFAULT_RF[fuel] ?? 0.35}; the input gives none)` };
};
/** The test fuel, its CO2 yield by the carbon balance: the fuel's carbon less the soot's and the CO's, as CO2. */
const testFuel = (name, soot, co, source) => {
  const f = TEST_FUELS[name];
  const coY = co ?? f.co;
  const co2 = +((f.carbon - soot - (coY * 12.011) / 28.01) * (44.01 / 12.011)).toFixed(4);
  return { name, heatOfCombustion: f.heatOfCombustion, yields: { co2, co: coY, soot }, source: `${source}; CO2 by the carbon balance` };
};

async function steckler(sources, dataplot) {
  // Every test with an FDS input and measured profiles (the guide's 55).
  const ids = [10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 30, 41, 114, 116, 122, 144, 160, 161, 162, 163, 164, 165, 166, 167, 210, 212, 220, 221, 224, 240, 242, 310, 324, 410, 510, 512, 513, 514, 517, 520, 521, 522, 524, 540, 541, 542, 544, 610, 612, 622, 710, 810];
  const tests = [];
  for (const id of ids) {
    const fdsFile = `Validation/Steckler_Compartment/FDS_Input_Files/Steckler_${String(id).padStart(3, "0")}.fds`;
    const f = await fetchFile("fds", fdsFile);
    const p = await fetchFile("exp", `Steckler_Compartment/Steckler_Test_${id}.csv`);
    const l = await fetchFile("exp", `Steckler_Compartment/Steckler_Test_${id}_layer.csv`);
    sources.push(f.source, p.source, l.source);
    const geo = readGeometry(f.text);
    // The burner: the active fire vents (a 0.3 m cross of vents), at the floor or on a 0.3 m block.
    const fire = geo.fire;
    if (!fire.length) throw new Error(`Steckler ${id}: no burner`);
    const area = fire.reduce((a, v) => a + (v.xb[1] - v.xb[0]) * (v.xb[3] - v.xb[2]), 0);
    const hrrpua = geo.surfs[fire[0].surf].hrrpua;
    const xs = fire.flatMap((v) => [v.xb[0], v.xb[1]]), ys = fire.flatMap((v) => [v.xb[2], v.xb[3]]);
    const Q = hrrpua * area * 1000;
    const prof = csv(p.text);
    const col = (n) => prof.rows.map((r) => r[prof.col(n)]);
    const z = col("Height");
    const Tin = col("T_in"), VC = col("V_C"), TC = col("T_C");
    const room = deviceXyz(geo, "TC_Room");
    const door = geo.devices.find((d) => d.id === "BP_Door");
    // The guide's own two-layer reduction (cross-check only).
    const lay = csv(l.text);
    const layer = { lower: lay.rows[0][lay.col("Temp")], interface: lay.rows[1][lay.col("Height")], upper: lay.rows[2][lay.col("Temp")] };
    const H = 2.13;
    const keep = z.map((_, q) => valid(Tin[q]));
    const jt = janssensTran(z.filter((_, q) => keep[q]), Tin.filter((_, q) => keep[q]).map((t) => t + K), H);
    const opening = geo.holes.map((h) => ({ width: r3(h[3] - h[2]), sill: r3(h[4]), soffit: r3(h[5]) }));
    tests.push({
      id: String(id),
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      opening,
      ambientC: geo.tmpa,
      burner: { centre: [r3((Math.min(...xs) + Math.max(...xs)) / 2), r3((Math.min(...ys) + Math.max(...ys)) / 2)], area: r4(area), z0: r3(fire[0].xb[4]), peakW: Math.round(Q), hrrW: [[0, 0], [10, Math.round(Q)], [1e6, Math.round(Q)]], ramp: "FDS: steady (its default 1 s rise); here a 10 s linear rise (as steckler-checks.mjs)", ...radiantOf(geo) },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", geo.reac?.soot ?? 0, null, "methane; soot yield from the FDS input's REAC (SOOT_YIELD 0)"),
      // Walls, ceiling, floor and the burner's block: the FDS input's default surface (13 mm ceramic fibre board).
      materials: materialsOf(geo, [defaultSurface(geo)]),
      steady: true,
      trees: [{ id: "room", x: room[0], y: room[1], H, heights: z.filter((_, q) => keep[q]).reverse(), quantities: ["hglTemperature", "hglDepth"], head: z.map((zz, q) => (keep[q] && zz >= HEAD_BAND[0] - 1e-9 && zz <= HEAD_BAND[1] + 1e-9 ? { id: `T_in ${zz}`, z: zz } : null)).filter(Boolean) }],
      door: { x: door.xb[0], y: door.xb[2], soffit: opening.length ? opening[0].soffit : 1.83, sill: opening.length ? opening[0].sill : 0 },
      measured: {
        trees: { room: { T: Tin.filter((_, q) => keep[q]).reverse(), Tu: r3(jt.upper - K), zi: r3(jt.height), H } },
        // The steady profile's readings at head height (°C), by device.
        head: { room: Object.fromEntries(z.map((zz, q) => (keep[q] && zz >= HEAD_BAND[0] - 1e-9 && zz <= HEAD_BAND[1] + 1e-9 ? [`T_in ${zz}`, r3(Tin[q])] : null)).filter(Boolean)) },
        door: { z: z.filter((_, q) => Number.isFinite(VC[q])), V: VC.filter(Number.isFinite), T: TC.filter((_, q) => Number.isFinite(VC[q])) },
        guideLayer: layer,
      },
    });
  }
  return { dataset: "steckler", tests };
}

async function nbs(sources, dataplot) {
  const plan = { "100A": [1, 3, 4, 5], "100O": [1, 3, 4, 5], "100Z": [1, 3, 5, 8] };
  const tests = [];
  for (const [id, trees] of Object.entries(plan)) {
    const fdsFile = `Validation/NBS_Multi-Room/FDS_Input_Files/NBS_${id}.fds`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", `NBS_Multi-Room/NBS_MV${id}.csv`);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const fire = geo.fire[0];
    const surf = geo.surfs[fire.surf];
    const area = (fire.xb[1] - fire.xb[0]) * (fire.xb[3] - fire.xb[2]);
    const Q = surf.hrrpua * area * 1000;
    const ramp = geo.ramps[surf.rampQ];
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const out = { times, trees: {}, head: {} };
    const treeSpecs = [];
    for (const n of trees) {
      // Columns "T#n hh": hh inches over the floor (T#1 84 = 2.13 m in the 2.2 m burn room; T#3 96 = 2.44 m, the corridor's 8 ft ceiling).
      const columns = table.head.filter((h) => new RegExp(`^T#${n}\\s+\\d+$`).test(h));
      const heights = columns.map((c) => +(Number(c.split(/\s+/)[1]) * 0.0254).toFixed(3));
      const [x, y] = deviceXyz(geo, `TC ${n}-1`);
      const H = ceilingAt(geo, x, y);
      const red = reduceTree(table, columns, heights, H);
      out.trees[String(n)] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(table, columns, heights);
      out.head[String(n)] = hd.series;
      treeSpecs.push({ id: String(n), x, y, H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
    }
    tests.push({
      id,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ambientC: geo.tmpa,
      burner: { centre: [r3((fire.xb[0] + fire.xb[1]) / 2), r3((fire.xb[2] + fire.xb[3]) / 2)], area: r4(area), z0: r3(fire.xb[4]), peakW: Math.round(Q), hrrW: ramp.map(([t, fr]) => [t, Math.round(Q * fr)]), ramp: `FDS RAMP_Q '${surf.rampQ}' x HRRPUA ${surf.hrrpua} kW/m2 x ${area.toFixed(3)} m2 (110 kW measured, Peacock et al. 1991, as the guide states)`, ...radiantOf(geo) },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", geo.reac?.soot ?? 0, null, `natural gas as methane; soot yield from the FDS input's REAC (SOOT_YIELD ${geo.reac?.soot ?? 0})`),
      // The corridor's walls and ceiling (13 mm calcium silicate on 13 mm gypsum), the burn room's (50 mm ceramic
      // fibre) and its floor (113 mm fire brick), as the FDS input's vents give them. Not matched: the corridor's
      // floor (FDS: the default 13 mm gypsum board) and the target room's walls and floor (gypsum board, concrete)
      // take the corridor's walls' material, three slots being all the solver has.
      materials: materialsOf(geo, ["HALLWAY", "INSULATION", "FIRE BRICK"], [{ slot: 2, xb: geo.meshes[0], floor: true }, { slot: 1, xb: geo.meshes[0] }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [300, 900] },
      trees: treeSpecs,
      points: [],
      measured: out,
    });
  }
  return { dataset: "nbs", tests };
}

async function nistNrc(sources, dataplotText) {
  // Natural ventilation (door open, no fan) and no ventilation (door closed, no fan). Left out: the fan
  // tests 4, 5, 10 and 16 (the harness has no fan), and test 17 (toluene: its soot yield, 0.195 +- 0.052,
  // leaves its heat release from the fuel flow uncertain, and no calorimetry was taken with the door shut).
  const ids = ["01", "02", "03", "07", "08", "09", "13", "14", "15", "18"];
  const trees = [1, 2, 3, 5, 6, 7]; // the guide's: tree 4 left out (its TC 4-9 failed)
  const tests = [];
  for (const id of ids) {
    const fdsFile = `Validation/NIST_NRC/FDS_Input_Files/NIST_NRC_${id}.fds`;
    const expFile = `NIST_NRC/NIST_NRC_Test_${id}_file_1.csv`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", expFile);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const fuel = geo.reac?.fuel === "TOLUENE" ? "toluene" : "heptane";
    // The heat release: the FDS input's fire ramp, scaled to the measured heat release's mean over its hold.
    const ramp = geo.ramps.FIRE_RAMP;
    const [h0, h1] = holdWindow(ramp);
    const hrrCol = table.col("HRR (kW)");
    const hold = table.rows.filter((r) => r[0] >= h0 + 60 && r[0] <= h1 - 30).map((r) => r[hrrCol]).filter(Number.isFinite);
    const Qcal = hold.length ? (hold.reduce((a, b) => a + b, 0) / hold.length) * 1000 : NaN;
    // FDS's prescribed fire: the nozzle's liquid flow (L/min) of heptane, 684 kg/m3, 44.6 MJ/kg.
    const flow = geo.props.nozzle.flowRate;
    const Qflow = fuel === "heptane" ? ((flow * 684) / 60000) * 44.6e6 : null;
    // The door-shut tests had no calorimetry (their HRR column reads zero): the FDS input's fuel flow instead.
    const calorimetry = Qcal > 50e3;
    const Qm = calorimetry ? Qcal : Qflow;
    // The pan (1 m x 2 m) and the pool, a 1 m circle at its centre (the guide's note from the test director).
    const pan = geo.obsts.find((o) => o.surf === "STEEL SHEET FLOOR").xb;
    const centre = [(pan[0] + pan[1]) / 2, (pan[2] + pan[3]) / 2];
    const rows = dataplotRows(dataplotText, expFile);
    const cj = rows.find((r) => r.quantity === "Ceiling Jet Temperature");
    const o2 = rows.find((r) => r.quantity === "Oxygen Concentration");
    const co2 = rows.find((r) => r.quantity === "Carbon Dioxide Concentration");
    const out = { times, trees: {}, points: {}, head: {} };
    const treeSpecs = [];
    for (const n of trees) {
      const columns = Array.from({ length: 10 }, (_, q) => `Tree ${n}-${q + 1}`);
      const heights = columns.map((_, q) => deviceXyz(geo, `Tr ${n}-${q + 1}`)[2]);
      const [x, y] = deviceXyz(geo, `Tr ${n}-1`);
      const H = 3.82;
      const red = reduceTree(table, columns, heights, H);
      out.trees[String(n)] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(table, columns, heights);
      out.head[String(n)] = hd.series;
      treeSpecs.push({ id: String(n), x, y, H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
    }
    const points = [];
    const addPoint = (column, device, quantity, scale = 1) => {
      const i = table.col(column);
      if (i < 0) return;
      const [x, y, z] = deviceXyz(geo, device);
      out.points[column] = table.rows.map((r) => (Number.isFinite(r[i]) ? r4(r[i] * scale) : null));
      points.push({ id: column, x, y, z, quantity });
    };
    for (const c of cj ? cj.columns : []) addPoint(c, c.replace("Tree", "Tr"), "ceilingJet");
    for (const c of o2 ? o2.columns : []) addPoint(c, c.replace("-", " "), "o2");
    if (fuel !== "toluene") for (const c of co2 ? co2.columns : []) addPoint(c, c.replace("-", " "), "co2");
    // Smoke: the laser path's transmission per metre (%), compared as the guide compares it, as the
    // obscuration 100 - T (%/m); the path from the FDS input's TRANSMISSION device.
    const smoke = rows.find((r) => r.quantity === "Smoke Obscuration");
    const path = geo.devices.find((d) => d.id === "Trans" && d.xb);
    if (smoke && path) {
      const i = table.col(smoke.columns[0]);
      if (i >= 0) {
        out.points.smoke = table.rows.map((r) => (Number.isFinite(r[i]) ? r4(100 - r[i]) : null));
        const b = path.xb;
        points.push({ id: "smoke", x: (b[0] + b[1]) / 2, y: (b[2] + b[3]) / 2, z: (b[4] + b[5]) / 2, path: [[b[0], b[2], b[4]], [b[1], b[3], b[5]]], quantity: "smoke" });
      }
    }
    const closed = !geo.open.length;
    // The door's bi-directional probes (file_2, BP Door-1 to -15 where recorded; positions from the FDS input's
    // 'BP Door n'), positive into the room (+x), in the open-door tests (13.2 item 5, added 2026-11-12); the
    // three columns of five as profiles for the neutral plane; the door's mass flow in and out (file_1, kg/s).
    let probes = [], profiles = [], doorFlow = null;
    if (!closed) {
      const f2 = `NIST_NRC/NIST_NRC_Test_${id}_file_2.csv`;
      const m2 = await fetchFile("exp", f2);
      sources.push(m2.source);
      const t2 = csv(m2.text);
      const series = {};
      for (let q = 1; q <= 15; q += 1) {
        const col = `BP Door-${q}`;
        const i = t2.col(col);
        const dev = geo.devices.find((d) => d.id === `BP Door ${q}`);
        if (i < 0 || !dev) continue;
        series[col] = t2.rows.map((r) => (Number.isFinite(r[i]) && Math.abs(r[i]) < 20 ? r4(r[i]) : null));
        probes.push({ id: col, x: dev.xyz[0], y: dev.xyz[1], z: dev.xyz[2], axis: "x", sign: 1, quantity: "doorVelocity" });
      }
      const ys = [...new Set(probes.map((p) => p.y))].sort((a, b) => a - b);
      profiles = ys.map((y) => ({ id: `door y ${y}`, probes: probes.filter((p) => p.y === y).sort((a, b) => a.z - b.z).map((p) => p.id), outSign: -1 }));
      out.probes = { times: t2.rows.map((r) => r[0]), series };
      const ci = table.col("Door flow in (kg/s)"), co = table.col("Door flow out (kg/s)");
      if (ci < 0 || co < 0) throw new Error(`NIST/NRC ${id}: no door flow columns`);
      // The file gives the outflow as negative; kept here as a positive rate.
      out.doorFlow = { times, in: table.rows.map((r) => (Number.isFinite(r[ci]) ? r4(r[ci]) : null)), out: table.rows.map((r) => (Number.isFinite(r[co]) ? r4(-r[co]) : null)) };
      const o = geo.open[0].xb;
      doorFlow = { plane: o[0], y: [o[2], o[3]], z: [o[4], o[5]], inward: 1, area: r3((o[3] - o[2]) * (o[5] - o[4])) };
    }
    tests.push({
      id,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ventilation: closed ? "door closed, no fan (the room sealed here; the test's measured leakage, 0.06 to 0.08 m2, is below a cell)" : "door open (2 m x 2 m), no fan",
      ambientC: geo.tmpa,
      fuel,
      burner: { centre: centre.map(r3), area: r4(Math.PI * 0.25), z0: 0, shape: "circle", peakW: Math.round(Qm), hrrW: ramp.map(([t, fr]) => [t, Math.round(Qm * fr)]), ramp: calorimetry ? `FDS FIRE_RAMP x the measured HRR's mean over ${h0 + 60} to ${h1 - 30} s (${(Qm / 1000).toFixed(0)} kW; the FDS nozzle flow gives ${(Qflow / 1000).toFixed(0)} kW)` : `FDS FIRE_RAMP x the FDS input's fuel flow, ${flow} L/min of heptane at 684 kg/m3 and 44.6 MJ/kg (${(Qm / 1000).toFixed(0)} kW; no calorimetry with the door shut)`, source: calorimetry ? "calorimetry" : "fuel flow", ...radiantOf(geo) },
      co2PerJ: CO2_PER_J[fuel],
      fuel: testFuel("heptane", 0.0149, 0.008, "heptane; soot yield 0.0149 as measured in the tests (Hamins et al. 2006, as the FDS Validation Guide quotes it); CO 0.008 from the FDS input's lumped products"),
      // Walls and ceiling: the default 25.4 mm marinite; the floor: 12.7 mm gypsum plaster on 18.3 mm pine (FDS's 'GYPSUM BOARD').
      materials: materialsOf(geo, [defaultSurface(geo), "GYPSUM BOARD"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [300, h1], ceilingJet: [cj?.start ?? 0, cj?.end ?? 10000], o2: [o2?.start ?? 0, o2?.end ?? 10000], co2: [co2?.start ?? 0, co2?.end ?? 10000], smoke: [0, 10000], doorVelocity: [0, 10000], neutralPlane: [300, h1], doorMassFlow: [0, 10000] },
      trees: treeSpecs,
      points,
      ...(probes.length ? { probes, profiles, doorFlow } : {}),
      measured: out,
    });
  }
  return { dataset: "nistnrc", tests };
}

async function wtc(sources, dataplotText) {
  const tests = [];
  for (let n = 1; n <= 6; n += 1) {
    const fdsFile = `Validation/WTC/FDS_Input_Files/WTC_0${n}.fds`;
    const expFile = `WTC/WTCP1T${n}.csv`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", expFile);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const ramp = geo.ramps.FIRE_RAMP;
    const [h0, h1] = holdWindow(ramp);
    const q = table.col("calHRR");
    const hold = table.rows.filter((r) => r[0] >= h0 + 60 && r[0] <= h1 - 30).map((r) => r[q]).filter(Number.isFinite);
    const Qm = (hold.reduce((a, b) => a + b, 0) / hold.length) * 1000;
    const nozzles = geo.devices.filter((d) => d.prop === "nozzle").length;
    const Qflow = ((geo.props.nozzle.flowRate * nozzles * 684) / 60000) * 44.6e6;
    const pan = geo.obsts.find((o) => o.xb[4] === o.xb[5] && o.xb[1] - o.xb[0] > 0.5 && o.xb[3] - o.xb[2] > 0.5).xb;
    const rows = dataplotRows(dataplotText, expFile);
    const cj = rows.find((r) => r.quantity === "Ceiling Jet Temperature");
    const o2 = rows.find((r) => r.quantity === "Oxygen Concentration");
    const co2 = rows.find((r) => r.quantity === "Carbon Dioxide Concentration");
    const out = { times, trees: {}, points: {}, head: {} };
    const treeSpecs = [];
    for (const [id, prefix] of [["west", "TTRW"], ["east", "TTRE"]]) {
      const devs = geo.devices.filter((d) => d.id && new RegExp(`^${prefix}\\d+$`).test(d.id) && table.col(d.id) >= 0);
      const columns = devs.map((d) => d.id);
      const heights = devs.map((d) => d.xyz[2]);
      const [x, y] = devs[0].xyz;
      const H = 3.82;
      const red = reduceTree(table, columns, heights, H);
      out.trees[id] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(table, columns, heights);
      out.head[id] = hd.series;
      treeSpecs.push({ id, x, y, H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
    }
    const points = [];
    const addPoint = (column, quantity) => {
      const i = table.col(column);
      if (i < 0) return;
      const [x, y, z] = deviceXyz(geo, column);
      out.points[column] = table.rows.map((r) => (Number.isFinite(r[i]) ? r4(r[i]) : null));
      points.push({ id: column, x, y, z, quantity });
    };
    for (const c of cj ? cj.columns : []) addPoint(c, "ceilingJet");
    for (const c of o2 ? o2.columns : []) addPoint(c, "o2");
    for (const c of co2 ? co2.columns : []) addPoint(c, "co2");
    const depth = dataplotRows(dataplotText, `WTC/WTCP1T${n}_layer.csv`).find((r) => r.quantity === "HGL Depth");
    // The opening probes (13.2 item 4, added 2026-11-12): bi-directional probes along x in the low west opening
    // ("n Vel in", PIn) and the high east one ("n Vel out", POn; 9 and 10 in tests 2 to 6), positive along +x
    // (into the room at the west, out of it at the east); the probes the guide compares, each with its row's
    // window. Each opening carries flow one way: no neutral plane.
    const vf = `WTC/WTCP1T${n}_velo.csv`;
    const vm = await fetchFile("exp", vf);
    sources.push(vm.source);
    const vt = csv(vm.text);
    const vrows = dataplotRows(dataplotText, vf);
    const probes = [], probeSeries = {};
    for (const side of ["in", "out"]) {
      for (let q = 1; q <= 10; q += 1) {
        const col = `${q} Vel ${side}`;
        const i = vt.col(col);
        const dev = geo.devices.find((d) => d.id === `P${side === "in" ? "I" : "O"}${q}`);
        const row = vrows.find((r) => r.columns.includes(col));
        // The probes the guide compares: in test 1 it leaves out inflow probes 4 and 5, whose thermocouples
        // read the hot layer's 230 to 280 °C while the probes read inflow (probe-checks.mjs, 2026-11-12).
        if (i < 0 || !dev || !row) continue;
        probeSeries[col] = vt.rows.map((r) => (Number.isFinite(r[i]) && Math.abs(r[i]) < 20 ? r4(r[i]) : null));
        probes.push({ id: col, x: dev.xyz[0], y: dev.xyz[1], z: dev.xyz[2], axis: "x", sign: 1, quantity: "doorVelocity", window: [row.start, row.end] });
      }
    }
    out.probes = { times: vt.rows.map((r) => r[0]), series: probeSeries, temperatures: Object.fromEntries(["in", "out"].flatMap((side) => Array.from({ length: 10 }, (_, q) => `T${side} ${q + 1}`)).map((c) => [c, vt.head.findIndex((x) => x.replace(/\s+/g, " ") === c)]).filter(([, i]) => i >= 0).map(([c, i]) => [c, vt.rows.map((r) => (Number.isFinite(r[i]) ? r3(r[i]) : null))])) };
    tests.push({
      id: String(n),
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ambientC: geo.tmpa,
      fuel: "heptane",
      burner: { centre: [r3((pan[0] + pan[1]) / 2), r3((pan[2] + pan[3]) / 2)], area: r4((pan[1] - pan[0]) * (pan[3] - pan[2])), z0: 0, peakW: Math.round(Qm), hrrW: ramp.map(([t, fr]) => [t, Math.round(Qm * fr)]), ramp: `FDS FIRE_RAMP x the measured HRR's (calHRR) mean over ${h0 + 60} to ${h1 - 30} s (${(Qm / 1000).toFixed(0)} kW; from the FDS nozzle flow ${(Qflow / 1000).toFixed(0)} kW)`, ...radiantOf(geo) },
      co2PerJ: CO2_PER_J.heptane,
      fuel: testFuel("heptane", geo.reac?.soot ?? 0.015, geo.reac?.co ?? 0.008, `heptane; soot ${geo.reac?.soot} and CO ${geo.reac?.co} from the FDS input's REAC`),
      // Walls and ceiling: the default 25.4 mm marinite; the floor: 25.4 mm gypsum board.
      materials: materialsOf(geo, [defaultSurface(geo), "GYPSUM BOARD"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [depth?.start ?? 0, depth?.end ?? h1], ceilingJet: [cj?.start ?? 0, cj?.end ?? 10000], o2: [o2?.start ?? 0, o2?.end ?? 10000], co2: [co2?.start ?? 0, co2?.end ?? 10000], doorVelocity: [0, 10000] },
      trees: treeSpecs,
      points,
      probes,
      measured: out,
    });
  }
  return { dataset: "wtc", tests };
}

async function atf(sources, dataplotText) {
  // The FDS validation inputs and the measured averages, test by test (the inputs' names round the heat release).
  const plan = [["050", "048kW"], ["100", "097kW"], ["240", "242kW"], ["250", "250kW"], ["500", "485kW"], ["Mix", "Pulse"]];
  const tests = [];
  for (const [fdsName, expName] of plan) {
    const fdsFile = `Validation/ATF_Corridors/FDS_Input_Files/ATF_Corridors_${fdsName}_kW.fds`;
    const expFile = `ATF_Corridors/ATF_Corridors_${expName}_avg.csv`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", expFile);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const fire = geo.fire[0];
    const surf = geo.surfs[fire.surf];
    const area = (fire.xb[1] - fire.xb[0]) * (fire.xb[3] - fire.xb[2]);
    const Q = surf.hrrpua * area * 1000;
    const ramp = geo.ramps[surf.rampQ];
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const rows = dataplotRows(dataplotText, expFile);
    const cj = rows.filter((r) => r.quantity === "Ceiling Jet Temperature");
    const out = { times, trees: {}, points: {}, head: {} };
    const treeSpecs = [];
    // The layer: trees A and B in the lower corridor and C at the foot of the stairs (floor 0, ceiling 2.4 m),
    // F and G in the upper corridor (floor 2.6, ceiling 5.0 m), each its own pair.
    for (const [id, z0, top] of [["A", 0, 2.4], ["B", 0, 2.4], ["C", 0, 2.4], ["F", 2.6, 5.0], ["G", 2.6, 5.0]]) {
      const devs = geo.devices.filter((d) => d.id && new RegExp(`^${id}-\\d\\d$`).test(d.id) && d.xyz);
      const columns = devs.map((d) => `TC_${d.id.replace("-", "")}`).filter((c) => table.col(c) >= 0);
      const used = devs.filter((d) => table.col(`TC_${d.id.replace("-", "")}`) >= 0);
      const heights = used.map((d) => +(d.xyz[2] - z0).toFixed(4));
      const [x, y] = used[0].xyz;
      const H = +(top - z0).toFixed(3);
      const red = reduceTree(table, columns, heights, H);
      out.trees[id] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(table, columns, heights);
      out.head[id] = hd.series;
      treeSpecs.push({ id, x, y, z0, H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
    }
    const points = [];
    for (const r of cj) {
      for (const c of r.columns) {
        const i = table.col(c);
        if (i < 0) continue;
        const id = c.replace("TC_", "").replace(/^([A-Z])/, "$1-");
        const [x, y, z] = deviceXyz(geo, id);
        out.points[c] = table.rows.map((row) => (Number.isFinite(row[i]) ? r4(row[i]) : null));
        points.push({ id: c, x, y, z, quantity: "ceilingJet" });
      }
    }
    tests.push({
      id: fdsName,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ambientC: geo.tmpa,
      burner: { centre: [r3((fire.xb[0] + fire.xb[1]) / 2), r3((fire.xb[2] + fire.xb[3]) / 2)], area: r4(area), z0: r3(fire.xb[4]), peakW: Math.round(Q * Math.max(...ramp.map((x) => x[1]))), hrrW: ramp.map(([t, fr]) => [t, Math.round(Q * fr)]), ramp: `FDS RAMP_Q '${surf.rampQ}' x HRRPUA ${surf.hrrpua} kW/m2 x ${area.toFixed(3)} m2`, ...radiantOf(geo) },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", geo.reac?.soot ?? 0, null, "natural gas as methane; soot yield from the FDS input's REAC (SOOT_YIELD 0)"),
      // Walls and ceilings: the default 12.7 mm gypsum board; the ground floor: concrete. Not matched: the
      // stairs (pine) and the second floor's concrete top take the gypsum board.
      materials: materialsOf(geo, [defaultSurface(geo), "CONCRETE FLOOR"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [0, 10000], ceilingJet: [cj[0]?.start ?? 0, cj[0]?.end ?? 10000] },
      trees: treeSpecs,
      points,
      measured: out,
    });
  }
  return { dataset: "atf", tests };
}

// ------------------------------------------------------------ UL FSRI 2018 (13.2 item 3, added 2026-11-12)

/** UL FSRI's licence (MIT), kept with the reduced numbers as it asks. */
export const FSRI_NOTICE =
  "Measurements from UL FSRI's 2018 compartment fire experiments (github.com/ulfsri/fsri-compartments-2018; Willi, Madrzykowski, McKinnon and Dow 2023, Data in Brief 47, 108934, doi:10.1016/j.dib.2023.108934; data doi:10.5281/zenodo.5703475). MIT License, Copyright (c) 2021 UL FSRI: Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. Changed here (2026-11-12): reduced to 5 s means of the compared channels from 60 s before ignition to the end of the compared run, and the trees' layer by Janssens and Tran's method.";

/**
 * UL FSRI 2018 (Willi et al. 2023): a 3.66 x 3.66 x 2.44 m room (the
 * README's plan), its walls and ceiling 13 mm marinite on 16 mm type X
 * gypsum, its floor 13 mm cement board (Data in Brief, section 2), one 0.9 x
 * 2.0 m door in the front wall 2.44 to 3.34 m from the left wall, open or
 * shut for the whole test. Natural gas burners, 0.6 m square (100, 250, 500
 * kW) or 0.3 m (100kWSB), their top 0.5 m over the floor, set by gas flow
 * from ignition to "Burner Off", in the corner (back left), at the back wall's
 * middle, the left wall's middle, or the centre, against the walls as the
 * README's schematic draws them. No FDS input exists: the scene is a box.
 *
 * Coordinates: x from the left wall, y from the front wall, z from the floor.
 * Trees at the quadrants' centres, 0.92 m from the walls: A front left, B back
 * left, C back right, D front right; thermocouples 25 mm and 0.3 to 2.1 m in
 * 0.3 m steps below the ceiling; O2 0.6 and 1.8 m below it. Seven doorway
 * probes on the door's centreline, 0.25 to 1.75 m below the lintel, positive
 * out of the room (the README's arrow; the data: the top probe reads out).
 *
 * The compared run: from ignition to the first of Burner Off, a door or fan
 * event, the file's end and 900 s; a test whose run would be under 120 s is
 * left out (500kW_Closed_Center_1, its door opened and shut at 12 s). The
 * ambient: the trees' mean over the 60 s before ignition (the room was
 * stratified by several kelvin before each test). Measured channels are
 * kept as 5 s means (the model's are its 1 s readings' means over the same
 * 5 s).
 */
async function fsri(sources) {
  const listing = await fetchFile("fsri", "01_Data/README.md");
  sources.push(listing.source);
  // The material properties: marinite and gypsum from NIST's WTC input, concrete from ATF's.
  const wtcGeo = readGeometry((await fetchFile("fds", "Validation/WTC/FDS_Input_Files/WTC_01.fds")).text);
  const atfGeo = readGeometry((await fetchFile("fds", "Validation/ATF_Corridors/FDS_Input_Files/ATF_Corridors_050_kW.fds")).text);
  const wall = surfaceMaterial({ ...wtcGeo, surfs: { ...wtcGeo.surfs, "FSRI WALL": { matl: ["MARINITE", "GYPSUM"], thickness: [0.013, 0.016] } } }, "FSRI WALL");
  const floor = surfaceMaterial(atfGeo, "CONCRETE FLOOR");
  const names = [];
  for (const fuel of ["100kWSB", "100kW", "250kW", "500kW"]) for (const door of ["Open", "Closed"]) for (const pos of ["Back", "Center", "Corner", "Side"]) for (const n of [1, 2, 3]) names.push(`${fuel}_${door}_${pos}_${n}`);
  const W = 3.66, D = 3.66, H = 2.44, STEP = 5;
  const below = ["25 mm", "0.3 m", "0.6 m", "0.9 m", "1.2 m", "1.5 m", "1.8 m", "2.1 m"];
  const depth = (b) => (b === "25 mm" ? 0.025 : Number(b.split(" ")[0]));
  const quads = { A: [0.92, 0.92], B: [0.92, D - 0.92], C: [W - 0.92, D - 0.92], D: [W - 0.92, 0.92] };
  const tests = [], left = [];
  for (const name of names) {
    let m, ev;
    try {
      ev = await fetchFile("fsri", `01_Data/${name}_Events.csv`);
      m = await fetchFile("fsri", `01_Data/${name}.csv`);
    } catch {
      continue; // not every configuration was run three times
    }
    sources.push(ev.source, m.source);
    const [fuelName, doorState, pos] = name.split("_");
    // Events: clock times; their offset from ignition in seconds.
    const clock = (s) => Date.parse(s.trim().replace(/^(\d{4}-\d\d-\d\d)-/, "$1T") + "Z") / 1000;
    const events = ev.text.split(/\r?\n/).slice(1).filter((l) => l.trim()).map((l) => l.split(",")).map(([t, e]) => ({ t: clock(t), e: e.trim() }));
    const ign = events.find((e) => /ignition/i.test(e.e));
    if (!ign) throw new Error(`FSRI ${name}: no ignition`);
    const table = csv(m.text);
    const t = table.rows.map((r) => r[table.col("Time")]);
    const stops = events.filter((e) => e !== ign && /burner off|door|fan/i.test(e.e)).map((e) => e.t - ign.t);
    const horizon = Math.floor(Math.min(900, Math.max(...t), ...stops));
    if (horizon < 120) {
      left.push(`${name}: run ${horizon} s (${events.map((e) => e.e).join(", ")})`);
      continue;
    }
    // 5 s means from -60 s to the horizon (the model's readings are averaged over the same 5 s).
    const times = [];
    for (let s = -60; s <= horizon; s += STEP) times.push(s);
    const blocks = times.map((s) => table.rows.filter((r, q) => t[q] > s - STEP / 2 && t[q] <= s + STEP / 2));
    const memo = new Map();
    const series = (column, scale = 1, ok = valid) => {
      const i = table.col(column);
      if (i < 0) return null;
      const key = `${column}|${scale}`;
      if (memo.has(key)) return memo.get(key);
      const v = blocks.map((rows) => {
        const v = rows.map((r) => r[i]).filter(ok);
        return v.length ? r4((v.reduce((a, b) => a + b, 0) / v.length) * scale) : null;
      });
      memo.set(key, v);
      return v;
    };
    // The ambient: every tree thermocouple's mean over the 60 s before ignition.
    const pre = [];
    for (const q of Object.keys(quads)) for (const b of below) {
      const i = table.col(`Quadrant ${q} Temperature ${b} Below Ceiling`);
      if (i >= 0) table.rows.forEach((r, k) => t[k] >= -60 && t[k] <= 0 && valid(r[i]) && pre.push(r[i]));
    }
    const ambientC = r3(pre.reduce((a, b) => a + b, 0) / pre.length);
    const side = fuelName === "100kWSB" ? 0.3 : 0.6;
    const Q = Number(fuelName.replace(/kW.*/, "")) * 1000;
    const cx = pos === "Corner" || pos === "Side" ? side / 2 : W / 2;
    const cy = pos === "Corner" || pos === "Back" ? D - side / 2 : D / 2;
    const foot = [cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2];
    const out = { times, trees: {}, points: {}, head: {} };
    const treeSpecs = [], points = [];
    for (const [q, [x, y]] of Object.entries(quads)) {
      const columns = below.map((b) => `Quadrant ${q} Temperature ${b} Below Ceiling`).filter((c) => table.col(c) >= 0);
      const heights = columns.map((c) => +(H - depth(c.split("Temperature ")[1].replace(" Below Ceiling", ""))).toFixed(3)).reverse();
      const cols = columns.slice().reverse();
      // The tree's columns as 5 s means, then each time's Janssens and Tran reduction.
      const avg = { head: [...cols], rows: times.map((_, k) => cols.map((c) => series(c)[k])), col: (n) => cols.indexOf(n) };
      const red = reduceTree(avg, cols, heights, H);
      out.trees[q] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(avg, cols, heights);
      out.head[q] = hd.series;
      treeSpecs.push({ id: q, x, y, H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
      // The ceiling jet: the top thermocouple, in each quadrant the burner's footprint does not overlap.
      const qx = x < W / 2 ? [0, W / 2] : [W / 2, W], qy = y < D / 2 ? [0, D / 2] : [D / 2, D];
      const overlaps = foot[0] < qx[1] && foot[1] > qx[0] && foot[2] < qy[1] && foot[3] > qy[0];
      const top = `Quadrant ${q} Temperature 25 mm Below Ceiling`;
      if (!overlaps && table.col(top) >= 0) {
        out.points[top] = series(top);
        points.push({ id: top, x, y, z: H - 0.025, quantity: "ceilingJet" });
      }
      for (const b of ["0.6 m", "1.8 m"]) {
        const c = `Quad ${q} Oxygen ${b} Below Ceiling`;
        if (table.col(c) < 0) continue;
        out.points[c] = series(c, 0.01, (v) => Number.isFinite(v) && v > 0 && v < 25);
        points.push({ id: c, x, y, z: +(H - depth(b)).toFixed(3), quantity: "o2" });
      }
    }
    const open = doorState === "Open";
    let probes = [], profiles = [], balance = null;
    if (open) {
      const series1 = {};
      for (const b of ["0.25", "0.50", "0.75", "1.00", "1.25", "1.50", "1.75"]) {
        const c = `Doorway Velocity ${b} m Below Lintel`;
        if (table.col(c) < 0) continue;
        series1[c] = series(c, 1, (v) => Number.isFinite(v) && Math.abs(v) < 20);
        probes.push({ id: c, x: 2.89, y: 0, z: +(2.0 - Number(b)).toFixed(3), axis: "y", sign: -1, quantity: "doorVelocity" });
      }
      // The doorway's thermocouples beside the probes (for probe-checks.mjs's mass balance).
      const temps = {};
      for (const b of ["0.25", "0.50", "0.75", "1.00", "1.25", "1.50", "1.75"]) {
        const c = `Doorway Temperature ${b} m Below Lintel`;
        if (table.col(c) >= 0) temps[`Doorway Velocity ${b} m Below Lintel`] = series(c);
      }
      out.probes = { times, series: series1, temperatures: temps };
      profiles = [{ id: "door", probes: probes.slice().sort((a, b) => a.z - b.z).map((p) => p.id), outSign: 1 }];
      balance = doorBalance(probes, out.probes, 0.9, 0, 2.0, [horizon - 180, horizon], 1);
      if (!balance.usable) (probes = []), (profiles = []);
    }
    tests.push({
      id: name,
      source: `01_Data/${name}.csv`,
      geometry: { meshes: [[0, W, 0, D, 0, H]], obsts: [{ xb: [...foot, 0, 0.5], surf: "BURNER" }], holes: [], open: open ? [{ xb: [2.44, 3.34, 0, 0, 0, 2.0] }] : [], tmpa: ambientC },
      ventilation: open ? "door open (0.9 m x 2.0 m)" : "door shut (the room sealed here; its leakage was not measured)",
      sealedByDesign: !open,
      ambientC,
      horizon,
      events: events.map((e) => `${e.e} @ ${Math.round(e.t - ign.t)} s`),
      burner: { centre: [r3(cx), r3(cy)], area: r4(side * side), z0: 0.5, peakW: Q, hrrW: [[0, 0], [10, Q], [1e6, Q]], ramp: `the gas flow's setting, ${Q / 1000} kW (calibrated by oxygen consumption calorimetry, Data in Brief section 2), from ignition; here a 10 s linear rise`, radiantFraction: FDS_DEFAULT_RF.METHANE, radiantSource: "FDS's default for METHANE (0.20; no FDS input)" },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", 0, null, "natural gas as methane, no soot (as the NIST methane inputs)"),
      // Walls and ceiling: 13 mm marinite on 16 mm type X gypsum (properties from NIST's WTC input); the floor:
      // concrete (ATF's input). Not matched: the floor's 13 mm cement board and the burner's steel frame (a solid block here).
      materials: { slots: [wall, floor], rules: [{ slot: 1, floor: true }] },
      windows: { hglTemperature: [0, horizon], hglDepth: [120, horizon], ceilingJet: [0, horizon], o2: [0, horizon], doorVelocity: [0, horizon], neutralPlane: [horizon - 180, horizon] },
      trees: treeSpecs,
      points,
      ...(probes.length ? { probes, profiles } : {}),
      ...(balance ? { doorBalance: balance } : {}),
      measured: out,
    });
  }
  if (left.length) console.log(`fsri: left out: ${left.join("; ")}`);
  const off = tests.filter((T) => T.doorBalance && !T.doorBalance.usable);
  if (off.length) console.log(`fsri: door probes left out by the door-probe rule: ${off.map((T) => `${T.id} (${T.doorBalance.outOverIn})`).join(", ")}`);
  return { dataset: "fsri", tests, notice: FSRI_NOTICE, leftOut: left };
}

/**
 * UL/NIST vents (NIST's Fire Fighting Technology Group at UL, 2012; plan 13.2
 * item 6, added 2026-11-12): four fires in a 6.1 x 4.3 x 2.4 m room with one
 * door and two ceiling vents opened and shut in time. Only each fire's first
 * interval is compared, before any vent changes: both vents shut (the FDS
 * inputs' lids, present at t = 0), the door open, the heat release the FDS
 * ramp's to the interval's end (the guide's table: 430, 2144, 476 and 1011
 * kW). The run ends at the interval's end; each quantity is compared over the
 * guide's window for that interval (its last 60 s), as the guide does: two
 * trees (Janssens and Tran), the ceiling jet at the guide's two thermocouples,
 * the trees' head-height thermocouples, and the door's ten probes with the
 * profile's neutral plane.
 */
async function ulnist(sources, dataplotText) {
  const tests = [];
  for (let n = 1; n <= 4; n += 1) {
    const fdsFile = `Validation/UL_NIST_Vents/FDS_Input_Files/UL_NIST_Vents_Test_${n}.fds`;
    const expFile = `UL_NIST_Vents/UL_NIST_Vents_Test_${n}.csv`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", expFile);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const rows = dataplotRows(dataplotText, expFile);
    const first = rows.filter((r) => /^HGL Temperature/.test(r.quantity)).sort((a, b) => a.start - b.start)[0];
    const [w0, w1] = [first.start, first.end];
    const fire = geo.fire[0];
    const surf = geo.surfs[fire.surf];
    const area = (fire.xb[1] - fire.xb[0]) * (fire.xb[3] - fire.xb[2]);
    const Q0 = surf.hrrpua * area * 1000;
    const ramp = geo.ramps[surf.rampQ];
    // The ramp to the first interval's end, then held (the run ends there).
    const hrrW = ramp.filter(([t]) => t <= w1).map(([t, fr]) => [t, Math.round(Q0 * fr)]);
    const out = { times, trees: {}, points: {}, head: {} };
    const treeSpecs = [], points = [];
    const H = 2.4;
    for (const k of [1, 2]) {
      const devs = geo.devices.filter((d) => new RegExp(`^T${k}-TC\\d+$`).test(d.id || ""));
      const columns = devs.map((d) => `T${k}-${d.id.split("TC")[1]}`).filter((c) => table.col(c) >= 0);
      const used = devs.filter((d) => table.col(`T${k}-${d.id.split("TC")[1]}`) >= 0);
      const heights = used.map((d) => d.xyz[2]);
      const red = reduceTree(table, columns, heights, H);
      out.trees[`T${k}`] = { Tu: red.Tu, zi: red.zi, H };
      const hd = headOf(table, columns, heights);
      out.head[`T${k}`] = hd.series;
      treeSpecs.push({ id: `T${k}`, x: used[0].xyz[0], y: used[0].xyz[1], H, heights, quantities: ["hglTemperature", "hglDepth"], head: hd.devices });
    }
    // The ceiling jet: the guide's thermocouples for the first interval.
    const cj = rows.filter((r) => r.quantity === "Ceiling Jet Temperature" && r.start === w0);
    for (const c of [...new Set(cj.flatMap((r) => r.columns))]) {
      const i = table.col(c);
      if (i < 0) continue;
      const [x, y, z] = deviceXyz(geo, c.replace("-", "-TC"));
      out.points[c] = table.rows.map((r) => (Number.isFinite(r[i]) ? r4(r[i]) : null));
      points.push({ id: c, x, y, z, quantity: "ceilingJet" });
    }
    // The door's probes (D-V1 to D-V10, along y, positive into the room) and thermocouples.
    const probes = [], series = {}, temps = {};
    for (let q = 1; q <= 10; q += 1) {
      const c = `Door_V${q}`, i = table.col(c);
      const dev = geo.devices.find((d) => d.id === `D-V${q}`);
      if (i < 0 || !dev) continue;
      series[c] = table.rows.map((r) => (Number.isFinite(r[i]) && Math.abs(r[i]) < 20 ? r4(r[i]) : null));
      const it = table.col(`Door_TC${q}`);
      if (it >= 0) temps[c] = table.rows.map((r) => (valid(r[it]) ? r3(r[it]) : null));
      probes.push({ id: c, x: dev.xyz[0], y: dev.xyz[1], z: dev.xyz[2], axis: "y", sign: 1, quantity: "doorVelocity" });
    }
    out.probes = { times, series, temperatures: temps };
    const door = geo.holes.find((hb) => hb[4] <= 0.01 && hb[5] > 1.5) || [0, 0.9, 0, 0, 0, 2.1];
    const balance = doorBalance(probes, out.probes, door[1] - door[0], door[4], door[5], [w0, w1], -1);
    tests.push({
      id: String(n),
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ventilation: "door open (0.9 m x 2.1 m), both ceiling vents shut (the first interval)",
      ambientC: geo.tmpa,
      horizon: w1,
      interval: [w0, w1],
      burner: { centre: [r3((fire.xb[0] + fire.xb[1]) / 2), r3((fire.xb[2] + fire.xb[3]) / 2)], area: r4(area), z0: r3(fire.xb[4]), peakW: Math.max(...hrrW.map((r) => r[1])), hrrW, ramp: `FDS RAMP_Q '${surf.rampQ}' x HRRPUA ${surf.hrrpua} kW/m2 x ${area.toFixed(3)} m2, to the first interval's end (${w1} s)`, ...radiantOf(geo) },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", geo.reac?.soot ?? 0, null, "natural gas as methane; soot yield from the FDS input's REAC (none given: 0)"),
      // Walls and ceiling: the default 16 mm type X gypsum board; the floor: concrete.
      materials: materialsOf(geo, [defaultSurface(geo), "CONCRETE FLOOR"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [w0, w1], hglDepth: [w0, w1], ceilingJet: [w0, w1], doorVelocity: [w0, w1], neutralPlane: [w0, w1] },
      trees: treeSpecs,
      points,
      ...(balance.usable ? { probes, profiles: [{ id: "door", probes: probes.slice().sort((a, b) => a.z - b.z).map((p) => p.id), outSign: -1 }] } : {}),
      doorBalance: balance,
      measured: out,
    });
  }
  const off = tests.filter((T) => !T.doorBalance.usable);
  if (off.length) console.log(`ulnist: door probes left out by the door-probe rule: ${off.map((T) => `${T.id} (${T.doorBalance.outOverIn})`).join(", ")}`);
  return { dataset: "ulnist", tests };
}

/**
 * NIST FSE 2008 (Lock et al.; plan 13.2 item 8, added 2026-11-12): an ISO 9705
 * room, 2.4 x 3.6 x 2.4 m, its door 0.1 to 0.8 m wide. Only the tests with a
 * door of 0.4 m or more are built (ISONG3, 0.8 m; ISOHept25 and 26, 0.4 m);
 * the rest have 0.1 to 0.2 m doors, which the grid seals. The fire: the FDS
 * input's fuel mass flux ramp times the burner's area and the fuel's heat of
 * combustion (natural gas as methane, 50.0 MJ/kg, which the measured heat
 * release confirms within 1 %; heptane 44.6 MJ/kg). The run ends at the end
 * of the guide's first window for the test, and each quantity is compared
 * over that window: O2 and CO2 at the front and rear sample points (dry, as
 * measured), their thermocouples as the guide's ceiling jet, and the soot
 * mass fraction where measured. The layer is not compared: the repository
 * holds only the guide's reduced layer, not the trees.
 */
async function fse(sources, dataplotText) {
  const tests = [];
  for (const [id, fuelName, door] of [["ISONG3", "methane", 0.8], ["ISOHept25", "heptane", 0.4], ["ISOHept26", "heptane", 0.4]]) {
    const fdsFile = `Validation/NIST_FSE_2008/FDS_Input_Files/${id}.fds`;
    const expFile = `NIST_FSE_2008/${id}.csv`;
    const f = await fetchFile("fds", fdsFile);
    const m = await fetchFile("exp", expFile);
    sources.push(f.source, m.source);
    const geo = readGeometry(f.text);
    const table = csv(m.text);
    const times = table.rows.map((r) => r[0]);
    const rows = dataplotRows(dataplotText, expFile);
    const o2row = rows.filter((r) => r.quantity === "Oxygen Concentration").sort((a, b) => a.start - b.start)[0];
    if (!o2row) {
      console.log(`fse: ${id}: the guide compares nothing; left out`);
      continue;
    }
    const [w0, w1] = [o2row.start, o2row.end];
    // The fire: the vent whose surface has a mass flux (readGeometry finds heat release per area only).
    const fireSurf = /&SURF\s+ID\s*=\s*'fire'[^/]*MASS_FLUX\(1\)\s*=\s*([0-9.]+)/i.exec(f.text);
    const fireVent = /&VENT\s+XB\s*=\s*([-0-9., ]+?),\s*SURF_ID\s*=\s*'fire'/i.exec(f.text);
    const dH = TEST_FUELS[fuelName].heatOfCombustion;
    const ramp = geo.ramps.fire_ramp;
    let xb, area, rate, how;
    if (fireSurf && fireVent) {
      // A gas burner: its vent's fuel mass flux (kg/(m2 s)).
      xb = fireVent[1].split(",").map(Number);
      area = (xb[1] - xb[0]) * (xb[3] - xb[2]);
      rate = Number(fireSurf[1]) * area;
      how = `FDS MASS_FLUX ${fireSurf[1]} kg/(m2 s) x ${area.toFixed(2)} m2`;
    } else if (geo.props.nozzle) {
      // A heptane spray onto a pan (as NIST/NRC and WTC): the nozzle's flow (L/min) at 684 kg/m3; the pan as the burner.
      const pan = geo.obsts.find((o) => o.xb[4] === o.xb[5] && o.xb[1] - o.xb[0] > 0.3 && o.xb[3] - o.xb[2] > 0.3);
      if (!pan) throw new Error(`fse ${id}: no pan`);
      xb = pan.xb;
      area = (xb[1] - xb[0]) * (xb[3] - xb[2]);
      rate = (geo.props.nozzle.flowRate * 684) / 60000;
      how = `the FDS nozzle's ${geo.props.nozzle.flowRate} L/min of heptane at 684 kg/m3, onto the ${area.toFixed(2)} m2 pan`;
    } else throw new Error(`fse ${id}: no fire`);
    const hrrW = ramp.filter(([t]) => t <= w1).map(([t, fr]) => [t, Math.round(rate * dH * fr)]);
    const out = { times, trees: {}, points: {} };
    const points = [];
    const add = (column, device, quantity, scale = 1) => {
      const i = table.col(column);
      if (i < 0) return;
      const vals = table.rows.map((r) => (Number.isFinite(r[i]) ? r4(r[i] * scale) : null));
      if (!vals.some((v) => v != null)) return;
      const d = geo.devices.find((x) => x.id === device);
      const [x, y, z] = d ? d.xyz : column.endsWith("F") || /Front|TF/.test(column) ? [1.89, 0.25, 2.08] : [1.89, 2.86, 2.08];
      out.points[column] = vals;
      points.push({ id: column, x, y, z, quantity });
    };
    for (const c of ["O2Front", "O2Rear"]) add(c, c, "o2");
    for (const c of ["CO2Front", "CO2Rear"]) add(c, c, "co2");
    for (const c of ["TFSampPtRh", "TRSampPtRh"]) add(c, c, "ceilingJet");
    if (fuelName === "heptane") for (const c of ["YsootF", "YsootR"]) add(c, c, "soot");
    tests.push({
      id,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ventilation: `door ${door} m wide, 1.9 to 2.0 m high`,
      ambientC: geo.tmpa,
      horizon: w1,
      burner: { centre: [r3((xb[0] + xb[1]) / 2), r3((xb[2] + xb[3]) / 2)], area: r4(area), z0: r3(xb[4]), peakW: Math.max(...hrrW.map((r) => r[1])), hrrW, ramp: `${how} x 'fire_ramp' x ${dH / 1e6} MJ/kg, to the guide's first window's end (${w1} s)`, radiantFraction: FDS_DEFAULT_RF[fuelName === "methane" ? "METHANE" : "N-HEPTANE"], radiantSource: `FDS's default for ${fuelName === "methane" ? "METHANE" : "N-HEPTANE"} (the input sets 0 because FDS there computes the flame's radiation; this solver's burner needs a fraction)` },
      co2PerJ: CO2_PER_J[fuelName],
      fuel: fuelName === "methane" ? testFuel("methane", 0, null, "natural gas (93 % methane) as methane; no soot") : testFuel("heptane", geo.reac?.soot ?? 0.0149, geo.reac?.co ?? 0.008, `heptane; soot ${geo.reac?.soot} and CO ${geo.reac?.co} from the FDS input's REAC`),
      // Walls and ceiling: the input's default (25 mm ceramic fibre board on 13 mm gypsum); the floor: concrete.
      materials: materialsOf(geo, [defaultSurface(geo), "CONCRETE"], [{ slot: 1, floor: true }]),
      windows: { ceilingJet: [w0, w1], o2: [w0, w1], co2: [w0, w1], soot: [w0, w1] },
      trees: [],
      points,
      measured: out,
    });
  }
  return { dataset: "fse", tests };
}

export const DATASETS = {
  steckler: {
    run: steckler,
    title: "Steckler, Quintiere and Rinkinen (1982), compartment doorway flows (NBS)",
    reference: "Steckler, K. D., Quintiere, J. G. and Rinkinen, W. J. (1982). Flow induced by fire in a compartment. NBSIR 82-2520. doi:10.6028/NBS.IR.82-2520",
  },
  nbs: {
    run: nbs,
    title: "NBS multi-room tests 100A, 100O, 100Z (Peacock, Davis and Babrauskas 1991; tests run 1985)",
    reference: "Peacock, R. D., Davis, S. and Babrauskas, V. (1991). Data for room fire model comparisons. Journal of Research of NIST 96(4), 411-462. doi:10.6028/jres.096.022",
  },
  nistnrc: {
    run: nistNrc,
    title: "NIST/NRC compartment tests (Hamins et al. 2003 tests; NUREG/CR-6905, NIST SP 1013-1)",
    reference: "Hamins, A., Maranghides, A., Johnsson, E. et al. (2006). Report of experimental results for the international fire model benchmarking and validation exercise #3. NUREG/CR-6905 and NIST SP 1013-1. doi:10.6028/NIST.SP.1013-1",
  },
  wtc: {
    run: wtc,
    title: "NIST WTC investigation, phase 1 spray-burner compartment tests (NIST NCSTAR 1-5B)",
    reference: "Hamins, A. et al. (2005). Federal building and fire safety investigation of the World Trade Center disaster: experiments and modeling of structural steel elements exposed to fire. NIST NCSTAR 1-5B, volume 1. doi:10.6028/NIST.NCSTAR.1-5Bv1",
  },
};

DATASETS.fse = {
  run: fse,
  title: "NIST FSE 2008: fires in an ISO 9705 room, the tests with a door of 0.4 m or more, first window (NIST, Lock et al.)",
  reference: "Lock, A., Bundy, M., Johnsson, E. L., Hamins, A., Ko, G. H., Hwang, C. and Fuss, P. (2008), as described in the FDS Validation Guide (NIST SP 1018-3), section NIST Full-Scale Enclosure Experiments, 2008",
};

DATASETS.ulnist = {
  run: ulnist,
  title: "UL/NIST vents: four natural gas fires in a 6.1 x 4.3 x 2.4 m room with a door and two ceiling vents, the first interval of each (NIST at UL, 2012)",
  reference: "Opert, K. (2012). Assessment of natural vertical ventilation for structural fire fighting. M.S. thesis, University of Maryland, as described in the FDS Validation Guide (NIST SP 1018-3), section UL/NIST Vent Experiments",
};

DATASETS.fsri = {
  run: fsri,
  title: "UL FSRI 2018 compartment fires: natural gas burners 100 to 500 kW in a 3.66 x 3.66 x 2.44 m room, door open or shut (UL FSRI)",
  reference: "Willi, J., Madrzykowski, D., McKinnon, M. and Dow, N. (2023). Experimental data from compartment fires with gas burner and upholstered furniture fuels. Data in Brief 47, 108934. doi:10.1016/j.dib.2023.108934 (data: doi:10.5281/zenodo.5703475)",
};

DATASETS.atf = {
  run: atf,
  title: "ATF corridors: two corridors joined by a stairwell, natural gas 48 to 485 kW (ATF Fire Research Laboratory, 2008)",
  reference: "Sheppard, D. T. (2008), ATF corridor experiments, as described and reduced in the FDS Validation Guide (NIST SP 1018-3, release 6.11.1), section ATF Corridors; no DOI found",
};

async function main() {
  const which = flag("--datasets", Object.keys(DATASETS).join(",")).split(",");
  mkdirSync(DATA, { recursive: true });
  const dp = await fetchFile("fds", "Utilities/Python/FDS_validation_dataplot_inputs.csv");
  for (const name of which) {
    const def = DATASETS[name];
    if (!def) throw new Error(`unknown dataset ${name}`);
    const file = path.join(DATA, `${name}.json`);
    known.clear();
    if (existsSync(file)) for (const s of JSON.parse(readFileSync(file, "utf8")).sources || []) known.set(`${s.repo.split("/").pop()}/${s.path}`, s.sha256);
    const sources = [dp.source];
    const t0 = Date.now();
    const reduced = await def.run(sources, dp.text);
    // Each test's leakage as its FDS input states it (plan 13.11 item 7; 13.B passes it to the solver).
    for (const T of reduced.tests) {
      if (!T.fdsInput) continue;
      const lk = leakageOf((await fetchFile("fds", T.fdsInput)).text);
      if (lk) T.leakage = lk;
    }
    const doc = {
      dataset: name,
      title: def.title,
      reference: def.reference,
      notice: reduced.notice || NOTICE,
      ...(reduced.leftOut ? { leftOut: reduced.leftOut } : {}),
      generated: new Date().toISOString().slice(0, 10),
      tool: "tools/calibration/validation-data.mjs",
      heldOut: "never used to fit any solver parameter (fire-calibration-plan-2026-11-11.md, section 12)",
      sources: sources.filter((s, q) => sources.findIndex((x) => x.path === s.path && x.repo === s.repo) === q),
      tests: reduced.tests,
    };
    writeFileSync(file, JSON.stringify(doc) + "\n");
    console.log(`${name}: ${reduced.tests.length} tests, ${doc.sources.length} files, ${(readFileSync(file).length / 1024).toFixed(0)} kB (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
