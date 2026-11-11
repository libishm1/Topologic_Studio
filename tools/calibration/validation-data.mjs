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
 *   - Steckler's steady profiles as measured (room tree, door centreline).
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
import { readGeometry, surfaceMaterial } from "./lib/fds.mjs";
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
});

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
    const url = `https://raw.githubusercontent.com/firemodels/${repo}/${COMMITS[repo]}/${file.split("/").map(encodeURIComponent).join("/")}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} for ${url}`);
    buf = Buffer.from(await r.arrayBuffer());
    mkdirSync(path.dirname(local), { recursive: true });
    writeFileSync(local, buf);
  }
  const sha256 = createHash("sha256").update(buf).digest("hex");
  const key = `${repo}/${file}`;
  if (known.has(key) && known.get(key) !== sha256) throw new Error(`digest mismatch for ${key}: ${sha256} against ${known.get(key)} recorded`);
  return { text: buf.toString("utf8"), source: { repo: `github.com/firemodels/${repo}`, commit: COMMITS[repo], path: file, sha256, bytes: buf.length } };
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
  methane: { heatOfCombustion: 50.0e6, yields: { co2: 2.744, co: 0 } },
  heptane: { heatOfCombustion: 44.6e6, yields: { co2: 3.006, co: 0.008 } },
};
const testFuel = (name, soot, co, source) => ({ name, heatOfCombustion: TEST_FUELS[name].heatOfCombustion, yields: { ...TEST_FUELS[name].yields, ...(co != null ? { co } : {}), soot }, source });

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
      burner: { centre: [r3((Math.min(...xs) + Math.max(...xs)) / 2), r3((Math.min(...ys) + Math.max(...ys)) / 2)], area: r4(area), z0: r3(fire[0].xb[4]), peakW: Math.round(Q), hrrW: [[0, 0], [10, Math.round(Q)], [1e6, Math.round(Q)]], ramp: "FDS: steady (its default 1 s rise); here a 10 s linear rise (as steckler-checks.mjs)" },
      co2PerJ: CO2_PER_J.methane,
      fuel: testFuel("methane", geo.reac?.soot ?? 0, null, "methane; soot yield from the FDS input's REAC (SOOT_YIELD 0)"),
      // Walls, ceiling, floor and the burner's block: the FDS input's default surface (13 mm ceramic fibre board).
      materials: materialsOf(geo, [defaultSurface(geo)]),
      steady: true,
      trees: [{ id: "room", x: room[0], y: room[1], H, heights: z.filter((_, q) => keep[q]).reverse(), quantities: ["hglTemperature", "hglDepth"] }],
      door: { x: door.xb[0], y: door.xb[2], soffit: opening.length ? opening[0].soffit : 1.83, sill: opening.length ? opening[0].sill : 0 },
      measured: {
        trees: { room: { T: Tin.filter((_, q) => keep[q]).reverse(), Tu: r3(jt.upper - K), zi: r3(jt.height), H } },
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
    const out = { times, trees: {} };
    const treeSpecs = [];
    for (const n of trees) {
      // Columns "T#n hh": hh inches over the floor (T#1 84 = 2.13 m in the 2.2 m burn room; T#3 96 = 2.44 m, the corridor's 8 ft ceiling).
      const columns = table.head.filter((h) => new RegExp(`^T#${n}\\s+\\d+$`).test(h));
      const heights = columns.map((c) => +(Number(c.split(/\s+/)[1]) * 0.0254).toFixed(3));
      const [x, y] = deviceXyz(geo, `TC ${n}-1`);
      const H = ceilingAt(geo, x, y);
      const red = reduceTree(table, columns, heights, H);
      out.trees[String(n)] = { Tu: red.Tu, zi: red.zi, H };
      treeSpecs.push({ id: String(n), x, y, H, heights, quantities: ["hglTemperature", "hglDepth"] });
    }
    tests.push({
      id,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ambientC: geo.tmpa,
      burner: { centre: [r3((fire.xb[0] + fire.xb[1]) / 2), r3((fire.xb[2] + fire.xb[3]) / 2)], area: r4(area), z0: r3(fire.xb[4]), peakW: Math.round(Q), hrrW: ramp.map(([t, fr]) => [t, Math.round(Q * fr)]), ramp: `FDS RAMP_Q '${surf.rampQ}' x HRRPUA ${surf.hrrpua} kW/m2 x ${area.toFixed(3)} m2 (110 kW measured, Peacock et al. 1991, as the guide states)` },
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
    const out = { times, trees: {}, points: {} };
    const treeSpecs = [];
    for (const n of trees) {
      const columns = Array.from({ length: 10 }, (_, q) => `Tree ${n}-${q + 1}`);
      const heights = columns.map((_, q) => deviceXyz(geo, `Tr ${n}-${q + 1}`)[2]);
      const [x, y] = deviceXyz(geo, `Tr ${n}-1`);
      const H = 3.82;
      const red = reduceTree(table, columns, heights, H);
      out.trees[String(n)] = { Tu: red.Tu, zi: red.zi, H };
      treeSpecs.push({ id: String(n), x, y, H, heights, quantities: ["hglTemperature", "hglDepth"] });
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
    tests.push({
      id,
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ventilation: closed ? "door closed, no fan (the room sealed here; the test's measured leakage, 0.06 to 0.08 m2, is below a cell)" : "door open (2 m x 2 m), no fan",
      ambientC: geo.tmpa,
      fuel,
      burner: { centre: centre.map(r3), area: r4(Math.PI * 0.25), z0: 0, shape: "circle", peakW: Math.round(Qm), hrrW: ramp.map(([t, fr]) => [t, Math.round(Qm * fr)]), ramp: calorimetry ? `FDS FIRE_RAMP x the measured HRR's mean over ${h0 + 60} to ${h1 - 30} s (${(Qm / 1000).toFixed(0)} kW; the FDS nozzle flow gives ${(Qflow / 1000).toFixed(0)} kW)` : `FDS FIRE_RAMP x the FDS input's fuel flow, ${flow} L/min of heptane at 684 kg/m3 and 44.6 MJ/kg (${(Qm / 1000).toFixed(0)} kW; no calorimetry with the door shut)`, source: calorimetry ? "calorimetry" : "fuel flow" },
      co2PerJ: CO2_PER_J[fuel],
      fuel: testFuel("heptane", 0.0149, 0.008, "heptane; soot yield 0.0149 as measured in the tests (Hamins et al. 2006, as the FDS Validation Guide quotes it); CO 0.008 from the FDS input's lumped products"),
      // Walls and ceiling: the default 25.4 mm marinite; the floor: 12.7 mm gypsum plaster on 18.3 mm pine (FDS's 'GYPSUM BOARD').
      materials: materialsOf(geo, [defaultSurface(geo), "GYPSUM BOARD"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [300, h1], ceilingJet: [cj?.start ?? 0, cj?.end ?? 10000], o2: [o2?.start ?? 0, o2?.end ?? 10000], co2: [co2?.start ?? 0, co2?.end ?? 10000], smoke: [0, 10000] },
      trees: treeSpecs,
      points,
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
    const out = { times, trees: {}, points: {} };
    const treeSpecs = [];
    for (const [id, prefix] of [["west", "TTRW"], ["east", "TTRE"]]) {
      const devs = geo.devices.filter((d) => d.id && new RegExp(`^${prefix}\\d+$`).test(d.id) && table.col(d.id) >= 0);
      const columns = devs.map((d) => d.id);
      const heights = devs.map((d) => d.xyz[2]);
      const [x, y] = devs[0].xyz;
      const H = 3.82;
      const red = reduceTree(table, columns, heights, H);
      out.trees[id] = { Tu: red.Tu, zi: red.zi, H };
      treeSpecs.push({ id, x, y, H, heights, quantities: ["hglTemperature", "hglDepth"] });
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
    tests.push({
      id: String(n),
      fdsInput: fdsFile,
      geometry: geometryOf(geo),
      ambientC: geo.tmpa,
      fuel: "heptane",
      burner: { centre: [r3((pan[0] + pan[1]) / 2), r3((pan[2] + pan[3]) / 2)], area: r4((pan[1] - pan[0]) * (pan[3] - pan[2])), z0: 0, peakW: Math.round(Qm), hrrW: ramp.map(([t, fr]) => [t, Math.round(Qm * fr)]), ramp: `FDS FIRE_RAMP x the measured HRR's (calHRR) mean over ${h0 + 60} to ${h1 - 30} s (${(Qm / 1000).toFixed(0)} kW; from the FDS nozzle flow ${(Qflow / 1000).toFixed(0)} kW)` },
      co2PerJ: CO2_PER_J.heptane,
      fuel: testFuel("heptane", geo.reac?.soot ?? 0.015, geo.reac?.co ?? 0.008, `heptane; soot ${geo.reac?.soot} and CO ${geo.reac?.co} from the FDS input's REAC`),
      // Walls and ceiling: the default 25.4 mm marinite; the floor: 25.4 mm gypsum board.
      materials: materialsOf(geo, [defaultSurface(geo), "GYPSUM BOARD"], [{ slot: 1, floor: true }]),
      windows: { hglTemperature: [0, 10000], hglDepth: [depth?.start ?? 0, depth?.end ?? h1], ceilingJet: [cj?.start ?? 0, cj?.end ?? 10000], o2: [o2?.start ?? 0, o2?.end ?? 10000], co2: [co2?.start ?? 0, co2?.end ?? 10000] },
      trees: treeSpecs,
      points,
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
    const out = { times, trees: {}, points: {} };
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
      treeSpecs.push({ id, x, y, z0, H, heights, quantities: ["hglTemperature", "hglDepth"] });
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
      burner: { centre: [r3((fire.xb[0] + fire.xb[1]) / 2), r3((fire.xb[2] + fire.xb[3]) / 2)], area: r4(area), z0: r3(fire.xb[4]), peakW: Math.round(Q * Math.max(...ramp.map((x) => x[1]))), hrrW: ramp.map(([t, fr]) => [t, Math.round(Q * fr)]), ramp: `FDS RAMP_Q '${surf.rampQ}' x HRRPUA ${surf.hrrpua} kW/m2 x ${area.toFixed(3)} m2` },
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
    const doc = {
      dataset: name,
      title: def.title,
      reference: def.reference,
      notice: NOTICE,
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
