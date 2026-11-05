/**
 * Floor-plan drawings to LOD1 rooms, headless, against a written schedule
 * (src/core/drawingToLod1.js):
 *
 *   1. read every DXF sheet of a building (readDxf), and register each storey
 *      layout as the schedule says (lod1_zones.json: "origin" per layout, or
 *      lettered grid bubbles found on the sheet);
 *   2. draft its rooms twice: "aided", with the schedule's reading settings
 *      (wall and door layers per sheet, door, fixture and furniture block
 *      patterns, text decoding), and "unaided", with suggestLayers on the
 *      sheet alone (layers, door blocks and scale), as a user who knows
 *      nothing of the drawing would get it;
 *   3. build the schedule's written rooms (rect or poly in grid tokens or
 *      metres; minus, carve and remainder) and match each to the draft room
 *      that overlaps it best, by intersection over union (IoU, sampled on a
 *      5 to 10 cm grid): found (IoU >= 0.8, the lab notebook's "found"),
 *      matched (>= 0.5), near (0.3 to 0.5), missed (< 0.3); draft rooms that
 *      match no written room at 0.3 are extra. The notebook's outcomes
 *      (found, close, merged, split, missed) are reported too;
 *   4. report the time per sheet (reading and drafting) and the issues found;
 *   5. with --variants, the aided draft again with one step changed at a
 *      time (no door blocks or swings bridged, no splitting by names, no
 *      growing, no outline closing, snapping tolerances, layers matched by
 *      name, the suggested layers), to show what each step is worth.
 *
 *   node tools/dxf-lod1-check.mjs [project folder] [--buildings a,b] [--variants]
 *                                 [--out file] [--no-out]
 *
 * The project folder holds data/<building>/lod1_zones.json and its DXF
 * sheets (data/<building>/dxf); default: $LOD1_PROJECT. The project is
 * kept outside the repository.
 *
 * The drawings are a real, private building: the results file
 * (benchmark/results/<date>-dxf-lod1.json) holds aggregates only, with the
 * buildings labelled "A", "B" ... and no room names, texts or coordinates.
 */
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { draftRooms, readDxf, suggestLayers } from "../topologicpy-web-frontend/src/core/drawingToLod1.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : fallback;
};
const positional = args.filter((a, i) => !a.startsWith("--") && !(i > 0 && ["--buildings", "--out"].includes(args[i - 1])));
const MAIN = import.meta.url === pathToFileURL(process.argv[1] || "").href;
const ROOT = [positional[0], process.env.LOD1_PROJECT]
  .filter(Boolean).find((p) => existsSync(path.join(p, "data")));
if (!ROOT && MAIN) {
  console.log("No LOD1 project found: pass its folder (the one holding data/<building>/lod1_zones.json) or set LOD1_PROJECT.");
  process.exit(1);
}
const only = flag("--buildings", null)?.split(",");
const VARIANTS = args.includes("--variants");

// ------------------------------------------------------------ registration

const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
const median = (a) => {
  const s = a.slice().sort((x, y) => x - y);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
};

/** The sheet's origin, scale and grid from its lettered grid bubbles (the lab's registration). */
function bubbles(entities, reg) {
  const rx = new RegExp(`^(${reg.labels})$`);
  const at = new Map();
  for (const e of entities) {
    if (e.type !== "text" || e.layer !== reg.layer || !rx.test(e.text.trim())) continue;
    const k = e.text.trim();
    if (!at.has(k)) at.set(k, []);
    at.get(k).push([e.x, e.y]);
  }
  const [gx, gy] = reg.origin;
  if (!at.has(gx) || !at.has(gy) || !at.has(reg.scale_check)) return null;
  const x0 = mean(at.get(gx).map((p) => p[0])), y0 = mean(at.get(gy).map((p) => p[1]));
  const s = mean(at.get(reg.scale_check).map((p) => p[0])) - x0 > 1000 ? 1e-3 : 1;
  const grid = {};
  for (const [k, pts] of at) grid[k] = Math.round((/[A-Za-z]/.test(k) ? mean(pts.map((p) => p[0])) - x0 : mean(pts.map((p) => p[1])) - y0) * s * 100) / 100;
  return { origin: [x0, y0], scale: s, grid };
}

// ------------------------------------------------------------ the written rooms

/** Crossing parity of (x, y) in a ring. */
function inRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j];
    if (yi > y !== yj > y && x < xj + ((y - yj) * (xi - xj)) / (yi - yj)) inside = !inside;
  }
  return inside;
}

/**
 * A layout's written rooms as membership tests (the lab's zone_polygons:
 * rect or poly, then minus, carve and remainder), with their boxes.
 */
function writtenRooms(spec, layout, grid) {
  const L = spec.layouts[layout];
  const xy = (v) => (typeof v === "number" ? v : grid[v]);
  const shape = (z) => {
    if (z.rect) {
      const [a, b, c, d] = z.rect.map(xy);
      return [[Math.min(a, c), Math.min(b, d)], [Math.max(a, c), Math.min(b, d)], [Math.max(a, c), Math.max(b, d)], [Math.min(a, c), Math.max(b, d)]];
    }
    return z.poly.map(([a, b]) => [xy(a), xy(b)]);
  };
  const zones = new Map();
  for (const z of L.zones) if (!z.remainder) zones.set(z.id, { id: z.id, cls: z.class, ring: shape(z), minus: z.minus || [], carve: Boolean(z.carve) });
  const carves = [...zones.values()].filter((z) => z.carve).map((z) => z.id);
  // In a room: in its shape, in none of the rooms it lists under minus, and
  // (unless it carves itself) in no room that carves.
  const inZone = (id, x, y) => {
    const z = zones.get(id);
    if (!inRing(x, y, z.ring)) return false;
    for (const m of z.minus) if (zones.has(m) && inZone(m, x, y)) return false;
    if (!z.carve) for (const c of carves) if (inZone(c, x, y)) return false;
    return true;
  };
  const box = (ring) => {
    const xs = ring.map((p) => p[0]), ys = ring.map((p) => p[1]);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  };
  const out = [];
  for (const z of zones.values()) out.push({ id: z.id, cls: z.cls, box: box(z.ring), test: (x, y) => inZone(z.id, x, y) });
  // A remainder room is the footprint less every other room; with no footprint
  // the footprint is their union, and the remainder is empty (as in the notebook).
  const footprint = L.footprint ? L.footprint.map(([a, b]) => [xy(a), xy(b)]) : null;
  for (const z of L.zones) {
    if (!z.remainder) continue;
    const others = out.slice();
    out.push(footprint
      ? { id: z.id, cls: z.class, box: box(footprint), test: (x, y) => inRing(x, y, footprint) && !others.some((q) => q.test(x, y)) }
      : { id: z.id, cls: z.class, box: [0, 0, 0, 0], test: () => false });
  }
  return out;
}

// ------------------------------------------------------------ comparison

/**
 * The draft rooms painted on a 5 cm grid (scanlines, holes by parity): the
 * room under each cell, -1 for none; where grown rooms overlap at a corner,
 * the later room wins.
 */
function roomRaster(rooms, written, res = 0.05) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const r of rooms) for (const [x, y] of r.polygon) [x0, y0, x1, y1] = [Math.min(x0, x), Math.min(y0, y), Math.max(x1, x), Math.max(y1, y)];
  for (const z of written) [x0, y0, x1, y1] = [Math.min(x0, z.box[0]), Math.min(y0, z.box[1]), Math.max(x1, z.box[2]), Math.max(y1, z.box[3])];
  const w = Math.ceil((x1 - x0) / res) + 1, h = Math.ceil((y1 - y0) / res) + 1;
  const grid = new Int32Array(w * h).fill(-1);
  rooms.forEach((r, i) => {
    const rings = [r.polygon, ...r.holes];
    const ys = r.polygon.map((p) => p[1]);
    const j0 = Math.max(0, Math.floor((Math.min(...ys) - y0) / res)), j1 = Math.min(h - 1, Math.ceil((Math.max(...ys) - y0) / res));
    for (let j = j0; j <= j1; j += 1) {
      const y = y0 + (j + 0.5) * res;
      const xs = [];
      for (const ring of rings) {
        for (let k = 0; k < ring.length; k += 1) {
          const [ax, ay] = ring[k], [bx, by] = ring[(k + 1) % ring.length];
          if (ay > y !== by > y) xs.push(ax + ((y - ay) * (bx - ax)) / (by - ay));
        }
      }
      xs.sort((a, b) => a - b);
      for (let k = 0; k + 1 < xs.length; k += 2) {
        for (let c = Math.max(0, Math.ceil((xs[k] - x0) / res - 0.5)); c < w && x0 + (c + 0.5) * res < xs[k + 1]; c += 1) grid[j * w + c] = i;
      }
    }
  });
  return (x, y) => {
    const c = Math.floor((x - x0) / res), j = Math.floor((y - y0) / res);
    return c >= 0 && j >= 0 && c < w && j < h ? grid[j * w + c] : -1;
  };
}

/**
 * Each written room against the draft rooms: the best IoU, its cover (share
 * of the written room) and purity (share of the draft room), sampled on a
 * grid (5 cm, coarser for rooms over 100 m²) over the written room's box.
 */
function compare(written, rooms) {
  const roomAt = roomRaster(rooms, written);
  const roomBest = new Float64Array(rooms.length);
  const rows = written.map((z) => {
    const [x0, y0, x1, y1] = z.box;
    const step = Math.max(0.05, Math.sqrt(((x1 - x0) * (y1 - y0)) / 40000));
    const inter = new Map();
    let area = 0;
    for (let x = x0 + step / 2; x < x1; x += step) {
      for (let y = y0 + step / 2; y < y1; y += step) {
        if (!z.test(x, y)) continue;
        area += 1;
        const r = roomAt(x, y);
        if (r >= 0) inter.set(r, (inter.get(r) || 0) + 1);
      }
    }
    const cell = step * step;
    area *= cell;
    let best = -1, iou = 0, cover = 0, purity = 0, hits = 0, hitArea = 0;
    inter.forEach((count, ri) => {
      const a = count * cell;
      if (a <= 0.01) return;
      hits += 1;
      hitArea += a;
      const u = area + rooms[ri].area - a;
      const v = u > 0 ? a / u : 0;
      roomBest[ri] = Math.max(roomBest[ri], v);
      if (v > iou) [best, iou, cover, purity] = [ri, v, a / area, a / rooms[ri].area];
    });
    // The lab notebook's outcomes.
    const outcome = iou >= 0.8 ? "found" : iou >= 0.5 ? "close" : cover >= 0.6 && purity < 0.6 ? "merged" : hits > 1 && hitArea >= 0.6 * area ? "split" : "missed";
    return { iou, cover, purity, outcome, cls: z.cls, draftCls: best >= 0 ? rooms[best].class : null };
  });
  return { rows, extra: [...roomBest].filter((v) => v < 0.3).length };
}

function tally(rows, extra, drafted) {
  const n = rows.length;
  const count = (f) => rows.filter(f).length;
  const notebook = {};
  for (const k of ["found", "close", "merged", "split", "missed"]) notebook[k] = count((r) => r.outcome === k);
  const matched = rows.filter((r) => r.iou >= 0.5);
  return {
    written: n, drafted,
    found: count((r) => r.iou >= 0.8), matched: matched.length, near: count((r) => r.iou >= 0.3 && r.iou < 0.5), missed: count((r) => r.iou < 0.3), extra,
    meanIoU: n ? Math.round((rows.reduce((s, r) => s + r.iou, 0) / n) * 1000) / 1000 : null,
    classAgreement: matched.length ? Math.round((matched.filter((r) => r.cls === r.draftCls).length / matched.length) * 100) / 100 : null,
    notebookOutcomes: notebook,
  };
}
const add = (a, b) => {
  const out = { ...a };
  for (const [k, v] of Object.entries(b)) {
    if (typeof v === "number" && typeof a[k] === "number" && !["meanIoU", "classAgreement"].includes(k)) out[k] = a[k] + v;
    else if (v && typeof v === "object" && !Array.isArray(v)) out[k] = add(a[k] || {}, v);
    else if (!(k in a)) out[k] = v;
  }
  return out;
};

// ------------------------------------------------------------ one building

const ms = (t) => Math.round(performance.now() - t);
function runBuilding(dir, label) {
  const spec = JSON.parse(readFileSync(path.join(dir, "lod1_zones.json"), "utf8"));
  const REG = spec.registration;
  const sheets = new Map();
  const sheet = (file) => {
    if (!sheets.has(file)) {
      const t = performance.now();
      const text = readFileSync(path.join(dir, "dxf", file), "utf8");
      const d = readDxf(text);
      sheets.set(file, { d, readMs: ms(t), mb: Math.round((text.length / 1e6) * 10) / 10 });
    }
    return sheets.get(file);
  };
  // Registration of every layout; for grid bubbles the grid is the sheets' median, as in the notebook.
  const reg = {};
  for (const [k, L] of Object.entries(spec.layouts)) {
    const { d } = sheet(L.dxf);
    if (REG.mode === "grid_bubbles") reg[k] = bubbles(d.entities, REG);
    else reg[k] = { origin: L.origin, scale: L.scale ?? 1, grid: { ...(spec.grid || {}) } };
  }
  let grid = { ...(spec.grid || {}) };
  if (REG.mode === "grid_bubbles") {
    const keys = new Set(Object.values(reg).filter(Boolean).flatMap((r) => Object.keys(r.grid)));
    grid = {};
    for (const key of keys) grid[key] = Math.round(median(Object.values(reg).filter((r) => r && key in r.grid).map((r) => r.grid[key])) * 100) / 100;
  }
  for (const [k, v] of Object.entries(spec.aux_lines || {})) grid[k] = v.value;

  const B = spec.blocks || {};
  const notWall = (B.not_wall || []).map((k) => B[k]).filter(Boolean).join("|") || null;
  const decode = spec.labels?.decode_arabic === "all" ? "all" : "auto";
  const layouts = [];
  const totals = { aided: {}, unaided: {} };
  const variants = {};
  for (const [k, L] of Object.entries(spec.layouts)) {
    const r = reg[k];
    if (!r) {
      layouts.push({ layout: k, error: "registration failed" });
      continue;
    }
    const { d, readMs, mb } = sheet(L.dxf);
    const written = writtenRooms(spec, k, grid);
    const cad = spec.cad_layers[k];
    const base = {
      origin: r.origin, scale: r.scale, clip: REG.clip, decode, ignoreTexts: L.ignore_labels || [],
    };
    const aidedOpts = {
      ...base, wallLayers: cad.walls, doorLayers: cad.doors, doorPattern: B.door, notWallPattern: notWall, fixturePattern: B.fixture || undefined, layerMatch: "path",
    };
    let t = performance.now();
    const aided = draftRooms(d.entities, aidedOpts);
    const aidedMs = ms(t);
    const ca = compare(written, aided.rooms);
    // Unaided: what the sheet says by itself, inside the same window.
    t = performance.now();
    const [x0, y0, x1, y1] = REG.clip;
    const inWindow = (x, y) => {
      const mx = (x - r.origin[0]) * r.scale, my = (y - r.origin[1]) * r.scale;
      return mx >= x0 && mx <= x1 && my >= y0 && my <= y1;
    };
    const region = d.entities.filter((e) => {
      if (e.type === "insert") return inWindow((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2);
      if (e.points) return inWindow(e.points[0], e.points[1]);
      return inWindow(e.x, e.y);
    });
    const sug = suggestLayers(region, { units: d.units });
    const unaided = draftRooms(d.entities, {
      origin: r.origin, scale: sug.scale, clip: REG.clip.map((v) => (v * r.scale) / sug.scale), decode: "auto",
      wallLayers: [...sug.walls, ...sug.windows], doorLayers: sug.doors, doorPattern: sug.doorPattern, layerMatch: "layer",
    });
    // Back to the registration's frame when the suggested scale differs.
    if (sug.scale !== r.scale) for (const room of unaided.rooms) {
      const f = sug.scale / r.scale;
      room.polygon = room.polygon.map(([x, y]) => [x / f, y / f]);
      room.holes = room.holes.map((h) => h.map(([x, y]) => [x / f, y / f]));
      room.area /= f * f;
    }
    const unaidedMs = ms(t);
    const cu = compare(written, unaided.rooms);
    const issueCount = (dr) => {
      const c = {};
      for (const i of dr.issues) c[i.type] = (c[i.type] || 0) + 1;
      return c;
    };
    const doorsOf = (dr) => ({
      doors: dr.doors.filter((x) => x.kind === "door").length,
      twoRooms: dr.doors.filter((x) => x.kind === "door" && x.rooms[0] && x.rooms[1]).length,
      toOutside: dr.doors.filter((x) => x.kind === "door" && x.sides.includes("outside")).length,
      openings: dr.doors.filter((x) => x.kind === "opening").length,
    });
    const sa = tally(ca.rows, ca.extra, aided.rooms.length), su = tally(cu.rows, cu.extra, unaided.rooms.length);
    totals.aided = add(totals.aided, sa);
    totals.unaided = add(totals.unaided, su);
    const own = new Set(cad.walls);
    const sugSet = new Set([...sug.walls, ...sug.windows]);
    layouts.push({
      layout: k, sheetMB: mb, readMs,
      aided: { ...sa, ...doorsOf(aided), ms: aidedMs, stages: aided.stats.ms, splits: aided.stats.splits, issues: issueCount(aided) },
      unaided: {
        ...su, ...doorsOf(unaided), ms: unaidedMs, splits: unaided.stats.splits, issues: issueCount(unaided),
        scale: sug.scale, scaleSource: sug.scaleSource, scaleRight: sug.scale === r.scale,
        wallLayersSuggested: sugSet.size, wallLayersOfSchedule: own.size,
        wallLayerOverlap: Math.round(([...sugSet].filter((x) => own.has(x) || own.has(x.split(">").pop())).length / Math.max(1, new Set([...sugSet, ...own]).size)) * 100) / 100,
        doorBlocksSuggested: sug.doorBlocks.length,
      },
    });
    console.log(`${label} ${k.padEnd(4)} ${String(mb).padStart(5)} MB read ${String(readMs).padStart(4)} ms | aided ${sa.found}/${sa.matched}/${sa.near} of ${sa.written} (extra ${sa.extra}) ${aidedMs} ms | unaided ${su.found}/${su.matched}/${su.near} (extra ${su.extra}) ${unaidedMs} ms, scale ${sug.scale}${sug.scale === r.scale ? "" : " WRONG"}`);
    if (VARIANTS) {
      const runs = {
        "no door blocks or swings bridged": { doorPattern: "^$", doorLayers: [] },
        "no splitting by names": { split: false },
        "no growing to centre-lines": { grow: false },
        "no outline closing": { close: 0 },
        "snap 1 cm, extend 10 cm": { snap: 0.01, extend: 0.1 },
        "snap 6 cm, extend 50 cm": { snap: 0.06, extend: 0.5 },
        "walls by layer name, not path": { layerMatch: "layer" },
        "suggested wall layers, schedule's scale": { wallLayers: [...sug.walls, ...sug.windows], doorLayers: sug.doors, doorPattern: sug.doorPattern, layerMatch: "layer" },
      };
      for (const [name, change] of Object.entries(runs)) {
        const dr = draftRooms(d.entities, { ...aidedOpts, ...change });
        const c = compare(written, dr.rooms);
        variants[name] = add(variants[name] || {}, tally(c.rows, c.extra, dr.rooms.length));
      }
    }
  }
  return { label, layouts: layouts.length, sheets: sheets.size, totals, perLayout: layouts, variants: VARIANTS ? variants : undefined };
}

// ------------------------------------------------------------ main

export { bubbles, compare, writtenRooms };
if (MAIN) await main();
async function main() {
  const buildings = readdirSync(path.join(ROOT, "data"), { withFileTypes: true })
    .filter((e) => e.isDirectory() && existsSync(path.join(ROOT, "data", e.name, "lod1_zones.json")) && existsSync(path.join(ROOT, "data", e.name, "dxf")))
    .map((e) => e.name)
    .filter((b) => !only || only.includes(b))
    .sort();
  const results = [];
  buildings.forEach((b, i) => {
    const label = String.fromCharCode(65 + i);
    const r = runBuilding(path.join(ROOT, "data", b), label);
    const t = r.totals;
    console.log(`${label}: aided found ${t.aided.found}, matched ${t.aided.matched}, near ${t.aided.near}, missed ${t.aided.missed} of ${t.aided.written} (extra ${t.aided.extra}); unaided found ${t.unaided.found}, matched ${t.unaided.matched}, near ${t.unaided.near} (extra ${t.unaided.extra})`);
    if (r.variants) for (const [k, v] of Object.entries(r.variants)) console.log(`   ${k.padEnd(42)} found ${v.found}, matched ${v.matched}, near ${v.near}, extra ${v.extra}`);
    results.push(r);
  });
  if (!args.includes("--no-out")) {
    const out = flag("--out", path.join(HERE, `../benchmark/results/${new Date().toISOString().slice(0, 10)}-dxf-lod1.json`));
    await mkdir(path.dirname(out), { recursive: true });
    await writeFile(out, JSON.stringify({
      date: new Date().toISOString().slice(0, 10),
      machine: { cpu: os.cpus()[0]?.model, cores: os.cpus().length, node: process.version },
      note: "Aggregates only: the drawings are a private building. found: IoU >= 0.8 (the lab notebook's 'found'); matched: IoU >= 0.5; near: 0.3 to 0.5; missed: < 0.3; extra: draft rooms matching no written room at 0.3. Times in ms.",
      results,
    }, null, 1));
    console.log(`wrote ${path.relative(process.cwd(), out)}`);
  }
}
