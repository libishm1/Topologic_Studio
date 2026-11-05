/**
 * Identifying IFC elements in the browser: rules (src/core/identify.js),
 * IFCNet's geometry classifier ported to ONNX (src/core/identifyGeometry.js,
 * tools/identify-ifcnet.py), a sentence-embedding model and small Qwen LLMs
 * (Transformers.js), on the same elements, in headless Chrome. Write-up:
 * wiki/research/ifc-identification-benchmark-2026-11-05.md.
 *
 *   node tools/identify-bench.mjs extract --data DIR [--cap 25]
 *       Reads the corpus (benchmark/manifest.json, $BENCHMARK_SAMPLES or
 *       ~/ifc-samples, its more2 folder, and the shipped examples) with
 *       web-ifc and writes DIR/dataset.json (what the app reads for each
 *       element, its class held apart as the truth), DIR/points.bin (2,048
 *       IFCNet-normalised points per element, float32) and
 *       DIR/hard-candidates.json (proxies and generic elements, for the
 *       hand-labelled subset: DIR/hard-labels.json, {id: label}).
 *   node tools/identify-bench.mjs prepare --data DIR
 *       Writes the evaluation set (DIR/eval.json, eval-points.bin) from the
 *       dataset and DIR/hard-labels.json, and DIR/fixture-points.bin for
 *       `python tools/identify-ifcnet.py fixtures` (the browser-against-Python check).
 *   node tools/identify-bench.mjs run --data DIR --models DIR [--onnx DIR]
 *        [--methods rules,ifcnet,embed,llm] [--llm-sample 300] [--out FILE]
 *       Serves the frontend, DIR and the model files, runs every method in
 *       headless Chrome (tools/identify-bench.html) and writes the aggregate
 *       numbers (no per-element text) to benchmark/results/2026-11-05-identify.json.
 *
 * Third-party models and IFCNet's data and weights stay local: DIR, the
 * model folders and the ONNX folder must be outside the repository.
 */
import { createRequire } from "node:module";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, readdir, readFile, stat, writeFile } from "node:fs/promises";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GENERIC_CLASSES, LABELS, boxOf, labelForClass } from "../topologicpy-web-frontend/src/core/identify.js";
import { samplePoints } from "../topologicpy-web-frontend/src/core/identifyGeometry.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..");
const FRONTEND = path.join(ROOT, "topologicpy-web-frontend");
const args = process.argv.slice(2);
const command = args[0];
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const DATA = path.resolve(flag("--data", path.join(os.tmpdir(), "identify-bench")));
const POINTS = 2048;

if (DATA.startsWith(ROOT)) throw new Error("--data must be outside the repository (third-party models stay local).");

// ---------------------------------------------------------------- corpus

/** The corpus, deduplicated: the manifest's models (not stress-only) and more2's buildings. */
async function corpus() {
  const manifest = JSON.parse(readFileSync(path.join(ROOT, "benchmark/manifest.json"), "utf8"));
  const samples = process.env.BENCHMARK_SAMPLES || path.join(os.homedir(), "ifc-samples");
  // Same building twice: keep one (the review's 45 distinct models did the same).
  const skip = new Set(["Ifc4_Revit_ARC_FireRatingAdded", "bsi-ifc4x3-Building-Architecture"]);
  const out = [];
  for (const m of manifest.models) {
    if (m.stressOnly || skip.has(m.name)) continue;
    const file = m.root === "examples" ? path.join(FRONTEND, "public/examples", m.path) : path.join(samples, m.path);
    if (existsSync(file)) out.push({ name: m.name, file });
  }
  const more2 = path.join(samples, "more2");
  if (existsSync(more2)) {
    for (const dir of await readdir(more2)) {
      const full = path.join(more2, dir);
      if (!(await stat(full)).isDirectory()) continue;
      for (const f of await readdir(full)) {
        if (!f.toLowerCase().endsWith(".ifc")) continue;
        const file = path.join(full, f);
        if ((await stat(file)).size >= 50e3) out.push({ name: `${dir}/${path.basename(f, ".ifc")}`, file });
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------- extract

const NOT_ELEMENTS = new Set([
  "IFCSITE", "IFCBUILDING", "IFCBUILDINGSTOREY", "IFCOPENINGELEMENT", "IFCOPENINGSTANDARDCASE", "IFCVIRTUALELEMENT",
  "IFCANNOTATION", "IFCDISTRIBUTIONPORT", "IFCPORT", "IFCGRID", "IFCVOIDINGFEATURE", "IFCPROJECTIONELEMENT",
  "IFCSURFACEFEATURE", "IFCFACILITY", "IFCFACILITYPART", "IFCBRIDGE", "IFCROAD", "IFCRAILWAY", "IFCMARINEFACILITY",
]);
const value = (a) => (a && typeof a === "object" && "value" in a ? a.value : a);

/** A small seeded shuffle, so the stratified sample is the same every run. */
function shuffled(list, seed) {
  let a = seed >>> 0;
  const r = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), a | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = list.slice();
  for (let i = out.length - 1; i > 0; i -= 1) {
    const j = Math.floor(r() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

const median = (xs) => {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

async function extractModel(api, WebIFC, entry, cap) {
  const model = api.OpenModel(new Uint8Array(readFileSync(entry.file)), { COORDINATE_TO_ORIGIN: true });
  try {
    // Elements and rooms, with what the app reads.
    const lines = new Map();
    for (const t of api.GetAllTypesOfModel(model)) {
      const name = String(t.typeName).toUpperCase();
      if (NOT_ELEMENTS.has(name) || !api.IsIfcElement(t.typeID)) continue;
      const ids = api.GetLineIDsWithType(model, t.typeID);
      for (let i = 0; i < ids.size(); i += 1) lines.set(ids.get(i), name);
    }
    // Containment: element -> spatial structure; aggregates: part -> whole.
    const container = new Map();
    const parent = new Map();
    const rel = (typeName, fn) => {
      const ids = api.GetLineIDsWithType(model, WebIFC[typeName]);
      for (let i = 0; i < ids.size(); i += 1) fn(api.GetLine(model, ids.get(i)));
    };
    rel("IFCRELCONTAINEDINSPATIALSTRUCTURE", (r) => {
      for (const e of r.RelatedElements || []) container.set(value(e), value(r.RelatingStructure));
    });
    rel("IFCRELAGGREGATES", (r) => {
      for (const e of r.RelatedObjects || []) parent.set(value(e), value(r.RelatingObject));
    });
    const typeOf = new Map();
    rel("IFCRELDEFINESBYTYPE", (r) => {
      for (const e of r.RelatedObjects || []) typeOf.set(value(e), value(r.RelatingType));
    });
    const storeyIds = new Set();
    {
      const ids = api.GetLineIDsWithType(model, WebIFC.IFCBUILDINGSTOREY);
      for (let i = 0; i < ids.size(); i += 1) storeyIds.add(ids.get(i));
    }
    const storeyOf = (id) => {
      let cur = id;
      for (let hop = 0; hop < 8 && cur != null; hop += 1) {
        if (storeyIds.has(cur)) return cur;
        cur = container.get(cur) ?? parent.get(cur);
      }
      return null;
    };
    const storeyName = new Map([...storeyIds].map((s) => [s, value(api.GetLine(model, s).Name) || ""]));

    const records = [];
    for (const [id, cls] of lines) {
      const line = api.GetLine(model, id);
      const own = value(line.PredefinedType) || "";
      let typePredefined = "";
      const t = typeOf.get(id);
      if (t != null) {
        try {
          typePredefined = value(api.GetLine(model, t)?.PredefinedType) || "";
        } catch {
          typePredefined = "";
        }
      }
      const truthPredefined = own && !/^(NOTDEFINED|USERDEFINED)$/.test(own) ? own : typePredefined || own;
      records.push({
        expressId: id,
        cls,
        truthPredefined,
        label: labelForClass(cls, truthPredefined),
        name: value(line.Name) || "",
        objectType: value(line.ObjectType) || "",
        longName: value(line.LongName) || "",
        // As the app reads it (src/viewer/categories.js elementRecords).
        predefined: own || value(line.OperationType) || "",
        storeyId: storeyOf(id),
      });
    }

    // The sample: up to `cap` per label per model (twice that drawn, trimmed
    // after geometry), and every generic element as a hard-subset candidate.
    const byLabel = new Map();
    for (const r of records) {
      if (!r.label) continue;
      if (!byLabel.has(r.label)) byLabel.set(r.label, []);
      byLabel.get(r.label).push(r);
    }
    const chosen = new Set();
    for (const [label, list] of byLabel) for (const r of shuffled(list, label.length * 7919 + list.length).slice(0, cap * 2)) chosen.add(r.expressId);
    for (const r of records) if (GENERIC_CLASSES.has(r.cls)) chosen.add(r.expressId);

    // Geometry: the chosen, and every door, wall and room (floor levels, rooms).
    const context = records.filter((r) => ["door", "wall", "space"].includes(r.label)).map((r) => r.expressId);
    const want = [...new Set([...chosen, ...context])];
    const meshesById = new Map();
    api.StreamMeshes(model, want, (mesh) => {
      const list = [];
      const placed = mesh.geometries;
      for (let g = 0; g < placed.size(); g += 1) {
        const pg = placed.get(g);
        const geom = api.GetGeometry(model, pg.geometryExpressID);
        const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
        const index = api.GetIndexArray(geom.GetIndexData(), geom.GetIndexDataSize());
        const m = pg.flatTransformation;
        const n = verts.length / 6;
        const positions = new Float32Array(n * 3);
        for (let v = 0; v < n; v += 1) {
          const x = verts[v * 6], y = verts[v * 6 + 1], z = verts[v * 6 + 2];
          positions[v * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
          positions[v * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
          positions[v * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
        }
        list.push({ positions, indices: Uint32Array.from(index) });
        geom.delete?.();
      }
      if (list.length) meshesById.set(mesh.expressID, (meshesById.get(mesh.expressID) || []).concat(list));
    });
    const boxes = new Map();
    for (const [id, meshes] of meshesById) {
      const b = boxOf(meshes);
      if (b && Number.isFinite(b.dims[0])) boxes.set(id, b);
    }

    // Floor level per storey: the median bottom of its doors, else of its walls.
    const floor = new Map();
    for (const s of storeyIds) {
      const doors = records.filter((r) => r.storeyId === s && r.label === "door" && boxes.has(r.expressId)).map((r) => boxes.get(r.expressId).bottom);
      const walls = records.filter((r) => r.storeyId === s && r.label === "wall" && boxes.has(r.expressId)).map((r) => boxes.get(r.expressId).bottom);
      const level = median(doors) ?? median(walls);
      if (level != null) floor.set(s, level);
    }
    const rooms = records
      .filter((r) => r.label === "space" && boxes.has(r.expressId))
      .map((r) => {
        let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
        for (const m of meshesById.get(r.expressId)) {
          for (let i = 0; i < m.positions.length; i += 3) {
            for (let c = 0; c < 3; c += 1) {
              lo[c] = Math.min(lo[c], m.positions[i + c]);
              hi[c] = Math.max(hi[c], m.positions[i + c]);
            }
          }
        }
        const volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
        return { id: r.expressId, name: r.longName || r.name, lo, hi, volume };
      });
    const roomOf = (centre) => {
      let best = null;
      for (const room of rooms) {
        const inside = centre[0] >= room.lo[0] && centre[0] <= room.hi[0] && centre[2] >= room.lo[2] && centre[2] <= room.hi[2] && centre[1] >= room.lo[1] - 0.5 && centre[1] <= room.hi[1];
        if (inside && (!best || room.volume < best.volume)) best = room;
      }
      return best;
    };

    const out = [];
    const perLabel = new Map();
    for (const r of shuffled(records.filter((x) => chosen.has(x.expressId)), 17)) {
      const box = boxes.get(r.expressId);
      if (!box) continue;
      if (r.label && !GENERIC_CLASSES.has(r.cls)) {
        const n = perLabel.get(r.label) || 0;
        if (n >= cap) continue;
        perLabel.set(r.label, n + 1);
      }
      const points = samplePoints(meshesById.get(r.expressId), { points: POINTS, seed: r.expressId, frame: "viewer" });
      if (!points) continue;
      const level = r.storeyId != null ? floor.get(r.storeyId) : undefined;
      const room = r.label === "space" ? null : roomOf(box.centre);
      out.push({
        model: entry.name,
        expressId: r.expressId,
        cls: r.cls,
        truthPredefined: r.truthPredefined,
        label: GENERIC_CLASSES.has(r.cls) ? null : r.label,
        name: r.name,
        objectType: r.objectType,
        longName: r.longName,
        predefined: r.predefined,
        storey: r.storeyId != null ? storeyName.get(r.storeyId) : "",
        room: room ? room.name : "",
        dims: box.dims.map((v) => Math.round(v * 1000) / 1000),
        above: level != null ? Math.round((box.bottom - level) * 1000) / 1000 : null,
        triangles: meshesById.get(r.expressId).reduce((s, m) => s + m.indices.length / 3, 0),
        points,
      });
    }
    return { records: out, elements: records.length };
  } finally {
    api.CloseModel(model);
  }
}

async function extractAll() {
  const cap = Number(flag("--cap", 25));
  const require = createRequire(import.meta.url);
  const WebIFC = require(path.join(FRONTEND, "node_modules/web-ifc/web-ifc-api-node.js"));
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(FRONTEND, "node_modules/web-ifc/"), true);
  await api.Init();
  await mkdir(DATA, { recursive: true });
  const models = await corpus();
  const all = [];
  const perModel = [];
  for (const entry of models) {
    const t0 = performance.now();
    try {
      const { records, elements } = await extractModel(api, WebIFC, entry, cap);
      all.push(...records);
      perModel.push({ model: entry.name, elements, sampled: records.length, generic: records.filter((r) => !r.label).length, ms: Math.round(performance.now() - t0) });
      console.log(`${entry.name}: ${elements} elements, ${records.length} sampled (${Math.round(performance.now() - t0)} ms)`);
    } catch (error) {
      console.log(`${entry.name}: failed (${error.message})`);
    }
  }
  // Points apart, as one float32 block [M, 2048, 3].
  const points = new Float32Array(all.length * POINTS * 3);
  all.forEach((r, i) => {
    points.set(r.points, i * POINTS * 3);
    r.index = i;
    r.id = `${r.model}#${r.expressId}`;
    delete r.points;
  });
  await writeFile(path.join(DATA, "points.bin"), Buffer.from(points.buffer));
  await writeFile(path.join(DATA, "dataset.json"), JSON.stringify({ points: POINTS, records: all, perModel }));
  const hard = all.filter((r) => !r.label).map((r) => ({ id: r.id, cls: r.cls, name: r.name, objectType: r.objectType, predefined: r.predefined, dims: r.dims, above: r.above }));
  await writeFile(path.join(DATA, "hard-candidates.json"), JSON.stringify(hard, null, 1));
  const byLabel = Object.fromEntries(LABELS.map((l) => [l, all.filter((r) => r.label === l).length]));
  console.log(`\n${all.length} records from ${perModel.length} models; ${hard.length} generic; by label:`, byLabel);
}

// ---------------------------------------------------------------- run

const EGRESS = ["door", "stair", "wall", "slab"];

/** Accuracy, macro-F1 over the labels present, per-label F1, egress confusion. */
function score(truth, pred) {
  const present = LABELS.filter((l) => truth.includes(l));
  let ok = 0;
  const tp = {}, fp = {}, fn = {};
  const egress = Object.fromEntries(EGRESS.map((t) => [t, Object.fromEntries([...EGRESS, "else"].map((p) => [p, 0]))]));
  truth.forEach((t, i) => {
    const p = pred[i];
    if (p === t) {
      ok += 1;
      tp[t] = (tp[t] || 0) + 1;
    } else {
      fp[p] = (fp[p] || 0) + 1;
      fn[t] = (fn[t] || 0) + 1;
    }
    if (EGRESS.includes(t)) egress[t][EGRESS.includes(p) ? p : "else"] += 1;
  });
  const f1 = {};
  for (const l of present) {
    const t = tp[l] || 0;
    const P = t + (fp[l] || 0) ? t / (t + (fp[l] || 0)) : 0;
    const R = t / (t + (fn[l] || 0));
    f1[l] = P + R ? Math.round(((2 * P * R) / (P + R)) * 1000) / 1000 : 0;
  }
  const macro = Object.values(f1).reduce((a, b) => a + b, 0) / present.length;
  return { n: truth.length, accuracy: Math.round((ok / truth.length) * 1000) / 1000, macroF1: Math.round(macro * 1000) / 1000, f1, egressConfusion: egress };
}

/** The evaluation set: every primary record, the hand-labelled hard subset, and the LLM sample. */
async function prepareEval() {
  const ds = JSON.parse(await readFile(path.join(DATA, "dataset.json"), "utf8"));
  const hard = JSON.parse(await readFile(path.join(DATA, "hard-labels.json"), "utf8"));
  const primary = ds.records.filter((r) => r.label);
  const hardRecs = ds.records.filter((r) => hard[r.id]);
  const recs = [...primary.map((r) => ({ r, subset: "primary", truth: r.label })), ...hardRecs.map((r) => ({ r, subset: "hard", truth: hard[r.id] }))];
  const buf = await readFile(path.join(DATA, "points.bin"));
  const all = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const pts = new Float32Array(recs.length * POINTS * 3);
  const records = recs.map(({ r, subset, truth }, i) => {
    pts.set(all.subarray(r.index * POINTS * 3, (r.index + 1) * POINTS * 3), i * POINTS * 3);
    return { i, p: i, subset, truth, model: r.model, cls: r.cls, name: r.name, objectType: r.objectType, longName: r.longName, predefined: r.predefined, storey: r.storey, room: r.room, dims: r.dims, above: r.above };
  });
  // The LLM sample: up to 20 per label from the primary set (seeded), and the whole hard subset.
  const per = new Map();
  for (const rec of shuffled(records.filter((x) => x.subset === "primary"), 4242)) {
    const n = per.get(rec.truth) || 0;
    if (n < 20) {
      per.set(rec.truth, n + 1);
      rec.llm = true;
    }
  }
  for (const rec of records) if (rec.subset === "hard") rec.llm = true;
  await writeFile(path.join(DATA, "eval.json"), JSON.stringify({ records }));
  await writeFile(path.join(DATA, "eval-points.bin"), Buffer.from(pts.buffer));
  // Fixture clouds for the browser-against-Python check: the first 16 records.
  await writeFile(path.join(DATA, "fixture-points.bin"), Buffer.from(pts.slice(0, 16 * POINTS * 3).buffer));
  return { records, perModel: ds.perModel };
}

const TYPES = { ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json", ".html": "text/html", ".wasm": "application/wasm", ".onnx": "application/octet-stream", ".bin": "application/octet-stream", ".txt": "text/plain" };

/** A static server; paths under /isolated/ get COOP and COEP (cross-origin isolation, for WASM threads). */
function serve(routes) {
  const server = http.createServer(async (req, res) => {
    const url = new URL(req.url, "http://x");
    let pathname = decodeURIComponent(url.pathname);
    const isolated = pathname.startsWith("/isolated");
    const headers = { "cache-control": "no-store" };
    if (isolated) Object.assign(headers, { "cross-origin-opener-policy": "same-origin", "cross-origin-embedder-policy": "require-corp" });
    if (pathname === "/" || pathname === "/isolated" || pathname === "/isolated/") {
      res.writeHead(200, { ...headers, "content-type": "text/html" });
      res.end(await readFile(path.join(HERE, "identify-bench.html")));
      return;
    }
    if (isolated) pathname = pathname.slice("/isolated".length);
    for (const [prefix, dir] of Object.entries(routes)) {
      if (!pathname.startsWith(prefix)) continue;
      const file = path.resolve(dir, "." + pathname.slice(prefix.length - 1));
      if (!file.startsWith(path.resolve(dir))) break;
      try {
        const body = await readFile(file);
        res.writeHead(200, { ...headers, "content-type": TYPES[path.extname(file)] || "application/octet-stream", "cross-origin-resource-policy": "same-origin" });
        res.end(body);
      } catch {
        res.writeHead(404, headers).end();
      }
      return;
    }
    res.writeHead(404, headers).end();
  });
  return new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server)));
}

/** Working set of the browser's process tree (Windows), for peak memory. */
async function treeBytes(rootPid) {
  const { execFile } = await import("node:child_process");
  const query = "Get-CimInstance Win32_Process -Filter \"name='chrome.exe'\" | Select-Object ProcessId,ParentProcessId,WorkingSetSize | ConvertTo-Json -Compress";
  const out = await new Promise((resolve) => execFile("powershell", ["-NoProfile", "-Command", query], { maxBuffer: 1 << 24, timeout: 60000, killSignal: "SIGKILL" }, (_e, stdout) => resolve(stdout || "[]")));
  let list = [];
  try {
    list = JSON.parse(out);
  } catch {
    return null;
  }
  if (!Array.isArray(list)) list = [list];
  const kids = new Map();
  for (const p of list) {
    if (!kids.has(p.ParentProcessId)) kids.set(p.ParentProcessId, []);
    kids.get(p.ParentProcessId).push(p);
  }
  let total = 0;
  const stack = [rootPid];
  const seen = new Set();
  while (stack.length) {
    const pid = stack.pop();
    if (seen.has(pid)) continue;
    seen.add(pid);
    const self = list.find((p) => p.ProcessId === pid);
    if (self) total += Number(self.WorkingSetSize) || 0;
    for (const k of kids.get(pid) || []) stack.push(k.ProcessId);
  }
  return total;
}

const IFCNET_VARIANT_ORDER = (v) => (v.graph === "naive" ? 0 : v.name.startsWith("fused") ? 1 : 2);

async function run() {
  const { chromium } = await import("playwright-core");
  const MODELS = path.resolve(flag("--models", path.join(os.homedir(), "ifcnet-src/hf")));
  const ONNX = path.resolve(flag("--onnx", path.join(os.homedir(), "ifcnet-src/onnx")));
  const ORT = path.resolve(flag("--ort", path.join(os.homedir(), "ifcnet-src/ort-1.30.0")));
  const V1 = flag("--rules-v1", null);
  const methods = new Set(flag("--methods", "rules,ifcnet,embed,llm").split(","));
  const llmLimit = Number(flag("--llm-sample", Infinity));
  const CHROME = flag("--chrome", "C:/Program Files/Google/Chrome/Application/chrome.exe");
  const OUT = path.resolve(flag("--out", path.join(ROOT, "benchmark/results/2026-11-05-identify.json")));
  const SHIP = flag("--ship", "pruned-fp32-n2048-b1");
  const only = flag("--only", null);
  for (const dir of [MODELS, ONNX, ORT]) if (dir.startsWith(ROOT)) throw new Error(`${dir} must be outside the repository.`);

  const { records } = await prepareEval();
  const ds = JSON.parse(await readFile(path.join(DATA, "dataset.json"), "utf8"));
  const fixtures = existsSync(path.join(ONNX, "fixtures.json")) ? JSON.parse(await readFile(path.join(ONNX, "fixtures.json"), "utf8")) : {};
  const routes = { "/src/": path.join(FRONTEND, "src"), "/ort/": ORT, "/onnx/": ONNX, "/models/": MODELS, "/data/": DATA };
  if (V1) routes["/v1/"] = path.dirname(path.resolve(V1));
  const server = await serve(routes);
  const origin = "http://127.0.0.1:" + server.address().port;
  const browserServer = await chromium.launchServer({ executablePath: CHROME, headless: true, args: ["--enable-unsafe-webgpu"] });
  const pid = browserServer.process().pid;
  const browser = await chromium.connect(browserServer.wsEndpoint());
  const prior = existsSync(OUT) ? JSON.parse(await readFile(OUT, "utf8")) : {};
  const result = {
    ...prior,
    date: "2026-11-05",
    machine: {
      cpu: os.cpus()[0]?.model,
      logicalCores: os.cpus().length,
      memoryGB: Math.round(os.totalmem() / 2 ** 30),
      platform: `${os.platform()} ${os.release()}`,
      chrome: browser.version(),
      note: "Another user's six-process CPU job ran on this machine throughout; timings are medians and carry that load.",
    },
  };
  result.methods ||= {};
  // IFCNet against its PyTorch reference (tools/identify-ifcnet.py verify): aggregate numbers only.
  if (existsSync(path.join(ONNX, "verify.json"))) {
    const v = JSON.parse(await readFile(path.join(ONNX, "verify.json"), "utf8"));
    result.ifcnetVerify = { note: v.note, perClass: v.perClass, reference: v.sets, variants: v.variants };
  }
  const save = async () => {
    await mkdir(path.dirname(OUT), { recursive: true });
    await writeFile(OUT, JSON.stringify(result, null, 2));
  };

  async function page(isolated = false) {
    const ctx = await browser.newContext();
    const pg = await ctx.newPage();
    pg.setDefaultTimeout(0);
    pg.on("console", (m) => {
      if (m.type() === "error" && !m.text().includes("404")) console.log("  [page error]", m.text().slice(0, 300));
      else if (process.env.BENCH_DEBUG) console.log("  [page]", m.text().slice(0, 300));
    });
    pg.on("pageerror", (e) => console.log("  [pageerror]", e.message.slice(0, 300)));
    await pg.goto(origin + (isolated ? "/isolated/" : "/"));
    await pg.waitForFunction(() => window.bench?.ready);
    return { pg, ctx };
  }
  /** Run fn in a fresh page, sampling the browser's memory while it runs. */
  async function measured(label, fn, isolated = false) {
    const { pg, ctx } = await page(isolated);
    await pg.evaluate(() => window.bench.data());
    const base = await treeBytes(pid);
    let peak = base || 0;
    let sampling = true;
    const sampler = (async () => {
      while (sampling) {
        const b = await treeBytes(pid);
        if (b) peak = Math.max(peak, b);
        await new Promise((r) => setTimeout(r, 1500));
      }
    })();
    const t0 = performance.now();
    let out;
    try {
      // A stage that hangs (a runtime waiting on a worker that never starts) must not stop the run.
      const limit = Number(flag("--stage-timeout", 1800)) * 1000;
      out = await Promise.race([fn(pg), new Promise((_, reject) => setTimeout(() => reject(new Error(`timed out after ${limit / 1000} s`)), limit))]);
    } catch (error) {
      out = { error: String(error?.message || error).slice(0, 500) };
    }
    sampling = false;
    await sampler;
    await ctx.close();
    const mem = { baselineMB: base ? Math.round(base / 2 ** 20) : null, peakMB: Math.round(peak / 2 ** 20), addedMB: base ? Math.round((peak - base) / 2 ** 20) : null };
    console.log(`${label}: ${Math.round((performance.now() - t0) / 1000)} s, peak +${mem.addedMB} MB${out?.error ? " ERROR " + out.error : ""}${out?.medianMs ? ` median ${out.medianMs.toFixed(1)} ms` : ""}`);
    return { ...out, memory: mem, wallSeconds: (performance.now() - t0) / 1000 };
  }

  const truthOf = (subset, sample = false) => records.filter((r) => r.subset === subset && (!sample || r.llm));
  const evaluate = (preds, keep = null) => {
    const take = (subset, sample) => {
      const rs = truthOf(subset, sample).filter((r) => (!keep || keep.has(r.i)) && preds[r.i]);
      return rs.length ? score(rs.map((r) => r.truth), rs.map((r) => preds[r.i].label)) : null;
    };
    return { primary: take("primary", false), primarySample: take("primary", true), hard: take("hard", false) };
  };
  const elementsPerModel = median(ds.perModel.map((m) => m.elements));
  const genericPerModel = median(ds.perModel.map((m) => m.generic));
  result.dataset = {
    models: ds.perModel.length,
    primary: truthOf("primary").length,
    hard: truthOf("hard").length,
    hardLabelsBy: "hand, from the name only, by the benchmark's author (Claude); 20 of 179 picked names were too vague to label and are left out",
    llmSample: { primary: truthOf("primary", true).length, hard: truthOf("hard", true).length },
    medianElementsPerModel: elementsPerModel,
    medianGenericSampledPerModel: genericPerModel,
    primaryByLabel: Object.fromEntries(LABELS.map((l) => [l, truthOf("primary").filter((r) => r.truth === l).length])),
    hardByLabel: Object.fromEntries(LABELS.map((l) => [l, truthOf("hard").filter((r) => r.truth === l).length])),
    primaryByClass: truthOf("primary").reduce((m, r) => ((m[r.cls] = (m[r.cls] || 0) + 1), m), {}),
    hardByClass: truthOf("hard").reduce((m, r) => ((m[r.cls] = (m[r.cls] || 0) + 1), m), {}),
  };
  const stash = async (name, preds) => writeFile(path.join(DATA, `preds-${name}.json`), JSON.stringify(preds));

  {
    const { pg, ctx } = await page();
    result.gpu = await pg.evaluate(() => window.bench.gpuInfo());
    await ctx.close();
    console.log("GPU", result.gpu);
  }

  if (methods.has("llmsanity")) {
    for (const [dtype, device] of flag("--sanity", "q4f16:webgpu,q4:wasm").split(",").map((x) => x.split(":"))) {
      const { pg, ctx } = await page();
      console.log(dtype, device, JSON.stringify(await pg.evaluate((o) => window.bench.llmSanity(o), { model: flag("--debug-model", "onnx-community/Qwen2.5-0.5B-Instruct"), dtype, device })));
      await ctx.close();
    }
  }

  if (methods.has("llmdebug")) {
    const { pg, ctx } = await page();
    console.log(JSON.stringify(await pg.evaluate((o) => window.bench.llmDebug(o), { model: flag("--debug-model", "onnx-community/Qwen2.5-0.5B-Instruct"), dtype: flag("--debug-dtype", "q4f16") }), null, 1));
    await ctx.close();
  }

  if (methods.has("rules")) {
    for (const [name, module] of [["rules", "/src/core/identify.js"], ...(V1 ? [["rules-v1", "/v1/" + path.basename(V1)]] : [])]) {
      const out = await measured(name, (pg) => pg.evaluate((m) => window.bench.rules({ module: m }), module));
      await stash(name, out.predictions);
      result.methods[name] = {
        download: { modelBytes: 0 },
        msPerElement: out.msPerElement,
        msPerModel: out.msPerElement * elementsPerModel,
        worksWithoutWebGPU: true,
        memory: out.memory,
        scores: evaluate(out.predictions),
        decidedBy: out.predictions.reduce((m, p) => ((m[p.by] = (m[p.by] || 0) + 1), m), {}),
      };
      await save();
    }
  }

  if (methods.has("ifcnet")) {
    const manifest = JSON.parse(await readFile(path.join(ONNX, "manifest.json"), "utf8"));
    // Batches of 8 only at 2,048 points (the accurate size), to keep the sweep bounded.
    // Scaled-k variants in fp32 only (they lose at every precision), and batches of 8 only at 2,048 points in fp32 and w8.
    const variants = manifest.variants.filter((v) => !v.name.includes("-raw-") && (v.batch === 1 ? !v.name.includes("k") || v.name.includes("fp32") : v.points === 2048 && /fp32|w8/.test(v.name)) && (!only || v.name.includes(only))).sort((a, b) => IFCNET_VARIANT_ORDER(a) - IFCNET_VARIANT_ORDER(b));
    const sweep = [];
    if (!methods.has("noSweep")) {
      for (const v of variants) {
        const opts = { name: v.name, points: v.points, batch: v.batch, ep: "wasm", threads: 1, runs: v.graph === "naive" ? 5 : 20, fixtures: v.batch === 1 && v.graph !== "naive" ? fixtures[v.name] || null : null };
        const out = await measured("wasm " + v.name, (pg) => pg.evaluate((o) => window.bench.ifcnetVariant(o), opts));
        sweep.push({ ...out, precision: v.precision, graph: v.graph });
        result.ifcnetSweep = sweep.map(({ memory, wallSeconds, ...rest }) => ({ ...rest, peakAddedMB: memory?.addedMB }));
        await save();
      }
      for (const name of ["pruned-fp32-n1024-b1", "pruned-fp16-n1024-b1", "pruned-int8-n1024-b1", "pruned-w8-n1024-b1", "pruned-fp32-n2048-b1", "pruned-fp16-n2048-b1", "pruned-fp32-n512-b1", "pruned-fp32-n1024-b8"]) {
        const v = manifest.variants.find((x) => x.name === name);
        if (!v) continue;
        result.ifcnetSweep = sweep.map(({ memory, wallSeconds, ...rest }) => ({ ...rest, peakAddedMB: memory?.addedMB }));
        await save();
        const out = await measured("webgpu " + name, (pg) => pg.evaluate((o) => window.bench.ifcnetVariant(o), { name, points: v.points, batch: v.batch, ep: "webgpu", runs: 20, fixtures: v.batch === 1 ? fixtures[name] || null : null }));
        sweep.push({ ...out, precision: v.precision, graph: v.graph });
      }
      for (const name of ["pruned-fp32-n1024-b1", "pruned-int8-n1024-b1", "pruned-w8-n1024-b1"]) {
        const v = manifest.variants.find((x) => x.name === name);
        result.ifcnetSweep = sweep.map(({ memory, wallSeconds, ...rest }) => ({ ...rest, peakAddedMB: memory?.addedMB }));
        await save();
        const out = await measured("wasm 4 threads " + name, (pg) => pg.evaluate((o) => window.bench.ifcnetVariant(o), { name, points: v.points, batch: 1, ep: "wasm", threads: 4, runs: 20 }), true);
        sweep.push({ ...out, precision: v.precision, graph: v.graph });
      }
      result.ifcnetSweep = sweep.map(({ memory, wallSeconds, ...rest }) => ({ ...rest, peakAddedMB: memory?.addedMB }));
      await save();
    }

    // The shipped variant through the app's own module and worker (onnxruntime-web from jsDelivr).
    const ship = manifest.variants.find((v) => v.name === SHIP);
    const app = await measured("app worker " + SHIP, (pg) => pg.evaluate((o) => window.bench.ifcnetApp(o), { modelUrl: "/onnx/" + SHIP + ".onnx", points: ship.points, backend: "wasm", limit: Number(flag("--app-limit", Infinity)) }));
    if (!app.error) {
      await stash(`ifcnet-${SHIP}`, app.predictions);
      const G = await import("../topologicpy-web-frontend/src/core/identifyGeometry.js");
      const covered = new Set(Object.values(G.IFCNET_LABELS));
      const keep = new Set(records.filter((r) => covered.has(r.truth)).map((r) => r.i));
      const rules = existsSync(path.join(DATA, "preds-rules.json")) ? JSON.parse(await readFile(path.join(DATA, "preds-rules.json"), "utf8")) : null;
      const combos = {};
      if (rules) {
        for (const th of [0.5, 0.6, 0.8, 0.9]) {
          const mixed = records.map((r, i) => G.combine(rules[i], app.predictions[i], th));
          await stash(`rules+ifcnet-${SHIP}@${th}`, mixed);
          combos[`rules+ifcnet-${SHIP}@${th}`] = { scores: evaluate(mixed), shapeUsed: { primary: mixed.filter((m, i) => m.by === "shape" && records[i].subset === "primary").length, hard: mixed.filter((m, i) => m.by === "shape" && records[i].subset === "hard").length } };
        }
      }
      result.methods[`ifcnet-${SHIP}`] = {
        variant: SHIP,
        download: { modelBytes: ship.bytes, runtimeBytesBrotli: 3073897 + 16610, runtimeBytesRaw: 14239897 + 50126 },
        loadMs: app.loadMs,
        msPerElement: app.msPerElement,
        msPerModelAllElements: app.msPerElement * elementsPerModel,
        msPerModelGenericOnly: app.msPerElement * genericPerModel,
        worstMainThreadGapMs: app.worstMainThreadGapMs,
        worksWithoutWebGPU: true,
        memory: app.memory,
        workerInfo: app.info,
        scores: evaluate(app.predictions),
        scoresCoveredLabelsOnly: evaluate(app.predictions, keep),
        coveredLabels: [...covered],
      };
      result.combined = { ...(result.combined || {}), ...combos };
    } else {
      result.methods[`ifcnet-${SHIP}`] = { error: app.error };
    }
    {
      const { pg, ctx } = await page();
      result.ifcnetFailures = await pg.evaluate(() => window.bench.ifcnetFailures());
      await ctx.close();
    }
    await save();
  }

  // Threads only (cross-origin isolated page), for the variants named by --threads-variants.
  if (methods.has("threads")) {
    const manifest = JSON.parse(await readFile(path.join(ONNX, "manifest.json"), "utf8"));
    result.ifcnetThreads ||= [];
    for (const name of flag("--threads-variants", "pruned-fp32-n2048-b1").split(",")) {
      const v = manifest.variants.find((x) => x.name === name);
      for (const threads of [2, 4, 8]) {
        const out = await measured(`wasm ${threads} threads ${name}`, (pg) => pg.evaluate((o) => window.bench.ifcnetVariant(o), { name, points: v.points, batch: 1, ep: "wasm", threads, runs: 20, fixtures: fixtures[name] || null }), true);
        const { memory, wallSeconds, ...rest } = out;
        result.ifcnetThreads = result.ifcnetThreads.filter((t) => !(t.name === name && t.threads === threads));
        result.ifcnetThreads.push({ ...rest, peakAddedMB: memory?.addedMB });
        await save();
      }
    }
  }

  // Recombine stashed rule and shape answers (after the rules change, without rerunning the shapes).
  if (methods.has("combine")) {
    const G = await import("../topologicpy-web-frontend/src/core/identifyGeometry.js");
    const rules = JSON.parse(await readFile(path.join(DATA, "preds-rules.json"), "utf8"));
    const shape = JSON.parse(await readFile(path.join(DATA, `preds-ifcnet-${SHIP}.json`), "utf8"));
    result.combined ||= {};
    for (const th of [0.5, 0.6, 0.8, 0.9]) {
      const mixed = records.map((r, i) => G.combine(rules[i], shape[i], th));
      await stash(`rules+ifcnet-${SHIP}@${th}`, mixed);
      result.combined[`rules+ifcnet-${SHIP}@${th}`] = { scores: evaluate(mixed), shapeUsed: { primary: mixed.filter((m, i) => m.by === "shape" && records[i].subset === "primary").length, hard: mixed.filter((m, i) => m.by === "shape" && records[i].subset === "hard").length } };
    }
    await save();
  }

  if (methods.has("embed")) {
    const list = [
      ["minilm-fp32-wasm", { model: "Xenova/all-MiniLM-L6-v2", dtype: "fp32", device: "wasm" }, 90387606],
      ["minilm-int8-wasm", { model: "Xenova/all-MiniLM-L6-v2", dtype: "q8", device: "wasm" }, 22972370],
      ["minilm-fp32-webgpu", { model: "Xenova/all-MiniLM-L6-v2", dtype: "fp32", device: "webgpu" }, 90387606],
      ["e5small-int8-wasm", { model: "Xenova/multilingual-e5-small", dtype: "q8", device: "wasm", prefix: "query: ", labelPrefix: "query: " }, 118308185],
    ].filter(([n]) => !only || n.includes(only));
    for (const [name, opts, bytes] of list) {
      const out = await measured(name, (pg) => pg.evaluate((o) => window.bench.embed(o), opts));
      if (out.error) {
        result.methods[name] = { error: out.error };
        continue;
      }
      await stash(name, out.predictions);
      result.methods[name] = { download: { modelBytes: bytes }, loadMs: out.loadMs, msPerElement: out.msPerElement, msPerModel: out.msPerElement * elementsPerModel, worksWithoutWebGPU: opts.device === "wasm", memory: out.memory, scores: evaluate(out.predictions) };
      await save();
    }
  }

  if (methods.has("llm") || methods.has("hybrid")) {
    const sample = records.filter((r) => r.llm);
    const full = await readFile(path.join(DATA, "eval.json"));
    // Qwen2.5-0.5B's q4f16 build (fp16 activations) gave nonsense on this machine's Intel UHD 770
    // under WebGPU ("What is the capital of France?" -> "Hello, welcome to the discussion..."; its q4
    // build and WASM answer "Paris"); Qwen3-0.6B's q4f16 answered sensibly. Qwen2.5-1.5B's q4f16
    // aborted on WebGPU and WASM alike. Check a build with --methods llmsanity before trusting it.
    const list = [
      ["qwen2.5-0.5b-q4f16-webgpu", { model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q4f16", device: "webgpu" }, 483003582],
      ["qwen2.5-0.5b-q4-webgpu", { model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q4", device: "webgpu", limit: 240 }, 786156820],
      ["qwen2.5-0.5b-int8-wasm", { model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "int8", device: "wasm", limit: 40 }, 512096557],
      ["qwen2.5-0.5b-q4-wasm", { model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q4", device: "wasm", limit: 40 }, 786156820],
      ["qwen3-0.6b-q4f16-webgpu", { model: "onnx-community/Qwen3-0.6B-ONNX", dtype: "q4f16", device: "webgpu", qwen3: true, limit: 240 }, 569789750],
      ["qwen2.5-1.5b-q4-webgpu", { model: "onnx-community/Qwen2.5-1.5B-Instruct", dtype: "q4", device: "webgpu", limit: 120 }, 1787600000],
      ["hybrid-qwen2.5-0.5b-q4-webgpu", { model: "onnx-community/Qwen2.5-0.5B-Instruct", dtype: "q4", device: "webgpu", hint: true, limit: 240 }, 786156820],
      ["hybrid-qwen2.5-1.5b-q4-webgpu", { model: "onnx-community/Qwen2.5-1.5B-Instruct", dtype: "q4", device: "webgpu", hint: true, limit: 120 }, 1787600000],
    ].filter(([n]) => ((n.startsWith("hybrid") ? methods.has("hybrid") : methods.has("llm")) && (!only || n.includes(only))));
    try {
      for (const [name, opts, bytes] of list) {
        const limit = Math.min(opts.limit ?? Infinity, llmLimit);
        // The page runs over every record it is given: give it the sample (spread out when cut short).
        const chosen = Number.isFinite(limit) && limit < sample.length ? Array.from({ length: limit }, (_, k) => sample[Math.floor((k * sample.length) / limit)]) : sample;
        await writeFile(path.join(DATA, "eval.json"), JSON.stringify({ records: chosen.map((r, k) => ({ ...r, i: k })) }));
        const at = new Map(chosen.map((r, k) => [r.i, k]));
        const out = await measured(name, (pg) => pg.evaluate((o) => window.bench.llm(o), opts));
        if (out.error) {
          result.methods[name] = { error: out.error, memory: out.memory };
          await save();
          continue;
        }
        const preds = records.map((r) => (at.has(r.i) ? out.predictions[at.get(r.i)] : null));
        await writeFile(path.join(DATA, `preds-${name}.json`), JSON.stringify(preds));
        const take = (subset) => {
          const rs = chosen.filter((r) => r.subset === subset);
          return rs.length ? score(rs.map((r) => r.truth), rs.map((r) => out.predictions[at.get(r.i)].label)) : null;
        };
        result.methods[name] = {
          download: { modelBytes: bytes },
          loadMs: out.loadMs,
          msPerElement: out.msPerElement,
          msPerModel: out.msPerElement * elementsPerModel,
          promptTokensPerElement: out.promptTokensPerElement,
          passesPerElement: out.passesPerElement,
          prefillTokensPerSecond: out.prefillTokensPerSecond,
          decode: out.decode,
          sampleSize: chosen.length,
          worksWithoutWebGPU: opts.device === "wasm",
          memory: out.memory,
          scores: { primarySample: take("primary"), hard: take("hard") },
        };
        if (out.predictions.every((x) => x.calibrated)) {
          const cal = out.predictions.map((x) => ({ label: x.calibrated }));
          await writeFile(path.join(DATA, `preds-${name}-calibrated.json`), JSON.stringify(records.map((r) => (at.has(r.i) ? cal[at.get(r.i)] : null))));
          const takeCal = (subset) => {
            const rs = chosen.filter((r) => r.subset === subset);
            return rs.length ? score(rs.map((r) => r.truth), rs.map((r) => cal[at.get(r.i)].label)) : null;
          };
          result.methods[`${name}-calibrated`] = { ...result.methods[name], note: "the same run, contextual calibration (label prior of a content-free prompt divided out)", scores: { primarySample: takeCal("primary"), hard: takeCal("hard") } };
        }
        await save();
      }
    } finally {
      await writeFile(path.join(DATA, "eval.json"), full);
    }
  }

  // Every method on the LLM sample, so all are compared on the same elements.
  const onSample = {};
  for (const f of await readdir(DATA)) {
    if (!f.startsWith("preds-") || !f.endsWith(".json")) continue;
    const preds = JSON.parse(await readFile(path.join(DATA, f), "utf8"));
    if (preds.length !== records.length) continue;
    const rs = truthOf("primary", true).filter((r) => preds[r.i]);
    if (rs.length === truthOf("primary", true).length) onSample[f.slice(6, -5)] = score(rs.map((r) => r.truth), rs.map((r) => preds[r.i].label));
  }
  result.onLlmSamplePrimary = { ...(result.onLlmSamplePrimary || {}), ...onSample };
  // Each LLM run against every full-coverage method on exactly its elements (some LLM runs used a sub-sample).
  const all = {};
  for (const f of await readdir(DATA)) {
    if (f.startsWith("preds-") && f.endsWith(".json")) all[f.slice(6, -5)] = JSON.parse(await readFile(path.join(DATA, f), "utf8"));
  }
  const fullCover = Object.entries(all).filter(([, p]) => p.length === records.length && p.every(Boolean));
  result.sameElements = {};
  for (const [name, preds] of Object.entries(all)) {
    if (preds.length !== records.length || preds.every(Boolean)) continue;
    const entry = {};
    for (const subset of ["primary", "hard"]) {
      const rs = records.filter((r) => r.subset === subset && preds[r.i]);
      if (!rs.length) continue;
      entry[subset] = { n: rs.length, [name]: score(rs.map((r) => r.truth), rs.map((r) => preds[r.i].label)) };
      for (const [other, op] of fullCover) entry[subset][other] = score(rs.map((r) => r.truth), rs.map((r) => op[r.i].label));
      for (const k of Object.keys(entry[subset])) if (k !== "n") entry[subset][k] = { accuracy: entry[subset][k].accuracy, macroF1: entry[subset][k].macroF1 };
    }
    result.sameElements[name] = entry;
  }
  await save();
  await browser.close();
  await browserServer.close();
  server.close();
  console.log("wrote", OUT);
  // WMI queries and the browser's pipes can keep Node alive; the work is done.
  process.exit(0);
}

// ---------------------------------------------------------------- main

if (command === "extract") await extractAll();
else if (command === "prepare") await prepareEval();
else if (command === "run") await run();
else console.log("usage: node tools/identify-bench.mjs extract|run --data DIR ... (see the header)");
