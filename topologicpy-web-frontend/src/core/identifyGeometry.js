/**
 * What an element is, from its shape: IFCNet's DGCNN point-cloud classifier
 * (Emunds, Pauen, Richter, Frisch and van Treeck 2021, "IFCNet: A Benchmark
 * Dataset for IFC Entity Classification", EG-ICE 2021, arXiv:2106.09712; code
 * github.com/RWTH-E3D/ifcnet-models, MIT), ported to ONNX
 * (tools/identify-ifcnet.py) and run with onnxruntime-web in a Web Worker
 * (src/workers/identify.worker.js), loaded only when asked for.
 *
 * Use: the name rules (identify.js) answer first. This is for the elements
 * they leave open, IfcBuildingElementProxy and the generic flow and
 * distribution elements whose names say nothing; its answer is a hint
 * ranked below a name, and the name rules keep the sub-types its 20 classes
 * do not separate (a cooktop from a refrigerator are both "IfcFurniture" to
 * it at best). See wiki/research/ifc-identification-benchmark-2026-11-05.md.
 *
 * Weights: the MIT licence covers IFCNet's code; the pretrained weights are
 * published separately (ifcnet.e3d.rwth-aachen.de) with no stated licence and
 * were trained on third-party models, so they are not in this repository and
 * there is no default model URL. `createGeometryIdentifier` needs `modelUrl`
 * (an ONNX file built by tools/identify-ifcnet.py from weights the user
 * downloaded, or a published copy once the authors licence one). Without it,
 * or when the runtime or model cannot load, it fails with a clear message and
 * the caller keeps the rule-based answer.
 *
 * Input, as IFCNet's preprocessing (src/data/make_dgcnn_data.py) makes it:
 * the element's triangles in IFC's frame (Z up; the viewer frame is
 * (x, z, -y) of it, so a viewer point (x, y, z) is (x, -z, y) here),
 * centred on the mean of its distinct vertices and scaled so the farthest
 * vertex is at distance 1, then `points` samples drawn uniformly by area.
 */

export const IFCNET_CLASSES = [
  "IfcAirTerminal", "IfcBeam", "IfcCableCarrierFitting", "IfcCableCarrierSegment", "IfcDoor",
  "IfcDuctFitting", "IfcDuctSegment", "IfcFurniture", "IfcLamp", "IfcOutlet", "IfcPipeFitting",
  "IfcPipeSegment", "IfcPlate", "IfcRailing", "IfcSanitaryTerminal", "IfcSlab", "IfcSpaceHeater",
  "IfcStair", "IfcValve", "IfcWall",
];

/** IFCNet's classes as the labels of identify.js (LABELS). */
export const IFCNET_LABELS = {
  IfcAirTerminal: "mep",
  IfcBeam: "beam",
  IfcCableCarrierFitting: "mep",
  IfcCableCarrierSegment: "mep",
  IfcDoor: "door",
  IfcDuctFitting: "mep",
  IfcDuctSegment: "mep",
  IfcFurniture: "furniture",
  IfcLamp: "lighting",
  IfcOutlet: "mep",
  IfcPipeFitting: "mep",
  IfcPipeSegment: "mep",
  IfcPlate: "other",
  IfcRailing: "railing",
  IfcSanitaryTerminal: "sanitary",
  IfcSlab: "slab",
  IfcSpaceHeater: "heater",
  IfcStair: "stair",
  IfcValve: "mep",
  IfcWall: "wall",
};

/**
 * Points per element. IFCNet was trained on 2,048; on its own test split the
 * balanced accuracy is 0.80 at 2,048, 0.72 at 1,024 and 0.39 at 512 points
 * (fewer neighbours, k = 20 at 1,024, does not recover it), so the input
 * stays at 2,048 and the speed comes from the pruned graph instead.
 */
export const DEFAULT_POINTS = 2048;

/** onnxruntime-web, pinned, from a CDN: never in the app bundle. */
export const ORT_VERSION = "1.30.0";
export const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

/** A small seeded generator (mulberry32), so one element always samples the same. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * IFCNet's input for one element.
 *
 * @param {Array<{positions: Float32Array, indices: Uint32Array}>} meshes  the element's meshes, viewer frame
 * @param {object} opts  points (default DEFAULT_POINTS); seed; frame "viewer" (Y up, default) or "ifc" (Z up)
 * @returns {Float32Array|null} [points * 3], or null when the meshes have no area
 */
export function samplePoints(meshes, { points = DEFAULT_POINTS, seed = 1, frame = "viewer" } = {}) {
  // Triangles in IFC's frame.
  let triangles = 0;
  for (const m of meshes || []) triangles += Math.floor((m.indices?.length || 0) / 3);
  if (!triangles) return null;
  const tri = new Float64Array(triangles * 9);
  const toIfc = (p, v, out, o) => {
    out[o] = p[v];
    out[o + 1] = frame === "viewer" ? -p[v + 2] : p[v + 1];
    out[o + 2] = frame === "viewer" ? p[v + 1] : p[v + 2];
  };
  // The centre is the mean of the referenced vertices as stored, a vertex
  // repeated per face counted each time: IFCNet's OBJ files repeat them so
  // (per-face normals), and pymesh averaged them so. Against IFCNet's own
  // point clouds this matches to a median 0.0003 of the unit sphere; the
  // mean of distinct vertices is 15 times further off.
  let cx = 0, cy = 0, cz = 0, count = 0;
  const xyz = new Float64Array(3);
  let t = 0;
  for (const m of meshes) {
    const p = m.positions;
    const idx = m.indices;
    const used = new Uint8Array(p.length / 3);
    for (let i = 0; i + 2 < idx.length; i += 3) {
      for (let c = 0; c < 3; c += 1) {
        toIfc(p, idx[i + c] * 3, tri, t * 9 + c * 3);
        used[idx[i + c]] = 1;
      }
      t += 1;
    }
    for (let v = 0; v < used.length; v += 1) {
      if (!used[v]) continue;
      toIfc(p, v * 3, xyz, 0);
      cx += xyz[0];
      cy += xyz[1];
      cz += xyz[2];
      count += 1;
    }
  }
  cx /= count;
  cy /= count;
  cz /= count;
  let r2 = 0;
  for (let i = 0; i < tri.length; i += 3) {
    const dx = tri[i] - cx, dy = tri[i + 1] - cy, dz = tri[i + 2] - cz;
    r2 = Math.max(r2, dx * dx + dy * dy + dz * dz);
  }
  const scale = r2 > 0 ? 1 / Math.sqrt(r2) : 1;

  // Cumulative areas, for sampling triangles in proportion to their area.
  const cum = new Float64Array(triangles);
  let total = 0;
  for (let i = 0; i < triangles; i += 1) {
    const o = i * 9;
    const ax = tri[o + 3] - tri[o], ay = tri[o + 4] - tri[o + 1], az = tri[o + 5] - tri[o + 2];
    const bx = tri[o + 6] - tri[o], by = tri[o + 7] - tri[o + 1], bz = tri[o + 8] - tri[o + 2];
    const nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
    total += 0.5 * Math.sqrt(nx * nx + ny * ny + nz * nz);
    cum[i] = total;
  }
  if (!(total > 0)) return null;

  const random = rng(seed);
  const out = new Float32Array(points * 3);
  for (let s = 0; s < points; s += 1) {
    const target = random() * total;
    let lo = 0, hi = triangles - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cum[mid] < target) lo = mid + 1;
      else hi = mid;
    }
    let u = random(), v = random();
    if (u + v > 1) {
      u = 1 - u;
      v = 1 - v;
    }
    const w = 1 - u - v;
    const o = lo * 9;
    for (let c = 0; c < 3; c += 1) {
      const value = tri[o + c] * u + tri[o + 3 + c] * v + tri[o + 6 + c] * w;
      out[s * 3 + c] = (value - (c === 0 ? cx : c === 1 ? cy : cz)) * scale;
    }
  }
  return out;
}

export function softmax(logits) {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  const out = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    out[i] = Math.exp(logits[i] - max);
    sum += out[i];
  }
  for (let i = 0; i < out.length; i += 1) out[i] /= sum;
  return out;
}

/** The classifier's answer for one element, from its 20 logits. */
export function readLogits(logits) {
  const probs = softmax(logits);
  let best = 0;
  for (let i = 1; i < probs.length; i += 1) if (probs[i] > probs[best]) best = i;
  const ifcClass = IFCNET_CLASSES[best];
  return { ifcClass, label: IFCNET_LABELS[ifcClass], confidence: probs[best], probs };
}

/**
 * One answer from the name rules and the shape: a class, a predefined type
 * or a name wins; the shape fills in only when the rules fell back to the
 * bounding box or found nothing, and only if it is confident; the rules keep
 * their sub-type (cooking, refrigeration). On the benchmark no threshold
 * made the pair better than the rules alone; at 0.9 it changes almost nothing.
 *
 * @param {{label, confidence, by, subtype?}} rule  identify.js's answer
 * @param {{label, ifcClass, confidence}|null} shape  readLogits's answer
 * @param {number} minConfidence  softmax probability the shape needs
 */
export function combine(rule, shape, minConfidence = 0.9) {
  if (!shape || (rule?.by !== "box" && rule?.by !== "none")) return rule;
  if (shape.confidence < minConfidence || shape.label === "other") return rule;
  return { label: shape.label, confidence: shape.confidence, by: "shape", ifcClass: shape.ifcClass, subtype: rule?.subtype };
}

/**
 * A lazily started classifier in a worker.
 *
 * @param {object} opts
 *   modelUrl      the ONNX model (required; see the module note on weights)
 *   points        the model's point count (static input shape)
 *   backend       "wasm" (default) or "webgpu" (falls back to "wasm")
 *   ortBase       where onnxruntime-web is loaded from
 *   workerFactory () => Worker, for tests
 * @returns {{ready: Promise<object>, classify: (clouds: Float32Array[]) => Promise<object[]>, terminate: () => void}}
 */
export function createGeometryIdentifier({ modelUrl, points = DEFAULT_POINTS, backend = "wasm", ortBase = ORT_BASE, workerFactory } = {}) {
  if (!modelUrl) {
    const error = new Error("No geometry model is configured (IFCNet's weights are not shipped; see identifyGeometry.js). The name rules' answers stand.");
    return { ready: Promise.reject(error), classify: () => Promise.reject(error), terminate() {} };
  }
  const worker = workerFactory ? workerFactory() : new Worker(new URL("../workers/identify.worker.js", import.meta.url), { type: "module" });
  let next = 1;
  const pending = new Map();
  worker.onmessage = (event) => {
    const { id, ok, result, error } = event.data || {};
    const job = pending.get(id);
    if (!job) return;
    pending.delete(id);
    if (ok) job.resolve(result);
    else job.reject(new Error(error || "The geometry classifier failed."));
  };
  worker.onerror = (event) => {
    const error = new Error(event?.message || "The geometry classifier's worker failed to start.");
    for (const job of pending.values()) job.reject(error);
    pending.clear();
  };
  const call = (type, payload, transfer = []) =>
    new Promise((resolve, reject) => {
      const id = next++;
      pending.set(id, { resolve, reject });
      worker.postMessage({ id, type, payload }, transfer);
    });
  const ready = call("init", { modelUrl, points, backend, ortBase });
  ready.catch(() => {});
  return {
    ready,
    async classify(clouds) {
      await ready;
      const batch = new Float32Array(clouds.length * points * 3);
      clouds.forEach((cloud, i) => batch.set(cloud.subarray(0, points * 3), i * points * 3));
      const { logits, classes } = await call("classify", { batch, count: clouds.length }, [batch.buffer]);
      const out = [];
      for (let i = 0; i < clouds.length; i += 1) out.push(readLogits(logits.subarray(i * classes, (i + 1) * classes)));
      return out;
    },
    terminate() {
      worker.terminate();
      const error = new Error("The geometry classifier was stopped.");
      for (const job of pending.values()) job.reject(error);
      pending.clear();
    },
  };
}
