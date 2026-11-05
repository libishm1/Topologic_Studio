/**
 * Identifying elements (identify.js, identifyGeometry.js): classes, names in
 * the languages the rules read, the bounding-box fallback, IFCNet's input
 * preparation, and the geometry classifier's failure paths. No model is
 * downloaded: the worker is replaced by a stand-in.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { LABELS, boxOf, countLabels, identify, labelForClass, labelFromBox, labelFromName } from "./identify.js";
import { IFCNET_CLASSES, IFCNET_LABELS, combine, createGeometryIdentifier, readLogits, samplePoints } from "./identifyGeometry.js";

test("typed elements take their class; the predefined type refines it", () => {
  assert.equal(labelForClass("IFCWALLSTANDARDCASE"), "wall");
  assert.equal(labelForClass("IfcSlab", "ROOF"), "roof");
  assert.equal(labelForClass("IFCSLAB", "LANDING"), "stair");
  assert.equal(labelForClass("IFCSLAB", "FLOOR"), "slab");
  assert.equal(labelForClass("IFCPIPESEGMENT"), "mep");
  assert.equal(labelForClass("IFCMEMBER"), "other");
  assert.equal(labelForClass("IFCBUILDINGELEMENTPROXY"), null);
  assert.equal(labelForClass("IFCFLOWTERMINAL"), null);
  assert.deepEqual(identify({ type: "IFCDOOR", name: "anything" }), { label: "door", confidence: 1, by: "class" });
});

test("a proxy named as an appliance is an appliance, and the sub-type tells a cooktop from a fridge", () => {
  const hob = identify({ type: "IFCBUILDINGELEMENTPROXY", name: "M_Cook Top-4 Unit:Standard:123" });
  assert.equal(hob.label, "appliance");
  assert.equal(hob.by, "name");
  assert.equal(hob.subtype, "cooking-range");
  const fridge = identify({ name: "M_Refrigerator:600 x 650mm:1" });
  assert.equal(fridge.label, "appliance");
  assert.equal(fridge.subtype, "refrigerator-freezer");
  // A range hood is an appliance but not a cooking range.
  const hood = identify({ name: "Range Hood 900" });
  assert.equal(hood.label, "appliance");
  assert.notEqual(hood.subtype, "cooking-range");
});

test("the specific wins over the generic in a name", () => {
  const cases = {
    "Basic Wall:Interior - Partition (92mm Stud)": "wall",
    "M_Base Cabinet-Double Door & 2 Drawer:600mm": "furniture",
    "Upper Cabinet-Wall:900mm": "furniture",
    "Floor Lamp": "lighting",
    "Floor Drain 100": "mep",
    "Ceiling Light 600x600": "lighting",
    "Skylight 1200": "window",
    "Lighting and Appliance Panelboard": "mep",
    "Stair:Steel | stringer": "other",
    "Curtain Wall Mullion 50x150": "other",
    "M_Single-Flush:0915 x 2134mm": "door",
    "Elevator Door-Center": "door",
    "Supply Diffuser 600x600": "mep",
    "Water Closet - Flush Tank": "sanitary",
    "Panel Radiator 600": "heater",
  };
  for (const [name, label] of Object.entries(cases)) assert.equal(labelFromName(name), label, name);
});

test("names in German, Norwegian, Dutch, French, Spanish, Italian and Arabic", () => {
  const cases = {
    "Innentür 1-flg": "door",
    "Außenwand 24": "wall",
    "Kühlschrank": "appliance",
    "Geschossdecke 20": "slab",
    "Waschtisch": "sanitary",
    "Toalett HC": "sanitary",
    "Kjøleskap 60": "appliance",
    "Yttervegg": "wall",
    "trap naar zolder": "stair",
    "Fenêtre double": "window",
    "Escalera de piedra": "stair",
    "Barandilla de acero": "railing",
    "Solaio 25": "slab",
    "باب خشبي": "door",
  };
  for (const [name, label] of Object.entries(cases)) assert.equal(labelFromName(name), label, name);
  // The Revit prefix "M_" does not hide a word ("_" is a word character to \b).
  assert.equal(labelFromName("M_Bed-Box"), "furniture");
});

test("a LongName marks a room; a predefined type is read before the name", () => {
  assert.equal(identify({ name: "101", longName: "Kitchen" }).label, "space");
  assert.equal(identify({ name: "Generic", predefined: "FLOOR" }).label, "slab");
  assert.equal(identify({ name: "Generic", predefined: "NOTDEFINED", dims: [0.9, 0.1, 2.1], above: 0 }).by, "box");
});

test("the bounding box decides when nothing else does", () => {
  assert.equal(labelFromBox([0.9, 0.1, 2.1], 0), "door");
  assert.equal(labelFromBox([1.2, 0.3, 1.4], 0.9), "window");
  assert.equal(labelFromBox([6, 0.2, 2.8], 0), "wall");
  assert.equal(labelFromBox([8, 5, 0.25], 0), "slab");
  assert.equal(labelFromBox([5, 4, 0.02], 2.6), "covering");
  assert.equal(labelFromBox([0.4, 0.4, 3], 0), "column");
  assert.equal(labelFromBox([0.6, 0.6, 0.1], 2.5), "lighting");
  assert.equal(labelFromBox([5, 4, 2.7], 0), "space");
  assert.equal(labelFromBox(null, 0), null);
});

test("every answer is a label, and counts add up", () => {
  const words = ["", "x", "Wall", "Tür", "M_", "???", "Floor", "lamp", "1234", "Proxy"];
  const records = [];
  for (const a of words) for (const b of words) records.push({ name: a + " " + b, dims: [Math.random() * 6, Math.random(), Math.random() * 4], above: Math.random() * 3 });
  for (const r of records) assert.ok(LABELS.includes(identify(r).label));
  const counts = countLabels(records);
  assert.equal(Object.values(counts).reduce((a, b) => a + b, 0), records.length);
});

/** A box mesh (viewer frame: Y up) of size sx x sy x sz at (x, y, z), 24 vertices as exporters write it. */
function boxMesh(sx, sy, sz, x = 0, y = 0, z = 0, angle = 0) {
  const corners = [];
  for (const [a, b, c] of [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]) {
    const px = (a - 0.5) * sx, pz = (c - 0.5) * sz;
    corners.push([x + px * Math.cos(angle) - pz * Math.sin(angle), y + b * sy, z + px * Math.sin(angle) + pz * Math.cos(angle)]);
  }
  const faces = [[0, 1, 2, 3], [4, 7, 6, 5], [0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]];
  const positions = [];
  const indices = [];
  for (const f of faces) {
    const o = positions.length / 3;
    for (const v of f) positions.push(...corners[v]);
    indices.push(o, o + 1, o + 2, o, o + 2, o + 3);
  }
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

test("boxOf measures a wall along its own axes", () => {
  const b = boxOf([boxMesh(5, 3, 0.2, 10, 1, 4, Math.PI / 6)]);
  assert.ok(Math.abs(b.dims[0] - 5) < 1e-3 && Math.abs(b.dims[1] - 0.2) < 1e-3 && Math.abs(b.dims[2] - 3) < 1e-3, JSON.stringify(b.dims));
  assert.ok(Math.abs(b.bottom - 1) < 1e-6);
});

test("IFCNet's input: Z up, centred, inside the unit sphere, the same for the same seed", () => {
  // A slab, flat in the viewer's Y: flat in IFC's Z.
  const slab = [boxMesh(6, 0.2, 4, 3, 7, -2)];
  const p = samplePoints(slab, { points: 1024, seed: 7 });
  assert.equal(p.length, 1024 * 3);
  let r = 0, zMin = Infinity, zMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (let i = 0; i < p.length; i += 3) {
    r = Math.max(r, Math.hypot(p[i], p[i + 1], p[i + 2]));
    zMin = Math.min(zMin, p[i + 2]);
    zMax = Math.max(zMax, p[i + 2]);
    yMin = Math.min(yMin, p[i + 1]);
    yMax = Math.max(yMax, p[i + 1]);
  }
  assert.ok(r <= 1 + 1e-6 && r > 0.95, `radius ${r}`);
  assert.ok(zMax - zMin < 0.1, "thin in Z");
  assert.ok(yMax - yMin > 1, "wide in Y (the viewer's -Z)");
  assert.deepEqual(samplePoints(slab, { points: 64, seed: 7 }), samplePoints(slab, { points: 64, seed: 7 }));
  assert.equal(samplePoints([{ positions: new Float32Array(9), indices: new Uint32Array([0, 1, 2]) }]), null);
});

test("IFCNet's classes map onto the labels; the shape only fills in where names fail", () => {
  for (const c of IFCNET_CLASSES) assert.ok(LABELS.includes(IFCNET_LABELS[c]), c);
  const logits = new Float32Array(20);
  logits[IFCNET_CLASSES.indexOf("IfcSanitaryTerminal")] = 8;
  const shape = readLogits(logits);
  assert.equal(shape.label, "sanitary");
  assert.ok(shape.confidence > 0.9);
  assert.equal(combine({ label: "furniture", by: "box", confidence: 0.4 }, shape).label, "sanitary");
  assert.equal(combine({ label: "appliance", by: "name", confidence: 0.8 }, shape).label, "appliance");
  assert.equal(combine({ label: "furniture", by: "box", confidence: 0.4 }, { ...shape, confidence: 0.3 }).label, "furniture");
  assert.equal(combine({ label: "furniture", by: "box" }, null).label, "furniture");
});

test("the geometry classifier fails with a message, never silently", async () => {
  const none = createGeometryIdentifier({});
  await assert.rejects(none.ready, /No geometry model/);
  await assert.rejects(none.classify([new Float32Array(3)]), /No geometry model/);
  // A worker that cannot load the runtime: the caller hears why.
  const failing = createGeometryIdentifier({
    modelUrl: "model.onnx",
    workerFactory: () => {
      const w = { postMessage: (msg) => setTimeout(() => w.onmessage({ data: { id: msg.id, ok: false, error: "Could not load onnxruntime-web" } }), 0), terminate() {} };
      return w;
    },
  });
  await assert.rejects(failing.ready, /onnxruntime-web/);
});

test("the geometry classifier's protocol: one batch out, one answer per cloud back", async () => {
  const points = 4;
  const id = createGeometryIdentifier({
    modelUrl: "model.onnx",
    points,
    workerFactory: () => {
      const w = {
        postMessage(msg) {
          setTimeout(() => {
            if (msg.type === "init") w.onmessage({ data: { id: msg.id, ok: true, result: { backend: "wasm" } } });
            else {
              const { count, batch } = msg.payload;
              assert.equal(batch.length, count * points * 3);
              const logits = new Float32Array(count * 20);
              for (let i = 0; i < count; i += 1) logits[i * 20 + (i % 20)] = 3;
              w.onmessage({ data: { id: msg.id, ok: true, result: { logits, classes: 20 } } });
            }
          }, 0);
        },
        terminate() {},
      };
      return w;
    },
  });
  assert.deepEqual(await id.ready, { backend: "wasm" });
  const out = await id.classify([new Float32Array(12), new Float32Array(12)]);
  assert.deepEqual(out.map((o) => o.ifcClass), [IFCNET_CLASSES[0], IFCNET_CLASSES[1]]);
  id.terminate();
});
