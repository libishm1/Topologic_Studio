/** "Describe this model": the facts sent are the app's own, with their units. */
import test from "node:test";
import assert from "node:assert/strict";
import { DESCRIBE_INSTRUCTIONS, describeFacts } from "./describeFacts.js";

test("the facts come from the app's state, with units, and leave out what it does not know", () => {
  const { lines, text } = describeFacts({
    file: { name: "duplex.ifc" },
    stats: { floors: 21, stairs: 4, doors: 14, walls: 57, storeys: 2, spaces: 21 },
    semantics: {
      schema: "IFC2X3",
      storeys: [{ name: "Level 1", elevation: 0 }, { name: "Level 2", elevation: 3.1 }],
      spaces: [{ name: "A101", longName: "Kitchen" }, { name: "A102", longName: "Living" }],
      doors: [{ exit: true, widthM: 0.9 }, { exit: false, widthM: 0.8 }],
      roofSlabs: [1],
    },
    identification: [{ label: "furniture", by: "class" }, { label: "furniture", by: "name" }, { label: "wall", by: "class" }],
    graph: { stats: { nodes: 1597, edges: 4597, components: 2, largest_component: 800, stair_nodes: 40 } },
    route: { found: true, length: 19.1 },
    walkSpeed: 1.2,
    fire: { model: "flamescapes", step: 60 },
    combustion: { time: 120, horizon: 120, peak: 2313, grid: [42, 40, 118], cell: 0.23, backend: { kind: "gpu", name: "x" }, openings: { doorCount: 14, doors: "closed", windowCount: 24, windows: "closed", terminals: 0, ventilation: "off" }, embersOn: false },
    rl: { method: "shielded", reached_exit: true, moves: 39, certified: true },
  });
  assert.match(text, /schema IFC2X3\. 2 storeys \(Level 1 at 0 m, Level 2 at 3\.1 m\)/);
  assert.match(text, /Rooms \(IfcSpace\): 2, named Kitchen, Living/);
  assert.match(text, /Doors: 2, 1 marked as exits in the file; clear widths 0\.8 to 0\.9 m/);
  assert.match(text, /2 furniture, 1 wall; 1 were generic/);
  assert.match(text, /in 2 parts \(the largest holds 50%\)/);
  assert.match(text, /19\.1 m, about 16 s at 1\.2 m\/s/);
  assert.match(text, /peaked at 2313 kW on a 42 x 40 x 118 grid of 0\.23 m cells, on the GPU\. The solver is not calibrated/);
  assert.match(text, /14 doors closed, 24 windows closed\./);
  assert.match(text, /out in 39 moves, certified as the earliest escape/);
  assert.ok(lines.every((l) => !/undefined|NaN/.test(l)), text);
  assert.match(DESCRIBE_INSTRUCTIONS, /Use only the facts given/);
});

test("with only a model open, the facts say only what the model gives", () => {
  const { lines } = describeFacts({ stats: { storeys: 1, doors: 3 }, semantics: { schema: "IFC4" } });
  assert.deepEqual(lines, ["Model: an IFC file, schema IFC4. 1 storeys.", "Doors: 3."]);
  assert.deepEqual(describeFacts().lines, ["Model: an IFC file. 0 storeys."]);
});
