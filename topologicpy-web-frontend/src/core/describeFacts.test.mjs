/** "Describe this model": the facts sent are the app's own, with their units. */
import test from "node:test";
import assert from "node:assert/strict";
import { DESCRIBE_INSTRUCTIONS, describeFacts } from "./describeFacts.js";
import { codeCheck } from "./codeCheck.js";

test("people leaving: the animation's outcome, said to be an illustration, with or without the fire", () => {
  const result = { people: 197, lastOut: 305, longestWait: 260, exits: [{ people: 189 }, { people: 8 }], cutOff: 0, inside: 0 };
  const without = describeFacts({ crowd: { result, withFire: false, occupantArea: 9 } }).text;
  assert.match(without, /People leaving \(the egress animation, an illustration, not an evacuation model\): 197 people at one per 9 m² of walkable floor, without the fire; the last out at 5.1 min through 2 exits \(189, 8 people\), the longest hold-up 4.3 min\./);
  const withFire = describeFacts({ crowd: { result: { ...result, cutOff: 3 }, withFire: true, occupantArea: 9 } }).text;
  assert.match(withFire, /with the fire, keeping out of places it made untenable;[^\n]*; 3 cut off by the fire stayed where they were\./);
  assert.doesNotMatch(describeFacts({}).text, /People leaving/);
});

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

/** Two 10 m corridors 3.5 m apart, a stair at the far end, an exit door at x = 0 below (as codeCheck.test.mjs builds them). */
function twoStoreys() {
  const nodes = [], kinds = [], edges = [];
  for (let s = 0; s < 2; s += 1) {
    for (let x = 0; x < 10; x += 1) {
      nodes.push(x, s * 3.5 + 0.75, 0);
      kinds.push(0);
      if (x) edges.push(s * 10 + x - 1, s * 10 + x);
    }
  }
  nodes.push(9, 0.75 + 1.75, 1);
  kinds.push(1);
  edges.push(9, 20, 20, 19);
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: Uint8Array.from(kinds), upAxis: "y", agentHeight: 0.75, spacing: 1 };
}

test("egress by storey: each storey's ways out and longest walk from the code check, named by the IFC's storeys; its verdicts only once the user ran it", () => {
  const egress = codeCheck(twoStoreys(), { exits: [0], levels: [0, 3.5] });
  const semantics = { storeys: [{ name: "EG", elevation: 0.1 }, { name: "OG1", elevation: 3.4 }] };
  const plain = describeFacts({ semantics, egress }).lines;
  assert.ok(plain.includes("Storey EG (floor at 0 m, with doors out of the building): 1 exit (doors out of the building, or stairs down); the longest walk to one is 9 m."), plain.join("\n"));
  assert.ok(plain.includes("Storey OG1 (floor at 3.5 m): 1 stair to leave by; the longest walk to one is 9 m."), plain.join("\n"));
  assert.ok(plain.includes("Highest occupied floor: 3.5 m above the storey with the doors out."));
  assert.ok(!plain.some((l) => /code check|Group B|pass|fail/i.test(l)), "no verdicts unasked");
  // The graph's floors are in the viewer frame: with a coordination shift of
  // 3.2 m the floor at 3.5 m is the IFC's 0.3 m storey, not the one at 3.4 m.
  const shifted = describeFacts({ semantics: { storeys: [{ name: "EG", elevation: -3.2 }, { name: "OG1", elevation: 0.3 }, { name: "OG2", elevation: 3.4 }] }, egress, levelShift: 3.2 }).lines;
  assert.ok(shifted.some((l) => l.startsWith("Storey EG (floor at -3.2 m, with doors out")), shifted.join("\n"));
  assert.ok(shifted.some((l) => l.startsWith("Storey OG1 (floor at 0.3 m)")), shifted.join("\n"));
  const checked = describeFacts({ semantics, egress, codeChecked: true }).lines;
  assert.match(checked.join("\n"), /Storey EG .*; code check: walk pass, number of exits /);
  assert.match(checked.join("\n"), /Code check \(run by the user\): as an office \(Group B\), each limit the stricter of SBC 801-CR-2018 and IBC 2021, without sprinklers; walking limit 60 m; not a high-rise building; exit door widths cannot tell/);
  assert.ok(checked.every((l) => !/undefined|NaN/.test(l)), checked.join("\n"));
});

test("egress by storey: places that reach no way out are counted; a tall building's storeys past twelve are summed up in one line", () => {
  // A storey whose far half is cut off from the door and the stair: 5 of its 10 places reach nothing.
  const g = twoStoreys();
  g.blocked = new Uint8Array(g.edges.length / 2);
  g.blocked[4] = 1; // the link from x = 4 to x = 5 on the ground storey
  g.blocked[18] = 1; // and from x = 9 onto the stair
  const cut = describeFacts({ egress: codeCheck(g, { exits: [0], levels: [0, 3.5] }) }).lines;
  assert.ok(cut.some((l) => /^Storey \(floor at 0 m, with doors out of the building\): .*; 5 of 10 places reach none on the graph/.test(l)), cut.join("\n"));
  const storey = (k) => ({ index: k, elevation: 3 * k, discharge: k === 0, places: 10, exits: 2, maxTravel: 10 + k, unreached: 0, checks: [] });
  const tall = describeFacts({ egress: { storeys: Array.from({ length: 15 }, (_, k) => storey(k)), building: { height: 42 }, limit: 60 } }).lines;
  assert.equal(tall.filter((l) => /^Storey /.test(l)).length, 12);
  assert.ok(tall.includes("3 more storeys; the longest walk to a way out among them is 24 m."), tall.join("\n"));
});

test("with only a model open, the facts say only what the model gives", () => {
  const { lines } = describeFacts({ stats: { storeys: 1, doors: 3 }, semantics: { schema: "IFC4" } });
  assert.deepEqual(lines, ["Model: an IFC file, schema IFC4. 1 storeys.", "Doors: 3."]);
  assert.deepEqual(describeFacts().lines, ["Model: an IFC file. 0 storeys."]);
});
