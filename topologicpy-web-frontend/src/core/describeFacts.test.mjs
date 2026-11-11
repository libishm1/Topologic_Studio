/** "Describe this model": the facts sent are the app's own, with their units. */
import test from "node:test";
import assert from "node:assert/strict";
import { DESCRIBE_INSTRUCTIONS, MAX_FACTS, describeFacts } from "./describeFacts.js";
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
  assert.match(text, /peaked at 2313 kW on a 42 x 40 x 118 grid of 0\.23 m cells, on the GPU\. Against held-out measured fires on cells of this size, the nearest tested \(0\.2 m\): smoke layer depth indicative/);
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

const crowdResult = { people: 481, lastOut: 410, longestWait: 120, exits: [{ people: 300 }, { people: 181 }], cutOff: 0, inside: 0, choice: "quickest", switches: [0, 2, 0, 1] };
const slimeResult = {
  agents: 1553,
  steps: 400,
  exits: [{ flux: 830 }, { flux: 750 }, { flux: 100 }],
  fire: "soft",
  smokeSource: "visibility",
  seconds: 168,
  closed: 412,
  fireShare: 0.08,
  route: { length: 18.9, shortest: 18.9 },
  medialRoute: { length: 20.4 },
  agreement: 0.71,
  bottlenecks: [{ people: 202, width: 0.92, door: true, seconds: 250 }, { people: 140, width: 1.6, door: false, seconds: 176 }],
  deadEnds: { limit: 6.096, sprinklered: false, area: 3494, worst: 41.4 },
  abandonedArea: 163,
};
const perfResult = {
  status: "fail",
  start: 540,
  factor: 2,
  detection: 60,
  detectionFrom: "no run",
  cue: 90,
  criterion: { label: "ratio 2 and 30 s (SFEG, Italian)" },
  storeys: [{ status: "pass" }, { status: "fail" }, { status: "fail" }],
  queues: { busiest: { persons: 120, wait: 95 } },
};

test("people leaving: how they chose their exits, and the range over seeds", () => {
  const lines = describeFacts({ crowd: { result: crowdResult, withFire: false, occupantArea: 9, exits: "all", fluctuate: true, steps: "wandering", runs: { of: 3, samples: [{ lastOut: 400 }, { lastOut: 430 }, { lastOut: 410 }] } } }).lines;
  const text = lines.join("\n");
  assert.match(text, /How they chose: every exit out of the building was open to them; each took the exit that got them out soonest, the walk or the queue ahead, and 2 turned to another exit on the way; walking speeds varied over time as measured \(Corbetta et al\. 2017\); each step chosen at random, wandering by the floor field/);
  assert.match(text, /Over 3 runs with different random seeds the last out ranged from 6\.7 min to 7\.2 min \(median 6\.8 min\)\./);
  const one = describeFacts({ crowd: { result: { ...crowdResult, choice: "nearest", switches: null }, withFire: false, occupantArea: 9, exits: "picked", fluctuate: false, steps: "exact", runs: null } }).text;
  assert.match(one, /How they chose: all were sent to the exit picked alone, to show one exit's load; each took the nearest exit by the shortest walk\./);
  assert.doesNotMatch(one, /Over \d+ runs|Corbetta|floor field/);
});

test("the slime mould: what it grew, held back by the fire, and its readings, said to be a reading for design", () => {
  const text = describeFacts({ slime: slimeResult }).text;
  assert.match(text, /Slime mould \(Jones's Physarum agents [^)]*not an evacuation model\): 1553 agents for 400 steps, 1680 reaching an exit through 3 exits; held back by the fire and the Flamescapes run's smoke/);
  assert.match(text, /412 places untenable by then and 8% of its trail laid in smoke or fire; from the chosen start along its tubes 18\.9 m against the shortest walk 18\.9 m, keeping to the floor's medial axis 20\.4 m; 71% of its desire lines/);
  assert.match(text, /Slime mould readings: bottlenecks [^:]*: 202 people through 0\.92 m of door, 250 s; 140 people through 1\.6 m of corridor, 176 s; the walk before a second, separate way out opens is longer than 6\.1 m \(IBC 2021 1020\.5's dead-end limit for corridors, applied to every place, rooms included, as a reference\) over 3494 m² of floor, the longest 41\.4 m; it explored and then gave up 163 m²/);
  const none = describeFacts({ slime: { ...slimeResult, fire: "none", route: null, medialRoute: null, bottlenecks: [], deadEnds: null, abandonedArea: 0 } }).text;
  assert.match(none, /grown as if there were no fire; 71%/);
  assert.match(none, /Slime mould readings: no bottleneck holds the desire lines' people 10 s or more\./);
  assert.doesNotMatch(describeFacts({ slime: { agents: 0 } }).text, /Slime/);
});

test("the performance check, the building's fire protection, the spaces left out and a nearer exit, each only when there", () => {
  const text = describeFacts({
    perf: perfResult,
    sprinklers: { heads: new Array(38), source: "rooms", rooms: [{ status: "pass" }, { status: "pass" }, { status: "fail" }, { status: "n/a" }], obstructionCount: 0 },
    systems: { lines: ["water in 18 rooms", "clean agent in 2 rooms"] },
    extinguishers: { sites: [1, 2, 3], source: "placed", check: { status: "pass", limit: 22.9 } },
    graph: { stats: { nodes: 10, edges: 9 }, outdoorSpaces: [{ why: "outside the building with no door onto it" }, { why: "outside the building with no door onto it" }, { why: "named as a void: its bottom is no floor" }] },
    route: { found: true, length: 54.9 },
    nearerExit: { walk: 5.7, route: 54.9 },
  }).text;
  assert.match(text, /Performance check \(run by the user, ASET against RSET from the forecast; ratio 2 and 30 s \(SFEG, Italian\)\): fail; occupants must be able to set off up to 540 s after ignition beside an exit, later farther away; detection at 60 s as set \(no Flamescapes run to detect\), the alarm at 90 s; storeys: 1 pass, 2 fail; the busiest door passes 120 people and holds them up to 95 s\./);
  assert.match(text, /Sprinklers \(laid out by the user\): 38 heads, placed by NFPA 13 in 3 rooms; 2 rooms meet every placement rule, 1 fail one, 0 cannot be told; beams and ducts not checked/);
  assert.match(text, /Suppression systems by room use: water in 18 rooms; clean agent in 2 rooms\./);
  assert.match(text, /Extinguishers: 3, placed on the walk by the travel rule; travel check pass \(every place within 22\.9 m of walking to one\)\./);
  assert.match(text, /Spaces not walked on: 2 outside the building with no door onto them \(roofs, terraces\) and 1 named as voids\./);
  assert.match(text, /54\.9 m, about[^\n]*\n- Another exit out of the building is much nearer that start: 5\.7 m away\./);
  const bare = describeFacts({ graph: { stats: { nodes: 10, edges: 9 }, outdoorSpaces: [] }, route: { found: true, length: 10 } }).text;
  assert.doesNotMatch(bare, /Performance|Sprinklers|Suppression|Extinguishers|not walked on|much nearer/);
});

test("with everything run, the facts stay within the Worker's 8,000 characters (its MAX_FACTS)", () => {
  const storey = (k) => ({ index: k, elevation: 3 * k, discharge: k === 0, places: 10, exits: 2, maxTravel: 10 + k, unreached: 1, checks: [{ id: "travel", status: "pass" }, { id: "exits", status: "pass" }, { id: "separation", status: "fail" }] });
  const { text, lines } = describeFacts({
    file: { name: "a-long-model-name-for-a-hospital-building.ifc" },
    stats: { floors: 40, stairs: 8, doors: 400, walls: 900, storeys: 15, spaces: 300 },
    semantics: {
      schema: "IFC4",
      storeys: Array.from({ length: 15 }, (_, k) => ({ name: `Level ${k}`, elevation: 3 * k })),
      spaces: Array.from({ length: 300 }, (_, k) => ({ name: `R${k}`, longName: `Patient room number ${k}` })),
      doors: Array.from({ length: 400 }, (_, k) => ({ exit: k < 6, widthM: 0.9 + (k % 3) * 0.1 })),
    },
    identification: Array.from({ length: 50 }, (_, k) => ({ label: ["furniture", "wall", "door", "column", "beam"][k % 5], by: k % 2 ? "class" : "name" })),
    graph: { stats: { nodes: 90000, edges: 300000, components: 3, largest_component: 88000, stair_nodes: 900 }, cores: [{ kind: "stair", linked: 1 }], outdoorSpaces: [{ why: "outside the building with no door onto it" }] },
    egress: { storeys: Array.from({ length: 15 }, (_, k) => storey(k)), building: { height: 42, sprinklered: true, highRise: true }, limit: 76.2, doors: { status: "pass", text: "every exit door at least 0.81 m clear" } },
    codeChecked: true,
    route: { found: true, length: 54.9 },
    nearerExit: { walk: 5.7, route: 54.9 },
    walkSpeed: 1.2,
    fire: { model: "flamescapes", step: 60 },
    combustion: { time: 300, horizon: 300, peak: 2313, grid: [42, 40, 118], cell: 0.23, backend: { kind: "gpu", name: "x" }, openings: { doorCount: 400, doors: "open", windowCount: 240, windows: "closed", terminals: 30, ventilation: "running", ductCells: 25 }, embersOn: true, embers: 40, spots: 2 },
    plan: { result: { front: [{ dose: 0.1, arrival: 200 }], earliest: { arrival: 180 } }, inputs: { departAfter: 60, doseLimit: 0.3 } },
    rl: { method: "shielded", reached_exit: true, moves: 39, certified: true },
    crowd: { result: crowdResult, withFire: true, occupantArea: 9, exits: "all", fluctuate: true, steps: "disoriented", runs: { of: 5, samples: [400, 410, 420, 430, 440].map((lastOut) => ({ lastOut })) } },
    slime: slimeResult,
    perf: perfResult,
    sprinklers: { heads: new Array(900), source: "rooms", rooms: Array.from({ length: 300 }, () => ({ status: "pass" })), obstructionCount: 0 },
    systems: { lines: ["water in 280 rooms", "clean agent in 12 rooms", "foam in 8 rooms"] },
    extinguishers: { sites: new Array(60), source: "placed", check: { status: "pass", limit: 22.9 } },
  });
  // Fewer room names and storey lines, so the last facts are not the ones the Worker drops.
  assert.ok(text.length <= MAX_FACTS, `${text.length} characters`);
  assert.match(text, /named Patient room number 0, [^\n]* and 290 more\./);
  assert.equal(lines.filter((l) => /^Storey /.test(l)).length, 6);
  assert.ok(lines.includes("9 more storeys; the longest walk to a way out among them is 24 m."), text);
  assert.match(text, /- Extinguishers: 60, /);
  assert.ok(lines.every((l) => !/undefined|NaN/.test(l)), text);
});

test("with only a model open, the facts say only what the model gives", () => {
  const { lines } = describeFacts({ stats: { storeys: 1, doors: 3 }, semantics: { schema: "IFC4" } });
  assert.deepEqual(lines, ["Model: an IFC file, schema IFC4. 1 storeys.", "Doors: 3."]);
  assert.deepEqual(describeFacts().lines, ["Model: an IFC file. 0 storeys."]);
});
