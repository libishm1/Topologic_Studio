// The egress animation's crowd (core/crowd.js): a floor-field walk to the
// nearest exit, timed by the SFPE hydraulic values.
import test from "node:test";
import assert from "node:assert/strict";

import { CORRIDOR_FLOW, DENSITY_READING_CALIBRATED, DENSITY_READING_DEFAULT, FREE_SPEED, GATHER_AREA, IMO_PASSENGERS, POPULATIONS, SFPE, STAIR_FLOW, STAIR_UP, STEP_RANDOMNESS, WALK_FLUCTUATION, fluctuationDrift, passageCapacity, passageSections, populationMix, simulateCrowd } from "./crowd.js";
import { doorCapacity } from "./performanceCheck.js";
import { KIND_DOOR, KIND_FLOOR, KIND_STAIR } from "./lattice.js";

/**
 * A room of nx x nz nodes 0.5 m apart, nodes 0.75 m above the floor, joined
 * to their eight neighbours; `exits` are door nodes added past its ends
 * ("left" at x = -0.5 m, "right" at the far side), `stairRow` makes one row stairs.
 */
function room(nx, nz, { exits = ["left"], stairRow = -1 } = {}) {
  const s = 0.5, h = 0.75;
  const nodes = [], kinds = [], edges = [];
  const id = (i, k) => i * nz + k;
  for (let i = 0; i < nx; i += 1) {
    for (let k = 0; k < nz; k += 1) {
      nodes.push(i * s, h, k * s);
      kinds.push(i === stairRow ? KIND_STAIR : KIND_FLOOR);
    }
  }
  for (let i = 0; i < nx; i += 1) {
    for (let k = 0; k < nz; k += 1) {
      for (const [di, dk] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const a = i + di, b = k + dk;
        if (a < nx && b >= 0 && b < nz) edges.push(id(i, k), id(a, b));
      }
    }
  }
  const exitNodes = [];
  const mid = Math.floor(nz / 2);
  for (const side of exits) {
    const v = nodes.length / 3;
    nodes.push(side === "left" ? -s : nx * s, h, mid * s);
    kinds.push(KIND_DOOR);
    edges.push(v, id(side === "left" ? 0 : nx - 1, mid));
    exitNodes.push(v);
  }
  return {
    graph: { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: s, agentHeight: h, upAxis: "y" },
    exits: exitNodes,
  };
}

/** Each person's segments as {t0, t1, a, b}. */
function walks(out) {
  return Array.from({ length: out.people }, (_, i) => {
    const list = [];
    for (let j = 0; j < out.count[i]; j += 1) {
      const q = (out.first[i] + j) * 4;
      list.push({ t0: out.segments[q], t1: out.segments[q + 1], a: out.segments[q + 2], b: out.segments[q + 3] });
    }
    return list;
  });
}

test("everyone walks out, each walk joined up and in time order", () => {
  const { graph, exits } = room(16, 8);
  const out = simulateCrowd(graph, { exits, occupantArea: 1 });
  assert.equal(out.people, 32); // 128 nodes x 0.25 m^2 / 1 m^2
  assert.equal(out.inside, 0);
  for (const list of walks(out)) {
    for (let j = 0; j < list.length; j += 1) {
      assert.ok(list[j].t1 >= list[j].t0);
      if (j) {
        assert.equal(list[j].a, list[j - 1].b, "a walk is joined up");
        assert.ok(list[j].t0 >= list[j - 1].t1 - 1e-4, "and in time order");
      }
    }
  }
  assert.ok(out.lastOut > 0 && Number.isFinite(out.lastOut));
  assert.deepEqual(out.exits.map((e) => e.people), [32]);
});

test("a node holds one person at a time", () => {
  const { graph, exits } = room(12, 6);
  const out = simulateCrowd(graph, { exits, occupantArea: 0.5 });
  // Who holds each node when: from the step towards it till the step away.
  const held = new Map();
  for (const list of walks(out)) {
    for (let j = 0; j < list.length; j += 1) {
      const { b } = list[j];
      if (b === exits[0] || b >= graph.nodes.length / 3) continue;
      const until = j + 1 < list.length ? list[j + 1].t0 : Infinity;
      if (!held.has(b)) held.set(b, []);
      held.get(b).push([list[j].t0, until]);
    }
  }
  for (const spans of held.values()) {
    spans.sort((p, q) => p[0] - q[0]);
    for (let j = 1; j < spans.length; j += 1) assert.ok(spans[j][0] >= spans[j - 1][1] - 1e-4, "no two at once");
  }
});

test("an exit passes no more than the SFPE specific flow through its width", () => {
  const { graph, exits } = room(10, 6);
  const out = simulateCrowd(graph, { exits, occupantArea: 0.25, doorWidth: 0.9 });
  const c = doorCapacity(0.9);
  assert.ok(Math.abs(c - 1.3155 * 0.6) < 1e-3, "about 0.8 persons a second through 0.9 m");
  const through = walks(out)
    .map((list) => list.find((s) => s.b === exits[0])?.t0)
    .filter((t) => t !== undefined)
    .sort((a, b) => a - b);
  assert.equal(through.length, out.people);
  for (let j = 1; j < through.length; j += 1) assert.ok(through[j] - through[j - 1] >= 1 / c - 1e-3, "one every 1/c s at most");
  // So the last is out no sooner than the door allows.
  assert.ok(out.lastOut >= (out.people - 1) / c);
});

test("people take their nearest exit", () => {
  const { graph, exits } = room(30, 6, { exits: ["left", "right"] });
  const out = simulateCrowd(graph, { exits, occupantArea: 1, choice: "nearest" });
  assert.equal(out.choice, "nearest");
  const middle = (30 * 0.5) / 2;
  let checked = 0;
  walks(out).forEach((list, i) => {
    const x = graph.nodes[list[0].a * 3];
    if (Math.abs(x - middle) < 2) return;
    assert.equal(out.exitOf[i], x < middle ? exits[0] : exits[1]);
    checked += 1;
  });
  assert.ok(checked > 10);
  assert.equal(out.exits.length, 2);
});

/**
 * A room 6 nodes across and 20 deep, full (a person a node), with the door
 * at the middle of one end's side and a second exit in the far corner: most
 * are nearest the first.
 */
function busyRoom() {
  const { graph, exits } = room(6, 20);
  const v = graph.nodes.length / 3;
  const corner = 5 * 20 + 19;
  return {
    graph: {
      ...graph,
      nodes: Float32Array.from([...graph.nodes, 3.0, 0.75, 9.5]),
      edges: Int32Array.from([...graph.edges, v, corner]),
      kinds: Uint8Array.from([...graph.kinds, KIND_DOOR]),
    },
    exits: [...exits, v],
  };
}

test("with the quickest choice the back of a long queue takes a farther free exit, and everyone is out sooner", () => {
  const { graph, exits } = busyRoom();
  const nearest = simulateCrowd(graph, { exits, occupantArea: 0.25, choice: "nearest" });
  const quickest = simulateCrowd(graph, { exits, occupantArea: 0.25 });
  assert.equal(quickest.choice, "quickest", "the default");
  assert.equal(quickest.people, nearest.people);
  assert.equal(quickest.inside, 0);
  const by = (out, e) => Array.from(out.exitOf).filter((x) => x === e).length;
  const gap = (out) => Math.abs(by(out, exits[0]) - by(out, exits[1]));
  assert.ok(gap(quickest) < gap(nearest), `the exits share the crowd more evenly (${gap(quickest)} against ${gap(nearest)})`);
  // Some whose nearest exit is the first leave by the second.
  const turned = Array.from(nearest.exitOf).filter((e, i) => e === exits[0] && quickest.exitOf[i] === exits[1]).length;
  assert.ok(turned > 0);
  assert.ok(quickest.lastOut < 0.8 * nearest.lastOut, `out by ${quickest.lastOut.toFixed(0)} s against ${nearest.lastOut.toFixed(0)} s`);
});

test("people keep the exit they chose unless another is clearly quicker: no one turns back and forth", () => {
  const { graph, exits } = busyRoom();
  const out = simulateCrowd(graph, { exits, occupantArea: 0.25 });
  // With two exits, a second turn would be back to the first.
  assert.ok(Math.max(...out.switches) <= 1, `at most ${Math.max(...out.switches)} turns`);
  assert.ok(out.switches.some((s) => s > 0), "some turn as the queues change");
  assert.ok(simulateCrowd(graph, { exits, occupantArea: 0.25, choice: "nearest" }).switches.every((s) => s === 0));
});

test("with one exit the quickest choice is the nearest", () => {
  const { graph, exits } = room(12, 6);
  const quickest = simulateCrowd(graph, { exits, occupantArea: 0.5 });
  const nearest = simulateCrowd(graph, { exits, occupantArea: 0.5, choice: "nearest" });
  assert.equal(quickest.choice, "nearest");
  assert.deepEqual(Array.from(quickest.segments), Array.from(nearest.segments));
});

test("walking speeds stay within the SFPE values, slower on a stair", () => {
  const { graph, exits } = room(20, 6, { stairRow: 10 });
  const out = simulateCrowd(graph, { exits, occupantArea: 1, spread: 0.25 });
  assert.ok(Math.abs(FREE_SPEED - 1.19) < 0.005);
  const n = graph.nodes.length / 3;
  for (const list of walks(out)) {
    for (const s of list) {
      if (s.b >= n || s.t1 <= s.t0) continue;
      const len = Math.hypot(...[0, 1, 2].map((c) => graph.nodes[s.b * 3 + c] - graph.nodes[s.a * 3 + c]));
      const v = len / (s.t1 - s.t0);
      const stair = graph.kinds[s.a] === KIND_STAIR || graph.kinds[s.b] === KIND_STAIR;
      const top = (stair ? FREE_SPEED * (SFPE.kStair / SFPE.k) : FREE_SPEED) * 1.25;
      assert.ok(v <= top + 1e-3, `${v.toFixed(3)} m/s at most ${top.toFixed(3)}`);
      assert.ok(v >= 0.15 - 1e-6);
    }
  }
});

test("the same seed gives the same crowd, another seed another", () => {
  const { graph, exits } = room(12, 6);
  const a = simulateCrowd(graph, { exits, occupantArea: 1, seed: 3 });
  const b = simulateCrowd(graph, { exits, occupantArea: 1, seed: 3 });
  const c = simulateCrowd(graph, { exits, occupantArea: 1, seed: 4 });
  assert.deepEqual(Array.from(a.segments), Array.from(b.segments));
  assert.notDeepEqual(Array.from(a.segments), Array.from(c.segments));
});

test("people who cannot reach an exit are counted, not placed", () => {
  const { graph, exits } = room(8, 4);
  // A second room with no way out.
  const extra = room(4, 4).graph;
  const n = graph.nodes.length / 3;
  const merged = {
    ...graph,
    nodes: Float32Array.from([...graph.nodes, ...Array.from(extra.nodes, (x, j) => (j % 3 === 0 ? x + 20 : x))]),
    edges: Int32Array.from([...graph.edges, ...Array.from(extra.edges, (v) => v + n)]),
    kinds: Uint8Array.from([...graph.kinds, ...extra.kinds]),
  };
  const out = simulateCrowd(merged, { exits, occupantArea: 1 });
  assert.equal(out.unreached, extra.nodes.length / 3 - 1); // its own door node is a door
  assert.equal(out.inside, 0);
});

test("a runner loops down the stair the crowd uses most, at the walk's own speeds", async () => {
  const { stairRunner } = await import("./crowd.js");
  // Two floors joined by a stair: the upper room's people walk down it.
  const s = 0.5, h = 0.75;
  const nodes = [], kinds = [], edges = [];
  // The upper floor, 3 m up: a 6 x 4 room, then a stair of 8 nodes down to a landing, then the exit.
  const upper = (i, k) => i * 4 + k;
  for (let i = 0; i < 6; i += 1) for (let k = 0; k < 4; k += 1) {
    nodes.push(i * s, 3 + h, k * s);
    kinds.push(KIND_FLOOR);
  }
  for (let i = 0; i < 6; i += 1) for (let k = 0; k < 4; k += 1) {
    if (i + 1 < 6) edges.push(upper(i, k), upper(i + 1, k));
    if (k + 1 < 4) edges.push(upper(i, k), upper(i, k + 1));
  }
  let prev = upper(5, 1);
  for (let j = 1; j <= 8; j += 1) {
    const v = nodes.length / 3;
    nodes.push(5 * s + j * 0.3, 3 - j * 0.375 + h, 1 * s);
    kinds.push(KIND_STAIR);
    edges.push(prev, v);
    prev = v;
  }
  for (let j = 1; j <= 5; j += 1) {
    const v = nodes.length / 3;
    nodes.push(5 * s + 2.4 + j * s, h, 1 * s);
    kinds.push(j === 5 ? KIND_DOOR : KIND_FLOOR);
    edges.push(prev, v);
    prev = v;
  }
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: s, agentHeight: h, upAxis: "y" };
  const crowd = simulateCrowd(graph, { exits: [prev], occupantArea: 1 });
  const runner = stairRunner(graph, crowd);
  assert.ok(runner, "a stair is found");
  assert.equal(runner.people, 1);
  // It starts above and ends below: it goes down the whole stair.
  assert.ok(runner.points[1] - runner.points[runner.points.length - 2] >= 3 - 1e-3);
  // Its segments run end to end from 0, the waits left out.
  for (let k = 0; k < runner.count[0]; k += 1) {
    assert.equal(runner.segments[k * 4 + 2], k);
    if (k) assert.ok(Math.abs(runner.segments[k * 4] - runner.segments[k * 4 - 3]) < 1e-4);
  }
  assert.ok(Math.abs(runner.duration - runner.segments[runner.count[0] * 4 - 3]) < 1e-3);
  // No stair, no runner.
  const flat = room(8, 4);
  assert.equal(stairRunner(flat.graph, simulateCrowd(flat.graph, { exits: flat.exits, occupantArea: 1 })), null);
});

/** Two floors joined by a straight stair of `steps` nodes, the upper room nx x nz; returns the graph and its exit. */
function twoFloors(nx = 6, nz = 4, steps = 8) {
  const s = 0.5, h = 0.75;
  const nodes = [], kinds = [], edges = [];
  const id = (i, k) => i * nz + k;
  for (let i = 0; i < nx; i += 1) for (let k = 0; k < nz; k += 1) {
    nodes.push(i * s, 3 + h, k * s);
    kinds.push(KIND_FLOOR);
  }
  for (let i = 0; i < nx; i += 1) for (let k = 0; k < nz; k += 1) {
    if (i + 1 < nx) edges.push(id(i, k), id(i + 1, k));
    if (k + 1 < nz) edges.push(id(i, k), id(i, k + 1));
  }
  let prev = id(nx - 1, 1);
  for (let j = 1; j <= steps; j += 1) {
    const v = nodes.length / 3;
    nodes.push((nx - 1) * s + j * 0.3, 3 - (j * 3) / steps + h, s);
    kinds.push(KIND_STAIR);
    edges.push(prev, v);
    prev = v;
  }
  for (let j = 1; j <= 4; j += 1) {
    const v = nodes.length / 3;
    nodes.push((nx - 1) * s + steps * 0.3 + j * s, h, s);
    kinds.push(j === 4 ? KIND_DOOR : KIND_FLOOR);
    edges.push(prev, v);
    prev = v;
  }
  return { graph: { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: s, agentHeight: h, upAxis: "y" }, exit: prev };
}

/** The closest two people on stair nodes at any one time (m), sampled every 0.1 s. */
function closestOnStair(graph, out) {
  const ws = walks(out);
  const n = graph.nodes.length / 3;
  const end = Math.max(...ws.map((list) => list.at(-1).t1));
  let closest = Infinity;
  for (let t = 0; t <= end; t += 0.1) {
    const held = [];
    for (const list of ws) {
      // The node a person holds at t: the target of their latest step begun.
      let node = list[0].a;
      for (const s of list) if (s.t0 <= t + 1e-9) node = s.b;
      if (node < n && graph.kinds[node] === KIND_STAIR) held.push(node);
    }
    for (let a = 0; a < held.length; a += 1) for (let b = a + 1; b < held.length; b += 1) {
      const d = Math.hypot(...[0, 1, 2].map((c) => graph.nodes[held[a] * 3 + c] - graph.nodes[held[b] * 3 + c]));
      closest = Math.min(closest, d);
    }
  }
  return closest;
}

test("on a stair people keep their gap; without it they would close up", () => {
  const { graph, exit } = twoFloors();
  const spaced = simulateCrowd(graph, { exits: [exit], occupantArea: 0.5 });
  assert.equal(spaced.inside, 0);
  assert.ok(closestOnStair(graph, spaced) > 0.9, `closest ${closestOnStair(graph, spaced).toFixed(2)} m`);
  const packed = simulateCrowd(graph, { exits: [exit], occupantArea: 0.5, stairGap: 0 });
  assert.ok(closestOnStair(graph, packed) < 0.9, "the test can tell");
});

test("a gathering place holds what its area takes; the earliest there leave first", () => {
  const { graph, exits } = room(10, 6);
  const out = simulateCrowd(graph, { exits, occupantArea: 0.5, shelterArea: 5 });
  const C = Math.floor(5 / GATHER_AREA);
  assert.equal(out.shelter, C);
  assert.ok(out.people > C + 5);
  const order = Array.from(out.outAt, (t, i) => [t, i]).sort((a, b) => a[0] - b[0]).map(([, i]) => i);
  // Everyone but the last C to arrive leaves, in the order they came.
  order.forEach((i, j) => assert.equal(Number.isFinite(out.goneAt[i]), j < out.people - C));
  for (let j = 1; j < out.people - C; j += 1) assert.ok(out.goneAt[order[j]] >= out.goneAt[order[j - 1]]);
  // Never more than C there at once.
  for (const t of out.outAt) {
    const there = Array.from(out.outAt).filter((a, i) => a <= t && out.goneAt[i] > t).length;
    assert.ok(there <= C, `${there} there at ${t.toFixed(1)} s`);
  }
  // Unlimited by default: no one leaves.
  const all = simulateCrowd(graph, { exits, occupantArea: 0.5 });
  assert.ok(Array.from(all.goneAt).every((t) => t === Infinity));
});

test("the forecast's runner keeps the plan's timeline", async () => {
  const { forecastRunner } = await import("./crowd.js");
  const { graph } = room(6, 2);
  const route = { nodes: [0, 2, 4, 6], times: [12, 13.5, 15, 17] };
  const runner = forecastRunner(graph, route);
  assert.equal(runner.count[0], 3);
  assert.deepEqual(Array.from(runner.segments.subarray(0, 4)), [12, 13.5, 0, 1]);
  assert.equal(runner.outAt[0], 17);
  // At foot level: the nodes stand agentHeight above the floor.
  assert.ok(Math.abs(runner.points[1] - (graph.nodes[1] - graph.agentHeight)) < 1e-6);
  assert.equal(forecastRunner(graph, null), null);
  assert.equal(forecastRunner(graph, { nodes: [3], times: [0] }), null);
});

test("with a fire, no one walks into a place it has made untenable; they turn to another exit or stay put", () => {
  // A corridor 30 nodes long, an exit at each end.
  const { graph, exits } = room(30, 1, { exits: ["left", "right"] });
  const n = graph.nodes.length / 3;
  // The fire closes nodes 6 to 9 from 2 s: those between it and the right
  // exit that would have gone left must turn right.
  const closes = new Float64Array(n).fill(Infinity);
  for (let v = 6; v <= 9; v += 1) closes[v] = 2;
  for (const choice of ["quickest", "nearest"]) {
    const out = simulateCrowd(graph, { exits, occupantArea: 0.25, closes, start: [0, 0], choice });
    const without = simulateCrowd(graph, { exits, occupantArea: 0.25, start: [0, 0], choice });
    assert.equal(out.choice, choice);
    // From an open place no one steps into the fire (those in it walk out).
    for (const list of walks(out)) for (const s of list) if (s.b < n && closes[s.a] > s.t0) assert.ok(closes[s.b] > s.t0, `${choice}: into node ${s.b} at ${s.t0.toFixed(1)} s`);
    const turned = walks(out).filter((list, i) => list[0].a >= 10 && list[0].a <= 13 && out.exitOf[i] === exits[1] && without.exitOf[i] === exits[0]);
    assert.ok(turned.length > 0, `${choice}: someone turned to the other exit`);
    assert.equal(out.cutOff, 0);
    assert.equal(out.inside, 0);
  }
  // One exit only, the fire between: those beyond it stay where they are.
  const single = room(30, 1);
  const shut = new Float64Array(single.graph.nodes.length / 3).fill(Infinity);
  for (let v = 5; v <= 7; v += 1) shut[v] = 0;
  const trapped = simulateCrowd(single.graph, { exits: single.exits, occupantArea: 0.25, closes: shut, start: [0, 0], horizon: 60 });
  const beyond = walks(trapped).filter((list) => list[0].a > 7).length;
  assert.equal(trapped.cutOff, beyond);
  assert.equal(trapped.inside, beyond);
  for (const list of walks(trapped)) for (const s of list) if (s.b < single.graph.nodes.length / 3 && shut[s.a] > s.t0) assert.ok(shut[s.b] > s.t0);
});

// ---- randomness in how people move (wiki/roadmap/crowd-motion-randomness-plan-2026-11-11.md, stage 1)

/** Each node's holders as [from, till] spans; throws if two overlap. */
function assertOneANode(graph, exits, out) {
  const held = new Map();
  for (const list of walks(out)) {
    for (let j = 0; j < list.length; j += 1) {
      const { b } = list[j];
      if (exits.includes(b) || b >= graph.nodes.length / 3) continue;
      const until = j + 1 < list.length ? list[j + 1].t0 : Infinity;
      if (!held.has(b)) held.set(b, []);
      held.get(b).push([list[j].t0, until]);
    }
  }
  for (const spans of held.values()) {
    spans.sort((p, q) => p[0] - q[0]);
    for (let j = 1; j < spans.length; j += 1) assert.ok(spans[j][0] >= spans[j - 1][1] - 1e-4, "no two at once");
  }
}

test("with the randomness off a run is exactly as before", () => {
  const { graph, exits } = room(30, 6, { exits: ["left", "right"] });
  const before = simulateCrowd(graph, { exits, occupantArea: 1, seed: 5 });
  const off = simulateCrowd(graph, { exits, occupantArea: 1, seed: 5, fluctuate: false, kS: STEP_RANDOMNESS.exact });
  assert.deepEqual(Array.from(off.segments), Array.from(before.segments));
});

test("speeds wobble as Corbetta et al. measured: about the person's own, a spread of about 16 %", () => {
  // The drift: zero at f = 1 and at the unstable point 0; restoring about 1 at 8 alpha up^2 = 0.5 a second.
  assert.equal(Math.abs(fluctuationDrift(1)), 0);
  assert.equal(Math.abs(fluctuationDrift(0)), 0);
  assert.ok(Math.abs((fluctuationDrift(1.001) - fluctuationDrift(0.999)) / 0.002 + 8 * WALK_FLUCTUATION.alpha) < 1e-6);
  // One person walking a corridor 500 m long at 0.6 m/s with no spread: alone,
  // nothing slows them, so each step's speed over 0.6 is the factor.
  const { graph, exits } = room(1000, 1, { exits: ["right"] });
  const n = graph.nodes.length / 3;
  const ratios = [];
  for (const seed of [1, 2, 3]) {
    const out = simulateCrowd(graph, { exits, occupantArea: 1e9, maxPeople: 1, speed: 0.6, spread: 0, start: [0, 0], horizon: 3000, fluctuate: true, seed });
    for (const s of walks(out)[0]) if (s.a < n && s.b < n && s.a !== s.b) ratios.push(0.5 / (s.t1 - s.t0) / 0.6);
  }
  const mean = ratios.reduce((a, b) => a + b, 0) / ratios.length;
  const sd = Math.sqrt(ratios.reduce((a, b) => a + (b - mean) ** 2, 0) / ratios.length);
  assert.ok(Math.abs(mean - 1) < 0.04, `mean factor ${mean.toFixed(3)}`);
  assert.ok(Math.abs(sd - 0.16) < 0.04, `spread ${sd.toFixed(3)}`);
  // Exact without it (times are kept in single precision).
  const steady = simulateCrowd(graph, { exits, occupantArea: 1e9, maxPeople: 1, speed: 0.6, spread: 0, start: [0, 0], horizon: 3000 });
  for (const s of walks(steady)[0]) if (s.a < n && s.b < n && s.a !== s.b) assert.ok(Math.abs(0.5 / (s.t1 - s.t0) - 0.6) < 1e-3);
});

test("the floor field's steps: still one a node and out of the fire; the larger kS the nearer the exact walk, the smaller the slower", () => {
  const { graph, exits } = room(30, 6, { exits: ["left", "right"] });
  const exact = simulateCrowd(graph, { exits, occupantArea: 0.5, start: [0, 0], seed: 1 });
  const lastOut = (kS) => [1, 2, 3].map((seed) => simulateCrowd(graph, { exits, occupantArea: 0.5, start: [0, 0], seed, kS }).lastOut).reduce((a, b) => a + b, 0) / 3;
  const direct = lastOut(50), ordered = lastOut(STEP_RANDOMNESS.direct), lost = lastOut(STEP_RANDOMNESS.disoriented);
  assert.ok(Math.abs(direct - exact.lastOut) < 0.15 * exact.lastOut, `kS 50: ${direct.toFixed(1)} s against exact ${exact.lastOut.toFixed(1)} s`);
  assert.ok(lost > ordered, `kS 0.4: ${lost.toFixed(1)} s against kS 10: ${ordered.toFixed(1)} s`);
  for (const kS of [STEP_RANDOMNESS.direct, STEP_RANDOMNESS.wandering, STEP_RANDOMNESS.disoriented]) {
    const out = simulateCrowd(graph, { exits, occupantArea: 0.5, seed: 4, kS, fluctuate: true, horizon: 3000 });
    assertOneANode(graph, exits, out);
    assert.equal(out.inside, 0, `kS ${kS}: everyone out`);
  }
  // With a fire, no one steps from an open place into one it has made untenable.
  const corridor = room(30, 1, { exits: ["left", "right"] });
  const m = corridor.graph.nodes.length / 3;
  const closes = new Float64Array(m).fill(Infinity);
  for (let v = 6; v <= 9; v += 1) closes[v] = 2;
  for (const seed of [1, 2, 3]) {
    const out = simulateCrowd(corridor.graph, { exits: corridor.exits, occupantArea: 0.25, closes, start: [0, 0], seed, kS: STEP_RANDOMNESS.wandering, fluctuate: true });
    for (const list of walks(out)) for (const s of list) if (s.b < m && closes[s.a] > s.t0) assert.ok(closes[s.b] > s.t0, `seed ${seed}: into node ${s.b} at ${s.t0.toFixed(1)} s`);
  }
  // One seed, one run; another, another.
  const a = simulateCrowd(graph, { exits, occupantArea: 0.5, seed: 9, kS: 2, fluctuate: true });
  const b = simulateCrowd(graph, { exits, occupantArea: 0.5, seed: 9, kS: 2, fluctuate: true });
  const c = simulateCrowd(graph, { exits, occupantArea: 0.5, seed: 10, kS: 2, fluctuate: true });
  assert.deepEqual(Array.from(a.segments), Array.from(b.segments));
  assert.notDeepEqual(Array.from(a.segments), Array.from(c.segments));
});

// ---- validation informs the model (wiki/research/crowd-validation-plan-2026-11-11.md; tools/validation/crowd-validation.mjs)

/** Each segment's speed, m/s, on the graph's own nodes. */
function speedsOf(graph, out) {
  const n = graph.nodes.length / 3;
  const out2 = [];
  for (const list of walks(out)) {
    for (const s of list) {
      if (s.b >= n || s.a === s.b || s.t1 <= s.t0) continue;
      const len = Math.hypot(...[0, 1, 2].map((c) => graph.nodes[s.b * 3 + c] - graph.nodes[s.a * 3 + c]));
      out2.push({ ...s, v: len / (s.t1 - s.t0), len });
    }
  }
  return out2;
}

test("a walker alone never counts themselves: they keep their own speed in single file and by a wall", () => {
  // Counting the mover capped a lone walker at 0.90 m/s in single file (RiMEA Test 16) and 1.15 by a wall.
  for (const [nx, nz] of [[60, 1], [40, 3]]) {
    const { graph, exits } = room(nx, nz);
    const out = simulateCrowd(graph, { exits, occupantArea: 1e9, maxPeople: 1, speed: 1.3, spread: 0, start: [0, 0], seed: 2 });
    const speeds = speedsOf(graph, out);
    assert.ok(speeds.length > 5);
    for (const s of speeds) assert.ok(Math.abs(s.v - 1.3) < 1e-3, `${nx} x ${nz}: ${s.v.toFixed(3)} m/s`);
  }
});

test("each walker keeps their own clock: unimpeded, their steps follow on without waiting for the time step, along the lattice or across it", () => {
  // Steps used to wait for the next 0.1 s: 1.33 m/s became 0.99 along the lattice and 1.16 across it (RiMEA Test 1).
  const { graph, exits } = room(40, 40);
  for (const seed of [1, 2, 3]) {
    const out = simulateCrowd(graph, { exits, occupantArea: 1e9, maxPeople: 1, speed: 1.33, spread: 0, start: [0, 0], seed });
    const list = walks(out)[0].filter((s) => s.b < graph.nodes.length / 3);
    for (let j = 1; j < list.length; j += 1) assert.ok(Math.abs(list[j].t0 - list[j - 1].t1) < 1e-4, `seed ${seed}: a gap of ${(list[j].t0 - list[j - 1].t1).toFixed(3)} s`);
    const length = speedsOf(graph, out).reduce((s, x) => s + x.len, 0);
    assert.ok(Math.abs(length / (out.outAt[0] - list[0].t0) - 1.33) < 0.005, `seed ${seed}: realised ${(length / (out.outAt[0] - list[0].t0)).toFixed(3)} m/s`);
  }
});

/** Two rooms of 12 x 12 nodes joined by one door node, the exit at the far side of the second; returns the graph, exit and door. */
function twoRooms() {
  const s = 0.5, h = 0.75, N = 12;
  const nodes = [], kinds = [], edges = [];
  const id = new Map();
  const add = (i, k, kind) => {
    id.set(`${i},${k}`, nodes.length / 3);
    nodes.push(i * s, h, k * s);
    kinds.push(kind);
  };
  for (let i = 0; i < N; i += 1) for (let k = 0; k < N; k += 1) add(i, k, KIND_FLOOR);
  add(N, N / 2, KIND_DOOR);
  for (let i = N + 1; i < 2 * N + 1; i += 1) for (let k = 0; k < N; k += 1) add(i, k, KIND_FLOOR);
  add(2 * N + 1, N / 2, KIND_DOOR);
  for (const [key, a] of id) {
    const [i, k] = key.split(",").map(Number);
    for (const [di, dk] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const b = id.get(`${i + di},${k + dk}`);
      if (b === undefined) continue;
      if (di && dk && (!id.has(`${i + di},${k}`) || !id.has(`${i},${k + dk}`))) continue;
      edges.push(a, b);
    }
  }
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: s, agentHeight: h, upAxis: "y" };
  return { graph, exit: id.get(`${2 * N + 1},${N / 2}`), door: id.get(`${N},${N / 2}`), n: nodes.length / 3 };
}

test("the door, not the crawl into its queue, sets what passes: the flow rises with each door node's own width", () => {
  // The crawl into a packed queue passed 0.29 persons/s through a door of any width (RiMEA Test 12d).
  const { graph, exit, door, n } = twoRooms();
  const flows = [];
  for (const w of [0.8, 1.0, 1.2]) {
    const doorWidths = new Float32Array(n);
    doorWidths[door] = w;
    doorWidths[exit] = 3;
    const out = simulateCrowd(graph, { exits: [exit], occupantArea: 0.3, start: [0, 0], seed: 1, doorWidths, choice: "nearest" });
    assert.equal(out.inside, 0);
    const through = speedsOf(graph, out).filter((s) => s.b === door).map((s) => s.t0).sort((a, b) => a - b);
    const c = doorCapacity(w);
    for (let j = 1; j < through.length; j += 1) assert.ok(through[j] - through[j - 1] >= 1 / c - 1e-4, `${w} m: at most its capacity`);
    // The steady part of the queue passes near the door's capacity.
    const k0 = 10, k1 = through.length - 10;
    const J = (k1 - k0) / (through[k1] - through[k0]);
    assert.ok(J > 0.85 * c && J <= c + 1e-6, `${w} m: ${J.toFixed(3)} persons/s against its capacity ${c.toFixed(3)}`);
    flows.push(J);
  }
  assert.ok(flows[1] > flows[0] && flows[2] > flows[1], flows.map((f) => f.toFixed(3)).join(" < "));
  // With no widths given, every door takes doorWidth.
  const plain = simulateCrowd(graph, { exits: [exit], occupantArea: 0.3, start: [0, 0], seed: 1, doorWidth: 1.2, choice: "nearest" });
  const through = speedsOf(graph, plain).filter((s) => s.b === door).map((s) => s.t0).sort((a, b) => a - b);
  for (let j = 1; j < through.length; j += 1) assert.ok(through[j] - through[j - 1] >= 1 / doorCapacity(1.2) - 1e-4);
});

test("up a stair at STAIR_UP of the speed down it", () => {
  // Down: twoFloors' upper room walks down its stair. Up: the same stair, the exit upstairs.
  const down = twoFloors(4, 2, 10);
  const outDown = simulateCrowd(down.graph, { exits: [down.exit], occupantArea: 1e9, maxPeople: 1, spread: 0, start: [0, 0], seed: 1 });
  const onStair = (graph, out) => speedsOf(graph, out).filter((s) => graph.kinds[s.a] === KIND_STAIR && graph.kinds[s.b] === KIND_STAIR);
  const vDown = onStair(down.graph, outDown);
  assert.ok(vDown.length > 3);
  for (const s of vDown) assert.ok(Math.abs(s.v - FREE_SPEED * (SFPE.kStair / SFPE.k)) < 1e-3, `down ${s.v.toFixed(3)}`);
  // The same graph mirrored in height: the room is now below, the stair climbs from it to the exit.
  const g = down.graph;
  const flipped = { ...g, nodes: Float32Array.from(g.nodes, (x, j) => (j % 3 === 1 ? 4.5 - x : x)) };
  let climbed = 0;
  for (const seed of [1, 2, 3, 4, 5, 6]) {
    const outUp = simulateCrowd(flipped, { exits: [down.exit], occupantArea: 1e9, maxPeople: 1, spread: 0, start: [0, 0], seed });
    for (const s of onStair(flipped, outUp)) {
      climbed += 1;
      assert.ok(Math.abs(s.v - FREE_SPEED * (SFPE.kStair / SFPE.k) * STAIR_UP) < 1e-3, `up ${s.v.toFixed(3)}`);
    }
  }
  assert.ok(climbed > 3, "someone climbed the stair");
});

// ---- pre-movement times of each person's own, and walking speed in smoke (tools/validation: NIST Verif.1.1 and 2.5)

import { SMOKE_SPEED, drawPreMovement, smokeSpeedFactor } from "./crowd.js";

/** A uniform generator for the draws below (a linear congruential one, not the model's). */
function uniform(seed) {
  let a = seed >>> 0;
  return () => {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

test("without pre-movement or smoke a run is exactly as before, whatever else of them is set", () => {
  const { graph, exits } = room(30, 6, { exits: ["left", "right"] });
  const n = graph.nodes.length / 3;
  const before = simulateCrowd(graph, { exits, occupantArea: 1, seed: 5 });
  const clearAir = { K: 2, frameSeconds: 10, vis: new Float32Array(2 * n).fill(100) };
  for (const more of [
    { preMovement: null, alarm: 120 },
    { smoke: { K: 1, frameSeconds: 1, vis: new Float32Array(n).fill(2) } },
    { smokeSpeed: "frantzichNilsson" },
    { smoke: clearAir, smokeSpeed: "frantzichNilsson" },
  ]) {
    const out = simulateCrowd(graph, { exits, occupantArea: 1, seed: 5, ...more });
    assert.deepEqual(Array.from(out.segments), Array.from(before.segments), JSON.stringify(Object.keys(more)));
    assert.equal(out.slowed, 0);
  }
  assert.equal(before.smokeSpeed, null);
  // The stagger each was given is reported, within `start`.
  assert.equal(before.startAt.length, before.people);
  assert.ok(Array.from(before.startAt).every((t) => t >= 0 && t <= 10));
});

test("pre-movement draws follow the distribution asked for, and keep to its cut-offs", () => {
  const draws = (d, k = 20000) => {
    const rand = uniform(7);
    return Array.from({ length: k }, () => drawPreMovement(d, rand)).sort((a, b) => a - b);
  };
  const share = (xs, t) => xs.filter((x) => x <= t).length / xs.length;
  const median = (xs) => xs[Math.floor(xs.length / 2)];
  // Uniform.
  const u = draws([10, 100]);
  assert.ok(u[0] >= 10 && u.at(-1) <= 100);
  assert.ok(Math.abs(median(u) - 55) < 1.5);
  // Log-normal: the median, and 84.1 % below median x e^sigma.
  const l = draws({ median: 60, sigma: 0.5 });
  assert.ok(Math.abs(median(l) - 60) < 1.5, `median ${median(l)}`);
  assert.ok(Math.abs(share(l, 60 * Math.exp(0.5)) - 0.8413) < 0.01);
  // The same by mu, and through its 1st and 99th percentiles.
  assert.ok(Math.abs(median(draws({ mu: Math.log(60), sigma: 0.5 })) - 60) < 1.5);
  const p = draws({ p1: 60, p99: 180 });
  assert.ok(Math.abs(share(p, 60) - 0.01) < 0.004 && Math.abs(share(p, 180) - 0.99) < 0.004);
  assert.ok(Math.abs(median(p) - Math.sqrt(60 * 180)) < 1.5);
  // Cut off: nothing outside, and the shape kept between (the share below the median of the uncut one, by hand).
  const c = draws({ median: 60, sigma: 0.5, min: 30, max: 150 });
  assert.ok(c[0] >= 30 && c.at(-1) <= 150);
  // P(30 < X < 60) / P(30 < X < 150), with Phi(ln(0.5) / 0.5) = 0.08283 and Phi(ln(2.5) / 0.5) = 0.96657.
  assert.ok(Math.abs(share(c, 60) - (0.5 - 0.08283) / (0.96657 - 0.08283)) < 0.01);
  // Normal, cut off at zero when no least time is given.
  const g = draws({ mean: 60, sd: 20, min: 20, max: 100 });
  assert.ok(g[0] >= 20 && g.at(-1) <= 100 && Math.abs(median(g) - 60) < 1);
  assert.ok(draws({ mean: 5, sd: 20 })[0] >= 0);
  // Weibull: the share below its scale is 1 - 1/e.
  const w = draws({ scale: 60, shape: 1.5 });
  assert.ok(Math.abs(share(w, 60) - (1 - Math.exp(-1))) < 0.01);
  assert.throws(() => drawPreMovement({ mode: 60 }, uniform(1)), /pre-movement/);
  assert.throws(() => drawPreMovement({ median: 60, sigma: 0.5, min: 100, max: 50 }, uniform(1)), /pre-movement/);
});

test("with pre-movement each person sets off at the alarm and a time of their own, the same people in the same places", () => {
  const { graph, exits } = room(20, 12);
  const n = graph.nodes.length / 3;
  const plain = simulateCrowd(graph, { exits, occupantArea: 3, seed: 4 });
  const spec = { median: 60, sigma: 0.5, min: 20, max: 200 };
  const out = simulateCrowd(graph, { exits, occupantArea: 3, seed: 4, preMovement: spec, alarm: 30 });
  assert.equal(out.people, plain.people);
  assert.equal(out.inside, 0);
  const first = (r) => walks(r).map((list) => list[0]);
  const a = first(plain), b = first(out);
  for (let i = 0; i < out.people; i += 1) {
    // The same place.
    assert.equal(b[i].a, a[i].a);
    assert.ok(out.startAt[i] >= 30 + 20 - 1e-3 && out.startAt[i] <= 30 + 200 + 1e-3, `set off at ${out.startAt[i]}`);
    // No one moves before their time.
    assert.ok(b[i].t0 >= out.startAt[i] - 1e-3);
  }
  // Not all at once: the times differ from person to person, and the last out is later than without.
  assert.ok(new Set(Array.from(out.startAt).map((t) => Math.round(t))).size > out.people / 3);
  assert.ok(out.lastOut > plain.lastOut + 30);
  // Most set off unhindered, exactly at their time.
  const onTime = b.filter((s, i) => Math.abs(s.t0 - out.startAt[i]) < 1e-3).length;
  assert.ok(onTime >= 0.9 * out.people, `${onTime} of ${out.people} on time`);
  // The same seed, the same times; the uniform form as well.
  const again = simulateCrowd(graph, { exits, occupantArea: 3, seed: 4, preMovement: spec, alarm: 30 });
  assert.deepEqual(Array.from(again.startAt), Array.from(out.startAt));
  const flat = simulateCrowd(graph, { exits, occupantArea: 3, seed: 4, preMovement: [40, 50] });
  assert.ok(Array.from(flat.startAt).every((t) => t >= 40 && t <= 50));
});

test("groups: a group's own pre-movement and alarm stand before the run's, and its own start before the run's distribution", () => {
  const { graph, exits } = room(20, 12);
  const id = (i, k) => i * 12 + k;
  const area = (i0, i1) => {
    const list = [];
    for (let i = i0; i <= i1; i += 1) for (let k = 0; k < 12; k += 1) list.push(id(i, k));
    return list;
  };
  const groups = [
    { nodes: area(0, 4), count: 10 },
    { nodes: area(6, 10), count: 10, preMovement: [100, 110], alarm: 5 },
    { nodes: area(12, 16), count: 10, start: [0, 2] },
  ];
  const out = simulateCrowd(graph, { exits, groups, preMovement: { mean: 60, sd: 5, min: 50, max: 70 }, alarm: 20, seed: 3 });
  assert.equal(out.people, 30);
  for (let i = 0; i < 30; i += 1) {
    const t = out.startAt[i];
    if (out.groupOf[i] === 0) assert.ok(t >= 70 - 1e-3 && t <= 90 + 1e-3, `the run's: ${t}`);
    if (out.groupOf[i] === 1) assert.ok(t >= 105 - 1e-3 && t <= 115 + 1e-3, `the group's: ${t}`);
    if (out.groupOf[i] === 2) assert.ok(t >= 0 && t <= 2 + 1e-3, `the group's start: ${t}`);
  }
});

test("the smoke relation built in is FDS+Evac's: a straight fall with K to a tenth; a curve of one's own is read between its points", () => {
  const fn = SMOKE_SPEED.frantzichNilsson;
  assert.deepEqual(fn, { alpha: 0.706, beta: -0.057, floor: 0.1 });
  assert.equal(smokeSpeedFactor(fn, 0), 1);
  assert.ok(Math.abs(smokeSpeedFactor(fn, 1) - (0.706 - 0.057) / 0.706) < 1e-12);
  assert.ok(Math.abs(smokeSpeedFactor(fn, 7.5) - (0.706 - 0.057 * 7.5) / 0.706) < 1e-12);
  // The line reaches a tenth at K = 0.9 alpha / 0.057, about 11.1 /m, and is held there.
  assert.ok(Math.abs(smokeSpeedFactor(fn, (0.9 * 0.706) / 0.057) - 0.1) < 1e-12);
  assert.equal(smokeSpeedFactor(fn, 20), 0.1);
  // A curve by K, and one by visibility (K = 3 / visibility; clear air is past its last point).
  const byK = { by: "K", points: [[0, 1], [2, 0.5], [4, 0.25]] };
  assert.equal(smokeSpeedFactor(byK, 1), 0.75);
  assert.equal(smokeSpeedFactor(byK, 3), 0.375);
  assert.equal(smokeSpeedFactor(byK, 9), 0.25);
  const byVis = { by: "visibility", points: [[1, 0.2], [3, 1]] };
  assert.equal(smokeSpeedFactor(byVis, 3), 0.2);
  assert.ok(Math.abs(smokeSpeedFactor(byVis, 1.5) - 0.6) < 1e-12);
  assert.equal(smokeSpeedFactor(byVis, 0.5), 1);
  assert.equal(smokeSpeedFactor(byVis, 0), 1);
});

test("in smoke a walker keeps the relation's share of their speed, from the smoke where they stand and when", () => {
  const { graph, exits } = room(80, 3);
  const n = graph.nodes.length / 3;
  const far = 79 * 3 + 1;
  const one = (more) => simulateCrowd(graph, { exits, groups: [{ nodes: [far], count: 1, speed: [1.25, 1.25] }], start: [0, 0], ...more });
  const share = (K) => (0.706 - 0.057 * K) / 0.706;
  // The same smoke everywhere: K = 1 /m is a visibility of 3 m.
  const even = one({ smoke: { K: 1, frameSeconds: 5, vis: new Float32Array(n).fill(3) }, smokeSpeed: "frantzichNilsson" });
  assert.equal(even.smokeSpeed, "frantzichNilsson");
  assert.equal(even.slowed, 1);
  const steps = speedsOf(graph, even);
  assert.ok(steps.length > 50);
  for (const s of steps) assert.ok(Math.abs(s.v - 1.25 * share(1)) < 2e-3, `${s.v.toFixed(4)} m/s`);
  const clear = one({});
  assert.ok(Math.abs((even.outAt[0] / clear.outAt[0]) * share(1) - 1) < 1e-3);
  // Smoke over the half nearer the exit only: full speed until they reach it.
  const half = new Float32Array(n).fill(100);
  for (let v = 0; v < 40 * 3; v += 1) half[v] = 3 / 5;
  const partly = one({ smoke: { K: 1, frameSeconds: 5, vis: half }, smokeSpeed: "frantzichNilsson" });
  for (const s of speedsOf(graph, partly)) assert.ok(Math.abs(s.v - 1.25 * (half[s.a] < 100 ? share(5) : 1)) < 2e-3, `from node ${s.a}: ${s.v.toFixed(4)} m/s`);
  // Smoke that comes at 10 s (the second frame, which holds to the end): full speed in steps begun before, slowed after.
  const later = new Float32Array(2 * n).fill(100);
  later.fill(3 / 3, n);
  const timed = one({ smoke: { K: 2, frameSeconds: 10, vis: later }, smokeSpeed: "frantzichNilsson" });
  const timedSteps = speedsOf(graph, timed);
  assert.ok(timedSteps.some((s) => s.t0 < 10) && timedSteps.some((s) => s.t0 > 30));
  for (const s of timedSteps) assert.ok(Math.abs(s.v - 1.25 * (s.t0 < 10 ? 1 : share(3))) < 2e-3, `at ${s.t0.toFixed(1)} s: ${s.v.toFixed(4)} m/s`);
  // A curve of one's own, by visibility: half speed at 3 m and under.
  const own = one({ smoke: { K: 1, frameSeconds: 5, vis: new Float32Array(n).fill(3) }, smokeSpeed: { by: "visibility", points: [[3, 0.5], [10, 1]] } });
  assert.equal(own.smokeSpeed, "custom");
  for (const s of speedsOf(graph, own)) assert.ok(Math.abs(s.v - 0.625) < 2e-3);
  // Smoke of the wrong size, or a relation with no such name, is refused.
  assert.throws(() => one({ smoke: { K: 2, frameSeconds: 5, vis: new Float32Array(n) }, smokeSpeed: "frantzichNilsson" }), /smoke/);
  assert.throws(() => one({ smoke: { K: 1, frameSeconds: 5, vis: new Float32Array(n) }, smokeSpeed: "jin" }), /no smoke speed relation/);
});

test("in smoke a crowd is slower out, and no faster anywhere than without it", () => {
  const { graph, exits } = room(30, 10);
  const n = graph.nodes.length / 3;
  const base = { exits, occupantArea: 2, seed: 6, start: [0, 0] };
  const clear = simulateCrowd(graph, base);
  const smoky = simulateCrowd(graph, { ...base, smoke: { K: 1, frameSeconds: 5, vis: new Float32Array(n).fill(0.5) }, smokeSpeed: "frantzichNilsson" });
  assert.equal(smoky.people, clear.people);
  assert.equal(smoky.inside, 0);
  assert.equal(smoky.slowed, smoky.people);
  assert.ok(smoky.lastOut > clear.lastOut);
  // K = 6 /m: no step faster than the share of the fastest walker's speed, 1.25 x 1.19 m/s.
  const cap = FREE_SPEED * 1.25 * ((0.706 - 0.057 * 6) / 0.706);
  for (const s of speedsOf(graph, smoky)) assert.ok(s.v <= cap + 2e-3, `${s.v.toFixed(3)} m/s`);
  assertOneANode(graph, exits, smoky);
});

/** The gathering places of a run (the points after the graph's own) that stand over a place of the graph in plan. */
function gatheredOverFloor(graph, out) {
  const n = graph.nodes.length / 3;
  const cells = new Set();
  for (let v = 0; v < n; v += 1) cells.add(`${Math.round(graph.nodes[v * 3] / 0.5)},${Math.round(graph.nodes[v * 3 + 2] / 0.5)}`);
  let over = 0;
  for (let p = n; p < out.points.length / 3; p += 1) if (cells.has(`${Math.round(out.points[p * 3] / 0.5)},${Math.round(out.points[p * 3 + 2] / 0.5)}`)) over += 1;
  return { over, places: out.points.length / 3 - n };
}

test("a large crowd gathers outside the wall, not back through it", () => {
  // 20 x 20 m, one door: about 300 people, whose gathering place would reach 7 m back from a point 3.5 m out.
  const { graph, exits } = room(40, 40);
  const out = simulateCrowd(graph, { exits, occupantArea: 1.3, horizon: 3600 });
  assert.ok(out.people > 250 && out.inside === 0);
  const { over, places } = gatheredOverFloor(graph, out);
  assert.ok(places >= out.people);
  assert.equal(over, 0);
  const n = graph.nodes.length / 3;
  for (let p = n; p < out.points.length / 3; p += 1) assert.ok(out.points[p * 3] < -0.5, "every place is beyond the wall the door is in");
});

test("a door with more floor beyond it than behind it still sends people away from the building", () => {
  // A porch of nine places past the door: away from the door's neighbours now points indoors.
  const { graph, exits } = room(12, 12);
  const nodes = Array.from(graph.nodes), kinds = Array.from(graph.kinds), edges = Array.from(graph.edges);
  const door = exits[0];
  for (let i = 0; i < 3; i += 1)
    for (let k = -1; k <= 1; k += 1) {
      const v = nodes.length / 3;
      nodes.push(-1 - i * 0.5, 0.75, 6 * 0.5 + k * 0.5);
      kinds.push(KIND_FLOOR);
      if (i === 0) edges.push(door, v);
    }
  const porch = { ...graph, nodes: Float32Array.from(nodes), kinds: Uint8Array.from(kinds), edges: Int32Array.from(edges) };
  const out = simulateCrowd(porch, { exits, occupantArea: 1.3, horizon: 3600 });
  assert.equal(out.inside, 0);
  assert.equal(gatheredOverFloor(porch, out).over, 0);
});

test("floor that reaches no exit is reported as the people it would hold", () => {
  // A second room of 6 x 6 places with no door to the first: 36 places, 9 m² at 0.25 m² a place.
  const { graph, exits } = room(8, 8);
  const nodes = Array.from(graph.nodes), kinds = Array.from(graph.kinds);
  for (let i = 0; i < 6; i += 1) for (let k = 0; k < 6; k += 1) { nodes.push(30 + i * 0.5, 0.75, k * 0.5); kinds.push(KIND_FLOOR); }
  const cut = { ...graph, nodes: Float32Array.from(nodes), kinds: Uint8Array.from(kinds) };
  const out = simulateCrowd(cut, { exits, occupantArea: 1 });
  assert.equal(out.unreached, 36);
  assert.equal(out.trapped, 9);
  assert.equal(simulateCrowd(graph, { exits, occupantArea: 1 }).trapped, 0);
});

test("people who leave by an upper exit stand at its level, and the exit says how high it is", () => {
  const { graph, exits } = room(10, 10, { exits: ["left", "right"] });
  const nodes = Float32Array.from(graph.nodes);
  nodes[exits[1] * 3 + 1] += 2.4; // the right-hand door is 2.4 m up
  const out = simulateCrowd({ ...graph, nodes }, { exits, occupantArea: 1 });
  const upper = out.exits.find((e) => e.node === exits[1]), lower = out.exits.find((e) => e.node === exits[0]);
  assert.ok(Math.abs(upper.above - 2.4) < 1e-5 && Math.abs(lower.above) < 1e-6);
  const n = nodes.length / 3;
  let high = 0;
  for (let p = n; p < out.points.length / 3; p += 1) if (Math.abs(out.points[p * 3 + 1] - 2.4) < 1e-5) high += 1;
  assert.ok(upper.people > 0 && high >= Math.min(upper.people, 1));
  assert.equal(high, upper.people);
});

test("groups: people start in their area, keep the exit given to them and walk at their population's speeds", () => {
  const { graph, exits } = room(20, 10, { exits: ["left", "right"] });
  const id = (i, k) => i * 10 + k;
  const leftEnd = [], rightEnd = [];
  for (let i = 0; i < 4; i += 1) for (let k = 0; k < 10; k += 1) { leftEnd.push(id(i, k)); rightEnd.push(id(19 - i, k)); }
  // Each end's people are sent to the far exit: the nearest and the quickest choice would both send them the other way.
  const groups = [
    { nodes: leftEnd, count: 12, exit: exits[1], speed: [1.0, 1.0] },
    { nodes: rightEnd, count: 8, exit: exits[0], speed: POPULATIONS.imoMales30to50, start: [5, 5] },
  ];
  const out = simulateCrowd(graph, { exits, groups, start: [0, 0] });
  assert.equal(out.people, 20);
  assert.equal(out.short, 0);
  assert.equal(out.inside, 0);
  for (let i = 0; i < out.people; i += 1) {
    const g = out.groupOf[i], startAt = out.segments[out.first[i] * 4 + 2];
    assert.ok((g === 0 ? leftEnd : rightEnd).includes(startAt), "starts in the group's area");
    assert.equal(out.exitOf[i], groups[g].exit, "leaves by the exit given");
    if (g === 1) assert.ok(out.segments[out.first[i] * 4] >= 5 - 1e-6, "sets off at the group's time");
  }
  assert.equal(Array.from(out.groupOf).filter((g) => g === 0).length, 12);
  // Without groups nothing changes: the same run as before the option existed.
  const plain = simulateCrowd(graph, { exits, occupantArea: 1 });
  assert.equal(plain.groupOf, null);
  assert.equal(plain.short, 0);
});

test("groups: an area too small for its people reports how many it could not place, and a wrong exit is refused", () => {
  const { graph, exits } = room(6, 6);
  const out = simulateCrowd(graph, { exits, groups: [{ nodes: [0, 1, 2], count: 5 }] });
  assert.equal(out.people, 3);
  assert.equal(out.short, 2);
  assert.throws(() => simulateCrowd(graph, { exits, groups: [{ nodes: [0, 1, 2], count: 1, exit: 7 }] }), /not one of the exits/);
});

test("groups: people given an exit are let go of it when the fire cuts their walk there", () => {
  // A corridor 30 nodes long, an exit at each end; eight people near the right end are given the left exit.
  const { graph, exits } = room(30, 1, { exits: ["left", "right"] });
  const n = graph.nodes.length / 3;
  const area = [20, 21, 22, 23, 24, 25, 26, 27];
  const closes = new Float64Array(n).fill(Infinity);
  for (let v = 6; v <= 9; v += 1) closes[v] = 2;
  for (const choice of ["quickest", "nearest"]) {
    const groups = [{ nodes: area, count: 8, exit: exits[0] }];
    // With no fire they all walk the length of the corridor to the exit given.
    const calm = simulateCrowd(graph, { exits, groups, start: [0, 0], choice });
    assert.equal(calm.released, 0);
    assert.ok(Array.from(calm.exitOf).every((e) => e === exits[0]));
    // With the fire across their way they are let go, and leave by the other exit; no one steps into the fire.
    const out = simulateCrowd(graph, { exits, groups, start: [0, 0], choice, closes });
    assert.equal(out.released, 8, choice);
    assert.equal(out.inside, 0);
    assert.ok(Array.from(out.exitOf).every((e) => e === exits[1]), choice);
    for (const list of walks(out)) for (const s of list) if (s.b < n && closes[s.a] > s.t0) assert.ok(closes[s.b] > s.t0, `${choice}: into node ${s.b}`);
  }
});

test("IMO's passengers: the mix gives whole head counts in its shares, each with its own speeds", () => {
  const groups = populationMix([0, 1, 2], 50);
  assert.equal(groups.reduce((s, g) => s + g.count, 0), 50);
  const by = Object.fromEntries(groups.map((g) => [g.name, g.count]));
  // 7, 7, 16, 10, 10 per cent of 50, for women and for men: 3.5, 3.5, 8, 5, 5 each, the halves shared out.
  assert.equal(by.imoFemalesOver50, 8);
  assert.equal(by.imoMalesOver50Impaired2, 5);
  assert.equal(by.imoFemalesUnder30 + by.imoFemales30to50 + by.imoMalesUnder30 + by.imoMales30to50, 14);
  assert.equal(Object.values(IMO_PASSENGERS).reduce((s, x) => s + x, 0), 100);
  for (const g of groups) assert.deepEqual(g.speed, POPULATIONS[g.name]);
  // In a run each person's speed lies in their own group's range.
  const { graph, exits } = room(30, 30);
  const floor = Array.from({ length: 900 }, (_, v) => v);
  const mix = populationMix(floor, 100, IMO_PASSENGERS);
  const out = simulateCrowd(graph, { exits, groups: mix, start: [0, 0] });
  assert.equal(out.people, 100);
  assert.equal(out.inside, 0);
});

/**
 * A plan of cells [i, j, kind?] 0.5 m apart joined to their eight neighbours (a diagonal only
 * where both its axis neighbours are cells), nodes 0.75 m above the floor; `at(i, j)` is a cell's node.
 */
function plan(cells) {
  const s = 0.5, h = 0.75;
  const index = new Map();
  const nodes = [], kinds = [], edges = [];
  for (const [i, j, kind = KIND_FLOOR] of cells) {
    index.set(`${i},${j}`, nodes.length / 3);
    nodes.push(i * s, h, j * s);
    kinds.push(kind);
  }
  for (const [key, a] of index) {
    const [i, j] = key.split(",").map(Number);
    for (const [di, dj] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const b = index.get(`${i + di},${j + dj}`);
      if (b === undefined || (di && dj && (!index.has(`${i + di},${j}`) || !index.has(`${i},${j + dj}`)))) continue;
      edges.push(a, b);
    }
  }
  return { graph: { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: s, agentHeight: h, upAxis: "y" }, at: (i, j) => index.get(`${i},${j}`) };
}
const block = (i0, i1, j0, j1, kind = KIND_FLOOR) => {
  const out = [];
  for (let i = i0; i <= i1; i += 1) for (let j = j0; j <= j1; j += 1) out.push([i, j, kind]);
  return out;
};

/** A room 8 m x 8 m (i 0..15, j 0..15), a corridor `lanes` nodes wide and 5 m long from its right wall, and a row of exits at the corridor's end. */
function roomAndCorridor(lanes) {
  const cells = [...block(0, 15, 0, 15), ...block(16, 25, 7, 6 + lanes), ...block(26, 26, 7, 6 + lanes, KIND_DOOR)];
  const { graph, at } = plan(cells);
  return { graph, at, exits: block(26, 26, 7, 6 + lanes).map(([i, j]) => at(i, j)), room: block(0, 15, 0, 15).map(([i, j]) => at(i, j)) };
}

/** A room 8 m x 8 m and, from the middle of its right wall, a stair as the app draws one: a single file of 16 nodes down to an exit, `width` m wide (0: no width). */
function roomAndStairFile(width) {
  const { graph, at } = plan([...block(0, 15, 0, 15), ...block(16, 31, 8, 8, KIND_STAIR), [32, 8, KIND_DOOR]]);
  // Down a slope of 7 in 11, 0.5 m between nodes along it.
  const theta = Math.atan(7 / 11);
  for (let c = 1; c <= 17; c += 1) {
    const v = at(15 + c, 8);
    graph.nodes[v * 3] = 7.5 + c * 0.5 * Math.cos(theta);
    graph.nodes[v * 3 + 1] = 0.75 - c * 0.5 * Math.sin(theta);
  }
  if (width > 0) {
    graph.widths = new Float32Array(graph.nodes.length / 3);
    for (let c = 1; c <= 16; c += 1) graph.widths[at(15 + c, 8)] = width;
  }
  return { graph, at, exits: [at(32, 8)], room: block(0, 15, 0, 15).map(([i, j]) => at(i, j)), first: at(16, 8) };
}

/** The steady flow (persons/s) of the steps on to any of `nodes` from outside them, the first and last ten left out; and the least time between two of them. */
function flowOnTo(graph, out, nodes) {
  const set = new Set(nodes);
  const times = speedsOf(graph, out).filter((s) => set.has(s.b) && !set.has(s.a)).map((s) => s.t0).sort((a, b) => a - b);
  let least = Infinity;
  for (let j = 1; j < times.length; j += 1) least = Math.min(least, times[j] - times[j - 1]);
  const k0 = 10, k1 = times.length - 11;
  return { flow: (k1 - k0) / (times[k1] - times[k0]), least, count: times.length };
}

test("passage capacities: SFPE's specific flow on the effective width, 0.79 persons/s for a 1 m corridor and 0.71 for a 1 m stair", () => {
  assert.ok(Math.abs(CORRIDOR_FLOW - 1.3157) < 1e-3);
  // The stair's follows from its k: (1 - a D) kStair D at D = 1.9, the Handbook's 1.01 persons/s/m for 7 in risers and 11 in treads.
  assert.ok(Math.abs(STAIR_FLOW - CORRIDOR_FLOW * (SFPE.kStair / SFPE.k)) < 1e-12);
  assert.ok(Math.abs(STAIR_FLOW - 1.01) < 0.01);
  assert.ok(Math.abs(passageCapacity(1.0) - 0.79) < 0.005, "corridor: 0.2 m lost a side");
  assert.ok(Math.abs(passageCapacity(1.0, true) - 0.71) < 0.005, "stair: 0.15 m lost a side");
  assert.ok(Math.abs(doorCapacity(1.0) - 0.92) < 0.005, "door: 0.15 m lost a side");
  assert.ok(passageCapacity(0.3) > 0, "never nothing");
});

test("passages found on the graph: a corridor's cross-sections wall to wall, not a room, a strip between doors or a stair without a width", () => {
  const { graph, at, exits } = roomAndCorridor(2);
  const found = passageSections(graph, { exits });
  // Each of the corridor's ten columns is one cross-section of two nodes, 1 m wide.
  assert.equal(found.corridors, 10);
  for (let i = 16; i <= 25; i += 1) {
    assert.equal(found.section[at(i, 7)], found.section[at(i, 8)], "the two lanes share it");
    assert.ok(found.section[at(i, 7)] >= 0);
    assert.ok(Math.abs(found.width[found.section[at(i, 7)]] - 1.0) < 1e-9);
    assert.ok(Math.abs(found.capacity[found.section[at(i, 7)]] - passageCapacity(1.0)) < 1e-9);
    if (i > 16) assert.notEqual(found.section[at(i, 7)], found.section[at(i - 1, 7)], "each column its own");
  }
  // The room is open floor; only the corridor's first column is a mouth.
  for (let i = 0; i <= 15; i += 1) for (let j = 0; j <= 15; j += 1) assert.equal(found.section[at(i, j)], -1);
  assert.equal(found.mouth[at(16, 7)], 1);
  assert.equal(found.mouth[at(17, 7)], 0);
  // A width given for a node is its passage's.
  const widths = new Float32Array(graph.nodes.length / 3);
  widths[at(20, 7)] = 1.4;
  const given = passageSections(graph, { exits, passageWidths: widths });
  assert.ok(Math.abs(given.capacity[given.section[at(20, 8)]] - passageCapacity(1.4)) < 1e-6);
  assert.ok(Math.abs(given.capacity[given.section[at(19, 8)]] - passageCapacity(1.0)) < 1e-9);
  // A single file of nodes is taken as two nodes wide.
  const file = roomAndCorridor(1);
  const one = passageSections(file.graph, { exits: file.exits });
  assert.ok(Math.abs(one.width[one.section[file.at(20, 7)]] - 1.0) < 1e-9);
  // A strip one node deep between a door and the exits is a way through, not a passage: its short run ends at a door.
  const strip = plan([...block(0, 15, 0, 15), [16, 8, KIND_DOOR], ...block(17, 17, 0, 15), ...block(18, 18, 0, 15, KIND_DOOR)]);
  const stripExits = block(18, 18, 0, 15).map(([i, j]) => strip.at(i, j));
  const none = passageSections(strip.graph, { exits: stripExits });
  assert.equal(none.corridors, 0);
  // A stair is a passage only with a width.
  const bare = roomAndStairFile(0), wide = roomAndStairFile(2.4);
  assert.equal(passageSections(bare.graph, { exits: bare.exits }).stairs, 0);
  const stair = passageSections(wide.graph, { exits: wide.exits });
  assert.equal(stair.stairs, 16);
  assert.equal(stair.slots[wide.first], 4, "2.4 m: four abreast");
  assert.equal(stair.slots[wide.at(15, 8)], 4, "and at its mouth on the floor");
  assert.ok(Math.abs(stair.capacity[stair.section[wide.first]] - passageCapacity(2.4, true)) < 1e-6);
});

test("passageFlow is off by default: a graph with stair widths walks exactly as one without", () => {
  const a = roomAndStairFile(0), b = roomAndStairFile(2.4);
  const opts = { occupantArea: 0.5, start: [0, 0], seed: 3 };
  const plain = simulateCrowd(a.graph, { ...opts, exits: a.exits });
  const withWidths = simulateCrowd(b.graph, { ...opts, exits: b.exits });
  assert.deepEqual(withWidths.segments, plain.segments);
  assert.equal(plain.passages, null);
  // And with passageFlow, a graph that has no narrow floor and no stair width walks as before too.
  const on = simulateCrowd(a.graph, { ...opts, exits: a.exits, passageFlow: true });
  assert.deepEqual(on.passages, { corridors: 0, stairs: 0 });
  assert.deepEqual(on.segments, plain.segments);
});

test("with passageFlow a corridor passes at most the SFPE flow for its width, and less than a door of that width", () => {
  const flows = {};
  for (const lanes of [2, 4]) {
    const { graph, at, exits, room } = roomAndCorridor(lanes);
    const mouth = Array.from({ length: lanes }, (_, k) => at(16, 7 + k));
    const opts = { exits, groups: [{ nodes: room, count: 120 }], doorWidth: 5, start: [0, 0], seed: 1, choice: "nearest" };
    const off = flowOnTo(graph, simulateCrowd(graph, opts), mouth);
    const out = simulateCrowd(graph, { ...opts, passageFlow: true });
    const on = flowOnTo(graph, out, mouth);
    const c = passageCapacity(lanes * 0.5);
    assert.equal(out.inside, 0);
    assert.equal(out.passages.corridors, 10);
    assert.ok(on.least >= 1 / c - 1e-4, `${lanes} lanes: never two steps into it closer than its capacity allows`);
    assert.ok(on.flow > 0.9 * c && on.flow <= c + 1e-6, `${lanes} lanes: ${on.flow.toFixed(3)} persons/s against ${c.toFixed(3)}`);
    if (lanes === 2) assert.ok(off.flow > c, `without it the lattice passes more (${off.flow.toFixed(3)})`);
    flows[lanes] = on.flow;
  }
  assert.ok(flows[2] < doorCapacity(1.0), "a corridor does not pass more than a door of its width");
  assert.ok(flows[4] > flows[2]);
});

test("with passageFlow a stair drawn as one file of nodes passes what its real width takes", () => {
  const flows = [];
  for (const width of [1.2, 2.4]) {
    const { graph, exits, room, first } = roomAndStairFile(width);
    const opts = { exits, groups: [{ nodes: room, count: 120 }], doorWidth: 5, start: [0, 0], seed: 1, choice: "nearest" };
    const out = simulateCrowd(graph, { ...opts, passageFlow: true });
    const on = flowOnTo(graph, out, [first]);
    const c = passageCapacity(width, true);
    assert.equal(out.inside, 0);
    assert.equal(out.passages.stairs, 16);
    assert.ok(on.least >= 1 / c - 1e-4, `${width} m: at most its capacity`);
    assert.ok(on.flow > 0.9 * c && on.flow <= c + 1e-6, `${width} m: ${on.flow.toFixed(3)} persons/s against ${c.toFixed(3)}`);
    // No stair node ever holds more than the stair takes abreast (two at 1.2 m, four at 2.4 m).
    const n = graph.nodes.length / 3;
    const events = [];
    for (const s of speedsOf(graph, out)) {
      if (s.b < n && graph.kinds[s.b] === KIND_STAIR) events.push([s.t0, 1, s.b]);
      if (graph.kinds[s.a] === KIND_STAIR) events.push([s.t0, -1, s.a]);
    }
    events.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    const standing = new Map();
    let most = 0;
    for (const [, d, v] of events) {
      standing.set(v, (standing.get(v) || 0) + d);
      most = Math.max(most, standing.get(v));
    }
    assert.ok(most <= Math.floor(width / 0.6 + 1e-6), `${width} m: ${most} on one node`);
    flows.push(on.flow);
  }
  assert.ok(flows[1] > 2 * flows[0], `2.4 m passes more than twice 1.2 m: ${flows.map((f) => f.toFixed(3)).join(", ")}`);
  // Without the option both pass the same, the single file's.
  const a = roomAndStairFile(1.2), b = roomAndStairFile(2.4);
  const opts = { groups: [{ nodes: a.room, count: 120 }], doorWidth: 5, start: [0, 0], seed: 1, choice: "nearest" };
  const fa = flowOnTo(a.graph, simulateCrowd(a.graph, { ...opts, exits: a.exits }), [a.first]).flow;
  const fb = flowOnTo(b.graph, simulateCrowd(b.graph, { ...opts, exits: b.exits }), [b.first]).flow;
  assert.equal(fa, fb);
  // Up the same stair, STAIR_UP of the capacity down it.
  const up = roomAndStairFile(1.2);
  up.graph.nodes = Float32Array.from(up.graph.nodes, (x, j) => (j % 3 === 1 ? 1.5 - x : x));
  const outUp = simulateCrowd(up.graph, { ...opts, exits: up.exits, passageFlow: true });
  const fu = flowOnTo(up.graph, outUp, [up.first]);
  assert.ok(fu.flow <= STAIR_UP * passageCapacity(1.2, true) + 1e-6 && fu.flow > 0.85 * STAIR_UP * passageCapacity(1.2, true), `up ${fu.flow.toFixed(3)}`);
});

/** A single file of `count` nodes 0.5 m apart with an exit at its far end. */
function singleFile(count) {
  const { graph, at } = plan([...block(0, count - 1, 0, 0), [count, 0, KIND_DOOR]]);
  return { graph, at, exits: [at(count, 0)] };
}

test("densityReading: left out, a run is exactly as with the default reading spelt out", () => {
  const { graph, exits } = room(16, 8);
  const opts = { exits, occupantArea: 0.4, start: [0, 0], seed: 2 };
  const plain = simulateCrowd(graph, opts);
  const spelt = simulateCrowd(graph, { ...opts, densityReading: { ...DENSITY_READING_DEFAULT } });
  assert.deepEqual(spelt.segments, plain.segments);
  assert.deepEqual(DENSITY_READING_DEFAULT, { centre: "ahead", radius: 1.0, area: "all", floor: 0.15 });
  // The reading chosen on the single-file data: a wider disc, over the floor the others could stand on.
  assert.deepEqual(DENSITY_READING_CALIBRATED, { centre: "ahead", radius: 2.0, area: "standing", floor: 0.15 });
});

test("densityReading 'calibrated': a dense single file walks slower than by default, and a lone walker still at their own speed", () => {
  // Every other node taken at the start: 1 person a metre, 2 persons/m2 on the lattice.
  const { graph, at, exits } = singleFile(200);
  const nodes = Array.from({ length: 80 }, (_, k) => at(2 * k, 0));
  const opts = { exits, groups: [{ nodes, count: 80, speed: [1, 1] }], doorWidth: 5, start: [0, 0], seed: 1, choice: "nearest", horizon: 40 };
  const meanSpeed = (out) => {
    // Metres walked between 10 and 30 s, by those still inside at 30 s.
    const n = graph.nodes.length / 3;
    let sum = 0, people = 0;
    walks(out).forEach((list, i) => {
      if (out.outAt[i] < 30) return;
      let d = 0;
      for (const s of list) if (s.b < n && s.t0 >= 10 && s.t1 <= 30) d += 0.5;
      sum += d / 20;
      people += 1;
    });
    return sum / people;
  };
  const before = meanSpeed(simulateCrowd(graph, opts));
  const after = meanSpeed(simulateCrowd(graph, { ...opts, densityReading: "calibrated" }));
  // SFPE's relation at 2 persons/m2 gives 0.66 m/s; the default reading lets them walk faster than that.
  assert.ok(after < before - 0.1, `calibrated ${after.toFixed(3)} m/s, default ${before.toFixed(3)}`);
  assert.ok(before > SFPE.k * (1 - SFPE.a * 2), `default ${before.toFixed(3)}`);
  // Alone, the walker counts no one and walks at their own speed, as by default.
  const lone = simulateCrowd(graph, { exits, groups: [{ nodes: [at(0, 0)], count: 1, speed: [1.4, 1.4] }], start: [0, 0], densityReading: "calibrated" });
  for (const s of speedsOf(graph, lone)) assert.ok(Math.abs(s.v - 1.4) < 1e-3, `alone at ${s.v.toFixed(3)} m/s`);
});
