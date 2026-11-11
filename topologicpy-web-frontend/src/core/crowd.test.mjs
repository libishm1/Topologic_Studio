// The egress animation's crowd (core/crowd.js): a floor-field walk to the
// nearest exit, timed by the SFPE hydraulic values.
import test from "node:test";
import assert from "node:assert/strict";

import { FREE_SPEED, GATHER_AREA, SFPE, STEP_RANDOMNESS, WALK_FLUCTUATION, fluctuationDrift, simulateCrowd } from "./crowd.js";
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
  // One person walking a corridor 500 m long at 0.6 m/s with no spread (slow
  // enough that the crowding cap, 0.9 m/s in a corridor one node wide, does not
  // clip the wobble): each step's speed over 0.6 is the factor.
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
