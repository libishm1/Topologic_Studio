// A slime mould's way out (core/slimeEgress.js): Jones's agents on the
// walkable graph, drawn to the exits and kept from the fire.
import test from "node:test";
import assert from "node:assert/strict";

import { SLIME_DEFAULTS, sensorReach, slimeEgress } from "./slimeEgress.js";
import { KIND_DOOR, KIND_FLOOR } from "./lattice.js";

/**
 * A floor nx x nz nodes 0.5 m apart joined to their eight neighbours, less
 * the nodes `hole(i, k)` takes out, with exit doors added beyond the given
 * nodes (each [i, k] gets a door 0.5 m further along x).
 */
function floor(nx, nz, { hole = () => false, exitsAt = [] } = {}) {
  const s = 0.5, h = 0.75;
  const nodes = [], kinds = [], edges = [];
  const id = new Map();
  for (let i = 0; i < nx; i += 1) {
    for (let k = 0; k < nz; k += 1) {
      if (hole(i, k)) continue;
      id.set(`${i},${k}`, nodes.length / 3);
      nodes.push(i * s, h, k * s);
      kinds.push(KIND_FLOOR);
    }
  }
  for (let i = 0; i < nx; i += 1) {
    for (let k = 0; k < nz; k += 1) {
      const a = id.get(`${i},${k}`);
      if (a === undefined) continue;
      for (const [di, dk] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const b = id.get(`${i + di},${k + dk}`);
        if (b !== undefined) edges.push(a, b);
      }
    }
  }
  const exits = [];
  for (const [i, k] of exitsAt) {
    const v = nodes.length / 3;
    nodes.push((i + 1) * s, h, k * s);
    kinds.push(KIND_DOOR);
    edges.push(v, id.get(`${i},${k}`));
    exits.push(v);
  }
  return { graph: { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), upAxis: "y" }, exits, id };
}

/**
 * A room at the left (columns 0 to 3) joined to the right by two corridors
 * two nodes wide round a block, the top one straight to an exit at the top
 * right, the bottom one longer (down, along and back up).
 */
function twoWays() {
  const nx = 30, nz = 14;
  const hole = (i, k) => i >= 4 && i <= 27 && k >= 2 && k <= 11 && !(i >= 26 && k >= 2);
  const built = floor(nx, nz, { hole, exitsAt: [[29, 0]] });
  const top = [], bottom = [], room = [];
  for (let i = 8; i <= 22; i += 1) {
    for (const k of [0, 1]) top.push(built.id.get(`${i},${k}`));
    for (const k of [12, 13]) bottom.push(built.id.get(`${i},${k}`));
  }
  // The room's upper half, from where the top corridor is clearly the shorter (from its bottom the two are nearly even).
  for (let i = 0; i <= 3; i += 1) for (let k = 2; k <= 6; k += 1) room.push(built.id.get(`${i},${k}`));
  return { ...built, top, bottom, room, start: built.id.get("1,7") };
}

const mean = (values, nodes) => nodes.reduce((s, v) => s + values[v], 0) / nodes.length;

test("the tubes form along the shorter way to the exit, and the route keeps to them", () => {
  const { graph, exits, top, bottom, room, start } = twoWays();
  for (const seed of [1, 2]) {
    const fromRoom = slimeEgress(graph, { exits, places: room, agents: 30, steps: 300, seed });
    assert.ok(fromRoom.exits[0]?.flux > 0, "agents reach the exit");
    // Seeds 1 to 8 gave 2.5 to 5.8 times (2026-11-11).
    assert.ok(mean(fromRoom.weights, top) > 2 * mean(fromRoom.weights, bottom), `seed ${seed}: top ${mean(fromRoom.weights, top).toFixed(3)} against bottom ${mean(fromRoom.weights, bottom).toFixed(3)}`);
  }
  // From the whole floor, the bottom corridor's own people take it: it carries trail too, less than the top.
  const out = slimeEgress(graph, { exits, start, agents: 60, steps: 300, seed: 1 });
  assert.ok(out.agents === 60 && mean(out.weights, top) > mean(out.weights, bottom));
  assert.ok(out.route, "a route from the start");
  assert.ok(out.route.nodes.some((v) => top.includes(v)) && !out.route.nodes.some((v) => bottom.includes(v)), "by the top corridor");
  assert.ok(out.route.length >= out.route.shortest - 1e-9);
  assert.ok(out.route.length < 1.1 * out.route.shortest, `${out.route.length.toFixed(1)} m against ${out.route.shortest.toFixed(1)} m`);
});

test("the hard fire: a place it has made untenable is a wall: no agent enters it, no trail is laid there, and the tubes go round", () => {
  const { graph, exits, top, bottom, start, id } = twoWays();
  const n = graph.nodes.length / 3;
  // The fire closes the top corridor's middle from 0 s.
  const closes = new Float64Array(n).fill(Infinity);
  for (let i = 14; i <= 16; i += 1) for (const k of [0, 1]) closes[id.get(`${i},${k}`)] = 0;
  const out = slimeEgress(graph, { exits, start, agents: 60, steps: 400, closes, time: 30, seed: 1, fire: "hard" });
  assert.equal(out.closed, 6);
  for (let v = 0; v < n; v += 1) if (closes[v] <= 30) assert.ok(out.visits[v] === 0 && out.weights[v] === 0, `node ${v}`);
  assert.ok(mean(out.weights, bottom) > mean(out.weights, top.filter((v) => closes[v] > 30)), "the bottom corridor carries the escape");
  assert.ok(out.route && out.route.nodes.some((v) => bottom.includes(v)), "the route goes round by the bottom");
  // Before the fire reaches it (time 0 is before nothing: every close is at 0), a later fire does not count.
  const later = Float64Array.from(closes, (c) => (c === 0 ? 100 : c));
  assert.equal(slimeEgress(graph, { exits, start, agents: 10, steps: 20, closes: later, time: 30, seed: 1, fire: "hard" }).closed, 0);
});

test("with the hard fire, a start it cuts off from every exit has no route", () => {
  const { graph, exits, start, id } = twoWays();
  const n = graph.nodes.length / 3;
  const closes = new Float64Array(n).fill(Infinity);
  // Close the room's whole right side, both corridors' mouths.
  for (let k = 0; k < 14; k += 1) {
    const v = id.get(`3,${k}`);
    if (v !== undefined) closes[v] = 0;
  }
  const out = slimeEgress(graph, { exits, start, agents: 20, steps: 50, closes, time: 0, seed: 1, fire: "hard" });
  assert.equal(out.route, null);
});

test("the trail repels near the fire: tubes keep off a corridor beside it even where it is open", () => {
  // Two exits, one at each end of a corridor 4 nodes wide; the fire beside the left half.
  const built = floor(40, 4, { exitsAt: [[39, 1]] });
  const { graph, id } = built;
  const v0 = graph.nodes.length / 3;
  const nodes = Array.from(graph.nodes);
  nodes.push(-0.5, 0.75, 0.5);
  const kinds = [...graph.kinds, KIND_DOOR];
  const edges = [...graph.edges, v0, id.get("0,1")];
  const g = { ...graph, nodes: Float32Array.from(nodes), kinds: Uint8Array.from(kinds), edges: Int32Array.from(edges) };
  const exits = [...built.exits, v0];
  const n = g.nodes.length / 3;
  // A fire on two nodes at the corridor's left quarter, on its far side.
  const closes = new Float64Array(n).fill(Infinity);
  closes[id.get("10,3")] = 0;
  closes[id.get("11,3")] = 0;
  const near = [id.get("10,2"), id.get("11,2"), id.get("10,1"), id.get("11,1")];
  const without = slimeEgress(g, { exits, agents: 30, steps: 300, seed: 2 });
  const repelled = slimeEgress(g, { exits, agents: 30, steps: 300, closes, time: 1, seed: 2, repel: 20 });
  assert.ok(mean(repelled.visits, near) < mean(without.visits, near), `${mean(repelled.visits, near)} against ${mean(without.visits, near)}`);
});

test("one seed, one run; another seed, another", () => {
  const { graph, exits, start } = twoWays();
  const a = slimeEgress(graph, { exits, start, agents: 30, steps: 80, seed: 3 });
  const b = slimeEgress(graph, { exits, start, agents: 30, steps: 80, seed: 3 });
  const c = slimeEgress(graph, { exits, start, agents: 30, steps: 80, seed: 4 });
  assert.deepEqual(Array.from(a.weights), Array.from(b.weights));
  assert.notDeepEqual(Array.from(a.visits), Array.from(c.visits));
});

test("the defaults are Jones's agent with the field weighed in metres", () => {
  assert.equal(SLIME_DEFAULTS.so, 3);
  assert.ok(SLIME_DEFAULTS.damp > 0 && SLIME_DEFAULTS.damp < 1);
  const { graph, exits } = twoWays();
  // With no exits there is nowhere to go: no agents, no weights.
  const none = slimeEgress(graph, { exits: [], agents: 10, steps: 5 });
  assert.equal(none.agents, 0);
  assert.equal(none.route, null);
  assert.ok(slimeEgress(graph, { exits, agents: 10, steps: 5 }).agents === 10);
});

test("the soft fire: still a way out when the fire fills the walls' way; the trail keeps out of it and fades there", () => {
  const { graph, exits, top, bottom, room, start, id } = twoWays();
  const n = graph.nodes.length / 3;
  // A fire that will close both corridors' mouths and most of the room, by 60 s.
  const closes = new Float64Array(n).fill(Infinity);
  for (let i = 0; i <= 5; i += 1) for (let k = 0; k < 14; k += 1) {
    const v = id.get(`${i},${k}`);
    if (v !== undefined && !(i === 1 && k === 7)) closes[v] = 20 + 2 * k;
  }
  // Hard, it walls off everything it ever closes: nothing to go by.
  const walled = slimeEgress(graph, { exits, start, agents: 30, steps: 200, closes, fire: "hard", seed: 1 });
  assert.equal(walled.route, null);
  // Soft, the slime still runs and finds the start a way out, at a cost.
  const soft = slimeEgress(graph, { exits, start, agents: 30, steps: 200, closes, seed: 1 });
  assert.equal(soft.fire, "soft");
  assert.ok(soft.exits[0]?.flux > 0, "agents reach the exit");
  assert.ok(soft.route, "a way out");
  // Little of its trail lies where the fire is.
  assert.ok(soft.fireShare < 0.25, `share of the trail in smoke ${soft.fireShare.toFixed(2)}`);
  // A fire on the top corridor alone, starting at 0: the soft slime moves its tubes to the bottom one.
  const topFire = new Float64Array(n).fill(Infinity);
  for (const v of top) topFire[v] = 0;
  const moved = slimeEgress(graph, { exits, places: room, agents: 30, steps: 300, closes: topFire, seed: 2 });
  const calm = slimeEgress(graph, { exits, places: room, agents: 30, steps: 300, seed: 2 });
  assert.ok(mean(moved.weights, bottom) > mean(calm.weights, bottom), "more on the bottom corridor with the fire on the top one");
  assert.ok(mean(moved.weights, top) < mean(calm.weights, top), "less on the top one");
});

test("the soft fire's strength rises over the lead before a place is untenable and stays full after", async () => {
  const { smokeAt } = await import("./slimeEgress.js");
  assert.equal(smokeAt(Infinity, 100, 30), 0);
  assert.equal(smokeAt(60, 20, 30), 0);
  assert.equal(smokeAt(60, 45, 30), 0.5);
  assert.equal(smokeAt(60, 60, 30), 1);
  assert.equal(smokeAt(60, 500, 30), 1);
  for (let t = 0; t < 100; t += 1) assert.ok(smokeAt(60, t + 1, 30) >= smokeAt(60, t, 30), "never weaker later");
});

test("the run's own smoke: thick smoke on a corridor holds the slime back from it, and sight shortens in it", () => {
  assert.equal(sensorReach(3, 30, 30), 3);
  assert.equal(sensorReach(3, 60, 30), 3);
  assert.equal(sensorReach(3, 20, 30), 2);
  assert.equal(sensorReach(3, 10, 30), 1);
  assert.equal(sensorReach(3, 0.5, 30), 1);
  const { graph, exits, top, bottom, room } = twoWays();
  const n = graph.nodes.length / 3;
  // Smoke on the top corridor from the start (visibility 2 m), clear elsewhere; nothing untenable.
  const K = 10, frameSeconds = 30;
  const vis = new Float32Array(K * n).fill(40);
  for (let k = 0; k < K; k += 1) for (const v of top) vis[k * n + v] = 2;
  const closes = new Float64Array(n).fill(Infinity);
  const smoky = slimeEgress(graph, { exits, places: room, agents: 30, steps: 300, closes, smokeFrames: { K, frameSeconds, vis }, seed: 2 });
  const calm = slimeEgress(graph, { exits, places: room, agents: 30, steps: 300, seed: 2 });
  assert.equal(smoky.smokeSource, "visibility");
  assert.ok(mean(smoky.weights, top) < mean(calm.weights, top), "less trail in the smoke");
  assert.ok(mean(smoky.weights, bottom) > mean(calm.weights, bottom), "more on the clear corridor");
  assert.equal(smoky.explored.length, n);
});
