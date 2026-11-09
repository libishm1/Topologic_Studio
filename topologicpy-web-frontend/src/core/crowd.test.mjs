// The egress animation's crowd (core/crowd.js): a floor-field walk to the
// nearest exit, timed by the SFPE hydraulic values.
import test from "node:test";
import assert from "node:assert/strict";

import { FREE_SPEED, SFPE, simulateCrowd } from "./crowd.js";
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
  const out = simulateCrowd(graph, { exits, occupantArea: 1 });
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
