/** The guided runs' choices on the graph (guided.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { GUIDED_HOLD_MS, GUIDED_RUNS, farthestFromExits, fireStageNext, guidedHeadline, guidedOrigin, guidedResultPlace, nodeAlong, stagesFor } from "./guided.js";

test("the start is the floor node most links from any exit, past blocked links", () => {
  // A path 0-1-2-3-4 with the exit at 0, and a door node 5 hanging off 4; node 6 joins 0 only by a blocked link.
  const graph = {
    nodes: new Float32Array(7 * 3),
    edges: Uint32Array.from([0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0, 6]),
    blocked: Uint8Array.from([0, 0, 0, 0, 0, 1]),
    kinds: Uint8Array.from([0, 0, 0, 0, 0, 2, 0]),
  };
  assert.equal(farthestFromExits(graph, [0]), 4, "node 5 is a door, node 6 cannot be reached");
  assert.equal(farthestFromExits(graph, [4]), 0);
  assert.equal(farthestFromExits(graph, []), -1);
});

test("a node part-way along a route, and the stages each preset passes through", () => {
  assert.equal(nodeAlong([10, 11, 12, 13, 14], 0.45), 12);
  assert.equal(nodeAlong([10, 11], 1), 11);
  assert.equal(nodeAlong([], 0.5), -1);
  for (const run of GUIDED_RUNS) {
    const ids = stagesFor(run).map((s) => s.id);
    assert.equal(ids[0], "open");
    assert.equal(ids.at(-1), "done");
    assert.equal(ids.includes("fire"), Boolean(run.fire));
    // The egress run maps every room instead of finding one route.
    assert.equal(ids.includes("map") && ids.includes("measure"), Boolean(run.egress));
    assert.equal(ids.includes("route"), !run.egress);
  }
  assert.deepEqual(
    stagesFor(GUIDED_RUNS.find((r) => r.egress)).map((s) => s.id),
    ["open", "graph", "map", "measure", "done"],
  );
});

test("a guided run lights its fire at the likeliest origin near the route, on the walkable graph", () => {
  // A corridor of 21 nodes 1 m apart along x (exit at 0), and an island of
  // two nodes at x = 5, z = 10 that reaches no exit.
  const nodes = [];
  for (let i = 0; i <= 20; i += 1) nodes.push(i, 0, 0);
  nodes.push(5, 0, 10, 6, 0, 10);
  const edges = [];
  for (let i = 0; i < 20; i += 1) edges.push(i, i + 1);
  edges.push(21, 22);
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: new Uint8Array(23), upAxis: "y" };
  const route = Array.from({ length: 21 }, (_, i) => 20 - i);
  const at = (x, z) => ({ position: [x, 0, z], scenario: { label: `at ${x}`, fireClass: "A" } });
  // The likeliest is on the island (no exit), the next beside the start
  // (too close), then two reachable ones: the nearer the route wins.
  const candidates = [at(5.2, 10), at(19.5, 0.2), at(12, 6), at(9, 1)];
  const pick = guidedOrigin(graph, { candidates, route, exits: [0], start: 20 });
  assert.equal(pick.candidate, candidates[3]);
  assert.equal(pick.node, 9);
  // Only beyond the top few, or none at all: a node part-way along the route.
  assert.deepEqual(guidedOrigin(graph, { candidates: [], route, exits: [0], start: 20 }), { node: nodeAlong(route, 0.45), candidate: null });
  assert.equal(guidedOrigin(graph, { candidates: [at(5.2, 10)], route, exits: [0], start: 20 }).candidate, null);
  // Never at the start or an exit.
  for (const c of [at(0, 0), at(20, 0)]) assert.equal(guidedOrigin(graph, { candidates: [c], route, exits: [0], start: 20 }).candidate, null);
});

test("a guided run's fire stage moves on when the fire ends by itself, and ends the run when it is stopped (state-machines.md SM-3)", () => {
  assert.equal(fireStageNext({ sawRunning: false, running: false, step: 0, stopped: false }), "wait");
  assert.equal(fireStageNext({ sawRunning: true, running: true, step: 12, stopped: false }), "wait");
  assert.equal(fireStageNext({ sawRunning: true, running: false, step: 40, stopped: false }), "advance");
  // Stopped: taken as ended, the plan stage computed the stopped run again from the start.
  assert.equal(fireStageNext({ sawRunning: true, running: false, step: 12, stopped: true }), "stopped");
  assert.equal(fireStageNext({ sawRunning: true, running: false, step: 0, stopped: true }), "wait");
});

test("a finished run leads with its numbers, from the result it ends on", () => {
  const run = (id) => GUIDED_RUNS.find((r) => r.id === id);
  assert.equal(guidedHeadline(run("way-out"), { route: { found: true, length: 20.63 } }), "Route 20.6 m, from the room furthest from an exit.");
  assert.equal(guidedHeadline(run("way-out"), { route: { found: false } }), null);
  // The route during the fire: its first change kept, and when it was clear again (the card's own words).
  assert.equal(guidedHeadline(run("fire-on-route"), { reroute: { found: true, step: 59, firstChange: 25, clearBy: 45 } }), "The route changed at step 25; clear again by step 45.");
  assert.equal(guidedHeadline(run("fire-on-route"), { reroute: { found: true, step: 59 } }), "The route stayed clear to step 59.");
  assert.equal(guidedHeadline(run("smoke-doors"), { combustion: { peak: 41.6, detectedAt: 23.2 } }), "Peak 42 kW; a smoke detector responds at 23 s.");
  assert.equal(guidedHeadline(run("smoke-doors"), { combustion: { peak: 41.6, detectedAt: null } }), "Peak 42 kW; no smoke detector responds.");
  assert.equal(guidedHeadline(run("vienna-gpu"), { plan: { out: 35.71, leave: 84.1, leaveAtLeast: false } }), "Out in 35.7 s by the earliest safe route; leave within 84.1 s.");
  assert.equal(guidedHeadline(run("vienna-gpu"), { plan: { out: null } }), "No safe route within the forecast.");
  assert.equal(
    guidedHeadline(run("egress-access"), { egress: { farthest: 61.24, deadEnds: 3, byChair: 40, onFoot: 52 } }),
    "Farthest walk 61.2 m; 3 dead-end corridors; a wheelchair reaches 40 of 52 rooms.",
  );
  assert.equal(guidedHeadline(run("egress-access"), {}), null);
});

test("a finished run's result is opened and brought into view where it is in the panel", () => {
  const run = (id) => GUIDED_RUNS.find((r) => r.id === id);
  assert.deepEqual(guidedResultPlace(run("way-out")), { tab: "route", sections: [], scroll: '[aria-label="Route result"]' });
  assert.deepEqual(guidedResultPlace(run("egress-access")).sections, ["egress-map", "accessibility"]);
  assert.equal(guidedResultPlace(run("smoke-doors")).scroll, '[aria-label="Flamescapes run"]');
  assert.equal(guidedResultPlace(run("fire-on-route")).scroll, '[aria-label="Route during the fire"]');
  assert.equal(guidedResultPlace(run("vienna-gpu")).tab, "simulate");
  // Held long enough to read a caption; the smoke run shows the smoke.
  assert.ok(GUIDED_HOLD_MS >= 2000 && GUIDED_HOLD_MS <= 3000);
  assert.equal(run("smoke-doors").fire.show, "smoke");
});
