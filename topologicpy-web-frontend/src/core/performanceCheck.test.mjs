/** The performance-based ("simulated fire") egress check (performanceCheck.js). */
import test from "node:test";
import assert from "node:assert/strict";
import { Z99, erf, lognormalCdf, lognormalFromPercentiles, performanceCheck } from "./performanceCheck.js";
import { smokeDetection } from "./forecast.js";

const close = (a, b, tol) => assert.ok(Math.abs(a - b) <= tol, `${a} vs ${b}`);

test("erf and the fitted log-normal return their known values and percentiles", () => {
  close(erf(0), 0, 1e-9);
  close(erf(1), 0.8427007929, 2e-7);
  close(erf(-1), -0.8427007929, 2e-7);
  close(erf(3), 0.9999779095, 2e-7);
  close(0.5 * (1 + erf(Z99 / Math.SQRT2)), 0.99, 1e-6);
  const d = lognormalFromPercentiles(30, 300);
  close(lognormalCdf(30, d), 0.01, 1e-6);
  close(lognormalCdf(300, d), 0.99, 1e-6);
  close(lognormalCdf(Math.sqrt(30 * 300), d), 0.5, 1e-9);
  assert.equal(lognormalCdf(0, d), 0);
  assert.equal(lognormalCdf(-5, d), 0);
  assert.equal(lognormalCdf(Infinity, d), 1);
});

/** A corridor of `n` nodes 1 m apart on one storey (y up, lifted 0.75 m), exit at node 0, and a separate pocket of two nodes. */
function corridor(n) {
  const nodes = [], edges = [];
  for (let x = 0; x < n; x += 1) nodes.push(x, 0.75, 0);
  for (let x = 1; x < n; x += 1) edges.push(x - 1, x);
  nodes.push(50, 0.75, 0, 51, 0.75, 0);
  edges.push(n, n + 1);
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: new Uint8Array(n + 2), upAxis: "y", agentHeight: 0.75, spacing: 1 };
}

test("a storey passes when every place can wait for detection, alarm and the 99th percentile, and fails otherwise", () => {
  const g = corridor(4);
  const opts = { levels: [0], exits: [0], horizon: 600, step: 1, detection: 30, alarm: 0, pre1: 30, pre99: 120 };
  // Required start: 30 + 120 = 150 s. Deadlines 400, 300, 200, 160 s: all pass.
  const ok = performanceCheck(g, Float64Array.from([400, 300, 200, 160, -Infinity, -Infinity]), opts);
  assert.equal(ok.start, 150);
  assert.equal(ok.status, "pass");
  const [s] = ok.storeys;
  assert.equal(s.judged, 4);
  assert.equal(s.unreached, 2, "the pocket reaches no exit and is not judged");
  close(s.worstMargin, 10, 1e-9);
  assert.ok(s.share > 0.99);
  // One place with 100 s fails; the share of occupants falls.
  const bad = performanceCheck(g, Float64Array.from([400, 300, 200, 100, -Infinity, -Infinity]), opts);
  assert.equal(bad.status, "fail");
  assert.equal(bad.storeys[0].fail, 1);
  assert.ok(bad.storeys[0].share < s.share);
  // No safe departure at all fails too.
  assert.equal(performanceCheck(g, Float64Array.from([400, 300, 200, -Infinity, 0, 0]), opts).status, "fail");
  // A margin on top raises the bar.
  assert.equal(performanceCheck(g, Float64Array.from([400, 300, 200, 160, 0, 0]), { ...opts, margin: 20 }).status, "fail");
});

test("a factor on RSET asks the margin to be at least factor - 1 times RSET, the walk included", () => {
  const g = corridor(4);
  // RSET less the walk 150 s; the walk is x s at node x (1 m/s). With factor
  // 2 node x needs L >= 300 + x: 300, 301, 302, 303 s.
  const opts = { levels: [0], exits: [0], horizon: 1200, step: 1, speed: 1, detection: 30, alarm: 0, pre1: 30, pre99: 120, factor: 2, margin: 30 };
  const ok = performanceCheck(g, Float64Array.from([400, 400, 400, 303, -Infinity, -Infinity]), opts);
  assert.equal(ok.status, "pass");
  assert.equal(ok.start, 300);
  // ASET / RSET at node 3: (303 + 3) / (150 + 3) = 2.
  close(ok.storeys[0].worstRatio, 2, 1e-12);
  close(ok.storeys[0].worstMargin, 153, 1e-12);
  assert.match(ok.storeys[0].text, /ASET at least 2 times RSET and at least 30 s over it/);
  const bad = performanceCheck(g, Float64Array.from([400, 400, 400, 302.5, -Infinity, -Infinity]), opts);
  assert.equal(bad.status, "fail");
  assert.equal(bad.storeys[0].fail, 1);
  // The absolute margin governs when the factor asks less: 1.1 times 153 s is under 150 + 30.
  const loose = { ...opts, factor: 1.1 };
  assert.equal(performanceCheck(g, Float64Array.from([400, 400, 400, 179, 0, 0]), loose).status, "fail");
  assert.equal(performanceCheck(g, Float64Array.from([400, 400, 400, 180, 0, 0]), loose).status, "pass");
});

test("detection from the frames is the end of the first frame whose smoke reaches the detector's optical density", () => {
  // 0.097 per metre is 13.4 m of visibility at C = 3.
  const frames = { n: 2, K: 3, frameSeconds: 10, vis: Float32Array.from([100, 100, 50, 20, 13, 100]) };
  assert.equal(smokeDetection(frames), 30);
  assert.equal(smokeDetection({ ...frames, vis: Float32Array.from([100, 100, 50, 20, 14, 100]) }), null);
  // A more sensitive detector (FDS's default 3.24 % per metre, OD 0.0143) responds at 91 m.
  assert.equal(smokeDetection(frames, { opticalDensity: 0.0143 }), 20);
});

test("a forecast that ends before the required start cannot tell for the places that stay safe to its end", () => {
  const g = corridor(3);
  // 120 s forecast, 1 s grid: a place safe throughout has the map's last time, 119 s or more.
  const opts = { levels: [0], exits: [0], horizon: 120, step: 1, detection: 30, alarm: 30, pre1: 60, pre99: 300 };
  const r = performanceCheck(g, Float64Array.from([120, 119.5, 119, -Infinity, -Infinity]), opts);
  assert.equal(r.status, "cannot tell");
  assert.equal(r.storeys[0].open, 3);
  // 360 s to start, the farthest place 2 m (1.7 s at 1.2 m/s) from the exit, and a 1 s grid.
  assert.equal(r.needHorizon, 363);
  assert.match(r.storeys[0].text, /plan one of at least 363 s/);
  // A place that becomes unsafe well within the forecast still fails.
  assert.equal(performanceCheck(g, Float64Array.from([120, 119.5, 40, 0, 0]), opts).status, "fail");
});
