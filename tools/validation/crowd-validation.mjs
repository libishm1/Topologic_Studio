// Verification and validation of the people-leaving model (core/crowd.js)
// against the RiMEA guideline's tests (version 4.1.1, 11.09.2025, CC BY-ND
// 4.0), NIST Technical Note 1822's suggested tests (Ronchi et al. 2013,
// doi:10.6028/NIST.TN.1822) and open experimental data (RiMEA Test 16's
// percentiles of single-file fundamental diagrams; bottleneck flows).
//
//   node tools/validation/crowd-validation.mjs [--quick] [--passage-flow] [--out FILE.json] [--date YYYY-MM-DD] [--no-write]
//
// --passage-flow runs every case with the corridors' and stairs' flow capacity
// (crowd.js `passageFlow`, off by default as in the app).
// --no-write records nothing and ends with a line "SUMMARY {json}"; the test
// suite runs it so (src/core/crowdValidation.test.mjs).
//
// Each test builds its geometry as a lattice graph (0.5 m nodes, eight
// neighbours, no corner cutting), runs simulateCrowd as the app does, and
// reads the walks back from the segments. A fail is a finding about the
// model, recorded as measured; nothing is tuned to pass.
// Writes benchmark/results/<--out> (default 2026-11-11-crowd-validation.json).
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as Crowd from "../../topologicpy-web-frontend/src/core/crowd.js";
const { FREE_SPEED, SFPE, simulateCrowd } = Crowd;
/** The populations a group's speeds are drawn from (crowd.js, since 2026-11-24). */
const POPULATIONS = Crowd.POPULATIONS ?? null;
/** The model's ascent share, where it has one (before the 2026-11-11 fixes it had none: up as down). */
const STAIR_UP = Crowd.STAIR_UP ?? 1;
import { doorCapacity } from "../../topologicpy-web-frontend/src/core/performanceCheck.js";
import { KIND_DOOR, KIND_FLOOR, KIND_STAIR } from "../../topologicpy-web-frontend/src/core/lattice.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const QUICK = process.argv.includes("--quick");
/** With --passage-flow every run has the corridors' and stairs' flow capacity (crowd.js passageFlow; off by default, as in the app). */
const PASSAGE = process.argv.includes("--passage-flow");
const BASE = PASSAGE ? { passageFlow: true } : {};
const S = 0.5;
const H = 0.75;

// ------------------------------------------------------------------ geometry

/**
 * A lattice graph. cells: [i, j, kind?, y?] (y: floor height at the cell);
 * pos(i, j, y) gives the node's floor point (default x = i S, z = j S).
 * Eight neighbours; a diagonal only where both axis neighbours exist.
 */
function lattice(cells, { pos = (i, j, y) => [i * S, y, j * S], extra = [] } = {}) {
  const index = new Map();
  const nodes = [], kinds = [], edges = [];
  for (const [i, j, kind = KIND_FLOOR, y = 0] of cells) {
    const key = `${i},${j}`;
    if (index.has(key)) continue;
    index.set(key, nodes.length / 3);
    const [x, h, z] = pos(i, j, y);
    nodes.push(x, h + H, z);
    kinds.push(kind);
  }
  for (const [key, a] of index) {
    const [i, j] = key.split(",").map(Number);
    for (const [di, dj] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
      const b = index.get(`${i + di},${j + dj}`);
      if (b === undefined) continue;
      if (di && dj && (!index.has(`${i + di},${j}`) || !index.has(`${i},${j + dj}`))) continue;
      edges.push(a, b);
    }
  }
  for (const [p, q] of extra) edges.push(index.get(p), index.get(q));
  const graph = { nodes: Float32Array.from(nodes), edges: Int32Array.from(edges), kinds: Uint8Array.from(kinds), spacing: S, agentHeight: H, upAxis: "y" };
  return { graph, at: (i, j) => index.get(`${i},${j}`), index };
}

const rect = (i0, i1, j0, j1, kind = KIND_FLOOR, y = 0) => {
  const out = [];
  for (let i = i0; i <= i1; i += 1) for (let j = j0; j <= j1; j += 1) out.push([i, j, kind, y]);
  return out;
};

/** Places people may start on, as crowd.js counts them: not doors, stairs, lifts or exits. */
const placesOf = (graph, exits) => {
  const isExit = new Set(exits);
  let k = 0;
  for (let v = 0; v < graph.kinds.length; v += 1) if (graph.kinds[v] === KIND_FLOOR && !isExit.has(v)) k += 1;
  return k;
};

/** Options that put exactly `people` on the floor (spread at random over its places). */
const crowdOf = (graph, exits, people) => ({ occupantArea: (placesOf(graph, exits) * S * S) / people, maxPeople: people });

// ------------------------------------------------------------------ reading the result

/** Each person's segments {t0, t1, a, b}, the walk out only (not the walk on to the gathering place). */
function walks(out, n) {
  return Array.from({ length: out.people }, (_, i) => {
    const list = [];
    for (let j = 0; j < out.count[i]; j += 1) {
      const q = (out.first[i] + j) * 4;
      const s = { t0: out.segments[q], t1: out.segments[q + 1], a: out.segments[q + 2], b: out.segments[q + 3] };
      if (s.b < n && s.a !== s.b) list.push(s);
    }
    return list;
  });
}

const dist3 = (g, a, b) => Math.hypot(g.nodes[b * 3] - g.nodes[a * 3], g.nodes[b * 3 + 1] - g.nodes[a * 3 + 1], g.nodes[b * 3 + 2] - g.nodes[a * 3 + 2]);
const startNode = (out, i) => out.segments[out.first[i] * 4 + 2];

/** Where a person is at t (x, y, z), interpolated along their segments; null once out. */
function positionAt(g, segs, t, outAt) {
  if (t >= outAt) return null;
  let p = segs.length ? segs[0].a : -1;
  for (const s of segs) {
    if (t < s.t0) break;
    if (t <= s.t1) {
      const f = s.t1 > s.t0 ? (t - s.t0) / (s.t1 - s.t0) : 1;
      return [0, 1, 2].map((c) => g.nodes[s.a * 3 + c] + f * (g.nodes[s.b * 3 + c] - g.nodes[s.a * 3 + c]));
    }
    p = s.b;
  }
  return p >= 0 ? [g.nodes[p * 3], g.nodes[p * 3 + 1], g.nodes[p * 3 + 2]] : null;
}

const mean = (xs) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : NaN);
const round = (x, d = 3) => (Number.isFinite(x) ? Math.round(x * 10 ** d) / 10 ** d : x);
const quantile = (xs, q) => {
  const s = xs.slice().sort((a, b) => a - b);
  if (!s.length) return NaN;
  const k = (s.length - 1) * q, lo = Math.floor(k), hi = Math.ceil(k);
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
};

/** One-sample Kolmogorov-Smirnov against a CDF: D and the asymptotic p-value (Stephens' correction). */
function ks(samples, cdf) {
  const s = samples.slice().sort((a, b) => a - b);
  const n = s.length;
  let d = 0;
  s.forEach((x, k) => {
    const f = cdf(x);
    d = Math.max(d, (k + 1) / n - f, f - k / n);
  });
  const lam = (Math.sqrt(n) + 0.12 + 0.11 / Math.sqrt(n)) * d;
  let p = 0;
  for (let k = 1; k <= 100; k += 1) p += 2 * (-1) ** (k - 1) * Math.exp(-2 * k * k * lam * lam);
  return { D: d, p: Math.min(1, Math.max(0, p)), n };
}

const results = [];
const record = (r) => {
  results.push(r);
  const mark = r.pass === true ? "PASS" : r.pass === false ? "FAIL" : "DOC ";
  console.log(`${mark}  ${r.id}  ${r.name}: ${r.summary}`);
};

const seeds = (k) => Array.from({ length: QUICK ? Math.min(2, k) : k }, (_, i) => i + 1);

// ------------------------------------------------------------------ tests

/** RiMEA Test 1 and NIST Verif.2.1: one person keeps their speed along a corridor 2 m wide and 40 m long, at 0 and 45 degrees to the lattice. */
function corridorSpeed() {
  const build = (angle) => {
    if (angle === 0) {
      const { graph, at } = lattice([...rect(0, 80, 0, 4), [81, 2, KIND_DOOR]]);
      return { graph, exits: [at(81, 2)] };
    }
    // Along the lattice's diagonal: |i - j| <= 2 is 2.1 m wide; i + j up to 113 is 40 m long.
    const cells = [];
    for (let i = 0; i <= 60; i += 1) for (let j = 0; j <= 60; j += 1) if (Math.abs(i - j) <= 2 && i + j <= 113) cells.push([i, j]);
    cells.push([57, 57, KIND_DOOR]);
    const { graph, at } = lattice(cells);
    return { graph, exits: [at(57, 57)] };
  };
  for (const { id, source, speed, expected, band } of [
    { id: "RiMEA-1", source: "RiMEA 4.1.1 Test 1", speed: 1.33, expected: 40 / 1.33, band: [26, 34] },
    { id: "NIST-Verif.2.1", source: "NIST TN 1822 Verif.2.1 (IMO Test 1)", speed: 1.0, expected: 40, band: [40 * 0.95, 40 * 1.05] },
  ]) {
    for (const angle of [0, 45]) {
      const { graph, exits } = build(angle);
      const n = graph.nodes.length / 3;
      const t40 = [], realised = [], segSpeed = [];
      for (const seed of seeds(10)) {
        const out = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 1), speed, spread: 0, start: [0, 0], seed, choice: "nearest" });
        const segs = walks(out, n)[0];
        if (!segs?.length) continue;
        const L = segs.reduce((s, x) => s + dist3(graph, x.a, x.b), 0);
        const T = out.outAt[0] - segs[0].t0;
        if (L < 5) continue;
        t40.push((40 * T) / L);
        realised.push(L / T);
        segSpeed.push(mean(segs.map((x) => dist3(graph, x.a, x.b) / (x.t1 - x.t0))));
      }
      const m = mean(t40);
      record({
        id: `${id}${angle ? "-45deg" : ""}`,
        source,
        name: `speed in a 2 m x 40 m corridor at ${speed} m/s, ${angle} degrees to the lattice`,
        criterion: `time for 40 m within [${round(band[0], 1)}, ${round(band[1], 1)}] s (expected ${round(expected, 2)} s)`,
        measured: { time40_mean_s: round(m, 2), time40_range_s: [round(Math.min(...t40), 2), round(Math.max(...t40), 2)], realised_speed_mps: round(mean(realised), 3), segment_speed_mps: round(mean(segSpeed), 3), runs: t40.length },
        pass: t40.length > 0 && t40.every((t) => t >= band[0] && t <= band[1]),
        summary: `40 m in ${round(m, 2)} s (expected ${round(expected, 2)}), realised ${round(mean(realised), 3)} m/s, while moving ${round(mean(segSpeed), 3)} m/s`,
      });
    }
  }
}

/** RiMEA Tests 2 and 3, NIST Verif.2.2, RiMEA Test 13 (direction): one person on a stair 2 m wide, 10 m along the slope (7 in risers, 11 in treads). */
function stairSpeed() {
  const theta = Math.atan(7 / 11);
  const hx = S * Math.cos(theta), dz = S * Math.sin(theta);
  const build = (up) => {
    // A landing of 5 x 5 nodes, then 20 stair columns (10 m along the slope), the exit door at the far end.
    const cells = [...rect(0, 4, 0, 4)];
    for (let c = 0; c < 20; c += 1) for (let j = 0; j <= 4; j += 1) cells.push([5 + c, j, KIND_STAIR, (up ? 1 : -1) * (c + 1) * dz]);
    cells.push([25, 2, KIND_DOOR, (up ? 1 : -1) * 21 * dz]);
    const pos = (i, j, y) => [i <= 4 ? i * S : 4 * S + (i - 4) * hx, y, j * S];
    const { graph, at } = lattice(cells, { pos });
    return { graph, exits: [at(25, 2)] };
  };
  const measure = (up, speed) => {
    const { graph, exits } = build(up);
    const n = graph.nodes.length / 3;
    const t10 = [], moving = [];
    for (const seed of seeds(10)) {
      const out = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 1), speed, spread: 0, start: [0, 0], seed, choice: "nearest", stairGap: 0 });
      const segs = walks(out, n)[0];
      const on = segs.filter((s) => graph.kinds[s.a] === KIND_STAIR || graph.kinds[s.b] === KIND_STAIR);
      if (!on.length) continue;
      const L = on.reduce((s, x) => s + dist3(graph, x.a, x.b), 0);
      const T = on.at(-1).t1 - on[0].t0;
      t10.push((10 * T) / L);
      moving.push(mean(on.map((x) => dist3(graph, x.a, x.b) / (x.t1 - x.t0))));
    }
    return { t10: mean(t10), range: [Math.min(...t10), Math.max(...t10)], moving: mean(moving), runs: t10.length };
  };
  // RiMEA 2 and 3: the model's own stair speed for the default walker, SFPE's 0.85 x 1.08 m/s.
  const vStair = FREE_SPEED * (SFPE.kStair / SFPE.k);
  for (const up of [true, false]) {
    const r = measure(up, FREE_SPEED);
    const expected = 10 / (vStair * (up ? STAIR_UP : 1));
    const band = [expected * (26 / 30.08), expected * (34 / 30.08)];
    record({
      id: up ? "RiMEA-2" : "RiMEA-3",
      source: `RiMEA 4.1.1 Test ${up ? 2 : 3}`,
      name: `speed ${up ? "up" : "down"} a stair 2 m wide, 10 m along the slope`,
      criterion: `time for 10 m within Test 1's tolerance scaled, [${round(band[0], 2)}, ${round(band[1], 2)}] s, at the model's stair speed ${round(vStair * (up ? STAIR_UP : 1), 3)} m/s ${up ? "up" : "down"} (expected ${round(expected, 2)} s)`,
      measured: { time10_mean_s: round(r.t10, 2), time10_range_s: r.range.map((x) => round(x, 2)), moving_speed_mps: round(r.moving, 3), runs: r.runs },
      pass: r.runs > 0 && r.range[0] >= band[0] && r.range[1] <= band[1],
      summary: `10 m in ${round(r.t10, 2)} s (expected ${round(expected, 2)}), moving at ${round(r.moving, 3)} m/s`,
      up: r,
    });
  }
  // NIST Verif.2.2: 1 m/s along the stair: the walker whose stair speed is 1 m/s.
  const nist = measure(false, 1.0 / (SFPE.kStair / SFPE.k));
  record({
    id: "NIST-Verif.2.2",
    source: "NIST TN 1822 Verif.2.2 (IMO Tests 2 and 3)",
    name: "1 m/s down a stair, 100 m along the incline (10 m run, scaled)",
    criterion: "100 s for 100 m, within 5 %",
    measured: { time100_s: round(nist.t10 * 10, 1), moving_speed_mps: round(nist.moving, 3), runs: nist.runs },
    pass: Math.abs(nist.t10 * 10 - 100) <= 5,
    summary: `100 m in ${round(nist.t10 * 10, 1)} s, moving at ${round(nist.moving, 3)} m/s`,
  });
  // RiMEA 13: down faster than up. The model's stair speed does not depend on the direction.
  const upR = measure(true, FREE_SPEED), downR = measure(false, FREE_SPEED);
  record({
    id: "RiMEA-13",
    source: "RiMEA 4.1.1 Test 13 (fundamental diagram on stairs: down faster than up)",
    name: "walking speeds down a stair higher than up",
    criterion: "speed down > speed up (and within the spread of Fig. 16)",
    measured: { up_mps: round(upR.moving, 3), down_mps: round(downR.moving, 3) },
    pass: downR.moving > upR.moving + 1e-6,
    summary: `up ${round(upR.moving, 3)} m/s, down ${round(downR.moving, 3)} m/s${Crowd.STAIR_UP ? ` (ascent at STAIR_UP ${Crowd.STAIR_UP} of kStair)` : " (SFPE kStair both ways)"}`,
  });
}

/** RiMEA Test 6, NIST Verif.2.3: twenty persons round a left-hand corner without passing through walls. */
function corner() {
  // Horizontal leg x 0..12, z 0..2; vertical leg x 10..12, z 2..12; the exit at its top.
  const cells = [...rect(0, 24, 0, 4), ...rect(20, 24, 5, 24), [22, 25, KIND_DOOR]];
  const { graph, at } = lattice(cells);
  const exits = [at(22, 25)];
  const n = graph.nodes.length / 3;
  const inside = (x, z) => (x >= -1e-6 && x <= 12 + 1e-6 && z >= -1e-6 && z <= 2 + 1e-6) || (x >= 10 - 1e-6 && x <= 12 + 1e-6 && z >= -1e-6 && z <= 12.5 + 1e-6);
  let bad = 0, out = 0, people = 0;
  for (const seed of seeds(5)) {
    const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 20), speed: 1.0, spread: 0, start: [0, 0], seed });
    people += r.people;
    out += r.people - r.inside;
    for (const segs of walks(r, n)) {
      for (const s of segs) {
        for (const f of [0, 0.25, 0.5, 0.75, 1]) {
          const x = graph.nodes[s.a * 3] + f * (graph.nodes[s.b * 3] - graph.nodes[s.a * 3]);
          const z = graph.nodes[s.a * 3 + 2] + f * (graph.nodes[s.b * 3 + 2] - graph.nodes[s.a * 3 + 2]);
          if (!inside(x, z)) bad += 1;
        }
      }
    }
  }
  record({
    id: "RiMEA-6",
    source: "RiMEA 4.1.1 Test 6; NIST TN 1822 Verif.2.3 (IMO Test 6)",
    name: "twenty persons round a corner",
    criterion: "all go round the corner without passing through walls",
    measured: { people, out, points_outside_walls: bad },
    pass: bad === 0 && out === people,
    summary: `${out} of ${people} out, ${bad} points of their walks outside the walls`,
    deviation: "persons start at random over the whole corridor, not in the 6 m x 2 m start area",
  });
  if (!POPULATIONS) return;
  // As written: the twenty start in the 6 m x 2 m area at the corridor's far end.
  const area = rect(0, 12, 0, 4).map(([i, j]) => at(i, j));
  let bad2 = 0, out2 = 0, people2 = 0, outsideArea = 0;
  for (const seed of seeds(5)) {
    const r = simulateCrowd(graph, { ...BASE, exits, groups: [{ nodes: area, count: 20 }], speed: 1.0, spread: 0, start: [0, 0], seed });
    people2 += r.people;
    out2 += r.people - r.inside;
    for (let i = 0; i < r.people; i += 1) if (!area.includes(startNode(r, i))) outsideArea += 1;
    for (const segs of walks(r, n)) {
      for (const sg of segs) {
        for (const f of [0, 0.25, 0.5, 0.75, 1]) {
          const x = graph.nodes[sg.a * 3] + f * (graph.nodes[sg.b * 3] - graph.nodes[sg.a * 3]);
          const z = graph.nodes[sg.a * 3 + 2] + f * (graph.nodes[sg.b * 3 + 2] - graph.nodes[sg.a * 3 + 2]);
          if (!inside(x, z)) bad2 += 1;
        }
      }
    }
  }
  record({
    id: "RiMEA-6-start-area",
    source: "RiMEA 4.1.1 Test 6; NIST TN 1822 Verif.2.3 (IMO Test 6)",
    name: "twenty persons round a corner, from the 6 m x 2 m start area",
    criterion: "all start in the area and go round the corner without passing through walls",
    measured: { people: people2, out: out2, started_outside_area: outsideArea, points_outside_walls: bad2 },
    pass: bad2 === 0 && out2 === people2 && outsideArea === 0 && people2 === 20 * seeds(5).length,
    summary: `${out2} of ${people2} out, ${outsideArea} started outside the area, ${bad2} points of their walks outside the walls`,
  });
}

/** RiMEA Test 5, NIST Verif.1.1: pre-movement times uniform between 10 and 100 s. */
function premovement() {
  const { graph, at } = lattice([...rect(0, 16, 0, 10), [-1, 5, KIND_DOOR]]);
  const exits = [at(-1, 5)];
  const n = graph.nodes.length / 3;
  const first = [];
  for (const seed of seeds(10)) {
    const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 10), start: [10, 100], seed });
    for (const segs of walks(r, n)) if (segs.length) first.push(segs[0].t0);
  }
  const test = ks(first, (x) => Math.min(1, Math.max(0, (x - 10) / 90)));
  record({
    id: "RiMEA-5",
    source: "RiMEA 4.1.1 Test 5; NIST TN 1822 Verif.1.1 (IMO Test 5)",
    name: "pre-movement uniform 10 to 100 s, ten persons in an 8 m x 5 m room",
    criterion: "each starts at an appropriate time: first moves within [10, 100] s, and uniform (KS p > 0.05)",
    measured: { samples: first.length, earliest_s: round(Math.min(...first), 2), latest_s: round(Math.max(...first), 2), ks_D: round(test.D, 3), ks_p: round(test.p, 3) },
    pass: Math.min(...first) >= 10 - 1e-9 && Math.max(...first) <= 100.2 && test.p > 0.05,
    summary: `first moves ${round(Math.min(...first), 1)} to ${round(Math.max(...first), 1)} s, KS p ${round(test.p, 3)} (crowd.js 'start' is a stagger; the performance check models pre-movement)`,
  });
}

/** The normal distribution's CDF (Abramowitz and Stegun 7.1.26 for erf). */
function normalCdf(x, mean, sd) {
  const z = (x - mean) / (sd * Math.SQRT2), t = 1 / (1 + 0.3275911 * Math.abs(z));
  const erf = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * (1 + (z >= 0 ? erf : -erf));
}

/** RiMEA Test 7, NIST Verif.2.4, IMO Test 7: the speeds of a named population are those asked for. */
function populations() {
  if (!POPULATIONS) return;
  const { graph, at } = lattice([...rect(0, 60, 0, 60), [-1, 30, KIND_DOOR]]);
  const exits = [at(-1, 30)];
  const n = graph.nodes.length / 3;
  const floor = rect(0, 60, 0, 60).map(([i, j]) => at(i, j));
  const cases = [
    {
      id: "RiMEA-7-weidmann", key: "weidmann", name: "400 persons of the Weidmann population (normal, mean 1.34 m/s, sd 0.26, cut off at 0.6 and 2.0)",
      cdf: (x) => {
        const p = POPULATIONS.weidmann, lo = normalCdf(p.min, p.mean, p.sd), hi = normalCdf(p.max, p.mean, p.sd);
        return Math.min(1, Math.max(0, (normalCdf(x, p.mean, p.sd) - lo) / (hi - lo)));
      },
      source: "RiMEA 4.1.1 Test 7; NIST TN 1822 Verif.2.4",
    },
    {
      id: "IMO-7", key: "imoMales30to50", name: "400 males of 30 to 50 years (uniform 0.97 to 1.62 m/s)",
      cdf: (x) => Math.min(1, Math.max(0, (x - 0.97) / (1.62 - 0.97))),
      source: "IMO MSC.1/Circ.1533 Test 7",
    },
  ];
  for (const c of cases) {
    const moving = [];
    // Eight runs in the quick mode too: 400 speeds, so the test of the distribution does not turn on a small sample.
    for (const seed of [1, 2, 3, 4, 5, 6, 7, 8]) {
      const r = simulateCrowd(graph, { ...BASE, exits, groups: [{ nodes: floor, count: 50, speed: POPULATIONS[c.key] }], seed, start: [0, 0] });
      for (const segs of walks(r, n)) {
        const first = segs.slice(0, 8);
        if (first.length < 4) continue;
        moving.push(quantile(first.map((sg) => dist3(graph, sg.a, sg.b) / (sg.t1 - sg.t0)), 0.5));
      }
    }
    const test = ks(moving, c.cdf);
    record({
      id: c.id,
      source: c.source,
      name: c.name,
      criterion: "unimpeded speeds consistent with the population assigned (KS p > 0.05)",
      measured: { samples: moving.length, moving_min_median_max: [round(Math.min(...moving), 3), round(quantile(moving, 0.5), 3), round(Math.max(...moving), 3)], ks_D: round(test.D, 3), ks_p: round(test.p, 4) },
      pass: test.p > 0.05,
      summary: `moving ${round(Math.min(...moving), 3)} to ${round(Math.max(...moving), 3)} m/s (median ${round(quantile(moving, 0.5), 3)}); KS p ${round(test.p, 4)}`,
    });
  }
}

/** RiMEA Test 7, NIST Verif.2.4: the speeds assigned are those asked for. */
function speedDistribution() {
  const { graph, at } = lattice([...rect(0, 60, 0, 60), [-1, 30, KIND_DOOR]]);
  const exits = [at(-1, 30)];
  const n = graph.nodes.length / 3;
  const spread = 0.25;
  const lo = FREE_SPEED * (1 - spread), hi = FREE_SPEED * (1 + spread);
  const moving = [], realised = [];
  for (const seed of seeds(4)) {
    const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 50), seed, start: [0, 0] });
    for (const segs of walks(r, n)) {
      const first = segs.slice(0, 8);
      if (first.length < 4) continue;
      moving.push(quantile(first.map((s) => dist3(graph, s.a, s.b) / (s.t1 - s.t0)), 0.5));
      const L = first.reduce((s, x) => s + dist3(graph, x.a, x.b), 0);
      realised.push(L / (first.at(-1).t1 - first[0].t0));
    }
  }
  const test = ks(moving, (x) => Math.min(1, Math.max(0, (x - lo) / (hi - lo))));
  record({
    id: "RiMEA-7",
    source: "RiMEA 4.1.1 Test 7; NIST TN 1822 Verif.2.4 (IMO Test 7)",
    name: "50 persons' unimpeded speeds against the distribution assigned (uniform 0.89 to 1.49 m/s)",
    criterion: "speeds consistent with the distribution assigned (KS p > 0.05)",
    measured: {
      samples: moving.length,
      assigned: [round(lo, 3), round(hi, 3)],
      moving_min_median_max: [round(Math.min(...moving), 3), round(quantile(moving, 0.5), 3), round(Math.max(...moving), 3)],
      realised_median: round(quantile(realised, 0.5), 3),
      ks_D: round(test.D, 3),
      ks_p: round(test.p, 4),
    },
    pass: test.p > 0.05,
    summary: `moving ${round(Math.min(...moving), 3)} to ${round(Math.max(...moving), 3)} m/s (median ${round(quantile(moving, 0.5), 3)}) against ${round(lo, 3)} to ${round(hi, 3)}; KS p ${round(test.p, 4)}`,
    deviation: "crowd.js assigns a uniform spread about one speed; RiMEA's population (Weidmann's speeds by age) cannot be set",
  });
}

/** Exit flow: times each person stepped out, sorted. */
const exitTimes = (r) => Array.from(r.outAt).filter(Number.isFinite).sort((a, b) => a - b);

/** NIST Verif.5.2 (IMO Test 4): 100 persons out of an 8 m x 5 m room through a 1 m exit; the flow kept under 1.33 persons/s per metre. */
function maxFlow() {
  const { graph, at } = lattice([...rect(0, 16, 0, 10), [-1, 5, KIND_DOOR]]);
  const exits = [at(-1, 5)];
  const flows = [], peaks = [];
  for (const seed of seeds(5)) {
    const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, 100), doorWidth: 1.0, start: [0, 0], seed });
    const t = exitTimes(r);
    flows.push((t.length - 1) / (t.at(-1) - t[0]));
    let peak = 0;
    for (let k = 0; k < t.length; k += 1) {
      let m = k;
      while (m + 1 < t.length && t[m + 1] - t[k] <= 10) m += 1;
      peak = Math.max(peak, (m - k) / 10);
    }
    peaks.push(peak);
  }
  record({
    id: "NIST-Verif.5.2",
    source: "NIST TN 1822 Verif.5.2 (IMO Test 4; MSC/Circ.1238's 1.33 persons/s/m)",
    name: "100 persons through a 1 m exit: the flow never above 1.33 persons/s",
    criterion: "flow at the exit over the whole period, and in any 10 s, at most 1.33 persons/s",
    measured: { mean_flow_ps: round(mean(flows), 3), peak_10s_flow_ps: round(Math.max(...peaks), 3), door_capacity_ps: round(doorCapacity(1.0), 3) },
    pass: Math.max(...peaks) <= 1.33,
    summary: `mean ${round(mean(flows), 3)} persons/s, peak 10 s ${round(Math.max(...peaks), 3)} (the door's SFPE capacity ${round(doorCapacity(1.0), 3)})`,
  });
}

/** RiMEA Test 9: 1,000 persons leave a 30 m x 20 m hall by four 1 m doors, then by two; about twice the time. */
function largeRoom() {
  const cells = [...rect(0, 60, 0, 40), [9, -1, KIND_DOOR], [51, -1, KIND_DOOR], [9, 41, KIND_DOOR], [51, 41, KIND_DOOR]];
  const { graph, at } = lattice(cells);
  const four = [at(9, -1), at(51, -1), at(9, 41), at(51, 41)];
  const two = [at(9, -1), at(51, -1)];
  const time = (exits) =>
    mean(
      seeds(3).map((seed) => {
        const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, four, QUICK ? 300 : 1000), doorWidth: 1.0, start: [0, 0], seed, horizon: 3000 });
        return r.lastOut;
      }),
    );
  const t4 = time(four), t2 = time(two);
  record({
    id: "RiMEA-9",
    source: "RiMEA 4.1.1 Test 9",
    name: "1,000 persons leave a hall by four doors, then by two",
    criterion: "about twice the time with two doors (documentation, not an exclusion criterion: footnote 8)",
    measured: { four_doors_s: round(t4, 1), two_doors_s: round(t2, 1), ratio: round(t2 / t4, 3) },
    pass: null,
    summary: `four doors ${round(t4, 1)} s, two ${round(t2, 1)} s: ${round(t2 / t4, 2)} times`,
    deviation: "persons spread over the whole hall rather than the 26 m x 16 m area; walking speeds crowd.js's uniform spread, not RiMEA's population",
  });
}

/** The bottleneck geometry of RiMEA Test 12: room 1 (10 m x 10 m), a 1 m opening (one door node) or a corridor, room 2. */
function bottleneckGraph({ goal = 10, long = 0, doorEnds = false } = {}) {
  const cells = [...rect(0, 20, 0, 20)];
  let first = 22;
  const door = [21, 10];
  if (long > 0) {
    // A corridor two nodes wide (z 5 and 5.5) and `long` m long, a door node at its ends when asked.
    const len = Math.round(long / S);
    for (let c = 0; c < len; c += 1) for (const j of [10, 11]) cells.push([21 + c, j, doorEnds && (c === 0 || c === len - 1) && j === 10 ? KIND_DOOR : KIND_FLOOR]);
    if (doorEnds) {
      // Only the door node row passes at the ends.
      for (const c of [0, len - 1]) {
        const k = cells.findIndex((x) => x[0] === 21 + c && x[1] === 11);
        cells.splice(k, 1);
      }
    }
    first = 21 + len;
  } else cells.push([door[0], door[1], KIND_DOOR]);
  const goalCol = first + Math.max(0, Math.round(goal / S) - 1);
  cells.push(...rect(first, goalCol - 1, 0, 20));
  for (let j = 0; j <= 20; j += 1) cells.push([goalCol, j, KIND_DOOR]);
  const { graph, at } = lattice(cells);
  const exits = [];
  for (let j = 0; j <= 20; j += 1) exits.push(at(goalCol, j));
  const room1 = new Set();
  for (let i = 0; i <= 20; i += 1) for (let j = 0; j <= 20; j += 1) room1.add(at(i, j));
  return { graph, exits, room1, door: long > 0 ? null : at(door[0], door[1]), lastRoom1Col: 20, at };
}

/**
 * Room 1's evacuation: when the last who started there left it, the flow out
 * of it, and (for the evidence) how long the steps on to the node before the
 * door and on to the door took, and how long between steps on to the door.
 */
function room1Out(g, r, room1) {
  const n = g.graph.nodes.length / 3;
  const ws = walks(r, n);
  const leave = [];
  const approach = g.door != null ? g.at(20, 10) : -1;
  const ontoApproach = [], ontoDoor = [];
  ws.forEach((segs, i) => {
    for (const s of segs) {
      if (s.b === approach) ontoApproach.push(s.t1 - s.t0);
      if (s.b === g.door) ontoDoor.push(s.t0);
    }
    if (!room1.has(startNode(r, i))) return;
    const s = segs.find((x) => room1.has(x.a) && !room1.has(x.b));
    if (s) leave.push(s.t1);
  });
  leave.sort((a, b) => a - b);
  ontoDoor.sort((a, b) => a - b);
  const k0 = Math.min(10, Math.floor(leave.length / 5)), k1 = leave.length - 1 - k0;
  const flow = k1 > k0 ? (k1 - k0) / (leave[k1] - leave[k0]) : NaN;
  const gaps = ontoDoor.slice(1).map((t, k) => t - ontoDoor[k]);
  return { people: leave.length, last: leave.at(-1), flow, stepOntoApproach_s: quantile(ontoApproach, 0.5), gapBetweenDoorSteps_s: quantile(gaps, 0.5) };
}

function bottlenecks() {
  // 12a: the goal a metres past the bottleneck.
  const rows12a = [];
  for (const a of [0.5, 1, 2, 5, 10]) {
    const g = bottleneckGraph({ goal: a });
    const runs = seeds(3).map((seed) => {
      const people = Math.round((150 * placesOf(g.graph, g.exits)) / g.room1.size);
      const r = simulateCrowd(g.graph, { ...BASE, exits: g.exits, ...crowdOf(g.graph, g.exits, people), doorWidth: 1.0, start: [0, 0], seed, choice: "nearest", horizon: 1500 });
      return room1Out(g, r, g.room1);
    });
    rows12a.push({ a, people: round(mean(runs.map((x) => x.people)), 0), room1_s: round(mean(runs.map((x) => x.last)), 1), flow_ps: round(mean(runs.map((x) => x.flow)), 3) });
  }
  const t = rows12a.map((r) => r.room1_s);
  const nonDecreasing = t.every((x, k) => k === 0 || x >= t[k - 1] * 0.97);
  record({
    id: "RiMEA-12a",
    source: "RiMEA 4.1.1 Test 12a",
    name: "room 1's evacuation against the goal's distance a past the bottleneck (0.5 to 10 m)",
    criterion: "the time rises with a until it is constant (documentation)",
    measured: rows12a,
    pass: null,
    summary: rows12a.map((r) => `a ${r.a} m: ${r.room1_s} s`).join("; ") + (nonDecreasing ? " (non-decreasing)" : " (not monotone)"),
    deviation: "about 150 start in room 1 (persons start at random over both rooms; those who start in room 2 are left out of the reading)",
  });
  // 12b: a 5 m long bottleneck against a short one.
  const run = (g) =>
    seeds(3).map((seed) => {
      const people = Math.round((150 * placesOf(g.graph, g.exits)) / g.room1.size);
      const r = simulateCrowd(g.graph, { ...BASE, exits: g.exits, ...crowdOf(g.graph, g.exits, people), doorWidth: 1.0, start: [0, 0], seed, choice: "nearest", horizon: 1500 });
      return room1Out(g, r, g.room1);
    });
  const short = run(bottleneckGraph({ goal: 10 }));
  const longCorr = run(bottleneckGraph({ goal: 10, long: 5 }));
  const longDoors = run(bottleneckGraph({ goal: 10, long: 5, doorEnds: true }));
  const ts = mean(short.map((x) => x.last)), tl = mean(longCorr.map((x) => x.last)), td = mean(longDoors.map((x) => x.last));
  record({
    id: "RiMEA-12b",
    source: "RiMEA 4.1.1 Test 12b",
    name: "a 5 m long, 1 m wide bottleneck against a short one",
    criterion: "room 1 takes longer to empty through the long bottleneck",
    measured: { short_door_s: round(ts, 1), long_corridor_s: round(tl, 1), long_with_doors_at_its_ends_s: round(td, 1), short_flow_ps: round(mean(short.map((x) => x.flow)), 3), long_corridor_flow_ps: round(mean(longCorr.map((x) => x.flow)), 3) },
    pass: tl > ts,
    summary: `short (one door node) ${round(ts, 1)} s; long corridor ${round(tl, 1)} s; long with door nodes at its ends ${round(td, 1)} s`,
    deviation: "the long bottleneck is two lattice rows (z 5 and 5.5 m); a corridor carries no door capacity in crowd.js, only one person a node",
  });
  // 12d: flow against the bottleneck's width, beside the measured flows (RiMEA Fig. 15: J about 1.9 w).
  const rows12d = [];
  for (const w of [0.8, 1.0, 1.2]) {
    const g = bottleneckGraph({ goal: 10 });
    const runs = seeds(3).map((seed) => {
      const people = Math.round((150 * placesOf(g.graph, g.exits)) / g.room1.size);
      const r = simulateCrowd(g.graph, { ...BASE, exits: g.exits, ...crowdOf(g.graph, g.exits, people), doorWidth: w, start: [0, 0], seed, choice: "nearest", horizon: 1500 });
      return room1Out(g, r, g.room1);
    });
    const J = mean(runs.map((x) => x.flow));
    rows12d.push({
      width_m: w,
      flow_ps: round(J, 3),
      sfpe_capacity_ps: round(doorCapacity(w), 3),
      measured_1_9w_ps: round(1.9 * w, 2),
      ratio_to_measured: round(J / (1.9 * w), 3),
      median_step_onto_node_before_door_s: round(mean(runs.map((x) => x.stepOntoApproach_s)), 2),
      median_gap_between_steps_onto_door_s: round(mean(runs.map((x) => x.gapBetweenDoorSteps_s)), 2),
    });
  }
  record({
    id: "RiMEA-12d",
    source: "RiMEA 4.1.1 Test 12d (Fig. 15, flows from experiments: Kretz et al. 2006; Seyfried et al. 2009; about 1.9 w persons/s)",
    name: "flow through a bottleneck 0.8, 1.0 and 1.2 m wide",
    criterion: "the flow rises with the width (verification); and, as validation, how near it is to the measured flows",
    measured: rows12d,
    pass: rows12d.every((r, k) => k === 0 || r.flow_ps > rows12d[k - 1].flow_ps),
    summary: rows12d.map((r) => `${r.width_m} m: ${r.flow_ps} persons/s (SFPE ${r.sfpe_capacity_ps}, measured about ${r.measured_1_9w_ps})`).join("; "),
    validation: "fails as a match to the measured flows by design: the SFPE capacity is a design value, about half the laboratory flows",
  });
}

/** A room 10 m x 10 m and, from the middle of its far wall, a stair as the app draws one: a single file of nodes `step` m apart along the slope (7 in risers, 11 in treads), `width` m wide, `long` m down (or up) to the exit. */
function stairFileGraph({ width, long = 10, up = false, step = S }) {
  const theta = Math.atan(7 / 11);
  const hx = step * Math.cos(theta), dz = step * Math.sin(theta);
  const steps = Math.round(long / step);
  const cells = [...rect(0, 20, 0, 20)];
  for (let c = 0; c < steps; c += 1) cells.push([21 + c, 10, KIND_STAIR, (up ? 1 : -1) * (c + 1) * dz]);
  cells.push([21 + steps, 10, KIND_DOOR, (up ? 1 : -1) * (steps + 1) * dz]);
  const pos = (i, j, y) => [i <= 20 ? i * S : 20 * S + (i - 20) * hx, y, j * S];
  const { graph, at } = lattice(cells, { pos });
  graph.widths = new Float32Array(graph.nodes.length / 3);
  for (let c = 0; c < steps; c += 1) graph.widths[at(21 + c, 10)] = width;
  const room1 = new Set(rect(0, 20, 0, 20).map(([i, j]) => at(i, j)));
  return { graph, exits: [at(21 + steps, 10)], room1, door: null, at };
}

/**
 * What a door, a corridor and a stair 1 m wide pass, beside the SFPE values (which Pathfinder's SFPE
 * mode reports for them: 0.92, 0.79 and 0.71 persons/s), and a stair against its width.
 */
function passageFlows() {
  const flowOf = (g, opts = {}) =>
    mean(
      seeds(3).map((seed) => {
        const r = simulateCrowd(g.graph, { ...BASE, exits: g.exits, groups: [{ nodes: [...g.room1], count: 150 }], doorWidth: 1.0, start: [0, 0], seed, choice: "nearest", horizon: 3000, ...opts });
        return room1Out(g, r, g.room1).flow;
      }),
    );
  const door = flowOf(bottleneckGraph({ goal: 10 }));
  const corridor = flowOf(bottleneckGraph({ goal: 10, long: 5 }));
  const stairs = [1.0, 1.2, 2.4].map((width) => ({ width_m: width, down_ps: round(flowOf(stairFileGraph({ width }), { doorWidth: 5 }), 3), up_ps: round(flowOf(stairFileGraph({ width, up: true }), { doorWidth: 5 }), 3), sfpe_down_ps: round(Crowd.passageCapacity ? Crowd.passageCapacity(width, true) : NaN, 3) }));
  // A node a tread, as the app draws a modelled stair (0.33 m along the slope for 0.28 m treads).
  const treads = round(flowOf(stairFileGraph({ width: 1.0, step: 0.33 }), { doorWidth: 5 }), 3);
  const expect = { door: 0.921, corridor: 0.789, stair: 0.71 };
  const near = (x, y) => Math.abs(x - y) <= 0.05 * y;
  const rising = stairs.every((r, k) => k === 0 || r.down_ps > stairs[k - 1].down_ps);
  record({
    id: "SFPE-flows",
    source: "SFPE Handbook, 5th ed., ch. 59 (specific flow and boundary layers); the values Pathfinder's SFPE mode reports (Thunderhead Engineering, Verification and Validation)",
    name: "what a door, a corridor and a stair 1 m wide pass, and a stair against its width",
    criterion: "150 persons queueing: the flows within 5 % of 0.92 (door), 0.79 (corridor) and 0.71 persons/s (stair, down), and a wider stair passes more",
    measured: { door_1m_ps: round(door, 3), corridor_1m_ps: round(corridor, 3), stair_1m_down_ps: stairs[0].down_ps, stair_1m_down_a_node_a_tread_ps: treads, sfpe: expect, stairs, passage_flow: PASSAGE },
    pass: near(door, expect.door) && near(corridor, expect.corridor) && near(stairs[0].down_ps, expect.stair) && rising,
    summary: `1 m door ${round(door, 3)}, corridor ${round(corridor, 3)}, stair ${stairs[0].down_ps} persons/s (SFPE 0.92, 0.79, 0.71); stairs down ` + stairs.map((r) => `${r.width_m} m: ${r.down_ps} (SFPE ${r.sfpe_down_ps}; up ${r.up_ps})`).join(", ") + `; 1 m stair with a node a tread: ${treads}`,
    deviation: "the stair is a single file of nodes 0.5 m apart carrying its width (graph.widths), as the app draws a stair, with a wide exit (5 m) at its far end; with a node a tread (0.33 m) the 1 m stair is reported beside the criterion, not in it; the corridor is two lattice rows",
  });
}

/** NIST Verif.5.1 (modified IMO Test 11): a room of 100 persons, a corridor 2 m wide and 12 m long, a stair down at its end. */
function congestion() {
  const theta = Math.atan(7 / 11);
  const hx = S * Math.cos(theta), dz = S * Math.sin(theta);
  // Room 8 m x 5 m (i 0..15, j 0..9); corridor four rows wide (j 3..6), i 16..39; the stair a single file from the corridor's end.
  const cells = [...rect(0, 15, 0, 9), ...rect(16, 39, 3, 6)];
  const steps = 10;
  for (let c = 0; c < steps; c += 1) cells.push([40 + c, 5, KIND_STAIR, -(c + 1) * dz]);
  cells.push([40 + steps, 5, KIND_DOOR, -(steps + 1) * dz]);
  const pos = (i, j, y) => [i <= 39 ? i * S : 39 * S + (i - 39) * hx, y, j * S];
  const { graph, at } = lattice(cells, { pos });
  graph.widths = new Float32Array(graph.nodes.length / 3);
  for (let c = 0; c < steps; c += 1) graph.widths[at(40 + c, 5)] = 2.0;
  const exits = [at(40 + steps, 5)];
  const n = graph.nodes.length / 3;
  const room = rect(0, 15, 0, 9).map(([i, j]) => at(i, j));
  const first = at(40, 5);
  // Measuring areas (x from, x to, m; all 2 m wide, z 1.25 to 3.25): before the room's exit, the corridor's middle, the stair's base.
  const zones = { room_exit: [5.75, 7.75], corridor_middle: [10.75, 16.75], stair_base: [17.75, 19.75] };
  const rows = [];
  for (const seed of seeds(3)) {
    const r = simulateCrowd(graph, { ...BASE, exits, groups: [{ nodes: room, count: 100 }], doorWidth: 5, start: [0, 0], seed, horizon: 1500 });
    const ws = walks(r, n);
    const leftRoom = [], ontoStair = [], pastMiddle = [];
    ws.forEach((segs) => {
      for (const sg of segs) {
        if (graph.nodes[sg.a * 3] <= 7.5 + 1e-6 && graph.nodes[sg.b * 3] > 7.5 + 1e-6 && graph.kinds[sg.b] === KIND_FLOOR) leftRoom.push(sg.t1);
        if (sg.b === first) ontoStair.push(sg.t0);
        if (graph.nodes[sg.a * 3] <= 13.5 + 1e-6 && graph.nodes[sg.b * 3] > 13.5 + 1e-6 && graph.kinds[sg.b] === KIND_FLOOR) pastMiddle.push(sg.t1);
      }
    });
    for (const list of [leftRoom, ontoStair, pastMiddle]) list.sort((a, b) => a - b);
    const emptied = leftRoom[Math.floor(leftRoom.length * 0.9) - 1] ?? NaN;
    const density = {}, peak = {};
    for (const [name, [x0, x1]] of Object.entries(zones)) {
      const series = [];
      for (let t = 5; t <= (name === "stair_base" ? ontoStair.at(-1) : emptied); t += 1) {
        let count = 0;
        ws.forEach((segs, i) => {
          const q = positionAt(graph, segs, t, r.outAt[i]);
          if (q && q[0] >= x0 && q[0] <= x1 && q[2] >= 1.25 && q[2] <= 3.25 && q[1] > H - 0.05) count += 1;
        });
        series.push(count / ((x1 - x0) * 2));
      }
      density[name] = mean(series);
      peak[name] = Math.max(...series);
    }
    // The flow past the corridor's middle in 5 s bins, between the first tenth and the last tenth of those who pass.
    const lo = pastMiddle[Math.floor(pastMiddle.length * 0.1)], hi = pastMiddle[Math.floor(pastMiddle.length * 0.9) - 1];
    const bins = [];
    for (let t = lo; t + 5 <= hi; t += 5) bins.push(pastMiddle.filter((x) => x >= t && x < t + 5).length / 5);
    const flowMid = mean(bins);
    const cv = Math.sqrt(mean(bins.map((x) => (x - flowMid) ** 2))) / flowMid;
    const k0 = 10, k1 = ontoStair.length - 11;
    rows.push({
      people: r.people, out: r.people - r.inside, last_out_s: r.lastOut, room_emptied_90pc_s: emptied,
      density_room_exit: density.room_exit, density_corridor_middle: density.corridor_middle, peak_corridor_middle: peak.corridor_middle, peak_stair_base: peak.stair_base,
      flow_out_of_room_ps: (Math.floor(leftRoom.length * 0.9) - 11) / (emptied - leftRoom[10]), flow_corridor_middle_ps: flowMid, flow_corridor_cv: cv,
      flow_onto_stair_ps: (k1 - k0) / (ontoStair[k1] - ontoStair[k0]),
    });
  }
  const m = (key) => mean(rows.map((x) => x[key]));
  const D = Crowd.QUEUE_DENSITY ?? 1.9;
  const measured = Object.fromEntries(Object.keys(rows[0]).map((key) => [key, round(m(key), 3)]));
  measured.sfpe = { corridor_2m_ps: round(Crowd.passageCapacity ? Crowd.passageCapacity(2.0) : NaN, 3), stair_2m_down_ps: round(Crowd.passageCapacity ? Crowd.passageCapacity(2.0, true) : NaN, 3) };
  measured.passage_flow = PASSAGE;
  const pass = m("out") === m("people") && m("density_room_exit") > D && m("peak_corridor_middle") < D && m("flow_corridor_cv") <= 0.25 && m("peak_stair_base") > D && m("flow_onto_stair_ps") < m("flow_corridor_middle_ps");
  record({
    id: "NIST-Verif.5.1",
    source: "NIST TN 1822 Verif.5.1 (modified IMO Test 11: the stair leads down)",
    name: "congestion: 100 persons in an 8 m x 5 m room, a corridor 2 m wide and 12 m long, a stair down at its end",
    criterion: `qualitative, read here as: congestion before the room's exit (mean density above SFPE's greatest-flow density, ${D} persons/m2, while the room empties), a steady flow in the corridor (its middle never above that density; the flow past it in 5 s bins varying by at most a quarter), and congestion at the base of the stair (peak density above ${D}; the stair takes less than the corridor brings)`,
    measured,
    pass,
    summary: `density before the room's exit ${measured.density_room_exit}/m2; corridor middle peak ${measured.peak_corridor_middle}/m2, flow ${measured.flow_corridor_middle_ps} persons/s (variation ${measured.flow_corridor_cv}); stair base peak ${measured.peak_stair_base}/m2, on to the stair ${measured.flow_onto_stair_ps} persons/s; last out ${measured.last_out_s} s`,
    deviation: "the stair is taken as 2 m wide, as the corridor (the figure's stair dimension was not read), 5 m along the slope, and is a single file of nodes carrying its width, with a wide exit (5 m) at its foot; the room opens on the corridor over its full width; the default population (1.19 m/s, 25 % either way)",
  });
}

/** RiMEA Test 12c: three rooms joined by two identical bottlenecks; the numbers in rooms 1 and 2 over time and the flows through both. */
function secondBottleneck() {
  const cells = [...rect(0, 20, 0, 20), [21, 10, KIND_DOOR], ...rect(22, 42, 0, 20), [43, 10, KIND_DOOR], ...rect(44, 64, 0, 20)];
  for (let j = 0; j <= 20; j += 1) cells.push([65, j, KIND_DOOR]);
  const { graph, at } = lattice(cells);
  const exits = Array.from({ length: 21 }, (_, j) => at(65, j));
  const n = graph.nodes.length / 3;
  const room1 = rect(0, 20, 0, 20).map(([i, j]) => at(i, j));
  const d1 = at(21, 10), d2 = at(43, 10);
  const rows = [];
  let series = null;
  for (const seed of seeds(3)) {
    const r = simulateCrowd(graph, { ...BASE, exits, groups: [{ nodes: room1, count: 150 }], doorWidth: 1.0, start: [0, 0], seed, choice: "nearest", horizon: 1500 });
    const ws = walks(r, n);
    const t1 = [], t2 = [], wait2 = [];
    ws.forEach((segs) => {
      segs.forEach((sg, k) => {
        if (sg.b === d1) t1.push(sg.t0);
        if (sg.b === d2) {
          t2.push(sg.t0);
          wait2.push(k ? sg.t0 - segs[k - 1].t1 : 0);
        }
      });
    });
    t1.sort((a, b) => a - b);
    t2.sort((a, b) => a - b);
    const flow = (t) => (t.length - 21) / (t[t.length - 11] - t[10]);
    const inRoom2 = (t) => t1.filter((x) => x <= t).length - t2.filter((x) => x <= t).length;
    let most = 0;
    for (let t = 0; t <= t2.at(-1); t += 1) most = Math.max(most, inRoom2(t));
    rows.push({ flow_bottleneck_1_ps: flow(t1), flow_bottleneck_2_ps: flow(t2), most_in_room_2: most, room_1_empty_s: t1.at(-1), room_2_empty_s: t2.at(-1), longest_wait_before_bottleneck_2_s: Math.max(...wait2) });
    if (!series) series = [0, 30, 60, 90, 120, 150, 180].map((t) => ({ t_s: t, room_1: 150 - t1.filter((x) => x <= t).length, room_2: inRoom2(t) }));
  }
  const measured = Object.fromEntries(Object.keys(rows[0]).map((key) => [key, round(mean(rows.map((x) => x[key])), 3)]));
  measured.numbers_over_time_seed_1 = series;
  record({
    id: "RiMEA-12c",
    source: "RiMEA 4.1.1 Test 12c",
    name: "150 persons through three rooms joined by two identical bottlenecks 1 m wide",
    criterion: "the numbers in rooms 1 and 2 over time and the flows through bottlenecks 1 and 2 (documentation: the test gives an aim, no expected result)",
    measured,
    pass: null,
    summary: `bottleneck 1 ${measured.flow_bottleneck_1_ps} persons/s, bottleneck 2 ${measured.flow_bottleneck_2_ps}; at most ${measured.most_in_room_2} in room 2 at once; longest wait before bottleneck 2 ${measured.longest_wait_before_bottleneck_2_s} s`,
    deviation: "the rooms are taken as 10 m x 10 m, as Test 12b's (Fig. 14's dimensions were not read); the bottlenecks are short (one door node each); the default population, not RiMEA's",
  });
}

/** RiMEA Test 10, NIST Verif.3.1: a corridor with rooms, a main and a secondary exit; deterministic exit choice. */
function exitAllocation() {
  // Corridor z 5..6 (rows 10..12), x 0..18; top rooms z 7..11 (rows 14..22) behind a wall row 13; bottom rooms z 0..4 (rows 0..8) behind wall row 9.
  const cells = [...rect(0, 36, 10, 12)];
  const topRooms = [["1", 0, 5], ["2", 7, 11], ["3", 16, 17], ["4", 19, 23], ["5", 25, 29], ["6", 31, 35]];
  const bottomRooms = [["7", 0, 5], ["8", 7, 11], ["9", 13, 17], ["10", 19, 23], ["11", 25, 29], ["12", 31, 35]];
  const roomOfCell = new Map();
  for (const [name, i0, i1] of topRooms) {
    for (const c of rect(i0, i1, 14, 22)) {
      cells.push(c);
      roomOfCell.set(`${c[0]},${c[1]}`, name);
    }
    cells.push([Math.round((i0 + i1) / 2), 13]);
  }
  for (const [name, i0, i1] of bottomRooms) {
    for (const c of rect(i0, i1, 0, 8)) {
      cells.push(c);
      roomOfCell.set(`${c[0]},${c[1]}`, name);
    }
    cells.push([Math.round((i0 + i1) / 2), 9]);
  }
  // The main exit's passage x 6.5..7 (columns 13, 14) up from the corridor; the main exit at its top; the secondary at the corridor's end.
  cells.push(...rect(13, 14, 13, 22), [13, 23, KIND_DOOR], [37, 11, KIND_DOOR]);
  const { graph, at, index } = lattice(cells);
  const main = at(13, 23), secondary = at(37, 11);
  const keyOf = new Map([...index].map(([k, v]) => [v, k]));
  const assigned = { main: new Set(["1", "2", "3", "4", "7", "8", "9", "10"]), secondary: new Set(["5", "6", "11", "12"]) };
  const tally = {};
  let deterministic = 0, total = 0, matchRiMEA = 0, inRooms = 0;
  for (const choice of ["nearest", "quickest"]) {
    const used = { main: 0, secondary: 0 };
    for (const seed of seeds(5)) {
      const r = simulateCrowd(graph, { ...BASE, exits: [main, secondary], ...crowdOf(graph, [main, secondary], 23), start: [0, 0], seed, choice });
      for (let i = 0; i < r.people; i += 1) {
        const exit = r.exitOf[i] === main ? "main" : r.exitOf[i] === secondary ? "secondary" : "none";
        if (exit !== "none") used[exit] += 1;
        if (choice !== "nearest") continue;
        const v = startNode(r, i);
        // Nearest by the walk: compare the two walk distances.
        const d = (e) => walkDistance(graph, v, e);
        const nearest = d(main) <= d(secondary) ? "main" : "secondary";
        total += 1;
        if (exit === nearest) deterministic += 1;
        const room = roomOfCell.get(keyOf.get(v));
        if (room) {
          inRooms += 1;
          if (assigned[exit]?.has(room)) matchRiMEA += 1;
        }
      }
    }
    tally[choice] = used;
  }
  record({
    id: "NIST-Verif.3.1",
    source: "NIST TN 1822 Verif.3.1 (IMO Test 10); RiMEA 4.1.1 Test 10",
    name: "exit choice in a corridor with twelve rooms, a main and a secondary exit",
    criterion: "a deterministic model sends everyone to the closest exit (NIST); RiMEA's assignment by room cannot be set in crowd.js",
    measured: { nearest_choice_deterministic: `${deterministic} of ${total}`, rimea_assignment_matched_by_nearest: `${matchRiMEA} of ${inRooms} starting in rooms`, exits_used: tally },
    pass: deterministic === total,
    summary: `nearest: ${deterministic} of ${total} to their closest exit; ${matchRiMEA} of ${inRooms} in rooms where RiMEA assigns them; quickest used main ${tally.quickest.main}, secondary ${tally.quickest.secondary}`,
    deviation: "persons start at random over rooms, corridor and passage, and choose their own exit",
  });
  if (!POPULATIONS) return;
  // As written: 23 persons in the rooms, each room's given its exit (rooms 1 to 4 and 7 to 10 the main one, the others the secondary).
  const nodesOf = new Map();
  for (const [key, room] of roomOfCell) {
    if (!nodesOf.has(room)) nodesOf.set(room, []);
    nodesOf.get(room).push(index.get(key));
  }
  const rooms = [...nodesOf.keys()].sort((a, b) => Number(a) - Number(b));
  let asGiven = 0, all = 0, left = 0;
  const byExit = { main: 0, secondary: 0 };
  for (const seed of seeds(5)) {
    const groups = rooms.map((room, k) => ({ nodes: nodesOf.get(room), count: room === "3" ? 1 : 2, exit: assigned.main.has(room) ? main : secondary }));
    const r = simulateCrowd(graph, { ...BASE, exits: [main, secondary], groups, start: [0, 0], seed });
    for (let i = 0; i < r.people; i += 1) {
      all += 1;
      if (r.exitOf[i] >= 0) left += 1;
      if (r.exitOf[i] === groups[r.groupOf[i]].exit) asGiven += 1;
      if (r.exitOf[i] === main) byExit.main += 1;
      if (r.exitOf[i] === secondary) byExit.secondary += 1;
    }
  }
  record({
    id: "RiMEA-10",
    source: "RiMEA 4.1.1 Test 10; NIST TN 1822 Verif.3.1 (IMO Test 10)",
    name: "23 persons in twelve rooms, each room given the main or the secondary exit",
    criterion: "everyone leaves by the exit given to them",
    measured: { people: all, out: left, by_the_exit_given: asGiven, exits_used: byExit, placing: "two persons a room, one in room 3, as IMO's figure has them" },
    pass: all === 23 * seeds(5).length && left === all && asGiven === all,
    summary: `${asGiven} of ${all} left by the exit given to them (main ${byExit.main}, secondary ${byExit.secondary})`,
  });
}

/** The walk from v to an exit node, by Dijkstra over link lengths. */
function walkDistance(graph, v, target) {
  const n = graph.nodes.length / 3;
  const adj = Array.from({ length: n }, () => []);
  for (let e = 0; e < graph.edges.length; e += 2) {
    const a = graph.edges[e], b = graph.edges[e + 1], w = dist3(graph, a, b);
    adj[a].push([b, w]);
    adj[b].push([a, w]);
  }
  const d = new Float64Array(n).fill(Infinity);
  d[v] = 0;
  const done = new Uint8Array(n);
  for (;;) {
    let u = -1;
    for (let k = 0; k < n; k += 1) if (!done[k] && d[k] < Infinity && (u < 0 || d[k] < d[u])) u = k;
    if (u < 0 || u === target) break;
    done[u] = 1;
    for (const [w, l] of adj[u]) if (d[u] + l < d[w]) d[w] = d[u] + l;
  }
  return d[target];
}

/** RiMEA Test 11: 1,000 persons, two exits; most take the nearer, with congestion there; some take the other. */
/** NIST Verif.4.1 as written: one occupant at 1 m/s in a 10 m x 15 m room, Exit 1 unavailable after 1 s. */
function dynamicExitAsWritten() {
  if (!POPULATIONS) return;
  // The room 10 m (x) by 15 m (z); the exits on the two 15 m walls, 2 m from the far end.
  const { graph, at } = lattice([...rect(0, 20, 0, 30), [-1, 26, KIND_DOOR], [21, 26, KIND_DOOR]]);
  const exit1 = at(-1, 26), exit2 = at(21, 26);
  const n = graph.nodes.length / 3;
  const closes = new Float64Array(n).fill(Infinity);
  closes[exit1] = 1;
  const rows = {};
  let ok = true;
  // The figure draws the occupant by the near wall, a little right of centre; a second place left of centre, nearer Exit 1, is the harder case.
  for (const [name, i] of [["right of centre", 11], ["left of centre, nearer Exit 1", 8]]) {
    const open = simulateCrowd(graph, { ...BASE, exits: [exit1, exit2], groups: [{ nodes: [at(i, 1)], count: 1, speed: [1, 1] }], start: [0, 0] });
    const r = simulateCrowd(graph, { ...BASE, exits: [exit1, exit2], groups: [{ nodes: [at(i, 1)], count: 1, speed: [1, 1] }], start: [0, 0], closes });
    const used = r.exitOf[0] === exit1 ? "Exit 1" : r.exitOf[0] === exit2 ? "Exit 2" : "none";
    rows[name] = { with_both_open: open.exitOf[0] === exit1 ? "Exit 1" : "Exit 2", with_exit_1_closed_at_1s: used, out_at_s: round(r.outAt[0], 1) };
    if (used !== "Exit 2") ok = false;
  }
  record({
    id: "NIST-Verif.4.1-as-written",
    source: "NIST TN 1822 Verif.4.1",
    name: "one occupant at 1 m/s in a 10 m x 15 m room, Exit 1 unavailable after 1 s",
    criterion: "Exit 1 is not used by the occupant",
    measured: rows,
    pass: ok,
    summary: Object.entries(rows).map(([k, v]) => `${k}: ${v.with_exit_1_closed_at_1s} (both open: ${v.with_both_open})`).join("; "),
  });
}

/** RiMEA Test 11 as written: 1,000 adults packed from the left side at the greatest density the lattice holds. */
function routeChoiceAsWritten() {
  if (!POPULATIONS) return;
  const cells = [...rect(0, 60, 0, 40), [43, 41, KIND_DOOR], [55, 41, KIND_DOOR]];
  const { graph, at } = lattice(cells);
  const e1 = at(43, 41), e2 = at(55, 41);
  // The block starts 2 m from the left and the near walls; one person a node (4 persons/m2), filled column by column from the left.
  const block = [];
  for (let i = 4; i <= 60 && block.length < 1000; i += 1) for (let j = 4; j <= 40 && block.length < 1000; j += 1) block.push(at(i, j));
  const people = QUICK ? 300 : 1000;
  const shares = [], waits = [];
  for (const seed of seeds(2)) {
    const r = simulateCrowd(graph, { ...BASE, exits: [e1, e2], groups: [{ nodes: block.slice(0, people), count: people, speed: POPULATIONS.weidmann }], doorWidth: 1.0, start: [0, 0], seed, horizon: 3000 });
    shares.push(Array.from(r.exitOf).filter((x) => x === e1).length / r.people);
    waits.push(r.longestWait);
  }
  const share = mean(shares), wait = mean(waits);
  record({
    id: "RiMEA-11-packed",
    source: "RiMEA 4.1.1 Test 11",
    name: `${people.toLocaleString("en")} adults packed from the left side, two exits 5 m apart in one wall`,
    criterion: "the persons prefer the closer exit 1, congestion occurs there, and some use exit 2",
    measured: { exit1_share: round(share, 3), longest_wait_s: round(wait, 1), density_per_m2: 4, speeds: "Weidmann's normal population (RiMEA gives its speeds by age as a curve only)" },
    pass: share > 0.5 && share < 1 && wait > 0,
    summary: `${round(share * 100, 1)} % by exit 1 (longest wait ${round(wait, 1)} s), the rest by exit 2`,
  });
}

function routeChoice() {
  const cells = [...rect(0, 60, 0, 40), [43, 41, KIND_DOOR], [55, 41, KIND_DOOR]];
  const { graph, at } = lattice(cells);
  const e1 = at(43, 41), e2 = at(55, 41);
  const rows = {};
  for (const choice of ["quickest", "nearest"]) {
    const shares = [], waits = [];
    for (const seed of seeds(2)) {
      const r = simulateCrowd(graph, { ...BASE, exits: [e1, e2], ...crowdOf(graph, [e1, e2], QUICK ? 300 : 1000), doorWidth: 1.0, start: [0, 0], seed, choice, horizon: 3000 });
      const by1 = Array.from(r.exitOf).filter((x) => x === e1).length;
      shares.push(by1 / r.people);
      waits.push(r.longestWait);
    }
    rows[choice] = { exit1_share: round(mean(shares), 3), longest_wait_s: round(mean(waits), 1) };
  }
  const q = rows.quickest;
  record({
    id: "RiMEA-11",
    source: "RiMEA 4.1.1 Test 11",
    name: "1,000 persons, two exits 5 m apart in one wall",
    criterion: "most prefer the closer exit 1, congestion there, some use exit 2",
    measured: rows,
    pass: q.exit1_share > 0.5 && q.exit1_share < 1 && q.longest_wait_s > 0,
    summary: `quickest: ${round(q.exit1_share * 100, 1)} % by exit 1 (longest wait ${q.longest_wait_s} s); nearest: ${round(rows.nearest.exit1_share * 100, 1)} %`,
    deviation: "persons spread over the whole room, not packed from the left side (crowd.js has no start areas), so exit 2 is the nearer for those far right",
  });
}

/** RiMEA Test 16 (long corridor, one agent wide): speed against line density, against the experiments' 10 and 90 % percentiles. */
function fundamental1D(percentiles) {
  const { graph, at } = lattice([...rect(0, 400, 0, 0), [401, 0, KIND_DOOR]]);
  const exits = [at(401, 0)];
  const n = graph.nodes.length / 3;
  const rows = [];
  for (const profile of [{ name: "1.0 m/s for all", speed: 1.0, spread: 0 }, { name: "the default population", speed: FREE_SPEED, spread: 0.25 }]) {
    for (const rho of [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]) {
      const people = Math.round(rho * 200);
      const v = [], dens = [];
      for (const seed of seeds(3)) {
        const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, people), speed: profile.speed, spread: profile.spread, doorWidth: 5, start: [0, 0], seed, choice: "nearest", horizon: 60 });
        const ws = walks(r, n);
        let count = 0;
        ws.forEach((segs, i) => {
          const p10 = positionAt(graph, segs, 10, r.outAt[i]), p40 = positionAt(graph, segs, 40, r.outAt[i]);
          if (!p10 || p10[0] < 40 || p10[0] > 80) return;
          count += 1;
          if (p40) v.push((p40[0] - p10[0]) / 30);
        });
        dens.push(count / 40);
      }
      const d = mean(dens), speed = mean(v);
      const ref = interpolate(percentiles, d);
      rows.push({ profile: profile.name, set_density_per_m: rho, measured_density_per_m: round(d, 3), mean_speed_mps: round(speed, 3), p10: round(ref.p10, 3), p50: round(ref.p50, 3), p90: round(ref.p90, 3), inside_band: speed >= ref.p10 && speed <= ref.p90 });
    }
  }
  const inBand = rows.filter((r) => r.inside_band).length;
  record({
    id: "RiMEA-16",
    source: "RiMEA 4.1.1 Test 16 (percentiles: rimea.de test_16_percentiles.xlsx; raw data Zenodo doi:10.5281/zenodo.8378592)",
    name: "single-file fundamental diagram, 200 m corridor one agent wide",
    criterion: "mean speeds between the experiments' 10 and 90 % percentiles at the measured density (validation)",
    measured: rows,
    pass: inBand === rows.length,
    summary: `${inBand} of ${rows.length} points inside the 10 to 90 % band; ` + rows.filter((r) => r.profile === "1.0 m/s for all").map((r) => `${r.measured_density_per_m}/m: ${r.mean_speed_mps} m/s [${r.p10}, ${r.p90}]`).join("; "),
    deviation: "the lattice holds at most 2 persons/m in single file (0.5 m nodes); measured in x 40 to 80 m between 10 and 40 s, before the exit's queue reaches it",
  });
}

const interpolate = (table, d) => {
  if (d <= table[0].d) return table[0];
  for (let k = 1; k < table.length; k += 1) {
    if (d <= table[k].d) {
      const f = (d - table[k - 1].d) / (table[k].d - table[k - 1].d);
      const mix = (key) => table[k - 1][key] + f * (table[k][key] - table[k - 1][key]);
      return { p10: mix("p10"), p50: mix("p50"), p90: mix("p90") };
    }
  }
  return table.at(-1);
};

/** RiMEA Test 4 (2D): speed against density in a 10 m wide corridor, beside SFPE's relation (the model's own) and Weidmann's (1993). */
function fundamental2D() {
  const cells = [...rect(0, 200, 0, 20)];
  for (let j = 0; j <= 20; j += 1) cells.push([201, j, KIND_DOOR]);
  const { graph, at } = lattice(cells);
  const exits = Array.from({ length: 21 }, (_, j) => at(201, j));
  const n = graph.nodes.length / 3;
  const weidmann = (d) => 1.34 * (1 - Math.exp(-1.913 * (1 / d - 1 / 5.4)));
  const sfpe = (d) => (d <= 0.54 ? FREE_SPEED : SFPE.k * (1 - SFPE.a * d));
  const rows = [];
  for (const rho of QUICK ? [0.5, 2] : [0.5, 1, 2, 3, 3.8]) {
    const people = Math.round(rho * 1000);
    const vs = [], ds = [];
    for (const seed of seeds(2)) {
      const r = simulateCrowd(graph, { ...BASE, exits, ...crowdOf(graph, exits, people), doorWidth: 5, start: [0, 0], seed, choice: "nearest", horizon: 45 });
      const ws = walks(r, n);
      for (let t = 10; t <= 40; t += 2) {
        let count = 0, sum = 0;
        ws.forEach((segs, i) => {
          const p = positionAt(graph, segs, t, r.outAt[i]);
          if (!p || p[0] < 49 || p[0] > 51 || p[2] < 4 || p[2] > 6) return;
          count += 1;
          const q = positionAt(graph, segs, t + 1, r.outAt[i]);
          if (q) sum += Math.hypot(q[0] - p[0], q[2] - p[2]);
        });
        if (count) {
          ds.push(count / 4);
          vs.push(sum / count);
        }
      }
    }
    const d = mean(ds), v = mean(vs);
    rows.push({ set_density_per_m2: rho, measured_density_per_m2: round(d, 3), mean_speed_mps: round(v, 3), sfpe_mps: round(sfpe(d), 3), weidmann_mps: round(weidmann(d), 3), rel_diff_weidmann: round((v - weidmann(d)) / weidmann(d), 3) });
  }
  record({
    id: "RiMEA-4",
    source: "RiMEA 4.1.1 Test 4; Weidmann (1993) for comparison",
    name: "fundamental diagram in a 10 m wide corridor (2 m x 2 m measuring area)",
    criterion: "documentation: speed against density beside the model's SFPE relation and Weidmann's",
    measured: rows,
    pass: null,
    summary: rows.map((r) => `${r.measured_density_per_m2}/m2: ${r.mean_speed_mps} m/s (SFPE ${r.sfpe_mps}, Weidmann ${r.weidmann_mps})`).join("; "),
    deviation: "100 m long, not 1,000 m; the lattice holds at most 4 persons/m2 (RiMEA asks up to 6); measured between 10 and 40 s",
  });
}

/** NIST Verif.4.1: an exit that closes at 10 s is not used after it. */
function dynamicExit() {
  const { graph, at } = lattice([...rect(0, 20, 0, 20), [-1, 10, KIND_DOOR], [21, 10, KIND_DOOR]]);
  const a = at(-1, 10), b = at(21, 10);
  const n = graph.nodes.length / 3;
  const closes = new Float64Array(n).fill(Infinity);
  closes[a] = 10;
  let after = 0, crossing = 0, people = 0, out = 0;
  for (const seed of seeds(3)) {
    const r = simulateCrowd(graph, { ...BASE, exits: [a, b], ...crowdOf(graph, [a, b], 80), start: [0, 0], seed, closes, doorWidth: 0.9 });
    people += r.people;
    out += r.people - r.inside;
    walks(r, n).forEach((segs, i) => {
      if (r.exitOf[i] !== a) return;
      const step = segs.find((s) => s.b === a);
      if (!step) return;
      if (step.t0 >= 10 - 1e-9) after += 1;
      else if (step.t1 > 10) crossing += 1;
    });
  }
  record({
    id: "NIST-Verif.4.1",
    source: "NIST TN 1822 Verif.4.1",
    name: "an exit closing at 10 s",
    criterion: "no one steps out by it after it closes; everyone still leaves",
    measured: { people, out, steps_on_to_closed_exit_begun_after_10s: after, steps_begun_before_and_ending_after_10s: crossing },
    pass: after === 0 && out === people,
    summary: `${after} steps on to the closed exit begun after 10 s (${crossing} begun before it closed, ending after); ${out} of ${people} out`,
  });
}

// ------------------------------------------------------------------ run

const pFile = path.join(here, "rimea-test16-percentiles.json");
let percentiles;
try {
  percentiles = JSON.parse(readFileSync(pFile, "utf8")).rows;
} catch {
  percentiles = null;
}

const t0 = Date.now();
corridorSpeed();
stairSpeed();
corner();
premovement();
speedDistribution();
populations();
maxFlow();
dynamicExit();
exitAllocation();
bottlenecks();
secondBottleneck();
passageFlows();
congestion();
if (percentiles) fundamental1D(percentiles);
fundamental2D();
largeRoom();
routeChoice();
routeChoiceAsWritten();
dynamicExitAsWritten();

const code = readFileSync(path.join(root, "topologicpy-web-frontend", "src", "core", "crowd.js"));
const report = {
  date: process.argv.includes("--date") ? process.argv[process.argv.indexOf("--date") + 1] : "2026-11-11",
  model: { file: "topologicpy-web-frontend/src/core/crowd.js", sha256: createHash("sha256").update(code).digest("hex"), lattice_spacing_m: S, dt_s: 0.1, options: BASE },
  quick: QUICK,
  seconds: Math.round((Date.now() - t0) / 1000),
  sources: {
    rimea: "RiMEA e.V., Guideline for Microscopic Evacuation Analysis, version 4.1.1 of 11.09.2025 (German authoritative), https://rimea.de, CC BY-ND 4.0",
    nist: "Ronchi E., Kuligowski E.D., Reneke P.A., Peacock R.D., Nilsson D. (2013) The Process of Verification and Validation of Building Fire Evacuation Models, NIST TN 1822, doi:10.6028/NIST.TN.1822",
    iso: "ISO 20414:2020, Fire safety engineering: verification and validation protocol for building fire evacuation models (paywalled; not read; scope from the ISO page)",
    percentiles: "RiMEA Test 16 percentiles, https://rimea.de/wp-content/uploads/2025/01/test_16_percentiles.xlsx (raw data: Chraibi and Subaih, SingleFileComparator, Zenodo doi:10.5281/zenodo.8378592)",
  },
  summary: { pass: results.filter((r) => r.pass === true).length, fail: results.filter((r) => r.pass === false).length, documentation: results.filter((r) => r.pass === null).length },
  tests: results,
};
const outArg = process.argv.indexOf("--out");
const outFile = path.join(root, "benchmark", "results", outArg > 0 ? process.argv[outArg + 1] : "2026-11-11-crowd-validation.json");
const NO_WRITE = process.argv.includes("--no-write");
if (!NO_WRITE) writeFileSync(outFile, JSON.stringify(report, null, 1));
if (NO_WRITE) console.log("SUMMARY " + JSON.stringify({ ...report.summary, failed: results.filter((r) => r.pass === false).map((r) => r.id) }));
else console.log(`\n${report.summary.pass} pass, ${report.summary.fail} fail, ${report.summary.documentation} documentation; ${report.seconds} s; ${path.relative(root, outFile)}`);
