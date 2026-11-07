/** Portable fire extinguishers (extinguishers.js): the code values, the placement check, the layout, the IFC reader and the attempt. */
import test from "node:test";
import assert from "node:assert/strict";
import {
  AREA_PER_EXTINGUISHER,
  CAPACITY,
  EXTINGUISHER_HAZARDS,
  STANDOFF,
  TRAVEL,
  T_TAKE,
  UNITS,
  attempt,
  capFor,
  extinguisherCap,
  extinguisherRecords,
  layoutExtinguishers,
  placementCheck,
  requiredCount,
  samplingMargin,
  suitability,
  travelLimit,
  ulA,
} from "./extinguishers.js";
import { createSuppression, evansFactor } from "./suppression.js";
import { KIND_DOOR } from "./lattice.js";

const close = (a, b, tol, what = "") => assert.ok(Math.abs(a - b) <= tol, `${what} ${a} vs ${b}`);
const RANK = { success: 0, "cannot tell": 1, failure: 2 };

/** Nodes at the given plan points [x, z] joined in a chain (y up, lifted 0.75 m). */
function chain(points, spacing = 0.5) {
  const nodes = [], edges = [];
  points.forEach(([x, z], i) => {
    nodes.push(x, 0.75, z);
    if (i) edges.push(i - 1, i);
  });
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: new Uint8Array(points.length), upAxis: "y", agentHeight: 0.75, spacing };
}

/**
 * One storey of two rooms on a lattice of `s` m (8 neighbours): x from 0 to
 * (nx - 1) s, z from 0 to (nz - 1) s, a wall between columns `wall` and
 * `wall + 1` whose links are blocked except the straight ones through a
 * doorway in rows door[0] to door[1], whose nodes are doors.
 */
function rooms({ nx = 121, nz = 21, s = 0.5, wall = 60, door = [9, 11] } = {}) {
  const nodes = [], edges = [], blocked = [];
  const id = (i, j) => j * nx + i;
  for (let j = 0; j < nz; j += 1) for (let i = 0; i < nx; i += 1) nodes.push(i * s, 0.75, j * s);
  for (let j = 0; j < nz; j += 1) {
    for (let i = 0; i < nx; i += 1) {
      for (const [di, dj] of [[1, 0], [0, 1], [1, 1], [1, -1]]) {
        const i2 = i + di, j2 = j + dj;
        if (i2 >= nx || j2 < 0 || j2 >= nz) continue;
        edges.push(id(i, j), id(i2, j2));
        const crosses = i <= wall && i2 > wall;
        blocked.push(crosses && !(dj === 0 && j >= door[0] && j <= door[1]) ? 1 : 0);
      }
    }
  }
  const kinds = new Uint8Array(nx * nz);
  for (let j = door[0]; j <= door[1]; j += 1) kinds[id(wall, j)] = kinds[id(wall + 1, j)] = KIND_DOOR;
  return { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), blocked: Uint8Array.from(blocked), kinds, upAxis: "y", agentHeight: 0.75, spacing: s, id };
}

test("the code values: travel, area per unit, area per extinguisher, least rating, the margin and the units", () => {
  assert.equal(TRAVEL.A.value, 22.86);
  assert.match(TRAVEL.A.clause, /SBC 801-CR-2018 906\.3\.1, Table 906\.3\(1\)/);
  assert.match(TRAVEL.A.clause, /NFPA 10 Table 6\.2\.1\.1/);
  assert.deepEqual(TRAVEL.B.light, [[5, 9], [10, 15]]);
  assert.deepEqual(TRAVEL.B.ordinary, [[10, 9], [20, 15]]);
  assert.deepEqual(TRAVEL.B.extra, [[40, 9], [80, 15]]);
  assert.equal(TRAVEL.K.value, 9);
  assert.equal(TRAVEL.D.value, 22.86);
  assert.deepEqual(
    Object.values(EXTINGUISHER_HAZARDS).map((h) => [h.areaPerUnit.value, h.minRating.value]),
    [[279, 2], [139, 2], [92.9, 4]],
  );
  assert.equal(AREA_PER_EXTINGUISHER.value, 1045);
  close(samplingMargin(0.5), 0.35355, 1e-5);
  // Class B by rating: the smaller unit within 9 m, the larger within 15 m, smaller still none.
  assert.equal(travelLimit("B", { hazard: "ordinary", rating: 10 }).value, 9);
  assert.equal(travelLimit("B", { hazard: "ordinary", rating: 20 }).value, 15);
  assert.equal(travelLimit("B", { hazard: "ordinary", rating: 5 }).value, null);
  assert.equal(travelLimit("K").value, 9);
  // The units and the capacity parameters.
  const w = UNITS["water-9.5"];
  assert.deepEqual([w.agent, w.rating.A, w.tD, w.reach], ["water", 2, 60, 9.1]);
  assert.deepEqual([UNITS["water-6"].rating.A, UNITS["water-6"].tD], [1, 40]);
  assert.deepEqual([UNITS["abc-2.3"].rating.A, UNITS["abc-2.3"].rating.B, UNITS["abc-2.3"].tD, UNITS["abc-2.3"].reach], [2, 10, 14, 1.5]);
  assert.deepEqual([UNITS["abc-4.5"].rating.A, UNITS["abc-4.5"].tD, UNITS["abc-4.5"].reach], [4, 20, 4.6]);
  assert.deepEqual([UNITS["co2-2"].agent, UNITS["co2-2"].tD, UNITS["co2-2"].reach], ["co2", 8.5, 0.9]);
  close(ulA("abc-6-en"), (34 * 2) / 13, 1e-12, "34A EN as UL A units");
  assert.deepEqual([CAPACITY.qA.value, CAPACITY.qA.low, CAPACITY.qA.high], [300, 200, 450]);
  assert.deepEqual([CAPACITY.eta.value, CAPACITY.eta.low, CAPACITY.eta.high], [0.4, 0.2, 1.0]);
  assert.equal(CAPACITY.areaPerB.ul, 0.0929);
  assert.equal(CAPACITY.epsilon.value, 0.01);
  assert.deepEqual([T_TAKE.value, T_TAKE.instructed, STANDOFF.value], [13.4, 7.2, 1.8]);
});

test("the travel rule on a corridor: pass at 22.4 m, near the limit at 22.6 m, fail at 23 m", () => {
  const xs = Array.from({ length: 45 }, (_, i) => i * 0.5); // 0 to 22 m
  const travel = (last) => {
    const g = chain([...xs, last].map((x) => [x, 0]));
    const r = placementCheck(g, [{ node: 0, rating: 2 }], { levels: [0], occupancy: "B", spacing: 0.5 });
    const st = r.storeys[0];
    return { r, st, check: st.checks.find((c) => c.id === "travel") };
  };
  const pass = travel(22.4);
  assert.equal(pass.check.status, "pass", pass.check.text);
  assert.equal(pass.r.status, "pass");
  assert.equal(pass.st.worst.node, 45);
  close(pass.st.worst.distance, 22.4, 1e-5);
  close(pass.r.passWithin, 22.86 - 0.5 / Math.SQRT2, 1e-12);
  const near = travel(22.6);
  assert.equal(near.check.status, "cannot tell", near.check.text);
  assert.match(near.check.text, /near the limit/);
  assert.equal(near.st.near, 1);
  const fail = travel(23);
  assert.equal(fail.check.status, "fail", fail.check.text);
  assert.equal(fail.st.over, 1);
  assert.equal(fail.r.status, "fail");
});

test("adding an extinguisher never lengthens a walk, and every walk is at least the straight line", () => {
  const g = rooms();
  const sites = [g.id(3, 3), g.id(100, 18)];
  const before = placementCheck(g, sites, { levels: [0] }).distance;
  const after = placementCheck(g, [...sites, g.id(58, 2)], { levels: [0] }).distance;
  let fell = 0;
  for (let v = 0; v < before.length; v += 1) {
    assert.ok(after[v] <= before[v] + 1e-12, `node ${v}: ${after[v]} after, ${before[v]} before`);
    if (after[v] < before[v]) fell += 1;
  }
  assert.ok(fell > 0, "the new unit shortens some walks");
  const P = g.nodes;
  const all = [...sites, g.id(58, 2)];
  for (let v = 0; v < after.length; v += 1) {
    const straight = Math.min(...all.map((s) => Math.hypot(P[v * 3] - P[s * 3], P[v * 3 + 2] - P[s * 3 + 2])));
    assert.ok(after[v] >= straight - 1e-9, `node ${v}: walk ${after[v]} under the straight ${straight}`);
  }
  // The wall makes a corner of the other room much farther by walk than in a straight line.
  const corner = g.id(61, 0);
  assert.ok(placementCheck(g, [g.id(59, 0)], { levels: [0] }).distance[corner] > 4);
});

test("the count rule: floor area over the area per extinguisher, at least one, never more with a larger unit or a lighter hazard", () => {
  assert.equal(requiredCount(558, "light", 2), 1);
  assert.equal(requiredCount(559, "light", 2), 2);
  assert.equal(requiredCount(600, "ordinary", 2), 3);
  assert.equal(requiredCount(600, "extra", 4), 2);
  assert.equal(requiredCount(2000, "light", 10), 2, "capped at 1045 m² per extinguisher");
  assert.equal(requiredCount(10, "extra", 4), 1, "a small floor still needs one");
  assert.equal(requiredCount(0, "light", 2), 0);
  for (const area of [50, 600, 1500, 5000]) {
    for (const hazard of ["light", "ordinary", "extra"]) {
      for (let r = 1; r < 20; r += 1) assert.ok(requiredCount(area, hazard, r + 1) <= requiredCount(area, hazard, r));
    }
    for (let r = 1; r <= 20; r += 1) {
      assert.ok(requiredCount(area, "light", r) <= requiredCount(area, "ordinary", r));
      assert.ok(requiredCount(area, "ordinary", r) <= requiredCount(area, "extra", r));
    }
  }
  // On the graph: 2535 places of 0.25 m², 634 m², need two 2-A units or one 4-A.
  const g = rooms();
  const count = (sites, opts = {}) => placementCheck(g, sites, { levels: [0], occupancy: "B", ...opts }).storeys[0].checks.find((c) => c.id === "count");
  assert.equal(count([{ node: g.id(30, 10), rating: 2 }]).status, "fail");
  assert.equal(count([{ node: g.id(30, 10), rating: 4 }]).status, "pass");
  assert.equal(count([g.id(30, 10)]).status, "cannot tell", "no rating: a larger unit might do");
  assert.equal(count([g.id(30, 10), g.id(90, 10)], { rating: 2 }).status, "pass");
  // A storey with no unit fails where 906.1 requires them; Group U does not.
  const none = placementCheck(g, [], { levels: [0], occupancy: "B" });
  assert.equal(none.required.status, "fail");
  assert.equal(none.status, "fail");
  const u = placementCheck(g, [], { levels: [0], occupancy: "U" });
  assert.equal(u.required.status, "pass");
  assert.equal(u.storeys[0].status, "n/a");
  assert.equal(placementCheck(g, [g.id(30, 10)], { levels: [0] }).required.status, "cannot tell");
});

test("the greedy cover reaches every place of a two-room storey and its layout passes the check", () => {
  const g = rooms();
  const routeNodes = Array.from({ length: 121 }, (_, i) => g.id(i, 10));
  const exits = [g.id(0, 10)];
  const out = layoutExtinguishers(g, { levels: [0], hazard: "light", exits, routeNodes, occupancy: "B" });
  const { check } = out;
  assert.equal(check.status, "pass", JSON.stringify(check.storeys[0].checks));
  const [st] = out.storeys;
  assert.ok(out.sites.length >= 2 && out.sites.length <= 4, `${out.sites.length} units`);
  assert.equal(st.fallback, 0, "every place is reached from a candidate by a door or on a route");
  for (let v = 0; v < g.kinds.length; v += 1) {
    if (g.kinds[v] === KIND_DOOR) continue;
    assert.ok(check.distance[v] <= check.passWithin + 1e-9, `node ${v} ${check.distance[v]} m from a unit`);
  }
  for (const s of out.sites) {
    assert.equal(s.rating, 2);
    assert.deepEqual(s.position, [g.nodes[s.node * 3], g.nodes[s.node * 3 + 1], g.nodes[s.node * 3 + 2]]);
  }
  // With no routes and no doors the cover still completes, from the walls.
  const bare = { ...g, kinds: new Uint8Array(g.kinds.length) };
  const plain = layoutExtinguishers(bare, { levels: [0], occupancy: "B" });
  assert.equal(plain.check.storeys[0].checks.find((c) => c.id === "travel").status, "pass");
});

test("extinguisher records: names in the corpus's forms, hose reels and hydrants left out, a cabinet one unit", () => {
  const recs = extinguisherRecords([
    { itemId: 1, name: "Extinguisher - C02: Red:123456", ifcClass: "IfcBuildingElementProxy" },
    { itemId: 2, name: "M_Fire Extinguisher Cabinet:305 x 215 x 813mm:4711", typeName: "305 x 215 x 813mm", ifcClass: "IFCFLOWTERMINAL" },
    { itemId: 3, name: 'Fire Extinguisher Cabinet:9 1/2 x 4 1/2 x 24":9876', ifcClass: "IfcBuildingElementProxy" },
    { itemId: 4, name: "FH", typeName: "Fire hose reel", predefinedType: "HOSEREEL", ifcClass: "IfcFlowTerminal" },
    { itemId: 5, name: "Feuerlöscher PG 6 kg ABC 43 A 233 B C", ifcClass: "IfcFurnishingElement" },
    { itemId: 6, name: "Fire extinguisher 2-A:10-B:C", ifcClass: "IfcFireSuppressionTerminal", predefinedType: "USERDEFINED" },
    { itemId: 7, name: "Hose reel and extinguisher cabinet", ifcClass: "IfcFlowTerminal" },
    { itemId: 8, name: "Sprinkler - Pendent - Hosted", ifcClass: "IfcFlowTerminal" },
    { itemId: 9, name: "FEC-1", objectType: "Fire equipment cabinet", ifcClass: "IfcBuildingElementProxy" },
    { itemId: 10, name: "Glazing", objectType: "fec", ifcClass: "IfcWindow" },
    { itemId: 11, name: "Fire Extinguisher - Wall Hosted", ifcClass: "IfcBuildingElementProxy" },
    { itemId: 12, name: "Fire Extinguisher", ifcClass: "IfcWall" },
    { itemId: 13, name: "Brandsläckare 6 kg pulver" },
    { itemId: 14, name: "Water fire extinguisher 9 L" },
    { itemId: 15, name: "Fire hydrant extinguisher point", ifcClass: "IfcFireSuppressionTerminal", predefinedType: "FIREHYDRANT" },
    { itemId: 16, name: "Feuerlöscher Kohlendioxid 2 kg", ifcClass: "IfcBuildingElementProxy" }, // the umlaut decomposed
  ]);
  assert.deepEqual(recs.map((r) => r.itemId), [1, 2, 3, 5, 6, 9, 11, 13, 14, 16]);
  assert.deepEqual([recs.at(-1).agent, recs.at(-1).unit], ["co2", "co2-2"]);
  const by = new Map(recs.map((r) => [r.itemId, r]));
  assert.deepEqual([by.get(1).agent, by.get(1).rating, by.get(1).cabinet], ["co2", null, false]);
  assert.deepEqual([by.get(2).agent, by.get(2).rating, by.get(2).unit, by.get(2).cabinet], ["unknown", null, null, true]);
  assert.equal(by.get(3).cabinet, true);
  assert.deepEqual([by.get(5).agent, by.get(5).unit], ["abc", "abc-6-en"]);
  assert.deepEqual(by.get(5).rating, { system: "EN", A: 43, B: 233, C: true, K: false });
  assert.deepEqual(by.get(6).rating, { system: "UL", A: 2, B: 10, C: true, K: false });
  assert.equal(by.get(9).cabinet, true);
  assert.deepEqual([by.get(13).agent, by.get(13).unit], ["powder", null]);
  assert.deepEqual([by.get(14).agent, by.get(14).unit], ["water", "water-9.5"]);
  assert.deepEqual(recs.excluded.map((e) => e.itemId), [7, 12, 15]);
});

/**
 * The note's worked example: a corridor every 0.5 m from the fire (node 0,
 * x = 0) past the responder (node 10, x = 5 m: 5 m from the fire) to the
 * extinguisher (node 40, x = 20 m: 15 m on) and an exit (node 50, x = 25 m).
 */
function worked() {
  const g = chain(Array.from({ length: 51 }, (_, i) => [i * 0.5, 0]));
  return { ...g, fire: 0, start: 10, ext: 40, exit: 50 };
}

test("the worked example: a medium t² fire, the responder 5 m away, the unit 15 m off: success at t_go 30 s, failure at 330 s", () => {
  const g = worked();
  const alpha = 0.0117;
  const base = { graph: g, fireNode: g.fire, startNode: g.start, sites: [g.ext], speed: 1.2, hrrAt: (t) => alpha * t * t, exits: [g.exit] };
  // A 2-A unit reaching 5 m is used from where the responder stood: 15 m out and 15 m back.
  const unit = { ...UNITS["water-9.5"], reach: 5 };
  const early = attempt({ ...base, unit, tGo: 30 });
  close(early.times.tS, 30 + 12.5 + 13.4 + 12.5, 1e-6, "t_s");
  close(early.hrr, 54.7, 0.1, "Q(t_s) kW");
  assert.equal(early.attack.node, g.start);
  assert.equal(early.outcome, "success", JSON.stringify(early.reasons));
  close(early.capacity.low, 80, 1e-9);
  close(early.capacity.high, 900, 1e-9);
  const late = attempt({ ...base, unit, tGo: 330 });
  close(late.hrr, 1588, 1, "Q(t_s) kW");
  assert.equal(late.outcome, "failure");
  assert.equal(late.reasons[0].id, "too-big");
  // The catalogue's 2-A units give the same verdicts: water from 9 m, ABC from 1.5 m.
  for (const [key, x] of [["water-9.5", 9], ["abc-2.3", 1.5]]) {
    const a = attempt({ ...base, unit: key, tGo: 30 });
    assert.equal(a.outcome, "success", key);
    close(a.attack.position[0], x, 1e-6, `${key} attack place`);
    assert.equal(attempt({ ...base, unit: key, tGo: 330 }).outcome, "failure", key);
  }
  // Time lost against walking straight out (the exit beyond the unit): t_take,
  // t_d and the 15 m back; never under t_take + t_d (time_lost_nonneg).
  close(early.timeLost, 13.4 + 60 + 12.5 + 12.5, 1e-6, "time lost");
  for (const key of Object.keys(UNITS)) {
    const a = attempt({ ...base, unit: key, tGo: 30 });
    if (a.timeLost != null) assert.ok(a.timeLost >= a.times.take + a.unit.tD - 1e-9, `${key}: ${a.timeLost}`);
  }
});

test("success never returns once a growing fire has outgrown the unit: the outcome is monotone in the arrival", () => {
  const g = worked();
  const unit = { ...UNITS["water-9.5"], reach: 5 };
  let previous = 0;
  const seen = new Set();
  for (let tGo = 0; tGo <= 600; tGo += 5) {
    const a = attempt({ graph: g, fireNode: g.fire, startNode: g.start, sites: [g.ext], unit, tGo, hrrAt: (t) => 0.0117 * t * t });
    assert.ok(RANK[a.outcome] >= previous, `t_go ${tGo}: ${a.outcome}`);
    previous = RANK[a.outcome];
    seen.add(a.outcome);
    // A t² fire: success exactly while t_s <= sqrt(Q_low / alpha) (success_iff_tsq).
    assert.equal(a.outcome === "success", a.times.tS <= Math.sqrt(80 / 0.0117));
  }
  assert.deepEqual([...seen], ["success", "cannot tell", "failure"]);
});

test("tau_x = t_d / ln(1/epsilon): the note's values, and the cap at the end of discharge is epsilon of the start", () => {
  close(capFor("water-9.5"), 13.0, 0.05);
  close(capFor("water-6"), 8.7, 0.05);
  close(capFor("abc-2.3"), 3.0, 0.05);
  close(capFor("abc-4.5"), 4.3, 0.05);
  close(capFor("abc-6-en"), 3.5, 0.05);
  close(capFor("co2-2"), 1.8, 0.05);
  assert.ok(capFor("water-9.5", 0.1) > capFor("water-9.5", 0.01), "tau rises with epsilon");
  // The cap on a suppression record: burning cells capped, the others left alone.
  const sup = createSuppression(6);
  const rates = Float32Array.from([0, 0.4, 0, 0.2, 0.3, 0.3]);
  sup.tau[4] = 200; // a sprinkler's slow cap, opened at 50 s from 0.5 kg/s
  sup.tAct[4] = 50;
  sup.rate[4] = 0.5;
  sup.tau[5] = 5; // one faster than the extinguisher's
  sup.tAct[5] = 90;
  sup.rate[5] = 0.3;
  const tau = capFor("water-9.5");
  const res = extinguisherCap(sup, [1, 2, 3, 4, 5], { tS: 100, tau, rates });
  assert.deepEqual(res.capped, [1, 3, 4]);
  assert.deepEqual(res.kept, [5]);
  assert.equal(res.idle, 1);
  assert.equal(sup.tau[2], 0, "a cell not burning at t_s is left alone");
  close(sup.rate[4], 0.3, 1e-7, "the start is the lesser of its rate and the sprinkler's cap");
  assert.equal(sup.tau[5], 5);
  // At t_s + t_d the allowance is epsilon times the rate at t_s (decay_at_discharge_end).
  const dt = 0.1;
  close(evansFactor(sup, 1, 1, 100 + 60, dt), 0.01 * 0.4 * dt, 1e-7);
  assert.equal(evansFactor(sup, 1, 1e-9, 100 + 60, dt), 1, "a loss under the cap is left alone");
});

test("the wrong agent fails with fireClasses.js's reason; the nearest suitable unit is fetched past a nearer unsuitable one", () => {
  const g = worked();
  const base = { graph: g, fireNode: g.fire, startNode: g.start, tGo: 30, hrrAt: (t) => 0.0117 * t * t };
  const water = attempt({ ...base, fireClass: "B", sites: [g.ext], unit: "water-9.5", burningArea: 0.1 });
  assert.equal(water.outcome, "failure");
  const why = water.reasons.find((r) => r.id === "unsuitable");
  assert.ok(why, JSON.stringify(water.reasons));
  assert.match(why.text, /Splashes and spreads the burning liquid/);
  assert.equal(suitability("co2", "A").suitable, false);
  assert.equal(suitability("abc", "B").suitable, true);
  assert.equal(suitability("wet-chemical", "K").suitable, true);
  assert.equal(suitability("powder", "A").suitable, null);
  // A CO2 unit farther away is the one fetched for a liquid fire.
  const mixed = attempt({ ...base, fireClass: "B", sites: [{ node: 19, unit: "water-9.5" }, { node: g.ext, unit: "co2-2" }], burningArea: 0.1 });
  assert.equal(mixed.extinguisher, g.ext);
  assert.equal(mixed.agent, "co2");
  // 34B (EN): 0.4 x pi x 34 / 100 = 0.43 m² for success; 0.1 m² is within it.
  assert.equal(mixed.outcome, "success", JSON.stringify(mixed.reasons));
  assert.equal(attempt({ ...base, fireClass: "B", sites: [{ node: g.ext, unit: "co2-2" }], burningArea: 2 }).outcome, "failure");
  assert.equal(attempt({ ...base, fireClass: "B", sites: [{ node: g.ext, unit: "co2-2" }] }).outcome, "cannot tell", "no burning area");
  // Class L: no attempt.
  const battery = attempt({ ...base, fireClass: "L", sites: [g.ext], exits: [g.exit] });
  assert.deepEqual([battery.outcome, battery.attempted, battery.reasons[0].id, battery.timeLost], ["failure", false, "leave", 0]);
});

test("speed: the placement check on a graph of 7,000 nodes", (t) => {
  const g = rooms({ nx: 140, nz: 50, wall: 70, door: [24, 26] });
  assert.equal(g.kinds.length, 7000);
  const sites = [g.id(5, 5), g.id(5, 44), g.id(65, 25), g.id(75, 25), g.id(134, 5), g.id(134, 44)].map((node) => ({ node, rating: 2 }));
  const opts = { levels: [0], occupancy: "B" };
  let t0 = performance.now();
  const first = placementCheck(g, sites, opts);
  const cold = performance.now() - t0;
  const times = [];
  for (let k = 0; k < 7; k += 1) {
    t0 = performance.now();
    placementCheck(g, sites, opts);
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  const median = times[3];
  t0 = performance.now();
  const layout = layoutExtinguishers(g, { levels: [0], exits: [g.id(0, 25)], routeNodes: Array.from({ length: 140 }, (_, i) => g.id(i, 25)), occupancy: "B" });
  const layoutMs = performance.now() - t0;
  t.diagnostic(`placementCheck, 7,000 nodes: ${cold.toFixed(1)} ms cold, ${median.toFixed(1)} ms median of 7; layoutExtinguishers ${layoutMs.toFixed(0)} ms (${layout.sites.length} units, ${layout.check.status})`);
  assert.equal(first.storeys[0].places, 7000 - 6);
  assert.ok(median < 50, `${median} ms`);
  assert.equal(layout.check.status, "pass");
});
