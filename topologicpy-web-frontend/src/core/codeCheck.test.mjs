/** The egress code check (codeCheck.js) on small graphs whose answers are known. */
import test from "node:test";
import assert from "node:assert/strict";
import { EGRESS_B, IBC2021_B, SBC2018_B, codeCheck } from "./codeCheck.js";

/**
 * A straight corridor of `length` nodes 1 m apart on each of `storeys`
 * storeys, 3.5 m apart, viewer frame (y up, lifted 0.75 m), with one stair
 * from the far end of each storey to the next and an exit door at x = 0 on
 * the ground storey (and a second exit at the far end when `twoExits`).
 */
function building({ length = 10, storeys = 2, twoExits = false } = {}) {
  const nodes = [], kinds = [], edges = [];
  const id = (s, x) => s * length + x;
  for (let s = 0; s < storeys; s += 1) {
    for (let x = 0; x < length; x += 1) {
      nodes.push(x, s * 3.5 + 0.75, 0);
      kinds.push(0);
      if (x) edges.push(id(s, x - 1), id(s, x));
    }
  }
  // A stair node between each pair of storeys, at the far end.
  for (let s = 0; s + 1 < storeys; s += 1) {
    const v = nodes.length / 3;
    nodes.push(length - 1, s * 3.5 + 0.75 + 1.75, 1);
    kinds.push(1);
    edges.push(id(s, length - 1), v, v, id(s + 1, length - 1));
  }
  const exits = [id(0, 0)];
  if (twoExits) exits.push(id(0, length - 1));
  return {
    graph: { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: Uint8Array.from(kinds), upAxis: "y", agentHeight: 0.75, spacing: 1 },
    exits,
    levels: Array.from({ length: storeys }, (_, s) => s * 3.5),
  };
}

test("walking distance to the nearest exit entrance: the exit door below, the stair door above", () => {
  const { graph, exits, levels } = building({ length: 10, storeys: 2 });
  const r = codeCheck(graph, { exits, levels });
  // Ground storey: from x = 9 to the door at x = 0, 9 m. Upper storey: the stair is entered at x = 9, so 9 m from x = 0.
  assert.equal(r.storeys[0].maxTravel, 9);
  assert.equal(r.storeys[1].maxTravel, 9);
  assert.equal(r.travel[0], 0);
  assert.equal(r.storeys[0].checks.find((c) => c.id === "travel").status, "pass");
  assert.equal(r.limit, EGRESS_B.travel.unsprinklered);
});

test("on a storey with an exit door, a part that reaches it only by a stair leaves by the stair", () => {
  // Two corridors on one storey (levels say so) 1.2 m apart in height, joined
  // only by a stair: the exit door is on the upper one.
  const nodes = [], kinds = [], edges = [];
  for (let x = 0; x < 6; x += 1) nodes.push(x, 0.75, 0), kinds.push(0);
  for (let x = 0; x < 6; x += 1) nodes.push(x, 1.95, 4), kinds.push(0);
  for (let x = 1; x < 6; x += 1) edges.push(x - 1, x, 6 + x - 1, 6 + x);
  nodes.push(5, 1.35, 2), kinds.push(1);
  edges.push(5, 12, 12, 11);
  const graph = { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: Uint8Array.from(kinds), upAxis: "y", agentHeight: 0.75, spacing: 1 };
  const r = codeCheck(graph, { exits: [6], levels: [0] });
  const [storey] = r.storeys;
  assert.equal(storey.checks.find((c) => c.id === "unreached"), undefined, "every place reaches an exit");
  assert.equal(r.travel[0], 5, "the lower corridor walks to the stair at x = 5");
  assert.equal(r.travel[11], 5, "the upper one to the door at x = 0");
  assert.equal(storey.exits, 2);
});

test("too far, too few exits and exits too close fail; sprinklers relax the limits", () => {
  // 70 m corridors: over the 60 m limit without sprinklers, within 90 m with them (SBC 801-CR-2018 Table 1017.2).
  const long = building({ length: 71, storeys: 1 });
  const dry = codeCheck(long.graph, { exits: long.exits, levels: long.levels });
  const wet = codeCheck(long.graph, { exits: long.exits, levels: long.levels, sprinklered: true });
  assert.equal(dry.storeys[0].checks.find((c) => c.id === "travel").status, "fail");
  assert.equal(wet.storeys[0].checks.find((c) => c.id === "travel").status, "pass");
  // 71 m2 holds at least 8 occupants: one exit is allowed on the first storey only within 22.86 m, so two are needed here.
  assert.equal(dry.storeys[0].checks.find((c) => c.id === "exits").status, "fail");
  // Two exits at the two ends of a straight corridor are a full diagonal apart.
  const both = building({ length: 71, storeys: 1, twoExits: true });
  const sep = codeCheck(both.graph, { exits: both.exits, levels: both.levels }).storeys[0].checks.find((c) => c.id === "separation");
  assert.equal(sep.status, "pass");
  // A short storey with few people may have one exit (Table 1006.3.2(2)).
  const short = building({ length: 10, storeys: 1 });
  assert.equal(codeCheck(short.graph, { exits: short.exits, levels: short.levels }).storeys[0].checks.find((c) => c.id === "exits").status, "pass");
});

test("high-rise from the highest occupied floor; exit door widths when the file has them", () => {
  const tall = building({ length: 5, storeys: 8 }); // top floor 24.5 m above the ground floor
  const r = codeCheck(tall.graph, { exits: tall.exits, levels: tall.levels, exitDoors: [{ widthM: 0.9 }, { widthM: 0.7 }] });
  assert.equal(r.building.highRise, true);
  assert.equal(r.building.provisions.length, 1);
  assert.equal(r.doors.status, "fail");
  assert.equal(codeCheck(tall.graph, { exits: tall.exits, levels: tall.levels }).doors.status, "cannot tell");
  // From the third storey up a single stair is never enough.
  assert.equal(r.storeys[2].checks.find((c) => c.id === "exits").status, "fail");
  for (const s of r.storeys) for (const c of s.checks) assert.match(c.clause, /\d/);
});

test("each limit is the stricter of SBC 801-CR-2018 and IBC 2021, so a pass meets both", () => {
  assert.equal(EGRESS_B.travel.unsprinklered, 60); // SBC; IBC allows 60.96
  assert.equal(EGRESS_B.travel.sprinklered, 90); // SBC; IBC 91.44
  assert.equal(EGRESS_B.occupantArea.value, 9); // SBC counts more people
  assert.equal(EGRESS_B.doorWidth.value, 0.813); // IBC
  assert.equal(EGRESS_B.highRise.value, 22.86); // IBC
  assert.equal(EGRESS_B.singleExit.first.travel, 22.86); // IBC
  assert.equal(EGRESS_B.singleExit.sprinkleredTravel, 30); // SBC
  assert.equal(EGRESS_B.highRiseProvisions[0].above, 22.86);
  for (const key of ["occupantArea", "travel", "exitsPerStorey", "singleExit", "separation", "doorWidth", "highRise"]) {
    assert.match(EGRESS_B[key].clause, /SBC 801-CR-2018/, key);
    assert.match(EGRESS_B[key].clause, /IBC 2021/, key);
  }
  // A door that meets SBC (800 mm) but not IBC (813 mm) fails.
  const { graph, exits, levels } = (() => {
    const nodes = [], edges = [];
    for (let x = 0; x < 6; x += 1) nodes.push(x, 0.75, 0);
    for (let x = 1; x < 6; x += 1) edges.push(x - 1, x);
    return { graph: { nodes: Float32Array.from(nodes), edges: Uint32Array.from(edges), kinds: new Uint8Array(6), upAxis: "y", agentHeight: 0.75, spacing: 1 }, exits: [0], levels: [0] };
  })();
  assert.equal(codeCheck(graph, { exits, levels, exitDoors: [{ widthM: 0.805 }] }).doors.status, "fail");
  assert.equal(codeCheck(graph, { exits, levels, exitDoors: [{ widthM: 0.82 }] }).doors.status, "pass");
  assert.ok(SBC2018_B.doorWidth.value < 0.805 && IBC2021_B.doorWidth.value > 0.805);
});
