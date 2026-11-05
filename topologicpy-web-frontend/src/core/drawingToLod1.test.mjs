/**
 * Floor-plan drawings to LOD1 (drawingToLod1.js) on synthetic DXF sheets
 * written here: a 10 x 6 m plan of three rooms behind 0.2 m walls drawn as
 * double lines, doors as blocks in gaps of the walls (one nested in another
 * block, one mirrored), room names as TEXT and MTEXT.
 *
 *   +-------------+---------------------+
 *   |             |  C (3.1..5.9)       |
 *   |  A          +----[door BC]--------+
 *   |  (0.1..3.9) [door AB]  B          |
 *   |             |  (4.1..9.9, 0.1..2.9)
 *   +-------------+-----------[exit]----+
 */
import assert from "node:assert/strict";
import test from "node:test";
import { decodeDxfBytes, draftRooms, draftSummary, mtextPlain, readDxf, roomClassFromNames, suggestLayers, toLod1Model } from "./drawingToLod1.js";
import { lod1ToEgressGeometry } from "./lod1.js";

/** A DXF file from group-code pairs. */
const dxfText = (pairs) => pairs.map(([c, v]) => `${c}\n${v}`).join("\n") + "\n";

/**
 * The sheet, at k drawing units per metre ($INSUNITS 6 for metres, 4 for
 * millimetres). texts: [text, x, y, kind] in metres; extra: more entities.
 */
function plan({ k = 1, units = 6, texts = [], extra = [], doorBC = true } = {}) {
  const P = (x) => x * k;
  const ent = [];
  const poly = (layer, pts, closed = false, linetype = null) => {
    ent.push([0, "LWPOLYLINE"], [8, layer], ...(linetype ? [[6, linetype]] : []), [90, pts.length], [70, closed ? 1 : 0]);
    for (const [x, y] of pts) ent.push([10, P(x)], [20, P(y)]);
  };
  const line = (layer, [x0, y0], [x1, y1]) => ent.push([0, "LINE"], [8, layer], [10, P(x0)], [20, P(y0)], [11, P(x1)], [21, P(y1)]);
  // Outer face (open at the exit, 8.0 to 8.9 on the south wall).
  poly("WALL", [[8.9, -0.1], [10.1, -0.1], [10.1, 6.1], [-0.1, 6.1], [-0.1, -0.1], [8.0, -0.1]]);
  // Room A, with the gap of door AB (x = 4, y 1.0 to 1.9).
  poly("WALL", [[3.9, 1.9], [3.9, 5.9], [0.1, 5.9], [0.1, 0.1], [3.9, 0.1], [3.9, 1.0]]);
  // Room B, with gaps for doors AB, BC (y = 3, x 6.0 to 6.9) and the exit.
  poly("WALL", [[4.1, 1.0], [4.1, 0.1], [8.0, 0.1]]);
  poly("WALL", [[8.9, 0.1], [9.9, 0.1], [9.9, 2.9], [6.9, 2.9]]);
  poly("WALL", [[6.0, 2.9], [4.1, 2.9], [4.1, 1.9]]);
  // Room C, as an old-style POLYLINE with VERTEX records.
  ent.push([0, "POLYLINE"], [8, "WALL"], [66, 1], [70, 0]);
  for (const [x, y] of [[6.0, 3.1], [4.1, 3.1], [4.1, 5.9], [9.9, 5.9], [9.9, 3.1], [6.9, 3.1]]) ent.push([0, "VERTEX"], [8, "WALL"], [10, P(x)], [20, P(y)]);
  ent.push([0, "SEQEND"]);
  // Jambs.
  for (const [a, b] of [[[3.9, 1.0], [4.1, 1.0]], [[3.9, 1.9], [4.1, 1.9]], [[6.0, 2.9], [6.0, 3.1]], [[6.9, 2.9], [6.9, 3.1]], [[8.0, -0.1], [8.0, 0.1]], [[8.9, -0.1], [8.9, 0.1]]]) line("WALL", a, b);
  // Doors: a block with its leaf drawn open along +y and its swing from +x to +y.
  ent.push([0, "INSERT"], [8, "DOORS"], [2, "DOOR90"], [10, P(4.0)], [20, P(1.0)], [41, k / 1000], [42, k / 1000], [50, 90]);
  if (doorBC) ent.push([0, "INSERT"], [8, "DOORS"], [2, "DOORSET"], [10, P(6.0)], [20, P(3.0)], [41, k], [42, k], [50, 0]);
  // The exit door mirrored (x scale -1): hinge at 8.9, closing towards 8.0.
  ent.push([0, "INSERT"], [8, "DOORS"], [2, "DOOR90"], [10, P(8.9)], [20, P(0.0)], [41, -k / 1000], [42, k / 1000], [50, 0]);
  for (const [text, x, y, kind = "TEXT"] of texts) {
    if (kind === "MTEXT") ent.push([0, "MTEXT"], [8, "TEXT"], [10, P(x)], [20, P(y)], [40, P(0.25)], [71, 5], [1, text]);
    else ent.push([0, "TEXT"], [8, "TEXT"], [10, P(x)], [20, P(y)], [40, P(0.25)], [72, 1], [11, P(x)], [21, P(y)], [1, text]);
  }
  for (const e of extra) ent.push(...e(P));
  // DOOR90 is drawn in millimetres (scaled by its inserts); DOORSET holds it in metres.
  const blocks = [
    [0, "BLOCK"], [8, "0"], [2, "DOOR90"], [70, 0], [10, 0], [20, 0],
    [0, "LINE"], [8, "0"], [10, 0], [20, 0], [11, 0], [21, 900],
    [0, "ARC"], [8, "0"], [10, 0], [20, 0], [40, 900], [50, 0], [51, 90],
    [0, "ENDBLK"],
    [0, "BLOCK"], [8, "0"], [2, "DOORSET"], [70, 0], [10, 0], [20, 0],
    [0, "INSERT"], [8, "0"], [2, "DOOR90"], [10, 0], [20, 0], [41, 0.001], [42, 0.001], [50, 0],
    [0, "ENDBLK"],
  ];
  return dxfText([
    [0, "SECTION"], [2, "HEADER"], [9, "$ACADVER"], [1, "AC1021"], [9, "$INSUNITS"], [70, units], [0, "ENDSEC"],
    [0, "SECTION"], [2, "TABLES"], [0, "TABLE"], [2, "LAYER"], [0, "LAYER"], [2, "WALL"], [62, 7], [70, 0], [6, "CONTINUOUS"], [0, "ENDTAB"], [0, "ENDSEC"],
    [0, "SECTION"], [2, "BLOCKS"], ...blocks, [0, "ENDSEC"],
    [0, "SECTION"], [2, "ENTITIES"], ...ent, [0, "ENDSEC"],
    [0, "EOF"],
  ]);
}

const OPTS = { wallLayers: ["WALL"], doorLayers: ["DOORS"], doorPattern: "^DOOR" };
const near = (a, b, tol = 0.02) => Math.abs(a - b) <= tol;
const roomWith = (draft, label) => draft.rooms.find((r) => r.labels.includes(label));

test("readDxf explodes nested and mirrored blocks and reads the units", () => {
  const d = readDxf(plan());
  assert.equal(d.units.name, "m");
  assert.equal(d.units.toMetres, 1);
  const arcs = d.entities.filter((e) => e.type === "arc");
  assert.equal(arcs.length, 3, "one swing per door, the nested one included");
  for (const a of arcs) assert.ok(near(a.r, 0.9, 1e-9));
  // The nested door: its hinge at (6, 3), its block chain two deep.
  const nested = arcs.find((a) => near(a.cx, 6.0) && near(a.cy, 3.0));
  assert.ok(nested, "the door inside DOORSET is exploded in place");
  assert.equal(d.entities[nested.owner].name, "DOOR90");
  assert.equal(d.entities[d.entities[nested.owner].owner].name, "DOORSET");
  assert.equal(nested.layer, "DOORS", "a part on layer 0 takes its insert's layer");
  // The mirrored door swings from the hinge at 8.9 towards 8.0.
  const mirrored = arcs.find((a) => near(a.cx, 8.9));
  const ends = [mirrored.points.slice(0, 2), mirrored.points.slice(-2)];
  assert.ok(ends.some(([x, y]) => near(x, 8.0) && near(y, 0)), "the closed position is at the other jamb");
  // The POLYLINE with VERTEX records.
  assert.ok(d.entities.some((e) => e.kind === "POLYLINE" && e.points.length === 12));
  assert.equal(mtextPlain("{\\fArial|b1;LIVING}\\PROOM \\H0.5;x"), "LIVING\nROOM x");
});

test("decodeDxfBytes: UTF-8 from AutoCAD 2007, the header's code page before", () => {
  const head = (ver, page) => dxfText([[0, "SECTION"], [2, "HEADER"], [9, "$ACADVER"], [1, ver], [9, "$DWGCODEPAGE"], [3, page], [0, "ENDSEC"]]);
  const utf8 = new TextEncoder().encode(head("AC1021", "ANSI_1252") + "1\nحمام\n");
  assert.ok(decodeDxfBytes(utf8).includes("حمام"));
  // "حمام" in Windows-1256 is CD E3 C7 E3.
  const ansi = Uint8Array.from([...new TextEncoder().encode(head("AC1015", "ANSI_1256") + "1\n"), 0xcd, 0xe3, 0xc7, 0xe3, 0x0a]);
  assert.ok(decodeDxfBytes(ansi).includes("حمام"));
});

test("three rooms with door gaps and door blocks", () => {
  const draft = draftRooms(readDxf(plan({ texts: [["BEDROOM", 2, 3], ["KITCHEN", 7, 1.5], ["{\\fArial|b1;LIVING}\\PROOM", 7, 4.5, "MTEXT"]] })).entities, OPTS);
  assert.equal(draft.rooms.length, 3, JSON.stringify(draft.rooms.map((r) => [r.label, r.area])));
  const A = roomWith(draft, "BEDROOM"), B = roomWith(draft, "KITCHEN"), C = roomWith(draft, "LIVING ROOM");
  assert.ok(A && B && C, "each room has its name");
  // Grown to the wall centre-lines: 4 x 6, 6 x 3, 6 x 3.
  assert.ok(near(A.area, 24, 0.05), `A ${A.area}`);
  assert.ok(near(B.area, 18, 0.05), `B ${B.area}`);
  assert.ok(near(C.area, 18, 0.05), `C ${C.area}`);
  assert.ok(near(A.innerArea, 3.8 * 5.8, 0.05), "the inner faces are kept too");
  assert.equal(A.class, "Unit");
  assert.equal(B.class, "Service");
  // Doors: A-B, B-C, and B to the outside.
  const pair = (d) => [...d.rooms].sort().join("|");
  const doors = draft.doors.filter((d) => d.kind === "door");
  assert.equal(doors.length, 3);
  assert.ok(doors.some((d) => pair(d) === [A.id, B.id].sort().join("|") && near(d.position[0], 4.0, 0.05) && near(d.position[1], 1.45, 0.05)));
  assert.ok(doors.some((d) => pair(d) === [B.id, C.id].sort().join("|") && near(d.position[1], 3.0, 0.05)));
  const exit = doors.find((d) => d.rooms.includes(null));
  assert.ok(exit && exit.rooms.includes(B.id) && exit.sides.includes("outside"));
  for (const d of doors) assert.ok(near(d.width, 0.9, 0.01));
  assert.ok(doors.every((d) => near(d.wall, 0.2, 0.01)), "the wall's thickness at each door");
  // Neighbours across the walls.
  assert.deepEqual([...A.neighbours].sort(), [B.id, C.id].sort());
  assert.equal(draft.issues.filter((i) => i.type.startsWith("door")).length, 0, JSON.stringify(draft.issues));
});

test("a room label, decoded from the Arabic keyboard font", () => {
  const texts = [["Hgplhl", 7, 1.5], ["+1.20", 7, 1.0], ["3.80x5.80", 2, 2], ["STAIR", 2, 3]];
  const draft = draftRooms(readDxf(plan({ texts })).entities, OPTS);
  const B = roomWith(draft, "الحمام");
  assert.ok(B, "'Hgplhl' reads as 'الحمام' (the bathroom)");
  assert.equal(B.label, "الحمام");
  assert.equal(B.class, "Toilet");
  assert.ok(!draft.rooms.some((r) => r.labels.includes("+1.20") || r.labels.includes("3.80x5.80")), "levels and dimensions are not names");
  assert.equal(roomWith(draft, "STAIR").class, "Stair");
  const C = draft.rooms.find((r) => !r.labels.length);
  assert.ok(draft.issues.some((i) => i.type === "no-label" && i.rooms.includes(C.id)));
  assert.equal(roomClassFromNames(["Guest bedroom"]), "Unit");
  assert.equal(roomClassFromNames(["Lift lobby"]), "Shaft", "the first class that matches wins");
});

test("millimetres: the scale from the door swings, the same rooms", () => {
  const texts = [["BEDROOM", 2, 3], ["KITCHEN", 7, 1.5], ["LIVING", 7, 4.5]];
  const mm = readDxf(plan({ k: 1000, units: 4, texts }));
  assert.equal(mm.units.name, "mm");
  const sug = suggestLayers(mm.entities, { units: mm.units });
  assert.equal(sug.scale, 0.001);
  assert.equal(sug.scaleSource, "door swings");
  assert.deepEqual(sug.walls, ["WALL"]);
  assert.deepEqual(sug.doors, ["DOORS"]);
  assert.deepEqual(sug.text, ["TEXT"]);
  assert.ok(sug.doorBlocks.includes("DOOR90"));
  const draft = draftRooms(mm.entities, { wallLayers: sug.walls, doorLayers: sug.doors, doorPattern: sug.doorPattern, scale: sug.scale });
  const areas = draft.rooms.map((r) => Math.round(r.area)).sort((a, b) => a - b);
  assert.deepEqual(areas, [18, 18, 24]);
  // A sheet in millimetres that declares metres (as these drawings do): the size gives it away.
  const liar = readDxf(plan({ k: 1000, units: 6, texts }));
  assert.equal(suggestLayers(liar.entities, { units: liar.units }).scale, 0.001);
});

test("dashed lines: a projection is not a wall, the head of an opening closes it", () => {
  const dashed = (pts, lt) => (P) => [[0, "LINE"], [8, "WALL"], [6, lt], [10, P(pts[0])], [20, P(pts[1])], [11, P(pts[2])], [21, P(pts[3])]];
  // A long hidden line across A and C (the floor below), and a head across a 1 m opening cut in B-C's wall.
  const texts = [["BEDROOM", 2, 3], ["KITCHEN", 7, 1.5], ["LIVING", 7, 4.5]];
  const draft = draftRooms(readDxf(plan({ texts, extra: [dashed([0.1, 4.5, 9.9, 4.5], "HIDDEN")] })).entities, OPTS);
  assert.equal(draft.rooms.length, 3, "the hidden line splits nothing");
  // Without its door block, the B-C gap joins the two rooms (unless their names split them there);
  // a dashed head across it closes it as an opening.
  const open = draftRooms(readDxf(plan({ texts, doorBC: false })).entities, { ...OPTS, split: false });
  assert.equal(open.rooms.length, 2);
  assert.ok(open.issues.some((i) => i.type === "several-labels"));
  const cut = draftRooms(readDxf(plan({ texts, doorBC: false })).entities, OPTS);
  assert.equal(cut.rooms.length, 3, "split by their names at the jambs");
  assert.ok(cut.doors.some((d) => d.kind === "opening" && d.source === "split"));
  const head = draftRooms(readDxf(plan({ texts, doorBC: false, extra: [dashed([6.0, 3.0, 6.9, 3.0], "DASHED")] })).entities, OPTS);
  assert.equal(head.rooms.length, 3);
  const opening = head.doors.find((d) => d.kind === "opening");
  assert.ok(opening, "the head is an opening");
  assert.deepEqual([...opening.rooms].sort(), [roomWith(head, "KITCHEN").id, roomWith(head, "LIVING").id].sort());
});

test("a room with two names is cut at the wall end between them", () => {
  // Two names in C, and a wall stub (double line) from C's north wall down to y = 4.2 at x = 7.
  const stub = (P) => [
    [0, "LWPOLYLINE"], [8, "WALL"], [90, 4], [70, 0], [10, P(6.9)], [20, P(5.9)], [10, P(6.9)], [20, P(4.2)], [10, P(7.1)], [20, P(4.2)], [10, P(7.1)], [20, P(5.9)],
  ];
  const draft = draftRooms(readDxf(plan({ texts: [["OFFICE", 5.5, 4.5], ["STORE", 8.5, 4.5]], extra: [stub] })).entities, OPTS);
  const office = roomWith(draft, "OFFICE"), store = roomWith(draft, "STORE");
  assert.ok(office && store && office.id !== store.id, "two rooms");
  assert.ok(near(office.area + store.area, 18, 0.15));
  const cut = draft.doors.find((d) => d.kind === "opening");
  assert.ok(cut, "the cut is an opening");
  assert.deepEqual([...cut.rooms].sort(), [office.id, store.id].sort());
  assert.ok(near(cut.position[0], 7.0, 0.15), `cut at ${cut.position}`);
  assert.equal(draft.stats.splits, 1);
});

test("draftSummary: rooms with area, names, centroid, neighbours and doors only", () => {
  const draft = draftRooms(readDxf(plan({ texts: [["BEDROOM", 2, 3], ["KITCHEN", 7, 1.5], ["LIVING", 7, 4.5]] })).entities, OPTS);
  const s = draftSummary(draft, { storey: "GF" });
  assert.equal(s.storey, "GF");
  assert.equal(s.rooms.length, 3);
  for (const r of s.rooms) {
    assert.deepEqual(Object.keys(r), ["id", "area", "label", "labels", "centroid", "neighbours", "doors"]);
    assert.equal(typeof r.id, "string");
    assert.equal(r.area, Math.round(r.area * 10) / 10);
    assert.ok(r.centroid.every((v) => v === Math.round(v * 10) / 10));
    for (const n of r.doors) assert.ok(r.neighbours.includes(n), "a room opened to is a neighbour");
  }
  const bed = s.rooms.find((r) => r.label === "BEDROOM"), kit = s.rooms.find((r) => r.label === "KITCHEN"), liv = s.rooms.find((r) => r.label === "LIVING");
  assert.deepEqual(bed.doors, [kit.id]);
  assert.deepEqual(bed.neighbours.sort(), [kit.id, liv.id].sort(), "A shares a wall with C but has no door to it");
  assert.deepEqual([...kit.doors].sort(), [bed.id, liv.id].sort());
  assert.deepEqual(bed.centroid, [2, 3]);
  assert.equal("storey" in draftSummary(draft), false);
});

test("two storeys through toLod1Model", () => {
  const gf = draftRooms(readDxf(plan({ texts: [["STAIR", 2, 3], ["LOBBY", 7, 1.5], ["OFFICE", 7, 4.5]] })).entities, OPTS);
  const f1 = draftRooms(readDxf(plan({ texts: [["STAIR", 2, 3], ["CORRIDOR", 7, 1.5], ["BEDROOM", 7, 4.5]] })).entities, OPTS);
  // A review: rename one room, set the class of another, remove nothing.
  roomWith(f1, "BEDROOM").name = "Guest room";
  roomWith(gf, "OFFICE").class = "Office";
  const levels = [{ id: "F1", name: "First floor", ffl: 3.2, height: 3.0 }, { id: "GF", name: "Ground floor", ffl: 0, height: 3.2 }];
  const model = toLod1Model([{ level: "GF", draft: gf }, { level: "F1", draft: f1 }], levels, { name: "test" });
  assert.deepEqual(model.levels.map((l) => l.id), ["GF", "F1"], "ascending floor level");
  assert.equal(model.rooms.length, 6);
  for (const r of model.rooms) {
    assert.ok(r.area > 17 && r.area < 24.1);
    assert.ok(r.footprint.length >= 4);
    let a = 0;
    for (let i = 0; i < r.footprint.length; i += 1) {
      const p = r.footprint[i], q = r.footprint[(i + 1) % r.footprint.length];
      a += p[0] * q[1] - q[0] * p[1];
    }
    assert.ok(a > 0, "footprints run anticlockwise");
    assert.equal(r.ffl, r.level === "GF" ? 0 : 3.2);
    assert.equal(r.height, r.level === "GF" ? 3.2 : 3.0);
  }
  const room = (level, name) => model.rooms.find((r) => r.level === level && r.name === name);
  assert.equal(room("GF", "OFFICE").class, "Office");
  assert.ok(room("F1", "Guest room"), "the review's name");
  assert.equal(room("GF", "STAIR").class, "Stair");
  assert.equal(room("GF", "STAIR").id.split(":")[0], "GF");
  // Doors: three a storey; the ground floor's door to the outside is an exit.
  assert.equal(model.doors.length, 6);
  const exits = model.doors.filter((d) => d.kind === "exit");
  assert.equal(exits.length, 1);
  assert.equal(exits[0].level, "GF");
  assert.ok(exits[0].rooms.includes(room("GF", "LOBBY").id) && exits[0].rooms.includes(null));
  assert.ok(model.doors.filter((d) => d.level === "F1").every((d) => d.kind === "door" && d.position[2] === 3.2));
  for (const d of model.doors) assert.ok(d.direction && near(Math.hypot(...d.direction), 1, 1e-6));
  // The stair over the stair.
  assert.equal(model.links.length, 1);
  assert.deepEqual(model.links[0].rooms, [room("GF", "STAIR").id, room("F1", "STAIR").id]);
  assert.equal(model.links[0].kind, "stair");
  assert.equal(model.meta.source, "dxf-draft");
  assert.equal(model.meta.boundary, "shared");
  assert.deepEqual(model.meta.counts, { levels: 2, rooms: 6, doors: 6, links: 1, kinds: { door: 5, exit: 1, stair: 1 } });
  // lod1.js builds the egress geometry from it.
  assert.doesNotThrow(() => lod1ToEgressGeometry(model));
});
