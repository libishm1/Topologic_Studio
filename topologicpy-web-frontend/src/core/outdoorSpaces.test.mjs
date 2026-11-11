// Spaces left out of the walkable floors (core/outdoorSpaces.js): roofs
// drawn as rooms that no door opens onto, and voids.
import test from "node:test";
import assert from "node:assert/strict";

import { VOID_NAME, outdoorSpaces } from "./outdoorSpaces.js";

/** A box mesh over the plan rectangle [x0, z0, x1, z1] from y0 to y1, faces down at y0 and up at y1. */
function box(itemId, [x0, z0, x1, z1], y0, y1) {
  const positions = [], indices = [];
  const vertex = (x, y, z) => {
    positions.push(x, y, z);
    return positions.length / 3 - 1;
  };
  for (const [y, down] of [[y0, true], [y1, false]]) {
    const a = vertex(x0, y, z0), b = vertex(x1, y, z0), d = vertex(x1, y, z1), e = vertex(x0, y, z1);
    if (down) indices.push(a, b, d, a, d, e);
    else indices.push(a, d, b, a, e, d);
  }
  return { itemId, positions: Float32Array.from(positions), indices: Uint32Array.from(indices) };
}

/**
 * A storey's room under a slab, a roof space on the slab with a parapet
 * 1 m high round it, a balcony beside the room at its floor's level with a
 * door onto it, and a space named as a void over the room.
 */
function building() {
  const names = new Map([[1, "Office"], [2, "ROOF"], [3, "Balcony"], [4, "OPEN TO BELOW"], [5, "Plant"]]);
  const spaces = [
    box(1, [0, 0, 6, 4], 0, 3),
    box(2, [0, 0, 6, 4], 3.2, 6.2),
    box(3, [0, 4.1, 6, 5.5], 0, 3),
    box(4, [2, 1, 4, 3], 3.2, 6.2),
    // A plant room on the roof with walls round it 2.6 m high: a room whose slab is not in the model.
    box(5, [10, 0, 14, 4], 3.2, 5.8),
  ];
  const overhead = [box("slab", [-0.2, -0.2, 6.2, 4.1], 3.0, 3.2)];
  const walls = [
    box("wall", [-0.2, -0.2, 6.2, 0], 0, 3.2),
    // The parapet: 1 m over the roof.
    box("parapet", [-0.2, -0.2, 6.2, 0], 3.2, 4.2),
    box("plant-wall", [9.8, -0.2, 14.2, 0], 3.2, 5.8),
  ];
  // The balcony door: in the room's wall on the balcony side, its sill at the floor.
  const doors = [box("door", [2.5, 3.95, 3.4, 4.15], 0, 2.1)];
  return { spaces, names, overhead, walls, doors };
}

test("a roof drawn as a room with only a parapet round it and no door onto it is not walked on", () => {
  const { spaces, names, overhead, walls, doors } = building();
  const out = outdoorSpaces(spaces, { names, overhead, walls, doors });
  const roof = out.find((s) => s.id === 2);
  assert.ok(roof, "the roof is left out");
  assert.match(roof.why, /outside the building with no door onto it/);
  assert.ok(!out.some((s) => s.id === 1), "the room under the slab stays");
  assert.ok(!out.some((s) => s.id === 5), "a plant room with walls round it stays");
});

test("a balcony with a door onto it stays walkable, so its door is not taken for an exit to the open air", () => {
  const { spaces, names, overhead, walls, doors } = building();
  assert.ok(!outdoorSpaces(spaces, { names, overhead, walls, doors }).some((s) => s.id === 3));
  // Without the door it is left out, as the roof is.
  assert.ok(outdoorSpaces(spaces, { names, overhead, walls, doors: [] }).some((s) => s.id === 3));
  // A door a storey up does not open onto it.
  const upstairs = [box("door", [2.5, 3.95, 3.4, 4.15], 3.2, 5.3)];
  assert.ok(outdoorSpaces(spaces, { names, overhead, walls, doors: upstairs }).some((s) => s.id === 3));
});

test("a space named as a void is not walked on, whatever is over it", () => {
  const { spaces, names, overhead, walls, doors } = building();
  const out = outdoorSpaces(spaces, { names, overhead, walls, doors });
  assert.match(out.find((s) => s.id === 4)?.why || "", /void/);
  assert.equal(out.filter((s) => s.id === 4).length, 1, "listed once");
  for (const name of ["OPEN TO BELOW", "Void", "Luftraum"]) assert.ok(VOID_NAME.test(name), name);
  for (const name of ["Office", "Avoidance lobby", "Lobby"]) assert.ok(!VOID_NAME.test(name), name);
});

test("no spaces, nothing left out", () => {
  assert.deepEqual(outdoorSpaces([], {}), []);
  assert.deepEqual(outdoorSpaces(null, {}), []);
});
