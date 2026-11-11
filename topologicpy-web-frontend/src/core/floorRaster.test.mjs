// Run with: node --test src/core/floorRaster.test.mjs
// The measuring grid on box-mesh floors with known answers: the exact
// distance transform against brute force, corridor widths, the 16-neighbour
// walk and Fast Marching against the straight line, and changes of level.
import test from "node:test";
import assert from "node:assert/strict";
import { SIXTEEN_BOUND, WIDTH_CLASSES, cellsNear, compareFields, edt, fastMarching, gridField, opening, storeyRaster, widthClasses } from "./floorRaster.js";

/** A closed box mesh in the viewer frame (y up). */
function box(x0, y0, z0, x1, y1, z1, itemId = 0) {
  const positions = Float32Array.from([x0, y0, z0, x1, y0, z0, x1, y0, z1, x0, y0, z1, x0, y1, z0, x1, y1, z0, x1, y1, z1, x0, y1, z1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]);
  return { itemId, positions, indices };
}

test("the distance transform is exact: it matches brute force on a random mask", () => {
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const w = 23, h = 17;
  const inside = new Uint8Array(w * h).map(() => (rand() < 0.85 ? 1 : 0));
  const got = edt(inside, w, h);
  for (let k = 0; k < w * h; k += 1) {
    let best = Infinity;
    for (let q = 0; q < w * h; q += 1) if (!inside[q]) best = Math.min(best, (k % w - (q % w)) ** 2 + (Math.floor(k / w) - Math.floor(q / w)) ** 2);
    assert.equal(got[k], inside[k] ? best : 0, `cell ${k}`);
  }
});

/** A corridor `width` m wide along x, 10 m long, walls 0.2 m thick, on a slab. */
function corridor(width, extra = []) {
  return {
    floors: [box(-1, -0.2, -1, 11, 0, width + 1)],
    walls: [box(-0.5, 0, -0.2, 10.5, 3, 0), box(-0.5, 0, width, 10.5, 3, width + 0.2), box(-0.2, 0, 0, 0, 3, width), box(10, 0, 0, 10.2, 3, width), ...extra],
  };
}

test("a 1.60 m corridor leaves exactly a 1.10 m strip for a body's centre (Poux's known answer), and twice the clearance is the width", () => {
  const R = storeyRaster(corridor(1.6), { level: 0, seeds: [[5, 0.8]], cell: 0.05 });
  const i = Math.floor((5 - R.a0) / R.cell);
  let strip = 0, widest = 0;
  for (let j = 0; j < R.h; j += 1) {
    const k = j * R.w + i;
    strip += R.body[k];
    widest = Math.max(widest, R.clear[k]);
  }
  assert.ok(Math.abs(strip * R.cell - 1.1) < 1e-6, `${strip * R.cell} m`);
  // An even number of cells across: twice the centre's clearance is the width less a cell.
  assert.ok(Math.abs(2 * widest - 1.6) <= R.cell + 1e-6, `${2 * widest} m`);
});

test("clear-width classes: an 0.85 m passage passes the 0.80 m disc and not the 0.90 m one", () => {
  const R = storeyRaster(corridor(0.85), { level: 0, seeds: [[5, 0.4]], cell: 0.05 });
  const classes = widthClasses(R);
  const k = Math.floor((0.425 - R.b0) / R.cell) * R.w + Math.floor((5 - R.a0) / R.cell);
  assert.equal(WIDTH_CLASSES[classes[k]], 0.815);
  const wide = opening(R, R.clear, 0.45);
  assert.equal(wide[k], 0, "no 0.90 m disc fits");
});

test("in the open, the 16-neighbour walk is never short and at most 2.75 % long; Fast Marching is within a few per cent", () => {
  // Seeds at the corners: the grid spans the seeds and a metre round them.
  const R = storeyRaster({ floors: [box(-1, -0.2, -1, 21, 0, 21)], walls: [] }, { level: 0, seeds: [[0, 0], [20, 20]], cell: 0.1 });
  // The four cells round the point (their centres 7 cm off it).
  const src = cellsNear(R, 10, 10, 0.08);
  assert.equal(src.cells.length, 4);
  const f = gridField(R, [{ ...src, id: 0 }]);
  const T = fastMarching(R, [{ ...src, id: 0 }]);
  let worst = 1, fmWorst = 0;
  for (let k = 0; k < R.w * R.h; k += 1) {
    if (!R.walk[k] || !Number.isFinite(f.dist[k])) continue;
    const [x, y] = [R.a0 + ((k % R.w) + 0.5) * R.cell, R.b0 + (Math.floor(k / R.w) + 0.5) * R.cell];
    const e = Math.hypot(x - 10, y - 10);
    if (e < 2) continue;
    const ratio = f.dist[k] / e;
    assert.ok(ratio >= 1 - 0.02, `short at ${x}, ${y}: ${ratio}`);
    worst = Math.max(worst, ratio);
    fmWorst = Math.max(fmWorst, Math.abs(T[k] / e - 1));
  }
  assert.ok(worst <= SIXTEEN_BOUND + 0.01, `${worst} against ${SIXTEEN_BOUND}`);
  assert.ok(worst > 1.01, "the bias is there to see");
  assert.ok(fmWorst < 0.06, `${fmWorst}`);
  const cmp = compareFields(f.dist, T, R.walk);
  assert.ok(Math.abs(cmp.mean) < 0.03, JSON.stringify(cmp));
});

test("a 15 cm step: walking crosses it, a wheelchair (13 mm, 1 in 12) does not", () => {
  const geometry = { floors: [box(-1, -0.2, -1, 5, 0, 3), box(5, -0.2, -1, 11, 0.15, 3)], walls: [] };
  const R = storeyRaster(geometry, { level: 0, seeds: [[2, 1], [8, 1]], cell: 0.05 });
  const start = cellsNear(R, 1, 1, 0.1);
  const walk = gridField(R, [{ ...start, id: 0 }]);
  const chair = gridField(R, [{ ...start, id: 0 }], { levelChange: 0.013, slope: 1 / 12 });
  // The grid spans the seeds and a metre round them: x from 1 to 9.
  const far = Math.floor((1 - R.b0) / R.cell) * R.w + Math.floor((8 - R.a0) / R.cell);
  assert.ok(Number.isFinite(walk.dist[far]), "a step is walked");
  assert.ok(!Number.isFinite(chair.dist[far]), "a wheelchair stops at it");
});

test("source cells start with their own step to the exit, not at nought", () => {
  const R = storeyRaster(corridor(1.6), { level: 0, seeds: [[0.5, 0.8], [9.5, 0.8]], cell: 0.05 });
  const near = cellsNear(R, 0.1, 0.8, 0.75);
  assert.ok(near.cells.length > 100);
  const f = gridField(R, [{ ...near, id: 0 }]);
  const k = near.cells[near.offsets.indexOf(Math.max(...near.offsets))];
  assert.ok(f.dist[k] > 0.6, `${f.dist[k]}`);
});
