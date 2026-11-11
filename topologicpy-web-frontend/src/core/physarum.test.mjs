// Jones's slime mould models on a grid (core/physarum.js): the network,
// erosion and chemotaxis models of wiki/research/physarum-jones-models-2026-11-11.md.
import test from "node:test";
import assert from "node:assert/strict";

import {
  EROSION_DEFAULTS,
  NETWORK_DEFAULTS,
  createChemotaxis,
  createPhysarum,
  diffuse,
  openWall,
  stepChemotaxis,
  stepPhysarum,
} from "./physarum.js";

/** A width x height grid walled round its border only. */
function arena(width, height) {
  const walls = new Uint8Array(width * height);
  for (let x = 0; x < width; x += 1) walls[x] = walls[(height - 1) * width + x] = 1;
  for (let y = 0; y < height; y += 1) walls[y * width] = walls[y * width + width - 1] = 1;
  return walls;
}

/** The free cells of a grid. */
function freeCells(walls) {
  return walls.reduce((n, w) => n + (w ? 0 : 1), 0);
}

/** The sum of a field, in doubles. */
function total(field) {
  let t = 0;
  for (const v of field) t += v;
  return t;
}

/** Throws unless every agent stands on a free cell of its own and `occupied` says so, both ways. */
function checkAgents(s, label) {
  let held = 0;
  for (let c = 0; c < s.occupied.length; c += 1) {
    const i = s.occupied[c];
    if (i < 0) continue;
    held += 1;
    if (i >= s.count || s.cell[i] !== c) assert.fail(`${label}: cell ${c} names agent ${i}, which is elsewhere`);
  }
  assert.equal(held, s.count, `${label}: ${held} cells held for ${s.count} agents`);
  for (let i = 0; i < s.count; i += 1) {
    const c = s.cell[i];
    if (s.walls[c]) assert.fail(`${label}: agent ${i} on a wall at ${c}`);
    if (s.occupied[c] !== i) assert.fail(`${label}: agent ${i} shares cell ${c} with ${s.occupied[c]}`);
  }
}

test("two food sources draw a strand of trail along the line between them", () => {
  // Food 30 cells apart on one of two lines, y 12 or y 28. The strand is
  // judged on its middle, 8 cells or more from either source, where the
  // food's own diffusion leaves little. Without food the agents still gather
  // into strands of their own, often along the arena's long axis, so the
  // test asks also that the line carries the strand only when the food is on
  // it. With SO 5, SA 45, RA 45 and 250 agents, on 30 seeds and both lines
  // the middle was at least 3.8 times the rest (the median 6.1), and at
  // least 34 times (the median 73) the same line's with the food on the other.
  const W = 70, H = 40, lines = [12, 28];
  const walls = arena(W, H);
  const measure = (s, line) => {
    let mid = 0, nMid = 0, rest = 0, nRest = 0;
    for (let y = 0; y < H; y += 1) {
      for (let x = 0; x < W; x += 1) {
        const i = y * W + x;
        if (walls[i]) continue;
        const band = Math.abs(y - line) <= 1;
        if (band && x >= 28 && x <= 42) {
          mid += s.trail[i];
          nMid += 1;
        } else if (!(band && x >= 20 && x <= 50)) {
          rest += s.trail[i];
          nRest += 1;
        }
      }
    }
    return { mid: mid / nMid, rest: rest / nRest };
  };
  for (const seed of [1, 2]) {
    const runs = lines.map((line) =>
      stepPhysarum(createPhysarum({ width: W, height: H, walls, food: [[20, line], [50, line]], agents: 250, seed, so: 5, sa: 45, ra: 45 }), 1000),
    );
    lines.forEach((line, k) => {
      const { mid, rest } = measure(runs[k], line);
      assert.ok(mid >= 3 * rest, `seed ${seed}, y ${line}: the strand's middle ${mid.toFixed(2)} is under 3x the rest's ${rest.toFixed(2)}`);
      const away = measure(runs[1 - k], line).mid;
      assert.ok(mid >= 10 * away, `seed ${seed}, y ${line}: ${mid.toFixed(2)} with the food there, ${away.toFixed(2)} with it on the other line`);
    });
  }
});

test("every step, no agent stands on a wall or on another's cell, and occupied agrees", () => {
  // Pillars in the arena; growth, deaths, random turns and jumps all on.
  const W = 48, H = 40;
  const walls = arena(W, H);
  for (let y = 8; y < 32; y += 1) for (const x of [15, 16, 31, 32]) if (y % 10 < 7) walls[y * W + x] = 1;
  const runs = [
    ["network", createPhysarum({ width: W, height: H, walls, agents: 400, food: [[8, 20], [40, 20]], adaptive: true, so: 9, sa: 22.5, ra: 45, pCD: 0.05, pTP: 0.01, seed: 3 }), stepPhysarum],
    ["erosion", createPhysarum({ width: W, height: H, walls, agents: 600, mode: "erosion", substrate: new Float32Array(W * H).fill(50), adaptive: true, every: 3, seed: 4 }), stepPhysarum],
    ["chemotaxis", createChemotaxis({ width: W, height: H, walls, exit: [44, 20], start: [5, 20, 4], agents: 40, seed: 5 }), stepChemotaxis],
  ];
  for (const [label, s, step] of runs) {
    checkAgents(s, `${label} at the start`);
    for (let t = 0; t < 300; t += 1) checkAgents(step(s), `${label} after step ${t + 1}`);
  }
  // The checks met every way an agent changes cells.
  const [, net] = runs[0], [, ero] = runs[1], [, chem] = runs[2];
  assert.ok(net.born > 0 && net.died > 0 && net.teleported > 0 && net.blocked > 0 && net.moved > 0);
  assert.ok(ero.teleported > 0 && ero.moved > 0);
  assert.ok(chem.arrived > 0);
});

test("a step past the border reflects the heading; a sensor past it turns the agent by RA", () => {
  const W = 10, H = 10;
  // Heading east from the last column: reflected west, and the step taken that way.
  const s = createPhysarum({ width: W, height: H, agents: 1, start: [9, 5, 0] });
  s.heading[0] = 0;
  stepPhysarum(s);
  assert.equal(s.cell[0], 5 * W + 8);
  assert.ok(Math.abs(s.heading[0] - Math.PI) < 1e-6, `heading ${s.heading[0]}`);
  // Heading north from row 1: the step reaches row 0, then the straight-on
  // sensor, 3 cells further, is past the border, so the heading turns by +RA.
  const u = createPhysarum({ width: W, height: H, agents: 1, start: [5, 1, 0] });
  u.heading[0] = 1.5 * Math.PI;
  stepPhysarum(u);
  assert.equal(u.cell[0], 5);
  assert.ok(Math.abs(u.heading[0] - (1.5 * Math.PI + NETWORK_DEFAULTS.ra * (Math.PI / 180))) < 1e-6, `heading ${u.heading[0]}`);
});

test("diffusion keeps an interior lump's total, takes (1 - damp) of it, and walls and the border absorb", () => {
  const W = 21, H = 21;
  const lump = () => {
    const f = new Float32Array(W * H);
    f[10 * W + 10] = 100;
    f[10 * W + 11] = 37.5;
    f[9 * W + 10] = 12.25;
    return f;
  };
  const before = total(lump());
  const conserved = diffuse(lump(), W, H, null, 0);
  assert.ok(Math.abs(total(conserved) - before) <= 1e-5 * before, `damp 0: ${total(conserved)} from ${before}`);
  assert.ok(conserved[10 * W + 10] < 100 && conserved[11 * W + 11] > 0 && conserved[12 * W + 12] === 0, "the lump spreads a cell a step");
  const damped = diffuse(lump(), W, H, null, 0.1);
  assert.ok(Math.abs(total(damped) - 0.9 * before) <= 1e-5 * before, `damp 0.1: ${total(damped)}, want ${0.9 * before}`);
  // The same through a step of the model with no agents.
  const s = createPhysarum({ width: W, height: H, agents: 0, damp: 0.25 });
  s.trail.set(lump());
  stepPhysarum(s);
  assert.ok(Math.abs(total(s.trail) - 0.75 * before) <= 1e-5 * before);

  // A unit in a corner: the border ring absorbs, as in the author's code, so
  // only the one cell of its 3x3 off the border keeps a ninth of it.
  const corner = new Float32Array(W * H);
  corner[0] = 9;
  const spread = diffuse(corner, W, H, null, 0);
  assert.ok(Math.abs(total(spread) - 1) < 1e-5 && Math.abs(spread[W + 1] - 1) < 1e-5);
  // With wrap-around the grid is a torus: nothing is absorbed and the corner's 3x3 wraps round.
  const round = new Float32Array(W * H);
  round[0] = 9;
  const wrapped = diffuse(round, W, H, null, 0, undefined, { wrap: true });
  assert.ok(Math.abs(total(wrapped) - 9) < 1e-5 && Math.abs(wrapped[(H - 1) * W + (W - 1)] - 1) < 1e-5);
  const damp = new Float32Array(W * H);
  damp[0] = 9;
  assert.ok(Math.abs(total(diffuse(damp, W, H, null, 0.2, undefined, { wrap: true })) - 7.2) < 1e-5, "on a torus the total falls by exactly (1 - damp)");

  // A wall beside the lump ends at zero, even holding trail before, and the total falls.
  const walls = new Uint8Array(W * H);
  for (let y = 0; y < H; y += 1) walls[y * W + 12] = 1;
  const f = lump();
  f[10 * W + 12] = 50;
  diffuse(f, W, H, walls, 0);
  for (let y = 0; y < H; y += 1) assert.equal(f[y * W + 12], 0);
  assert.ok(total(f) < before - 1, "the wall absorbed what reached it");
  // Nothing crosses it.
  for (let y = 0; y < H; y += 1) for (let x = 13; x < W; x += 1) assert.equal(f[y * W + x], 0);
});

test("the adaptive population grows from a small patch and stays within the bound its windows set", () => {
  // Jones 2011's growth (GW 9, (0, 10]) and shrinkage (SW 5, [0, 23]) with
  // the coarse sensors. A bud needs at most gMax agents in its growth window,
  // so the bound is twice the density that fills every window to gMax + 1:
  // on 8 seeds, from step 1000 to 5000, the most seen was 913 of its 1044.
  const W = 64, H = 64;
  const walls = arena(W, H);
  const p = NETWORK_DEFAULTS;
  const bound = (2 * freeCells(walls) * (p.gMax + 1)) / (p.gw * p.gw);
  const coarse = { so: 9, sa: 22.5, ra: 45 };
  for (const seed of [1, 2]) {
    const s = createPhysarum({ width: W, height: H, walls, agents: 20, start: [32, 32, 3], adaptive: true, seed, ...coarse });
    assert.equal(s.count, 20);
    stepPhysarum(s, 500);
    assert.ok(s.count >= 200, `seed ${seed}: ${s.count} after 500 steps from 20`);
    let most = 0;
    for (let t = 500; t < 2000; t += 10) most = Math.max(most, stepPhysarum(s, 10).count);
    assert.ok(most <= bound, `seed ${seed}: ${most} above the bound ${bound.toFixed(0)}`);
    assert.ok(s.count > 50, `seed ${seed}: died out (${s.count})`);
  }
  // With a growth window that never refuses, the same start runs past it.
  const loose = createPhysarum({ width: W, height: H, walls, agents: 20, start: [32, 32, 3], adaptive: true, seed: 1, ...coarse, gMax: p.gw * p.gw });
  stepPhysarum(loose, 300);
  assert.ok(loose.count > 2 * bound, `without the growth limit only ${loose.count}`);
});

test("with shrinkage limits that forbid crowds, agents in a packed block die", () => {
  const W = 40, H = 40;
  const walls = arena(W, H);
  // Tested every step (the author's every 5 falls on step 5, 10, ...).
  const block = (sMax) => createPhysarum({ width: W, height: H, walls, agents: 10000, start: [20, 20, 6], adaptive: true, every: 1, gMax: 0, sMax, seed: 7 });
  const packed = block(8);
  const n = packed.count;
  // A disc of radius 6, every cell taken.
  assert.equal(n, 113);
  stepPhysarum(packed);
  assert.equal(packed.born, 0);
  assert.ok(packed.died >= n / 2, `only ${packed.died} of ${n} died`);
  assert.equal(packed.count, n - packed.died);
  for (let y = 18; y <= 22; y += 1) for (let x = 18; x <= 22; x += 1) assert.equal(packed.occupied[y * W + x], -1, `(${x}, ${y}) survived`);
  // The default limit (more than 23 of 25) takes only the block's core.
  const usual = block(NETWORK_DEFAULTS.sMax);
  stepPhysarum(usual);
  assert.ok(usual.died > 0 && usual.died < packed.died, `${usual.died} died at the default limit, ${packed.died} at 8`);
  // And scattered agents none.
  const sparse = createPhysarum({ width: W, height: H, walls, agents: 60, adaptive: true, every: 1, gMax: 0, seed: 7 });
  stepPhysarum(sparse);
  assert.equal(sparse.died, 0);
});

test("erosion: agents lay no trail, each step wears exactly erodeBy, never below zero, and jumps come at pTP", () => {
  const W = 50, H = 50;
  const walls = arena(W, H);
  const field = (v) => {
    const f = new Float32Array(W * H);
    for (let i = 0; i < W * H; i += 1) if (!walls[i]) f[i] = v;
    return f;
  };
  assert.equal(EROSION_DEFAULTS.depT, 0);
  // With nothing projected, the trail stays at zero however much the agents move.
  const quiet = createPhysarum({ width: W, height: H, walls, agents: 300, mode: "erosion", substrate: field(1000), projectScale: 0, seed: 1 });
  stepPhysarum(quiet, 50);
  assert.ok(quiet.moved > 1000);
  assert.ok(quiet.trail.every((v) => v === 0));

  // Every step taken wears erodeBy off: the substrate never rises, and its
  // total falls by erodeBy a step taken (a cell is entered at most once a
  // step, so 1000 lasts the 60 steps without reaching zero).
  const s = createPhysarum({ width: W, height: H, walls, agents: 400, mode: "erosion", substrate: field(1000), seed: 2 });
  let previous = Float32Array.from(s.substrate);
  for (let t = 0; t < 60; t += 1) {
    const moved = s.moved;
    stepPhysarum(s);
    for (let i = 0; i < W * H; i += 1) if (s.substrate[i] > previous[i]) assert.fail(`step ${t}: cell ${i} rose`);
    assert.equal(total(previous) - total(s.substrate), EROSION_DEFAULTS.erodeBy * (s.moved - moved), `step ${t}`);
    previous = Float32Array.from(s.substrate);
  }
  assert.ok(s.moved > 10000);

  // Thin substrate wears to zero and stops there.
  const thin = createPhysarum({ width: W, height: H, walls, agents: 400, mode: "erosion", substrate: field(25), seed: 3 });
  stepPhysarum(thin, 300);
  let zero = 0;
  for (let i = 0; i < W * H; i += 1) {
    assert.ok(thin.substrate[i] >= 0);
    if (!walls[i] && thin.substrate[i] === 0) zero += 1;
  }
  assert.ok(zero > freeCells(walls) / 2, `only ${zero} free cells worn through`);

  // Jumps: about pTP of the agent-steps (2000 expected, a standard deviation of 45).
  const j = createPhysarum({ width: W, height: H, walls, agents: 1000, mode: "erosion", substrate: field(1000), seed: 4 });
  stepPhysarum(j, 200);
  const rate = j.teleported / (200 * 1000);
  assert.ok(Math.abs(rate - EROSION_DEFAULTS.pTP) < 0.15 * EROSION_DEFAULTS.pTP, `jump rate ${rate}`);
});

test("erosion agents seek the low: they gather in a channel already worn", () => {
  // A band 5 cells high at zero across substrate of 1000: it projects no
  // trail, so the agents turning towards the least reading collect there
  // (1.7 to 2 times its share of the floor after 100 steps on 4 seeds;
  // wandering would give 1).
  const W = 50, H = 50;
  const walls = arena(W, H);
  const substrate = new Float32Array(W * H);
  for (let y = 1; y < H - 1; y += 1) for (let x = 1; x < W - 1; x += 1) substrate[y * W + x] = Math.abs(y - 25) <= 2 ? 0 : 1000;
  const share = (5 * (W - 2)) / freeCells(walls);
  for (const seed of [1, 2]) {
    const s = stepPhysarum(createPhysarum({ width: W, height: H, walls, agents: 200, mode: "erosion", substrate, seed }), 100);
    let inside = 0;
    for (let i = 0; i < s.count; i += 1) if (Math.abs(Math.floor(s.cell[i] / W) - 25) <= 2) inside += 1;
    assert.ok(inside / s.count >= 1.5 * share, `seed ${seed}: ${inside} of ${s.count} in the channel, ${(share * 100).toFixed(0)}% of the floor`);
  }
});

test("chemotaxis takes the short corridor through a maze, and a shorter one when a wall opens", () => {
  // A start shaft (left) and an exit shaft (right) joined by an upper
  // corridor (about 52 cells from start to exit) and a lower one (about 72),
  // with a middle corridor (36) plugged by a wall one cell thick.
  const W = 45, H = 36;
  const maze = new Uint8Array(W * H).fill(1);
  const carve = (x0, y0, x1, y1) => {
    for (let y = y0; y <= y1; y += 1) for (let x = x0; x <= x1; x += 1) maze[y * W + x] = 0;
  };
  carve(2, 3, 6, 32);
  carve(38, 3, 42, 32);
  carve(2, 3, 42, 5);
  carve(2, 30, 42, 32);
  carve(7, 11, 37, 13);
  for (let y = 11; y <= 13; y += 1) maze[y * W + 22] = 1;
  // The three corridors' runs between the shafts, 29 x 3 cells each.
  const along = (s, y0) => {
    let t = 0;
    for (let y = y0; y < y0 + 3; y += 1) for (let x = 8; x <= 36; x += 1) t += s.trail[y * W + x];
    return t;
  };
  for (const seed of [1, 2]) {
    const s = createChemotaxis({ width: W, height: H, walls: maze, exit: [40, 12], start: [4, 12, 3], agents: 20, seed });
    stepChemotaxis(s, 600);
    const upper = along(s, 3), lower = along(s, 30);
    assert.ok(s.arrived > 100, `seed ${seed}: ${s.arrived} arrived`);
    assert.ok(upper > 0 && upper > 10 * lower, `seed ${seed}: upper ${upper.toFixed(1)}, lower ${lower.toFixed(1)}`);
    for (let y = 11; y <= 13; y += 1) openWall(s, 22, y);
    stepChemotaxis(s, 400);
    const middle = along(s, 11), upperAfter = along(s, 3);
    assert.ok(middle > upperAfter, `seed ${seed}: middle ${middle.toFixed(1)}, upper ${upperAfter.toFixed(1)}`);
    // The old way fades.
    assert.ok(upperAfter < 0.1 * upper, `seed ${seed}: upper still ${upperAfter.toFixed(1)} of ${upper.toFixed(1)}`);
  }
});

test("one seed gives one run, another a different one", () => {
  const W = 40, H = 30;
  const walls = arena(W, H);
  const runs = {
    network: (seed) => stepPhysarum(createPhysarum({ width: W, height: H, walls, agents: 150, food: [[10, 15], [30, 15]], adaptive: true, pCD: 0.02, seed }), 120),
    erosion: (seed) => stepPhysarum(createPhysarum({ width: W, height: H, walls, agents: 150, mode: "erosion", substrate: new Float32Array(W * H).fill(40), seed }), 120),
    chemotaxis: (seed) => stepChemotaxis(createChemotaxis({ width: W, height: H, walls, exit: [35, 15], start: [5, 15, 4], agents: 30, seed }), 120),
  };
  const snapshot = (s) => ({
    count: s.count,
    trail: Array.from(s.trail),
    x: Array.from(s.x.subarray(0, s.count)),
    y: Array.from(s.y.subarray(0, s.count)),
    heading: Array.from(s.heading.subarray(0, s.count)),
    cell: Array.from(s.cell.subarray(0, s.count)),
    substrate: s.substrate ? Array.from(s.substrate) : null,
  });
  for (const [mode, run] of Object.entries(runs)) {
    assert.deepEqual(snapshot(run(11)), snapshot(run(11)), mode);
    assert.notDeepEqual(snapshot(run(11)), snapshot(run(12)), mode);
  }
});

test("the author's rules: a step back into its own cell is blocked, the tests fall on step `every`, starts and jumps keep to startOn, the grid can wrap", () => {
  // One agent at (4.51, 4.51), cell (5, 5), heading 45 degrees: the step lands
  // at (5.22, 5.22), its own cell again, which its code counts as taken.
  const one = createPhysarum({ width: 12, height: 12, agents: 1, start: [5, 5, 0], seed: 1 });
  one.x[0] = 4.51;
  one.y[0] = 4.51;
  one.heading[0] = Math.PI / 4;
  stepPhysarum(one);
  assert.equal(one.moved, 0);
  assert.equal(one.blocked, 1);
  assert.equal(one.cell[0], 5 * 12 + 5);
  // The adaptive tests fall on steps every, 2 every, ... (the author's counter starts at 1).
  const W = 40, H = 40;
  const walls = arena(W, H);
  const packed = createPhysarum({ width: W, height: H, walls, agents: 10000, start: [20, 20, 6], adaptive: true, gMax: 0, sMax: 8, seed: 7 });
  stepPhysarum(packed, NETWORK_DEFAULTS.every - 1);
  assert.equal(packed.died, 0, "no test before step `every`");
  stepPhysarum(packed);
  assert.ok(packed.died > 0, "the test on step `every`");
  // Not within `border` cells of the edge: a crowd at the edge neither buds nor dies.
  const edge = createPhysarum({ width: W, height: H, walls, agents: 10000, start: [2, 20, 1], adaptive: true, every: 1, gMax: 0, sMax: 0, seed: 7 });
  stepPhysarum(edge);
  assert.equal(edge.died, 0);
  // Erosion with startOn 0: agents start, and jump, only onto worn-through cells.
  const substrate = new Float32Array(W * H).fill(200);
  for (let y = 10; y < 14; y += 1) for (let x = 10; x < 30; x += 1) substrate[y * W + x] = 0;
  const worn = createPhysarum({ width: W, height: H, mode: "erosion", substrate, agents: 30, startOn: 0, pTP: 1, pCD: 0, erodeBy: 0, seed: 3 });
  // Every agent jumps every step (pTP 1), so where they stand is where the jumps land; walking may leave the worn cells.
  const onWorn = (s) => Array.from(s.cell.subarray(0, s.count)).every((c) => s.substrate[c] === 0);
  for (let i = 0; i < worn.count; i += 1) assert.equal(worn.substrate[worn.cell[i]], 0, "started on a worn cell");
  const jumpsBefore = worn.teleported;
  stepPhysarum(worn, 5);
  assert.ok(worn.teleported > jumpsBefore && onWorn(worn));
  // The erosion example wraps round: an agent heading out past the right edge comes back in on the left.
  assert.equal(EROSION_DEFAULTS.wrap, true);
  const ring = createPhysarum({ width: 10, height: 10, mode: "erosion", agents: 1, start: [9, 5, 0], pTP: 0, pCD: 0, so: 2, seed: 1 });
  ring.heading[0] = 0;
  stepPhysarum(ring);
  assert.equal(ring.cell[0] % 10, 0, "past x = 9 is x = 0");
});
