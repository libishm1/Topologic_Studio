// Run with: node --test src/core/zone.test.mjs
//
// The two-zone fire model (gap LG8; wiki/research/zone-model-spec-2026-10-28.md)
// against its specification, on seeded random cases (mulberry32):
//   1. Hand checks of the correlations against FireDynamics.lean's numbers:
//      FD-02 growth classes, FD-03 Babrauskas's worked example, FD-04 and FD-17
//      (MQH, its constant and its inverse), FD-07 virtual origin, FD-08 plume
//      mass (expanded form, clamp, growth with height), FD-18 penetration time
//      and the factor-2 drop at t_p.
//   2. P1: Q never falls, h_k never rises, the fire room's rise never falls,
//      across t_p and flashover.
//   3. P2: each Euler step (S3) never raises the interface and keeps it in
//      [0, H]; every simulated interface is non-increasing.
//   4. P3: door outflow (S2) rises with T_U and falls with z; (R3) rises with
//      m_in and dT_in; every room's rise is non-decreasing in simulations on
//      random room graphs.
//   5. P4: every node's temperature never falls, the stored FED rates never
//      fall, untenability is an up-set and each node closes at the first step
//      its room is untenable; the coarsened forecast is never optimistic.
//   6. P6: a closed room filling at constant Q (linear plume term omitted)
//      against the closed form s(t) = (s0^(-2/3) + (2/3) a t)^(-3/2): the error
//      at dt = 0.5 s and its first-order convergence as dt falls.
//   7. rooms.js on box rooms: areas, perimeters, A_T, node assignment, door
//      sides, vents and exits, overlapping spaces.
//   8. The routers on the zone forecast: earliestArrival equals the dose
//      front's first point, and the deadline map is never later than
//      doseDeadline (section 6: they run unchanged).
// Properties are checked exactly (no tolerance): each formula is a chain of
// floating-point operations monotone in the argument concerned.
import assert from "node:assert/strict";
import { test } from "node:test";

import {
  GROWTH,
  GYPSUM,
  HEAD_HEIGHT,
  KELVIN,
  POST_FLASHOVER_RISE,
  T_AMBIENT,
  buildZoneForecast,
  doorOutflow,
  fedRoomRise,
  fireDiameter,
  fireRoomHistory,
  heatRelease,
  inUpperLayer,
  layerStep,
  liningConductance,
  mqhRise,
  nodeTemperatureC,
  penetrationTime,
  plumeMass,
  plumePowerTerm,
  simulateRooms,
  ventilationLimit,
  virtualOrigin,
} from "./zone.js";
import { buildRoomGraph } from "./rooms.js";
import { convectiveDoseRate, doseFront, earliestArrival } from "./forecast.js";
import { doseDeadline, doseDeadlineMap, escapeDeadlines } from "./deadline.js";

/** mulberry32, as rngFrom in src/core/rl.js. */
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const between = (x, lo, hi, label) => assert.ok(x > lo && x < hi, `${label}: ${x} not in (${lo}, ${hi})`);
const close = (x, y, rel, label) => assert.ok(Math.abs(x - y) <= rel * Math.max(1, Math.abs(y)), `${label}: ${x} vs ${y}`);

// ---------------------------------------------------------------- 1. hand checks

test("FD-02: growth classes are 1055 / t_g^2, and Q reaches 1055 kW at t_g", () => {
  const tg = { slow: 600, medium: 300, fast: 150, "ultra-fast": 75 };
  const within = { slow: 0.000005, medium: 0.00005, fast: 0.00005, "ultra-fast": 0.0005 };
  for (const [name, t] of Object.entries(tg)) {
    assert.ok(Math.abs(1055 / t ** 2 - GROWTH[name]) <= within[name], name);
    close(heatRelease(t, 1055 / t ** 2, Infinity), 1055, 1e-12, `FD01_reaches_ref ${name}`);
  }
  // Q is capped at Q_max.
  assert.equal(heatRelease(1000, GROWTH.fast, 2000), 2000);
});

test("FD-03: Babrauskas's worked example, and the ventilation limit is twice it", () => {
  const qfo = 750 * 1.8 * Math.sqrt(2);
  between(qfo, 1909, 1910, "Q_fo for a 1.8 m2 door 2 m high");
  const alpha = 1055 / 150 ** 2;
  const tfo = Math.sqrt(qfo / alpha);
  between(tfo, 201, 202, "fast fire reaches Q_fo");
  close(heatRelease(tfo, alpha, Infinity), qfo, 1e-12, "FD03_tfo_reaches");
  close(ventilationLimit(1.8, 2), 2 * qfo, 1e-12, "1500 A_o sqrt(H_o)");
});

test("FD-04, FD-17: MQH's constant, its inverse, and its monotonicity", () => {
  // FD17_constant_685: the note's 6.85 is Quintiere's reduction, rounded.
  const c = 1.63 * Math.cbrt(295) * Math.pow(Math.sqrt(9.8) * 1.2 * 1.05, -1 / 3);
  between(c, 6.86, 6.88, "1.63 T0^(1/3) (sqrt(g) rho0 cp)^(-1/3)");
  // FD04_740_vs_685.
  between(Math.sqrt((575 / 6.85) ** 3), 769, 770, "sqrt((575/6.85)^3)");
  // FD04_mqh_inverse: Q = sqrt((dT/c)^3) sqrt(A_o sqrt(H_o) h_k A_T) gives dT.
  const rng = mulberry32(17);
  for (let i = 0; i < 500; i += 1) {
    const Ao = 0.2 + rng() * 6, Ho = 1 + rng() * 2, hk = 0.005 + rng() * 0.1, AT = 20 + rng() * 800, dT = rng() * 900;
    const Q = Math.sqrt((dT / 6.85) ** 3) * Math.sqrt(Ao * Math.sqrt(Ho) * hk * AT);
    close(mqhRise(Q, Ao, Ho, hk, AT), dT, 1e-9, "MQH inverse");
    // FD17_strictMono_Q and FD17_antitone_hk.
    assert.ok(mqhRise(Q * 1.01 + 1e-6, Ao, Ho, hk, AT) > mqhRise(Q, Ao, Ho, hk, AT) || Q === 0);
    assert.ok(mqhRise(Q, Ao, Ho, hk * 1.5, AT) <= mqhRise(Q, Ao, Ho, hk, AT));
  }
  // A worked number: 500 kW, a 1.8 m2 door 2 m high, h_k = 0.03, A_T = 75 m2.
  const dT = mqhRise(500, 1.8, 2, 0.03, 75);
  close(dT, 6.85 * Math.cbrt(250000 / (1.8 * Math.SQRT2 * 0.03 * 75)), 1e-12, "MQH by hand");
  // 250000 / (1.8 * 1.41421 * 0.03 * 75) = 43648.6, cube root 35.209, x 6.85.
  between(dT, 241.1, 241.3, "500 kW in a 20 m2 room");
});

test("FD-07: the virtual origin sits 0.152 Q^(2/5) below the mean flame tip", () => {
  const rng = mulberry32(7);
  for (let i = 0; i < 200; i += 1) {
    const Q = rng() * 5000, D = rng() * 3;
    const flame = 0.235 * Q ** 0.4 - 1.02 * D;
    close(flame - virtualOrigin(Q, D), 0.152 * Q ** 0.4, 1e-9, "FD07_origin_below_flame");
  }
  between(virtualOrigin(1000, 1), 0.2954, 0.2955, "z0 at 1 MW, D = 1 m");
  // D for 2 MW at 500 kW m^-2: sqrt(4 * 2000 / (500 pi)) = 2.2568 m.
  between(fireDiameter(2000), 2.2567, 2.2569, "fire diameter");
});

test("FD-08: Heskestad's plume mass, expanded, clamped, growing with height", () => {
  const rng = mulberry32(8);
  for (let i = 0; i < 500; i += 1) {
    const Qc = rng() * 3000, s = 0.1 + rng() * 8;
    const factored = 0.071 * Math.cbrt(Qc) * s ** (5 / 3) * (1 + 0.026 * Qc ** (2 / 3) * s ** (-5 / 3));
    close(plumeMass(Qc, s), factored, 1e-9, "FD08_expand");
    if (Qc > 0) assert.ok(plumeMass(Qc, s + 0.01) > plumeMass(Qc, s), "FD08_strictMono_height");
  }
  // s below 0.1 m is taken as 0.1 m (S1).
  assert.equal(plumeMass(700, 0.02), plumeMass(700, 0.1));
  assert.equal(plumeMass(700, -3), plumeMass(700, 0.1));
  // 1 MW convective at 2 m: 0.071 * 10 * 2^(5/3) + 1.846 = 2.2541 + 1.846 = 4.1001 kg/s.
  between(plumeMass(1000, 2), 4.1, 4.1002, "plume at 1 MW, 2 m");
});

test("FD-18: gypsum's penetration time and the drop by half at t_p", () => {
  const tp = penetrationTime(GYPSUM);
  close(tp, 161.28, 1e-12, "t_p");
  close(liningConductance(tp * (1 - 1e-12), GYPSUM), (2 * GYPSUM.k) / GYPSUM.delta / 1000, 1e-9, "thick wall at t_p");
  close(liningConductance(tp, GYPSUM), GYPSUM.k / GYPSUM.delta / 1000, 1e-12, "thin wall from t_p");
});

test("S2: door outflow by hand", () => {
  // 0.9 m by 2.0 m door, layer 300 K above ambient, interface 1 m up.
  const TU = T_AMBIENT + 300;
  const expected = (2 / 3) * 0.7 * 0.9 * (353 / T_AMBIENT) * Math.sqrt(2 * 9.81 * (1 - T_AMBIENT / TU)) * 1;
  close(doorOutflow(0.9, 2.0, TU, 1.0), expected, 1e-12, "S2");
  // 0.42 x 1.20416 x sqrt(19.62 x 0.50577) = 0.50575 x 3.15013.
  between(doorOutflow(0.9, 2.0, TU, 1.0), 1.5931, 1.5932, "S2 value");
  assert.equal(doorOutflow(0.9, 2.0, TU, 2.0), 0, "layer at the door's top");
  assert.equal(doorOutflow(0.9, 2.0, T_AMBIENT, 0.5), 0, "no buoyancy");
});

// ------------------------------------------------------------- random room graphs

/** A random room graph: a spanning tree of doors, extra doors, vents, rooms with no door. */
function randomRoomGraph(rng) {
  const R = 1 + Math.floor(rng() * 12);
  const rooms = Array.from({ length: R }, () => {
    const floorArea = 3 + rng() * 200;
    const height = 1.5 + rng() * 5;
    const perimeter = 4 * Math.sqrt(floorArea) * (1 + rng() * 0.5);
    return { floorArea, height, perimeter };
  });
  const doors = [];
  const door = (a, b) => doors.push({ a, b, width: 0.6 + rng() * 1.6, height: 1.8 + rng() * 0.9 });
  const isolated = rng() < 0.15;
  for (let r = 1; r < R; r += 1) if (!isolated || rng() < 0.5) door(r, Math.floor(rng() * r));
  for (let e = 0; e < Math.floor(rng() * R); e += 1) {
    const a = Math.floor(rng() * R), b = Math.floor(rng() * R);
    if (a !== b) door(a, b);
  }
  for (let e = 0; e < Math.floor(rng() * 3); e += 1) door(Math.floor(rng() * R), -1);
  rooms.forEach((room, r) => {
    let doorArea = 0;
    for (const d of doors) if (d.a === r || d.b === r) doorArea += d.width * d.height;
    room.AT = Math.max(room.floorArea, 2 * room.floorArea + room.perimeter * room.height - doorArea);
  });
  return { rooms, doors };
}

function randomFire(rng, R) {
  const classes = Object.values(GROWTH);
  return {
    fireRoom: Math.floor(rng() * R),
    alpha: rng() < 0.7 ? classes[Math.floor(rng() * 4)] : rng() * 0.3,
    peak: 50 + rng() * 8000,
    dt: [0.5, 0.25, 1][Math.floor(rng() * 3)],
    horizon: 60 + rng() * 600,
  };
}

// ------------------------------------------------------------------------ 2. P1

test("P1: Q never falls, h_k never rises, the fire room's rise never falls (across t_p and flashover)", () => {
  const rng = mulberry32(101);
  const tp = penetrationTime(GYPSUM);
  let flashovers = 0, capped = 0;
  for (let c = 0; c < 300; c += 1) {
    const alpha = rng() * 0.3, Qmax = rng() * 6000;
    // Times on a fine grid with t_p and the cap time inserted, and their neighbours.
    const times = [];
    for (let t = 0; t <= 900; t += 0.37) times.push(t);
    const tc = Math.sqrt(Qmax / Math.max(alpha, 1e-12));
    for (const t of [tp, tc]) times.push(t * (1 - 1e-12), t, t * (1 + 1e-12));
    times.sort((p, q) => p - q);
    for (let i = 1; i < times.length; i += 1) {
      assert.ok(heatRelease(times[i], alpha, Qmax) >= heatRelease(times[i - 1], alpha, Qmax), "Q non-decreasing");
      if (times[i - 1] > 0) assert.ok(liningConductance(times[i], GYPSUM) <= liningConductance(times[i - 1], GYPSUM), "h_k non-increasing");
    }
    const Ao = 0.1 + rng() * 6, Ho = 1 + rng() * 1.6, AT = 10 + rng() * 900, dt = [0.5, 0.25, 0.1][c % 3];
    const steps = Math.ceil(900 / dt);
    const { rise, flashover } = fireRoomHistory(steps, dt, { alpha, Qmax, Ao, Ho, AT });
    for (let i = 1; i <= steps; i += 1) assert.ok(rise[i] >= rise[i - 1], `rise falls at step ${i}`);
    if (flashover !== null) {
      flashovers += 1;
      for (let i = Math.round(flashover / dt); i <= steps; i += 1) assert.equal(rise[i], POST_FLASHOVER_RISE);
    }
    if (Qmax < alpha * 900 ** 2) capped += 1;
  }
  assert.ok(flashovers > 20 && flashovers < 300, `flashover cases ${flashovers}`);
  assert.ok(capped > 20, `capped cases ${capped}`);
});

// ------------------------------------------------------------------------ 3. P2

test("P2: an Euler step never raises the interface and keeps it in [0, H]", () => {
  const rng = mulberry32(202);
  for (let c = 0; c < 20000; c += 1) {
    const H = 0.5 + rng() * 6, z = rng() < 0.1 ? H : rng() * H;
    const mIn = rng() < 0.1 ? 0 : rng() * 20, mOut = rng() < 0.3 ? 0 : rng() * 20;
    const next = layerStep(z, mIn, mOut, 0.3 + rng() * 0.9, 1 + rng() * 300, H, rng() * 2);
    assert.ok(next <= z && next >= 0 && next <= H, `${z} -> ${next}`);
  }
});

// --------------------------------------------------------------------- 4. P2, P3

test("P3: door outflow rises with T_U and falls with z; (R3) rises with its inputs", () => {
  const rng = mulberry32(303);
  for (let c = 0; c < 20000; c += 1) {
    const W = 0.5 + rng() * 2, Hd = 1.5 + rng(), T1 = T_AMBIENT + rng() * 1000, T2 = T1 + rng() * 200 * (rng() < 0.2 ? 0 : 1);
    const z1 = rng() * 4, z2 = z1 - rng() * 2;
    assert.ok(doorOutflow(W, Hd, T2, z1) >= doorOutflow(W, Hd, T1, z1), "S2 in T_U");
    assert.ok(doorOutflow(W, Hd, T1, z2) >= doorOutflow(W, Hd, T1, z1), "S2 in z");
    const m1 = rng() * 10, m2 = m1 + (rng() < 0.2 ? 0 : rng() * 5), d1 = rng() * 900, d2 = d1 + rng() * 100, AT = 5 + rng() * 800;
    assert.ok(fedRoomRise(d1, m2, AT) >= fedRoomRise(d1, m1, AT), "R3 in m_in");
    assert.ok(fedRoomRise(d2, m1, AT) >= fedRoomRise(d1, m1, AT), "R3 in dT_in");
    assert.ok(fedRoomRise(d1, m1, AT) <= d1, "R3 loses heat to the walls");
  }
});

test("P2, P3: on random room graphs every room heats and fills monotonically", () => {
  const rng = mulberry32(404);
  let reached = 0, roomsTotal = 0, belowDoor = 0;
  for (let c = 0; c < 250; c += 1) {
    const g = randomRoomGraph(rng);
    const fire = randomFire(rng, g.rooms.length);
    const sim = simulateRooms(g, fire);
    roomsTotal += g.rooms.length;
    sim.z.forEach((zk, k) => {
      if (!zk) return;
      const H = g.rooms[k].height;
      assert.equal(zk[0], H);
      for (let i = 1; i <= sim.steps; i += 1) {
        assert.ok(zk[i] <= zk[i - 1], `z rises in room ${k} at step ${i}`);
        assert.ok(zk[i] >= 0 && zk[i] <= H);
        assert.ok(sim.rise[k][i] >= sim.rise[k][i - 1], `T_U falls in room ${k} at step ${i}`);
      }
      if (k !== fire.fireRoom && zk[sim.steps] < H) reached += 1;
      if (zk[sim.steps] < 1.8) belowDoor += 1;
    });
  }
  // The cases exercise spread: smoke reaches rooms beyond the fire room.
  assert.ok(reached > 50, `rooms reached ${reached} of ${roomsTotal}`);
  assert.ok(belowDoor > 50, `layers below 1.8 m ${belowDoor}`);
});

// ------------------------------------------------------------------------ 5. P4

/** Random lattice-like graph: n nodes on a line with extra links, rooms by blocks. */
function randomNodes(rng, R, n) {
  const nodes = new Float64Array(n * 3);
  for (let v = 0; v < n; v += 1) {
    nodes[v * 3] = v * 0.5;
    nodes[v * 3 + 1] = 0.75;
    nodes[v * 3 + 2] = (rng() - 0.5) * 0.2;
  }
  const edges = [];
  for (let v = 1; v < n; v += 1) edges.push(v - 1, v);
  for (let e = 0; e < n / 3; e += 1) {
    const a = Math.floor(rng() * n), b = Math.min(n - 1, a + 1 + Math.floor(rng() * 4));
    if (a !== b) edges.push(a, b);
  }
  const nodeRoom = new Int32Array(n);
  for (let v = 0; v < n; v += 1) nodeRoom[v] = rng() < 0.08 ? -1 : Math.floor((v / n) * R);
  return { graph: { nodes, edges: Uint32Array.from(edges) }, nodeRoom };
}

test("P4: node temperatures never fall, untenability is an up-set, rates are the running maximum", () => {
  const rng = mulberry32(505);
  let closedSome = 0, smokeOff = 0;
  for (let c = 0; c < 150; c += 1) {
    const g = randomRoomGraph(rng);
    const R = g.rooms.length;
    const { graph, nodeRoom } = randomNodes(rng, R, 20 + Math.floor(rng() * 60));
    const n = graph.nodes.length / 3;
    let seed = Math.floor(rng() * n);
    while (nodeRoom[seed] < 0) seed = (seed + 1) % n;
    const smoke = rng() < 0.7;
    if (!smoke) smokeOff += 1;
    const opts = {
      seed,
      alpha: randomFire(rng, R).alpha,
      peak: 100 + rng() * 6000,
      untenableC: 30 + rng() * 150,
      smoke,
      horizon: 30 + rng() * 500,
      frameSeconds: [0.5, 1, 2.5][Math.floor(rng() * 3)],
    };
    const fc = buildZoneForecast(graph, { ...g, nodeRoom }, opts);
    const { sim, K, n: nn, rate, untenable, closes, horizon, frameSeconds } = fc;
    assert.equal(nn, n);
    assert.equal(fc.zone.fireRoom, nodeRoom[seed]);
    // Rates never fall, and are the FED rate of the frame's hottest state.
    for (let v = 0; v < n; v += 1) {
      for (let k = 1; k < K; k += 1) assert.ok(rate[k * n + v] >= rate[(k - 1) * n + v], `rate falls at node ${v}`);
      assert.ok(closes[v] <= horizon && closes[v] === Math.min(untenable[v], horizon));
    }
    for (let v = 0; v < n; v += 1) {
      const r = nodeRoom[v];
      if (r < 0 || !sim.z[r]) {
        assert.equal(untenable[v], Infinity);
        assert.equal(rate[v], Math.fround(convectiveDoseRate(T_AMBIENT - KELVIN)));
        continue;
      }
      const zk = sim.z[r], rk = sim.rise[r], H = g.rooms[r].height;
      let prev = -Infinity, first = Infinity;
      for (let i = 0; i <= sim.steps && i * sim.dt < horizon; i += 1) {
        const T = nodeTemperatureC(zk[i], H, rk[i]);
        assert.ok(T >= prev, "T_n falls");
        prev = T;
        const bad = inUpperLayer(zk[i], H, HEAD_HEIGHT) && (smoke || T >= opts.untenableC);
        if (bad && first === Infinity) first = i * sim.dt;
        if (first < Infinity) assert.ok(bad, "untenability is not an up-set");
        // The frame holding this step has at least its rate.
        const k = Math.min(K - 1, Math.floor((i * sim.dt) / frameSeconds));
        assert.ok(rate[k * n + v] >= Math.fround(convectiveDoseRate(T)) * (1 - 1e-7), "frame below its state");
      }
      assert.equal(untenable[v], first, `node ${v} closes at the first untenable step`);
      if (first < Infinity) closedSome += 1;
    }
  }
  assert.ok(closedSome > 100, `closed nodes ${closedSome}`);
  assert.ok(smokeOff > 10);
});

test("P4: a coarsened zone forecast holds each window's hottest state (never optimistic)", () => {
  // 30,000 nodes x 600 one-second frames is over forecast.js's MAX_CELLS.
  const rng = mulberry32(606);
  const g = randomRoomGraph(rng);
  while (g.rooms.length < 4) Object.assign(g, randomRoomGraph(rng));
  const { graph, nodeRoom } = randomNodes(rng, g.rooms.length, 30000);
  const fc = buildZoneForecast(graph, { ...g, nodeRoom }, { seed: nodeRoom.indexOf(0), growth: "fast", peak: 3000 });
  assert.ok(fc.stride > 1, `stride ${fc.stride}`);
  const { sim, K, n, rate, frameSeconds } = fc;
  for (let v = 0; v < n; v += 97) {
    const r = nodeRoom[v];
    if (r < 0 || !sim.z[r]) continue;
    for (let i = 0; i <= sim.steps && i * sim.dt < fc.horizon; i += 1) {
      const T = nodeTemperatureC(sim.z[r][i], g.rooms[r].height, sim.rise[r][i]);
      const k = Math.min(K - 1, Math.floor((i * sim.dt) / frameSeconds));
      assert.ok(rate[k * n + v] >= Math.fround(convectiveDoseRate(T)) * (1 - 1e-7));
    }
  }
});

// ------------------------------------------------------------------------ 6. P6

test("P6: a closed room fills as the closed form, first order in dt", () => {
  // Constant Q = 400 kW (Q_c = 280 kW), a 25 m2 room 3 m high, rho_U held at
  // 353 / 600 K, the linear term of (S1) omitted: ds/dt = -a s^(5/3).
  const Q = 400, Qc = 0.7 * Q, Af = 25, H = 3, rhoU = 353 / 600;
  const z0 = virtualOrigin(Q, fireDiameter(Q));
  const s0 = H - z0;
  const a = (0.071 * Math.cbrt(Qc)) / (rhoU * Af);
  const exact = (t) => Math.pow(Math.pow(s0, -2 / 3) + (2 / 3) * a * t, -3 / 2);
  const T = 60;
  const errors = [];
  for (const dt of [0.5, 0.25, 0.125, 0.0625, 0.03125]) {
    let z = H, worst = 0;
    const steps = Math.round(T / dt);
    for (let i = 0; i < steps; i += 1) {
      z = layerStep(z, plumePowerTerm(Qc, z - z0), 0, rhoU, Af, H, dt);
      worst = Math.max(worst, Math.abs(z - z0 - exact((i + 1) * dt)));
    }
    assert.ok(z - z0 > 0.1, "the clamp at 0.1 m never acts");
    errors.push({ dt, maxError: worst, sAtEnd: z - z0, exactAtEnd: exact(T) });
  }
  console.log(JSON.stringify({ P6: errors.map((e) => ({ dt: e.dt, maxErrorM: +e.maxError.toExponential(3) })), sEnd: +errors[0].exactAtEnd.toFixed(4) }));
  // The error at dt = 0.5 s is a centimetre or two, and halves with dt.
  assert.ok(errors[0].maxError < 0.03, `error at 0.5 s: ${errors[0].maxError}`);
  for (let i = 1; i < errors.length; i += 1) {
    const ratio = errors[i - 1].maxError / errors[i].maxError;
    assert.ok(ratio > 1.8 && ratio < 2.2, `convergence ratio ${ratio}`);
  }
});

// ------------------------------------------------------------------- 7. rooms.js

/** A box space mesh, y up: [x0, x1] x [y0, y1] x [z0, z1]. */
function box(itemId, x0, x1, z0, z1, y0 = 0, y1 = 3) {
  const v = [
    [x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1],
    [x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1],
  ];
  const f = [0, 1, 2, 0, 2, 3, 4, 6, 5, 4, 7, 6, 0, 4, 5, 0, 5, 1, 1, 5, 6, 1, 6, 2, 2, 6, 7, 2, 7, 3, 3, 7, 4, 3, 4, 0];
  return { itemId, positions: Float32Array.from(v.flat()), indices: Uint32Array.from(f) };
}

/**
 * Two rooms, A (x 0 to 3.9) and B (x 4.1 to 10), z 0 to 5, a 0.2 m wall
 * between with a door at x = 4, an exterior door at x = 10.1 to two outside
 * nodes. Nodes stand 0.75 m above the floor on a 0.5 m grid.
 */
function twoRooms() {
  const pts = [];
  const id = new Map();
  const add = (x, z) => {
    id.set(`${x},${z}`, pts.length / 3);
    pts.push(x, 0.75, z);
  };
  const xsA = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75];
  const xsB = [4.25, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.75];
  const zs = [0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75];
  for (const x of [...xsA, ...xsB]) for (const z of zs) add(x, z);
  add(4.0, 2.25); // door A-B
  add(10.1, 2.25); // exterior door
  add(10.6, 2.25); // outside
  add(11.1, 2.25);
  const edges = [];
  const link = (a, b) => edges.push(id.get(a), id.get(b));
  for (const xs of [xsA, xsB]) {
    for (let i = 0; i < xs.length; i += 1) {
      for (let j = 0; j < zs.length; j += 1) {
        if (i + 1 < xs.length) link(`${xs[i]},${zs[j]}`, `${xs[i + 1]},${zs[j]}`);
        if (j + 1 < zs.length) link(`${xs[i]},${zs[j]}`, `${xs[i]},${zs[j + 1]}`);
      }
    }
  }
  link("3.75,2.25", "4,2.25");
  link("4,2.25", "4.25,2.25");
  link("9.75,2.25", "10.1,2.25");
  link("10.1,2.25", "10.6,2.25");
  link("10.6,2.25", "11.1,2.25");
  const graph = { nodes: Float64Array.from(pts), edges: Uint32Array.from(edges) };
  graph.blocked = new Uint8Array(edges.length / 2);
  const doors = [
    { itemId: 501, node: id.get("4,2.25"), sides: [1, 1], x: 4.0, y: 2.25, sill: 0, width: 0.9, ux: 0, uy: 1 },
    { itemId: 502, node: id.get("10.1,2.25"), sides: [1, 1], x: 10.1, y: 2.25, sill: 0, width: 1.0, ux: 0, uy: 1 },
  ];
  return { graph, doors, id, spaces: [box(11, 0, 3.9, 0, 5), box(12, 4.1, 10, 0, 5)] };
}

test("rooms.js: box rooms, node assignment, a door joining them and one venting", () => {
  const { graph, doors, id, spaces } = twoRooms();
  const rg = buildRoomGraph(graph, { spaces, doors, declaredDoors: [{ id: 501, widthM: 0.8, heightM: 2.1 }] });
  assert.equal(rg.rooms.length, 2);
  const [A, B] = rg.rooms;
  close(A.floorArea, 19.5, 1e-6, "A_f of A");
  close(B.floorArea, 29.5, 1e-6, "A_f of B");
  close(A.perimeter, 2 * (3.9 + 5), 1e-6, "P of A");
  close(B.perimeter, 2 * (5.9 + 5), 1e-6, "P of B");
  close(A.height, 3, 1e-6, "H");
  assert.equal(A.perimeterFromMesh, true);
  // A_T = 2 A_f + P H less the door (0.8 x 2.1 from the IFC).
  close(A.AT, 2 * 19.5 + 17.8 * 3 - 0.8 * 2.1, 1e-6, "A_T of A");
  close(B.AT, 2 * 29.5 + 21.8 * 3 - 0.8 * 2.1 - 1.0 * 2.0, 1e-6, "A_T of B");
  // Every grid node is in its footprint; the door nodes and outside nodes
  // take a room from the search; nothing is left without one.
  assert.equal(rg.stats.inFootprint, 200);
  assert.equal(rg.stats.assigned, graph.nodes.length / 3);
  assert.equal(rg.nodeRoom[id.get("0.25,0.25")], 0);
  assert.equal(rg.nodeRoom[id.get("9.75,4.75")], 1);
  assert.ok(rg.nodeRoom[id.get("4,2.25")] >= 0);
  assert.equal(rg.nodeRoom[id.get("11.1,2.25")], 1, "outside nodes take the room they are reached from");
  // The doors: A-B joined with the IFC's size; B vents through the mesh-measured door.
  assert.equal(rg.doors.length, 2);
  const [join, vent] = rg.doors;
  assert.deepEqual([join.kind, Math.min(join.a, join.b), Math.max(join.a, join.b), join.width, join.height], ["join", 0, 1, 0.8, 2.1]);
  assert.deepEqual([vent.kind, vent.a, vent.b, vent.width, vent.height], ["vent", 1, -1, 1.0, 2.0]);
  assert.equal(rg.stats.joins, 1);
  assert.equal(rg.stats.vents, 1);
});

test("rooms.js: an exit flag vents both rooms; an enclosing space loses to the rooms; no direction falls back", () => {
  const { graph, doors, spaces } = twoRooms();
  const exitFlag = buildRoomGraph(graph, { spaces, doors, declaredDoors: [{ id: 501, exit: "Pset_DoorCommon.FireExit" }] });
  const flagged = exitFlag.doors.filter((d) => d.itemId === 501);
  assert.deepEqual(flagged.map((d) => [d.kind, d.a, d.b]).sort(), [["vent", 0, -1], ["vent", 1, -1]]);
  // A gross-area space over both rooms: nodes still go to the smaller rooms.
  const gross = buildRoomGraph(graph, { spaces: [box(99, -1, 12, -1, 6), ...spaces], doors });
  assert.equal(gross.rooms.length, 3);
  const small = gross.rooms.findIndex((r) => r.itemId === 11);
  assert.equal(gross.nodeRoom[0], small);
  // Doors without a direction: neighbours' rooms decide.
  const bare = doors.map(({ itemId, node, sides }) => ({ itemId, node, sides }));
  const old = buildRoomGraph(graph, { spaces, doors: bare });
  assert.equal(old.doors.find((d) => d.itemId === 501).kind, "join");
});

test("rooms.js: the search stays within 1 m of the room's floor; unbounded it climbs the stairs", () => {
  // Room A's grid, a stair of 0.3 m risers from its west edge, and a landing
  // 3 m up with no space over it.
  const { graph: base, doors, id, spaces } = twoRooms();
  const pts = Array.from(base.nodes);
  const edges = Array.from(base.edges);
  let prev = id.get("0.25,2.25");
  const stair = [];
  for (let k = 1; k <= 10; k += 1) {
    const v = pts.length / 3;
    pts.push(-0.25 * k, 0.75 + 0.3 * k, 2.25);
    edges.push(prev, v);
    stair.push(v);
    prev = v;
  }
  const graph = { nodes: Float64Array.from(pts), edges: Uint32Array.from(edges) };
  graph.blocked = new Uint8Array(edges.length / 2);
  const banded = buildRoomGraph(graph, { spaces, doors });
  const literal = buildRoomGraph(graph, { spaces, doors, band: Infinity });
  // Floors 0.3 to 3.0 m: within 1 m of A's floor are the first three treads.
  assert.deepEqual(stair.map((v) => banded.nodeRoom[v] >= 0), [true, true, true, false, false, false, false, false, false, false]);
  assert.ok(stair.every((v) => literal.nodeRoom[v] === 0));
});

// --------------------------------------------------------------- 8. the routers

test("Section 6: the routers run unchanged on the zone forecast and keep their guarantees", () => {
  const { graph, doors, id, spaces } = twoRooms();
  const rg = buildRoomGraph(graph, { spaces, doors });
  const exits = [id.get("11.1,2.25")];
  const blocked = graph.blocked;
  let checked = 0;
  for (const [fireAt, growth, peak] of [["5.25,1.25", "fast", 2500], ["1.25,3.25", "medium", 1500], ["8.25,4.25", "ultra-fast", 4000], ["2.25,0.25", "slow", 800]]) {
    const fc = buildZoneForecast(graph, rg, { seed: id.get(fireAt), growth, peak, horizon: 300 });
    assert.ok(fc.zone.roomsReached >= 1);
    const hard = escapeDeadlines(graph, { exits, untenable: fc.closes, speed: 1.2, blocked });
    const map = doseDeadlineMap(graph, { exits, forecast: fc, limit: 0.3, speed: 1.2, blocked });
    for (const start of [id.get("0.25,0.25"), id.get("3.75,4.75"), id.get("6.25,2.25"), id.get("9.75,0.25")]) {
      const depart = hard[start] > 0 ? Math.min(5, hard[start] / 2) : 0;
      const ea = earliestArrival(graph, start, depart, { exits, closes: fc.closes, speed: 1.2, blocked });
      const { front } = doseFront(graph, start, { exits, forecast: fc, depart, speed: 1.2, blocked });
      assert.equal(ea.exit >= 0, front.length > 0, "both find a route or neither does");
      if (ea.exit >= 0) assert.ok(Math.abs(ea.arrive[ea.exit] - front[0].arrival) <= 1e-9 * Math.max(1, front[0].arrival));
      if (map.deadline[start] > -Infinity) {
        const exact = doseDeadline(graph, start, { exits, forecast: fc, limit: 0.3, speed: 1.2, blocked, tolerance: 0.01 });
        assert.ok(exact && map.deadline[start] <= exact.bad + 1e-9, "the map is never later than the exact deadline");
        const at = doseFront(graph, start, { exits, forecast: fc, depart: map.deadline[start], speed: 1.2, blocked }).front;
        assert.ok(at.length && at[at.length - 1].dose <= 0.3, "leaving at the map's deadline works");
        checked += 1;
      }
    }
  }
  assert.ok(checked >= 6, `deadlines checked ${checked}`);
});

test("Section 5: a room lower than head height is not untenable before smoke reaches it", () => {
  assert.equal(inUpperLayer(1.6, 1.6, HEAD_HEIGHT), false);
  assert.equal(inUpperLayer(1.59, 1.6, HEAD_HEIGHT), true);
  assert.equal(nodeTemperatureC(3, 3, 400), T_AMBIENT - KELVIN);
  close(nodeTemperatureC(1.0, 3, 400), T_AMBIENT + 400 - KELVIN, 1e-12, "N1");
});
