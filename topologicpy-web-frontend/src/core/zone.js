/**
 * A quasi-steady two-zone fire model on the IFC room graph (gap LG8), as
 * specified in wiki/research/zone-model-spec-2026-10-28.md. The room graph
 * (rooms, their doors, which room each lattice node is in) comes from
 * src/core/rooms.js. Equation and property labels (F1-F3, S1-S3, R1-R3, N1;
 * P1-P6) are the specification's, and formal/fireegress Zone.lean is written
 * from the same text; each function below implements the equation named in
 * its comment.
 *
 * The model. A t-squared fire grows in one room up to a ventilation-limited
 * peak (F1); MQH gives that room's upper-layer temperature (F2) until it
 * reaches 500 K above ambient, when the room flashes over and stays at 900 K
 * (F3). Heskestad's plume fills the room's upper layer (S1), the layer spills
 * through doors once it is below their tops (S2), and the interface falls by
 * an explicit Euler step (S3). Rooms are then taken in breadth-first order
 * from the fire room: each is fed by the doors to rooms taken before it (R1),
 * at the hottest of their temperatures (R2) less what its walls take (R3), and
 * fills by (S3) in turn. A lattice node is in its room's upper layer once the
 * interface is at or below head height (section 5), and then sees the layer's
 * temperature in degrees C (N1); otherwise it is at ambient.
 *
 * Clock. Seconds after ignition. The Euler step is dt (0.5 s); the state
 * computed for t_i = i dt holds on [t_i, t_{i+1}). The forecast frame k covers
 * [k f, (k + 1) f) and holds the hottest state in it, so it is never
 * optimistic, as buildForecast's frames are.
 *
 * Monotonicity (P1-P4). Q never falls and h_k never rises, so the fire room's
 * rise never falls (P1); each Euler step subtracts a non-negative amount and
 * clamps, so no interface ever rises (P2); door outflow rises with the
 * upstream temperature and falls with its interface, and (R3) rises with the
 * inflow and its temperature, so by induction along the breadth-first order
 * every room heats and fills monotonically (P3), and every node's temperature
 * never falls and its untenability is an up-set in time (P4): the hypotheses
 * under which the routers of forecast.js and deadline.js are exact and
 * conservative. Each step of each formula is a floating-point operation that
 * is monotone in the argument concerned, so the properties hold exactly in
 * doubles as well (src/core/zone.test.mjs checks them without a tolerance).
 *
 * Limits (spec section 8): pre-flashover correlations for rooms of ordinary
 * size; one fire; open doors; no ceiling jet, radiant heat, toxic gases or
 * visibility beyond the layer height; one-way smoke spread along the
 * breadth-first order; lower layers at ambient; IfcSpace rooms needed. It
 * ranks routes and deadlines under these assumptions, and is not a substitute
 * for CFAST, FDS or an engineer's assessment.
 */
import { MAX_CELLS, convectiveDoseRate } from "./forecast.js";

/** Ambient temperature T_a (K; spec section 3) and the Celsius offset (N1). */
export const T_AMBIENT = 293.15;
export const KELVIN = 273.15;
/** Gas density 353 / T (kg m^-3; spec section 3). */
export const gasDensity = (T) => 353 / T;
const RHO_AMBIENT = gasDensity(T_AMBIENT);

/** Growth coefficients alpha (kW s^-2; FD-02). */
export const GROWTH = { slow: 0.00293, medium: 0.0117, fast: 0.0469, "ultra-fast": 0.188 };
/** Default lining, gypsum board: k (W m^-1 K^-1), rho (kg m^-3), c (J kg^-1 K^-1), delta (m). */
export const GYPSUM = { k: 0.48, rho: 1440, c: 840, delta: 0.016 };

const CONVECTIVE_FRACTION = 0.7; // chi_c (F1)
const FIRE_FLUX = 500; // q'' (kW m^-2; F1)
const MQH = 6.85; // K^(2/3) kW^(-1/3) m^(5/6) (F2, FD-17)
export const FLASHOVER_RISE = 500; // K (F3)
export const POST_FLASHOVER_RISE = 900; // K (F3)
const MIN_PLUME_HEIGHT = 0.1; // m (S1)
const DOOR_CD = 0.7; // C_d (S2)
const GRAVITY = 9.81; // m s^-2 (S2)
const CP = 1.0; // kJ kg^-1 K^-1 (R3)
const WALL_LOSS = 0.015; // h_c, kW m^-2 K^-1 (R3)
/** A room with no door leaks through A_o = 0.1 m^2 at H_o = 2.0 m (F2). */
const LEAK_AREA = 0.1;
const LEAK_HEIGHT = 2.0;
/** Head height above the room's floor (m; section 5). */
export const HEAD_HEIGHT = 1.8;

// ------------------------------------------------------------------- fire

/** (F1) Heat release rate Q(t) = min(alpha t^2, Q_max), kW, t >= 0. */
export function heatRelease(t, alpha, Qmax) {
  return Math.min(alpha * t * t, Qmax);
}

/** (F1) Ventilation limit 1500 A_o sqrt(H_o), kW. */
export function ventilationLimit(Ao, Ho) {
  return 1500 * Ao * Math.sqrt(Ho);
}

/** (F1) Fire diameter D = sqrt(4 Q_max / (pi q'')), m. */
export function fireDiameter(Qmax, flux = FIRE_FLUX) {
  return Math.sqrt((4 * Qmax) / (Math.PI * flux));
}

/** (F1, FD-07) Heskestad's virtual origin z_0 = 0.083 Q^(2/5) - 1.02 D, m above the floor. */
export function virtualOrigin(Q, D) {
  return 0.083 * Math.pow(Math.max(Q, 0), 0.4) - 1.02 * D;
}

/** (F2, FD-18) Thermal penetration time t_p = (rho c / k) (delta / 2)^2, s. */
export function penetrationTime({ k, rho, c, delta } = GYPSUM) {
  return ((rho * c) / k) * (delta / 2) ** 2;
}

/**
 * (F2, FD-18) The lining's effective conductance h_k(t), kW m^-2 K^-1:
 * sqrt(k rho c / t) / 1000 before t_p, k / (1000 delta) from t_p on (a drop
 * by half at t_p; the thick-wall branch at t_p would be 2 k / delta).
 */
export function liningConductance(t, lining = GYPSUM) {
  const { k, rho, c, delta } = lining;
  return t < penetrationTime(lining) ? Math.sqrt((k * rho * c) / t) / 1000 : k / (1000 * delta);
}

/** (F2, FD-17) MQH upper-layer temperature rise 6.85 (Q^2 / (A_o sqrt(H_o) h_k A_T))^(1/3), K. */
export function mqhRise(Q, Ao, Ho, hk, AT) {
  if (!(Q > 0)) return 0;
  return MQH * Math.cbrt((Q * Q) / (Ao * Math.sqrt(Ho) * hk * AT));
}

/** (F2) The fire room's vent: A_o = sum of W_d H_d over its doors, H_o their area-weighted height. */
export function fireVent(doors) {
  let Ao = 0, weighted = 0;
  for (const d of doors) {
    const area = d.width * d.height;
    Ao += area;
    weighted += area * d.height;
  }
  return Ao > 0 ? { Ao, Ho: weighted / Ao, leakage: false } : { Ao: LEAK_AREA, Ho: LEAK_HEIGHT, leakage: true };
}

/**
 * (F1-F3) The fire room's upper-layer rise at every Euler step, with the
 * flashover latch: from the first step at which MQH reaches 500 K, 900 K.
 *
 * @returns {{rise: Float64Array, flashover: number|null, Q: Float64Array}}
 */
export function fireRoomHistory(steps, dt, { alpha, Qmax, Ao, Ho, AT, lining = GYPSUM }) {
  const rise = new Float64Array(steps + 1);
  const Q = new Float64Array(steps + 1);
  let flashover = null;
  for (let i = 0; i <= steps; i += 1) {
    const t = i * dt;
    Q[i] = heatRelease(t, alpha, Qmax);
    const raw = mqhRise(Q[i], Ao, Ho, liningConductance(t, lining), AT);
    if (flashover === null && raw >= FLASHOVER_RISE) flashover = t;
    rise[i] = flashover === null ? raw : POST_FLASHOVER_RISE;
  }
  return { rise, flashover, Q };
}

// ----------------------------------------------------------------- layers

/** (S1) The 5/3-power term of Heskestad's plume mass flow, kg s^-1 (s clamped to >= 0.1 m). */
export function plumePowerTerm(Qc, s) {
  return 0.071 * Math.cbrt(Math.max(Qc, 0)) * Math.pow(Math.max(s, MIN_PLUME_HEIGHT), 5 / 3);
}

/** (S1, FD-08) Plume mass flow into the fire room's layer, 0.071 Q_c^(1/3) s^(5/3) + 0.001846 Q_c, kg s^-1. */
export function plumeMass(Qc, s) {
  return plumePowerTerm(Qc, s) + 0.001846 * Math.max(Qc, 0);
}

/**
 * (S2) Upper-layer outflow through a door of a room whose layer is at TU (K)
 * with its interface z (m above the floor):
 * (2/3) C_d W_d rho_a sqrt(2 g (1 - T_a / T_U)) max(0, H_d - z)^(3/2), kg s^-1.
 */
export function doorOutflow(Wd, Hd, TU, z) {
  const depth = Hd - z;
  if (!(depth > 0) || !(TU > T_AMBIENT)) return 0;
  return ((2 / 3) * DOOR_CD * Wd * RHO_AMBIENT) * Math.sqrt(2 * GRAVITY * (1 - T_AMBIENT / TU)) * Math.pow(depth, 1.5);
}

/**
 * (S3) One explicit Euler step of a room's interface:
 * clamp(z - dt max(0, m_in - m_out) / (rho_U A_f), 0, H). Never rises (P2).
 */
export function layerStep(z, mIn, mOut, rhoU, Af, H, dt) {
  const fall = (dt * Math.max(0, mIn - mOut)) / (rhoU * Af);
  return Math.min(H, Math.max(0, fall > 0 ? z - fall : z));
}

/**
 * (R3) A fed room's layer rise, the incoming rise less the walls' share:
 * dT_in m_in c_p / (m_in c_p + h_c A_T), written as dT_in / (1 + h_c A_T /
 * (m_in c_p)) so that every operation is monotone in m_in.
 */
export function fedRoomRise(dTin, mIn, AT) {
  if (!(mIn > 0) || !(dTin > 0)) return 0;
  return dTin / (1 + (WALL_LOSS * AT) / (mIn * CP));
}

// ------------------------------------------------------------------ nodes

/**
 * (Section 5) Whether a node of a room is in its upper layer: the interface
 * at or below head height, and the layer begun (z below the ceiling), so a
 * room lower than head height is not untenable before any smoke reaches it.
 */
export function inUpperLayer(z, H, zHead = HEAD_HEIGHT) {
  return z <= zHead && z < H;
}

/** (N1) A node's gas temperature in degrees C: the layer's if it is in it, else ambient. */
export function nodeTemperatureC(z, H, rise, zHead = HEAD_HEIGHT) {
  return (inUpperLayer(z, H, zHead) ? T_AMBIENT + rise : T_AMBIENT) - KELVIN;
}

// ------------------------------------------------------------- simulation

/** Rooms in breadth-first order from `start` over the doors that join two rooms (section 4). */
export function roomOrder(roomCount, doors, start) {
  const adjacent = Array.from({ length: roomCount }, () => []);
  for (const d of doors) {
    if (d.a >= 0 && d.b >= 0 && d.a !== d.b) {
      adjacent[d.a].push(d.b);
      adjacent[d.b].push(d.a);
    }
  }
  const pos = new Int32Array(roomCount).fill(-1);
  const order = [start];
  pos[start] = 0;
  for (let q = 0; q < order.length; q += 1) {
    for (const r of adjacent[order[q]]) {
      if (pos[r] >= 0) continue;
      pos[r] = order.length;
      order.push(r);
    }
  }
  return { order, pos };
}

/**
 * Run the model on a room graph to the horizon: the fire room first, then
 * each room reached from it in breadth-first order, each room's whole history
 * before its successors' (section 4).
 *
 * @param {object} roomGraph  rooms: [{floorArea, height, AT}]; doors: [{a, b,
 *                            width, height}], b = -1 for a vent to outside
 *                            (rooms.js buildRoomGraph).
 * @param {object} opts       fireRoom; alpha (kW s^-2); peak: Q_user (kW); dt
 *                            (s); horizon (s); lining.
 * @returns {{dt, steps, order, pos, z: Array<Float64Array|null>, rise:
 *           Array<Float64Array|null>, flashover: number|null, fire: object}}
 *          z and rise per room and step (null: never reached, z = H and no
 *          rise throughout).
 */
export function simulateRooms(roomGraph, { fireRoom, alpha = GROWTH.medium, peak = 2000, dt = 0.5, horizon = 600, lining = GYPSUM }) {
  const { rooms, doors } = roomGraph;
  const R = rooms.length;
  if (!(fireRoom >= 0 && fireRoom < R)) throw new Error("The fire is not in a room.");
  const steps = Math.max(1, Math.ceil(horizon / dt - 1e-9));
  const doorsOf = Array.from({ length: R }, () => []);
  for (const d of doors) {
    if (d.a >= 0) doorsOf[d.a].push(d);
    if (d.b >= 0 && d.b !== d.a) doorsOf[d.b].push(d);
  }
  const { order, pos } = roomOrder(R, doors, fireRoom);

  // (F1, F2) The fire and its room's vent.
  const vent = fireVent(doorsOf[fireRoom]);
  const Qmax = Math.min(peak, ventilationLimit(vent.Ao, vent.Ho));
  const D = fireDiameter(Qmax);
  const fireRoomData = rooms[fireRoom];
  const history = fireRoomHistory(steps, dt, { alpha, Qmax, Ao: vent.Ao, Ho: vent.Ho, AT: fireRoomData.AT, lining });

  const z = new Array(R).fill(null);
  const rise = new Array(R).fill(null);
  for (const j of order) {
    const room = rooms[j];
    const zj = new Float64Array(steps + 1);
    const rj = j === fireRoom ? history.rise : new Float64Array(steps + 1);
    // (R1) The doors that feed room j: those to rooms taken before it.
    const feeds = [];
    for (const d of doorsOf[j]) {
      if (d.b < 0 || d.a === d.b) continue;
      const from = d.a === j ? d.b : d.a;
      if (pos[from] < pos[j]) feeds.push([d, from]);
    }
    zj[0] = room.height;
    for (let i = 0; i <= steps; i += 1) {
      let mIn = 0;
      if (j === fireRoom) {
        // (S1) The plume, at height z - z_0 above the virtual origin.
        const Q = history.Q[i];
        mIn = plumeMass(CONVECTIVE_FRACTION * Q, zj[i] - virtualOrigin(Q, D));
      } else {
        // (R1) inflow, (R2) its temperature, (R3) less the walls' share.
        let dTin = 0;
        for (const [d, from] of feeds) {
          mIn += doorOutflow(d.width, d.height, T_AMBIENT + rise[from][i], z[from][i]);
          if (rise[from][i] > dTin) dTin = rise[from][i];
        }
        rj[i] = fedRoomRise(dTin, mIn, room.AT);
      }
      if (i === steps) break;
      // (S2) Outflow through every door of room j, as (S3) has it: vents,
      // doors onward, and the doors it is fed through (whose flow back is lost
      // to j but not added to the room behind: one-way spread). (S3) the step.
      const TU = T_AMBIENT + rj[i];
      let mOut = 0;
      for (const d of doorsOf[j]) mOut += doorOutflow(d.width, d.height, TU, zj[i]);
      zj[i + 1] = layerStep(zj[i], mIn, mOut, gasDensity(TU), room.floorArea, room.height, dt);
    }
    z[j] = zj;
    rise[j] = rj;
  }
  return {
    dt,
    steps,
    order,
    pos,
    z,
    rise,
    flashover: history.flashover,
    fire: { room: fireRoom, alpha, peak, Qmax, ventLimited: Qmax < peak, D, ...vent, tp: penetrationTime(lining) },
  };
}

/** The fire room k_f: the room of the fire origin's node (section 2). */
export function fireRoomOf(nodeRoom, seed) {
  const room = seed >= 0 && seed < nodeRoom.length ? nodeRoom[seed] : -1;
  if (room < 0) throw new Error("The fire origin is not in a room: the two-zone model needs the fire inside an IfcSpace.");
  return room;
}

/**
 * For the live simulation: each room's gas temperature as its nodes see it
 * (N1, degrees C) at times 0, f, 2f, ... (frame-major, rooms + 1 per frame;
 * the last entry is ambient, for nodes in no room).
 */
export function roomTemperatureFrames(sim, rooms, frames, frameSeconds, zHead = HEAD_HEIGHT) {
  const R = rooms.length;
  const table = new Float32Array(frames * (R + 1)).fill(T_AMBIENT - KELVIN);
  for (let frame = 0; frame < frames; frame += 1) {
    const i = Math.min(sim.steps, Math.round((frame * frameSeconds) / sim.dt));
    for (let k = 0; k < R; k += 1) {
      if (sim.z[k]) table[frame * (R + 1) + k] = nodeTemperatureC(sim.z[k][i], rooms[k].height, sim.rise[k][i], zHead);
    }
  }
  return table;
}

/** Node temperatures (degrees C), one Float64Array per frame, from roomTemperatureFrames' table. */
export function* zoneNodeFrames(nodeRoom, table, roomCount, frames) {
  const n = nodeRoom.length;
  for (let frame = 0; frame < frames; frame += 1) {
    const out = new Float64Array(n);
    const row = frame * (roomCount + 1);
    for (let v = 0; v < n; v += 1) out[v] = table[row + (nodeRoom[v] >= 0 ? nodeRoom[v] : roomCount)];
    yield out;
  }
}

/** Rooms whose layer has begun (interface below the ceiling) by time t. */
export function roomsReachedBy(sim, rooms, t) {
  const i = Math.min(sim.steps, Math.max(0, Math.floor(t / sim.dt + 1e-9)));
  let count = 0;
  sim.z.forEach((zk, k) => {
    if (zk && zk[i] < rooms[k].height) count += 1;
  });
  return count;
}

/**
 * The two-zone model as a forecast for the routers (section 6): the same
 * object buildForecast returns (forecast.js), with T a gas temperature in
 * degrees C, so earliestArrival, doseFront, planWithForecast, escapeDeadlines,
 * doseDeadline and doseDeadlineMap run on it unchanged.
 *
 * @param {object} graph      {nodes, edges}
 * @param {object} roomGraph  rooms.js buildRoomGraph output
 * @param {object} opts       seed: the fire's node; growth: a GROWTH key, or
 *                            alpha; peak: Q_user (kW); untenableC: theta (C);
 *                            smoke: the layer at head height is untenable;
 *                            zHead (m); horizon (s); frameSeconds: Delta t_f
 *                            (s); dt: the Euler step (s); lining.
 */
export function buildZoneForecast(graph, roomGraph, {
  seed,
  growth = "medium",
  alpha = null,
  peak = 2000,
  untenableC = 60,
  smoke = true,
  zHead = HEAD_HEIGHT,
  horizon = 600,
  frameSeconds = 1,
  dt = 0.5,
  lining = GYPSUM,
}) {
  const n = graph.nodes.length / 3;
  const { rooms, nodeRoom } = roomGraph;
  const fireRoom = fireRoomOf(nodeRoom, seed);
  const a = alpha ?? GROWTH[growth] ?? GROWTH.medium;
  const K0 = Math.max(1, Math.round(horizon / frameSeconds));
  const H = K0 * frameSeconds;
  const sim = simulateRooms(roomGraph, { fireRoom, alpha: a, peak, dt, horizon: H, lining });
  const stride = Math.max(1, Math.ceil((K0 * n) / MAX_CELLS));
  const K = Math.ceil(K0 / stride);
  const f = stride * frameSeconds;

  // Per room: the FED rate of each frame's hottest state, and the first step
  // at which the room is untenable (section 5).
  const R = rooms.length;
  const ambientRate = convectiveDoseRate(T_AMBIENT - KELVIN);
  const frameRate = new Float32Array((R + 1) * K).fill(ambientRate);
  const roomUntenable = new Float64Array(R + 1).fill(Infinity);
  const peakC = new Float64Array(R + 1).fill(T_AMBIENT - KELVIN);
  for (let k = 0; k < R; k += 1) {
    const zk = sim.z[k], rk = sim.rise[k];
    if (!zk) continue;
    const Hk = rooms[k].height;
    let i = 0, hottest = -Infinity;
    for (let frame = 0; frame < K; frame += 1) {
      const end = Math.min((frame + 1) * f, H);
      for (; i <= sim.steps && i * dt < end; i += 1) {
        const T = nodeTemperatureC(zk[i], Hk, rk[i], zHead);
        if (T > hottest) hottest = T;
        if (roomUntenable[k] === Infinity && inUpperLayer(zk[i], Hk, zHead) && (smoke || T >= untenableC)) {
          roomUntenable[k] = i * dt;
        }
      }
      frameRate[k * K + frame] = convectiveDoseRate(hottest);
    }
    peakC[k] = hottest;
  }

  // Nodes take their room's frames; index R stands for "no room" (ambient).
  const rate = new Float32Array(K * n);
  const untenable = new Float64Array(n);
  const closes = new Float64Array(n);
  const roomOf = new Int32Array(n);
  for (let v = 0; v < n; v += 1) {
    roomOf[v] = nodeRoom[v] >= 0 ? nodeRoom[v] : R;
    untenable[v] = roomUntenable[roomOf[v]];
    closes[v] = Math.min(untenable[v], H);
  }
  for (let frame = 0; frame < K; frame += 1) {
    const row = frame * n;
    for (let v = 0; v < n; v += 1) rate[row + v] = frameRate[roomOf[v] * K + frame];
  }

  let reached = 0, untenableRooms = 0, hottestC = T_AMBIENT - KELVIN;
  for (let k = 0; k < R; k += 1) {
    if (sim.z[k] && sim.z[k][sim.steps] < rooms[k].height) reached += 1;
    if (roomUntenable[k] < H) untenableRooms += 1;
    if (peakC[k] > hottestC) hottestC = peakC[k];
  }
  let inRooms = 0;
  for (let v = 0; v < n; v += 1) if (nodeRoom[v] >= 0) inRooms += 1;
  const fire = rooms[fireRoom];
  return {
    n,
    K,
    stride,
    dt: frameSeconds,
    frameSeconds: f,
    horizon: H,
    rate,
    untenable,
    closes,
    threshold: untenableC,
    model: "zone",
    zone: {
      fireRoom,
      fireRoomId: fire.itemId ?? null,
      fireRoomArea: fire.floorArea,
      fireRoomHeight: fire.height,
      fireRoomUntenable: roomUntenable[fireRoom],
      alpha: a,
      growth: alpha === null ? growth : null,
      peak,
      Qmax: sim.fire.Qmax,
      ventLimited: sim.fire.ventLimited,
      ventArea: sim.fire.Ao,
      ventHeight: sim.fire.Ho,
      leakage: sim.fire.leakage,
      flashover: sim.flashover,
      rooms: R,
      roomsReached: reached,
      roomsReached300: H >= 300 ? roomsReachedBy(sim, rooms, 300) : null,
      roomsUntenable: untenableRooms,
      hottestC,
      nodesInRooms: n ? inRooms / n : 0,
      untenableC,
      smoke,
      zHead,
      eulerStep: dt,
    },
    sim,
  };
}
