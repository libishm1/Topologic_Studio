/**
 * Jeff Jones's multi-agent slime mould (Physarum polycephalum) on a 2-D
 * grid: many small agents that lay a trail and follow it, whose trail settles
 * into transport networks and shortest paths. Written from the pseudocode in
 * wiki/research/physarum-jones-models-2026-11-11.md, sections 1 to 3.
 *
 * The network model (Jones 2010, doi:10.1162/artl.2010.16.2.16202; Jones
 * 2011, doi:10.1007/s11047-010-9223-z). Each agent stands on one cell, at
 * most one agent a cell, with a real position and a heading. Each step the
 * food cells add attractant to the trail over their 3x3 (less while an agent
 * is within it). Then every agent, in a fresh random order, steps `ss` cells
 * ahead and lays `depT` trail on the cell it reaches, or, blocked by a wall
 * or another agent, goes back to its cell's centre with a random heading and
 * lays nothing. Then every agent, in another random order, reads the trail
 * at three sensors `so` cells away (straight on, and `sa` degrees to either
 * side) and turns `ra` degrees towards the strongest. Last the trail
 * diffuses: each cell takes the mean of its 3x3, less the share `damp`, and
 * walls and the border absorb what reaches them. With `adaptive`, every
 * `every` steps an agent that has just moved into a sparse place (its
 * `gw` x `gw` window holds more than `gMin` and at most `gMax` agents) buds a
 * new one beside it, and one in a crowd (its `sw` x `sw` window holds fewer
 * than `sMin` or more than `sMax`) dies, so the population grows into empty
 * floor and thins where it piles up (Jones 2011).
 *
 * The erosion model. The same agents wear a substrate down instead of laying
 * trail: each step every cell projects `projectScale` times its substrate
 * into the trail, the agents turn towards the weakest of their three
 * readings, each step an agent takes wears `erodeBy` off the cell it reaches
 * (never below zero), and with probability `pTP` an agent jumps to a random
 * free cell instead. Worn cells project less, so the agents gather there and
 * wear them more, and channels deepen as desire lines do in grass.
 *
 * Chemotactic shortest paths (Jones and Saeed 2007, section 6.4,
 * doi:10.3233/mgs-2007-3108; after the chemical waves that find the way
 * through a labyrinth, Steinbock et al. 1995, doi:10.1126/science.267.5199.868).
 * A chemical is released at the exit each step and diffuses over the free
 * cells, decaying by `damp` and absorbed by the walls. Each agent faces the
 * neighbouring cell with the most chemical and steps there if no one stands
 * on it, laying `depT` trail; the trail fades by `erode` a step. The trail
 * that persists runs along the shortest way from the start to the exit, and
 * when a wall opens a shorter way, it takes over and the old one fades.
 *
 * Checked against the author's example code (2026-11-11; the code is not in
 * this repository): motor stage
 * before sensory, each in a fresh random order; a step blocked by a wall or
 * by any agent, itself included (a step that rounds back to its own cell);
 * the sensors measured from the agent's cell; when both sides beat ahead,
 * the turn towards the larger side and left on a tie (Jones 2010 describes a
 * random choice; FireEgress.Physarum.turn_tie_left); a random turn drawn
 * before a jump, and a jump to a cell the agent may start on with a random
 * heading; the adaptive tests on steps `every`, 2 `every`, ... and not within
 * `border` cells of the edge. Without `wrap` a sensor past the border reads
 * -1 and turns its agent by `ra`, a step past it reflects the heading (the
 * author's reflection can round onto the far edge at an unwalled border;
 * this one does not), and the diffusion zeroes the border ring as his does;
 * with `wrap` (his erosion example) the grid is a torus. Where this differs
 * on purpose: an erosion agent reads a wall as the least attractive cell
 * (in his, a wall's zero trail would draw the low-seekers to it), and the
 * sensor angles are not truncated to whole degrees. Cells are indexed
 * y * width + x and cell (x, y) is the unit square centred on (x, y).
 * Headings are radians; the angles among the parameters are degrees, as in
 * the papers. One seed gives one run (mulberry32), as the crowd's does
 * (crowd.js). The turn, the one-agent-a-cell movement, the diffusion's
 * totals and bounds and the erosion are proved in
 * formal/fireegress/FireEgress/Physarum.lean.
 */

/** The network model's parameters: the defaults of the author's map example (Jones 2011). */
export const NETWORK_DEFAULTS = {
  // Sensor offset (cells), sensor angle and rotation angle (degrees), step size (cells).
  so: 3,
  sa: 60,
  ra: 60,
  ss: 1,
  // Trail laid a successful step; the chance of a random turn instead of a step.
  depT: 5,
  pCD: 0,
  // The share of the trail lost each step.
  damp: 0.1,
  // Attractant each food cell adds over its 3x3 a step, and while an agent is within it.
  food: 10,
  foodCovered: 2.5,
  // Growth and shrinkage of the population, tested every `every` steps (on
  // steps every, 2 every, ...), and not within `border` cells of the edge.
  adaptive: false,
  every: 5,
  gw: 9,
  gMin: 0,
  gMax: 10,
  sw: 5,
  sMin: 0,
  sMax: 23,
  border: 5,
  // Whether the grid wraps round (a torus) or its border absorbs.
  wrap: false,
  // Agents start, and jump, only onto cells whose substrate is this value
  // (the author's examples: 0, the floor's colour); null: any free cell.
  startOn: null,
};

/** The erosion model's parameters, over the network model's: the author's erosion example. */
export const EROSION_DEFAULTS = {
  so: 11,
  sa: 22.5,
  ra: 22.5,
  ss: 1,
  depT: 0,
  pCD: 0.05,
  // The chance an agent jumps to a random free cell instead of stepping.
  pTP: 0.01,
  damp: 0.1,
  // The share of each cell's substrate projected into the trail a step, and what a step wears off.
  projectScale: 0.01,
  erodeBy: 10,
  // The author's erosion example wraps round.
  wrap: true,
};

/** The chemotaxis model's parameters (the note gives none; these find the paths in the tests). */
export const CHEMOTAXIS_DEFAULTS = {
  // Chemical released at the exit a step, and the share of it lost a step.
  source: 100,
  damp: 0.1,
  // Trail laid a step taken, and the share of it lost a step.
  depT: 5,
  erode: 0.02,
};

/** The two models createPhysarum runs. */
export const MODES = ["network", "erosion"];

const TAU = 2 * Math.PI;
const RAD = Math.PI / 180;
// Marks for the end of a step.
const DIVIDE = 1;
const DIE = 2;
// Random cells tried before a jump gives up.
const TRIES = 32;

/** A seeded uniform generator (mulberry32), so one seed gives one run. */
function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** An angle in [0, 2 pi). */
function wrap(a) {
  a %= TAU;
  return a < 0 ? a + TAU : a;
}

/**
 * One diffusion step of a field on the grid, in place: each cell becomes
 * (1 - damp) times the mean of its 3x3 (divisor 9), walls and the cells past
 * the border counting as zero, and walls end at zero; without wrap-around
 * the cells on the border end at zero too, as in the author's code. So walls
 * and the border absorb what reaches them, and away from both the total falls
 * by exactly (1 - damp). With `wrap` the grid is a torus and only walls
 * absorb: with none the total falls by exactly (1 - damp)
 * (FireEgress.Physarum.diffuse_total; diffuse_absorbing_total_le bounds the
 * rest). The 3x3 sum is taken as a row sum then a column sum; `scratch` is a
 * buffer of the field's length.
 */
export function diffuse(field, width, height, walls, damp, scratch = new Float64Array(field.length), { wrap: torus = false } = {}) {
  if (walls) for (let i = 0; i < width * height; i += 1) if (walls[i]) field[i] = 0;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const left = x > 0 ? field[row + x - 1] : torus ? field[row + width - 1] : 0;
      const right = x + 1 < width ? field[row + x + 1] : torus ? field[row] : 0;
      scratch[row + x] = left + field[row + x] + right;
    }
  }
  const k = (1 - damp) / 9;
  for (let y = 0; y < height; y += 1) {
    const row = y * width;
    for (let x = 0; x < width; x += 1) {
      const i = row + x;
      // Walls, and without wrap-around the cells on the border, end at zero,
      // as the author's code has them (his border ring absorbs).
      if ((walls && walls[i]) || (!torus && (x === 0 || y === 0 || x === width - 1 || y === height - 1))) {
        field[i] = 0;
        continue;
      }
      const up = y > 0 ? scratch[i - width] : torus ? scratch[(height - 1) * width + x] : 0;
      const down = y + 1 < height ? scratch[i + width] : torus ? scratch[x] : 0;
      field[i] = (up + scratch[i] + down) * k;
    }
  }
  return field;
}

/** The cells of the grid within `radius` of (cx, cy), or all of them without a centre. */
function cellsWithin(width, height, around) {
  const out = [];
  if (!around) {
    for (let i = 0; i < width * height; i += 1) out.push(i);
    return out;
  }
  const [cx, cy, r = 0] = around;
  for (let y = Math.max(0, Math.ceil(cy - r)); y <= Math.min(height - 1, Math.floor(cy + r)); y += 1) {
    for (let x = Math.max(0, Math.ceil(cx - r)); x <= Math.min(width - 1, Math.floor(cx + r)); x += 1) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= r * r) out.push(y * width + x);
    }
  }
  return out;
}

/** Room for at least `need` agents, the arrays doubled as they fill. */
function reserve(s, need) {
  if (need <= s.capacity) return;
  const capacity = Math.max(need, s.capacity * 2, 16);
  const grow = (a, Type) => {
    const b = new Type(capacity);
    if (a) b.set(a.subarray(0, s.count));
    return b;
  };
  s.x = grow(s.x, Float32Array);
  s.y = grow(s.y, Float32Array);
  s.heading = grow(s.heading, Float32Array);
  s.cell = grow(s.cell, Int32Array);
  s.mark = grow(s.mark, Uint8Array);
  s.order = new Int32Array(capacity);
  s.capacity = capacity;
}

/** A new agent at the centre of free cell c. */
function addAgent(s, c, heading) {
  reserve(s, s.count + 1);
  const i = s.count;
  s.x[i] = c % s.width;
  s.y[i] = Math.floor(c / s.width);
  s.heading[i] = heading;
  s.cell[i] = c;
  s.mark[i] = 0;
  s.occupied[c] = i;
  s.count += 1;
  return i;
}

/** Agent i gone, the last agent moved into its place. */
function removeAgent(s, i) {
  s.occupied[s.cell[i]] = -1;
  const last = s.count - 1;
  if (i !== last) {
    s.x[i] = s.x[last];
    s.y[i] = s.y[last];
    s.heading[i] = s.heading[last];
    s.cell[i] = s.cell[last];
    s.mark[i] = s.mark[last];
    s.occupied[s.cell[i]] = i;
  }
  s.count = last;
}

/**
 * Whether an agent may start, or jump, onto cell c: no wall, no agent, and
 * with `startOn` set, a substrate of that value (the erosion model's), or
 * no food (the network model's, whose substrate is the floor).
 */
function startable(s, c) {
  if (s.walls[c] || s.occupied[c] >= 0) return false;
  const on = s.params.startOn;
  if (on === null || on === undefined) return true;
  return s.substrate ? s.substrate[c] === on : !s.isFood[c];
}

/** Up to `want` agents on random free cells among `cells` (a partial Fisher-Yates shuffle), random headings. */
function scatter(s, cells, want) {
  const free = cells.filter((c) => startable(s, c));
  const n = Math.min(want, free.length);
  for (let k = 0; k < n; k += 1) {
    const j = k + Math.floor(s.rand() * (free.length - k));
    [free[k], free[j]] = [free[j], free[k]];
    addAgent(s, free[k], s.rand() * TAU);
  }
}

/** The first `count` agent indices in a fresh random order. */
function shuffled(s) {
  const { order, count, rand } = s;
  for (let k = 0; k < count; k += 1) order[k] = k;
  for (let k = count - 1; k > 0; k -= 1) {
    const j = Math.floor(rand() * (k + 1));
    const t = order[k];
    order[k] = order[j];
    order[j] = t;
  }
  return order;
}

/** How many agents stand in the size x size window centred on (cx, cy), the agent there counted. */
function crowdAround(s, cx, cy, size) {
  const r = size >> 1;
  const { width, height, occupied } = s;
  let n = 0;
  for (let y = Math.max(0, cy - r); y <= Math.min(height - 1, cy + r); y += 1) {
    const row = y * width;
    for (let x = Math.max(0, cx - r); x <= Math.min(width - 1, cx + r); x += 1) if (occupied[row + x] >= 0) n += 1;
  }
  return n;
}

/**
 * Agent i jumps to a random cell it may start on, with a random heading (the
 * author's code places it afresh), if one turns up in TRIES draws; whether it did.
 */
function teleport(s, i) {
  const { width, height, occupied, rand } = s;
  for (let k = 0; k < TRIES; k += 1) {
    const c = Math.floor(rand() * width * height);
    if (!startable(s, c)) continue;
    occupied[s.cell[i]] = -1;
    occupied[c] = i;
    s.cell[i] = c;
    s.x[i] = c % width;
    s.y[i] = Math.floor(c / width);
    s.heading[i] = rand() * TAU;
    return true;
  }
  return false;
}

/**
 * A network or erosion model on a grid.
 *
 * @param {object} opts
 *   width, height  the grid, in cells
 *   walls          Uint8Array width x height, 1 a wall (copied; null: none)
 *   food           [x, y] cells that project attractant each step
 *   substrate      Float32Array width x height, the erosion model's (copied; null: zero)
 *   agents         how many to place on random free cells
 *   start          [x, y, radius]: place them within this disc only (as many as fit)
 *   mode           "network" (default) or "erosion" (MODES)
 *   seed           for the placement and every random choice after it
 *   maxAgents      at most this many agents, placed or budded (the free cells
 *                  bound them anyway; the arrays start at the placed count and double)
 *   ...params      over NETWORK_DEFAULTS, and EROSION_DEFAULTS in the erosion mode
 * @returns the state stepPhysarum advances: trail (Float32Array), occupied
 *   (Int32Array, the agent on each cell or -1), the agents as x, y, heading
 *   (Float32Array, radians), cell (Int32Array) and count, substrate in the
 *   erosion mode, t (steps taken), and running totals moved (steps taken by
 *   agents), blocked, teleported, born and died.
 */
export function createPhysarum({
  width,
  height,
  walls = null,
  food = [],
  substrate = null,
  agents = 0,
  start = null,
  mode = "network",
  seed = 1,
  maxAgents = Infinity,
  ...params
} = {}) {
  if (!MODES.includes(mode)) throw new Error(`physarum: unknown mode "${mode}"`);
  const n = width * height;
  const p = { ...NETWORK_DEFAULTS, pTP: 0, projectScale: 0, erodeBy: 0, ...(mode === "erosion" ? EROSION_DEFAULTS : {}), ...params };
  const wall = walls ? Uint8Array.from(walls) : new Uint8Array(n);
  let freeCells = 0;
  for (let i = 0; i < n; i += 1) if (!wall[i]) freeCells += 1;
  const foodCells = [];
  for (const [fx, fy] of food) if (fx >= 0 && fx < width && fy >= 0 && fy < height) foodCells.push(fy * width + fx);
  const s = {
    mode,
    width,
    height,
    params: p,
    walls: wall,
    trail: new Float32Array(n),
    substrate: mode === "erosion" ? (substrate ? Float32Array.from(substrate) : new Float32Array(n)) : null,
    food: Int32Array.from(foodCells),
    isFood: new Uint8Array(n),
    occupied: new Int32Array(n).fill(-1),
    count: 0,
    capacity: 0,
    maxAgents: Math.min(maxAgents, freeCells),
    x: null,
    y: null,
    heading: null,
    cell: null,
    mark: null,
    order: null,
    scratch: new Float64Array(n),
    seed,
    rand: random(seed),
    t: 0,
    moved: 0,
    blocked: 0,
    teleported: 0,
    born: 0,
    died: 0,
  };
  for (const c of foodCells) s.isFood[c] = 1;
  reserve(s, Math.max(1, Math.min(agents, s.maxAgents)));
  scatter(s, cellsWithin(width, height, start), Math.min(agents, s.maxAgents));
  return s;
}

/** Steps the network or erosion model on; returns the state. */
export function stepPhysarum(s, steps = 1) {
  const { width: W, height: H, walls, trail, occupied, substrate, food, scratch, params: p } = s;
  const erosion = s.mode === "erosion";
  const ss = p.ss, so = p.so, sa = p.sa * RAD, ra = p.ra * RAD, torus = Boolean(p.wrap);
  const mod = (v, m) => ((v % m) + m) % m;
  // The cell a sensor reads, measured from the agent's cell as the author's
  // code does: wrapped round, or -1 past the border (where the agent turns by
  // ra whatever the other two read).
  const sensor = (px, py, a) => {
    let sx = Math.round(px + so * Math.cos(a)), sy = Math.round(py + so * Math.sin(a));
    if (torus) {
      sx = mod(sx, W);
      sy = mod(sy, H);
    }
    return sx < 0 || sx >= W || sy < 0 || sy >= H ? -1 : sy * W + sx;
  };
  // Not within `border` cells of the edge: the adaptive tests' windows stay on the grid.
  const inside = (cx, cy) => cx >= p.border && cy >= p.border && cx < W - p.border && cy < H - p.border;
  // In the erosion mode the readings are negated, so the same rules turn
  // towards the least, and a wall reads the most of all, so those agents keep
  // off walls as the network's do (a wall's trail is zero, the least there is).
  const read = erosion ? (c) => (walls[c] ? -Infinity : -trail[c]) : (c) => trail[c];
  for (let step = 0; step < steps; step += 1) {
    const rand = s.rand;
    // The author's step counter starts at 1: the tests fall on steps every, 2 every, ...
    const test = p.adaptive && (s.t + 1) % p.every === 0;

    // Food over its 3x3, less while an agent is within it.
    for (let f = 0; f < food.length; f += 1) {
      const fx = food[f] % W, fy = Math.floor(food[f] / W);
      let covered = false;
      for (let y = Math.max(0, fy - 1); y <= Math.min(H - 1, fy + 1) && !covered; y += 1) {
        for (let x = Math.max(0, fx - 1); x <= Math.min(W - 1, fx + 1); x += 1) if (occupied[y * W + x] >= 0) covered = true;
      }
      const v = covered ? p.foodCovered : p.food;
      for (let y = Math.max(0, fy - 1); y <= Math.min(H - 1, fy + 1); y += 1) {
        for (let x = Math.max(0, fx - 1); x <= Math.min(W - 1, fx + 1); x += 1) if (!walls[y * W + x]) trail[y * W + x] += v;
      }
    }
    // The substrate projected into the trail.
    if (erosion && p.projectScale) for (let i = 0; i < W * H; i += 1) if (!walls[i]) trail[i] += p.projectScale * substrate[i];

    // Motor stage.
    const { x, y, heading, cell, mark } = s;
    let order = shuffled(s);
    for (let k = 0; k < s.count; k += 1) {
      const i = order[k];
      // A random turn first, then (erosion) a jump, in the author's order.
      if (p.pCD > 0 && rand() < p.pCD) {
        heading[i] = rand() * TAU;
        x[i] = cell[i] % W;
        y[i] = Math.floor(cell[i] / W);
        continue;
      }
      if (p.pTP > 0 && rand() < p.pTP) {
        if (teleport(s, i)) s.teleported += 1;
        continue;
      }
      let h = heading[i];
      let tx = x[i] + ss * Math.cos(h), ty = y[i] + ss * Math.sin(h);
      if (torus) {
        tx = mod(tx, W);
        ty = mod(ty, H);
      }
      let cx = Math.round(tx), cy = Math.round(ty);
      if (torus) {
        cx = mod(cx, W);
        cy = mod(cy, H);
      }
      if (!torus && (cx < 0 || cx >= W || cy < 0 || cy >= H)) {
        // Past the border: the heading reflected off it, and the step taken that way.
        if (cx < 0 || cx >= W) h = Math.PI - h;
        if (cy < 0 || cy >= H) h = -h;
        h = wrap(h);
        heading[i] = h;
        tx = x[i] + ss * Math.cos(h);
        ty = y[i] + ss * Math.sin(h);
        cx = Math.round(tx);
        cy = Math.round(ty);
      }
      const here = cell[i];
      const c = cy * W + cx;
      // Any agent on the target blocks, itself included: a step that rounds
      // back to its own cell is blocked, as in the author's code.
      if (cx < 0 || cx >= W || cy < 0 || cy >= H || walls[c] || occupied[c] >= 0) {
        // Blocked: a random heading, back to the cell's centre, no trail.
        heading[i] = rand() * TAU;
        x[i] = here % W;
        y[i] = Math.floor(here / W);
        s.blocked += 1;
        continue;
      }
      occupied[here] = -1;
      occupied[c] = i;
      cell[i] = c;
      x[i] = tx;
      y[i] = ty;
      trail[c] += p.depT;
      if (erosion) substrate[c] = Math.max(0, substrate[c] - p.erodeBy);
      s.moved += 1;
      if (test && inside(cx, cy)) {
        const m = crowdAround(s, cx, cy, p.gw);
        if (m > p.gMin && m <= p.gMax) mark[i] |= DIVIDE;
      }
    }

    // Sensory stage.
    order = shuffled(s);
    for (let k = 0; k < s.count; k += 1) {
      const i = order[k];
      const ax = cell[i] % W, ay = Math.floor(cell[i] / W);
      if (test && inside(ax, ay)) {
        const m = crowdAround(s, ax, ay, p.sw);
        if (m < p.sMin || m > p.sMax) mark[i] |= DIE;
      }
      // The sensors are measured from the agent's cell, not its real position.
      const h = heading[i], px = ax, py = ay;
      const cF = sensor(px, py, h), cL = sensor(px, py, h - sa), cR = sensor(px, py, h + sa);
      let turn = 0;
      if (cF < 0 || cL < 0 || cR < 0) turn = ra;
      else {
        const F = read(cF), FL = read(cL), FR = read(cR);
        if (F > FL && F > FR) turn = 0;
        else if (F < FL && F < FR) turn = FL < FR ? ra : -ra;
        else if (FL < FR) turn = ra;
        else if (FR < FL) turn = -ra;
      }
      if (turn) heading[i] = wrap(h + turn);
    }

    diffuse(trail, W, H, walls, p.damp, scratch, { wrap: torus });

    // Buds, then deaths.
    if (test) {
      const before = s.count;
      for (let i = 0; i < before; i += 1) {
        if (!(s.mark[i] & DIVIDE)) continue;
        s.mark[i] &= ~DIVIDE;
        if (s.count >= s.maxAgents) continue;
        const ax = s.cell[i] % W, ay = Math.floor(s.cell[i] / W);
        // A random free cell of its 3x3 (reservoir sampling).
        let pick = -1, seen = 0;
        for (let yy = Math.max(0, ay - 1); yy <= Math.min(H - 1, ay + 1); yy += 1) {
          for (let xx = Math.max(0, ax - 1); xx <= Math.min(W - 1, ax + 1); xx += 1) {
            const c = yy * W + xx;
            if (walls[c] || occupied[c] >= 0) continue;
            seen += 1;
            if (rand() * seen < 1) pick = c;
          }
        }
        if (pick < 0) continue;
        addAgent(s, pick, rand() * TAU);
        s.born += 1;
      }
      for (let i = s.count - 1; i >= 0; i -= 1) {
        if (!(s.mark[i] & DIE)) continue;
        removeAgent(s, i);
        s.died += 1;
      }
    }
    s.t += 1;
  }
  return s;
}

/**
 * The chemotaxis model on a grid: agents climbing a chemical released at an
 * exit, from a start region they go back to when they reach it.
 *
 * @param {object} opts
 *   width, height  the grid, in cells
 *   walls          Uint8Array width x height, 1 a wall (copied)
 *   exit           [x, y], the cell the chemical is released at
 *   start          [x, y, radius]: the agents begin within this disc, and go back to it from the exit
 *   agents         how many (as many as fit in the start)
 *   seed           for the placement and every random choice after it
 *   ...params      over CHEMOTAXIS_DEFAULTS: source, damp, depT, erode
 * @returns the state stepChemotaxis advances: chemical (Float64Array, which
 *   keeps the exponential tail of the field a long way from the exit), trail
 *   (Float32Array), occupied, the agents as x, y, heading, cell and count,
 *   t, and running totals moved and arrived.
 */
export function createChemotaxis({ width, height, walls = null, exit, start, agents = 0, seed = 1, ...params } = {}) {
  const n = width * height;
  const p = { ...CHEMOTAXIS_DEFAULTS, ...params };
  const s = {
    mode: "chemotaxis",
    width,
    height,
    params: p,
    walls: walls ? Uint8Array.from(walls) : new Uint8Array(n),
    chemical: new Float64Array(n),
    trail: new Float32Array(n),
    occupied: new Int32Array(n).fill(-1),
    exit: exit[1] * width + exit[0],
    startCells: Int32Array.from(cellsWithin(width, height, start)),
    count: 0,
    capacity: 0,
    x: null,
    y: null,
    heading: null,
    cell: null,
    mark: null,
    order: null,
    scratch: new Float64Array(n),
    seed,
    rand: random(seed),
    t: 0,
    moved: 0,
    arrived: 0,
  };
  reserve(s, Math.max(1, agents));
  scatter(s, Array.from(s.startCells), agents);
  return s;
}

/** Agent i back to a random free cell of the start; whether one was free. */
function backToStart(s, i) {
  const { startCells, walls, occupied, rand } = s;
  let pick = -1, seen = 0;
  for (let k = 0; k < startCells.length; k += 1) {
    const c = startCells[k];
    if (walls[c] || occupied[c] >= 0) continue;
    seen += 1;
    if (rand() * seen < 1) pick = c;
  }
  if (pick < 0) return false;
  occupied[s.cell[i]] = -1;
  occupied[pick] = i;
  s.cell[i] = pick;
  s.x[i] = pick % s.width;
  s.y[i] = Math.floor(pick / s.width);
  s.heading[i] = rand() * TAU;
  return true;
}

/** Steps the chemotaxis model on; returns the state. */
export function stepChemotaxis(s, steps = 1) {
  const { width: W, height: H, walls, chemical, trail, occupied, scratch, exit, params: p } = s;
  const keep = 1 - p.erode;
  for (let step = 0; step < steps; step += 1) {
    const rand = s.rand;
    chemical[exit] += p.source;
    diffuse(chemical, W, H, walls, p.damp, scratch);
    const { x, y, heading, cell } = s;
    const order = shuffled(s);
    for (let k = 0; k < s.count; k += 1) {
      const i = order[k];
      const here = cell[i];
      // One left on the exit when the start was full tries again.
      if (here === exit) {
        backToStart(s, i);
        continue;
      }
      const ax = here % W, ay = Math.floor(here / W);
      // The neighbour (of eight, not walls) with the most chemical, ties at random.
      let best = -Infinity, pick = -1, ties = 0;
      for (let yy = Math.max(0, ay - 1); yy <= Math.min(H - 1, ay + 1); yy += 1) {
        for (let xx = Math.max(0, ax - 1); xx <= Math.min(W - 1, ax + 1); xx += 1) {
          const c = yy * W + xx;
          if (c === here || walls[c]) continue;
          const v = chemical[c];
          if (v > best) {
            best = v;
            pick = c;
            ties = 1;
          } else if (v === best) {
            ties += 1;
            if (rand() * ties < 1) pick = c;
          }
        }
      }
      if (pick < 0) continue;
      const px = pick % W, py = Math.floor(pick / W);
      heading[i] = wrap(Math.atan2(py - ay, px - ax));
      if (occupied[pick] >= 0) continue;
      occupied[here] = -1;
      occupied[pick] = i;
      cell[i] = pick;
      x[i] = px;
      y[i] = py;
      trail[pick] += p.depT;
      s.moved += 1;
      if (pick === exit) {
        s.arrived += 1;
        backToStart(s, i);
      }
    }
    for (let c = 0; c < W * H; c += 1) trail[c] *= keep;
    s.t += 1;
  }
  return s;
}

/**
 * Makes cell (x, y) a wall (`wall` true) or free floor, mid-run, in any of
 * the models' states. A new wall loses its trail and chemical, and an agent
 * standing on it is removed.
 */
export function setWall(s, x, y, wall = true) {
  if (x < 0 || x >= s.width || y < 0 || y >= s.height) return s;
  const c = y * s.width + x;
  s.walls[c] = wall ? 1 : 0;
  if (!wall) return s;
  s.trail[c] = 0;
  if (s.chemical) s.chemical[c] = 0;
  if (s.occupied[c] >= 0) removeAgent(s, s.occupied[c]);
  return s;
}

/** Opens cell (x, y): no longer a wall. */
export function openWall(s, x, y) {
  return setWall(s, x, y, false);
}
