/**
 * A small reader of FDS input files (NIST's Fire Dynamics Simulator, the
 * validation inputs of github.com/firemodels/fds, Validation/<name>/FDS_Input_Files)
 * and a translator of their simple geometry to the Flamescapes solver's scene.
 * Only what the held-out validation needs (tools/calibration/validation-suite.mjs):
 *
 *   &MESH XB (with &MULT), &OBST XB (minus &HOLE XB), &VENT XB or MB with
 *   SURF_ID='OPEN', &SURF HRRPUA with RAMP_Q, &RAMP T/F, &MISC TMPA,
 *   &DEVC XYZ or XB with ID and QUANTITY, &REAC FUEL, &PROP FLOW_RATE.
 *
 * FDS reads only namelist groups that begin with '&'; a line without one is a
 * comment (the Steckler inputs switch their burners and doors on and off that
 * way), and so it is here. The input files are data: nothing in them is run.
 *
 * Coordinates: FDS's (x, y, z up). The solver's grid is the viewer's (x, y up,
 * z), so FDS (x, y, z) maps to grid (x, z, y), as tools/sprinkler-validation.mjs
 * maps its data frame.
 */
import * as V from "../../../topologicpy-web-frontend/src/core/voxelize.js";

// ------------------------------------------------------------ the namelists

/** Split a group body into its key = value pairs (values as arrays of numbers, strings or booleans). */
function parseBody(body) {
  const out = {};
  const keyRe = /([A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?)\s*=/g;
  const keys = [];
  let m;
  // Keys outside quoted strings only.
  const quoted = [];
  {
    let q = null, start = 0;
    for (let i = 0; i < body.length; i += 1) {
      const ch = body[i];
      if (q) {
        if (ch === q) {
          quoted.push([start, i]);
          q = null;
        }
      } else if (ch === "'" || ch === '"') {
        q = ch;
        start = i;
      }
    }
  }
  const inQuote = (i) => quoted.some(([a, b]) => i > a && i < b);
  while ((m = keyRe.exec(body))) if (!inQuote(m.index)) keys.push({ key: m[1].toUpperCase(), at: m.index, end: keyRe.lastIndex });
  for (let q = 0; q < keys.length; q += 1) {
    const raw = body.slice(keys[q].end, q + 1 < keys.length ? keys[q + 1].at : body.length);
    const values = [];
    const tokenRe = /'([^']*)'|"([^"]*)"|([^,\s]+)/g;
    let t;
    while ((t = tokenRe.exec(raw))) {
      if (t[1] !== undefined || t[2] !== undefined) values.push(t[1] ?? t[2]);
      else {
        const s = t[3];
        if (/^\.?(true|t)\.?$/i.test(s)) values.push(true);
        else if (/^\.?(false|f)\.?$/i.test(s)) values.push(false);
        else {
          const x = Number(s.replace(/[dD]/, "e"));
          values.push(Number.isFinite(x) ? x : s);
        }
      }
    }
    out[keys[q].key] = values;
  }
  return out;
}

/**
 * The namelist groups of an FDS input as [{group, ...keys}], in file order.
 * A group starts with '&' at the start of a line (after blanks) and ends at
 * the first '/' outside quotes.
 */
export function parseFds(text) {
  const groups = [];
  const lines = text.split(/\r?\n/);
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const start = line.search(/^\s*&[A-Za-z]/);
    if (start < 0) {
      i += 1;
      continue;
    }
    let buf = line.slice(line.indexOf("&") + 1);
    let end = -1;
    for (;;) {
      let q = null;
      for (let c = 0; c < buf.length; c += 1) {
        const ch = buf[c];
        if (q) {
          if (ch === q) q = null;
        } else if (ch === "'" || ch === '"') q = ch;
        else if (ch === "/") {
          end = c;
          break;
        }
      }
      if (end >= 0 || i + 1 >= lines.length) break;
      i += 1;
      buf += "\n" + lines[i];
    }
    const body = end >= 0 ? buf.slice(0, end) : buf;
    const name = body.match(/^[A-Za-z]+/)[0].toUpperCase();
    groups.push({ group: name, ...parseBody(body.slice(name.length)) });
    i += 1;
  }
  return groups;
}

const one = (g, k, fallback = undefined) => (g[k] && g[k].length ? g[k][0] : fallback);

/** Apply an &MULT to an XB: every copy. */
function multiplied(xb, mult) {
  if (!mult) return [xb];
  const [dx, dy, dz] = [one(mult, "DX", 0), one(mult, "DY", 0), one(mult, "DZ", 0)];
  const out = [];
  for (let i = one(mult, "I_LOWER", 0); i <= one(mult, "I_UPPER", 0); i += 1) {
    for (let j = one(mult, "J_LOWER", 0); j <= one(mult, "J_UPPER", 0); j += 1) {
      for (let k = one(mult, "K_LOWER", 0); k <= one(mult, "K_UPPER", 0); k += 1) {
        out.push([xb[0] + i * dx, xb[1] + i * dx, xb[2] + j * dy, xb[3] + j * dy, xb[4] + k * dz, xb[5] + k * dz]);
      }
    }
  }
  return out;
}

const sorted = (xb) => [Math.min(xb[0], xb[1]), Math.max(xb[0], xb[1]), Math.min(xb[2], xb[3]), Math.max(xb[2], xb[3]), Math.min(xb[4], xb[5]), Math.max(xb[4], xb[5])];

/**
 * The geometry and fire of an FDS input, reduced to plain data:
 * { meshes: [xb], obsts: [{xb, surf, surfs, label}], holes: [xb], open: [xb],
 *   surfs: {id: {hrrpua, rampQ}}, ramps: {id: [[t, f]]}, tmpa, fuel,
 *   devices: [{id, quantity, xyz | xb}], props: {id: {flowRate, flowRamp}},
 *   burnerVents: [{xb, surf}] }
 */
export function readGeometry(text) {
  const groups = parseFds(text);
  const mults = Object.fromEntries(groups.filter((g) => g.group === "MULT").map((g) => [one(g, "ID"), g]));
  const out = { meshes: [], obsts: [], holes: [], open: [], surfs: {}, matls: {}, surfaceVents: [], ramps: {}, tmpa: 20, fuel: null, devices: [], props: {}, burnerVents: [], reac: null };
  for (const g of groups) {
    if (g.group === "MESH") for (const xb of multiplied(sorted(g.XB), mults[one(g, "MULT_ID")])) out.meshes.push(xb);
    else if (g.group === "OBST" && g.XB) for (const xb of multiplied(sorted(g.XB), mults[one(g, "MULT_ID")])) out.obsts.push({ xb, surf: one(g, "SURF_ID", null), surfs: g.SURF_IDS || null });
    else if (g.group === "HOLE" && g.XB) for (const xb of multiplied(sorted(g.XB), mults[one(g, "MULT_ID")])) out.holes.push(xb);
    else if (g.group === "VENT") {
      const surf = one(g, "SURF_ID", null);
      const spec = g.XB ? { xb: sorted(g.XB) } : g.MB ? { mb: String(one(g, "MB")).toUpperCase() } : null;
      if (!spec) continue;
      if (String(surf).toUpperCase() === "OPEN") out.open.push(spec);
      else if (surf) {
        out.burnerVents.push({ ...spec, surf });
        // A vent that gives a mesh side or a patch its own wall surface.
        out.surfaceVents.push({ ...spec, surf });
      }
    } else if (g.group === "SURF") {
      const ids = [];
      for (const [k, v] of Object.entries(g)) if (/^MATL_ID/.test(k)) ids.push(...v.filter((x) => typeof x === "string"));
      const thick = [];
      for (const [k, v] of Object.entries(g)) if (/^THICKNESS/.test(k)) thick.push(...v.filter((x) => typeof x === "number"));
      out.surfs[one(g, "ID")] = { hrrpua: one(g, "HRRPUA", null), rampQ: one(g, "RAMP_Q", null), tauQ: one(g, "TAU_Q", null), matl: ids, thickness: thick, default: one(g, "DEFAULT", false) === true };
    } else if (g.group === "MATL") out.matls[one(g, "ID")] = { rho: one(g, "DENSITY", null), k: one(g, "CONDUCTIVITY", null), kRamp: one(g, "CONDUCTIVITY_RAMP", null), c: one(g, "SPECIFIC_HEAT", null), cRamp: one(g, "SPECIFIC_HEAT_RAMP", null), eps: one(g, "EMISSIVITY", null) };
    else if (g.group === "RAMP") {
      const id = one(g, "ID");
      (out.ramps[id] ||= []).push([one(g, "T"), one(g, "F")]);
    } else if (g.group === "MISC" && g.TMPA) out.tmpa = one(g, "TMPA");
    else if (g.group === "REAC") out.reac = { fuel: one(g, "FUEL", null), soot: one(g, "SOOT_YIELD", null), co: one(g, "CO_YIELD", null), radiativeFraction: one(g, "RADIATIVE_FRACTION", null) };
    else if (g.group === "PROP") out.props[one(g, "ID")] = { flowRate: one(g, "FLOW_RATE", null), flowRamp: one(g, "FLOW_RAMP", null) };
    else if (g.group === "DEVC") out.devices.push({ id: one(g, "ID", null), quantity: one(g, "QUANTITY", null), prop: one(g, "PROP_ID", null), spec: one(g, "SPEC_ID", null), ...(g.XYZ ? { xyz: g.XYZ.slice(0, 3) } : {}), ...(g.XB ? { xb: sorted(g.XB) } : {}) });
  }
  // Burners: vents whose surface releases heat, and obstructions whose top does.
  const fires = new Set(Object.entries(out.surfs).filter(([, s]) => s.hrrpua > 0).map(([id]) => id));
  out.fire = [];
  for (const v of out.burnerVents) if (fires.has(v.surf) && v.xb) out.fire.push({ xb: v.xb, surf: v.surf });
  for (const o of out.obsts) {
    const top = o.surfs ? o.surfs[0] : o.surf;
    if (fires.has(top)) out.fire.push({ xb: [o.xb[0], o.xb[1], o.xb[2], o.xb[3], o.xb[5], o.xb[5]], surf: top });
  }
  delete out.burnerVents;
  return out;
}

/** A ramp's value at its first point (the property at ambient), else the constant. */
const atAmbient = (geo, value, ramp) => (value != null ? value : ramp && geo.ramps[ramp] ? geo.ramps[ramp][0][1] : null);

/**
 * A wall surface of an FDS input as the solver's inert material (the
 * calibration plan, section 12): its layers' areal heat capacity, sum of
 * rho c L, and areal mass, sum of rho L; the exposed layer's conductivity
 * and emissivity at ambient (FDS's default emissivity 0.9). The solver lumps
 * a wall into whole cells, so validation-suite.mjs sets the density to the
 * areal mass over the cell (rho = sum rho L / h): a wall one cell thick then
 * holds the board's heat capacity per unit area, not many times it.
 */
export function surfaceMaterial(geo, id) {
  const s = geo.surfs[id];
  if (!s) throw new Error(`no surface ${id}`);
  const layers = s.matl.map((m, q) => {
    const M = geo.matls[m];
    if (!M) throw new Error(`no material ${m}`);
    const c = atAmbient(geo, M.c, M.cRamp);
    return { matl: m, rho: M.rho, c: c * 1000, k: atAmbient(geo, M.k, M.kRamp), eps: M.eps ?? 0.9, thickness: s.thickness[q] ?? s.thickness[0] };
  });
  const arealMass = layers.reduce((a, l) => a + l.rho * l.thickness, 0);
  const arealHeat = layers.reduce((a, l) => a + l.rho * l.c * l.thickness, 0);
  return { surface: id, layers, arealMass, arealHeat, c: arealHeat / arealMass, k: layers[0].k, eps: layers[0].eps, thickness: layers.reduce((a, l) => a + l.thickness, 0) };
}

// ------------------------------------------------------- boxes and meshes

const volume = (b) => Math.max(0, b[1] - b[0]) * Math.max(0, b[3] - b[2]) * Math.max(0, b[5] - b[4]);
const overlaps = (a, b) => a[0] < b[1] && b[0] < a[1] && a[2] < b[3] && b[2] < a[3] && a[4] < b[5] && b[4] < a[5];

/** Box a minus box b, as at most six boxes. */
export function subtract(a, b) {
  if (!overlaps(a, b)) return [a];
  const out = [];
  let [x0, x1, y0, y1, z0, z1] = a;
  if (b[0] > x0) out.push([x0, b[0], y0, y1, z0, z1]), (x0 = b[0]);
  if (b[1] < x1) out.push([b[1], x1, y0, y1, z0, z1]), (x1 = b[1]);
  if (b[2] > y0) out.push([x0, x1, y0, b[2], z0, z1]), (y0 = b[2]);
  if (b[3] < y1) out.push([x0, x1, b[3], y1, z0, z1]), (y1 = b[3]);
  if (b[4] > z0) out.push([x0, x1, y0, y1, z0, b[4]]), (z0 = b[4]);
  if (b[5] < z1) out.push([x0, x1, y0, y1, b[5], z1]);
  return out.filter((p) => volume(p) > 1e-12);
}

const subtractAll = (pieces, cut) => pieces.flatMap((p) => subtract(p, cut));

/** A box in FDS coordinates as a mesh in the viewer frame (x, z, y), stone. */
function meshOf([x0, x1, y0, y1, z0, z1]) {
  const p = [x0, z0, y0, x1, z0, y0, x1, z1, y0, x0, z1, y0, x0, z0, y1, x1, z0, y1, x1, z1, y1, x0, z1, y1];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  return { positions: Float32Array.from(p), indices: Uint32Array.from(f), material: V.MATERIAL.STONE };
}

/**
 * The solid boxes of a geometry at cell h, in FDS coordinates:
 *   - the domain is the meshes' bounding box and one cell more on every side
 *     (the floor a cell thick under it); everything in it outside the meshes
 *     is solid (an FDS mesh's sides are walls unless a vent opens them);
 *   - an OPEN vent on a mesh's side opens the solid beyond it, out to the
 *     domain's edge (the solver's open boundary); one on the floor is left
 *     closed (the solver's ground is closed);
 *   - each obstruction with a volume, less the holes through it.
 * Obstructions of no thickness (FDS's thin plates: trusses, pans, flanges)
 * have no volume here and are left out, as the app's voxelisation would.
 */
export function solidBoxes(geo, h, { only = null } = {}) {
  const m = geo.meshes;
  const bb = [Math.min(...m.map((b) => b[0])), Math.max(...m.map((b) => b[1])), Math.min(...m.map((b) => b[2])), Math.max(...m.map((b) => b[3])), Math.min(...m.map((b) => b[4])), Math.max(...m.map((b) => b[5]))];
  const domain = [bb[0] - h, bb[1] + h, bb[2] - h, bb[3] + h, bb[4] - h, bb[5] + h];
  let outside = [domain];
  for (const mesh of m) outside = subtractAll(outside, mesh);
  const inside = (x, y, z) => m.some((b) => x >= b[0] - 1e-9 && x <= b[1] + 1e-9 && y >= b[2] - 1e-9 && y <= b[3] + 1e-9 && z >= b[4] - 1e-9 && z <= b[5] + 1e-9);
  const openings = [];
  for (const v of geo.open) {
    const faces = [];
    if (v.mb) {
      // The named side of every mesh on the bounding box's side.
      const d = { XMIN: 0, XMAX: 1, YMIN: 2, YMAX: 3, ZMIN: 4, ZMAX: 5 }[v.mb];
      for (const b of m) if (Math.abs(b[d] - bb[d]) < 1e-9) faces.push({ d, xb: b.map((x, q) => (q >> 1 === d >> 1 ? b[d] : x)) });
    } else {
      const xb = v.xb;
      for (let d = 0; d < 6; d += 1) {
        const axis = d >> 1;
        if (Math.abs(xb[2 * axis] - xb[2 * axis + 1]) > 1e-9) continue;
        const at = xb[2 * axis];
        // The vent's side: the one whose outside is not in a mesh.
        const c = [(xb[0] + xb[1]) / 2, (xb[2] + xb[3]) / 2, (xb[4] + xb[5]) / 2];
        const probe = c.slice();
        probe[axis] = at + (d & 1 ? 1 : -1) * 1e-3;
        const back = c.slice();
        back[axis] = at - (d & 1 ? 1 : -1) * 1e-3;
        if (!inside(...probe) && inside(...back)) faces.push({ d, xb });
      }
    }
    for (const { d, xb } of faces) {
      const axis = d >> 1;
      if (axis === 2 && d === 4) continue; // the floor stays closed
      const cut = xb.slice();
      cut[2 * axis] = d & 1 ? xb[2 * axis] : domain[2 * axis];
      cut[2 * axis + 1] = d & 1 ? domain[2 * axis + 1] : xb[2 * axis + 1];
      openings.push(cut);
      outside = subtractAll(outside, cut);
    }
  }
  const obstacles = [];
  for (const o of geo.obsts) {
    if (volume(o.xb) <= 1e-12) continue;
    if (only && !only(o)) continue;
    let pieces = [o.xb];
    for (const hole of geo.holes) pieces = subtractAll(pieces, hole);
    obstacles.push(...pieces);
  }
  return { domain, boxes: [...outside, ...obstacles], openings, bbox: bb };
}

/**
 * The solver's grid over a geometry at cell h, its volume fractions and
 * solid cells by the app's own voxelisation (voxelize.js volumeFractions
 * and occupancy: a wall thinner than a cell still seals, as in the app).
 * Returns {grid, fraction, solid, domain, bbox, toCell, centre}.
 */
export function gridFor(geo, h, Vx = V) {
  const { domain, boxes, bbox, openings } = solidBoxes(geo, h);
  const n = [Math.ceil((domain[1] - domain[0]) / h - 1e-6), Math.ceil((domain[5] - domain[4]) / h - 1e-6), Math.ceil((domain[3] - domain[2]) / h - 1e-6)];
  const origin = [domain[0], domain[4], domain[2]];
  const grid = { nx: n[0], ny: n[1], nz: n[2], h, origin, axes: [0, 1, 2], upAxis: "y", cells: n[0] * n[1] * n[2] };
  const fraction = Vx.volumeFractions(grid, boxes.map(meshOf), { rays: Vx.raysFor(grid) });
  const solid = Vx.occupancy(grid, fraction);
  const at = (i, j, k) => i + grid.nx * (j + grid.ny * k);
  /** The cell (i, j, k) holding an FDS point (x, y, z), clamped to the grid. */
  const toCell = (x, y, z) => [
    Math.min(grid.nx - 1, Math.max(0, Math.floor((x - origin[0]) / h))),
    Math.min(grid.ny - 1, Math.max(0, Math.floor((z - origin[1]) / h))),
    Math.min(grid.nz - 1, Math.max(0, Math.floor((y - origin[2]) / h))),
  ];
  /** A cell's centre in FDS coordinates. */
  const centre = (i, j, k) => [origin[0] + (i + 0.5) * h, origin[2] + (k + 0.5) * h, origin[1] + (j + 0.5) * h];
  return { grid, fraction, solid, domain, bbox, openings, at, toCell, centre, boxes: boxes.length };
}

/**
 * The open cells of the column over FDS plan point (x, y) between the floor
 * under it and the ceiling over it, from the first open cell at or above z0:
 * [{z (cell centre), c}] bottom up, and the column's open top (the ceiling's
 * underside on this grid). With `z0` null the column starts at the room's
 * floor under the point.
 */
export function column(G, x, y, z0 = null) {
  const { grid, solid, at, toCell } = G;
  const [i, , k] = toCell(x, y, 0);
  let j = z0 == null ? toCell(x, y, G.bbox[4] + 1e-6)[1] : toCell(x, y, z0)[1];
  while (j < grid.ny && solid[at(i, j, k)]) j += 1;
  const out = [];
  for (; j < grid.ny && !solid[at(i, j, k)]; j += 1) out.push({ z: grid.origin[1] + (j + 0.5) * grid.h, c: at(i, j, k), j });
  const top = out.length ? out[out.length - 1].z + grid.h / 2 : null;
  const floor = out.length ? out[0].z - grid.h / 2 : null;
  return { i, k, cells: out, top, floor };
}

/**
 * A thermocouple tree's cells: every open cell of the column over FDS plan
 * point (x, y) from its floor (zBottom; the domain's floor by default) up
 * whose centre lies below zTop (the test's nominal ceiling), skipping solid cells (a bracket or beam sealed into whole
 * cells interrupts the string, as an obstruction would a real tree, but does
 * not end it). [{z (cell centre), c, j}] bottom up, with i and k.
 */
export function treeCells(G, x, y, zTop, zBottom = null) {
  const { grid, solid, at, toCell } = G;
  const [i, j0, k] = toCell(x, y, (zBottom ?? G.bbox[4]) + 1e-6);
  const cells = [];
  for (let j = j0; j < grid.ny; j += 1) {
    const z = grid.origin[1] + (j + 0.5) * grid.h;
    if (z >= zTop) break;
    if (!solid[at(i, j, k)]) cells.push({ z, c: at(i, j, k), j });
  }
  return { i, k, cells };
}

/**
 * The open cell for a point device at FDS (x, y, z): its own cell, or, when
 * that is solid (a ceiling snapped down past it) or fails `ok` (not in the
 * fire's connected gas), the nearest one that is open and passes, within
 * four cells. Returns {c, moved (m), at: [x, y, z]} or null.
 */
export function deviceCell(G, x, y, z, ok = () => true) {
  const { grid, solid, at, toCell, centre } = G;
  const [i0, j0, k0] = toCell(x, y, z);
  let best = null;
  for (let r = 0; r <= 4 && !best; r += 1) {
    for (let dj = -r; dj <= r; dj += 1) {
      for (let di = -r; di <= r; di += 1) {
        for (let dk = -r; dk <= r; dk += 1) {
          if (Math.max(Math.abs(di), Math.abs(dj), Math.abs(dk)) !== r) continue;
          const i = i0 + di, j = j0 + dj, k = k0 + dk;
          if (i < 0 || j < 0 || k < 0 || i >= grid.nx || j >= grid.ny || k >= grid.nz) continue;
          if (solid[at(i, j, k)] || !ok(at(i, j, k))) continue;
          const p = centre(i, j, k);
          const d = Math.hypot(p[0] - x, p[1] - y, p[2] - z);
          if (!best || d < best.d) best = { c: at(i, j, k), d, p, i, j, k };
        }
      }
    }
  }
  return best ? { c: best.c, i: best.i, j: best.j, k: best.k, moved: +best.d.toFixed(3), at: best.p.map((v) => +v.toFixed(3)) } : null;
}
