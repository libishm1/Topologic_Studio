/**
 * Floor-plan drawings to a draft LOD1 model, in the browser, by rules: no
 * server and no AI. One DXF sheet per storey becomes rooms (footprint
 * polygons in metres, named from the texts inside them), the doors that join
 * them, and, with the storeys' levels, the LOD1 model of lod1.js.
 *
 * The method follows the CM-iTAD lab's LOD1 pipeline (notebook
 * LOD1-00-DraftingRooms, lod1/drafting.py: walls and closed doors enclose
 * spaces, one room per space, grown to the wall centre-lines, named from the
 * texts inside), rebuilt as a planar arrangement instead of polygon buffers:
 *
 *   decodeDxfBytes a file's bytes as text (UTF-8, or the header's code page);
 *   readDxf        the sheet's entities, blocks exploded with their transforms
 *                  applied, and the declared units (an own reader: see
 *                  wiki/research/dxf-floor-plans-to-lod1-2026-11-05.md);
 *   suggestLayers  likely wall, door, window and text layers, door blocks and
 *                  the drawing scale, from layer names and geometry;
 *   draftRooms     walls snapped and joined, door openings bridged, the
 *                  arrangement's faces, rooms named and split by their texts,
 *                  grown to the wall centre-lines, doors given their rooms;
 *   draftSummary   compact JSON for an optional AI suggester (no geometry
 *                  beyond centroids);
 *   toLod1Model    per-storey drafts and levels to the model of lod1.js.
 *
 * Everything here is a pure function of its arguments (no DOM, no globals),
 * so it runs in a Web Worker. Hot paths use a uniform spatial hash: noding,
 * snapping, ray casts and point location are near linear in the number of
 * segments.
 *
 * Coordinates: readDxf returns drawing units; draftRooms works in metres,
 * (p - origin) * scale, in the drawing's own axes (x right, y up), which are
 * the IFC plan axes of lod1.js.
 */

import { classFromName as lod1ClassFromName, doorOnBoundary, isWalkableClass } from "./lod1.js";

// ===================================================================== reading DXF

/**
 * A DXF file's bytes as text: UTF-8 from AutoCAD 2007 (AC1021) on, else the
 * code page its header names ($DWGCODEPAGE, e.g. ANSI_1256 for Arabic),
 * through TextDecoder (browsers, workers and Node all have it).
 *
 * @param {ArrayBuffer|Uint8Array} bytes
 */
export function decodeDxfBytes(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const head = new TextDecoder("latin1").decode(u8.subarray(0, Math.min(u8.length, 1 << 16)));
  const value = (name) => head.match(new RegExp(`\\$${name}\\s*\\r?\\n\\s*\\d+\\s*\\r?\\n([^\\r\\n]*)`))?.[1]?.trim() || null;
  const version = value("ACADVER");
  if (!version || version >= "AC1021") return new TextDecoder("utf-8").decode(u8);
  const page = /^ANSI_(\d+)$/i.exec(value("DWGCODEPAGE") || "")?.[1];
  try {
    return new TextDecoder(page && page !== "1252" ? `windows-${page}` : "windows-1252").decode(u8);
  } catch {
    return new TextDecoder("windows-1252").decode(u8);
  }
}

/** Metres per drawing unit for each $INSUNITS code (0: unitless). */
const INSUNITS = {
  1: ["in", 0.0254], 2: ["ft", 0.3048], 3: ["mi", 1609.344], 4: ["mm", 0.001], 5: ["cm", 0.01], 6: ["m", 1],
  7: ["km", 1000], 8: ["µin", 2.54e-8], 9: ["mil", 2.54e-5], 10: ["yd", 0.9144], 14: ["dm", 0.1], 15: ["dam", 10], 16: ["hm", 100],
};

/** Affine 2D transforms [a, b, c, d, e, f]: x' = a x + c y + e, y' = b x + d y + f. */
const IDENTITY = [1, 0, 0, 1, 0, 0];
const compose = (m, n) => [
  m[0] * n[0] + m[2] * n[1], m[1] * n[0] + m[3] * n[1],
  m[0] * n[2] + m[2] * n[3], m[1] * n[2] + m[3] * n[3],
  m[0] * n[4] + m[2] * n[5] + m[4], m[1] * n[4] + m[3] * n[5] + m[5],
];
const apply = (m, x, y) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
const det = (m) => m[0] * m[3] - m[1] * m[2];
/** Whether a transform keeps circles circles (rotation, uniform scale, mirror). */
const conformal = (m) => Math.abs(m[0] * m[0] + m[1] * m[1] - m[2] * m[2] - m[3] * m[3]) < 1e-9 * (1 + Math.abs(det(m))) && Math.abs(m[0] * m[2] + m[1] * m[3]) < 1e-9 * (1 + Math.abs(det(m)));

/**
 * The object coordinate system of planar entities (ARC, CIRCLE, LWPOLYLINE,
 * 2D POLYLINE, TEXT, INSERT): with extrusion (0, 0, -1), the usual result of
 * mirroring in CAD, OCS x runs along world -x. Other tilts are projected.
 */
function ocs(raw) {
  const z = num(raw, 230, 1);
  return z < 0 ? [-1, 0, 0, 1, 0, 0] : IDENTITY;
}

/** The first value of a group code in a raw entity, as a number. */
function num(raw, code, fallback = 0) {
  const i = raw.c.indexOf(code);
  if (i < 0) return fallback;
  const v = parseFloat(raw.v[i]);
  return Number.isFinite(v) ? v : fallback;
}
/** The first value of a group code, as a string. */
function str(raw, code, fallback = "") {
  const i = raw.c.indexOf(code);
  return i < 0 ? fallback : raw.v[i];
}

/** \U+XXXX escapes and %% codes of DXF strings. */
function dxfString(s) {
  if (!s) return "";
  return s
    .replace(/\\U\+([0-9a-fA-F]{4})/g, (_, h) => String.fromCharCode(parseInt(h, 16)))
    .replace(/%%[cC]/g, "⌀").replace(/%%[dD]/g, "°").replace(/%%[pP]/g, "±").replace(/%%[uUoOkK]/g, "").replace(/%%%/g, "%");
}

/** MTEXT without its inline formatting (fonts, heights, colours, stacking, paragraphs). */
export function mtextPlain(s) {
  // Escaped braces are kept through the removal of grouping braces (private-use placeholders).
  const t = dxfString(s)
    .replace(/\\P/g, "\n").replace(/\\~/g, " ")
    .replace(/\\S([^;]*?)[\^/#]([^;]*?);/g, "$1/$2")
    .replace(/\\[fFcCHhTtQqWwAap][^;\\{}]*;/g, "")
    .replace(/\\[LlOoKkNX]/g, "")
    .replace(/\\\\/g, "\\").replace(/\\\{/g, "").replace(/\\\}/g, "")
    .replace(/[{}]/g, "")
    .replace(//g, "{").replace(//g, "}");
  return t.replace(/[ \t]+/g, " ").trim();
}

/** Points of an arc from a0 to a1 (radians, anticlockwise), at most `step` apart in angle. */
function arcPoints(cx, cy, r, a0, a1, step) {
  let sweep = a1 - a0;
  while (sweep <= 0) sweep += 2 * Math.PI;
  while (sweep > 2 * Math.PI + 1e-12) sweep -= 2 * Math.PI;
  const n = Math.max(2, Math.ceil(sweep / step));
  const out = new Array(2 * (n + 1));
  for (let k = 0; k <= n; k += 1) {
    const a = a0 + (sweep * k) / n;
    out[2 * k] = cx + r * Math.cos(a);
    out[2 * k + 1] = cy + r * Math.sin(a);
  }
  return out;
}

/** A polyline with bulges (LWPOLYLINE, POLYLINE) as points: each bulge an arc. */
function bulgePoints(xs, ys, bulges, closed, step) {
  const out = [];
  const n = xs.length;
  for (let i = 0; i < n; i += 1) {
    out.push(xs[i], ys[i]);
    const j = i + 1 < n ? i + 1 : closed ? 0 : -1;
    const b = bulges[i] || 0;
    if (j < 0 || Math.abs(b) < 1e-9) continue;
    // Arc from vertex i to j: included angle 4 atan(b), anticlockwise when b > 0.
    const x0 = xs[i], y0 = ys[i], x1 = xs[j], y1 = ys[j];
    const chord = Math.hypot(x1 - x0, y1 - y0);
    if (chord < 1e-12) continue;
    const theta = 4 * Math.atan(b);
    const r = chord / (2 * Math.sin(Math.abs(theta) / 2));
    const mx = (x0 + x1) / 2, my = (y0 + y1) / 2;
    const h = r * Math.cos(Math.abs(theta) / 2) * Math.sign(b);
    const cx = mx - ((y1 - y0) / chord) * h, cy = my + ((x1 - x0) / chord) * h;
    const a0 = Math.atan2(y0 - cy, x0 - cx);
    const steps = Math.max(1, Math.ceil(Math.abs(theta) / step));
    for (let k = 1; k < steps; k += 1) {
      const a = a0 + (theta * k) / steps;
      out.push(cx + Math.abs(r) * Math.cos(a), cy + Math.abs(r) * Math.sin(a));
    }
  }
  if (closed && n > 1) out.push(xs[0], ys[0]);
  return out;
}

/** A B-spline's points by de Boor's algorithm (weights ignored), or its fit points. */
function splinePoints(raw) {
  const deg = num(raw, 71, 3);
  const knots = [], ctrl = [], fit = [];
  for (let i = 0; i < raw.c.length; i += 1) {
    const c = raw.c[i];
    if (c === 40) knots.push(parseFloat(raw.v[i]));
    else if (c === 10) ctrl.push([parseFloat(raw.v[i]), 0]);
    else if (c === 20 && ctrl.length) ctrl[ctrl.length - 1][1] = parseFloat(raw.v[i]);
    else if (c === 11) fit.push([parseFloat(raw.v[i]), 0]);
    else if (c === 21 && fit.length) fit[fit.length - 1][1] = parseFloat(raw.v[i]);
  }
  if (ctrl.length < 2 || knots.length !== ctrl.length + deg + 1) return (fit.length >= 2 ? fit : ctrl).flat();
  const out = [];
  const t0 = knots[deg], t1 = knots[ctrl.length];
  const samples = Math.min(200, ctrl.length * 8);
  for (let s = 0; s <= samples; s += 1) {
    const t = Math.min(t0 + ((t1 - t0) * s) / samples, t1 - 1e-12 * Math.abs(t1 - t0));
    let k = deg;
    while (k < ctrl.length - 1 && knots[k + 1] <= t) k += 1;
    const d = [];
    for (let j = 0; j <= deg; j += 1) d.push(ctrl[j + k - deg].slice());
    for (let r = 1; r <= deg; r += 1) {
      for (let j = deg; j >= r; j -= 1) {
        const i = j + k - deg;
        const den = knots[i + deg - r + 1] - knots[i];
        const a = den > 0 ? (t - knots[i]) / den : 0;
        d[j][0] = (1 - a) * d[j - 1][0] + a * d[j][0];
        d[j][1] = (1 - a) * d[j - 1][1] + a * d[j][1];
      }
    }
    out.push(d[deg][0], d[deg][1]);
  }
  return out;
}

/** The text's approximate width: 0.6 of its height per character, per line. */
const textWidth = (text, h, factor = 1) => 0.6 * h * factor * Math.max(1, ...text.split("\n").map((l) => l.length));

/**
 * Raw DXF records: group-code pairs split into sections, tables, blocks and
 * entities. Each record is {type, c: codes, v: values}; a POLYLINE carries
 * its VERTEX records, an INSERT its ATTRIBs.
 */
function dxfRecords(text) {
  const lines = text.split(/\r\n|\n|\r/);
  const header = {};
  const layers = [];
  const blocks = new Map();
  const entities = [];
  let section = null, current = null, list = null, block = null, headerVar = null;
  let parent = null; // the POLYLINE or INSERT whose VERTEX / ATTRIB records follow
  const close = () => {
    if (!current) return;
    const rec = current;
    current = null;
    if (section === "TABLES") {
      if (rec.type === "LAYER") {
        const color = num(rec, 62, 7), flags = num(rec, 70, 0);
        layers.push({ name: str(rec, 2), color: Math.abs(color), off: color < 0, frozen: (flags & 1) === 1, linetype: str(rec, 6, "CONTINUOUS") });
      }
      return;
    }
    if (section !== "ENTITIES" && section !== "BLOCKS") return;
    if (rec.type === "BLOCK") {
      block = { name: str(rec, 2), base: [num(rec, 10), num(rec, 20)], flags: num(rec, 70, 0), entities: [] };
      list = block.entities;
      return;
    }
    if (rec.type === "ENDBLK") {
      if (block) blocks.set(block.name, block);
      block = null;
      list = null;
      return;
    }
    const target = section === "ENTITIES" ? entities : list;
    if (!target) return;
    if (rec.type === "VERTEX" && parent?.type === "POLYLINE") return parent.vertices.push(rec);
    if (rec.type === "ATTRIB" && parent?.type === "INSERT") return parent.attribs.push(rec);
    if (rec.type === "SEQEND") {
      parent = null;
      return;
    }
    parent = null;
    if (rec.type === "POLYLINE") {
      rec.vertices = [];
      parent = rec;
    } else if (rec.type === "INSERT") {
      rec.attribs = [];
      if (num(rec, 66, 0) === 1) parent = rec;
    }
    target.push(rec);
  };
  for (let i = 0; i + 1 < lines.length; i += 2) {
    const code = parseInt(lines[i], 10);
    const value = lines[i + 1].trim();
    if (code === 0) {
      close();
      if (value === "SECTION") {
        section = "?";
        continue;
      }
      if (value === "ENDSEC") {
        section = null;
        continue;
      }
      if (value === "EOF") break;
      if (section && section !== "HEADER") current = { type: value, c: [], v: [] };
      continue;
    }
    if (section === "?" && code === 2) {
      section = value;
      continue;
    }
    if (section === "HEADER") {
      if (code === 9) header[(headerVar = value)] = [];
      else if (headerVar) header[headerVar].push([code, value]);
      continue;
    }
    if (current) {
      current.c.push(code);
      current.v.push(value);
    }
  }
  close();
  return { header, layers, blocks, entities };
}

/**
 * A DXF sheet's entities, with every block exploded (nested blocks too) and
 * its transform applied, in drawing units.
 *
 * Each entity is one of:
 *   {type: "polyline", kind, layer, path, owner, points: [x0, y0, x1, y1, ...], closed}
 *      LINE, LWPOLYLINE and POLYLINE (bulges as arcs), ELLIPSE, SPLINE and
 *      MLINE (its reference line), and arcs a non-uniform scale distorts;
 *   {type: "arc", kind, layer, path, owner, cx, cy, r, a0, a1, points}
 *      ARC and CIRCLE: centre, radius, start and end angle (radians,
 *      anticlockwise), with `points` the arc flattened;
 *   {type: "text", kind, layer, path, owner, text, x, y, h, rotation, tag}
 *      TEXT, MTEXT (formatting removed) and ATTRIB; (x, y) is the text's
 *      estimated centre, which is where a room name points;
 *   {type: "insert", kind: "INSERT", layer, path, owner, name, x, y, rotation, sx, sy, bbox}
 *      every block reference at any depth; bbox [x0, y0, x1, y1] covers its
 *      exploded geometry, and its parts carry its index as `owner`.
 *
 * `layer` is the layer AutoCAD shows (a part on layer 0 takes its insert's),
 * `linetype` its linetype (BYLAYER and BYBLOCK resolved);
 * `path` is the lab's key: the layer, or "A>B" for a part on layer B inside a
 * block inserted on layer A. `owner` is the index of the innermost insert
 * holding the entity, -1 for none. Paper-space entities, dimensions, hatches
 * and images are left out.
 *
 * @param {string} text  the DXF file's text (ASCII DXF; R12 to 2018)
 * @param {{arcStep?: number, maxDepth?: number}} opts  arcStep: the largest
 *   angle between two points of a flattened arc (radians)
 * @returns {{entities: object[], units: {code: number, name: string|null, toMetres: number|null},
 *   layers: object[], blocks: number, extents: number[]|null, version: string|null, warnings: string[]}}
 */
export function readDxf(text, { arcStep = Math.PI / 24, maxDepth = 12 } = {}) {
  if (typeof text !== "string" || !/^\s*0\s*\r?\n\s*SECTION/.test(text.slice(0, 200))) {
    throw new Error("Not an ASCII DXF file (a binary DXF or a DWG must be saved as ASCII DXF first).");
  }
  const { header, layers, blocks, entities: raws } = dxfRecords(text);
  const warnings = [];
  const out = [];
  const code = Number(header.$INSUNITS?.[0]?.[1] ?? 0) || 0;
  const unit = INSUNITS[code] || null;
  let missing = 0, cycles = 0;

  const layerType = new Map(layers.map((l) => [l.name, l.linetype.toUpperCase()]));
  /** Layer, lab path and linetype (BYLAYER resolved through the layer, BYBLOCK through the insert). */
  const layerOf = (raw, ctx) => {
    const own = str(raw, 8, "0");
    const lt = str(raw, 6, "BYLAYER").toUpperCase();
    if (!ctx) return { layer: own, path: own, linetype: lt === "BYLAYER" || lt === "BYBLOCK" ? layerType.get(own) || "CONTINUOUS" : lt };
    const layer = own === "0" ? ctx.layer : own;
    const path = own === "0" || own === ctx.own ? ctx.path : `${ctx.path}>${own}`;
    const linetype = lt === "BYBLOCK" ? ctx.linetype : lt === "BYLAYER" ? layerType.get(own === "0" ? ctx.layer : own) || "CONTINUOUS" : lt;
    return { layer, path, linetype };
  };
  const push = (e) => {
    out.push(e);
    return out.length - 1;
  };
  const polyline = (kind, raw, ctx, m, pts, closed) => {
    if (pts.length < 4) return;
    const p = new Array(pts.length);
    for (let i = 0; i < pts.length; i += 2) {
      p[i] = m[0] * pts[i] + m[2] * pts[i + 1] + m[4];
      p[i + 1] = m[1] * pts[i] + m[3] * pts[i + 1] + m[5];
    }
    push({ type: "polyline", kind, ...layerOf(raw, ctx), owner: ctx?.owner ?? -1, points: p, closed });
  };
  const arc = (kind, raw, ctx, m, cx, cy, r, a0, a1) => {
    if (!(r > 0)) return;
    if (!conformal(m)) return polyline(kind, raw, ctx, m, arcPoints(cx, cy, r, a0, a1, arcStep), kind === "CIRCLE");
    const [x, y] = apply(m, cx, cy);
    const s = Math.sqrt(Math.abs(det(m)));
    const [sx, sy] = apply(m, cx + r * Math.cos(a0), cy + r * Math.sin(a0));
    const [ex, ey] = apply(m, cx + r * Math.cos(a1), cy + r * Math.sin(a1));
    let b0 = Math.atan2(sy - y, sx - x), b1 = Math.atan2(ey - y, ex - x);
    if (kind === "CIRCLE") [b0, b1] = [0, 2 * Math.PI];
    else if (det(m) < 0) [b0, b1] = [b1, b0];
    push({ type: "arc", kind, ...layerOf(raw, ctx), owner: ctx?.owner ?? -1, cx: x, cy: y, r: r * s, a0: b0, a1: b1, points: arcPoints(x, y, r * s, b0, b1, arcStep) });
  };
  const textEntity = (kind, raw, ctx, m, value, x, y, h, rot, tag = null) => {
    const t = value.trim();
    if (!t) return;
    const [wx, wy] = apply(m, x, y);
    const [ux, uy] = [m[0] * Math.cos(rot) + m[2] * Math.sin(rot), m[1] * Math.cos(rot) + m[3] * Math.sin(rot)];
    push({ type: "text", kind, ...layerOf(raw, ctx), owner: ctx?.owner ?? -1, text: t, x: wx, y: wy, h: h * Math.sqrt(Math.abs(det(m))), rotation: Math.atan2(uy, ux), tag });
  };
  /** TEXT and ATTRIB: the centre of the text from its alignment (72; 73 or 74). */
  const plainText = (kind, raw, ctx, m) => {
    if (kind === "ATTRIB" && (num(raw, 70, 0) & 1)) return; // invisible
    const value = dxfString(str(raw, 1));
    const h = num(raw, 40, 1), rot = (num(raw, 50, 0) * Math.PI) / 180, wf = num(raw, 41, 1) || 1;
    const ha = num(raw, 72, 0), va = num(raw, kind === "ATTRIB" ? 74 : 73, 0);
    let x = num(raw, 10), y = num(raw, 20);
    const aligned = ha !== 0 || va !== 0;
    if (aligned && raw.c.includes(11)) {
      const ax = num(raw, 11), ay = num(raw, 21);
      if (ha === 3 || ha === 5) [x, y] = [(x + ax) / 2, (y + ay) / 2];
      else [x, y] = [ax, ay];
    }
    const w = textWidth(value, h, wf);
    const fx = ha === 3 || ha === 5 ? 0.5 : [0, 0.5, 1, 0.5, 0.5][ha] ?? 0, fy = ha === 4 ? 0.5 : [0, 0, 0.5, 1][va] ?? 0;
    const dx = ha === 3 || ha === 5 ? 0 : (0.5 - fx) * w, dy = (0.5 - fy) * h;
    const c = Math.cos(rot), s = Math.sin(rot);
    textEntity(kind, raw, ctx, compose(m, ocs(raw)), value, x + dx * c - dy * s, y + dx * s + dy * c, h, rot, kind === "ATTRIB" ? str(raw, 2) : null);
  };
  const mtext = (raw, ctx, m) => {
    let value = "";
    for (let i = 0; i < raw.c.length; i += 1) if (raw.c[i] === 3) value += raw.v[i];
    value = mtextPlain(value + str(raw, 1));
    const h = num(raw, 40, 1), attach = num(raw, 71, 1), col = num(raw, 41, 0);
    let rot = num(raw, 50, 0);
    if (raw.c.includes(11)) rot = Math.atan2(num(raw, 21), num(raw, 11));
    const lines = value.split("\n");
    const est = textWidth(value, h);
    const w = col > 0 ? Math.min(col, est) : est, th = h * (1 + 0.67 * (lines.length - 1));
    const cx = (attach - 1) % 3, cy = Math.floor((attach - 1) / 3);
    const dx = (0.5 - cx / 2) * w, dy = (cy / 2 - 0.5) * th;
    const c = Math.cos(rot), s = Math.sin(rot);
    textEntity("MTEXT", raw, ctx, m, value.replace(/\n/g, " "), num(raw, 10) + dx * c - dy * s, num(raw, 20) + dx * s + dy * c, h, rot);
  };

  const emit = (raw, m, ctx, depth) => {
    if (num(raw, 67, 0) === 1) return; // paper space
    const t = raw.type;
    if (t === "LINE") {
      polyline("LINE", raw, ctx, m, [num(raw, 10), num(raw, 20), num(raw, 11), num(raw, 21)], false);
    } else if (t === "LWPOLYLINE") {
      const xs = [], ys = [], bs = [];
      for (let i = 0; i < raw.c.length; i += 1) {
        const c = raw.c[i];
        if (c === 10) {
          xs.push(parseFloat(raw.v[i]));
          bs.push(0);
        } else if (c === 20) ys.push(parseFloat(raw.v[i]));
        else if (c === 42 && bs.length) bs[bs.length - 1] = parseFloat(raw.v[i]);
      }
      const closed = (num(raw, 70, 0) & 1) === 1;
      polyline("LWPOLYLINE", raw, ctx, compose(m, ocs(raw)), bulgePoints(xs, ys, bs, closed, arcStep), closed);
    } else if (t === "POLYLINE") {
      const flags = num(raw, 70, 0);
      if (flags & (16 | 64)) return; // meshes
      const xs = [], ys = [], bs = [];
      for (const v of raw.vertices) {
        if (num(v, 70, 0) & 16) continue; // spline frame control points
        xs.push(num(v, 10));
        ys.push(num(v, 20));
        bs.push(num(v, 42, 0));
      }
      const closed = (flags & 1) === 1;
      polyline("POLYLINE", raw, ctx, flags & 8 ? m : compose(m, ocs(raw)), bulgePoints(xs, ys, bs, closed, arcStep), closed);
    } else if (t === "ARC" || t === "CIRCLE") {
      const mo = compose(m, ocs(raw));
      const a0 = t === "ARC" ? (num(raw, 50) * Math.PI) / 180 : 0, a1 = t === "ARC" ? (num(raw, 51) * Math.PI) / 180 : 2 * Math.PI;
      arc(t, raw, ctx, mo, num(raw, 10), num(raw, 20), num(raw, 40), a0, a1);
    } else if (t === "ELLIPSE") {
      const cx = num(raw, 10), cy = num(raw, 20), mx = num(raw, 11), my = num(raw, 21), ratio = num(raw, 40, 1);
      let p0 = num(raw, 41, 0), p1 = num(raw, 42, 2 * Math.PI);
      if (num(raw, 230, 1) < 0) [p0, p1] = [-p1, -p0];
      while (p1 <= p0) p1 += 2 * Math.PI;
      const n = Math.max(4, Math.ceil((p1 - p0) / arcStep));
      const pts = [];
      for (let k = 0; k <= n; k += 1) {
        const a = p0 + ((p1 - p0) * k) / n;
        pts.push(cx + mx * Math.cos(a) - my * ratio * Math.sin(a), cy + my * Math.cos(a) + mx * ratio * Math.sin(a));
      }
      polyline("ELLIPSE", raw, ctx, m, pts, Math.abs(p1 - p0 - 2 * Math.PI) < 1e-6);
    } else if (t === "SPLINE") {
      polyline("SPLINE", raw, ctx, m, splinePoints(raw), (num(raw, 70, 0) & 1) === 1);
    } else if (t === "MLINE") {
      const pts = [];
      for (let i = 0; i < raw.c.length; i += 1) {
        if (raw.c[i] === 11) pts.push(parseFloat(raw.v[i]), 0);
        else if (raw.c[i] === 21 && pts.length) pts[pts.length - 1] = parseFloat(raw.v[i]);
      }
      polyline("MLINE", raw, ctx, m, pts, (num(raw, 71, 0) & 2) === 2);
    } else if (t === "TEXT" || t === "ATTRIB") {
      plainText(t, raw, ctx, m);
    } else if (t === "MTEXT") {
      mtext(raw, ctx, m);
    } else if (t === "INSERT") {
      const name = str(raw, 2);
      const def = blocks.get(name);
      if (!def) {
        missing += 1;
        return;
      }
      if (depth >= maxDepth || ctx?.chain?.includes(name)) {
        cycles += 1;
        return;
      }
      const sx = num(raw, 41, 1), sy = num(raw, 42, 1), rot = (num(raw, 50, 0) * Math.PI) / 180;
      const cols = Math.max(1, num(raw, 70, 1)), rows = Math.max(1, num(raw, 71, 1)), dc = num(raw, 44, 0), dr = num(raw, 45, 0);
      const { layer, path, linetype } = layerOf(raw, ctx);
      const own = str(raw, 8, "0");
      const mo = compose(m, ocs(raw));
      for (let r = 0; r < rows; r += 1) {
        for (let k = 0; k < cols; k += 1) {
          const c = Math.cos(rot), s = Math.sin(rot);
          const ox = num(raw, 10) + c * k * dc - s * r * dr, oy = num(raw, 20) + s * k * dc + c * r * dr;
          // insert point . rotation . scale . (- base point)
          const local = [c * sx, s * sx, -s * sy, c * sy, ox - (c * sx * def.base[0] - s * sy * def.base[1]), oy - (s * sx * def.base[0] + c * sy * def.base[1])];
          const mi = compose(mo, local);
          const [x, y] = apply(mo, ox, oy);
          const ux = mi[0], uy = mi[1];
          const id = push({ type: "insert", kind: "INSERT", name, layer, path, owner: ctx?.owner ?? -1, x, y, rotation: Math.atan2(uy, ux), sx, sy, bbox: null });
          const start = out.length;
          const inner = { layer, path, linetype, own: own === "0" && ctx ? ctx.own : own, owner: id, chain: [...(ctx?.chain || []), name] };
          for (const child of def.entities) emit(child, mi, inner, depth + 1);
          // The insert's box: its exploded geometry (texts by their centre).
          let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
          for (let j = start; j < out.length; j += 1) {
            const e = out[j];
            if (e.points) {
              for (let q = 0; q < e.points.length; q += 2) {
                const px = e.points[q], py = e.points[q + 1];
                if (px < x0) x0 = px;
                if (px > x1) x1 = px;
                if (py < y0) y0 = py;
                if (py > y1) y1 = py;
              }
            } else if (e.type === "text" || e.type === "insert") {
              const b = e.bbox || [e.x, e.y, e.x, e.y];
              x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]); x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
            }
          }
          out[id].bbox = x0 <= x1 ? [x0, y0, x1, y1] : [x, y, x, y];
        }
      }
      // Attributes are placed in the insert's parent frame already.
      for (const a of raw.attribs) plainText("ATTRIB", a, ctx ? { ...ctx } : null, m);
    }
  };
  for (const raw of raws) emit(raw, IDENTITY, null, 0);
  if (missing) warnings.push(`${missing} block references name blocks the file does not define; left out.`);
  if (cycles) warnings.push(`${cycles} block references nest too deep or refer to themselves; left out.`);

  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const e of out) {
    if (e.points) {
      for (let q = 0; q < e.points.length; q += 2) {
        if (e.points[q] < x0) x0 = e.points[q];
        if (e.points[q] > x1) x1 = e.points[q];
        if (e.points[q + 1] < y0) y0 = e.points[q + 1];
        if (e.points[q + 1] > y1) y1 = e.points[q + 1];
      }
    }
  }
  return {
    entities: out,
    units: { code, name: unit ? unit[0] : null, toMetres: unit ? unit[1] : null },
    layers,
    blocks: blocks.size,
    extents: x0 <= x1 ? [x0, y0, x1, y1] : null,
    version: header.$ACADVER?.[0]?.[1] ?? null,
    warnings,
  };
}

// ===================================================================== spatial hash and plan geometry

/**
 * A uniform grid over boxes (x0, y0, x1, y1 per item, in one Float64Array):
 * each cell lists the items whose box touches it, in CSR arrays. The cell
 * grows when the grid would pass `maxCells`.
 */
function boxGrid(boxes, n, cell, maxCells = 4e6) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i += 1) {
    if (boxes[4 * i] < x0) x0 = boxes[4 * i];
    if (boxes[4 * i + 1] < y0) y0 = boxes[4 * i + 1];
    if (boxes[4 * i + 2] > x1) x1 = boxes[4 * i + 2];
    if (boxes[4 * i + 3] > y1) y1 = boxes[4 * i + 3];
  }
  if (!(x0 <= x1)) [x0, y0, x1, y1] = [0, 0, 1, 1];
  while (((x1 - x0) / cell + 1) * ((y1 - y0) / cell + 1) > maxCells) cell *= 2;
  const nx = Math.floor((x1 - x0) / cell) + 1, ny = Math.floor((y1 - y0) / cell) + 1;
  const cx = (x) => Math.min(nx - 1, Math.max(0, Math.floor((x - x0) / cell)));
  const cy = (y) => Math.min(ny - 1, Math.max(0, Math.floor((y - y0) / cell)));
  const start = new Int32Array(nx * ny + 1);
  for (let i = 0; i < n; i += 1) {
    const i0 = cx(boxes[4 * i]), i1 = cx(boxes[4 * i + 2]), j0 = cy(boxes[4 * i + 1]), j1 = cy(boxes[4 * i + 3]);
    for (let j = j0; j <= j1; j += 1) for (let k = i0; k <= i1; k += 1) start[j * nx + k + 1] += 1;
  }
  for (let c = 0; c < nx * ny; c += 1) start[c + 1] += start[c];
  const items = new Int32Array(start[nx * ny]);
  const fill = start.slice();
  for (let i = 0; i < n; i += 1) {
    const i0 = cx(boxes[4 * i]), i1 = cx(boxes[4 * i + 2]), j0 = cy(boxes[4 * i + 1]), j1 = cy(boxes[4 * i + 3]);
    for (let j = j0; j <= j1; j += 1) for (let k = i0; k <= i1; k += 1) items[fill[j * nx + k]++] = i;
  }
  return { x0, y0, cell, nx, ny, start, items, cx, cy, stamp: new Int32Array(Math.max(1, n)), tick: 0 };
}

/** Calls fn(item) once for every item whose cells meet the box; fn returning false stops. */
function gridQuery(g, bx0, by0, bx1, by1, fn) {
  g.tick += 1;
  if (g.tick > 2e9) {
    g.stamp.fill(0);
    g.tick = 1;
  }
  const i0 = g.cx(bx0), i1 = g.cx(bx1), j0 = g.cy(by0), j1 = g.cy(by1);
  for (let j = j0; j <= j1; j += 1) {
    for (let k = i0; k <= i1; k += 1) {
      const c = j * g.nx + k;
      for (let q = g.start[c]; q < g.start[c + 1]; q += 1) {
        const it = g.items[q];
        if (g.stamp[it] === g.tick) continue;
        g.stamp[it] = g.tick;
        if (fn(it) === false) return;
      }
    }
  }
}

/** Boxes of segments [x0, y0, x1, y1, ...]. */
function segmentBoxes(s, n) {
  const b = new Float64Array(4 * n);
  for (let i = 0; i < n; i += 1) {
    b[4 * i] = Math.min(s[4 * i], s[4 * i + 2]);
    b[4 * i + 1] = Math.min(s[4 * i + 1], s[4 * i + 3]);
    b[4 * i + 2] = Math.max(s[4 * i], s[4 * i + 2]);
    b[4 * i + 3] = Math.max(s[4 * i + 1], s[4 * i + 3]);
  }
  return b;
}

/** Boxes of rings. */
function ringBoxes(rings) {
  const b = new Float64Array(4 * rings.length);
  rings.forEach((ring, i) => {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const [x, y] of ring) {
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
    b[4 * i] = x0;
    b[4 * i + 1] = y0;
    b[4 * i + 2] = x1;
    b[4 * i + 3] = y1;
  });
  return b;
}

/** Signed area of a ring of [x, y] (positive: anticlockwise). */
function areaOf(ring) {
  let a = 0;
  for (let i = 0, n = ring.length; i < n; i += 1) {
    const p = ring[i], q = ring[(i + 1) % n];
    a += p[0] * q[1] - q[0] * p[1];
  }
  return a / 2;
}
function perimeterOf(ring) {
  let s = 0;
  for (let i = 0; i < ring.length; i += 1) {
    const p = ring[i], q = ring[(i + 1) % ring.length];
    s += Math.hypot(q[0] - p[0], q[1] - p[1]);
  }
  return s;
}
/** Crossing parity of (x, y) in a ring. */
function inRing(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i, i += 1) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if (yi > y !== yj > y && x < xj + ((y - yj) * (xi - xj)) / (yi - yj)) inside = !inside;
  }
  return inside;
}
/** Area centroid of a ring. */
function centroidOf(ring) {
  let cx = 0, cy = 0, a = 0;
  for (let i = 0; i < ring.length; i += 1) {
    const p = ring[i], q = ring[(i + 1) % ring.length];
    const f = p[0] * q[1] - q[0] * p[1];
    cx += (p[0] + q[0]) * f;
    cy += (p[1] + q[1]) * f;
    a += f;
  }
  if (Math.abs(a) < 1e-12) return [ring.reduce((s, p) => s + p[0], 0) / ring.length, ring.reduce((s, p) => s + p[1], 0) / ring.length];
  return [cx / (3 * a), cy / (3 * a)];
}
/** A point inside a ring (and outside its holes): the centroid if it is, else the middle of the widest horizontal chord. */
function insidePoint(ring, holes = []) {
  const c = centroidOf(ring);
  const ok = (x, y) => inRing(x, y, ring) && !holes.some((h) => inRing(x, y, h));
  if (ok(c[0], c[1])) return c;
  let best = c, bw = -1;
  const ys = ring.map((p) => p[1]).sort((a, b) => a - b);
  for (let k = 1; k <= 9; k += 1) {
    const y = ys[0] + ((ys[ys.length - 1] - ys[0]) * k) / 10;
    const xs = [];
    for (const r of [ring, ...holes]) {
      for (let i = 0; i < r.length; i += 1) {
        const p = r[i], q = r[(i + 1) % r.length];
        if (p[1] > y !== q[1] > y) xs.push(p[0] + ((y - p[1]) * (q[0] - p[0])) / (q[1] - p[1]));
      }
    }
    xs.sort((a, b) => a - b);
    for (let i = 0; i + 1 < xs.length; i += 2) if (xs[i + 1] - xs[i] > bw) [bw, best] = [xs[i + 1] - xs[i], [(xs[i] + xs[i + 1]) / 2, y]];
  }
  return best;
}
/** Distance from (x, y) to segment a-b, and the parameter of the nearest point. */
function toSeg(x, y, ax, ay, bx, by) {
  const vx = bx - ax, vy = by - ay, l2 = vx * vx + vy * vy;
  const t = l2 > 0 ? Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / l2)) : 0;
  return [Math.hypot(x - ax - t * vx, y - ay - t * vy), t];
}
/** A ring without repeated and straight-through points (tolerance in metres). */
function cleanRing(ring, tol = 1e-3) {
  let pts = ring.filter((p, i) => {
    const q = ring[(i + 1) % ring.length];
    return Math.hypot(p[0] - q[0], p[1] - q[1]) > tol;
  });
  for (let changed = true; changed && pts.length > 3; ) {
    changed = false;
    for (let i = 0; i < pts.length && pts.length > 3; i += 1) {
      const a = pts[(i + pts.length - 1) % pts.length], b = pts[i], c = pts[(i + 1) % pts.length];
      if (toSeg(b[0], b[1], a[0], a[1], c[0], c[1])[0] <= tol) {
        pts.splice(i, 1);
        changed = true;
        i -= 1;
      }
    }
  }
  return pts;
}
/** Douglas-Peucker on a closed ring, anchored at the two points furthest apart. */
function simplifyClosed(ring, tol) {
  const n = ring.length;
  if (n <= 4 || tol <= 0) return ring;
  let a = 0, b = 0, best = -1;
  for (let i = 0; i < n; i += 1) {
    const d = Math.hypot(ring[i][0] - ring[0][0], ring[i][1] - ring[0][1]);
    if (d > best) [best, b] = [d, i];
  }
  best = -1;
  for (let i = 0; i < n; i += 1) {
    const d = Math.hypot(ring[i][0] - ring[b][0], ring[i][1] - ring[b][1]);
    if (d > best) [best, a] = [d, i];
  }
  const keep = new Uint8Array(n);
  const dp = (i, j) => {
    const stack = [[i, j]];
    while (stack.length) {
      const [s, e] = stack.pop();
      let far = -1, fd = tol;
      for (let k = (s + 1) % n; k !== e; k = (k + 1) % n) {
        const d = toSeg(ring[k][0], ring[k][1], ring[s][0], ring[s][1], ring[e][0], ring[e][1])[0];
        if (d > fd) [fd, far] = [d, k];
      }
      if (far >= 0) {
        keep[far] = 1;
        stack.push([s, far], [far, e]);
      }
    }
  };
  keep[a] = keep[b] = 1;
  dp(a, b);
  dp(b, a);
  const out = ring.filter((_, i) => keep[i]);
  return out.length >= 3 ? out : ring;
}
/** Whether two segments cross at a point inside both. */
function segmentsCross(ax, ay, bx, by, cx, cy, dx, dy) {
  const d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx);
  const d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx);
  const d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
  const d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax);
  return ((d1 > 1e-12 && d2 < -1e-12) || (d1 < -1e-12 && d2 > 1e-12)) && ((d3 > 1e-12 && d4 < -1e-12) || (d3 < -1e-12 && d4 > 1e-12));
}
/** Whether a ring crosses itself (edges that are not neighbours cross). */
function selfCrossing(ring) {
  const n = ring.length;
  if (n > 600) return false;
  for (let i = 0; i < n; i += 1) {
    const a = ring[i], b = ring[(i + 1) % n];
    for (let j = i + 2; j < n; j += 1) {
      if (i === 0 && j === n - 1) continue;
      const c = ring[j], d = ring[(j + 1) % n];
      if (segmentsCross(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1])) return true;
    }
  }
  return false;
}

/**
 * The first crossing of a ray from (ox, oy) along unit (dx, dy) with the
 * grid's segments, within maxT: its distance, or Infinity.
 */
function rayHit(grid, segs, ox, oy, dx, dy, maxT, skip = -1) {
  let best = maxT, hit = Infinity;
  const ex = ox + dx * maxT, ey = oy + dy * maxT;
  gridQuery(grid, Math.min(ox, ex), Math.min(oy, ey), Math.max(ox, ex), Math.max(oy, ey), (i) => {
    if (i === skip) return;
    const ax = segs[4 * i], ay = segs[4 * i + 1], sx = segs[4 * i + 2] - ax, sy = segs[4 * i + 3] - ay;
    const den = dx * sy - dy * sx;
    if (Math.abs(den) < 1e-12) return;
    const t = ((ax - ox) * sy - (ay - oy) * sx) / den;
    const u = ((ax - ox) * dy - (ay - oy) * dx) / den;
    if (t > 1e-9 && t < best && u >= -1e-9 && u <= 1 + 1e-9) {
      best = t;
      hit = t;
    }
  });
  return hit;
}

// ===================================================================== the planar arrangement

/**
 * Wall segments to a planar graph: endpoints within `snap` merged; a
 * dangling end moved onto a segment within `snap` or, failing that, carried
 * along its own line up to `extend` to the first segment it meets (lines
 * drawn short of a wall); then every crossing, touch and collinear overlap
 * noded. Returns the vertices and the unique edges.
 */
function arrangement(segIn, nIn, { snap, extend }) {
  const segs = Float64Array.from(segIn.subarray(0, 4 * nIn));
  // 1. Endpoints within snap become one point (the first seen).
  if (snap > 0) {
    const reps = new Map();
    for (let p = 0; p < 2 * nIn; p += 1) {
      const x = segs[2 * p], y = segs[2 * p + 1];
      const ix = Math.floor(x / snap), iy = Math.floor(y / snap);
      let best = null, bd = snap;
      for (let dx = -1; dx <= 1; dx += 1) {
        for (let dy = -1; dy <= 1; dy += 1) {
          for (const q of reps.get(`${ix + dx},${iy + dy}`) || []) {
            const d = Math.hypot(q[0] - x, q[1] - y);
            if (d <= bd) [best, bd] = [q, d];
          }
        }
      }
      if (best) {
        segs[2 * p] = best[0];
        segs[2 * p + 1] = best[1];
      } else {
        const k = `${ix},${iy}`;
        if (!reps.has(k)) reps.set(k, []);
        reps.get(k).push([x, y]);
      }
    }
  }
  let n = 0;
  for (let i = 0; i < nIn; i += 1) {
    if (segs[4 * i] === segs[4 * i + 2] && segs[4 * i + 1] === segs[4 * i + 3]) continue;
    for (let k = 0; k < 4; k += 1) segs[4 * n + k] = segs[4 * i + k];
    n += 1;
  }
  // 2. Dangling ends: onto a segment within snap, else along their own line.
  const degree = new Map();
  const pk = (x, y) => `${x},${y}`;
  for (let p = 0; p < 2 * n; p += 1) {
    const k = pk(segs[2 * p], segs[2 * p + 1]);
    degree.set(k, (degree.get(k) || 0) + 1);
  }
  let grid = boxGrid(segmentBoxes(segs, n), n, 1.0);
  let moved = 0;
  for (let p = 0; p < 2 * n; p += 1) {
    const x = segs[2 * p], y = segs[2 * p + 1];
    if (degree.get(pk(x, y)) !== 1) continue;
    const s = p >> 1, other = p ^ 1;
    const ox = segs[2 * other], oy = segs[2 * other + 1];
    const len = Math.hypot(x - ox, y - oy);
    if (len < 1e-9) continue;
    let best = null, bd = snap;
    gridQuery(grid, x - snap, y - snap, x + snap, y + snap, (i) => {
      if (i === s) return;
      const [d, t] = toSeg(x, y, segs[4 * i], segs[4 * i + 1], segs[4 * i + 2], segs[4 * i + 3]);
      if (d > 0 && d <= bd) {
        bd = d;
        best = [segs[4 * i] + t * (segs[4 * i + 2] - segs[4 * i]), segs[4 * i + 1] + t * (segs[4 * i + 3] - segs[4 * i + 1])];
      }
    });
    if (!best && extend > 0) {
      const dx = (x - ox) / len, dy = (y - oy) / len;
      const t = rayHit(grid, segs, x, y, dx, dy, extend, s);
      if (Number.isFinite(t)) best = [x + dx * t, y + dy * t];
    }
    if (best) {
      segs[2 * p] = best[0];
      segs[2 * p + 1] = best[1];
      moved += 1;
    }
  }
  if (moved) grid = boxGrid(segmentBoxes(segs, n), n, 1.0);
  // 3. Node every crossing, touch and collinear overlap.
  const splits = new Array(n);
  const addSplit = (s, t, x, y) => (splits[s] ||= []).push([t, x, y]);
  const boxes = segmentBoxes(segs, n);
  const { nx, start, items } = grid;
  const TOL = 1e-6;
  for (let c = 0; c < nx * grid.ny; c += 1) {
    const cxi = c % nx, cyi = Math.floor(c / nx);
    for (let qa = start[c]; qa < start[c + 1]; qa += 1) {
      const i = items[qa];
      for (let qb = qa + 1; qb < start[c + 1]; qb += 1) {
        const j = items[qb];
        // Each pair once: in the cell holding the low corner of the boxes' overlap.
        const ox0 = Math.max(boxes[4 * i], boxes[4 * j]), oy0 = Math.max(boxes[4 * i + 1], boxes[4 * j + 1]);
        const ox1 = Math.min(boxes[4 * i + 2], boxes[4 * j + 2]), oy1 = Math.min(boxes[4 * i + 3], boxes[4 * j + 3]);
        if (ox0 > ox1 + TOL || oy0 > oy1 + TOL) continue;
        if (grid.cx(ox0) !== cxi || grid.cy(oy0) !== cyi) continue;
        nodePair(segs, i, j, addSplit, TOL);
      }
    }
  }
  // 4. Sub-edges between consecutive split points; vertices merged within 1e-5 m.
  const vx = [], vy = [];
  const vmap = new Map();
  const Q = 1e-5;
  const vertex = (x, y) => {
    const ix = Math.round(x / Q), iy = Math.round(y / Q);
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        const v = vmap.get(`${ix + dx},${iy + dy}`);
        if (v !== undefined && Math.abs(vx[v] - x) <= Q && Math.abs(vy[v] - y) <= Q) return v;
      }
    }
    vx.push(x);
    vy.push(y);
    vmap.set(`${ix},${iy}`, vx.length - 1);
    return vx.length - 1;
  };
  const edgeSet = new Set();
  const eu = [], ev = [];
  for (let s = 0; s < n; s += 1) {
    const pts = [[0, segs[4 * s], segs[4 * s + 1]], ...(splits[s] || []), [1, segs[4 * s + 2], segs[4 * s + 3]]];
    pts.sort((a, b) => a[0] - b[0]);
    let prev = vertex(pts[0][1], pts[0][2]);
    for (let k = 1; k < pts.length; k += 1) {
      const v = vertex(pts[k][1], pts[k][2]);
      if (v === prev) continue;
      const key = prev < v ? `${prev},${v}` : `${v},${prev}`;
      if (!edgeSet.has(key)) {
        edgeSet.add(key);
        eu.push(prev);
        ev.push(v);
      }
      prev = v;
    }
  }
  return { vx: Float64Array.from(vx), vy: Float64Array.from(vy), eu: Int32Array.from(eu), ev: Int32Array.from(ev), moved, segments: n };
}

/** Records where segments i and j cross, touch or overlap. */
function nodePair(segs, i, j, addSplit, TOL) {
  const ax = segs[4 * i], ay = segs[4 * i + 1], bx = segs[4 * i + 2], by = segs[4 * i + 3];
  const cx = segs[4 * j], cy = segs[4 * j + 1], dx = segs[4 * j + 2], dy = segs[4 * j + 3];
  const rx = bx - ax, ry = by - ay, sx = dx - cx, sy = dy - cy;
  const lr = Math.hypot(rx, ry), ls = Math.hypot(sx, sy);
  const den = rx * sy - ry * sx;
  // An end of one segment lying on the other: a T-junction or a collinear overlap.
  const onto = (px, py, s, x0, y0, vx0, vy0, len) => {
    const t = ((px - x0) * vx0 + (py - y0) * vy0) / (len * len);
    if (t <= TOL / len || t >= 1 - TOL / len) return;
    if (Math.abs((px - x0) * vy0 - (py - y0) * vx0) / len > 10 * TOL) return;
    addSplit(s, t, px, py);
  };
  if (Math.abs(den) <= 1e-9 * lr * ls) {
    onto(cx, cy, i, ax, ay, rx, ry, lr);
    onto(dx, dy, i, ax, ay, rx, ry, lr);
    onto(ax, ay, j, cx, cy, sx, sy, ls);
    onto(bx, by, j, cx, cy, sx, sy, ls);
    return;
  }
  const t = ((cx - ax) * sy - (cy - ay) * sx) / den;
  const u = ((cx - ax) * ry - (cy - ay) * rx) / den;
  const ti = TOL / lr, tj = TOL / ls;
  if (t < -ti || t > 1 + ti || u < -tj || u > 1 + tj) return;
  const endI = t <= ti || t >= 1 - ti, endJ = u <= tj || u >= 1 - tj;
  if (endI && endJ) return; // they share an end
  if (endI) addSplit(j, u, t <= ti ? ax : bx, t <= ti ? ay : by);
  else if (endJ) addSplit(i, t, u <= tj ? cx : dx, u <= tj ? cy : dy);
  else {
    const px = ax + t * rx, py = ay + t * ry;
    addSplit(i, t, px, py);
    addSplit(j, u, px, py);
  }
}

/**
 * The faces of a planar graph. Dangling edges are pruned first (they bound
 * nothing). Each cycle has its face on the left: anticlockwise for a bounded
 * face, clockwise for the outer boundary of a connected part, which becomes
 * a hole of the face round it, or the outside.
 * Returns {faces: [{ring, area, holes}], outer: rings, liveEdges, pruned}.
 */
function planarFaces(g) {
  const { vx, vy, eu, ev } = g;
  const nv = vx.length, ne = eu.length;
  const deg = new Int32Array(nv);
  const alive = new Uint8Array(ne).fill(1);
  const inc = new Array(nv);
  for (let e = 0; e < ne; e += 1) {
    deg[eu[e]] += 1;
    deg[ev[e]] += 1;
    (inc[eu[e]] ||= []).push(e);
    (inc[ev[e]] ||= []).push(e);
  }
  const queue = [];
  for (let v = 0; v < nv; v += 1) if (deg[v] === 1) queue.push(v);
  let pruned = 0;
  while (queue.length) {
    const v = queue.pop();
    if (deg[v] !== 1) continue;
    for (const e of inc[v]) {
      if (!alive[e]) continue;
      alive[e] = 0;
      pruned += 1;
      deg[eu[e]] -= 1;
      deg[ev[e]] -= 1;
      const w = eu[e] === v ? ev[e] : eu[e];
      if (deg[w] === 1) queue.push(w);
      break;
    }
  }
  // Half-edges 2e (u -> v) and 2e + 1 (v -> u), sorted anticlockwise round their origin.
  const out = new Array(nv);
  for (let e = 0; e < ne; e += 1) {
    if (!alive[e]) continue;
    (out[eu[e]] ||= []).push(2 * e);
    (out[ev[e]] ||= []).push(2 * e + 1);
  }
  const org = (h) => (h & 1 ? ev[h >> 1] : eu[h >> 1]);
  const dst = (h) => (h & 1 ? eu[h >> 1] : ev[h >> 1]);
  const pos = new Int32Array(2 * ne);
  const ang = new Float64Array(2 * ne);
  for (let v = 0; v < nv; v += 1) {
    const list = out[v];
    if (!list) continue;
    for (const h of list) ang[h] = Math.atan2(vy[dst(h)] - vy[v], vx[dst(h)] - vx[v]);
    list.sort((a, b) => ang[a] - ang[b]);
    list.forEach((h, i) => (pos[h] = i));
  }
  // The face on the left of h continues with the edge clockwise from h's twin.
  const next = (h) => {
    const t = h ^ 1, list = out[org(t)];
    return list[(pos[t] - 1 + list.length) % list.length];
  };
  const seen = new Uint8Array(2 * ne);
  const cycles = [];
  for (let e = 0; e < ne; e += 1) {
    if (!alive[e]) continue;
    for (const h0 of [2 * e, 2 * e + 1]) {
      if (seen[h0]) continue;
      const ring = [];
      let h = h0, guard = 0;
      do {
        seen[h] = 1;
        ring.push([vx[org(h)], vy[org(h)]]);
        h = next(h);
      } while (h !== h0 && (guard += 1) < 2 * ne + 2);
      cycles.push({ ring, area: areaOf(ring), v0: org(h0) });
    }
  }
  // Connected parts, so a part's outer cycle is not taken for a hole in its own faces.
  const parent = Int32Array.from({ length: nv }, (_, i) => i);
  const find = (a) => {
    while (parent[a] !== a) a = parent[a] = parent[parent[a]];
    return a;
  };
  for (let e = 0; e < ne; e += 1) if (alive[e]) parent[find(eu[e])] = find(ev[e]);
  const faces = [];
  for (const c of cycles) if (c.area > 1e-10) faces.push({ ring: c.ring, area: c.area, holes: [], part: find(c.v0) });
  const fb = ringBoxes(faces.map((f) => f.ring));
  const fgrid = boxGrid(fb, faces.length, 2.0);
  const outer = [];
  for (const c of cycles) {
    if (c.area > 1e-10) continue;
    const part = find(c.v0);
    const [px, py] = c.ring[0];
    let best = -1, ba = Infinity;
    gridQuery(fgrid, px, py, px, py, (f) => {
      const F = faces[f];
      if (F.part === part || F.area >= ba) return;
      if (px < fb[4 * f] || px > fb[4 * f + 2] || py < fb[4 * f + 1] || py > fb[4 * f + 3]) return;
      if (inRing(px, py, F.ring)) [best, ba] = [f, F.area];
    });
    if (best >= 0) faces[best].holes.push(c.ring);
    else outer.push(c.ring);
  }
  const live = [];
  for (let e = 0; e < ne; e += 1) if (alive[e]) live.push(e);
  return { faces, outer, live, pruned };
}

// ===================================================================== the building outline (raster closing)

/**
 * Exact squared Euclidean distance transform of a binary grid (Felzenszwalb
 * and Huttenlocher): for every pixel, the squared distance in pixels to the
 * nearest pixel where `seed` is 1.
 */
function distanceTransform(seed, w, h) {
  const INF = 1e20;
  const d = new Float64Array(w * h);
  for (let i = 0; i < w * h; i += 1) d[i] = seed[i] ? 0 : INF;
  const n = Math.max(w, h);
  const f = new Float64Array(n), z = new Float64Array(n + 1), out = new Float64Array(n);
  const v = new Int32Array(n);
  const pass = (len, get, set) => {
    for (let q = 0; q < len; q += 1) f[q] = get(q);
    let k = 0;
    v[0] = 0;
    z[0] = -INF;
    z[1] = INF;
    for (let q = 1; q < len; q += 1) {
      let s;
      while (true) {
        s = (f[q] + q * q - (f[v[k]] + v[k] * v[k])) / (2 * q - 2 * v[k]);
        if (s > z[k] || k === 0) break;
        k -= 1;
      }
      if (s <= z[k]) {
        v[k] = q;
        z[k] = -INF;
        z[k + 1] = INF;
        continue;
      }
      k += 1;
      v[k] = q;
      z[k] = s;
      z[k + 1] = INF;
    }
    k = 0;
    for (let q = 0; q < len; q += 1) {
      while (z[k + 1] < q) k += 1;
      out[q] = (q - v[k]) * (q - v[k]) + f[v[k]];
    }
    for (let q = 0; q < len; q += 1) set(q, out[q]);
  };
  for (let x = 0; x < w; x += 1) pass(h, (q) => d[q * w + x], (q, val) => (d[q * w + x] = val));
  for (let y = 0; y < h; y += 1) pass(w, (q) => d[y * w + q], (q, val) => (d[y * w + q] = val));
  return d;
}

/**
 * The outline of the drawn building: the walls rasterised, closed (dilated
 * and eroded by `close` metres, so openings up to twice that are bridged),
 * their inside filled; parts smaller than minMass dropped; the boundary
 * traced along pixel edges and simplified. Returns rings (anticlockwise
 * outer boundaries, clockwise holes) in metres.
 */
function closedOutline(segs, n, { close = 1.0, minMass = 4, res = 0.1 } = {}) {
  if (!n) return [];
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < 2 * n; i += 1) {
    x0 = Math.min(x0, segs[2 * i]);
    x1 = Math.max(x1, segs[2 * i]);
    y0 = Math.min(y0, segs[2 * i + 1]);
    y1 = Math.max(y1, segs[2 * i + 1]);
  }
  while (((x1 - x0) / res + 4) * ((y1 - y0) / res + 4) > 6e6) res *= 1.5;
  const pad = Math.ceil(close / res) + 3;
  const w = Math.ceil((x1 - x0) / res) + 2 * pad, h = Math.ceil((y1 - y0) / res) + 2 * pad;
  const ox = x0 - pad * res, oy = y0 - pad * res;
  const wall = new Uint8Array(w * h);
  for (let i = 0; i < n; i += 1) {
    const ax = (segs[4 * i] - ox) / res, ay = (segs[4 * i + 1] - oy) / res, bx = (segs[4 * i + 2] - ox) / res, by = (segs[4 * i + 3] - oy) / res;
    const steps = Math.ceil(2 * Math.max(Math.abs(bx - ax), Math.abs(by - ay))) + 1;
    for (let k = 0; k <= steps; k += 1) {
      const px = Math.floor(ax + ((bx - ax) * k) / steps), py = Math.floor(ay + ((by - ay) * k) / steps);
      if (px >= 0 && py >= 0 && px < w && py < h) wall[py * w + px] = 1;
    }
  }
  const r2 = (close / res) ** 2;
  const dw = distanceTransform(wall, w, h);
  // Outside: reached from the border without crossing the dilated walls.
  const outside = new Uint8Array(w * h);
  const stack = [];
  for (let x = 0; x < w; x += 1) stack.push(x, (h - 1) * w + x);
  for (let y = 0; y < h; y += 1) stack.push(y * w, y * w + w - 1);
  while (stack.length) {
    const p = stack.pop();
    if (outside[p] || dw[p] <= r2) continue;
    outside[p] = 1;
    const x = p % w, y = (p / w) | 0;
    if (x > 0) stack.push(p - 1);
    if (x < w - 1) stack.push(p + 1);
    if (y > 0) stack.push(p - w);
    if (y < h - 1) stack.push(p + w);
  }
  const de = distanceTransform(outside, w, h);
  const mass = new Uint8Array(w * h);
  for (let p = 0; p < w * h; p += 1) mass[p] = !outside[p] && (de[p] > r2 || wall[p]) ? 1 : 0;
  // Drop small parts.
  const label = new Int32Array(w * h).fill(-1);
  const keep = [];
  for (let p0 = 0; p0 < w * h; p0 += 1) {
    if (!mass[p0] || label[p0] >= 0) continue;
    const id = keep.length;
    let count = 0;
    stack.push(p0);
    label[p0] = id;
    while (stack.length) {
      const p = stack.pop();
      count += 1;
      const x = p % w, y = (p / w) | 0;
      for (const q of [x > 0 ? p - 1 : -1, x < w - 1 ? p + 1 : -1, y > 0 ? p - w : -1, y < h - 1 ? p + w : -1]) {
        if (q >= 0 && mass[q] && label[q] < 0) {
          label[q] = id;
          stack.push(q);
        }
      }
    }
    keep.push(count * res * res >= minMass);
  }
  for (let p = 0; p < w * h; p += 1) if (mass[p] && !keep[label[p]]) mass[p] = 0;
  // One pixel more, so the outline stands clear of the facade and never runs along it.
  const grown = mass.slice();
  for (let p = 0; p < w * h; p += 1) {
    if (mass[p]) continue;
    const x = p % w, y = (p / w) | 0;
    if ((x > 0 && mass[p - 1]) || (x < w - 1 && mass[p + 1]) || (y > 0 && mass[p - w]) || (y < h - 1 && mass[p + w])) grown[p] = 1;
  }
  return traceMask(grown, w, h).map((ring) => cleanRing(simplifyClosed(ring.map(([x, y]) => [ox + x * res, oy + y * res]), res * 0.75), 1e-6));
}

/**
 * Boundaries of a binary mask along pixel edges, inside on the left: outer
 * boundaries anticlockwise, holes clockwise, in pixel units.
 */
function traceMask(mask, w, h) {
  const W = w + 1;
  // At most two directed edges leave a corner (two at a saddle).
  const nxt = new Int32Array(W * (h + 1) * 2).fill(-1);
  const add = (a, b) => {
    if (nxt[2 * a] < 0) nxt[2 * a] = b;
    else nxt[2 * a + 1] = b;
  };
  const at = (x, y) => (x >= 0 && y >= 0 && x < w && y < h ? mask[y * w + x] : 0);
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      if (!mask[y * w + x]) continue;
      if (!at(x, y - 1)) add(y * W + x, y * W + x + 1);
      if (!at(x + 1, y)) add(y * W + x + 1, (y + 1) * W + x + 1);
      if (!at(x, y + 1)) add((y + 1) * W + x + 1, (y + 1) * W + x);
      if (!at(x - 1, y)) add((y + 1) * W + x, y * W + x);
    }
  }
  const rings = [];
  for (let s = 0; s < W * (h + 1); s += 1) {
    while (nxt[2 * s] >= 0) {
      const ring = [];
      let a = s, prev = -1, guard = 0;
      do {
        let k = 2 * a;
        if (nxt[k + 1] >= 0 && prev >= 0) {
          // A saddle (pixels touching at a corner): turn right, joining them.
          const dxIn = (a % W) - (prev % W), dyIn = Math.floor(a / W) - Math.floor(prev / W);
          const b0 = nxt[k], dx0 = (b0 % W) - (a % W), dy0 = Math.floor(b0 / W) - Math.floor(a / W);
          if (dxIn * dy0 - dyIn * dx0 > 0) k += 1;
        }
        const b = nxt[k];
        if (b < 0) break;
        // Remove the edge used.
        if (k % 2 === 0) {
          nxt[k] = nxt[k + 1];
          nxt[k + 1] = -1;
        } else nxt[k] = -1;
        ring.push([a % W, Math.floor(a / W)]);
        prev = a;
        a = b;
      } while (a !== s && (guard += 1) < 4 * W * (h + 1));
      if (ring.length >= 4) rings.push(ring);
      if (nxt[2 * s] < 0) break;
    }
  }
  return rings;
}

// ===================================================================== words: names, classes, layers

/**
 * Texts typed in the old AutoCAD Arabic fonts are stored as the Latin keys of
 * the Arabic 101 keyboard ('Hg' = 'ال', 'plhL' = 'حمام').
 */
const ARABIC_KEYS = (() => {
  const m = {};
  const keys = "qwertyuiop[]asdfghjkl;'zxcvbnm,./`";
  const arabic = "ض ص ث ق ف غ ع ه خ ح ج د ش س ي ب ل ا ت ن م ك ط ئ ء ؤ ر لا ى ة و ز ظ ذ".split(" ");
  [...keys].forEach((k, i) => (m[k] = arabic[i]));
  for (const k of keys) if (/[a-z]/.test(k)) m[k.toUpperCase()] = m[k];
  Object.assign(m, { ":": "ك", _: "", ">": "ز", B: "لأ" });
  return m;
})();

/** A text decoded from the Arabic keyboard font: 'auto' when it looks encoded ('Hg', '__'), 'all' when it has Latin letters. */
export function decodeArabicKeys(s, mode = "auto") {
  if (mode === "none" || !s) return s;
  const encoded = mode === "all" ? /[A-Za-z]/.test(s) : s.includes("Hg") || s.includes("__");
  return encoded ? [...s].map((c) => ARABIC_KEYS[c] ?? c).join("") : s;
}

/**
 * Room classes from the words of a name, the first class that matches
 * winning (English, Arabic and common transliterations). The classes are the
 * hotel schedule's space classes, which the AI suggester also uses.
 */
export const ROOM_CLASSES = ["Unit", "Corridor", "Stair", "Ramp", "Toilet", "Service", "Office", "Commercial", "Parking", "Arcade", "Shaft", "Void", "Tank"];
const VOCABULARY = [
  ["Toilet", /bath|toilet|\bw\.?\s?c\b|wash\s?room|lavator|shower|restroom|حمام|دورة مياه|دورات مياه|مغاسل|مرحاض|hammam/i],
  ["Shaft", /\blift|elevator|shaft|\bduct|riser|مصعد|مناور|منور خدمة|mas'?ad/i],
  ["Tank", /tank|خزان|cistern/i],
  ["Stair", /stair|درج|سلم|staircase|escalier|daraj/i],
  ["Ramp", /ramp|منحدر|منزلق/i],
  ["Parking", /parking|garage|loading|موقف|مواقف|car\s?park/i],
  ["Void", /void|فراغ|منور|open to below|light\s?well/i],
  ["Arcade", /arcade|porch|terrace|balcon|veranda|رواق|مظلة|شرفة|بلكون|تراس/i],
  ["Corridor", /corridor|lobby|looby|\bhall\b|entrance|passage|foyer|vestibule|ممر|مدخل|ردهة|موزع|بهو|صالة مدخل|dahliz|mamar/i],
  ["Commercial", /shop|commer|cafeteria|restaurant|retail|store front|محل|محلات|مطعم|كافتيريا|معرض/i],
  ["Office", /office|manage|admin|reception|مكتب|إدارة|ادارة|استقبال|maktab/i],
  ["Service", /store|storage|elec|pump|transform|generator|chiller|kitchen|pantry|garbage|machine|refrig|servi|recic|recycl|control|laundry|deliver|dilivery|plant|mech|janitor|مطبخ|غسيل|مخزن|مستودع|كهرباء|مضخات|غرفة خدمة|غرفة الكهرباء|matbakh/i],
  ["Unit", /bed|room|suite|majlis|living|dining|guest|family|maid|study|sitting|^\d{3,4}$|غرفة|نوم|مجلس|معيشة|طعام|ضيوف|صالة|صاله|اولاد|بنات|شغالة|خادمة|رئيسي|جلوس|ghurfa|salah/i],
];

/** The class a room's names give, or null. */
export function roomClassFromNames(names) {
  for (const [cls, rx] of VOCABULARY) if (names.some((n) => rx.test(n))) return cls;
  // lod1.js's word list (more languages for stairs, lifts and corridors).
  for (const n of names) {
    const c = lod1ClassFromName(n);
    if (c !== "Unit" && ROOM_CLASSES.includes(c)) return c;
  }
  return null;
}

/**
 * What a text is to a room: "name" (words), "number" (a room number or tag:
 * 101, 2A, G-12), or null (levels, areas, dimensions, grid bubbles, notes).
 */
function textRole(s) {
  const t = s.trim();
  if (t.length < 2 || t.length > 60) return null;
  if (/^[+\-±]?\s*\d+([.,]\d+)\s*(m2|m²|sqm|sq\.?\s?m|m|mm)?$/i.test(t)) return null; // levels and areas
  if (/^[+\-±]?\s*\d+\s*(m2|m²|sqm|sq\.?\s?m)$/i.test(t)) return null;
  if (/^\d+([.,]\d+)?\s*[x×X*]\s*\d+([.,]\d+)?/.test(t)) return null; // dimensions
  if (/^(ffl|f\.f\.l|fl|sl|ssl|tos|lvl|level|el|ceiling|clg|h\s?=)\b/i.test(t)) return null;
  if (/^(scale|date|drawn|checked|sheet|dwg|project|client|rev|note|notes|title)\b/i.test(t)) return null;
  if (/^\d{2,5}[A-Za-z]?$/.test(t) || /^[A-Za-z]{1,2}[-.]?\d{1,4}[A-Za-z]?$/.test(t)) return "number";
  if (/[A-Za-z؀-ۿ]{2,}/.test(t)) return "name";
  return null;
}

// Layer names offices use, in English, Arabic and transliterated Arabic.
const LAYER_WORDS = {
  wall: /wall|wand|\bmur|muro|parete|partition|stud|masonry|brick|block|beton|concrete|const|struct|plaster|col(umn)?s?\b|\bcol|pillar|جدار|جدران|حائط|حوائط|حيط|عمود|أعمدة|اعمدة|اعمده|خرسان|بلوك|طوب|jidar|jdar|hait|haet|amod|a-wall|awall|s-col|s-wall/i,
  door: /door|\bdr\b|tür|tuer|porte|puerta|باب|ابواب|أبواب|bab\b|abwab|a-door|opening/i,
  window: /window|windo|\bwin\b|glaz|glass|fenster|fenetre|ventana|نافذ|نوافذ|شباك|شبابيك|زجاج|shubbak|nafitha|a-glaz/i,
  text: /text|txt|name|label|room|anno|tag|نص|نصوص|كتابة|كتابه|اسماء|أسماء|a-anno|a-area|iden/i,
  never: /dim|grid|axis|axes|محاور|محور|title|frame|border|إطار|اطار|hatch|furn|فرش|farsh|أثاث|اثاث|athath|marble|tile|floor|pattern|paving|بلاط|رخام|ceramic|elec|light|power|fire|alarm|hvac|plumb|sanit|صحي|صحية|tree|أشجار|landscape|site|سور|elev|section|defpoints|viewport|vport|cable|socket|data|cctv|sprink|chw|thin|centre|center|ceil|سقف|درج|stair|text|txt|كتابة/i,
};

// ===================================================================== doors from the drawing

/**
 * Circle through a polyline's first, middle and last points, if every point
 * lies on it: {cx, cy, r, sweep (radians), p0, p1} or null.
 */
function fitArc(pts) {
  const n = pts.length / 2;
  if (n < 4) return null;
  const ax = pts[0], ay = pts[1], m = Math.floor(n / 2), bx = pts[2 * m], by = pts[2 * m + 1], cx = pts[2 * n - 2], cy = pts[2 * n - 1];
  if (Math.hypot(ax - cx, ay - cy) < 1e-9) return null; // closed
  const d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
  if (Math.abs(d) < 1e-12) return null;
  const a2 = ax * ax + ay * ay, b2 = bx * bx + by * by, c2 = cx * cx + cy * cy;
  const ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d, uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
  const r = Math.hypot(ax - ux, ay - uy);
  for (let i = 0; i < n; i += 1) if (Math.abs(Math.hypot(pts[2 * i] - ux, pts[2 * i + 1] - uy) - r) > Math.max(0.02 * r, 1e-9)) return null;
  const ta = Math.atan2(ay - uy, ax - ux), tb = Math.atan2(by - uy, bx - ux), tc = Math.atan2(cy - uy, cx - ux);
  const ccw = (a, b) => ((b - a) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI);
  const s = ccw(ta, tc);
  const sweep = ccw(ta, tb) < s ? s : 2 * Math.PI - s;
  return { cx: ux, cy: uy, r, sweep, p0: [ax, ay], p1: [cx, cy] };
}

/** A door swing in metres (a quarter-ish circle of door-leaf radius), from an arc or a polyline. */
function swingOf(e, tf) {
  let f = null;
  if (e.type === "arc") {
    let sweep = e.a1 - e.a0;
    while (sweep <= 0) sweep += 2 * Math.PI;
    if (e.kind === "CIRCLE") return null;
    const [cx, cy] = tf(e.cx, e.cy);
    const p0 = tf(e.cx + (e.r * Math.cos(e.a0)), e.cy + e.r * Math.sin(e.a0)), p1 = tf(e.cx + e.r * Math.cos(e.a1), e.cy + e.r * Math.sin(e.a1));
    f = { cx, cy, r: Math.hypot(p0[0] - cx, p0[1] - cy), sweep, p0, p1 };
  } else if (e.type === "polyline" && !e.closed && e.points.length >= 8) {
    const pts = new Array(e.points.length);
    for (let i = 0; i < e.points.length; i += 2) [pts[i], pts[i + 1]] = tf(e.points[i], e.points[i + 1]);
    f = fitArc(pts);
  }
  if (!f) return null;
  const deg = (f.sweep * 180) / Math.PI;
  return f.r >= 0.45 && f.r <= 1.6 && deg >= 25 && deg <= 135 ? f : null;
}

// ===================================================================== layer suggestions

/**
 * Likely wall, door, window and text layers of a sheet, the door blocks and
 * the drawing's scale, from layer names (English, Arabic, transliterated) and
 * geometry: door layers hold door swings, wall layers hold long runs of
 * parallel line pairs a wall's thickness apart, text layers hold room names.
 *
 * @param {object[]} entities  readDxf's entities (one sheet, or a region of one)
 * @param {{units?: {toMetres: number|null}}} opts  the declared units, a fallback
 * @returns {{walls: string[], doors: string[], windows: string[], text: string[],
 *   doorBlocks: string[], doorPattern: string|null, scale: number, scaleSource: string, layers: object[]}}
 */
export function suggestLayers(entities, { units = null } = {}) {
  // 1. Scale: door swings are 0.6 to 1.2 m in radius, whatever the unit.
  const radii = [];
  const fits = new Map();
  const swingLayers = new Map();
  const swingBlocks = new Map();
  const blockName = (e) => (e.owner >= 0 ? entities[e.owner]?.name : null);
  for (const e of entities) {
    if (e.type !== "arc" && !(e.type === "polyline" && !e.closed && e.points.length >= 8)) continue;
    const f = e.type === "arc" ? (e.kind === "ARC" ? { r: e.r, sweep: ((e.a1 - e.a0) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) } : null) : fitArc(e.points);
    if (!f) continue;
    const deg = (f.sweep * 180) / Math.PI;
    if (deg < 60 || deg > 120) continue;
    radii.push(f.r);
    f.layer = e.layer;
    f.block = blockName(e);
    fits.set(e, f);
  }
  // The drawing's size (the middle 90 % of its linework), which a building's
  // plan keeps between 3 and 400 m whatever the unit.
  const xs = [], ys = [];
  for (const e of entities) {
    if (!e.points) continue;
    for (let i = 0; i < e.points.length; i += 2 * Math.max(1, e.points.length >> 3)) {
      xs.push(e.points[i]);
      ys.push(e.points[i + 1]);
    }
  }
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  const pct = (a, q) => a[Math.min(a.length - 1, Math.floor(a.length * q))];
  const span = xs.length ? Math.max(pct(xs, 0.95) - pct(xs, 0.05), pct(ys, 0.95) - pct(ys, 0.05)) : 0;
  const declared = units?.toMetres || null;
  const plausible = [0.001, 0.01, 0.0254, 0.1, 0.3048, 1].filter((s) => !span || (span * s >= 3 && span * s <= 400));
  if (declared && !plausible.includes(declared) && (!span || (span * declared >= 3 && span * declared <= 400))) plausible.push(declared);
  // Of the plausible units, the one that puts most swings at a door's radius;
  // ties go to the declared unit, then to metric units.
  const order = (s) => (s === declared ? 0 : [0.001, 1, 0.01, 0.1, 0.0254, 0.3048].indexOf(s) + 1);
  let scale = !span ? declared || 1 : declared && plausible.includes(declared) ? declared : plausible.sort((a, b) => order(a) - order(b))[0] || declared || 1;
  let scaleSource = declared && scale === declared ? "declared units" : span ? "drawing size" : "assumed metres";
  if (radii.length >= 2) {
    let best = 0;
    for (const s of plausible.sort((a, b) => order(a) - order(b))) {
      const hits = radii.filter((r) => r * s >= 0.55 && r * s <= 1.4).length;
      if (hits > best) [best, scale, scaleSource] = [hits, s, "door swings"];
    }
  }
  for (const f of fits.values()) {
    if (f.r * scale < 0.45 || f.r * scale > 1.6) continue;
    swingLayers.set(f.layer, (swingLayers.get(f.layer) || 0) + 1);
    if (f.block) swingBlocks.set(f.block, (swingBlocks.get(f.block) || 0) + 1);
  }
  // 2. Per layer: linework, wall-like pairs, texts.
  const stats = new Map();
  const of = (layer) => {
    if (!stats.has(layer)) stats.set(layer, { layer, entities: 0, length: 0, segments: [], texts: 0, names: 0, swings: swingLayers.get(layer) || 0, inserts: 0, box: [Infinity, Infinity, -Infinity, -Infinity] });
    return stats.get(layer);
  };
  for (const e of entities) {
    const s = of(e.layer);
    s.entities += 1;
    if (e.type === "text") {
      s.texts += 1;
      if (textRole(e.text) === "name") s.names += 1;
    } else if (e.type === "insert") s.inserts += 1;
    else if (e.points) {
      for (let i = 0; i < e.points.length; i += 2) {
        s.box[0] = Math.min(s.box[0], e.points[i]);
        s.box[1] = Math.min(s.box[1], e.points[i + 1]);
        s.box[2] = Math.max(s.box[2], e.points[i]);
        s.box[3] = Math.max(s.box[3], e.points[i + 1]);
      }
      for (let i = 2; i < e.points.length; i += 2) {
        const len = Math.hypot(e.points[i] - e.points[i - 2], e.points[i + 1] - e.points[i - 1]) * scale;
        s.length += len;
        if (len >= 0.3 && s.segments.length < 6000) s.segments.push([e.points[i - 2] * scale, e.points[i - 1] * scale, e.points[i] * scale, e.points[i + 1] * scale]);
      }
    }
  }
  // A wall is drawn as two parallel lines 5 to 60 cm apart that run side by
  // side; a line with partners on both sides is a pattern (tiles, hatching).
  for (const s of stats.values()) {
    const segs = s.segments;
    s.paired = 0;
    s.both = 0;
    if (segs.length < 8) continue;
    const flat = Float64Array.from(segs.flat());
    const grid = boxGrid(segmentBoxes(flat, segs.length), segs.length, 1.0);
    let paired = 0, both = 0, total = 0;
    segs.forEach(([ax, ay, bx, by], i) => {
      const len = Math.hypot(bx - ax, by - ay);
      const ux = (bx - ax) / len, uy = (by - ay) / len;
      total += len;
      let left = false, right = false;
      gridQuery(grid, Math.min(ax, bx) - 0.6, Math.min(ay, by) - 0.6, Math.max(ax, bx) + 0.6, Math.max(ay, by) + 0.6, (j) => {
        if (j === i) return;
        const [cx, cy, dx, dy] = segs[j];
        const l2 = Math.hypot(dx - cx, dy - cy);
        if (Math.abs(((dx - cx) / l2) * uy - ((dy - cy) / l2) * ux) > 0.03) return;
        const signed = (cx - ax) * uy - (cy - ay) * ux, off = Math.abs(signed);
        if (off < 0.05 || off > 0.6) return;
        const t0 = (cx - ax) * ux + (cy - ay) * uy, t1 = (dx - ax) * ux + (dy - ay) * uy;
        const lo = Math.max(0, Math.min(t0, t1)), hi = Math.min(len, Math.max(t0, t1));
        if (hi - lo >= 0.5 * Math.min(len, l2)) {
          if (signed > 0) left = true;
          else right = true;
          if (left && right) return false;
        }
      });
      if (left || right) paired += len;
      if (left && right) both += len;
    });
    s.paired = total ? paired / total : 0;
    s.both = total ? both / total : 0;
  }
  // A sheet frame: a few lines round most of the drawing.
  const W = span ? (pct(xs, 0.95) - pct(xs, 0.05)) : 0, H = span ? (pct(ys, 0.95) - pct(ys, 0.05)) : 0;
  const frame = (s) => s.entities <= 8 && s.box[2] - s.box[0] >= 0.8 * W && s.box[3] - s.box[1] >= 0.8 * H && W > 0;
  const layers = [...stats.values()].map((s) => ({
    layer: s.layer, entities: s.entities, length: Math.round(s.length * 10) / 10, paired: Math.round(s.paired * 100) / 100, both: Math.round(s.both * 100) / 100,
    swings: s.swings, texts: s.texts, names: s.names, inserts: s.inserts, frame: frame(s),
  })).sort((a, b) => b.length - a.length);
  const named = (l, kind) => LAYER_WORDS[kind].test(l.layer);
  // Annotation layers: many texts for their linework (tags, labels).
  const annotation = (l) => l.texts >= 10 && l.length < 5 * l.texts;
  const walls = layers.filter((l) => !named(l, "never") && !named(l, "door") && !named(l, "window") && !annotation(l) && !l.frame &&
    ((named(l, "wall") && l.length >= 0.5) || (l.paired >= 0.45 && l.both < 0.35 && l.length >= 30))).map((l) => l.layer);
  const doors = layers.filter((l) => l.swings >= 2 || (named(l, "door") && l.length > 0)).map((l) => l.layer);
  // Windows close the facade, so they are barriers even on a layer that holds doors too.
  const windows = layers.filter((l) => named(l, "window") && l.length > 0 && !named(l, "never")).map((l) => l.layer);
  const text = layers.filter((l) => l.names >= 3 && !/dim|title|frame|border|grid|axis|محاور/i.test(l.layer)).map((l) => l.layer);
  // Door blocks: named like doors, or holding a swing.
  const blockNames = new Set(entities.filter((e) => e.type === "insert").map((e) => e.name));
  const doorBlocks = [...blockNames].filter((n) => !/^\*/.test(n) && (LAYER_WORDS.door.test(n) || swingBlocks.has(n)) && !LAYER_WORDS.window.test(n));
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  return {
    walls, doors, windows, text, doorBlocks,
    doorPattern: doorBlocks.length ? `^(${doorBlocks.map(esc).join("|")})$` : null,
    scale, scaleSource, layers,
  };
}

// ===================================================================== drafting the rooms

const DEFAULTS = Object.freeze({
  scale: 1, origin: [0, 0], clip: null,
  snap: 0.03, extend: 0.3, minArea: 0.8, minWidth: 0.35, holeArea: 2.0,
  close: 1.0, minMass: 4, maxWall: 0.8, maxCut: 3.5, grow: true, split: true, dashed: false,
  mergeOpenings: true, mergeArea: 8,
  decode: "auto", layerMatch: "either",
  // Blocks drawn on wall layers that are not walls: sanitary fixtures, furniture, cars, plants.
  notWallPattern: "wc|toilet|bidet|bedeh|bath|basin|sink|wash|shower|urinal|lav\\b|chair|bed|sofa|table|desk|stove|cooker|oven|fridge|furn|kitchen|counter|car\\b|tree|plant|مغسل|مرحاض|سرير|كرسي|طاولة",
  fixturePattern: "wc|toilet|bidet|bedeh|bath|basin|handwash|shower|urinal|lav\\b|مغسل|مرحاض",
});

/**
 * Rooms and doors drafted from one storey's sheet.
 *
 *  1. Walls: the linework of the wall layers (and window layers, which close
 *     the facade), less dashed and dotted lines (projections of the floor
 *     below or of beams above), the parts of fixture, furniture and door
 *     blocks, and the door swings, in metres: (p - origin) * scale, kept when every point
 *     is inside `clip` [x0, y0, x1, y1] (metres, after the transform).
 *  2. Doors: every door block (name matching `doorPattern`) and every swing on
 *     a door layer gives a hinge, a leaf and its closed position: of the
 *     arc's two chords from the hinge, the one the drawing draws along (the
 *     leaf, open) is not the opening; with no leaf drawn, the closed one is
 *     the chord that runs along the wall. Double doors meet in the middle. The opening
 *     is bridged jamb to jamb on both faces of the wall (rays find the jambs
 *     and the faces), so the doorway is a piece of wall, not a leak. Doors
 *     with no swing (folding, sliding) are bridged along their block's long
 *     axis, and a dashed line as long as a door is wide (0.5 to 2.5 m) on a
 *     wall or door layer, the head of an opening, is bridged as an opening.
 *  3. Outline: the walls closed by `close` (1 m: openings up to 2 m are
 *     bridged, as the lab's buffer and its inverse), filled and traced; it
 *     bounds the plan where a facade is left open.
 *  4. The arrangement: endpoints within `snap` joined, short lines extended up
 *     to `extend` onto the wall they stop short of, every crossing noded,
 *     dangling lines pruned; its faces. A face is a room when its area is at
 *     least `minArea` and its width (2 area / perimeter) at least `minWidth`,
 *     and it is not a long thin strip (width under 0.6 m and 4 pi A / P^2
 *     under 0.1): the rest are walls, columns and slivers. The outside and
 *     the strips between the outline and the facade are not rooms.
 *  5. Names: each text inside a room is one of its labels ("name" texts
 *     first; levels, areas, dimensions and notes are not labels). An
 *     opening head between a named room and a small unnamed space (up to
 *     `mergeArea`) that opens onto no other named room (an entrance lobby, a
 *     wardrobe recess) is dropped and steps 3 to 5 run again, so the two are
 *     one room (`mergeOpenings`).
 *  6. Split by names: a room holding two names more than 1.5 m apart is cut
 *     along the shortest line (up to `maxCut`) from a re-entrant corner (a
 *     wall end) that puts the names on different sides: across an opening
 *     between two wall ends, or along a wall's line to the far side. The cut
 *     becomes an opening between the two rooms.
 *  7. Grow to the wall centre-lines: every edge moves out by half the
 *     thickness of the wall behind it (rays across the wall to the next room
 *     or the outside, up to `maxWall`), so neighbours meet on one line as
 *     the lab's rooms do; holes under `holeArea` (columns) are filled.
 *  8. Doors join the rooms found on either side of the opening (probes up to
 *     1 m); a side in no room is the outside, or unknown inside the walls.
 *
 * @param {object[]} entities  readDxf's entities
 * @param {object} opts  wallLayers, doorLayers, windowLayers, textLayers (null: every layer),
 *   doorPattern (block names, a RegExp or its source), notWallPattern, scale, origin, clip,
 *   snap, extend, minArea, minWidth, holeArea, close, minMass, maxWall, maxCut, grow,
 *   split, mergeOpenings, mergeArea, dashed (keep dashed lines as walls), decode
 *   ("auto" | "all" | "none"), layerMatch ("path" | "layer" | "either"), ignoreTexts
 *   (texts that are not room names: notes, equipment tags), fixturePattern, keepLines
 *   (return the wall linework and door bridges as `lines`, flat [x0, y0, x1, y1, ...]
 *   in metres, to draw under the draft)
 * @returns {{rooms: object[], doors: object[], outline: number[][]|null, issues: object[], stats: object, settings: object}}
 *   rooms: {id, polygon (grown, anticlockwise), holes, area, inner (the inner faces),
 *   innerArea, label, labels, class (or null), classSource, fixtures, centroid, at (a
 *   point inside, for the label), neighbours}; doors: {id, kind ("door" | "opening"),
 *   position, direction, width, rooms [a, b] (null: outside or unknown), sides
 *   ("room" | "outside" | "unknown"), wall (thickness), source, block, jambs};
 *   issues: {type, message, at, rooms}.
 */
export function draftRooms(entities, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const t0 = now();
  const scale = Number(o.scale) || 1;
  const [ox, oy] = o.origin || [0, 0];
  const tf = (x, y) => [(x - ox) * scale, (y - oy) * scale];
  const clip = o.clip;
  const inClip = (x, y) => !clip || (x >= clip[0] && x <= clip[2] && y >= clip[1] && y <= clip[3]);
  const set = (list) => new Set((list || []).map(String));
  const wallSet = set([...(o.wallLayers || []), ...(o.windowLayers || [])]);
  const doorSet = set(o.doorLayers), textSet = o.textLayers ? set(o.textLayers) : null;
  const matches = (s, e) => (o.layerMatch === "path" ? s.has(e.path) : o.layerMatch === "layer" ? s.has(e.layer) : s.has(e.path) || s.has(e.layer));
  const rx = (p) => (p instanceof RegExp ? p : p ? new RegExp(p, "i") : null);
  const doorRx = rx(o.doorPattern), notWallRx = rx(o.notWallPattern), fixtureRx = rx(o.fixturePattern);
  const issues = [];

  // Block ownership: the chain of inserts above each entity.
  const doorOwner = new Int32Array(entities.length).fill(-1); // the outermost door insert holding the entity
  const excluded = new Uint8Array(entities.length); // inside a fixture or furniture block
  entities.forEach((e, i) => {
    const up = e.owner;
    if (up >= 0) {
      doorOwner[i] = doorOwner[up];
      excluded[i] = excluded[up];
    }
    if (e.type === "insert") {
      if (doorOwner[i] < 0 && doorRx?.test(e.name)) doorOwner[i] = i;
      if (notWallRx?.test(e.name) && doorOwner[i] < 0) excluded[i] = 1;
    }
  });
  const inClipAll = (e) => {
    if (e.points) {
      for (let i = 0; i < e.points.length; i += 2) {
        const [x, y] = tf(e.points[i], e.points[i + 1]);
        if (!inClip(x, y)) return false;
      }
      return true;
    }
    if (e.type === "insert") {
      const [x, y] = tf((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2);
      return inClip(x, y);
    }
    const [x, y] = tf(e.x, e.y);
    return inClip(x, y);
  };

  // 1. Walls.
  const walls = [];
  const swings = []; // free swings on door layers
  const heads = []; // dashed lines across openings
  entities.forEach((e, i) => {
    if (!e.points || doorOwner[i] >= 0 || excluded[i]) return;
    let onWall = matches(wallSet, e);
    const onDoor = matches(doorSet, e);
    if (!onWall && !onDoor) return;
    if (!inClipAll(e)) return;
    // A dashed line is a projection (the floor below, a beam overhead) or, as long
    // as a door is wide and straight, the head of an opening: bridged as a door.
    if (!o.dashed && e.linetype && !CONTINUOUS.test(e.linetype)) {
      const p = e.points;
      let len = 0;
      for (let k = 2; k < p.length; k += 2) len += Math.hypot(p[k] - p[k - 2], p[k + 1] - p[k - 1]) * scale;
      const a = tf(p[0], p[1]), b = tf(p[p.length - 2], p[p.length - 1]);
      const chord = Math.hypot(b[0] - a[0], b[1] - a[1]);
      if (len >= 0.5 && len <= 2.5 && chord >= 0.9 * len) heads.push({ H: a, C: b, width: chord, source: "head", name: null });
      onWall = false;
      if (!onDoor) return;
    }
    // Swings on door layers are doors; a curve on a wall layer is a wall, whatever its radius.
    const sw = onDoor ? swingOf(e, tf) : null;
    if (sw) {
      swings.push({ ...sw, layer: e.path, entity: i });
      return;
    }
    if (!onWall) return;
    const p = e.points;
    for (let k = 2; k < p.length; k += 2) {
      const [ax, ay] = tf(p[k - 2], p[k - 1]), [bx, by] = tf(p[k], p[k + 1]);
      if (ax !== bx || ay !== by) walls.push(ax, ay, bx, by);
    }
  });
  const nWall = walls.length / 4;
  const wallSegs = Float64Array.from(walls);
  const wallGrid = boxGrid(segmentBoxes(wallSegs, nWall), nWall, 1.0);

  // 2. Doors: hinge, closed leaf, width.
  const leaves = []; // straight lines that may be door leaves: door-layer lines and lines of door blocks
  const leafOf = (e) => {
    if (e.type !== "polyline") return;
    for (let k = 2; k < e.points.length; k += 2) {
      const a = tf(e.points[k - 2], e.points[k - 1]), b = tf(e.points[k], e.points[k + 1]);
      leaves.push([a[0], a[1], b[0], b[1]]);
    }
  };
  const blockParts = new Map();
  entities.forEach((e, i) => {
    if (doorOwner[i] >= 0 && doorOwner[i] !== i) {
      if (!blockParts.has(doorOwner[i])) blockParts.set(doorOwner[i], []);
      blockParts.get(doorOwner[i]).push(e);
    } else if (doorOwner[i] < 0 && matches(doorSet, e) && e.type === "polyline" && inClipAll(e)) leafOf(e);
  });
  // How much of the chord H-P is drawn: the open leaf is (as a line or a thin
  // rectangle, often in pieces), the closed position is the empty opening.
  const drawnAlong = (lines, H, P, tol) => {
    let n = 0;
    for (const t of [0.2, 0.35, 0.5, 0.65, 0.8]) {
      const x = H[0] + (P[0] - H[0]) * t, y = H[1] + (P[1] - H[1]) * t;
      if (lines.some(([ax, ay, bx, by]) => toSeg(x, y, ax, ay, bx, by)[0] <= tol)) n += 1;
    }
    return n;
  };
  // Wall lines near the hinge running along a chord: the wall the closed leaf sits in.
  const support = (H, P) => {
    const ux = (P[0] - H[0]) / Math.hypot(P[0] - H[0], P[1] - H[1]), uy = (P[1] - H[1]) / Math.hypot(P[0] - H[0], P[1] - H[1]);
    let along = 0;
    gridQuery(wallGrid, H[0] - 0.4, H[1] - 0.4, H[0] + 0.4, H[1] + 0.4, (s) => {
      const ax = wallSegs[4 * s], ay = wallSegs[4 * s + 1], bx = wallSegs[4 * s + 2], by = wallSegs[4 * s + 3];
      const len = Math.hypot(bx - ax, by - ay);
      if (len < 1e-6 || toSeg(H[0], H[1], ax, ay, bx, by)[0] > 0.4) return;
      if (Math.abs(((bx - ax) * uy - (by - ay) * ux) / len) < 0.15) along += Math.min(len, 1);
    });
    return along;
  };
  const found = [];
  const fromSwing = (f, lines, source, name) => {
    const H = [f.cx, f.cy];
    const tol = Math.max(0.03, 0.06 * f.r);
    const d0 = drawnAlong(lines, H, f.p0, tol), d1 = drawnAlong(lines, H, f.p1, tol);
    let C;
    if (d0 !== d1) C = d0 < d1 ? f.p0 : f.p1;
    else {
      // No leaf drawn: the closed leaf runs along the wall it closes.
      const s0 = support(H, f.p0), s1 = support(H, f.p1);
      C = s0 >= s1 ? f.p0 : f.p1;
    }
    found.push({ H, C, width: f.r, source, name });
  };
  for (const [id, parts] of blockParts) {
    const ins = entities[id];
    if (!inClipAll(ins)) continue;
    const lines = [];
    const fits = [];
    for (const e of parts) {
      const f = swingOf(e, tf);
      if (f) fits.push(f);
      else if (e.type === "polyline") {
        for (let k = 2; k < e.points.length; k += 2) {
          const a = tf(e.points[k - 2], e.points[k - 1]), b = tf(e.points[k], e.points[k + 1]);
          lines.push([a[0], a[1], b[0], b[1]]);
        }
      }
    }
    if (fits.length) {
      for (const f of fits) fromSwing(f, lines, "block", ins.name);
      continue;
    }
    // No swing (folding or sliding door): the block's long axis, in its own frame.
    const c = Math.cos(ins.rotation), s = Math.sin(ins.rotation);
    let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
    const [px, py] = tf(ins.x, ins.y);
    for (const [ax, ay, bx, by] of lines) {
      for (const [x, y] of [[ax, ay], [bx, by]]) {
        const u = (x - px) * c + (y - py) * s, v = -(x - px) * s + (y - py) * c;
        u0 = Math.min(u0, u); u1 = Math.max(u1, u); v0 = Math.min(v0, v); v1 = Math.max(v1, v);
      }
    }
    if (!(u0 < u1)) continue;
    const along = u1 - u0 >= v1 - v0;
    const mu = (u0 + u1) / 2, mv = (v0 + v1) / 2;
    const half = along ? (u1 - u0) / 2 : (v1 - v0) / 2;
    const dir = along ? [c, s] : [-s, c];
    const mx = px + mu * c - mv * s, my = py + mu * s + mv * c;
    if (half * 2 < 0.4 || half * 2 > 3.5) continue;
    found.push({ H: [mx - dir[0] * half, my - dir[1] * half], C: [mx + dir[0] * half, my + dir[1] * half], width: 2 * half, source: "folding", name: ins.name });
  }
  for (const f of swings) fromSwing(f, leaves, "swing", null);
  found.push(...heads);
  // Double doors: two leaves closing on one point.
  const doorsRaw = [];
  const used = new Uint8Array(found.length);
  for (let i = 0; i < found.length; i += 1) {
    if (used[i]) continue;
    const a = found[i];
    let merged = null;
    for (let j = i + 1; j < found.length && !merged; j += 1) {
      const b = found[j];
      if (used[j] || a.source === "folding" || b.source === "folding" || a.source === "head" || b.source === "head") continue;
      if (Math.hypot(a.C[0] - b.C[0], a.C[1] - b.C[1]) > 0.12 + 0.1 * Math.min(a.width, b.width)) continue;
      used[j] = 1;
      merged = { H: a.H, C: b.H, width: a.width + b.width, source: a.source, name: a.name, double: true };
    }
    // The same swing drawn twice (a block and a loose arc on top of it).
    if (!merged && doorsRaw.some((d) => Math.hypot(d.H[0] - a.H[0], d.H[1] - a.H[1]) < 0.05 && Math.hypot(d.C[0] - a.C[0], d.C[1] - a.C[1]) < 0.05)) continue;
    doorsRaw.push(merged || a);
  }
  // Bridge each opening jamb to jamb, on both faces of the wall when they are found.
  const doors = [];
  // A face is the last of the lines a few centimetres apart (plaster, finish, core).
  const face = (px, py, dx, dy) => {
    let t = rayHit(wallGrid, wallSegs, px, py, dx, dy, 0.6);
    while (Number.isFinite(t)) {
      const next = rayHit(wallGrid, wallSegs, px + dx * (t + 1e-4), py + dy * (t + 1e-4), dx, dy, 0.12);
      if (!Number.isFinite(next) || t + next > 0.6) break;
      t += next + 1e-4;
    }
    return t;
  };
  for (const d of doorsRaw) {
    const L = Math.hypot(d.C[0] - d.H[0], d.C[1] - d.H[1]);
    if (L < 0.3) continue;
    const ux = (d.C[0] - d.H[0]) / L, uy = (d.C[1] - d.H[1]) / L, nx = -uy, ny = ux;
    const ta = rayHit(wallGrid, wallSegs, d.H[0], d.H[1], -ux, -uy, 0.6);
    const tb = rayHit(wallGrid, wallSegs, d.C[0], d.C[1], ux, uy, 0.6);
    const ea = Number.isFinite(ta) ? ta : 0.15, eb = Number.isFinite(tb) ? tb : 0.15;
    const A = [d.H[0] - ux * ea, d.H[1] - uy * ea], B = [d.C[0] + ux * eb, d.C[1] + uy * eb];
    // The faces of the wall at the two jambs, from just inside the wall.
    const sides = [];
    for (const [px, py] of [[A[0] - ux * 0.05, A[1] - uy * 0.05], [B[0] + ux * 0.05, B[1] + uy * 0.05]]) {
      const dp = face(px, py, nx, ny), dm = face(px, py, -nx, -ny);
      if (Number.isFinite(dp) && Number.isFinite(dm) && dp + dm <= o.maxWall) sides.push([dp, dm]);
    }
    const ext = 0.05;
    const lines = [];
    const line = (off) => lines.push(A[0] + nx * off - ux * ext, A[1] + ny * off - uy * ext, B[0] + nx * off + ux * ext, B[1] + ny * off + uy * ext);
    let thickness = null, off = 0;
    if (sides.length) {
      const dp = Math.min(...sides.map((q) => q[0])), dm = Math.min(...sides.map((q) => q[1]));
      thickness = dp + dm;
      off = (dp - dm) / 2;
      line(dp);
      line(-dm);
      if (dp > 0.02 && dm > 0.02) line(0);
    } else line(0);
    doors.push({
      a: [A[0] + nx * off, A[1] + ny * off], b: [B[0] + nx * off, B[1] + ny * off],
      mid: [(A[0] + B[0]) / 2 + nx * off, (A[1] + B[1]) / 2 + ny * off], u: [ux, uy], n: [nx, ny],
      width: d.width, thickness, source: d.source, block: d.name ?? null, double: Boolean(d.double), lines,
    });
  }

  // Texts that may name rooms.
  const texts = [];
  const ignore = new Set((o.ignoreTexts || []).map((t) => String(t).trim().toLowerCase()));
  entities.forEach((e) => {
    if (e.type !== "text") return;
    if (textSet && !matches(textSet, e)) return;
    if (ignore.has(e.text.trim().toLowerCase())) return;
    const [x, y] = tf(e.x, e.y);
    if (!inClip(x, y)) return;
    const text = decodeArabicKeys(e.text, o.decode);
    const role = textRole(text);
    if (role) texts.push({ text, role, x, y, h: e.h * scale });
  });
  const roomIndex = (rs) => {
    const boxes = ringBoxes(rs.map((r) => r.ring));
    const grid = boxGrid(boxes, rs.length, 2.0);
    return (x, y) => {
      let best = -1;
      gridQuery(grid, x, y, x, y, (i) => {
        if (x < boxes[4 * i] || x > boxes[4 * i + 2] || y < boxes[4 * i + 1] || y > boxes[4 * i + 3]) return;
        const r = rs[i];
        if (inRing(x, y, r.ring) && !r.holes.some((h) => inRing(x, y, h))) {
          best = i;
          return false;
        }
      });
      return best;
    };
  };

  /** Steps 3 to 5 with the given doors bridged: outline, arrangement, faces, rooms and their names. */
  const trace = (bridged) => {
    const bridges = bridged.flatMap((d) => d.lines);
    // 3. The outline, closed over open facades.
    const barrier = new Float64Array(walls.length + bridges.length);
    barrier.set(wallSegs, 0);
    barrier.set(bridges, walls.length);
    const ta = now();
    const outlineRings = o.close > 0 ? closedOutline(barrier, barrier.length / 4, { close: o.close, minMass: o.minMass }) : [];
    const tb = now();
    const outlineSegs = [];
    for (const ring of outlineRings) {
      for (let i = 0; i < ring.length; i += 1) {
        const p = ring[i], q = ring[(i + 1) % ring.length];
        outlineSegs.push(p[0], p[1], q[0], q[1]);
      }
    }
    const all = new Float64Array(barrier.length + outlineSegs.length);
    all.set(barrier, 0);
    all.set(outlineSegs, barrier.length);
    // 4. The arrangement and its faces.
    const g = arrangement(all, all.length / 4, { snap: o.snap, extend: o.extend });
    const pf = planarFaces(g);
    const tc = now();
    const nOutline = outlineSegs.length / 4;
    const oSegs = Float64Array.from(outlineSegs);
    const oGrid = boxGrid(segmentBoxes(oSegs, nOutline), nOutline, 1.0);
    const onOutline = (x, y) => {
      let hit = false;
      gridQuery(oGrid, x - 1e-4, y - 1e-4, x + 1e-4, y + 1e-4, (k) => {
        if (toSeg(x, y, oSegs[4 * k], oSegs[4 * k + 1], oSegs[4 * k + 2], oSegs[4 * k + 3])[0] < 1e-4) {
          hit = true;
          return false;
        }
      });
      return hit;
    };
    // Faces: rooms, walls, and strips between the outline and the facade (outside).
    const faces = pf.faces.map((fc) => {
      const holeArea = fc.holes.reduce((sum, h) => sum + Math.abs(areaOf(h)), 0);
      const area = fc.area - holeArea;
      const per = perimeterOf(fc.ring) + fc.holes.reduce((sum, h) => sum + perimeterOf(h), 0);
      const width = per > 0 ? (2 * area) / per : 0;
      const compact = per > 0 ? (4 * Math.PI * area) / (per * per) : 0;
      const room = area >= o.minArea && width >= o.minWidth && !(width < 0.6 && compact < 0.1);
      return { ring: fc.ring, holes: fc.holes, area, width, room, outside: false };
    });
    if (nOutline) {
      for (const fc of faces) {
        if (fc.width >= o.minWidth) continue;
        if (fc.ring.some(([x, y]) => onOutline(x, y))) {
          fc.outside = true;
          fc.room = false;
        }
      }
    }
    const faceBoxes = ringBoxes(faces.map((fc) => fc.ring));
    const faceGrid = boxGrid(faceBoxes, faces.length, 2.0);
    const faceAt = (x, y) => {
      let best = -1, ba = Infinity;
      gridQuery(faceGrid, x, y, x, y, (i) => {
        const fc = faces[i];
        if (fc.area >= ba || x < faceBoxes[4 * i] || x > faceBoxes[4 * i + 2] || y < faceBoxes[4 * i + 1] || y > faceBoxes[4 * i + 3]) return;
        if (inRing(x, y, fc.ring) && !fc.holes.some((h) => inRing(x, y, h))) [best, ba] = [i, fc.area];
      });
      return best;
    };
    // Rooms, with small holes (columns) filled.
    const rooms = [];
    faces.forEach((fc, i) => {
      if (!fc.room) return;
      const holes = fc.holes.filter((h) => Math.abs(areaOf(h)) >= o.holeArea).map((h) => (areaOf(h) > 0 ? h.slice().reverse() : h));
      rooms.push({ ring: cleanRing(fc.ring, 1e-4), holes: holes.map((h) => cleanRing(h, 1e-4)), face: i, labels: [], cuts: [] });
    });
    // 5. Names: texts inside rooms.
    const roomAt = roomIndex(rooms);
    for (const t of texts) {
      const r = roomAt(t.x, t.y);
      if (r >= 0) rooms[r].labels.push(t);
    }
    const isOutside = (x, y) => {
      const k = faceAt(x, y);
      return k < 0 || faces[k].outside;
    };
    return { bridges, outlineRings, nOutline, g, pf, faces, faceAt, isOutside, rooms, roomAt, ms: { outline: tb - ta, faces: tc - tb } };
  };
  // The room on one side of a door: probes across the opening, up to 1 m.
  const sideOf = (T, d, sgn) => {
    for (let k = 1; k <= 20; k += 1) {
      const x = d.mid[0] + d.n[0] * 0.05 * k * sgn, y = d.mid[1] + d.n[1] * 0.05 * k * sgn;
      const r = T.roomAt(x, y);
      if (r >= 0) return ["room", r];
      if (T.isOutside(x, y)) return ["outside", -1];
    }
    return ["unknown", -1];
  };
  const t1 = now();
  let active = doors;
  let T = trace(active);
  // An opening with no leaf between a named room and a small unnamed space
  // that serves it alone (an entrance lobby, a wardrobe recess: the space opens
  // onto no other named room) does not divide a room: trace again without it.
  let merged = 0;
  if (o.mergeOpenings) {
    const named = (r) => r.labels.length > 0;
    const pairs = doors.map((d) => [sideOf(T, d, 1)[1], sideOf(T, d, -1)[1]]);
    const reach = new Map();
    pairs.forEach(([a, b]) => {
      if (a < 0 || b < 0 || a === b) return;
      if (!reach.has(a)) reach.set(a, new Set());
      if (!reach.has(b)) reach.set(b, new Set());
      reach.get(a).add(b);
      reach.get(b).add(a);
    });
    const drop = new Set();
    doors.forEach((d, k) => {
      if (d.source !== "head") return;
      const [a, b] = pairs[k];
      if (a < 0 || b < 0 || a === b) return;
      const ra = T.rooms[a], rb = T.rooms[b];
      if (named(ra) === named(rb)) return;
      const [si, ni] = named(ra) ? [b, a] : [a, b];
      if (Math.abs(areaOf(T.rooms[si].ring)) > o.mergeArea) return;
      if ([...reach.get(si)].some((r) => r !== ni && named(T.rooms[r]))) return;
      drop.add(d);
    });
    if (drop.size) {
      merged = drop.size;
      active = doors.filter((d) => !drop.has(d));
      T = trace(active);
    }
  }
  const { faces, isOutside, pf, g, nOutline, outlineRings } = T;
  const bridges = T.bridges;
  let rooms = T.rooms;
  let roomAt = T.roomAt;
  const t2 = t1 + T.ms.outline, t3 = now();

  // 6. Rooms holding two names far apart are cut where a wall end allows it.
  let splits = 0;
  if (o.split) {
    const next = [];
    for (const r of rooms) {
      const parts = splitByNames(r, o);
      splits += parts.length - 1;
      next.push(...parts);
    }
    rooms = next;
    roomAt = roomIndex(rooms);
  }

  // 7. Grow to the wall centre-lines; neighbours from the same rays.
  const live = pf.live;
  const eSegs = new Float64Array(4 * live.length);
  live.forEach((e, k) => eSegs.set([g.vx[g.eu[e]], g.vy[g.eu[e]], g.vx[g.ev[e]], g.vy[g.ev[e]]], 4 * k));
  const eGrid = boxGrid(segmentBoxes(eSegs, live.length), live.length, 1.0);
  const shared = new Map(); // "i,j" -> metres of boundary seen across a wall or a line
  const addShared = (i, j, len) => {
    if (i === j) return;
    const k = i < j ? `${i},${j}` : `${j},${i}`;
    shared.set(k, (shared.get(k) || 0) + len);
  };
  const probe = (self, x, y, nx, ny) => {
    const q0 = 1e-3;
    let r = roomAt(x + nx * q0, y + ny * q0);
    if (r >= 0) return [0, r];
    if (isOutside(x + nx * q0, y + ny * q0)) return [0, -1];
    let cur = q0;
    while (cur < o.maxWall) {
      const t = rayHit(eGrid, eSegs, x + nx * cur, y + ny * cur, nx, ny, o.maxWall - cur + q0);
      if (!Number.isFinite(t)) return [0, -2];
      cur += t + q0;
      const px = x + nx * cur, py = y + ny * cur;
      r = roomAt(px, py);
      if (r >= 0) return [(cur - q0) / 2, r];
      if (isOutside(px, py)) return [(cur - q0) / 2, -1];
    }
    return [0, -2];
  };
  rooms.forEach((room, self) => {
    room.inner = room.ring;
    room.innerHoles = room.holes;
    const grown = [];
    for (const ring of [room.ring, ...room.holes]) {
      const n = ring.length;
      const offs = new Float64Array(n);
      for (let i = 0; i < n; i += 1) {
        const p = ring[i], q = ring[(i + 1) % n];
        const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
        if (len < 1e-6) continue;
        const ux = (q[0] - p[0]) / len, uy = (q[1] - p[1]) / len, nx = uy, ny = -ux;
        const ts = len < 0.3 ? [0.5] : [0.2, 0.5, 0.8];
        const vals = [];
        for (const t of ts) {
          const [d, other] = probe(self, p[0] + ux * len * t, p[1] + uy * len * t, nx, ny);
          vals.push(d);
          if (other >= 0) addShared(self, other, len / ts.length);
        }
        vals.sort((a, b) => a - b);
        offs[i] = o.grow ? vals[Math.floor(vals.length / 2)] : 0;
      }
      grown.push(offsetRing(ring, offs));
    }
    const outer = grown[0];
    const ok = outer && areaOf(outer) > 0 && !selfCrossing(outer);
    room.ring = ok ? cleanRing(simplifyClosed(outer, 0.02), 1e-3) : room.inner;
    room.holes = grown.slice(1).map((h, k) => (h && areaOf(h) < 0 && Math.abs(areaOf(h)) > 0.25 && !selfCrossing(h) ? cleanRing(h, 1e-3) : room.innerHoles[k])).filter(Boolean);
  });
  const t4 = now();

  // 8. Doors join the rooms on either side of the opening.
  const doorList = [];
  for (const d of active) {
    const [ka, ra] = sideOf({ roomAt, isOutside }, d, 1), [kb, rb] = sideOf({ roomAt, isOutside }, d, -1);
    if (ra >= 0 && ra === rb) {
      issues.push({ type: "door-inside-room", message: "A door with the same room on both sides: a wall the drawing leaves open, or a room to split here.", at: d.mid, rooms: [ra] });
    }
    if (ra < 0 && rb < 0) {
      issues.push({ type: "door-no-room", message: "A door with no room on either side.", at: d.mid, rooms: [] });
      continue;
    }
    // A second leaf of the same opening.
    const dup = doorList.find((x) => Math.hypot(x.mid[0] - d.mid[0], x.mid[1] - d.mid[1]) < 0.6 && ((x.ra === ra && x.rb === rb) || (x.ra === rb && x.rb === ra)));
    if (dup) continue;
    doorList.push({ ...d, ra, rb, sa: ka, sb: kb });
  }
  // Cuts made in step 6 are openings between the rooms they separate.
  const openings = [];
  rooms.forEach((r) => {
    for (const c of r.cuts) {
      const mid = [(c[0][0] + c[1][0]) / 2, (c[0][1] + c[1][1]) / 2];
      const len = Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]);
      const n = [-(c[1][1] - c[0][1]) / len, (c[1][0] - c[0][0]) / len];
      const a = roomAt(mid[0] + n[0] * 0.05, mid[1] + n[1] * 0.05), b = roomAt(mid[0] - n[0] * 0.05, mid[1] - n[1] * 0.05);
      if (a >= 0 && b >= 0 && a !== b && !openings.some((x) => Math.hypot(x.mid[0] - mid[0], x.mid[1] - mid[1]) < 0.1)) {
        openings.push({ mid, a: c[0], b: c[1], u: [-n[1], n[0]], width: len, ra: a, rb: b });
      }
    }
  });

  // Ids, labels and classes; rooms ordered top to bottom, left to right.
  const order = rooms.map((r, i) => [i, insidePoint(r.inner, r.innerHoles)]).sort((a, b) => Math.round(b[1][1]) - Math.round(a[1][1]) || a[1][0] - b[1][0]);
  const idOf = new Map(order.map(([i], k) => [i, `R${k + 1}`]));
  const fixtures = [];
  entities.forEach((e) => {
    if (e.type !== "insert" || !fixtureRx?.test(e.name)) return;
    const [x, y] = tf((e.bbox[0] + e.bbox[2]) / 2, (e.bbox[1] + e.bbox[3]) / 2);
    if (inClip(x, y)) fixtures.push([x, y]);
  });
  const fixtureCount = new Int32Array(rooms.length);
  for (const [x, y] of fixtures) {
    const r = roomAt(x, y);
    if (r >= 0) fixtureCount[r] += 1;
  }
  const outRooms = order.map(([i, at]) => {
    const r = rooms[i];
    const labels = r.labels.slice().sort((a, b) => (a.role === b.role ? b.h - a.h : a.role === "name" ? -1 : 1));
    const names = labels.filter((t) => t.role === "name").map((t) => t.text);
    const polygon = r.ring.map(([x, y]) => [round3(x), round3(y)]);
    const holes = r.holes.map((h) => h.map(([x, y]) => [round3(x), round3(y)]));
    const area = Math.abs(areaOf(polygon)) - holes.reduce((s, h) => s + Math.abs(areaOf(h)), 0);
    let cls = roomClassFromNames(labels.map((t) => t.text)), why = cls ? "name" : null;
    if (!cls && fixtureCount[i] > 0) [cls, why] = ["Toilet", "fixture"];
    if (!cls) {
      const shape = rectangleSides(r.inner);
      if (shape && shape[0] < 2.4 && shape[1] > 3.5 * shape[0]) [cls, why] = ["Corridor", "shape"];
    }
    return {
      id: idOf.get(i), polygon, holes, area: round3(area), innerArea: round3(Math.abs(areaOf(r.inner)) - r.innerHoles.reduce((s, h) => s + Math.abs(areaOf(h)), 0)),
      inner: r.inner.map(([x, y]) => [round3(x), round3(y)]),
      label: names[0] ?? labels[0]?.text ?? null, labels: labels.map((t) => t.text),
      class: cls, classSource: why, fixtures: fixtureCount[i], centroid: insidePoint(r.ring, r.holes).map(round3), at: at.map(round3),
    };
  });
  const byIndex = new Map(order.map(([i], k) => [i, outRooms[k]]));
  const outDoors = [];
  doorList.forEach((d, k) => {
    const A = d.ra >= 0 ? byIndex.get(d.ra).id : null, B = d.rb >= 0 ? byIndex.get(d.rb).id : null;
    outDoors.push({
      id: `D${k + 1}`, kind: d.source === "head" ? "opening" : "door", position: d.mid.map(round3), direction: d.u.map(round3), width: round3(d.width),
      rooms: [A, B], sides: [d.sa, d.sb], wall: d.thickness === null ? null : round3(d.thickness), source: d.source, block: d.block,
      jambs: [d.a.map(round3), d.b.map(round3)],
    });
  });
  openings.forEach((d, k) => {
    outDoors.push({
      id: `O${k + 1}`, kind: "opening", position: d.mid.map(round3), direction: d.u.map(round3), width: round3(d.width),
      rooms: [byIndex.get(d.ra).id, byIndex.get(d.rb).id], sides: ["room", "room"], wall: null, source: "split", block: null,
      jambs: [d.a.map(round3), d.b.map(round3)],
    });
  });
  // Neighbours: rooms seen across one wall or line, over at least 0.3 m.
  for (const r of outRooms) r.neighbours = [];
  for (const [k, len] of shared) {
    if (len < 0.3) continue;
    const [i, j] = k.split(",").map(Number);
    byIndex.get(i).neighbours.push(byIndex.get(j).id);
    byIndex.get(j).neighbours.push(byIndex.get(i).id);
  }
  const idNum = (s) => Number(s.slice(1));
  for (const r of outRooms) r.neighbours.sort((a, b) => idNum(a) - idNum(b));

  // Issues for the reviewer.
  for (const r of outRooms) {
    const names = new Set(r.labels.filter((t) => textRole(t) === "name").map(normaliseName));
    if (!r.labels.length) issues.push({ type: "no-label", message: "A room with no name inside.", at: r.at, rooms: [r.id] });
    else if (names.size > 1) issues.push({ type: "several-labels", message: "A room with several names: rooms merged where the drawing leaves no wall to cut along?", at: r.at, rooms: [r.id] });
    if (r.area > 150 && names.size !== 1) issues.push({ type: "large-room", message: `A ${Math.round(r.area)} m² room: check that it is one room.`, at: r.at, rooms: [r.id] });
    if (r.area < 1.5) issues.push({ type: "small-room", message: "A room under 1.5 m²: a shaft, or a gap between walls?", at: r.at, rooms: [r.id] });
  }
  for (const d of outDoors) {
    if (d.kind === "door" && d.sides.includes("unknown")) issues.push({ type: "door-one-side", message: "A door with a room on one side only; the other side is inside the walls.", at: d.position, rooms: d.rooms.filter(Boolean) });
  }
  const unplaced = texts.filter((t) => t.role === "name" && roomAt(t.x, t.y) < 0 && !isOutside(t.x, t.y)).length;
  if (unplaced) issues.push({ type: "names-outside-rooms", message: `${unplaced} room names fall on walls or in spaces too small to be rooms.`, at: null, rooms: [] });
  if (!nWall) issues.push({ type: "no-walls", message: "No linework on the wall layers inside the sheet.", at: null, rooms: [] });

  const outline = outlineRings.filter((r) => areaOf(r) > 0).sort((a, b) => areaOf(b) - areaOf(a))[0] || null;
  return {
    rooms: outRooms,
    doors: outDoors,
    outline: outline ? outline.map(([x, y]) => [round3(x), round3(y)]) : null,
    issues,
    stats: {
      wallSegments: nWall, doorsFound: doorsRaw.length, bridges: bridges.length / 4, outlineSegments: nOutline,
      vertices: g.vx.length, edges: g.eu.length, pruned: pf.pruned, extended: g.moved, faces: faces.length,
      rooms: outRooms.length, splits, merged, texts: texts.length,
      ms: { walls: Math.round(t1 - t0), outline: Math.round(t2 - t1), faces: Math.round(t3 - t2), rooms: Math.round(t4 - t3), total: Math.round(now() - t0) },
    },
    settings: { scale, origin: [ox, oy], clip, snap: o.snap, extend: o.extend, minArea: o.minArea, minWidth: o.minWidth, close: o.close, grow: o.grow, split: o.split },
    // The linework read as walls and the door bridges, for showing under the draft.
    lines: o.keepLines ? { walls: wallSegs, bridges: Float64Array.from(bridges) } : undefined,
  };
}

const CONTINUOUS = /^(CONTINUOUS|BYLAYER|BYBLOCK|SOLID)$/i;
const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const round3 = (v) => Math.round(v * 1000) / 1000;
const normaliseName = (s) => s.toLowerCase().replace(/[^a-z؀-ۿ]+/g, "");

/** The sides of a ring's minimum-area bounding rectangle (over its edge directions), shorter first. */
function rectangleSides(ring) {
  let best = null;
  for (let i = 0; i < ring.length; i += 1) {
    const p = ring[i], q = ring[(i + 1) % ring.length];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (len < 1e-6) continue;
    const ux = (q[0] - p[0]) / len, uy = (q[1] - p[1]) / len;
    let a0 = Infinity, a1 = -Infinity, b0 = Infinity, b1 = -Infinity;
    for (const [x, y] of ring) {
      const a = x * ux + y * uy, b = -x * uy + y * ux;
      a0 = Math.min(a0, a); a1 = Math.max(a1, a); b0 = Math.min(b0, b); b1 = Math.max(b1, b);
    }
    const s = [a1 - a0, b1 - b0].sort((m, n) => m - n);
    if (!best || s[0] * s[1] < best[0] * best[1]) best = s;
  }
  return best;
}

/**
 * A ring with each edge moved out (to its right: the room is on the left) by
 * its own distance: neighbouring edges meet at their lines' crossing, or,
 * where they run straight on or meet at a sharp angle, by a short step.
 */
function offsetRing(ring, offs) {
  const n = ring.length;
  if (n < 3) return null;
  const dir = [];
  for (let i = 0; i < n; i += 1) {
    const p = ring[i], q = ring[(i + 1) % n];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1;
    dir.push([(q[0] - p[0]) / len, (q[1] - p[1]) / len]);
  }
  const out = [];
  for (let i = 0; i < n; i += 1) {
    const a = (i - 1 + n) % n, v = ring[i];
    const da = dir[a], db = dir[i];
    const oa = offs[a], ob = offs[i];
    const pa = [v[0] + da[1] * oa, v[1] - da[0] * oa], pb = [v[0] + db[1] * ob, v[1] - db[0] * ob];
    const cross = da[0] * db[1] - da[1] * db[0];
    if (Math.abs(cross) < 0.05) {
      out.push(pa);
      if (Math.hypot(pa[0] - pb[0], pa[1] - pb[1]) > 1e-4) out.push(pb);
      continue;
    }
    const t = ((pb[0] - pa[0]) * db[1] - (pb[1] - pa[1]) * db[0]) / cross;
    const x = [pa[0] + da[0] * t, pa[1] + da[1] * t];
    if (Math.hypot(x[0] - v[0], x[1] - v[1]) > 4 * Math.max(oa, ob) + 0.01) out.push(pa, pb);
    else out.push(x);
  }
  return cleanRing(out, 1e-4);
}

/**
 * A room cut into rooms that each hold one name (group of nearby texts),
 * along the shortest admissible cuts: from a re-entrant corner of the room
 * (a wall end) to another corner, or along one of its edges' lines to the far
 * side. Returns the parts, each with the cuts that bound it.
 */
function splitByNames(room, o, depth = 0) {
  const groups = nameGroups(room.labels);
  if (groups.length < 2 || depth > 6) return [room];
  const ring = room.ring;
  const n = ring.length;
  if (n < 4 || n > 800) return [room];
  const edges = [];
  for (const r of [ring, ...room.holes]) for (let i = 0; i < r.length; i += 1) edges.push([r[i], r[(i + 1) % r.length]]);
  const crossesEdge = (p, q) => edges.some(([a, b]) => segmentsCross(p[0], p[1], q[0], q[1], a[0], a[1], b[0], b[1]));
  const inside = (x, y) => inRing(x, y, ring) && !room.holes.some((h) => inRing(x, y, h));
  const reflex = [];
  for (let i = 0; i < n; i += 1) {
    const a = ring[(i - 1 + n) % n], b = ring[i], c = ring[(i + 1) % n];
    if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) < -1e-9) reflex.push(i);
  }
  let best = null;
  const consider = (i, j, X, k) => {
    // Cut from vertex i to vertex j, or to point X on edge k (k, k+1).
    const p = ring[i], q = X || ring[j];
    const len = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (len < 0.3 || len > o.maxCut || (best && len >= best.len)) return;
    const mx = (p[0] + q[0]) / 2, my = (p[1] + q[1]) / 2;
    if (!inside(mx, my) || crossesEdge(p, q)) return;
    let A, B;
    if (X) {
      // Ring with X inserted after vertex k.
      const r2 = [...ring.slice(0, k + 1), X, ...ring.slice(k + 1)];
      const ii = i <= k ? i : i + 1, jj = k + 1;
      [A, B] = splitRing(r2, ii, jj);
    } else [A, B] = splitRing(ring, i, j);
    if (!A || !B) return;
    const aA = areaOf(A), aB = areaOf(B);
    if (aA < o.minArea || aB < o.minArea) return;
    if ((2 * aA) / perimeterOf(A) < o.minWidth || (2 * aB) / perimeterOf(B) < o.minWidth) return;
    let inA = 0, inB = 0;
    for (const g of groups) {
      const a = g.filter((t) => inRing(t.x, t.y, A)).length;
      if (a * 2 >= g.length) inA += 1;
      else inB += 1;
    }
    if (!inA || !inB) return;
    best = { len, A, B, cut: [p, q] };
  };
  for (const i of reflex) {
    for (let j = 0; j < n; j += 1) if (j !== i && j !== (i + 1) % n && j !== (i - 1 + n) % n) consider(i, j, null, -1);
    // Along the two edges meeting at the corner, carried on into the room.
    for (const [from, to] of [[(i - 1 + n) % n, i], [(i + 1) % n, i]]) {
      const len = Math.hypot(ring[to][0] - ring[from][0], ring[to][1] - ring[from][1]);
      if (len < 1e-9) continue;
      const ux = (ring[to][0] - ring[from][0]) / len, uy = (ring[to][1] - ring[from][1]) / len;
      let bt = o.maxCut, bk = -1;
      for (let k = 0; k < n; k += 1) {
        if (k === i || (k + 1) % n === i) continue;
        const a = ring[k], b = ring[(k + 1) % n];
        const sx = b[0] - a[0], sy = b[1] - a[1];
        const den = ux * sy - uy * sx;
        if (Math.abs(den) < 1e-12) continue;
        const t = ((a[0] - ring[i][0]) * sy - (a[1] - ring[i][1]) * sx) / den;
        const u = ((a[0] - ring[i][0]) * uy - (a[1] - ring[i][1]) * ux) / den;
        if (t > 1e-6 && t < bt && u > 1e-6 && u < 1 - 1e-6) [bt, bk] = [t, k];
      }
      if (bk >= 0) consider(i, -1, [ring[i][0] + ux * bt, ring[i][1] + uy * bt], bk);
    }
  }
  if (!best) return [room];
  const holesOf = (R) => room.holes.filter((h) => inRing(h[0][0], h[0][1], R));
  const make = (R) => ({ ring: R, holes: holesOf(R), face: room.face, labels: room.labels.filter((t) => inRing(t.x, t.y, R)), cuts: [...room.cuts, best.cut] });
  return [...splitByNames(make(best.A), o, depth + 1), ...splitByNames(make(best.B), o, depth + 1)];
}

/** The two rings a chord from vertex i to vertex j cuts a ring into. */
function splitRing(ring, i, j) {
  const n = ring.length;
  if (i === j) return [null, null];
  const A = [], B = [];
  for (let k = i; ; k = (k + 1) % n) {
    A.push(ring[k]);
    if (k === j) break;
  }
  for (let k = j; ; k = (k + 1) % n) {
    B.push(ring[k]);
    if (k === i) break;
  }
  return A.length >= 3 && B.length >= 3 ? [A, B] : [null, null];
}

/** Name texts grouped by distance (1.5 m), groups with the same words merged. */
function nameGroups(labels) {
  const names = labels.filter((t) => t.role === "name");
  const groups = [];
  for (const t of names) {
    const near = groups.filter((g) => g.some((u) => Math.hypot(u.x - t.x, u.y - t.y) <= 1.5));
    const merged = near.flat().concat(t);
    for (const g of near) groups.splice(groups.indexOf(g), 1);
    groups.push(merged);
  }
  // A name written twice in one room (a long corridor) is one room.
  const out = [];
  for (const g of groups) {
    const key = [...new Set(g.map((t) => normaliseName(t.text)))].sort().join("|");
    const same = out.find((x) => x.key === key);
    if (same) same.g.push(...g);
    else out.push({ key, g });
  }
  return out.map((x) => x.g);
}

// ===================================================================== for an AI suggester

/**
 * A storey's draft as compact JSON for an optional AI step that suggests
 * names, classes, merges and splits (its instructions live server-side; the
 * Worker checks every suggestion against this summary):
 *
 *   {storey?, rooms: [{id, area, label, labels, centroid, neighbours, doors}]}
 *
 *   id          the draft room's id (a string);
 *   area        m², to 0.1;
 *   label       the name the draft gave it, or null;
 *   labels      every text found inside it, as written (decoded from the
 *               Arabic keyboard font when draftRooms decoded it);
 *   centroid    [x, y] in metres, to 0.1 (the only geometry);
 *   neighbours  ids of the rooms sharing a wall or a boundary with it,
 *               whether or not a door joins them;
 *   doors       ids of the rooms it opens to through a door or an opening.
 *
 * @param {object} draft  draftRooms' result (or a reviewed copy of it)
 * @param {{storey?: string}} opts
 */
export function draftSummary(draft, { storey } = {}) {
  const r1 = (v) => Math.round(v * 10) / 10;
  const through = new Map(draft.rooms.map((r) => [r.id, new Set()]));
  for (const d of draft.doors || []) {
    const [a, b] = d.rooms || [];
    if (a && b && a !== b && through.has(a) && through.has(b)) {
      through.get(a).add(b);
      through.get(b).add(a);
    }
  }
  const order = (ids) => [...ids].sort((p, q) => String(p).localeCompare(String(q), "en", { numeric: true }));
  const rooms = draft.rooms.map((r) => {
    const ring = r.polygon;
    const c = centroidOf(ring);
    const neighbours = new Set([...(r.neighbours || []), ...through.get(r.id)]);
    neighbours.delete(r.id);
    return {
      id: String(r.id),
      area: r1(r.area ?? Math.abs(areaOf(ring))),
      label: r.label ?? null,
      labels: [...(r.labels || [])],
      centroid: [r1(c[0]), r1(c[1])],
      neighbours: order(neighbours).map(String),
      doors: order(through.get(r.id)).map(String),
    };
  });
  return storey === undefined ? { rooms } : { storey: String(storey), rooms };
}

// ===================================================================== the LOD1 model

/**
 * The LOD1 model of lod1.js from per-storey drafts and the storeys' levels.
 *
 *   drafts  [{level, draft}], or {levelId: draft}, or drafts in the order
 *           of `levels`; a draft is draftRooms' result, possibly reviewed:
 *           a room may carry name, class, ffl and height of its own, and a
 *           room or door may be removed or added (rooms by id, doors with
 *           position, width, kind and rooms).
 *   levels  [{id, name, ffl, height}] (metres).
 *
 * Rooms become prisms on their level: id "<level>:<room id>", the name the
 * review gave or the draft's label, the class the review gave or the one the
 * names give (Unit when none), footprint anticlockwise and holes clockwise.
 * Doors are placed on the shared boundary of their two rooms (lod1.js
 * doorOnBoundary) at the sill of the lower floor; a door to the outside on
 * the entrance level (the level nearest ±0.00, or opts.exitLevels) is an
 * "exit"; a door with a side the draft could not place is left out, so it
 * does not become a false exit. Stairs and lifts are linked between
 * consecutive levels where rooms of the same kind overlap in plan (flagged in
 * meta.warnings: a drawing's stair shows on both floors, but which flight
 * joins which is a judgement).
 *
 * @param {object[]|object} drafts
 * @param {{id: string, name?: string, ffl: number, height: number}[]} levels
 * @param {{name?: string, north?: number|number[], exitLevels?: string[]}} opts
 */
export function toLod1Model(drafts, levels, { name = null, north = null, exitLevels = null } = {}) {
  const lv = (levels || []).map((l) => ({ id: String(l.id), name: l.name ?? String(l.id), ffl: Number(l.ffl) || 0, height: Number(l.height) > 0 ? Number(l.height) : 3 }));
  lv.sort((a, b) => a.ffl - b.ffl);
  const byLevel = new Map(lv.map((l) => [l.id, l]));
  // The drafts, keyed by level.
  let list;
  if (Array.isArray(drafts)) {
    list = drafts.map((d, i) => (d && d.draft ? { level: String(d.level), draft: d.draft } : { level: String(d?.level ?? levels?.[i]?.id), draft: d }));
  } else list = Object.entries(drafts || {}).map(([level, draft]) => ({ level, draft }));
  const warnings = [];
  const rooms = [];
  const doors = [];
  const entrance = exitLevels ? new Set(exitLevels.map(String)) : new Set(lv.length ? [lv.reduce((a, b) => (Math.abs(b.ffl) < Math.abs(a.ffl) ? b : a)).id] : []);
  let boundary = null;
  for (const { level, draft } of list) {
    const L = byLevel.get(level);
    if (!L) {
      warnings.push(`A draft for level ${level}, which the levels do not list; left out.`);
      continue;
    }
    if (!draft?.rooms) continue;
    const b = draft.settings?.grow === false ? "inner" : "shared";
    if (boundary && boundary !== b) warnings.push("Some storeys were drafted on the walls' centre lines and some on their inner faces.");
    boundary ||= b;
    const ids = new Map();
    for (const r of draft.rooms) {
      const ring = (r.polygon || r.footprint || []).map(([x, y]) => [x, y]);
      if (ring.length < 3 || Math.abs(areaOf(ring)) < 1e-6) {
        warnings.push(`Room ${r.id} on ${level} has no area; left out.`);
        continue;
      }
      const footprint = areaOf(ring) > 0 ? ring : ring.reverse();
      const holes = (r.holes || []).filter((h) => h.length >= 3).map((h) => {
        const hh = h.map(([x, y]) => [x, y]);
        return areaOf(hh) < 0 ? hh : hh.reverse();
      });
      const labels = r.labels || [];
      const cls = ROOM_CLASSES.includes(r.class) ? r.class : roomClassFromNames(labels) || "Unit";
      const id = `${level}:${r.id}`;
      ids.set(r.id, id);
      rooms.push({
        id, name: r.name ?? r.label ?? String(r.id), level, ffl: Number.isFinite(r.ffl) ? r.ffl : L.ffl, height: r.height > 0 ? r.height : L.height,
        class: cls, area: Math.abs(areaOf(footprint)) - holes.reduce((s, h) => s + Math.abs(areaOf(h)), 0), footprint, holes,
        walkable: r.walkable === false ? false : isWalkableClass(cls),
      });
    }
    const levelRooms = rooms.filter((r) => r.level === level);
    const byId = new Map(levelRooms.map((r) => [r.id, r]));
    for (const d of draft.doors || []) {
      const [a, b] = d.rooms || [];
      const ra = a ? byId.get(ids.get(a)) : null, rb = b ? byId.get(ids.get(b)) : null;
      if ((a && !ra) || (b && !rb) || (!ra && !rb)) continue; // a removed room
      const sides = d.sides || [a ? "room" : "outside", b ? "room" : "outside"];
      if (sides.includes("unknown") && !(ra && rb)) {
        warnings.push(`Door ${d.id} on ${level} has a side in no room; left out (it would be a false exit).`);
        continue;
      }
      const at = d.position || [0, 0];
      const placed = doorOnBoundary(ra, rb, [at[0], at[1]], d.width || 0.9, levelRooms);
      const outside = !ra || !rb;
      const kind = d.kind === "opening" ? "opening" : outside && entrance.has(level) ? "exit" : "door";
      doors.push({
        id: `${level}:${d.id}`,
        position: [...(placed?.at || [at[0], at[1]]), Math.min(ra?.ffl ?? Infinity, rb?.ffl ?? Infinity)],
        direction: placed?.direction ?? d.direction ?? null,
        level, kind,
        rooms: [ra?.id ?? null, rb?.id ?? null],
        width: d.width > 0 ? d.width : null,
        assumed: Boolean(d.assumed),
      });
    }
  }
  // Stairs and lifts: rooms of one kind over one another on consecutive levels.
  const links = [];
  const kindOf = (r) => (r.class === "Stair" ? "stair" : r.class === "Ramp" ? "ramp" : r.class === "Shaft" && /lift|elevator|مصعد|aufzug|ascens/i.test(r.name) ? "lift" : null);
  for (let k = 0; k + 1 < lv.length; k += 1) {
    const lower = rooms.filter((r) => r.level === lv[k].id && kindOf(r)), upper = rooms.filter((r) => r.level === lv[k + 1].id && kindOf(r));
    for (const a of lower) {
      for (const b of upper) {
        if (kindOf(a) !== kindOf(b)) continue;
        const pa = insidePoint(a.footprint, a.holes), pb = insidePoint(b.footprint, b.holes);
        const over = inRing(pa[0], pa[1], b.footprint) ? pa : inRing(pb[0], pb[1], a.footprint) ? pb : null;
        if (!over) continue;
        links.push({ id: `${a.id}~${b.id}`, kind: kindOf(a), rooms: [a.id, b.id], position: over, width: null, footprint: a.footprint });
      }
    }
  }
  if (links.length) warnings.push(`${links.length} stair and lift links joined by rooms of one kind over one another; check them.`);
  const kinds = {};
  for (const d of doors) kinds[d.kind] = (kinds[d.kind] || 0) + 1;
  for (const l of links) kinds[l.kind] = (kinds[l.kind] || 0) + 1;
  return {
    levels: lv,
    rooms,
    doors,
    links,
    meta: {
      source: "dxf-draft", name, north, boundary: boundary || "shared",
      counts: { levels: lv.length, rooms: rooms.length, doors: doors.length, links: links.length, kinds },
      warnings,
    },
  };
}
