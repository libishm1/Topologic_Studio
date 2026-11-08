/**
 * Where can smoke leave a room other than through its doors? The fire grid
 * of a model is built with every door closed (each leaf solid in its
 * opening) and the windows closed, and the open cells are flooded
 * (6-connected, as gas moves between faces) from the middle of a room. A
 * room the flood reaches is joined to the fire's room through something the
 * grid leaves open: a gap in a wall, an opening without a door, a shaft, an
 * element type the fire grid does not take as solid. For each joined room
 * the tool gives the first open cells outside the fire's room on the way
 * (where the air leaves it), and which IFC element types lie within a cell
 * of them but are not in the fire grid.
 *
 *   node tools/smoke-path-check.mjs [model.ifc ...] [--cells 0.27,0.34] [--rooms 3] [--json out.json]
 *
 * Without models, the Duplex and Vienna examples.
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { extract } from "./ifc-extract.mjs";
import { buildScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { SOLID, createSolver } from "../topologicpy-web-frontend/src/core/flamescapes.js";
import { roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.join(HERE, "../topologicpy-web-frontend/public/examples");
const require = createRequire(import.meta.url);
const WebIFC = require("../topologicpy-web-frontend/node_modules/web-ifc/web-ifc-api-node.js");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const cells = flag("--cells", "0.27,0.34").split(",").map(Number);
const roomCount = Number(flag("--rooms", 3));
// Rooms to start from by name (comma-separated, e.g. Zimmer,Bad): the first of each name, instead of the largest rooms.
const roomNames = flag("--names", "").split(",").map((s) => s.trim()).filter(Boolean);
// IFC types to add to the fire grid as solid, to test whether leaving them out opens a path (e.g. IFCBEAM,IFCCOVERING).
const addTypes = flag("--add", "").split(",").map((s) => s.trim().toUpperCase()).filter(Boolean);

/** The meshes of `types`, in the viewer frame as ifc-extract's (for --add). */
async function meshesOfTypes(file, types) {
  if (!types.length) return [];
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(file)), { COORDINATE_TO_ORIGIN: true });
  const meshes = [];
  api.StreamAllMeshesWithTypes(model, types.map((t) => WebIFC[t]).filter((t) => typeof t === "number"), (mesh) => {
    for (let g = 0; g < mesh.geometries.size(); g += 1) {
      const pg = mesh.geometries.get(g);
      const geom = api.GetGeometry(model, pg.geometryExpressID);
      const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
      const index = api.GetIndexArray(geom.GetIndexData(), geom.GetIndexDataSize());
      const m = pg.flatTransformation;
      const positions = new Float32Array((verts.length / 6) * 3);
      for (let v = 0; v < verts.length / 6; v += 1) {
        const x = verts[v * 6], y = verts[v * 6 + 1], z = verts[v * 6 + 2];
        positions[v * 3] = m[0] * x + m[4] * y + m[8] * z + m[12];
        positions[v * 3 + 1] = m[1] * x + m[5] * y + m[9] * z + m[13];
        positions[v * 3 + 2] = m[2] * x + m[6] * y + m[10] * z + m[14];
      }
      meshes.push({ positions, indices: Uint32Array.from(index), itemId: mesh.expressID });
      geom.delete?.();
    }
  });
  api.CloseModel(model);
  return meshes;
}
const jsonOut = flag("--json", null);
let models = args.filter((a, i) => a.toLowerCase().endsWith(".ifc") && args[i - 1] !== "--json");
if (!models.length) models = ["duplex-apartment.ifc", "vienna-escape-route-model.ifc"].map((m) => path.join(EXAMPLES, m));

/** Every element's axis-aligned box and IFC type, for naming what lies near a leak (viewer frame as ifc-extract's meshes). */
async function elementBoxes(file) {
  const api = new WebIFC.IfcAPI();
  api.SetWasmPath(path.join(HERE, "../topologicpy-web-frontend/node_modules/web-ifc/"), true);
  await api.Init();
  const model = api.OpenModel(new Uint8Array(readFileSync(file)), { COORDINATE_TO_ORIGIN: true });
  const boxes = [];
  api.StreamAllMeshes(model, (mesh) => {
    const type = api.GetNameFromTypeCode(api.GetLineType(model, mesh.expressID));
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (let g = 0; g < mesh.geometries.size(); g += 1) {
      const pg = mesh.geometries.get(g);
      const geom = api.GetGeometry(model, pg.geometryExpressID);
      const verts = api.GetVertexArray(geom.GetVertexData(), geom.GetVertexDataSize());
      const m = pg.flatTransformation;
      for (let v = 0; v < verts.length; v += 6) {
        const x = verts[v], y = verts[v + 1], z = verts[v + 2];
        const p = [m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]];
        for (let a = 0; a < 3; a += 1) {
          lo[a] = Math.min(lo[a], p[a]);
          hi[a] = Math.max(hi[a], p[a]);
        }
      }
      geom.delete?.();
    }
    if (lo[0] <= hi[0]) boxes.push({ id: mesh.expressID, type, lo, hi });
  });
  api.CloseModel(model);
  return boxes;
}

const IN_GRID = /^(IFCWALL|IFCWALLSTANDARDCASE|IFCWALLELEMENTEDCASE|IFCSLAB|IFCSLABSTANDARDCASE|IFCSLABELEMENTEDCASE|IFCPLATE|IFCROOF|IFCSTAIR|IFCSTAIRFLIGHT|IFCRAMP|IFCRAMPFLIGHT|IFCDOOR|IFCDOORSTANDARDCASE|IFCWINDOW|IFCWINDOWSTANDARDCASE|IFCRAILING|IFCCOLUMN|IFCCOLUMNSTANDARDCASE|IFCCURTAINWALL|IFCMEMBER|IFCMEMBERSTANDARDCASE)$/i;

const { contentsFuel, ignitionFuel, item } = fuelsFor("A");
const report = [];
for (const file of models) {
  const geo = await extract(file, { withDoorFlags: true });
  const boxes = await elementBoxes(file);
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const rooms = roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls }).filter((r) => r.rect && !r.problem && r.ceiling - r.floor > 1.8);
  const base = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
  // Doors closed: their leaves fill their openings (the fire grid's "Closed" doors); windows closed (obstacles).
  const extra = await meshesOfTypes(file, addTypes);
  const closed = { ...base, doors: geo.doors || [], obstacles: [...(base.obstacles || []), ...extra] };
  if (addTypes.length) console.log(`  added to the fire grid as solid: ${extra.length} meshes of ${addTypes.join(", ")}`);
  const fires = roomNames.length
    ? roomNames.map((name) => rooms.filter((r) => (r.name || "") === name).sort((a, b) => b.floor - a.floor)[0]).filter(Boolean)
    : rooms.slice().sort((a, b) => b.area - a.area).slice(0, roomCount);
  console.log(`\n${path.basename(file)}: ${rooms.length} rooms; ${geo.doors?.length ?? 0} door meshes closed; fires in ${fires.map((r) => r.name || r.id).join(", ")}`);
  for (const h of cells) {
    for (const room of fires) {
      const [cx, cz] = room.rect.centre;
      const fire = [cx, room.floor + 0.75, cz];
      const scene = buildScene(closed, { upAxis: "y", fire, minCell: h, maxCell: h + 1, maxCells: 1e7, contentsFuel, ignitionFuel, item, fireLoad: 0 });
      const g = scene.grid;
      const sim = createSolver(scene, { contentsFuel, ignitionFuel });
      const cellAt = (x, y, z) => {
        const i = Math.floor((x - g.origin[0]) / g.h), j = Math.floor((y - g.origin[1]) / g.h), k = Math.floor((z - g.origin[2]) / g.h);
        return i < 0 || j < 0 || k < 0 || i >= g.nx || j >= g.ny || k >= g.nz ? -1 : i + g.nx * (j + g.ny * k);
      };
      const centreOf = (c) => [g.origin[0] + ((c % g.nx) + 0.5) * g.h, g.origin[1] + ((Math.floor(c / g.nx) % g.ny) + 0.5) * g.h, g.origin[2] + (Math.floor(c / (g.nx * g.ny)) + 0.5) * g.h];
      const inRoom = (r, x, y, z) => {
        const { centre: [a, b], axis: [ua, ub], half: [hu, hv] } = r.rect;
        const u = (x - a) * ua + (z - b) * ub, v = -(x - a) * ub + (z - b) * ua;
        return Math.abs(u) <= hu && Math.abs(v) <= hv && y > r.floor && y < r.ceiling;
      };
      // Flood from the room's middle at 1.2 m, remembering where each cell was reached from.
      const start = cellAt(cx, room.floor + 1.2, cz);
      if (start < 0 || sim.kind[start] === SOLID) continue;
      const from = new Int32Array(g.cells).fill(-2);
      from[start] = -1;
      const queue = [start];
      for (let q = 0; q < queue.length; q += 1) {
        const c = queue[q];
        const i = c % g.nx, j = Math.floor(c / g.nx) % g.ny, k = Math.floor(c / (g.nx * g.ny));
        for (const [di, dj, dk] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]) {
          const a = i + di, b = j + dj, e = k + dk;
          if (a < 0 || b < 0 || e < 0 || a >= g.nx || b >= g.ny || e >= g.nz) continue;
          const n = a + g.nx * (b + g.ny * e);
          if (from[n] !== -2 || sim.kind[n] === SOLID) continue;
          from[n] = c;
          queue.push(n);
        }
      }
      // Which other rooms the flood reaches, and where the air left the fire's room on the way.
      const joined = [];
      for (const other of rooms) {
        if (other === room) continue;
        const [ox, oz] = other.rect.centre;
        const c = cellAt(ox, other.floor + 1.2, oz);
        if (c < 0 || from[c] === -2) continue;
        // Walk back to the fire's room: the first cell outside it is where the air left.
        let path = c, exit = c, steps = 0;
        while (from[path] >= 0) {
          const [x, y, z] = centreOf(path);
          if (!inRoom(room, x, y, z)) exit = path;
          path = from[path];
          steps += 1;
        }
        const at = centreOf(exit).map((v) => +v.toFixed(2));
        const near = boxes.filter((b) => [0, 1, 2].every((a) => at[a] >= b.lo[a] - g.h && at[a] <= b.hi[a] + g.h));
        const missing = [...new Set(near.filter((b) => !IN_GRID.test(b.type) && !addTypes.includes(b.type.toUpperCase()) && b.type !== "IFCSPACE" && b.type !== "IFCOPENINGELEMENT").map((b) => b.type))];
        const types = [...new Set(near.map((b) => b.type))].filter((t) => t !== "IFCSPACE");
        // Through an element: the cell where the air left lies inside a wall's, slab's or door's box
        // (a hole the grid leaves in it); otherwise in the open (an archway, an open plan, a stairwell).
        const inside = boxes.filter((b) => /WALL|SLAB|DOOR|BEAM|COVERING|PLATE|CURTAIN/i.test(b.type) && [0, 1, 2].every((a) => at[a] >= b.lo[a] && at[a] <= b.hi[a]));
        joined.push({ room: other.name || other.id, storey: +other.floor.toFixed(2), leaves: at, steps, nearTypes: types, notInGrid: missing, throughElement: [...new Set(inside.map((b) => b.type))] });
      }
      report.push({ model: path.basename(file), cell: h, fire: room.name || room.id, grid: [g.nx, g.ny, g.nz], joined });
      const own = Math.round((room.area * (room.ceiling - room.floor)) / h ** 3);
      console.log(`  cell ${h} m, fire in ${room.name || room.id} (floor ${room.floor.toFixed(2)} m): its air reaches ${queue.length} open cells (the room holds about ${own}); ${joined.length} other rooms joined with every door closed`);
      const through = joined.filter((j) => j.throughElement.length).length;
      console.log(`    ${through} of them through a cell inside a wall, slab, door, beam or covering; ${joined.length - through} through the open`);
      for (const j of joined.slice(0, 8)) console.log(`    ${j.room} (floor ${j.storey}): leaves at ${j.leaves.join(", ")}${j.throughElement.length ? ` INSIDE ${j.throughElement.join("/")}` : ""}; near ${j.nearTypes.join(", ") || "nothing"}${j.notInGrid.length ? `; not in the fire grid: ${j.notInGrid.join(", ")}` : ""}`);
      if (joined.length > 8) console.log(`    and ${joined.length - 8} more`);
    }
  }
}
if (jsonOut) {
  await writeFile(jsonOut, JSON.stringify({ generated: new Date().toISOString(), report }, null, 1));
  console.log(`wrote ${jsonOut}`);
}
