/**
 * Do the suppression systems by room use work (src/core/suppressionSystems.js)?
 * On each model, the rooms get their systems (planSystems: water, clean
 * agent, foam), and fires are run in rooms of each system with no system
 * and with it, on several room-contents seeds (buildScene's `seed`; single
 * runs near flashover are noisy, tools/sprinkler-seeds.mjs): heat released,
 * fuel burnt, the peak heat release after the system acted, and when each
 * system acted.
 *
 *   node tools/systems-check.mjs [model.ifc ...] [--horizon 180] [--seeds 1,2,3] [--cells 30000] [--json out.json]
 *                                [--cases water,clean-agent,foam,foam-spill] [--water-area 60]
 *
 * "foam-spill" (not run by default) is the foam room again with a 0.1 m²,
 * 5 kg heptane spill kept alight by the pilot for the whole run: on this
 * grid a pool's own flame does not keep its surface boiling, so the app's
 * heptane item goes out when its 30 s pilot stops, before foam arrives.
 *
 * Without models: the NIBS Office and Clinic (commercial: Group B) and the
 * Duplex (apartments: Group R-2) from public/examples. Per commercial model:
 * a water room (the largest room on sprinklers of at most 60 m², not a
 * corridor), a clean agent room (a server room, else another IT room, else
 * an electrical room) and a
 * foam room if one qualifies by name (generator, fuel, flammable or
 * hazardous storage); the Duplex: water only. Fires: class A in water
 * rooms, class C (electrical equipment's plastics) in clean-agent rooms,
 * class B (the app's heptane pool) in foam rooms, each at the room's centre.
 * Modes:
 *   none         no sprinklers, no systems
 *   water        sprinklers in every room ("wetting", the app's default);
 *                in a foam room, ordinary sprinklers in its place
 *   agent        HFC-227ea in the room, its sprinklers left out (agent only)
 *   agent+water  HFC-227ea with the room's sprinklers kept (the default
 *                assignment: NFPA 75 9.1.1, 9.4.1)
 *   foam         foam-water sprinklers in the room (its ordinary heads left out)
 * "After" is read from when the case's system acted in its run: the first
 * head's opening for water, the agent reaching the class A extinguishing
 * concentration, the foam's discharge; the run with no system is read from
 * the same time.
 */
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { extract } from "./ifc-extract.mjs";
import { buildScene, runScene } from "../topologicpy-web-frontend/src/core/flamescapesRun.js";
import { layoutRooms, roomsFromSpaces } from "../topologicpy-web-frontend/src/core/sprinklers.js";
import { planSystems } from "../topologicpy-web-frontend/src/core/suppressionSystems.js";
import { fuelsFor } from "../topologicpy-web-frontend/src/core/fireClasses.js";
import { floorsFromSpaces, slabsMissing } from "../topologicpy-web-frontend/src/core/spaceFloors.js";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const EXAMPLES = path.join(HERE, "../topologicpy-web-frontend/public/examples");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const horizon = Number(flag("--horizon", 180));
const maxCells = Number(flag("--cells", 30000));
const seeds = flag("--seeds", "1,2,3").split(",").map(Number);
const jsonOut = flag("--json", null);
// --cases water,clean-agent,foam: which cases to run; --water-area: the largest room (m²) the water case takes.
const only = new Set(flag("--cases", "water,clean-agent,foam").split(","));
const waterArea = Number(flag("--water-area", 60));
let models = args.filter((a) => a.toLowerCase().endsWith(".ifc")).map((file) => ({ file, occupancy: /duplex|apartment|resid/i.test(file) ? "R-2" : "B" }));
if (!models.length) {
  models = [
    { file: path.join(EXAMPLES, "nibs-office.ifc"), occupancy: "B" },
    { file: path.join(EXAMPLES, "nibs-clinic.ifc"), occupancy: "B" },
    { file: path.join(EXAMPLES, "duplex-apartment.ifc"), occupancy: "R-2" },
  ];
}

/** The room of a case: the largest of those whose use matches, in order of preference. */
function pick(plan, rooms, prefs) {
  for (const test of prefs) {
    const hits = plan.rooms.map((p, i) => ({ p, r: rooms[i] })).filter(({ p, r }) => r.rect && !r.problem && test(p, r));
    if (hits.length) return hits.sort((a, b) => b.r.area - a.r.area)[0].r;
  }
  return null;
}

const report = { horizon, seeds, maxCells, models: [] };
for (const { file, occupancy } of models) {
  const t0 = performance.now();
  const geo = await extract(file, { withDoorFlags: true });
  const names = new Map((geo.semantics?.spaces || []).map((sp) => [sp.id, sp.longName || sp.name]));
  const rooms = geo.spaces.length ? roomsFromSpaces(geo.spaces, { upAxis: "y", names, overhead: [...geo.floors, ...(geo.roofs || [])], ceilings: geo.ceilings, walls: geo.walls }) : [];
  const plan = planSystems(rooms, { occupancy });
  const heads = layoutRooms(rooms).heads;
  const fireGeo = slabsMissing(geo.floors, geo.spaces) ? { ...geo, floors: [...geo.floors, ...floorsFromSpaces(geo.spaces)] } : geo;
  const name = path.basename(file);
  console.log(`\n${name} (Group ${occupancy}): ${rooms.length} spaces, ${heads.length} heads; ${plan.lines[0]}`);
  for (const line of plan.lines.slice(1)) console.log(`  ${line}`);
  const special = plan.rooms.filter((r) => r.system === "clean-agent" || r.system === "foam");
  for (const r of special) console.log(`  ${String(r.name).padEnd(22)} ${r.system.padEnd(11)} ${r.use}${r.agentKg ? `, ${r.agentKg.toFixed(1)} kg of agent for ${r.volume.toFixed(0)} m³` : ""}`);
  const entry = { model: name, occupancy, rooms: rooms.length, heads: heads.length, counts: plan.counts, lines: plan.lines, special: special.map((r) => ({ name: r.name, system: r.system, use: r.use, agentKg: r.agentKg ?? null })), cases: [] };
  report.models.push(entry);

  // Nodes: one at head height in each room (only the readings need them).
  const nodeList = [];
  for (const r of rooms) if (r.rect && !r.problem) nodeList.push(r.rect.centre[0], r.floor + 0.75, r.rect.centre[1]);
  const nodes = Float32Array.from(nodeList);
  const systemOf = new Map(plan.rooms.map((p, i) => [rooms[i], p]));
  const residential = occupancy.startsWith("R");
  const cases = [];
  // A room of everyday size (at most 60 m²): in a 600 m² open office no head opens within minutes.
  const everyday = (p, r) => p.system === "water" && p.use === "general" && !/corridor|stair|vest|toilet|\brr\b|elev|open to below|hall|foyer/i.test(r.name || "") && r.area <= waterArea;
  const water = pick(plan, rooms, [everyday, (p) => p.system === "water"]);
  if (water && only.has("water")) cases.push({ kind: "water", room: water, cls: "A", modes: ["none", "water"] });
  if (!residential) {
    const agent = pick(plan, rooms, [(p, r) => p.system === "clean-agent" && /server/i.test(r.name || ""), (p) => p.system === "clean-agent" && p.use === "it", (p) => p.system === "clean-agent"]);
    if (agent && only.has("clean-agent")) cases.push({ kind: "clean-agent", room: agent, cls: "C", modes: ["none", "water", "agent", "agent+water"] });
    const foam = pick(plan, rooms, [(p) => p.system === "foam"]);
    if (foam && only.has("foam")) cases.push({ kind: "foam", room: foam, cls: "B", modes: ["none", "water", "foam"] });
    // The same room with a spill kept alight (a leak feeding it): on this grid
    // a pool's own flame does not keep its surface boiling, so the app's
    // heptane item goes out when its 30 s pilot stops, before any foam
    // arrives; here the pilot holds the 0.1 m², 5 kg spill hot for the run.
    if (foam && only.has("foam-spill")) cases.push({ kind: "foam", room: foam, cls: "B", spill: true, modes: ["none", "water", "foam"] });
    else if (!foam) console.log("  no room qualifies for foam by name in this model");
  }
  for (const c of cases) {
    const r = c.room;
    const fire = [r.rect.centre[0], r.floor + 0.75, r.rect.centre[1]];
    const roomHeads = heads.filter((h) => h.room !== r.id);
    const fuels = fuelsFor(c.cls);
    const { contentsFuel, ignitionFuel } = fuels;
    const item = c.spill ? { mass: 5, area: 0.1 } : fuels.item;
    const sysRoom = { ...r, system: c.kind === "water" ? "water" : c.kind };
    const options = (mode) => ({
      sprinklers: mode === "water" || mode === "agent+water" ? { heads, mode: "wetting" } : mode === "agent" || mode === "foam" ? { heads: roomHeads, mode: "wetting" } : null,
      systems: mode === "agent" || mode === "agent+water" || mode === "foam" ? { rooms: [sysRoom] } : null,
    });
    console.log(`\n  ${c.kind} case${c.spill ? " (a 0.1 m² heptane spill kept alight)" : ""}: fire (class ${c.cls}) in ${r.name} (${r.area.toFixed(1)} m², floor at ${r.floor.toFixed(1)} m), ${systemOf.get(r).system}; ${horizon} s; seeds ${seeds.join(", ")}`);
    const runs = [];
    for (const seed of seeds) {
      for (const mode of c.modes) {
        const scene = buildScene(fireGeo, { upAxis: "y", fire, maxCells, contentsFuel, ignitionFuel, item, seed });
        if (c.spill && scene.pilot) scene.pilot = { ...scene.pilot, duration: 1e9 };
        const t1 = performance.now();
        const out = runScene(scene, { nodes, horizon, frameSeconds: 5, particles: 0, ...options(mode) });
        const s = out.systems?.rooms[0] ?? null;
        const firstHead = out.sprinklers ? Math.min(...out.sprinklers.heads.map((h) => h.estimate ?? Infinity)) : Infinity;
        // When this run's system acted: the agent at the class A value, the foam's discharge, else the first head.
        const acted = s ? (s.system === "clean-agent" ? s.reachesA : s.dischargeAt) : Number.isFinite(firstHead) ? firstHead : null;
        runs.push({
          seed, mode, acted,
          detectedAt: s?.detectedAt ?? null,
          dischargeAt: s?.dischargeAt ?? null,
          firstHead: Number.isFinite(firstHead) ? firstHead : null,
          heads: out.sprinklers ? out.sprinklers.heads.filter((h) => h.estimate !== null).length : 0,
          released: out.stats.released / 1e6,
          burnt: out.stats.burnt,
          hrr: Array.from(out.hrr),
          peak: Math.max(...out.hrr),
          end: out.hrr[out.K - 1],
          grid: scene.grid.cells,
          clipped: scene.clipped,
          seconds: (performance.now() - t1) / 1000,
          stats: s?.stats ?? null,
        });
      }
    }
    // "After": from when the case's own system acted in each seed's run (the water case: the first head).
    const own = c.kind === "water" ? "water" : c.kind === "clean-agent" ? "agent+water" : "foam";
    for (const run of runs) {
      const ref = runs.find((q) => q.seed === run.seed && q.mode === own)?.acted ?? null;
      run.after = ref;
      run.peakAfter = ref === null ? null : Math.max(0, ...run.hrr.filter((_, k) => (k + 1) * 5 > ref));
      run.releasedAfter = ref === null ? null : run.hrr.reduce((a, q, k) => a + ((k + 1) * 5 > ref ? q * 5 : 0), 0) / 1000;
      const f = (x, d = 1) => (x === null || x === undefined ? "-" : x.toFixed(d));
      console.log(
        `    seed ${run.seed} ${run.mode.padEnd(11)} detected ${f(run.detectedAt, 0).padStart(4)} s, acted ${f(run.acted).padStart(6)} s, heads ${String(run.heads).padStart(2)}  ` +
          `released ${f(run.released).padStart(6)} MJ (after ${f(run.releasedAfter).padStart(6)})  burnt ${f(run.burnt, 2).padStart(6)} kg  peak ${f(run.peak, 0).padStart(5)} kW, after ${f(run.peakAfter, 0).padStart(5)} kW, end ${f(run.end, 0).padStart(5)} kW  [${run.grid} cells, ${run.seconds.toFixed(0)} s]`,
      );
    }
    // Means over the seeds, and each mode against none.
    const summary = c.modes.map((mode) => {
      const rs = runs.filter((q) => q.mode === mode);
      const mean = (key) => (rs.every((q) => q[key] !== null) ? rs.reduce((a, q) => a + q[key], 0) / rs.length : null);
      return { mode, released: mean("released"), releasedAfter: mean("releasedAfter"), burnt: mean("burnt"), peakAfter: mean("peakAfter"), end: mean("end"), acted: mean("acted"), detected: mean("detectedAt") };
    });
    const none = summary.find((q) => q.mode === "none");
    for (const q of summary) {
      if (q === none) continue;
      const pct = (a, b) => (a === null || b === null || !(b > 0) ? "-" : `${((a / b - 1) * 100).toFixed(0)} %`);
      console.log(`    mean ${q.mode.padEnd(11)} against none: released ${pct(q.released, none.released)}, after acting ${pct(q.releasedAfter, none.releasedAfter)}, burnt ${pct(q.burnt, none.burnt)}, peak after acting ${pct(q.peakAfter, none.peakAfter)}; acted at ${q.acted === null ? "-" : q.acted.toFixed(1)} s`);
    }
    entry.cases.push({ kind: c.kind, spill: Boolean(c.spill), room: r.name, area: r.area, cls: c.cls, modes: c.modes, summary, runs: runs.map(({ hrr, ...rest }) => rest) });
  }
  console.log(`  [${((performance.now() - t0) / 1000).toFixed(0)} s for ${name}]`);
}
if (jsonOut) await writeFile(jsonOut, JSON.stringify(report, null, 1));
