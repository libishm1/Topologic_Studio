/**
 * Forecast worker: runs the fire model to its horizon, plans against it and
 * maps the escape deadlines (src/core/forecast.js, src/core/deadline.js) off
 * the main thread. The deadline map is the slow part, a few seconds on a
 * 30,000-node graph.
 *
 * With model "zone" the fire is the two-zone room model (src/core/rooms.js,
 * src/core/zone.js): the payload then carries `zone`, the IfcSpace meshes, the
 * graph's doors with their directions, the IFC's declared door sizes and exit
 * flags, and the fire's growth, peak, horizon and smoke criterion. The same
 * routers run on its forecast unchanged.
 *
 * With model "flamescapes" or "flamescapes-embers" the payload carries the
 * Flamescapes solver's frames (gas temperature, visibility and CO at every
 * node; src/core/flamescapesRun.js), already computed by its own worker.
 *
 * The graph arrives as copies of its typed arrays (the page keeps its own);
 * the deadline map goes back as a transferable buffer.
 */
import { buildForecast, planWithForecast } from "../core/forecast.js";
import { doseDeadline, doseDeadlineMap, escapeDeadlines } from "../core/deadline.js";
import { buildRoomGraph } from "../core/rooms.js";
import { GROWTH, buildZoneForecast, fireRoomOf, roomTemperatureFrames, simulateRooms } from "../core/zone.js";
import { ceilingDetection, forecastFromFrames, smokeDetection } from "../core/forecast.js";
import { memoryKind } from "../core/memoryNotice.js";

/** Exit nodes this close (m) to the one nearest the fire are the same exit: the other leaf of a double door. */
const SAME_EXIT_M = 2;

/** The exits left when the one nearest the fire (and any within SAME_EXIT_M of it) is blocked, and that exit. */
function withoutNearestExit(nodes, exits, seed) {
  const dist = (a, b) => Math.hypot(nodes[a * 3] - nodes[b * 3], nodes[a * 3 + 1] - nodes[b * 3 + 1], nodes[a * 3 + 2] - nodes[b * 3 + 2]);
  let near = -1;
  for (const e of exits) if (near < 0 || dist(e, seed) < dist(near, seed)) near = e;
  const rest = exits.filter((e) => dist(e, near) > SAME_EXIT_M);
  return { blocked: near, rest };
}

/** The room graph, or a plain error when the model's spaces give no rooms. */
function roomGraphFor(graph, zone) {
  const rooms = buildRoomGraph(graph, {
    spaces: zone.spaces,
    doors: zone.doors,
    declaredDoors: zone.declaredDoors,
    upAxis: zone.upAxis,
    agentHeight: zone.agentHeight,
  });
  if (!rooms.rooms.length) {
    throw new Error("The two-zone model needs IfcSpace rooms, and this model's spaces give none. Choose a graph model instead.");
  }
  return rooms;
}

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    if (payload?.kind === "zoneFrames") {
      // The live simulation's frames: each room's gas temperature (degrees C)
      // every `frameSeconds`, and each node's room (zone.js zoneNodeFrames).
      const { graph: g, seed, zone, frames, frameSeconds } = payload;
      const graph = { nodes: g.nodes, edges: g.edges, blocked: g.blocked };
      const rooms = roomGraphFor(graph, zone);
      const sim = simulateRooms(rooms, {
        fireRoom: fireRoomOf(rooms.nodeRoom, seed),
        alpha: GROWTH[zone.growth] ?? GROWTH.medium,
        peak: zone.peak,
        horizon: frames * frameSeconds,
      });
      const table = roomTemperatureFrames(sim, rooms.rooms, frames, frameSeconds);
      const nodeRoom = Int32Array.from(rooms.nodeRoom);
      self.postMessage(
        { id, ok: true, result: { nodeRoom, table, rooms: rooms.rooms.length, frames, frameSeconds, flashover: sim.flashover } },
        [nodeRoom.buffer, table.buffer],
      );
      return;
    }
    const t0 = performance.now();
    const { graph: g, start, exits, seed, model, steps, secondsPerStep, threshold, speed, depart, limit, useWalls, withMap, blockNearExit, zone, frames } = payload;
    const graph = { nodes: g.nodes, edges: g.edges, blocked: g.blocked };
    const blocked = useWalls ? graph.blocked : null;
    let forecast, zoneSummary = null, coverage = null, detection = null, ceiling = null;
    if (model === "zone") {
      const rooms = roomGraphFor(graph, zone);
      forecast = buildZoneForecast(graph, rooms, {
        seed,
        growth: zone.growth,
        peak: zone.peak,
        untenableC: threshold,
        smoke: zone.smoke,
        horizon: zone.horizon,
      });
      // What the result card says; the per-room histories stay here.
      zoneSummary = { ...forecast.zone, roomStats: rooms.stats };
    } else if (model === "flamescapes" || model === "flamescapes-embers") {
      if (!frames) throw new Error("Run the Flamescapes fire first.");
      forecast = forecastFromFrames(frames, { threshold, smoke: frames.smoke, visibilityLimit: frames.visibilityLimit });
      coverage = { clipped: Boolean(frames.clipped), inGrid: frames.inGrid ?? 1 };
      // For the performance check (core/performanceCheck.js): the ceiling
      // detectors when the run read them, else smoke at head height.
      ceiling = ceilingDetection(frames);
      detection = ceiling ? ceiling.first : smokeDetection(frames);
    } else {
      forecast = buildForecast(graph, { model, seed, steps, dt: secondsPerStep, threshold, blocked });
    }
    const tForecast = performance.now();
    const plan = planWithForecast(graph, { start, exits, forecast, depart, speed, blocked });
    const hard = escapeDeadlines(graph, { exits, untenable: forecast.closes, speed, blocked })[start];
    const leaveBy = doseDeadline(graph, start, { exits, forecast, limit, speed, blocked, tolerance: 0.1 });
    const tPlan = performance.now();
    const map = withMap ? doseDeadlineMap(graph, { exits, forecast, limit, speed, blocked }) : null;
    // A second map with the exit nearest the fire blocked, for the
    // performance check's sensitivity case (SFEG 2025 test 5), when asked.
    const cut = withMap && blockNearExit && exits.length > 1 ? withoutNearestExit(graph.nodes, exits, seed) : null;
    const mapBlocked = cut?.rest.length ? doseDeadlineMap(graph, { exits: cut.rest, forecast, limit, speed, blocked }) : null;
    const tMap = performance.now();
    const result = {
      model,
      horizon: forecast.horizon,
      coarsened: forecast.stride > 1 ? forecast.frameSeconds : null,
      startUntenable: forecast.untenable[start],
      hardDeadline: hard,
      leaveBy,
      earliest: plan.earliest,
      fastest: plan.fastest,
      leastDose: plan.leastDose,
      front: plan.front,
      truncated: plan.truncated,
      labels: plan.labels,
      snapshot: plan.snapshot,
      deadlines: map?.deadline ?? null,
      mapStep: map?.step ?? null,
      zone: zoneSummary,
      coverage,
      detection,
      detectionAt: ceiling ? "ceiling" : detection === null ? null : "head height",
      detectionSlowest: ceiling?.slowest ?? null,
      detectors: ceiling?.count ?? 0,
      deadlinesBlocked: mapBlocked?.deadline ?? null,
      blockedExit: mapBlocked ? cut.blocked : null,
      exitsLeft: mapBlocked ? cut.rest : null,
      ms: { forecast: tForecast - t0, plan: tPlan - tForecast, map: tMap - tPlan, total: tMap - t0 },
    };
    const transfers = [];
    if (map) transfers.push(map.deadline.buffer);
    if (mapBlocked) transfers.push(mapBlocked.deadline.buffer);
    self.postMessage({ id, ok: true, result }, transfers);
  } catch (error) {
    // Typed when it is running out of memory (core/memoryNotice.js), so the page can offer to clear its caches.
    self.postMessage({ id, ok: false, error: error?.message || String(error), kind: memoryKind(error) });
  }
};
