/**
 * Navigation-graph worker.
 *
 * The browser already holds the geometry in memory to draw it, so the whole
 * graph is built here, off the main thread: one lattice for the building,
 * aligned to its walls, linked to lattice neighbours only (src/core/lattice.js).
 *
 * This replaced per-triangle sampling plus a server-side k-nearest-neighbour
 * linker, whose edges ran at every angle ("the grid has diagonals"; see
 * wiki/roadmap/browser-core-2026-10-27.md). The server no longer builds the
 * graph; when one is reachable the finished graph is uploaded so its features
 * can use it.
 *
 * Input geometry arrives as transferable typed arrays, and the graph goes back
 * the same way, so neither crossing is a copy.
 */
import { buildLatticeGraph } from "../core/lattice.js";

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const graph = buildLatticeGraph({
      floors: payload.floors || [],
      stairs: payload.stairs || [],
      doors: payload.doors || [],
      walls: payload.walls || [],
      spaces: payload.spaces || [],
      obstacles: payload.obstacles || [],
      upAxis: payload.upAxis ?? "y",
      spacing: payload.floorSpacing ?? 0.5,
      agentHeight: payload.agentHeight ?? 0.75,
      maxEdgeRise: payload.maxEdgeRise ?? 0.35,
      columnGap: payload.columnGap ?? 0.9,
      rectilinear: Boolean(payload.rectilinear),
      maxPoints: payload.maxPoints ?? 40000,
    });
    const result = {
      points: graph.points,
      edges: graph.edges,
      kinds: graph.kinds,
      blocked: graph.blocked,
      angle: graph.angle,
      spacing: graph.spacing,
      sections: graph.sections,
      sectionHeight: graph.sectionHeight,
      stats: { ...graph.stats, walls: graph.walls.length },
      // Doors as the graph sees them, to join with the IFC's exit flags. The
      // plan centre, direction across the opening (doorPoints, in the same
      // order) and measured width tell the two-zone fire model which room
      // lies on each side (src/core/rooms.js).
      doors: graph.doorReport.map((d, i) => ({
        itemId: d.itemId,
        node: d.node,
        sides: d.sides,
        x: d.x,
        y: d.y,
        sill: d.sill,
        width: d.width,
        ux: graph.doorPoints[i]?.[3] ?? null,
        uy: graph.doorPoints[i]?.[4] ?? null,
      })),
    };
    self.postMessage({ id, ok: true, result }, [
      result.points.buffer,
      result.edges.buffer,
      result.kinds.buffer,
      result.blocked.buffer,
      ...result.sections.map((section) => section.segs.buffer),
    ]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
