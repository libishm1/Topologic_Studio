/**
 * The egress map and the accessibility audit off the main thread
 * (core/egressRun.js). Loaded only when asked for, from the Route step.
 *
 *   {kind: "map"}   the egress map on the walkable graph: tens of milliseconds a solve
 *   {kind: "fine"}  each storey's measuring grid, the walk on 16 neighbours, the
 *                   Fast Marching check, the wheelchair audit, accessible means of
 *                   egress, and the map again with the grid's widths
 */
import { rooms, runFine, runMap } from "../core/egressRun.js";

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const { kind, graph, opts, geometry } = payload;
    const names = new Map(opts.spaceNames || []);
    const roomList = rooms(geometry, names, graph.upAxis || "y");
    if (kind === "map") {
      const result = runMap(graph, opts, roomList, opts.widthAt || null);
      self.postMessage({ id, ok: true, result }, [result.routeLinks.buffer, result.deadLinks.buffer, result.contours.buffer]);
    } else {
      const result = runFine(graph, opts, geometry, roomList, names);
      const transfer = result.images.flatMap((im) => [im.distance.buffer, im.access.buffer]);
      self.postMessage({ id, ok: true, result }, transfer);
    }
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
