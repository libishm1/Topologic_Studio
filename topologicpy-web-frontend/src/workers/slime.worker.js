/**
 * The slime mould's way out (src/core/slimeEgress.js) off the main thread:
 * two thousand agents for four hundred steps take most of a second on the
 * NIBS Clinic's graph. Loaded only when the Route step's slime mould is run.
 * With it, the skeletons (src/core/skeleton.js): the floor's medial axis,
 * the slime's desire lines, how far the two agree, and the medial way out
 * from the start.
 */
import { SLIME_DEFAULTS, slimeEgress } from "../core/slimeEgress.js";
import { agreement, desireLines, medialAxis, medialRoute, openAdjacency } from "../core/skeleton.js";

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const { graph, opts } = payload;
    const result = slimeEgress(graph, opts);
    const adj = openAdjacency(graph, { blocked: opts.blocked });
    const medial = medialAxis(graph, adj);
    // With no agent grown (a hard fire over the whole building) there are no
    // desire lines: the tree over zero weights would be the shortest walks.
    const desire = result.agents
      ? desireLines(graph, adj, result.weights, opts.exits, {
          tube: opts.tube ?? SLIME_DEFAULTS.tube,
          fireCost: result.fire === "soft" ? (opts.fireCost ?? SLIME_DEFAULTS.fireCost) : 0,
          smoke: result.smoke,
        })
      : { links: new Int32Array(0), flow: new Int32Array(0), on: new Uint8Array(0), maxFlow: 0 };
    result.medialLinks = medial.links;
    result.desireLinks = desire.links;
    result.desireFlow = desire.flow;
    result.desireMax = desire.maxFlow;
    result.agreement = result.agents ? agreement(adj, desire.on, medial.axis, { reach: 1 }) : null;
    result.medialRoute = opts.start >= 0 ? medialRoute(adj, medial.clearance, opts.start, opts.exits, { beta: 1 }) : null;
    delete result.smoke;
    self.postMessage({ id, ok: true, result }, [
      result.weights.buffer,
      result.visits.buffer,
      result.dExit.buffer,
      result.medialLinks.buffer,
      result.desireLinks.buffer,
      result.desireFlow.buffer,
    ]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
