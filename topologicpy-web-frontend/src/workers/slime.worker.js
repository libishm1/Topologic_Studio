/**
 * The slime mould's way out (src/core/slimeEgress.js) off the main thread:
 * two thousand agents for four hundred steps take most of a second on the
 * NIBS Clinic's graph. Loaded only when the Route step's slime mould is run.
 * With it, the readings (src/core/skeleton.js;
 * wiki/roadmap/egress-desire-lines-plan-2026-11-11.md): the floor's medial
 * axis, the slime's desire lines and how far they agree, the medial way out
 * from the start, the bottlenecks on the desire lines, the dead ends (the
 * walk before a second way out opens) and the branches the slime explored
 * and abandoned.
 */
import { SLIME_DEFAULTS, slimeEgress } from "../core/slimeEgress.js";
import { agreement, bottlenecks, commonPath, desireLines, markedLinks, medialAxis, medialRoute, openAdjacency } from "../core/skeleton.js";
import { KIND_FLOOR } from "../core/lattice.js";

/** IBC 2021 1020.5's dead-end limits, m: 20 ft; 50 ft in sprinklered buildings of the listed groups. */
const DEAD_END = { plain: 6.096, sprinklered: 15.24 };

/** A node explored early with less than this trail late is a branch given up. */
const ABANDONED = { explored: 0.25, weight: 0.05 };

self.onmessage = (event) => {
  const { id, payload } = event.data || {};
  try {
    const { graph, opts } = payload;
    const n = graph.nodes.length / 3;
    const kinds = graph.kinds || new Uint8Array(n);
    const result = slimeEgress(graph, opts);
    const adj = openAdjacency(graph, { blocked: opts.blocked });
    const medial = medialAxis(graph, adj);
    // Persons on each floor node, at the occupant area the performance check uses.
    const cellArea = (graph.spacing || 0.5) ** 2;
    const people = new Float32Array(n);
    for (let v = 0; v < n; v += 1) if (kinds[v] === KIND_FLOOR) people[v] = cellArea / (opts.occupantArea || 9);
    // With no agent grown (a hard fire over the whole building) there are no
    // desire lines: the tree over zero weights would be the shortest walks.
    const desire = result.agents
      ? desireLines(graph, adj, result.weights, opts.exits, {
          tube: opts.tube ?? SLIME_DEFAULTS.tube,
          fireCost: result.fire === "soft" ? (opts.fireCost ?? SLIME_DEFAULTS.fireCost) : 0,
          smoke: result.smoke,
          people,
        })
      : { links: new Int32Array(0), flow: new Int32Array(0), people: new Float32Array(0), on: new Uint8Array(0), maxFlow: 0 };
    result.medialLinks = medial.links;
    result.desireLinks = desire.links;
    result.desireFlow = desire.flow;
    result.desireMax = desire.maxFlow;
    result.agreement = result.agents ? agreement(adj, desire.on, medial.axis, { reach: 1 }) : null;
    result.medialRoute = opts.start >= 0 ? medialRoute(adj, medial.clearance, opts.start, opts.exits, { beta: 1 }) : null;
    // Bottlenecks: the people the desire lines carry over what each passage's clear width passes.
    result.bottlenecks = result.agents ? bottlenecks(graph, desire.links, desire.people, { clearance: medial.clearance, doorWidth: opts.doorWidth, adj }) : [];
    // Dead ends: floor nodes whose walk before a second way out opens is past the limit.
    const limit = opts.sprinklered ? DEAD_END.sprinklered : DEAD_END.plain;
    const common = commonPath(adj, opts.exits);
    const dead = new Uint8Array(n);
    let deadCount = 0, worst = 0, worstNode = -1;
    if (common) {
      for (let v = 0; v < n; v += 1) {
        if (kinds[v] !== KIND_FLOOR || !(common[v] < Infinity)) continue;
        if (common[v] > worst) {
          worst = common[v];
          worstNode = v;
        }
        if (common[v] > limit) {
          dead[v] = 1;
          deadCount += 1;
        }
      }
    }
    result.deadLinks = markedLinks(adj, dead);
    result.deadEnds = common ? { limit, sprinklered: Boolean(opts.sprinklered), area: deadCount * cellArea, worst, worstNode } : null;
    // Branches explored early and abandoned.
    const abandoned = new Uint8Array(n);
    let given = 0;
    if (result.agents) {
      for (let v = 0; v < n; v += 1) {
        if (result.explored[v] >= ABANDONED.explored && result.weights[v] < ABANDONED.weight) {
          abandoned[v] = 1;
          given += 1;
        }
      }
    }
    result.abandonedLinks = markedLinks(adj, abandoned);
    result.abandonedArea = given * cellArea;
    delete result.smoke;
    delete result.explored;
    self.postMessage({ id, ok: true, result }, [
      result.weights.buffer,
      result.visits.buffer,
      result.dExit.buffer,
      result.medialLinks.buffer,
      result.desireLinks.buffer,
      result.desireFlow.buffer,
      result.deadLinks.buffer,
      result.abandonedLinks.buffer,
    ]);
  } catch (error) {
    self.postMessage({ id, ok: false, error: error?.message || String(error) });
  }
};
