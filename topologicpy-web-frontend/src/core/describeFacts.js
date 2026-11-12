/**
 * What "Describe this model" gives the language model: the facts Studio has
 * already worked out from the IFC and from its own simulations, as short
 * lines with their units. The description is grounded in these alone, so it
 * cannot add a room, a door or a number the app did not compute; the page
 * shows the lines before anything is sent. Pure: App.jsx gathers the state,
 * this turns it into text.
 */
import { validationLine } from "./calibrationStatus.js";

const round = (v, d = 1) => (Number.isFinite(v) ? Math.round(v * 10 ** d) / 10 ** d : null);
const list = (items, max) => (items.length > max ? `${items.slice(0, max).join(", ")} and ${items.length - max} more` : items.join(", "));
const duration = (s) => (s >= 90 ? `${round(s / 60, 1)} min` : `${Math.round(s)} s`);
/**
 * The Worker reads this many characters of facts and drops the rest
 * (cloudflare/describe-worker MAX_FACTS), so the facts fit within it: room
 * names listed and storeys given a line each (the rest summed up in one),
 * fewer of each until they do, for a tall building with everything run.
 */
export const MAX_FACTS = 8000;
const DETAIL = [
  { roomNames: 25, storeyLines: 12 },
  { roomNames: 10, storeyLines: 6 },
  { roomNames: 0, storeyLines: 3 },
];
/** A graph floor takes the name of the IFC storey within this many metres of it. */
const STOREY_NAME_WITHIN = 0.6;

const FIRE_MODELS = {
  temperature: "heat diffusion on the walkable graph (an index, not a measured temperature)",
  flood: "spread along links of the walkable graph",
  radial: "distance bands from the origin",
  zone: "the two-zone room model",
  flamescapes: "the Flamescapes combustion solver (fire and smoke)",
  "flamescapes-embers": "the Flamescapes combustion solver (fire, smoke and embers)",
};

/** How each person's next step is drawn (core/crowd.js STEP_RANDOMNESS). */
const STEP_WORDS = { direct: "a little at random (direct)", wandering: "at random, wandering", disoriented: "at random, as if disoriented" };
/** Where the performance check's detection time came from (Panels.jsx PerformanceCheck). */
const DETECTED_BY = {
  ceiling: "by the run's ceiling detectors",
  run: "from the run's smoke at head height",
  "not in the run": "as set (the run's smoke reached no detector)",
  "no run": "as set (no Flamescapes run to detect)",
  set: "as set",
};
const VERDICT = { pass: "pass", fail: "fail", "cannot tell": "cannot tell" };

/**
 * @param {object} s  what the app knows (every part optional):
 *   file {name}; stats (Model step counts); semantics (ifcSemantics.js);
 *   identification answers [{label, by}]; graph {stats, cores,
 *   outdoorSpaces}; egress (codeCheck.js's result on the graph); codeChecked
 *   (the user ran the code check, so its verdicts may be given); levelShift
 *   (m, the model's coordination shift: a viewer height is the IFC's plus
 *   it); route {found, length, node_ids}; nearerExit {walk, route} (m);
 *   walkSpeed (m/s); fire {model, step}; combustion (the Flamescapes run
 *   card); plan {result, inputs}; rl (result); crowd {result, withFire,
 *   occupantArea, exits, fluctuate, steps, runs}; slime (slimeEgress.js's
 *   result with the worker's readings); perf (performanceCheck.js's result);
 *   sprinklers, systems, extinguishers (their layouts, when switched on)
 * @returns {{lines: string[], text: string}}
 */
export function describeFacts(s = {}) {
  let facts = null;
  for (const detail of DETAIL) {
    const lines = factLines(s, detail);
    facts = { lines, text: lines.map((l) => `- ${l}`).join("\n") };
    if (facts.text.length <= MAX_FACTS) break;
  }
  return facts;
}

function factLines(s, { roomNames, storeyLines }) {
  const lines = [];
  const sem = s.semantics || {};
  const st = s.stats || {};

  // The building, from the IFC.
  const storeys = (sem.storeys || []).filter((x) => x && (x.name || Number.isFinite(x.elevation)));
  lines.push(
    `Model: ${s.file?.name || "an IFC file"}${sem.schema ? `, schema ${sem.schema}` : ""}. ` +
      `${storeys.length || st.storeys || 0} storeys` +
      (storeys.length ? ` (${list(storeys.map((x) => `${x.name || "unnamed"} at ${round(x.elevation, 2)} m`), 8)})` : "") +
      ".",
  );
  const rooms = (sem.spaces || []).map((r) => r.longName || r.name).filter(Boolean);
  if (rooms.length || st.spaces) lines.push(`Rooms (IfcSpace): ${rooms.length || st.spaces}${rooms.length && roomNames ? `, named ${list([...new Set(rooms)], roomNames)}` : ""}.`);
  const doors = sem.doors || [];
  const exits = doors.filter((d) => d.exit);
  const widths = doors.map((d) => d.widthM).filter((w) => w > 0);
  if (doors.length || st.doors) {
    lines.push(
      `Doors: ${doors.length || st.doors}` +
        (doors.length ? `, ${exits.length} marked as exits in the file` : "") +
        (widths.length ? `; clear widths ${round(Math.min(...widths), 2)} to ${round(Math.max(...widths), 2)} m` : "") +
        ".",
    );
  }
  const counts = [
    ["stairs", st.stairs],
    ["floor slabs", st.floors],
    ["walls", st.walls],
    ["roof slabs", sem.roofSlabs?.length],
  ].filter(([, n]) => n > 0);
  if (counts.length) lines.push(`Elements: ${counts.map(([k, n]) => `${n} ${k}`).join(", ")}.`);

  // What the elements are (identify.js).
  const answers = s.identification || [];
  if (answers.length) {
    const by = new Map();
    for (const a of answers) by.set(a.label, (by.get(a.label) || 0) + 1);
    const generic = answers.filter((a) => a.by !== "class").length;
    lines.push(
      `Identified elements: ${[...by].sort((a, b) => b[1] - a[1]).map(([k, n]) => `${n} ${k}`).join(", ")}` +
        (generic ? `; ${generic} were generic and named from their name, size or shape` : "") +
        ".",
    );
  }

  // Fire protection the user laid out (sprinklers.js, suppressionSystems.js, extinguishers.js).
  const sp = s.sprinklers;
  if (sp?.heads) {
    const rooms = (sp.rooms || []).filter((r) => r.status !== "n/a");
    const placement = (r) => r.placement ?? r.status;
    const failing = rooms.filter((r) => placement(r) === "fail").length;
    const open = rooms.filter((r) => placement(r) === "cannot tell").length;
    const where = { ifc: "read from the IFC", rooms: `placed by NFPA 13 in ${rooms.length} rooms`, storeys: "placed on a lattice over the walkable floor (the model has no rooms)" }[sp.source] || "laid out";
    lines.push(
      `Sprinklers (laid out by the user): ${sp.heads.length} heads, ${where}` +
        (rooms.length ? `; ${rooms.length - failing - open} rooms meet every placement rule, ${failing} fail one, ${open} cannot be told` : "") +
        (rooms.length && !sp.obstructionCount ? "; beams and ducts not checked (no structural or MEP model)" : "") +
        ".",
    );
  }
  const sys = s.systems;
  if (sys?.lines?.length && !sys.noRooms) lines.push(`Suppression systems by room use: ${sys.lines.join("; ")}.`);
  const ex = s.extinguishers;
  if (ex?.sites) {
    const where = { ifc: "read from the IFC", placed: "placed on the walk by the travel rule" }[ex.source] || "none found in the IFC";
    lines.push(`Extinguishers: ${ex.sites.length}, ${where}` + (ex.check ? `; travel check ${VERDICT[ex.check.status] || ex.check.status} (every place within ${ex.check.limit} m of walking to one)` : "") + ".");
  }

  // The walkable graph and the route.
  const gs = s.graph?.stats;
  if (gs?.nodes) {
    const reach = gs.nodes ? Math.round(((gs.largest_component ?? gs.nodes) / gs.nodes) * 100) : null;
    lines.push(
      `Walkable graph: ${gs.nodes} nodes, ${gs.edges ?? "?"} links, in ${gs.components ?? 1} part${(gs.components ?? 1) === 1 ? "" : "s"}` +
        (reach != null && (gs.components ?? 1) > 1 ? ` (the largest holds ${reach}%)` : "") +
        (gs.stair_nodes ? `; ${gs.stair_nodes} stair nodes` : "") +
        ".",
    );
    const assumed = (s.graph?.cores || []).filter((c) => c.kind === "stair" && c.linked > 0).length;
    if (assumed) lines.push(`Stairs without modelled treads: ${assumed}, walked along an assumed line.`);
    // Spaces left out of the floor (core/outdoorSpaces.js).
    const outside = s.graph?.outdoorSpaces || [];
    if (outside.length) {
      const voids = outside.filter((o) => /void/.test(o.why)).length;
      const parts = [
        outside.length - voids ? `${outside.length - voids} outside the building with no door onto them (roofs, terraces)` : "",
        voids ? `${voids} named as voids` : "",
      ].filter(Boolean);
      lines.push(`Spaces not walked on: ${parts.join(" and ")}.`);
    }
  }

  // Egress by storey (codeCheck.js): each storey's ways out and its longest
  // walk to one, worked out on the graph, so the model words them instead of
  // finding paths itself. The code's verdicts only when the user ran the
  // check: it judges every building as an office (Group B).
  const eg = s.egress;
  if (eg?.storeys?.length) {
    // The graph's floors are in the viewer frame; the IFC's storeys are the
    // viewer's heights less the model's coordination shift.
    const ifcHeight = (x) => x.elevation - (s.levelShift || 0);
    const levelled = storeys.filter((x) => Number.isFinite(x.elevation));
    const nameOf = (x) => {
      const h = ifcHeight(x);
      const near = levelled.reduce((best, y) => (!best || Math.abs(y.elevation - h) < Math.abs(best.elevation - h) ? y : best), null);
      return near && near.name && Math.abs(near.elevation - h) <= STOREY_NAME_WITHIN ? `Storey ${near.name}` : "Storey";
    };
    const verdict = (x) => {
      if (!s.codeChecked) return "";
      const of = (id) => x.checks.find((k) => k.id === id)?.status;
      const separation = of("separation");
      return `; code check: walk ${of("travel")}, number of exits ${of("exits")}${separation && separation !== "n/a" ? `, separation ${separation}` : ""}`;
    };
    for (const x of eg.storeys.slice(0, storeyLines)) {
      const reached = x.unreached < x.places;
      const ways = x.discharge ? `${x.exits} exit${x.exits === 1 ? "" : "s"} (doors out of the building, or stairs down)` : `${x.exits} stair${x.exits === 1 ? "" : "s"} to leave by`;
      lines.push(
        `${nameOf(x)} (floor at ${round(ifcHeight(x), 1)} m${x.discharge ? ", with doors out of the building" : ""}): ` +
          (reached ? `${ways}; the longest walk to one is ${round(x.maxTravel, 1)} m` : "no place reaches a way out on the graph") +
          (reached && x.unreached ? `; ${x.unreached} of ${x.places} places reach none on the graph (often a door or stair the model lacks)` : "") +
          verdict(x) +
          ".",
      );
    }
    const rest = eg.storeys.slice(storeyLines);
    if (rest.length) lines.push(`${rest.length} more storeys; the longest walk to a way out among them is ${round(Math.max(...rest.map((x) => x.maxTravel)), 1)} m.`);
    if (Number.isFinite(eg.building?.height)) lines.push(`Highest occupied floor: ${round(eg.building.height, 1)} m above the storey with the doors out.`);
    if (s.codeChecked) {
      lines.push(
        `Code check (run by the user): as an office (Group B), each limit the stricter of SBC 801-CR-2018 and IBC 2021, ${eg.building?.sprinklered ? "with" : "without"} sprinklers; ` +
          `walking limit ${round(eg.limit, 2)} m; ${eg.building?.highRise ? "a high-rise building" : "not a high-rise building"}; exit door widths ${eg.doors?.status}: ${eg.doors?.text}.`,
      );
    }
  }
  if (s.route) {
    if (s.route.found === false) lines.push("Route: none found between the chosen start and exit.");
    else if (s.route.length > 0) {
      const speed = s.walkSpeed > 0 ? s.walkSpeed : 1.2;
      lines.push(`Shortest route from the chosen start to the chosen exit: ${round(s.route.length, 1)} m, about ${duration(s.route.length / speed)} at ${speed} m/s.`);
      if (s.nearerExit?.walk >= 0) lines.push(`Another exit out of the building is much nearer that start: ${round(s.nearerExit.walk, 1)} m away.`);
    }
  }

  // The fire.
  if (s.fire?.step > 0 && s.fire.model) {
    lines.push(`Fire simulation: ${FIRE_MODELS[s.fire.model] || s.fire.model}, ${s.fire.step} steps played.`);
  }
  const c = s.combustion;
  if (c && c.time > 0) {
    lines.push(
      `Flamescapes run: ${Math.round(c.time)} of ${Math.round(c.horizon ?? c.time)} s simulated; heat release peaked at ${Math.round(c.peak ?? 0)} kW` +
        (c.grid ? ` on a ${c.grid.join(" x ")} grid of ${round(c.cell, 2)} m cells` : "") +
        (c.backend?.name ? `, on ${c.backend.kind === "gpu" ? "the GPU" : "one CPU core"}` : "") +
        // What held-out measured fires support on its cells (core/calibrationStatus.js).
        `. ${
          c.solver === "previous"
            ? "It ran on the previous solver (the site's until 2026-11-12), which was not checked against measured fires."
            : c.cell > 0
              ? validationLine(c.cell, { peakKW: c.peak ?? null, horizon: c.horizon ?? null })
              : "The solver is not calibrated."
        } Its heat release depends on the grid.`,
    );
    const o = c.openings;
    if (o) {
      lines.push(
        `Openings in that run: ${o.doorCount} doors ${o.doors}, ${o.windowCount} windows ${o.windows}` +
          (o.ventilation === "running" ? `, ventilation running through ${o.ductCells ?? "?"} of ${o.terminals} air terminals` : o.terminals ? `, ventilation off (${o.terminals} air terminals)` : "") +
          ".",
      );
    }
    lines.push(c.embersOn ? `Embers: on, ${c.embers ?? 0} thrown, ${c.spots ?? 0} spot fires.` : "Embers: off.");
  }

  // People leaving (core/crowd.js), when the animation was run: an illustration.
  const cr = s.crowd?.result;
  if (cr?.people > 0) {
    const exits = cr.exits || [];
    lines.push(
      `People leaving (the egress animation, an illustration, not an evacuation model): ${cr.people} people at one per ${s.crowd.occupantArea ?? "?"} m² of walkable floor, ` +
        (s.crowd.withFire ? "with the fire, keeping out of places it made untenable" : "without the fire") +
        `; the last out at ${duration(cr.lastOut)} through ${exits.length} exit${exits.length === 1 ? "" : "s"} (${exits.map((e) => e.people).join(", ")} people), the longest hold-up ${duration(cr.longestWait)}` +
        (cr.cutOff > 0 ? `; ${cr.cutOff} cut off by the fire stayed where they were` : cr.inside > 0 ? `; ${cr.inside} still inside when the walk stopped` : "") +
        ".",
    );
    const turned = cr.switches ? cr.switches.reduce((n, x) => n + (x > 0 ? 1 : 0), 0) : 0;
    const how = [
      s.crowd.exits === "picked" ? "all were sent to the exit picked alone, to show one exit's load" : "every exit out of the building was open to them",
      cr.tooLarge
        ? "each took the nearest exit (the graph is too large to weigh every exit for everyone)"
        : cr.choice === "quickest"
          ? `each took the exit that got them out soonest, the walk or the queue ahead${turned ? `, and ${turned} turned to another exit on the way` : ""}`
          : "each took the nearest exit by the shortest walk",
      s.crowd.fluctuate ? "walking speeds varied over time as measured (Corbetta et al. 2017)" : "",
      STEP_WORDS[s.crowd.steps] ? `each step chosen ${STEP_WORDS[s.crowd.steps]} by the floor field (Kirchner and Schadschneider 2002)` : "",
    ].filter(Boolean);
    lines.push(`How they chose: ${how.join("; ")}.`);
    const outs = (s.crowd.runs?.samples || []).map((x) => x.lastOut).filter(Number.isFinite).sort((a, b) => a - b);
    if (outs.length > 1) {
      const mid = outs.length % 2 ? outs[(outs.length - 1) / 2] : (outs[outs.length / 2 - 1] + outs[outs.length / 2]) / 2;
      lines.push(`Over ${outs.length} runs with different random seeds the last out ranged from ${duration(outs[0])} to ${duration(outs.at(-1))} (median ${duration(mid)}).`);
    }
  }

  // The slime mould (core/slimeEgress.js, workers/slime.worker.js), when grown.
  const sl = s.slime;
  if (sl?.agents > 0) {
    const reached = (sl.exits || []).reduce((n, e) => n + e.flux, 0);
    const fire =
      sl.fire === "soft"
        ? `held back by the fire${sl.smokeSource === "visibility" ? " and the Flamescapes run's smoke (its visibility, and the agents' sight shortened in it)" : ""} over ${Math.round(sl.seconds)} s of it, ${sl.closed} places untenable by then and ${Math.round(sl.fireShare * 100)}% of its trail laid in smoke or fire`
        : sl.fire === "hard"
          ? `walled off from ${sl.closed} places the fire made untenable`
          : "grown as if there were no fire";
    lines.push(
      `Slime mould (Jones's Physarum agents grown over the walkable graph towards the exits; a reading of where the escape concentrates, for design, not an evacuation model): ` +
        `${sl.agents} agents for ${sl.steps} steps, ${reached} reaching an exit through ${(sl.exits || []).length} exits; ${fire}` +
        (sl.route ? `; from the chosen start along its tubes ${round(sl.route.length, 1)} m against the shortest walk ${round(sl.route.shortest, 1)} m` : "") +
        (sl.medialRoute ? `, keeping to the floor's medial axis ${round(sl.medialRoute.length, 1)} m` : "") +
        (sl.agreement != null ? `; ${Math.round(sl.agreement * 100)}% of its desire lines lie within half a metre of the medial axis` : "") +
        ".",
    );
    const necks = (sl.bottlenecks || []).slice(0, 3);
    const readings = [
      necks.length
        ? `bottlenecks (the people the desire lines carry, one per occupant area, at 1.3 people a second per metre of clear width): ${necks.map((b) => `${Math.round(b.people)} people through ${round(b.width, 2)} m of ${b.door ? "door" : "corridor"}, ${Math.round(b.seconds)} s`).join("; ")}`
        : sl.bottlenecks
          ? "no bottleneck holds the desire lines' people 10 s or more"
          : "",
      sl.deadEnds
        ? `the walk before a second, separate way out opens is longer than ${round(sl.deadEnds.limit, 1)} m (IBC 2021 1020.5's dead-end limit for corridors${sl.deadEnds.sprinklered ? ", sprinklered" : ""}, applied to every place, rooms included, as a reference) over ${Math.round(sl.deadEnds.area)} m² of floor, the longest ${round(sl.deadEnds.worst, 1)} m`
        : "",
      sl.abandonedArea > 0 ? `it explored and then gave up ${Math.round(sl.abandonedArea)} m² (branches that look like ways out)` : "",
    ].filter(Boolean);
    if (readings.length) lines.push(`Slime mould readings: ${readings.join("; ")}.`);
  }

  // The forecast plan (Panels.jsx planHeading).
  const plan = s.plan?.result;
  if (plan?.front) {
    const inputs = s.plan.inputs || {};
    const depart = inputs.departAfter || 0;
    const within = plan.front.find((p) => p.dose <= inputs.doseLimit);
    if (within) lines.push(`Forecast plan: a safe route leaves the building in ${duration(within.arrival - depart)}, within the dose limit.`);
    else if (plan.earliest) lines.push(`Forecast plan: the earliest way out takes ${duration(plan.earliest.arrival - depart)}, above the dose limit.`);
    else lines.push("Forecast plan: no safe route within the forecast.");
  }

  // The performance-based check (core/performanceCheck.js), when the user ran it.
  const pf = s.perf;
  if (pf?.status) {
    const count = (st) => (pf.storeys || []).filter((x) => x.status === st).length;
    const storeys = ["pass", "fail", "cannot tell"].map((st) => [st, count(st)]).filter(([, n]) => n > 0);
    const busiest = pf.queues?.busiest;
    lines.push(
      `Performance check (run by the user, ASET against RSET from the forecast${pf.criterion?.label ? `; ${pf.criterion.label}` : ""}): ${VERDICT[pf.status] || pf.status}; ` +
        `occupants must be able to set off up to ${Math.round(pf.start)} s after ignition${pf.factor > 1 ? " beside an exit, later farther away" : ""}; ` +
        `detection at ${Math.round(pf.detection)} s ${DETECTED_BY[pf.detectionFrom] || "as set"}, the alarm at ${Math.round(pf.cue)} s` +
        (storeys.length ? `; storeys: ${storeys.map(([st, n]) => `${n} ${st}`).join(", ")}` : "") +
        (busiest?.wait > 0 ? `; the busiest door passes ${Math.round(busiest.persons)} people and holds them up to ${Math.round(busiest.wait)} s` : "") +
        ".",
    );
  }

  // The learned escape.
  const rl = s.rl;
  if (rl) {
    if (rl.method === "shielded") {
      lines.push(
        rl.reached_exit
          ? `Learned escape (shielded): out in ${rl.moves} moves${rl.certified ? ", certified as the earliest escape" : ""}.`
          : rl.escape_steps == null
            ? "Learned escape (shielded): no escape exists from the chosen start against that fire."
            : "Learned escape (shielded): did not reach an exit.",
      );
    } else lines.push(`Learned escape (Q-learning): ${rl.reached_exit ? "reached the exit" : "did not reach the exit"}.`);
  }

  return lines;
}

/** The instructions the model gets with the facts (the Worker holds its own copy). */
export const DESCRIBE_INSTRUCTIONS = [
  "You describe a building model for an architect or fire-safety reviewer, from facts that Topologic Studio computed from the model's IFC file and its own simulations.",
  "Use only the facts given. Never invent rooms, numbers, materials or code compliance; when the facts do not say something, say that they do not.",
  "Write plain English in four short parts: the building; how people get out, storey by storey, with the code check's verdicts if they are given; what the fire simulation showed, or that none was run; what to check next.",
  "Say that the fire results come from uncalibrated models. Keep it under 250 words. No headings in capitals, no bullet lists longer than five items.",
].join(" ");
