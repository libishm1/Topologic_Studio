/**
 * What "Describe this model" gives the language model: the facts Studio has
 * already worked out from the IFC and from its own simulations, as short
 * lines with their units. The description is grounded in these alone, so it
 * cannot add a room, a door or a number the app did not compute; the page
 * shows the lines before anything is sent. Pure: App.jsx gathers the state,
 * this turns it into text.
 */

const round = (v, d = 1) => (Number.isFinite(v) ? Math.round(v * 10 ** d) / 10 ** d : null);
const list = (items, max) => (items.length > max ? `${items.slice(0, max).join(", ")} and ${items.length - max} more` : items.join(", "));
const duration = (s) => (s >= 90 ? `${round(s / 60, 1)} min` : `${Math.round(s)} s`);

const FIRE_MODELS = {
  temperature: "heat diffusion on the walkable graph (an index, not a measured temperature)",
  flood: "spread along links of the walkable graph",
  radial: "distance bands from the origin",
  zone: "the two-zone room model",
  flamescapes: "the Flamescapes combustion solver (fire and smoke)",
  "flamescapes-embers": "the Flamescapes combustion solver (fire, smoke and embers)",
};

/**
 * @param {object} s  what the app knows (every part optional):
 *   file {name}; stats (Model step counts); semantics (ifcSemantics.js);
 *   identification answers [{label, by}]; graph {stats, cores};
 *   route {found, length, node_ids}; walkSpeed (m/s); fire {model, step};
 *   combustion (the Flamescapes run card); plan {result, inputs}; rl (result)
 * @returns {{lines: string[], text: string}}
 */
export function describeFacts(s = {}) {
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
  if (rooms.length || st.spaces) lines.push(`Rooms (IfcSpace): ${rooms.length || st.spaces}${rooms.length ? `, named ${list([...new Set(rooms)], 25)}` : ""}.`);
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
  }
  if (s.route) {
    if (s.route.found === false) lines.push("Route: none found between the chosen start and exit.");
    else if (s.route.length > 0) {
      const speed = s.walkSpeed > 0 ? s.walkSpeed : 1.2;
      lines.push(`Shortest route from the chosen start to the chosen exit: ${round(s.route.length, 1)} m, about ${duration(s.route.length / speed)} at ${speed} m/s.`);
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
        ". The solver is not calibrated and its heat release depends strongly on the grid.",
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

  return { lines, text: lines.map((l) => `- ${l}`).join("\n") };
}

/** The instructions the model gets with the facts (the Worker holds its own copy). */
export const DESCRIBE_INSTRUCTIONS = [
  "You describe a building model for an architect or fire-safety reviewer, from facts that Topologic Studio computed from the model's IFC file and its own simulations.",
  "Use only the facts given. Never invent rooms, numbers, materials or code compliance; when the facts do not say something, say that they do not.",
  "Write plain English in four short parts: the building; how people get out; what the fire simulation showed, if one was run; what to check next.",
  "Say that the fire results come from uncalibrated models. Keep it under 250 words. No headings in capitals, no bullet lists longer than five items.",
].join(" ");
