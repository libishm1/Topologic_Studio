/**
 * Field notes and hints (plan: wiki/roadmap/hints-and-discoveries-plan-2026-11-06.md).
 *
 * Field notes are things a user can find out about their building. Each is
 * shown first as its teaser question; the app marks it found from its own
 * state (App.jsx calls `found` where the result appears), with the details
 * the note's text needs. The header counts them.
 *
 * Hints say where to look next, once, when they apply: mostly at the steps
 * and sections past the first screen, so the app is not taken to end at the
 * model. One at a time, for the step on screen, never while something runs.
 * A hint stops applying once its feature has been used (`used`), or once
 * dismissed.
 *
 * Pure data and functions: the state lives in state/useFieldNotes.js.
 */

const m = (x) => `${Number(x).toFixed(1)} m`;

/** In the order they are listed. `text(detail)` is said once found. */
export const FIELD_NOTES = [
  {
    id: "walk-out",
    teaser: "How far is the walk out?",
    text: (d) =>
      `From this start the walk out is ${m(d.length)}. The code check in step 3 compares each storey's longest walk with the limit for an office.`,
  },
  {
    id: "second-way",
    teaser: "Does every room have a second way out?",
    text: (d) => {
      const longer = d.first > 0 ? Math.round((100 * (d.second - d.first)) / d.first) : 0;
      return `The next way out is ${m(d.second)} against ${m(d.first)}${longer > 0 ? `, ${longer}% longer` : ""}. If the first is blocked, this is what is left.`;
    },
  },
  {
    id: "reroute",
    teaser: "Is the shortest way out always the right one?",
    text: () => "As the fire spread, the hazard-aware route (magenta) changed to keep clear of it: the way out that was shortest at the start was not the one to take.",
  },
  {
    id: "closed-doors",
    teaser: "What do closed doors do?",
    text: () =>
      "This fire burned with every door shut. Run it again with the doors open (the switch beside the fire class) and compare the smoke view: the doors were holding it back.",
  },
  {
    id: "cut-off",
    teaser: "What does the model leave out?",
    text: (d) =>
      `${Math.round(d.largest)}% of the floor is in the graph's largest part; the rest cannot reach it. Either the building really is in separate parts, such as two homes side by side, or a door or a stair is missing from the file.`,
  },
  {
    id: "exit-separation",
    teaser: "Are the exits far enough apart?",
    text: (d) =>
      `${d.passed} of ${d.checked} storeys have their exits far enough apart. Two exits close together can be lost to the same fire, so the code asks for half the floor's diagonal between them, a third with sprinklers.`,
  },
  {
    id: "sprinklers",
    teaser: "Do sprinklers change the rules?",
    text: () => "With sprinklers the walking limit rises from 60.96 m to 91.44 m, and exits may stand a third of the diagonal apart instead of half.",
  },
  {
    id: "no-safe-time",
    teaser: "When is it too late to leave?",
    text: (d) =>
      `In this fire ${d.places.toLocaleString()} points of the graph have no safe time to leave (grey on the deadline map): they need another way out, or an earlier warning.`,
  },
  {
    id: "smoke-first",
    teaser: "Where does the smoke go first?",
    text: () => "In the smoke view, the places that darken first are where visibility at head height falls below the limit first: where people lose their way first.",
  },
  {
    id: "embers",
    teaser: "Can embers start a second fire?",
    text: (d) => `Embers carried by the hot gas lit ${d.spots} spot fire${d.spots === 1 ? "" : "s"} away from the first.`,
  },
  {
    id: "certified",
    teaser: "Can a learner be trusted?",
    text: () => "The shielded agent's route was checked against the fire step by step and certified as the earliest escape.",
  },
  {
    id: "ignition",
    teaser: "What starts fires here?",
    text: (d) =>
      `The fire starts at a likely ignition point for this building: ${d.label}. The list ranks each by its share of fire deaths for the kind of building.`,
  },
];

export const FIELD_NOTE_COUNT = FIELD_NOTES.length;
export const fieldNote = (id) => FIELD_NOTES.find((n) => n.id === id) || null;

/**
 * The hints, in order of priority within a step. `when(s, used)` reads the
 * state summary (App.jsx `hintState`) and the features used so far. An
 * `action` is offered as a button: a step to go to, or settings to set.
 */
export const HINTS = [
  {
    id: "more-steps",
    steps: ["model", "graph"],
    when: (s) => s.model && !s.graph,
    text: "The model is only the start: build the walkable graph here, then find the way out (step 3) and start a fire (step 4).",
    action: { label: "Go to step 2", tab: "graph" },
  },
  {
    id: "ghost",
    steps: ["graph"],
    when: (s) => s.graph && s.appearance === "solid",
    text: "Graph hidden inside the walls? Show the building as Ghost, or cut it at one storey with Show, at the bottom right of the view.",
    action: { label: "Show as Ghost", settings: { modelAppearance: "ghost" } },
  },
  {
    id: "parts",
    steps: ["graph"],
    when: (s) => s.graph && s.split,
    text: "Part of the floor cannot be reached. A level change with no stair in the model joins up if you raise Max vertical step in Graph settings; otherwise a door is likely missing from the file.",
  },
  {
    id: "route-next",
    steps: ["graph"],
    when: (s) => s.graph && !s.route,
    text: "Next, the way out: in step 3, set a start on the graph and press Nearest exit.",
    action: { label: "Go to step 3", tab: "route" },
  },
  {
    id: "alternatives",
    steps: ["route"],
    when: (s, used) => s.route && !used.has("alternatives"),
    text: "Is there a second way out? Find alternative routes shows how much longer it is.",
  },
  {
    id: "code-check",
    steps: ["route"],
    when: (s, used) => s.route && !used.has("codeCheck"),
    text: "Is this walk within the limit for an office? Open Code check at the foot of this step: walking distance, exits per storey, their separation and door widths.",
  },
  {
    id: "fire-next",
    steps: ["route"],
    when: (s, used) => s.route && (used.has("codeCheck") || used.has("alternatives")) && !used.has("fire"),
    text: "See what a fire does to this route: place a fire in step 4 and watch the route react.",
    action: { label: "Go to step 4", tab: "simulate" },
  },
  {
    id: "ignition",
    steps: ["simulate"],
    when: (s, used) => s.graph && !s.fireOrigin && !used.has("ignition"),
    text: "Not sure where a fire would start? Likely ignition points ranks this building's, and one click sets the fire there.",
  },
  {
    id: "flamescapes",
    steps: ["simulate"],
    when: (s, used) => used.has("fire") && !used.has("flamescapes"),
    text: "For smoke, flames and closed doors, switch on the Flamescapes solver and start the fire again.",
  },
  {
    id: "doors-closed",
    steps: ["simulate"],
    when: (s, used) => used.has("flamescapes") && !used.has("flamescapesClosed"),
    text: "Run it again with the doors closed: how far does the smoke get now?",
    action: { label: "Close the doors", settings: { flamescapesDoors: "closed" } },
  },
  {
    id: "plan",
    steps: ["simulate"],
    when: (s, used) => used.has("fire") && !used.has("plan"),
    text: "Leave now or wait? Plan with the forecast gives each place its latest safe time to leave.",
  },
  {
    id: "agent",
    steps: ["simulate"],
    when: (s, used) => used.has("plan") && !used.has("agent"),
    text: "Can a learner find the way out on its own? Train an agent against this fire and compare it with the earliest escape.",
  },
];

/**
 * The hint to show: the first that is for this step, applies, and has not
 * been dismissed; none while something is running or hints are off.
 */
export function nextHint(s, { used = new Set(), dismissed = new Set(), on = true } = {}) {
  if (!on || s.busy) return null;
  return HINTS.find((h) => h.steps.includes(s.step) && !dismissed.has(h.id) && h.when(s, used)) || null;
}
