/**
 * What the view draws of the sprinklers and of the suppression systems' rooms
 * (ViewerManager.setSprinklers, setSystemRooms): their switches decide, for
 * the laid-out plan and for a run's frames alike. `on` is the switch with a
 * Flamescapes model chosen, the only fire they act in.
 *
 * A frame of a run made with sprinklers drew its heads and spray whatever the
 * switch said, so once it was turned off they came back at the next repaint
 * (a change of what the graph shows, a replay) and stayed through every later
 * run, which drew none of its own (the user's report of 2026-11-12, "the
 * sprinklers stayed on when I turned it off"). The rooms a run floods or
 * foams were drawn the same way.
 */

/** The heads to draw, open by `frame` of `run` when one is on screen, else the plan's, shut; null for none. */
export function sprinklerDrawing({ on, plan = null, run = null, frame = null }) {
  if (!on) return null;
  if (run?.sprinklerPoints?.length && frame) return { points: run.sprinklerPoints, open: new Set(frame.sprinklers?.open || []), drops: run.sprinklerDrops ?? null };
  const heads = plan?.heads;
  return heads?.length ? { points: heads.map((h) => h.position), open: null, drops: null } : null;
}

/** The rooms with a clean-agent or foam system to draw, discharged by `frame` of `run`, else the plan's; null for none. */
export function systemsDrawing({ on, plan = null, run = null, frame = null }) {
  if (!on) return null;
  if (run?.systemRooms?.length && frame) return { rooms: run.systemRooms, discharged: new Set(frame.systems?.discharged || []), units: run.systemUnits ?? null };
  return plan?.special?.length ? { rooms: plan.special, discharged: null, units: plan.units ?? null } : null;
}
