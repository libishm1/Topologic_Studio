/**
 * The Flamescapes solver a run uses (Simulate, Flamescapes settings, "Fire
 * solver"): the validated one by default, or the previous one, as the site ran
 * it until 2026-11-12, for continuity and comparison. The previous transport is
 * reached through the solver's own switches (flamescapes.js DEFAULTS): back-
 * traced T and Y (conservative 0), no expansion, the confinement of 10, one
 * step a frame (cfl 0, vorticityStep 0). The run-health guard, the particles'
 * wall check and the porous faces' weighting stay on in both: they only stop
 * impossible runs and close leaks.
 */
export const SOLVERS = Object.freeze({
  validated: Object.freeze({ options: Object.freeze({}), label: "Current (2026-11-12): conserves heat; what it is checked for is on the run card" }),
  previous: Object.freeze({
    options: Object.freeze({ conservative: 0, expansion: 0, vorticity: 10, cfl: 0, vorticityStep: 0 }),
    label: "Previous (the site's until 2026-11-12): not validated; its transport made heat",
  }),
});

/** A solver's options for createSolver: the validated one's (none) unless "previous" is asked for. */
export function solverOptionsFor(name) {
  return (SOLVERS[name] || SOLVERS.validated).options;
}
