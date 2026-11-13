/**
 * Why a Flamescapes run was stopped, in words (proposal A of
 * wiki/research/fire-calibration-plan-2026-11-11.md; the guard and its limits
 * are flamescapes.js HEALTH). The card and the notice read the same words.
 */
import { HEALTH } from "./flamescapes.js";

const round = (x) => Math.round(x).toLocaleString("en-US");

/** The reason a run stopped, from its health {reason, value}, as a clause. */
export function healthReason(health) {
  switch (health?.reason) {
    case "non-finite":
      return "the gas held a value that is not a number";
    case "temperature":
      return `the gas reached ${round(health.value)} K, past the ${round(HEALTH.maxT)} K no flame in air reaches`;
    case "speed":
      return `the gas moved at ${round(health.value)} m/s, past the ${round(HEALTH.maxSpeed)} m/s no building fire drives`;
    case "energy":
      return "the gas held over one and a half times the heat the fire had put in";
    default:
      return "the solver went unstable";
  }
}
