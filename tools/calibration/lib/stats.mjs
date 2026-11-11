/**
 * The statistics and reductions of the fire calibration, shared by
 * tools/calibration/validation-suite.mjs and validation-data.mjs. The
 * functions are those of transport-checks.mjs and steckler-checks.mjs
 * (wiki/research/fire-calibration-plan-2026-11-11.md), gathered here so the
 * measured data and the model are reduced by the same code.
 */

/**
 * The experimental relative standard deviations sigma_E of the FDS Validation
 * Guide (NIST SP 1018-3, release 6.11.1), Table 4.1, as its scatter-plot
 * inputs (Utilities/Python/FDS_validation_scatterplot_inputs.csv) give them:
 * HGL temperature 0.07, HGL depth 0.05, ceiling jet 0.07, plume 0.07, O2 and
 * CO2 0.08, smoke obscuration 0.12; gas velocity 0.08 (Table 4.1).
 * neutralPlane 0.05 is NOT the guide's: it is this project's assumption,
 * taken from the layer depth's (the guide gives none for a neutral plane).
 */
export const SIGMA_E = Object.freeze({ hglTemperature: 0.07, hglDepth: 0.05, ceilingJet: 0.07, plume: 0.07, o2: 0.08, co2: 0.08, smoke: 0.12, velocity: 0.08, neutralPlane: 0.05 });

/**
 * The FDS Validation Guide's statistics (section 4.4, Eqs. 4.19 to 4.21) on
 * pairs of (model, measured) values, rises above ambient: sigma_M from the
 * scatter of ln(M/E) less the experiment's, never below it; the bias factor
 * delta. Pairs with a non-positive member are left out (and counted).
 */
export function statistics(pairs, sigmaE) {
  const usable = pairs.filter(([m, e]) => m > 0 && e > 0 && Number.isFinite(m) && Number.isFinite(e));
  const logs = usable.map(([m, e]) => Math.log(m / e));
  const dropped = pairs.length - usable.length;
  if (logs.length < 2) return { n: logs.length, dropped, delta: logs.length ? +Math.exp(logs[0]).toFixed(3) : null, sigmaM: null, meanRatio: logs.length ? +Math.exp(logs[0]).toFixed(3) : null };
  const mean = logs.reduce((a, b) => a + b, 0) / logs.length;
  const varLog = logs.reduce((a, b) => a + (b - mean) ** 2, 0) / (logs.length - 1);
  const sigmaM = Math.max(sigmaE, Math.sqrt(Math.max(0, varLog - sigmaE * sigmaE)));
  const delta = Math.exp(mean + (sigmaM * sigmaM) / 2 - (sigmaE * sigmaE) / 2);
  return { n: logs.length, dropped, delta: +delta.toFixed(3), sigmaM: +sigmaM.toFixed(3), meanRatio: +Math.exp(mean).toFixed(3) };
}

/**
 * This project's tiers (calibration plan 2026-11-08, section 8), on the FDS
 * Validation Guide's statistics (the guide defines the statistics, not the
 * bands). "Calibrated" also needs three independent datasets.
 */
export const tier = (s) => (!s || s.delta == null || s.sigmaM == null ? "too few points" : s.delta >= 0.8 && s.delta <= 1.25 && s.sigmaM <= 0.35 ? "within the calibrated band" : s.delta >= 0.67 && s.delta <= 1.5 && s.sigmaM <= 0.5 ? "indicative" : "not calibrated");

/**
 * Janssens and Tran's (1992) reduction of one tree, as the FDS Validation
 * Guide gives it (HGL chapter, section 1): heights over the floor (rising),
 * temperatures in kelvin, the ceiling H; the lowest reading is the lower
 * layer's temperature, the profile is held at its ends to the floor and the
 * ceiling. Returns {height (interface, m), upper (K), lower (K)}.
 */
export function janssensTran(heights, T, H) {
  const pts = heights.map((z, q) => [Math.min(z, H), T[q]]).filter(([, t]) => Number.isFinite(t) && t > 0).sort((a, b) => a[0] - b[0]);
  if (pts.length < 2) return { height: null, upper: null, lower: null };
  const z = [0, ...pts.map((p) => p[0]), H];
  const t = [pts[0][1], ...pts.map((p) => p[1]), pts[pts.length - 1][1]];
  let I1 = 0, I2 = 0;
  for (let i = 1; i < z.length; i += 1) {
    const dz = z[i] - z[i - 1];
    I1 += 0.5 * (t[i] + t[i - 1]) * dz;
    I2 += 0.5 * (1 / t[i] + 1 / t[i - 1]) * dz;
  }
  const Tl = t[0];
  const denom = I1 + I2 * Tl * Tl - 2 * Tl * H;
  let zi = Math.abs(denom) > 1e-12 ? (Tl * (I1 * I2 - H * H)) / denom : H;
  zi = Math.min(H, Math.max(0, zi));
  let upper = 0, span = 0;
  for (let i = 1; i < z.length; i += 1) {
    const a = Math.max(z[i - 1], zi), b = z[i];
    if (b <= a) continue;
    const f = (x) => (z[i] > z[i - 1] ? t[i - 1] + ((t[i] - t[i - 1]) * (x - z[i - 1])) / (z[i] - z[i - 1]) : t[i]);
    upper += 0.5 * (f(a) + f(b)) * (b - a);
    span += b - a;
  }
  return { height: zi, upper: span > 0 ? upper / span : Tl, lower: Tl };
}

/** Where a profile (heights falling) crosses zero, by linear interpolation; null if it does not. */
export function crossing(z, v) {
  for (let i = 1; i < z.length; i += 1) {
    if (v[i - 1] > 0 && v[i] <= 0) return z[i - 1] + ((z[i] - z[i - 1]) * v[i - 1]) / (v[i - 1] - v[i]);
  }
  return null;
}

/** D* / dx (FDS User's Guide, mesh resolution): D* = (Q / (rho cp T sqrt g))^(2/5), Q in W. */
export const plumeIndex = (Q, dx) => (Q / 1000 / (1.204 * 1.005 * 293.15 * Math.sqrt(9.81))) ** 0.4 / dx;
