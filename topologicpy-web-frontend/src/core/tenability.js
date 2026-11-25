/**
 * Smoke, gas and tenability measures at each graph node, from the
 * Flamescapes solver's own fields at head height (flamescapesRun.js
 * readNodes): gas temperature, visibility, CO, and, when recorded, the soot
 * density, CO2 and O2.
 *
 * Smoke. The soot mass concentration rho_s gives, by Bouguer's law, the
 * extinction coefficient K = sigma_m rho_s, with the mass-specific extinction
 * coefficient of flaming smoke sigma_m = 8.7 m^2/g (Mulholland and Croarkin
 * 2000; 4.5 to 8.7 m^2/g across fuels); the visibility S = C / K of a
 * light-reflecting sign (C = 3, Jin), as flamescapesRun.js's visibility();
 * the obscuration over one metre, 1 - exp(-K); and the optical density per
 * metre, K / ln 10. None needs storing: each follows from rho_s.
 *
 * Tenability index. ISO 13571 (2007 cl. 4.1; 2012 cl. 4.1) takes the time to
 * compromised tenability as the shortest of the times for asphyxiant gases,
 * irritant gases, heat and visual obscuration, each acting independently
 * (cl. 4.6 a). The index is therefore the largest of these ratios, each 1 at
 * its limit:
 *
 *   heat     (T - T_amb) / (T_lim - T_amb): the gas temperature against the
 *            app's untenable temperature (60 C by default; ISO 13571:2007
 *            8.1 NOTE: respiratory burns from saturated air above 60 C);
 *   smoke    S_lim / S: visibility against its limit (10 m by default;
 *            PD 7974-6 gives 5 m in small rooms and 10 m in large spaces);
 *            0 in clear air (visibility at the 100 m cap), where the ratio
 *            would otherwise never fall below S_lim / 100;
 *   gases    X_FED / FED_crit, the asphyxiant dose (ISO 13571:2007 Eq. 2-3):
 *            sum over minutes of phi_CO / 35 000, the CO term multiplied by
 *            nu_CO2 = exp(phi_CO2 / 5) when CO2 is above 2 % by volume; plus
 *            the low-oxygen dose of Purser (ISO 13571 ref. [5]; SFPE
 *            Handbook, Purser and McAllister 2016), sum of dt /
 *            exp(8.13 - 0.54 (20.9 - phi_O2)), which ISO notes is negligible
 *            while O2 stays above 13 %. With the option `doseForm` "purser"
 *            the rate is instead the one FDS integrates (purserRate below),
 *            for the same three gases; ISO's form stays the default;
 *   heat dose X_FED / FED_crit of convected heat (ISO 13571:2007 Eq. 10-11):
 *            sum over minutes of T^3.4 / 5e7 (lightly clothed);
 *
 * with FED_crit = 0.3, ISO 13571:2007 cl. 5.2's threshold for general
 * occupancies (11.4 % of people are more susceptible). The first two are the
 * criteria the app already closes a Flamescapes node on (forecast.js
 * forecastFromFrames); the two doses add ISO's dose criteria. Not included,
 * as the solver does not give them: radiant heat (no flux at head height),
 * irritant gases and HCN (not modelled), particle sizes (no PM2.5 or PM10).
 * The inputs come from an uncalibrated simulation, so the index ranks
 * places and times under the model; it is not a measured exposure.
 *
 * Smoke detectors. With the option `detectorEntryLag` a detector reads the
 * smoke inside its sensing chamber, which follows the smoke outside with a
 * lag (chamberStep below; Heskestad's characteristic length, as FDS's smoke
 * detector). Off by default: a detector then reads the smoke at its cell.
 */

export const SOOT_EXTINCTION_G = 8.7; // m^2 / g (Mulholland and Croarkin 2000)
export const SIGN_CONSTANT = 3; // light-reflecting signs (Jin)
export const MAX_VISIBILITY = 100; // m, as flamescapesRun.js
export const CO_DOSE = 35000; // ppm min for an FED of 1 (ISO 13571 Eq. 2)
export const FED_CRIT = 0.3; // ISO 13571:2007 cl. 5.2
export const AIR_O2 = 20.9; // % by volume

/** Extinction coefficient (1/m) of smoke with `soot` mg/m^3. */
export const extinction = (soot) => (SOOT_EXTINCTION_G * Math.max(0, soot)) / 1000;
/** Visibility (m) of a light-reflecting sign, as flamescapesRun.js's visibility(). */
export function visibilityOf(soot) {
  const K = extinction(soot);
  return K > SIGN_CONSTANT / MAX_VISIBILITY ? SIGN_CONSTANT / K : MAX_VISIBILITY;
}
/** Share of light lost over one metre of smoke, in %. */
export const obscuration = (soot) => 100 * (1 - Math.exp(-extinction(soot)));
/** Optical density per metre (base 10). */
export const opticalDensity = (soot) => extinction(soot) / Math.LN10;

/** CO2 hyperventilation factor (ISO 13571:2007 Eq. 3), applied above 2 % by volume. */
export const hyperventilation = (co2) => (co2 > 2 ? Math.exp(co2 / 5) : 1);

/** The asphyxiant dose's two forms: ISO 13571's (the default) and Purser's as FDS integrates it. */
export const DOSE_ISO = "iso13571";
export const DOSE_PURSER = "purser";
/**
 * Purser's CO term, FED per minute = 3.317e-5 ppm^1.036 RMV / D, by activity:
 * RMV the volume of air breathed (L/min) and D the carboxyhaemoglobin at
 * incapacitation (%). At rest 8.5 and 40, light work 25 and 30, heavy work
 * 50 and 20, so 0.705e-5, 2.764e-5 and 8.293e-5. D. A. Purser, "Combustion
 * Toxicity", SFPE Handbook of Fire Protection Engineering, 5th ed.,
 * Springer, New York, 2016, as the FDS Verification Guide cites it for its
 * FED (section "Fractional Effective Dose (FED) and Fractional Irritant
 * Concentration (FIC)"); FDS's default activity is light work.
 */
export const PURSER_CO_EXPONENT = 1.036;
export const PURSER_CO = Object.freeze({ rest: (3.317e-5 * 8.5) / 40, light: (3.317e-5 * 25) / 30, heavy: (3.317e-5 * 50) / 20 });
/** Below this O2 (% by volume) FDS counts the low-oxygen term; at or above it, none. */
export const PURSER_O2_FROM = 20;

/**
 * Asphyxiant FED per minute in the form FDS integrates (its FED quantity,
 * after Purser, as above), for the gases this solver gives:
 *
 *   FED_CO  = k ppm_CO^1.036, k by activity (PURSER_CO; light work 2.764e-5);
 *   HV_CO2  = exp(0.1903 %CO2 + 2.0004) / 7.1, at any CO2 level (about 1.04
 *             with next to no CO2, 2 at 3.4 %), where ISO's factor starts
 *             at 2 %;
 *   FED_O2  = 1 / exp(8.13 - 0.54 (20.9 - %O2)), only below 20 % O2;
 *
 * and the rate FED_CO HV_CO2 + FED_O2. FDS's sum also has HCN, NOx and the
 * irritants, which the solver does not model: they are left out, so this is
 * FDS's FED for a fire that gives only CO, CO2 and low oxygen. `co2` and
 * `o2` null (a run that kept only CO) leave their terms out.
 */
export function purserRate(co, co2, o2, activity = "light") {
  const c = co > 0 ? (PURSER_CO[activity] ?? PURSER_CO.light) * Math.pow(co, PURSER_CO_EXPONENT) : 0;
  const hv = co2 == null || !(co2 > 0) ? 1 : Math.exp(0.1903 * co2 + 2.0004) / 7.1;
  const lowO2 = o2 == null || !(o2 < PURSER_O2_FROM) ? 0 : 1 / Math.exp(8.13 - 0.54 * (AIR_O2 - o2));
  return c * hv + lowO2;
}

/**
 * Asphyxiant FED per minute: CO (with CO2's factor) and low O2 (Purser).
 * `form` DOSE_ISO (the default, ISO 13571's) or DOSE_PURSER (purserRate,
 * with its `activity`); any other value reads as ISO's.
 */
export function asphyxiantRate(co, co2, o2, form = DOSE_ISO, activity = "light") {
  if (form === DOSE_PURSER) return purserRate(co, co2, o2, activity);
  const c = Math.max(0, co) / CO_DOSE;
  const nu = co2 == null ? 1 : hyperventilation(co2);
  const lowO2 = o2 == null ? 0 : 1 / Math.exp(8.13 - 0.54 * (AIR_O2 - Math.min(AIR_O2, o2)));
  return c * nu + lowO2;
}
/** Convected-heat FED per minute at T degrees C (ISO 13571:2007 Eq. 10). */
export const heatRate = (T) => (T > 0 ? Math.pow(T, 3.4) / 5e7 : 0);

/**
 * A smoke detector's entry lag. The smoke in the sensing chamber, Y_c,
 * follows the smoke outside, Y_e, as
 *
 *   dY_c / dt = (Y_e - Y_c) / (L / u),
 *
 * with L the detector's characteristic length and u the gas speed past it
 * (Heskestad's model, as the FDS Verification Guide's section "Smoke
 * Detector Model" gives it, citing the FDS Technical Reference Guide). L is
 * 1.8 m, FDS's default for its smoke detector (ALPHA_C, "Heskestad Length
 * Scale"); FDS's default has no second, dwell lag (ALPHA_E 0), and none is
 * modelled here. FDS floors the speed at 1e-10 m/s only to avoid a division
 * by zero, so its detector in still gas never responds; here the floor is
 * DETECTOR_MIN_SPEED, 0.01 m/s, which makes the longest time constant 180 s.
 * That floor is this app's assumption, not a measured entry speed. FDS lags
 * the soot mass fraction and multiplies by the gas density at the detector;
 * the app lags the soot mass concentration (its reading at a cell), which is
 * the same while the density there holds.
 */
export const DETECTOR_LENGTH = 1.8; // m
export const DETECTOR_MIN_SPEED = 0.01; // m/s

/**
 * The chamber's smoke after `seconds`, from `chamber`, with the smoke outside
 * going in a straight line from `before` to `after` over the step and the
 * speed `speed` (m/s) held: the equation above solved exactly for that
 * input, so a step's length does not change the answer. With before = after
 * = Y_0 and an empty chamber it is the guide's closed form,
 * Y_c = Y_0 (1 - exp(-t u / L)). Any unit of smoke; the same unit comes back.
 */
export function chamberStep(chamber, before, after, speed, seconds, { length = DETECTOR_LENGTH, minSpeed = DETECTOR_MIN_SPEED } = {}) {
  if (!(seconds > 0)) return chamber;
  const tau = length / Math.max(minSpeed, speed || 0);
  const slope = (after - before) / seconds;
  return Math.max(0, after - slope * tau + (chamber - before + slope * tau) * Math.exp(-seconds / tau));
}

/**
 * The chambers of `count` detectors over a run, a frame at a time:
 * `step(soot, speed, seconds, out)` takes the soot outside each (mg/m^3) and
 * the gas speed at each (m/s) at the frame's end, moves every chamber on by
 * chamberStep (the soot outside in a straight line from the frame before's
 * reading, none before the first; the frame's speed held over it) and
 * returns the visibility each detector reads (m, visibilityOf its chamber's
 * soot), in `out` when given. `soot` holds the chambers' soot.
 */
export function detectorChambers(count, opts = {}) {
  const soot = new Float64Array(count);
  const before = new Float64Array(count);
  return {
    soot,
    step(outside, speed, seconds, out = new Float32Array(count)) {
      for (let q = 0; q < count; q += 1) {
        const now = Math.max(0, outside[q]);
        soot[q] = chamberStep(soot[q], before[q], now, speed[q], seconds, opts);
        before[q] = now;
        out[q] = visibilityOf(soot[q]);
      }
      return out;
    },
  };
}

/**
 * The running doses of a run, one frame at a time: `step(frame, minutes)`
 * adds a frame of `minutes` and returns its tenability index per node
 * (Float32Array). `fedGas` and `fedHeat` hold the doses so far.
 *
 * @param {number} n     nodes
 * @param {object} opts  ambient (C), limitC (C), visibilityLimit (m), smoke
 *                       (whether visibility counts), fedCrit, doseForm
 *                       (DOSE_ISO by default, or DOSE_PURSER with `activity`)
 */
export function tenabilityTracker(n, { ambient = 20, limitC = 60, visibilityLimit = 10, smoke = true, fedCrit = FED_CRIT, doseForm = DOSE_ISO, activity = "light" } = {}) {
  const fedGas = new Float64Array(n);
  const fedHeat = new Float64Array(n);
  const span = Math.max(1e-6, limitC - ambient);
  return {
    fedGas,
    fedHeat,
    step(frame, minutes) {
      const { T, vis, co, co2, o2 } = frame;
      const ti = new Float32Array(n);
      for (let v = 0; v < n; v += 1) {
        // Doses over the frame at its readings (a step function in time, as ISO's sums).
        fedGas[v] += asphyxiantRate(co[v], co2 ? co2[v] : null, o2 ? o2[v] : null, doseForm, activity) * minutes;
        fedHeat[v] += heatRate(T[v]) * minutes;
        let x = Math.max(0, (T[v] - ambient) / span);
        if (smoke && vis[v] < MAX_VISIBILITY) x = Math.max(x, visibilityLimit / Math.max(1e-6, vis[v]));
        x = Math.max(x, fedGas[v] / fedCrit, fedHeat[v] / fedCrit);
        ti[v] = x;
      }
      return ti;
    },
  };
}

/**
 * The tenability index of frame k of a run (`run.frames`, each with its
 * `time` in seconds), worked out in order and kept on the run. It starts
 * again when the limits change, or when the frames were replaced: a run that
 * restarts on the CPU after the GPU stopped drops its frames, and the doses
 * of the abandoned frames must not carry into the new ones
 * (wiki/architecture/state-machines.md, SM-1).
 *
 * @param {object} run   {frames}; ti, tiKey, tiFirst and tiTracker are kept on it
 * @param {number} k     the frame
 * @param {object} opts  as tenabilityTracker
 */
export function runIndex(run, k, opts = {}) {
  const key = JSON.stringify(opts);
  const n = run.frames[0]?.T.length || 0;
  if (!run.ti || run.tiKey !== key || run.tiFirst !== run.frames[0] || run.ti.length > run.frames.length) {
    run.ti = [];
    run.tiKey = key;
    run.tiFirst = run.frames[0];
    run.tiTracker = tenabilityTracker(n, opts);
  }
  let before = run.ti.length ? run.frames[run.ti.length - 1].time : 0;
  while (run.ti.length <= k && run.ti.length < run.frames.length) {
    const frame = run.frames[run.ti.length];
    run.ti.push(run.tiTracker.step(frame, Math.max(0, frame.time - before) / 60));
    before = frame.time;
  }
  return run.ti[Math.min(k, run.ti.length - 1)] || null;
}

/**
 * Which criterion is nearest its limit at a node now (for a tooltip or the
 * export): "heat", "smoke", "gases" or "heat dose".
 */
export function limitingCriterion({ T, vis, fedGas, fedHeat }, { ambient = 20, limitC = 60, visibilityLimit = 10, smoke = true, fedCrit = FED_CRIT } = {}) {
  const parts = [
    ["heat", (T - ambient) / Math.max(1e-6, limitC - ambient)],
    ["smoke", smoke && vis < MAX_VISIBILITY ? visibilityLimit / Math.max(1e-6, vis) : -Infinity],
    ["gases", fedGas / fedCrit],
    ["heat dose", fedHeat / fedCrit],
  ];
  return parts.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
}
