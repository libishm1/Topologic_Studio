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
 *            while O2 stays above 13 %;
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
/** Asphyxiant FED per minute: CO (with CO2's factor) and low O2 (Purser). */
export function asphyxiantRate(co, co2, o2) {
  const c = Math.max(0, co) / CO_DOSE;
  const nu = co2 == null ? 1 : hyperventilation(co2);
  const lowO2 = o2 == null ? 0 : 1 / Math.exp(8.13 - 0.54 * (AIR_O2 - Math.min(AIR_O2, o2)));
  return c * nu + lowO2;
}
/** Convected-heat FED per minute at T degrees C (ISO 13571:2007 Eq. 10). */
export const heatRate = (T) => (T > 0 ? Math.pow(T, 3.4) / 5e7 : 0);

/**
 * The running doses of a run, one frame at a time: `step(frame, minutes)`
 * adds a frame of `minutes` and returns its tenability index per node
 * (Float32Array). `fedGas` and `fedHeat` hold the doses so far.
 *
 * @param {number} n     nodes
 * @param {object} opts  ambient (C), limitC (C), visibilityLimit (m), smoke
 *                       (whether visibility counts), fedCrit
 */
export function tenabilityTracker(n, { ambient = 20, limitC = 60, visibilityLimit = 10, smoke = true, fedCrit = FED_CRIT } = {}) {
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
        fedGas[v] += asphyxiantRate(co[v], co2 ? co2[v] : null, o2 ? o2[v] : null) * minutes;
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
