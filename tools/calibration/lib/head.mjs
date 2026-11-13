/**
 * Gas temperature at head height (fire-calibration-plan-2026-11-11.md,
 * section 13.2 item 1; written 2026-11-12, before any head-height comparison
 * was run). One reducer for the measured data and the model:
 *
 *   - devices: each tree's thermocouples 1.5 to 2.0 m over its floor
 *     (validation-data.mjs headOf), left out where they stand in the burner's
 *     plume (inPlume below);
 *   - the model: the tree's open cell that holds the device's height over the
 *     column's model floor (the nearest cell centre; the suite stores each
 *     tree's cells around head height every second, run.series.head);
 *   - transient tests: the rise above ambient, its peak over the run, each
 *     model value the mean of its 1 s readings within half the measured
 *     interval of a measured time (as the suite's point devices);
 *   - steady tests (Steckler): the 200 to 300 s mean against the steady
 *     profile's reading.
 */

/**
 * The plume rule (fixed 2026-11-12, before any head-height comparison, from
 * geometry only): a device is in the burner's plume when its horizontal
 * distance from the burner's centre is under the burner's equivalent radius
 * plus 0.25 times its height over the burner,
 *
 *   r < D/2 + 0.25 (z - z_burner),   D = sqrt(4 A / pi),
 *
 * 0.25 being about twice Heskestad's spread of the plume's temperature
 * half-width (0.12 (z - z0) (Tc/Ta)^(1/2) with Tc/Ta near 1, SFPE Handbook),
 * so the cone holds the plume's edge, not only its core. A device over a
 * floor that lies between it and the burner (an FDS obstruction covering its
 * plan position between the burner's height and its own) is not in the
 * plume: the floor parts them.
 */
export const PLUME_SPREAD = 0.25;

export function plumeCheck(T, tree, dev) {
  const [bx, by] = T.burner.centre;
  const zb = T.burner.z0 ?? 0;
  const D = Math.sqrt((4 * T.burner.area) / Math.PI);
  const z = (tree.z0 ?? 0) + dev.z;
  const r = Math.hypot(tree.x - bx, tree.y - by);
  const reach = D / 2 + PLUME_SPREAD * (z - zb);
  const floorBetween = (T.geometry?.obsts || []).some(({ xb }) => xb[0] <= tree.x && xb[1] >= tree.x && xb[2] <= tree.y && xb[3] >= tree.y && xb[4] >= zb && xb[5] <= z && xb[5] > zb);
  return { r: +r.toFixed(3), reach: +reach.toFixed(3), floorBetween, inPlume: r < reach && !floorBetween };
}

/**
 * The head-height pairs of one run. run: a suite result (cell, horizon,
 * health, series.head); T: its test from the reduced data; steadyFrom: the
 * plan's steady window start (Steckler 200 s). Returns the pairs and the
 * devices left out, with why.
 */
export function headPairs(run, T, steadyFrom = 200) {
  const out = { pairs: [], excluded: [], missing: [] };
  const series = run.series?.head;
  if (!series || !run.health?.ok) return out;
  const h = run.cell;
  const Ta = T.ambientC + 273.15;
  const from = run.series.from ?? 1, step = run.series.step ?? 1;
  const M = T.measured;
  for (const tree of T.trees || []) {
    for (const dev of tree.head || []) {
      const plume = plumeCheck(T, tree, dev);
      const device = `${tree.id}/${dev.id}`;
      if (plume.inPlume) {
        out.excluded.push({ device, why: `in the plume: r ${plume.r} m < ${plume.reach} m` });
        continue;
      }
      const s = series[tree.id];
      if (!s || !s.z.length) {
        out.missing.push({ device, why: "no stored cells at head height" });
        continue;
      }
      // A tree that ends below a pocket (plan 13.22) has no cell above its end: compared with nothing.
      if (s.end != null && dev.z > s.end) {
        out.missing.push({ device, why: `above the tree's end at ${s.end} m: its cell was dropped with a pocket (plan 13.22)` });
        continue;
      }
      let best = -1, dz = Infinity;
      s.z.forEach((z, q) => {
        if (Math.abs(z - dev.z) < dz) (dz = Math.abs(z - dev.z)), (best = q);
      });
      if (dz > h + 1e-6) {
        out.missing.push({ device, why: `nearest open cell ${dz.toFixed(2)} m away` });
        continue;
      }
      const temps = s.T[best];
      const moved = +Math.max(0, dz - h / 2).toFixed(3);
      if (T.steady) {
        const pick = temps.filter((_, q) => from + q * step > steadyFrom);
        const meas = M.head?.[tree.id]?.[dev.id];
        if (!pick.length || meas == null) continue;
        const model = pick.reduce((a, b) => a + b, 0) / pick.length - Ta;
        out.pairs.push(pair(run, T, device, model, meas - T.ambientC, moved));
        continue;
      }
      const meas = M.head?.[tree.id]?.[dev.id];
      if (!meas) continue;
      const dts = M.times.slice(1).map((t, q) => t - M.times[q]).sort((a, b) => a - b);
      const spacing = dts[dts.length >> 1] || 1;
      let bestM = null, bestE = null;
      M.times.forEach((t, iq) => {
        if (t < 1 || t > run.horizon || meas[iq] == null) return;
        const lo = t - spacing / 2, hi = t + spacing / 2;
        let sum = 0, n = 0;
        temps.forEach((v, q) => {
          const ts = from + q * step;
          if (ts > lo && ts <= hi) (sum += v), (n += 1);
        });
        if (!n) return;
        const mv = sum / n - Ta, ev = meas[iq] - T.ambientC;
        bestM = bestM == null ? mv : Math.max(bestM, mv);
        bestE = bestE == null ? ev : Math.max(bestE, ev);
      });
      if (bestM != null) out.pairs.push(pair(run, T, device, bestM, bestE, moved));
    }
  }
  return out;
}

const pair = (run, T, device, model, measured, moved) => ({
  dataset: run.dataset,
  test: T.id,
  cell: run.cell,
  quantity: "headTemperature",
  device,
  model: Number.isFinite(model) ? +model.toFixed(4) : null,
  measured: Number.isFinite(measured) ? +measured.toFixed(4) : null,
  moved,
});
