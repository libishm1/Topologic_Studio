/**
 * Velocity probes in openings, their profiles' neutral planes, and the mass
 * flow through a door (fire-calibration-plan-2026-11-11.md, 13.2 items 4 and
 * 5; written 2026-11-12, before any of these comparisons was run). One
 * reducer for the measured data and the model, as lib/head.mjs:
 *
 *   - doorVelocity: each probe's largest speed through its opening over the
 *     window, either way (the FDS Validation Guide's "maxabs" for the WTC
 *     probes), the model's value at a measured time the mean of its 1 s
 *     readings within half the measured interval;
 *   - neutralPlane: each probe column's flow reversal height (lib/stats.mjs
 *     crossing, as Steckler's) on its mean profile over the window;
 *   - doorMassFlow: the largest inflow and the largest outflow through the
 *     door over the window (kg/s).
 *
 * The suite keeps the model's series in run.series: probes (each probe's
 * velocity along its axis, positive as the measurement's, m/s) and doorFlow
 * ({in, out}, kg/s, both positive), every second from 1 s.
 */
import { crossing } from "./stats.mjs";

/** The model's mean over the 1 s readings within half the measured interval of t. */
function atTime(series, t, spacing, from = 1, step = 1) {
  const lo = t - spacing / 2, hi = t + spacing / 2;
  let sum = 0, n = 0;
  for (let q = 0; q < series.length; q += 1) {
    const ts = from + q * step;
    if (ts > lo && ts <= hi && series[q] != null) (sum += series[q]), (n += 1);
  }
  return n ? sum / n : null;
}

const spacingOf = (times) => {
  const d = times.slice(1).map((t, q) => t - times[q]).sort((a, b) => a - b);
  return d[d.length >> 1] || 1;
};

/**
 * The pairs of one run. run: a suite result with series.probes and
 * series.doorFlow; T: its test. Returns {pairs, notes}.
 */
export function probePairs(run, T) {
  const out = { pairs: [], notes: [] };
  if (!run.health?.ok) return out;
  const from = run.series?.from ?? 1, step = run.series?.step ?? 1;
  const push = (quantity, device, model, measured) =>
    out.pairs.push({ dataset: run.dataset, test: T.id, cell: run.cell, quantity, device, model: Number.isFinite(model) ? +model.toFixed(4) : null, measured: Number.isFinite(measured) ? +measured.toFixed(4) : null });
  const MP = T.measured?.probes;
  const model = run.series?.probes || {};
  if (MP && T.probes?.length) {
    const spacing = spacingOf(MP.times);
    const win = (q) => {
      const w = T.windows?.[q] || [0, 1e9];
      return [Math.max(1, w[0]), Math.min(w[1], run.horizon)];
    };
    // doorVelocity
    for (const p of T.probes) {
      // A probe's own window (WTC: the guide's row for it), else the quantity's.
      const [a, b] = p.window ? [Math.max(1, p.window[0]), Math.min(p.window[1], run.horizon)] : win("doorVelocity");
      const ms = MP.series[p.id], mm = model[p.id];
      if (!ms || !mm) {
        out.notes.push(`${p.id}: ${!ms ? "no measured series" : "no model cell"}`);
        continue;
      }
      let bestM = null, bestE = null;
      MP.times.forEach((t, iq) => {
        if (t < a || t > b || ms[iq] == null) return;
        const mv = atTime(mm, t, spacing, from, step);
        if (mv == null) return;
        bestM = bestM == null ? Math.abs(mv) : Math.max(bestM, Math.abs(mv));
        bestE = bestE == null ? Math.abs(ms[iq]) : Math.max(bestE, Math.abs(ms[iq]));
      });
      if (bestM != null) push("doorVelocity", p.id, bestM, bestE);
    }
    // neutralPlane, per profile: the mean profile over the window, outflow positive, top down.
    const [c, d] = win("neutralPlane");
    for (const pr of T.profiles || []) {
      const rows = pr.probes.map((id) => T.probes.find((p) => p.id === id)).filter(Boolean);
      const meanOf = (getter) =>
        rows.map((p) => {
          let s = 0, n = 0;
          MP.times.forEach((t, iq) => {
            if (t < c || t > d) return;
            const v = getter(p, t, iq);
            if (v != null) (s += v), (n += 1);
          });
          return n ? (s / n) * pr.outSign : null;
        });
      const zs = rows.map((p) => p.z - (pr.floor ?? 0));
      const em = meanOf((p, t, iq) => MP.series[p.id]?.[iq] ?? null);
      const mo = meanOf((p, t) => (model[p.id] ? atTime(model[p.id], t, spacing, from, step) : null));
      if (em.some((v) => v == null) || mo.some((v) => v == null)) {
        out.notes.push(`${pr.id}: a probe without readings in the window`);
        continue;
      }
      const order = zs.map((z, q) => q).sort((p, q) => zs[q] - zs[p]);
      const z = order.map((q) => zs[q]);
      const ze = crossing(z, order.map((q) => em[q])), zm = crossing(z, order.map((q) => mo[q]));
      if (ze == null) {
        out.notes.push(`${pr.id}: the measured profile does not reverse`);
        continue;
      }
      push("neutralPlane", pr.id, zm, ze);
    }
  }
  // doorMassFlow
  const MF = T.measured?.doorFlow, mf = run.series?.doorFlow;
  if (MF && mf) {
    const spacing = spacingOf(MF.times);
    const w = T.windows?.doorMassFlow || [0, 1e9];
    const [a, b] = [Math.max(1, w[0]), Math.min(w[1], run.horizon)];
    for (const dir of ["in", "out"]) {
      let bestM = null, bestE = null;
      MF.times.forEach((t, iq) => {
        const e = MF[dir][iq];
        if (t < a || t > b || e == null) return;
        const mv = atTime(mf[dir], t, spacing, from, step);
        if (mv == null) return;
        bestM = bestM == null ? mv : Math.max(bestM, mv);
        bestE = bestE == null ? e : Math.max(bestE, e);
      });
      if (bestM != null) push("doorMassFlow", `door ${dir}`, bestM, bestE);
    }
  }
  return out;
}
