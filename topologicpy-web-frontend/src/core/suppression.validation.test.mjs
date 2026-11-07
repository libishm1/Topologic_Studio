/**
 * The sprinkler model against measurements, without the fire solver
 * (wiki/research/sprinkler-validation-data-2026-11-07.md, "Results
 * (2026-11-07)"; data in suppression.validation.json, NIST, with its notice).
 *
 * 1. Link replay. The gas temperature and speed measured 25 mm below the
 *    ceiling at heads 1 and 2 of Vettori's 27 smooth-ceiling tests (NISTIR
 *    6253) drive the link as activationFrame does for the "late" reading:
 *    gas held at the mean of each interval's two readings, speed floored at
 *    SPEED_FLOOR, the exact step (linkStep) and the exact crossing, with
 *    the heads' RTI 55 (m s)^1/2, C 0.25 (m/s)^1/2 (UL's plunge tests) and
 *    68 C. Of the 45 records with a measured activation:
 *      - 18 open within the record: predicted over measured has a median of
 *        1.00 and a range of 0.81 to 1.18 (the stated tolerance is 0.8 to
 *        1.2, median 0.95 to 1.05);
 *      - 18 run past the measured activation without opening: the replay is
 *        late in those, by at least 0 to 31 %. They are mostly the fast and
 *        medium fires and the wall and corner fires, where the rows about
 *        7 s apart and the thermocouple's own lag under-read a fast-rising
 *        gas;
 *      - 9 end before the measured activation and say nothing.
 *    So the link equation with these parameters is never more than 19 %
 *    early on the measured gas. With C = 0 (the FDS inputs' default) it is
 *    earlier (median 0.96, down to 0.76) and opens in 25 records.
 *
 * 2. Evans. The cap's decay time at each test's spray density, evansTau(w)
 *    = 3.0 w^-1.85 s, against the decay constants Evans fitted (NISTIR 5254
 *    Tables 1 and 2). The cap is the slower of the two, so it removes heat
 *    no faster than measured, for 19 of the 23 decay constants (9 in Table 1,
 *    the 0.61 m crib with two; 14 in Table 2). The exceptions:
 *      - Workstation I (713 s measured against the cap's 411 s at
 *        0.07 mm/s);
 *      - the 0.61 m crib in Table 1, second run (560 s against 411 s; the
 *        first run is 375 s);
 *      - the 0.61 m cribs of Table 2 at 0.066 and 0.081 mm/s (478 s against
 *        458 s, 326 s against 314 s): within Evans's own bracketing fits,
 *        4 % faster. The furnishing fit is the crib fit at a 0.61 m crib
 *        height, so these are the runs it was fitted to.
 *    The 0.305 m cribs decayed 1.8 to 7.1 times faster than the cap.
 */
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { EVANS_MIN_DENSITY, SPEED_FLOOR, createSuppression, crossing, evansFactor, evansTau, linkStep } from "./suppression.js";

const data = JSON.parse(readFileSync(new URL("./suppression.validation.json", import.meta.url), "utf8"));
const K = 273.15;

/** The link driven by a measured gas record: the activation time (s), or null and the link's last temperature (C). */
function replay(time, gas, speed, ambientC, rti, c, activationC) {
  const Tm = ambientC + K, Ta = activationC + K;
  let T = Tm;
  for (let i = 0; i + 1 < time.length; i += 1) {
    const Tg = 0.5 * (gas[i] + gas[i + 1]) + K;
    const u = Math.max(SPEED_FLOOR, 0.5 * (speed[i] + speed[i + 1]));
    const step = linkStep(T, Tg, Tm, u, rti, c, time[i + 1] - time[i]);
    if (T < Ta && step.T >= Ta) return { time: time[i] + crossing(T, step.Tinf, step.k, Ta), link: null };
    T = step.T;
  }
  return { time: null, link: T - K };
}

/** Every record, sorted into: opens within the record, runs past the measured time without opening, ends before it. */
function classify(rti, c) {
  const opens = [], late = [], silent = [];
  for (const rec of data.vettori_link_records.tests) {
    for (const h of rec.heads) {
      const r = replay(rec.time_s, h.gas_C, h.speed_m_per_s, rec.ambient_C, rti, c, 68);
      const end = rec.time_s.at(-1);
      const id = `test ${rec.test} head ${h.head} (${rec.growth}, ${rec.burner_position})`;
      if (r.time !== null) opens.push({ id, ratio: r.time / h.activation_s });
      else if (end > h.activation_s) late.push({ id, atLeast: end / h.activation_s, link: r.link });
      else silent.push({ id });
    }
  }
  const ratios = opens.map((o) => o.ratio).sort((a, b) => a - b);
  return { opens, late, silent, ratios, median: ratios[ratios.length >> 1] };
}

test("link replay: on Vettori's measured gas, RTI 55 and C 0.25 open within 0.8 to 1.2 of the measured time, never earlier", () => {
  const { head } = data.vettori_link_records;
  assert.equal(head.RTI_m05_s05, 55);
  assert.equal(head.C_m05_per_s05, 0.25);
  const r = classify(55, 0.25);
  assert.equal(r.opens.length + r.late.length + r.silent.length, 45);
  assert.equal(r.opens.length, 18);
  assert.equal(r.late.length, 18);
  assert.equal(r.silent.length, 9);
  assert.ok(r.median >= 0.95 && r.median <= 1.05, `median ${r.median}`);
  for (const o of r.opens) assert.ok(o.ratio >= 0.8 && o.ratio <= 1.2, `${o.id}: ${o.ratio.toFixed(2)}`);
  // Where it does not open, the record ran past the measured time: late, not early.
  for (const l of r.late) assert.ok(l.atLeast >= 1 && l.link < 68, `${l.id}`);
  // The late records are the fast-rising gas: no slow detached fire is late by more than 12 %.
  assert.ok(r.late.filter((l) => l.id.includes("slow, detached")).every((l) => l.atLeast < 1.12));
});

test("link replay: C matters; with C = 0 (the FDS inputs' default) the link opens earlier", () => {
  const withC = classify(55, 0.25);
  const without = classify(55, 0);
  assert.ok(without.opens.length > withC.opens.length);
  assert.ok(without.median < withC.median, `${without.median} against ${withC.median}`);
  assert.ok(without.ratios[0] < 0.8, "without C one record opens more than 20 % early");
});

test("Evans: the cap's decay time is evansTau at the spray density, and the cap removes heat no faster than measured in 19 of 23 decay constants", () => {
  // The cap as solver code applies it: a cell armed at t = 0 and burning at a steady rate is allowed rate exp(-t / tau).
  for (const w of [0.026, 0.07, 0.133]) {
    const sup = createSuppression(1);
    sup.tau[0] = evansTau(w);
    sup.tAct[0] = 0;
    const dt = 0.5, loss = 2e-3;
    assert.equal(evansFactor(sup, 0, loss, 0, dt), 1); // arms: reads the rate
    const tau = evansTau(w);
    const f = evansFactor(sup, 0, loss, tau, dt);
    assert.ok(Math.abs(f - Math.exp(-1)) < 1e-6, `w ${w}: factor ${f} at t = tau`);
  }
  assert.ok(Math.abs(evansTau(0.07) - 410.9) < 0.1);

  const exceptions = [];
  let runs = 0;
  for (const f of data.evans.table1_furnishings) {
    const tau = evansTau(f.density_mm_per_s);
    for (const d of f.decay_s) {
      runs += 1;
      if (tau < d) exceptions.push(`${f.item.split(" (")[0]}: ${d} s measured, cap ${tau.toFixed(0)} s`);
    }
  }
  for (const c of data.evans.table2_cribs) {
    runs += 1;
    assert.ok(c.density_mm_per_s >= EVANS_MIN_DENSITY, "Evans's crib runs are all at or above the cap's lowest density");
    const tau = evansTau(c.density_mm_per_s);
    if (tau < c.decay_s) {
      exceptions.push(`${c.crib_height_mm} mm crib at ${c.density_mm_per_s} mm/s: ${c.decay_s} s measured, cap ${tau.toFixed(0)} s`);
      // Faster only within Evans's own bracketing fits.
      assert.ok(tau >= c.decay_range_s[0], `${c.crib_height_mm} mm crib at ${c.density_mm_per_s} mm/s`);
    } else if (c.crib_height_mm === 305) {
      assert.ok(tau / c.decay_s >= 1.8, "the 0.305 m cribs decayed at least 1.8 times faster than the cap");
    }
  }
  assert.equal(runs, 23);
  assert.deepEqual(exceptions.map((e) => e.split(":")[0]), [
    "Workstation I",
    "Square base wood crib",
    "610 mm crib at 0.066 mm/s",
    "610 mm crib at 0.081 mm/s",
  ]);
});
