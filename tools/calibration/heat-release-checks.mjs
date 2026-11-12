/**
 * The fire's own heat release in UL FSRI's 2018 series
 * (fire-calibration-plan-2026-11-11.md, section 15), from the series'
 * oxygen consumption record (channel "Heat Release Rate", kW: 02_Info's
 * channel list and 03_Scripts/plot.py's axis label), no model run:
 *
 *   1. (15.5) the burner's nominal size: for each open-door burner test of
 *      250 or 500 kW with a record, the mean measured heat release over the
 *      steady period against the nominal size, as a ratio, and the median
 *      ratio against 0.95 to 1.05. The steady period is the events file's,
 *      from "Ignition" to "Burner Off" (the gas held at its setting), the
 *      record's 1 s readings in it as they are (no lag shift, no baseline
 *      taken off). Beside it, not pre-registered: the same period less its
 *      first 60 s, and the record's energy from ignition to its end over the
 *      period's length (the hood's delay moves energy past Burner Off but
 *      keeps it in the record).
 *   2. (15.3, the FSRI part) the design fire's curve against the chair
 *      tests: for each open-door chair test with a record, the peak, the
 *      energy released and a growth coefficient fitted by least squares as
 *      alpha (t - t0)^2 with t0 free, from the first reading above 25 kW to
 *      the first at 90 % of the peak; set beside the app's medium and fast
 *      classes (fireClasses.js GROWTH_CLASSES). The record is noisy, so the
 *      readings are the suite's 5 s means (validation-data.mjs: each the mean
 *      of the 1 s readings in (s - 2.5, s + 2.5], s = 0, 5, 10 ... s from
 *      ignition); the raw 1 s peak is reported beside, and two second routes
 *      for alpha (the same method on the raw readings; a straight-line fit of
 *      the square root of the 5 s means). No verdict word is attached (15.3).
 *
 * The raw files come from github.com/ulfsri/fsri-compartments-2018 at the
 * commit validation-data.mjs pins (COMMITS.fsri), into its git-ignored cache
 * (fetched when missing; --offline reads the cache only). Only the reduced
 * numbers are written:
 *
 *   node tools/calibration/heat-release-checks.mjs [--offline] [--json out.json]
 *
 * (default out: benchmark/results/2026-11-12-heat-release-checks.json).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GROWTH_CLASSES } from "../../topologicpy-web-frontend/src/core/fireClasses.js";
import { COMMITS, FSRI_NOTICE, fetchFile } from "./validation-data.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const OUT = flag("--json", path.join(ROOT, "benchmark", "results", "2026-11-12-heat-release-checks.json"));

/** The series' open-door tests at the pinned commit (its file tree, read 2026-11-12). */
const BURNER_TESTS = [
  ...["Back", "Center", "Corner", "Side"].flatMap((p) => [1, 2, 3].map((n) => `250kW_Open_${p}_${n}`)),
  ...["Back", "Center"].flatMap((p) => [1, 2, 3].map((n) => `500kW_Open_${p}_${n}`)),
  "500kW_Open_Corner_1",
  ...[1, 2, 3].map((n) => `500kW_Open_Side_${n}`),
];
/** The open-door 100 kW tests: outside 15.5's set (not above 100 kW); their ratio is kept apart. */
const ASIDE_TESTS = ["Back", "Center", "Corner", "Side"].flatMap((p) => [1, 2, 3].map((n) => `100kW_Open_${p}_${n}`));
const CHAIR_TESTS = [
  "RedAccent_Open_Back_1",
  ...[1, 2, 3].map((n) => `RedAccent_Open_Center_${n}`),
  ...[1, 2, 3].map((n) => `RedAccent_Open_Corner_${n}`),
  ...[1, 2, 3].map((n) => `Overstuffed_Open_Back_${n}`),
  "Overstuffed_Open_Center_1",
  ...[1, 2, 3].map((n) => `Overstuffed_Open_Corner_${n}`),
];

const HRR = "Heat Release Rate";
const BAND = [0.95, 1.05]; // 15.5
const START_KW = 25; // 15.3
const PEAK_FRACTION = 0.9; // 15.3
const STEP = 5; // the suite's 5 s means
const LEAD_S = 60; // the sensitivity's dropped lead (not pre-registered)

const r = (x, d = 3) => (x == null || !Number.isFinite(x) ? null : +x.toFixed(d));
const sum = (v) => v.reduce((a, b) => a + b, 0);
const mean = (v) => (v.length ? sum(v) / v.length : NaN);
const median = (v) => {
  const s = v.slice().sort((a, b) => a - b), m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
};
const clock = (s) => Date.parse(s.trim().replace(/^(\d{4}-\d\d-\d\d)[- ]/, "$1T") + "Z") / 1000;

const sources = [];
async function read(file) {
  const f = await fetchFile("fsri", file);
  sources.push(f.source);
  return f.text;
}

/** A test's record: {t (s from ignition), q (kW, NaN for blanks), events [{e, t from ignition}], timeCheck} or {missing}. */
async function record(name) {
  const evText = await read(`01_Data/${name}_Events.csv`);
  const text = await read(`01_Data/${name}.csv`);
  const ev = evText.split(/\r?\n/).slice(1).filter((l) => l.trim()).map((l) => l.split(",")).map(([t, e]) => ({ clock: clock(t), e: e.trim() }));
  const ign = ev.find((e) => /^ignition$/i.test(e.e));
  if (!ign) throw new Error(`${name}: no Ignition event`);
  const events = ev.map((e) => ({ e: e.e, t: e.clock - ign.clock }));
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const head = lines[0].split(",").map((s) => s.trim());
  const iT = head.indexOf("Time"), iS = head.indexOf("Timestamp"), iQ = head.indexOf(HRR);
  if (iQ < 0) return { name, missing: `no "${HRR}" column`, events };
  const rows = lines.slice(1).map((l) => l.split(","));
  const t = rows.map((x) => Number(x[iT]));
  const q = rows.map((x) => (x[iQ] == null || x[iQ].trim() === "" ? NaN : Number(x[iQ])));
  // The file's Time against the events' clock: the row at Time 0 is stamped at Ignition.
  const k0 = t.indexOf(0);
  const timeCheck = k0 >= 0 && iS >= 0 ? clock(rows[k0][iS]) - ign.clock : null;
  return { name, t, q, events, timeCheck };
}

/** The suite's 5 s means of the readings at t >= 0: [{t, q}] (blocks with no reading dropped). */
function fiveSecond(t, q) {
  const end = Math.max(...t.filter((x, k) => Number.isFinite(q[k])));
  const out = [];
  for (let s = 0; s <= end; s += STEP) {
    const v = q.filter((x, k) => Number.isFinite(x) && t[k] > s - STEP / 2 && t[k] <= s + STEP / 2);
    if (v.length) out.push({ t: s, q: mean(v) });
  }
  return out;
}

/**
 * Least squares of q = alpha max(0, t - t0)^2 with t0 free: for each t0 the
 * best alpha is sum(q u^2) / sum(u^4) (u = max(0, t - t0)); t0 is scanned in
 * 0.5 s steps over [first t - 1200, last t), then refined by golden section
 * on the bracketing steps.
 */
function fitAlpha(pts) {
  const sse = (t0) => {
    let a = 0, b = 0, c = 0;
    for (const p of pts) {
      const u = Math.max(0, p.t - t0), u2 = u * u;
      a += p.q * u2;
      b += u2 * u2;
      c += p.q * p.q;
    }
    return b > 0 ? { sse: c - (a * a) / b, alpha: a / b } : { sse: c, alpha: 0 };
  };
  const lo = pts[0].t - 1200, hi = pts[pts.length - 1].t;
  let best = lo;
  for (let t0 = lo; t0 < hi; t0 += 0.5) if (sse(t0).sse < sse(best).sse) best = t0;
  let a = best - 0.5, b = Math.min(best + 0.5, hi - 1e-6);
  const g = (Math.sqrt(5) - 1) / 2;
  for (let i = 0; i < 60; i += 1) {
    const x1 = b - g * (b - a), x2 = a + g * (b - a);
    if (sse(x1).sse < sse(x2).sse) b = x2;
    else a = x1;
  }
  const t0 = (a + b) / 2, s = sse(t0);
  const qm = mean(pts.map((p) => p.q));
  const sst = sum(pts.map((p) => (p.q - qm) ** 2));
  return { alpha: s.alpha, t0, rmsKW: Math.sqrt(s.sse / pts.length), r2: 1 - s.sse / sst, n: pts.length };
}

/** Ordinary least squares of sqrt(q) = a (t - t0): alpha = a^2 (a second route; it weights early readings more). */
function fitSqrt(pts) {
  const x = pts.map((p) => p.t), y = pts.map((p) => Math.sqrt(Math.max(0, p.q)));
  const mx = mean(x), my = mean(y);
  const a = sum(x.map((xi, k) => (xi - mx) * (y[k] - my))) / sum(x.map((xi) => (xi - mx) ** 2));
  return { alpha: a * a, t0: mx - my / a };
}

/** The 15.3 window on a series [{t, q}] at t >= 0: first q > 25 kW to the first q >= 0.9 peak. */
function window(pts) {
  let peak = pts[0];
  for (const p of pts) if (p.q > peak.q) peak = p;
  const i0 = pts.findIndex((p) => p.q > START_KW);
  const i1 = pts.findIndex((p) => p.q >= PEAK_FRACTION * peak.q);
  const win = pts.slice(i0, i1 + 1);
  // Readings in the window that fall back to 25 kW or below (a blip before the growth would show here).
  const dips = win.filter((p) => p.q <= START_KW).map((p) => p.t);
  return { peak, start: pts[i0], end: pts[i1], win, dips };
}

const placeOf = (alpha) => {
  const names = Object.entries(GROWTH_CLASSES).sort((a, b) => a[1] - b[1]);
  if (alpha < names[0][1]) return `below ${names[0][0]}`;
  for (let i = 0; i + 1 < names.length; i += 1) if (alpha < names[i + 1][1]) return `between ${names[i][0]} and ${names[i + 1][0]}`;
  return `above ${names[names.length - 1][0]}`;
};

// ------------------------------------------------------------ 1. the burner's nominal size (15.5)

async function burnerRow(name) {
  const R = await record(name);
  const nominal = Number(name.split("kW")[0]);
  if (R.missing) return { id: name, nominalKW: nominal, record: false, why: R.missing };
  const off = R.events.find((e) => /^burner off$/i.test(e.e));
  if (!off) return { id: name, nominalKW: nominal, record: true, blocked: "no Burner Off event" };
  const inP = (a, b) => R.q.filter((v, k) => R.t[k] >= a && R.t[k] < b);
  const steady = inP(0, off.t);
  const blanks = steady.filter((v) => !Number.isFinite(v)).length;
  const m = mean(steady.filter(Number.isFinite));
  const mLead = mean(inP(LEAD_S, off.t).filter(Number.isFinite));
  let E = 0; // kJ, 1 s readings from ignition to the record's end
  for (let k = 0; k < R.t.length; k += 1) if (R.t[k] >= 0 && Number.isFinite(R.q[k])) E += R.q[k] * (k + 1 < R.t.length ? R.t[k + 1] - R.t[k] : 1);
  const pre = mean(inP(-60, 0).filter(Number.isFinite));
  // How steady the record is: the standard deviation of its 5 s means over the period less its first 60 s.
  const five = fiveSecond(R.t, R.q).filter((p) => p.t >= LEAD_S && p.t < off.t).map((p) => p.q);
  const m5 = mean(five);
  const sd5 = Math.sqrt(sum(five.map((x) => (x - m5) ** 2)) / (five.length - 1));
  return {
    id: name,
    nominalKW: nominal,
    record: true,
    steadyPeriodS: [0, off.t],
    readings: steady.length,
    blankReadings: blanks,
    timeColumnVsIgnitionS: R.timeCheck,
    meanKW: r(m, 1),
    ratio: r(m / nominal),
    preIgnitionMeanKW: r(pre, 1),
    sd5sAfter60sKW: r(sd5, 1),
    sensitivity: {
      lessFirst60s: { meanKW: r(mLead, 1), ratio: r(mLead / nominal) },
      energyOverPeriod: { energyMJ: r(E / 1000, 2), meanKW: r(E / off.t, 1), ratio: r(E / off.t / nominal) },
    },
  };
}

/** n, median (the mean of the middle two when n is even), min and max; with band, whether the median is inside 15.5's band. */
const summarise = (rows, key, d = 3, band = false) => {
  const v = rows.map(key).filter((x) => x != null);
  if (!v.length) return null;
  const med = median(v);
  return { n: v.length, median: r(med, d), min: r(Math.min(...v), d), max: r(Math.max(...v), d), ...(band ? { insideBand: med >= BAND[0] && med <= BAND[1] } : {}) };
};

const burner = [];
for (const name of BURNER_TESTS) burner.push(await burnerRow(name));
const aside = [];
for (const name of ASIDE_TESTS) {
  const row = await burnerRow(name);
  if (row.record) aside.push(row);
}
const withRecord = burner.filter((b) => b.record && !b.blocked);
const check1 = {
  plan: "fire-calibration-plan-2026-11-11.md, section 15.5",
  set: "open-door gas-burner tests of nominal 250 or 500 kW (above 100 kW)",
  steadyPeriod: "from the events file's Ignition to its Burner Off (the file's Time column is seconds from Ignition; checked: the row at Time 0 carries Ignition's clock time); every 1 s reading with 0 <= t < Burner Off, as recorded (no lag shift, no pre-ignition baseline taken off)",
  band: BAND,
  tests: burner,
  noRecord: burner.filter((b) => !b.record).map((b) => `${b.id} (${b.why})`),
  summary: {
    preRegistered: summarise(withRecord, (b) => b.ratio, 3, true),
    by250: summarise(withRecord.filter((b) => b.nominalKW === 250), (b) => b.ratio, 3, true),
    by500: summarise(withRecord.filter((b) => b.nominalKW === 500), (b) => b.ratio, 3, true),
    notPreRegistered: {
      lessFirst60s: summarise(withRecord, (b) => b.sensitivity.lessFirst60s.ratio, 3, true),
      energyOverPeriod: summarise(withRecord, (b) => b.sensitivity.energyOverPeriod.ratio, 3, true),
    },
  },
  aside100kW: { note: "nominal 100 kW: outside 15.5's set (above 100 kW); kept apart, not in any summary", tests: aside },
};

// ------------------------------------------------------------ 2. the design fire against the chair tests (15.3)

const massText = await read("02_Info/furniture_mass.csv");
const MASS = Object.fromEntries(massText.split(/\r?\n/).slice(1).filter((l) => l.trim()).map((l) => l.split(",")).map(([n, m]) => [n.trim(), Number(m)]));

const chairs = [];
for (const name of CHAIR_TESTS) {
  const R = await record(name);
  const kind = name.split("_")[0];
  if (R.missing) {
    chairs.push({ id: name, kind, record: false, why: R.missing });
    continue;
  }
  const supp = R.events.find((e) => /suppression/i.test(e.e));
  const five = fiveSecond(R.t, R.q);
  const W = window(five);
  const raw = R.t.map((t, k) => ({ t, q: R.q[k] })).filter((p) => p.t >= 0 && Number.isFinite(p.q));
  const WR = window(raw);
  const fit = fitAlpha(W.win);
  const fitRaw = fitAlpha(WR.win);
  const lin = fitSqrt(W.win);
  let E = 0, Es = 0; // kJ
  for (let k = 0; k < R.t.length; k += 1) {
    if (R.t[k] < 0 || !Number.isFinite(R.q[k])) continue;
    const dt = k + 1 < R.t.length ? R.t[k + 1] - R.t[k] : 1;
    E += R.q[k] * dt;
    if (supp && R.t[k] < supp.t) Es += R.q[k] * dt;
  }
  const atSupp = supp ? five.reduce((a, p) => (Math.abs(p.t - supp.t) < Math.abs(a.t - supp.t) ? p : a), five[0]) : null;
  const mass = MASS[name] ?? null;
  chairs.push({
    id: name,
    kind,
    record: true,
    timeColumnVsIgnitionS: R.timeCheck,
    events: R.events.map((e) => `${e.e} @ ${e.t} s`),
    recordEndS: Math.max(...raw.map((p) => p.t)),
    peakKW: r(W.peak.q, 1),
    peakAtS: W.peak.t,
    rawPeakKW: r(WR.peak.q, 1),
    rawPeakAtS: WR.peak.t,
    energyMJ: r(E / 1000, 1),
    energyToSuppressionMJ: supp ? r(Es / 1000, 1) : null,
    hrrAtSuppressionKW: atSupp ? r(atSupp.q, 1) : null,
    massKg: mass,
    energyPerInitialMassMJkg: mass ? r(E / 1000 / mass, 2) : null,
    fit: {
      windowS: [W.start.t, W.end.t],
      windowKW: [r(W.start.q, 1), r(W.end.q, 1)],
      readingsAtOrBelow25kWInWindow: W.dips,
      alpha: r(fit.alpha, 5),
      t0S: r(fit.t0, 1),
      // t0 after the window's start: the readings before it are fitted as zero (a slower first stage the curve skips).
      t0AfterWindowStart: fit.t0 > W.start.t,
      n: fit.n,
      rmsKW: r(fit.rmsKW, 1),
      r2: r(fit.r2, 4),
      overMedium: r(fit.alpha / GROWTH_CLASSES.medium, 2),
      overFast: r(fit.alpha / GROWTH_CLASSES.fast, 2),
      place: placeOf(fit.alpha),
    },
    secondRoutes: {
      raw1s: { windowS: [WR.start.t, WR.end.t], alpha: r(fitRaw.alpha, 5), t0S: r(fitRaw.t0, 1), n: fitRaw.n, readingsAtOrBelow25kWInWindow: WR.dips.length },
      sqrtLine5s: { alpha: r(lin.alpha, 5), t0S: r(lin.t0, 1) },
    },
  });
}
const byKind = {};
for (const kind of ["RedAccent", "Overstuffed"]) {
  const rows = chairs.filter((c) => c.kind === kind && c.record);
  const where = (s) => ({
    ...s,
    ...Object.fromEntries(["medium", "fast"].map((k) => [k, GROWTH_CLASSES[k] < s.min ? "below the range" : GROWTH_CLASSES[k] > s.max ? "above the range" : "inside the range"])),
  });
  byKind[kind] = {
    n: rows.length,
    alpha: where(summarise(rows, (c) => c.fit.alpha, 5)),
    alphaRaw1s: where(summarise(rows, (c) => c.secondRoutes.raw1s.alpha, 5)),
    alphaSqrtLine5s: where(summarise(rows, (c) => c.secondRoutes.sqrtLine5s.alpha, 5)),
    peakKW: summarise(rows, (c) => c.peakKW, 1),
    rawPeakKW: summarise(rows, (c) => c.rawPeakKW, 1),
    energyMJ: summarise(rows, (c) => c.energyMJ, 1),
    energyPerInitialMassMJkg: summarise(rows, (c) => c.energyPerInitialMassMJkg, 2),
  };
}
const check2 = {
  plan: "fire-calibration-plan-2026-11-11.md, section 15.3 (the FSRI part; NIST's Fire Calorimetry Database not read here)",
  set: "open-door upholstered chair tests (RedAccent: the README's Red Accent Chair; Overstuffed: the README's Overstuffed Sofa, the plan's overstuffed chair)",
  readings: `the suite's ${STEP} s means of the 1 s record (each the mean of the readings in (s - 2.5, s + 2.5], s = 0, 5, 10 ... s from Ignition), for the peak, the window and the fit`,
  window: `from the first 5 s mean above ${START_KW} kW (t >= 0) to the first at ${PEAK_FRACTION * 100} % of the 5 s peak, both included`,
  fit: "least squares of q = alpha max(0, t - t0)^2, t0 free (scanned in 0.5 s steps over [window start - 1200 s, window end), refined by golden section)",
  energy: "the 1 s readings summed from Ignition to the record's end (kJ to MJ), and to the Suppression event; no baseline taken off",
  massSource: "02_Info/furniture_mass.csv (initial mass only; no mass-loss record exists in the series, so the energy is not recomputed from mass loss)",
  classes: { medium: GROWTH_CLASSES.medium, fast: GROWTH_CLASSES.fast, all: GROWTH_CLASSES, source: "topologicpy-web-frontend/src/core/fireClasses.js GROWTH_CLASSES" },
  tests: chairs,
  noRecord: chairs.filter((c) => !c.record).map((c) => `${c.id} (${c.why})`),
  byKind,
};

const doc = {
  generated: new Date().toISOString().slice(0, 10),
  tool: "tools/calibration/heat-release-checks.mjs",
  plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 15 (15.3 and 15.5)",
  data: { repo: "github.com/ulfsri/fsri-compartments-2018", commit: COMMITS.fsri, channel: HRR, units: "kW (03_Scripts/plot.py's axis label 'Heat Release Rate (kW)')" },
  notice: `${FSRI_NOTICE.split(" Changed here")[0]} Changed here (2026-11-12): the heat release channel reduced to means, peaks, energies and fitted growth coefficients by tools/calibration/heat-release-checks.mjs.`,
  parameters: { band: BAND, startKW: START_KW, peakFraction: PEAK_FRACTION, meanStepS: STEP, sensitivityLeadS: LEAD_S },
  check1,
  check2,
  sources: sources.filter((s, q) => sources.findIndex((x) => x.path === s.path) === q),
};
mkdirSync(path.dirname(path.resolve(OUT)), { recursive: true });
writeFileSync(path.resolve(OUT), JSON.stringify(doc, null, 1) + "\n");

// ------------------------------------------------------------ the tables

const f = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));
console.log("1. the burner's nominal size (15.5): mean over Ignition to Burner Off, against nominal");
console.log("   test                      nominal  period s  mean kW  ratio   | less 60 s  energy/period  pre-ign kW  sd 5 s kW");
for (const b of burner) {
  if (!b.record) {
    console.log(`   ${b.id.padEnd(25)} ${String(b.nominalKW).padStart(7)}  no record (${b.why})`);
    continue;
  }
  console.log(`   ${b.id.padEnd(25)} ${String(b.nominalKW).padStart(7)}  ${String(b.steadyPeriodS[1]).padStart(8)}  ${f(b.meanKW, 1).padStart(7)}  ${f(b.ratio)}  | ${f(b.sensitivity.lessFirst60s.ratio).padStart(9)}  ${f(b.sensitivity.energyOverPeriod.ratio).padStart(13)}  ${f(b.preIgnitionMeanKW, 1).padStart(10)}  ${f(b.sd5sAfter60sKW, 1).padStart(9)}`);
}
const S1 = check1.summary;
const line = (label, s) => console.log(`   ${label}: n ${s.n}, median ${f(s.median)} (range ${f(s.min)} to ${f(s.max)}), median inside ${BAND[0]} to ${BAND[1]}: ${s.insideBand ? "yes" : "no"}`);
line("pre-registered", S1.preRegistered);
line("250 kW only", S1.by250);
line("500 kW only", S1.by500);
line("less first 60 s (not pre-registered)", S1.notPreRegistered.lessFirst60s);
line("energy over period (not pre-registered)", S1.notPreRegistered.energyOverPeriod);
console.log(`   no record: ${check1.noRecord.length} of ${burner.length}: ${check1.noRecord.join("; ")}`);
const offsets = {};
for (const x of [...burner, ...chairs]) if (x.record) (offsets[x.timeColumnVsIgnitionS] ||= []).push(x.id);
console.log(`   the row at Time 0 stamped this many s after Ignition (the events' clock): ${Object.entries(offsets).map(([k, v]) => `${k} s in ${v.length} tests`).join(", ")}`);
for (const b of aside) console.log(`   aside (100 kW, outside the set): ${b.id} ratio ${f(b.ratio)}`);

console.log("\n2. the design fire against the chair tests (15.3): 5 s means; alpha in kW/s^2");
console.log("   test                         peak kW (raw)      at s   energy MJ (to supp.)  MJ/kg  window s    alpha    t0 s    r2     raw 1 s  sqrt-line  place");
for (const c of chairs) {
  if (!c.record) {
    console.log(`   ${c.id.padEnd(28)} no record (${c.why})`);
    continue;
  }
  console.log(`   ${c.id.padEnd(28)} ${f(c.peakKW, 0).padStart(5)} (${f(c.rawPeakKW, 0).padStart(5)})  ${String(c.peakAtS).padStart(5)}   ${f(c.energyMJ, 1).padStart(6)} (${f(c.energyToSuppressionMJ, 1).padStart(6)})   ${f(c.energyPerInitialMassMJkg, 1).padStart(5)}  ${`${c.fit.windowS[0]}-${c.fit.windowS[1]}`.padEnd(10)} ${f(c.fit.alpha, 4)}  ${f(c.fit.t0S, 1).padStart(6)}${c.fit.t0AfterWindowStart ? "*" : " "} ${f(c.fit.r2, 3)}   ${f(c.secondRoutes.raw1s.alpha, 4)}   ${f(c.secondRoutes.sqrtLine5s.alpha, 4)}    ${c.fit.place}`);
}
console.log("   (* t0 after the window's start: the readings before it are fitted as zero)");
const rng = (s, d) => `median ${f(s.median, d)} (range ${f(s.min, d)} to ${f(s.max, d)})`;
for (const [kind, s] of Object.entries(byKind)) {
  console.log(`   ${kind} (n ${s.n}): alpha ${rng(s.alpha, 4)}; medium ${GROWTH_CLASSES.medium} ${s.alpha.medium}, fast ${GROWTH_CLASSES.fast} ${s.alpha.fast}`);
  console.log(`      raw 1 s alpha ${rng(s.alphaRaw1s, 4)} (medium ${s.alphaRaw1s.medium}, fast ${s.alphaRaw1s.fast}); sqrt-line alpha ${rng(s.alphaSqrtLine5s, 4)} (medium ${s.alphaSqrtLine5s.medium}, fast ${s.alphaSqrtLine5s.fast})`);
  console.log(`      peak ${rng(s.peakKW, 0)} kW; raw peak ${rng(s.rawPeakKW, 0)} kW; energy ${rng(s.energyMJ, 1)} MJ; per initial mass ${rng(s.energyPerInitialMassMJkg, 1)} MJ/kg`);
}
console.log(`   no record: ${check2.noRecord.join("; ") || "none"}`);
console.log(`\nwritten: ${path.relative(ROOT, path.resolve(OUT)).split(path.sep).join("/")}`);
