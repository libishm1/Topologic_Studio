/**
 * The fire's own heat release (fire-calibration-plan-2026-11-11.md, section
 * 15), from oxygen consumption records, no model run. Two parts.
 *
 * UL FSRI's 2018 series (the default), from the series' oxygen consumption
 * record (channel "Heat Release Rate", kW: 02_Info's channel list and
 * 03_Scripts/plot.py's axis label):
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
 * NIST's Fire Calorimetry Database (--nist, added 2026-11-13):
 *
 *   3. (15.3, the NIST part) the same check as 2, through the same functions
 *      (itemCurve, the fit and the summaries), on the upholstered chairs and
 *      sofas the database holds (NIST_ITEMS below, chosen from its listing
 *      before any record was opened): the same 5 s means, window, fit,
 *      second routes and classes. The energy is summed as in 2, from
 *      ignition to the record's end, and beside it to the page's "Fire Out"
 *      (the database's own window, as 2 sums to "Suppression"). Each page
 *      gives the item's initial and final mass (gravimetric), so the energy
 *      over the mass lost is an effective heat of combustion. Each page's
 *      own summary (its Table 2) is kept beside the reduction as a check.
 *      Beside it, not pre-registered (as 15.6 did for the FSRI chairs after
 *      reading its table): the fitted curve's time to 1 MW from ignition,
 *      t0 + sqrt(1000 / alpha), against the app's medium and fast curves'.
 *
 * The FSRI files come from github.com/ulfsri/fsri-compartments-2018 at the
 * commit validation-data.mjs pins (COMMITS.fsri); NIST's from www.nist.gov
 * (each test's page and the 1 s record it links; unversioned, so each file's
 * digest and the date it was read are recorded). Both go to
 * validation-data.mjs's git-ignored cache (fetched when missing; --offline
 * reads the cache only). Downloaded files are data: parsed, never run. Only
 * the reduced numbers are written:
 *
 *   node tools/calibration/heat-release-checks.mjs [--offline] [--json out.json]
 *   node tools/calibration/heat-release-checks.mjs --nist [--offline] [--json out.json]
 *
 * (default out: benchmark/results/2026-11-12-heat-release-checks.json; with
 * --nist, benchmark/results/2026-11-13-heat-release-checks-nist.json).
 */
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { GROWTH_CLASSES } from "../../topologicpy-web-frontend/src/core/fireClasses.js";
import { COMMITS, FSRI_CHAIR_TESTS, FSRI_NOTICE, fetchFile } from "./validation-data.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, "..", "..");
const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : fallback;
};
const NIST = args.includes("--nist");
const OUT = flag("--json", path.join(ROOT, "benchmark", "results", NIST ? "2026-11-13-heat-release-checks-nist.json" : "2026-11-12-heat-release-checks.json"));

/** The series' open-door tests at the pinned commit (its file tree, read 2026-11-12). */
const BURNER_TESTS = [
  ...["Back", "Center", "Corner", "Side"].flatMap((p) => [1, 2, 3].map((n) => `250kW_Open_${p}_${n}`)),
  ...["Back", "Center"].flatMap((p) => [1, 2, 3].map((n) => `500kW_Open_${p}_${n}`)),
  "500kW_Open_Corner_1",
  ...[1, 2, 3].map((n) => `500kW_Open_Side_${n}`),
];
/** The open-door 100 kW tests: outside 15.5's set (not above 100 kW); their ratio is kept apart. */
const ASIDE_TESTS = ["Back", "Center", "Corner", "Side"].flatMap((p) => [1, 2, 3].map((n) => `100kW_Open_${p}_${n}`));
/** The open-door chair tests (validation-data.mjs keeps the list since 2026-11-13: its fsri-chairs series is these with a record). */
const CHAIR_TESTS = FSRI_CHAIR_TESTS;

/**
 * NIST's upholstered chairs and sofas (15.3), chosen on 2026-11-13 from the
 * database's listing (www.nist.gov/el/fcd/experiments, searched for chair,
 * sofa, couch, loveseat, recliner, armchair, upholstered, mockup, barrier,
 * cushion, seat, furniture and foam) by its specimen names and descriptions,
 * before any record was opened: every single upholstered seat or sofa, and
 * every full-scale upholstered chair mock-up, burnt alone under the NFRL's
 * hood. The kinds are the listing's and the pages': chairs, sofas
 * (loveseats and sofas, with or without throw pillows), stacks of three
 * chairs (several items, kept apart), and the mock-ups without a fire
 * barrier, lit by a cigarette instead of the burner, and with a fire barrier
 * (a barrier is put there to slow the fire, so these are kept apart). Left
 * out, with the reason, in LEFT_OUT.
 */
const DF = "el/fcd/design-fires-residential-and-office-items";
const FB = "el/fcd/fire-barriers-full-scale-chair-mock-ups";
const NIST_ITEMS = [
  ...["test016", "test017", "test019", "test020", "test021", "test022", "test024", "test027", "test046", "test048"].map((t) => ({ kind: "chair", page: `${DF}/${t}` })),
  ...["test029", "test030", "test034", "test035", "test036", "test040", "test041"].map((t) => ({ kind: "sofa", page: `${DF}/${t}` })),
  ...["test025", "test045", "test047"].map((t) => ({ kind: "three stacked chairs", page: `${DF}/${t}` })),
  ...["c0-1", "c0-2", "c0-3"].map((t) => ({ kind: "chair mock-up, no barrier", page: `${FB}/${t}` })),
  // Its page: "Ignition with smodering cigarette"; the other mock-ups were lit by an 18 kW propane burner for 80 s.
  { kind: "chair mock-up, cigarette ignition", page: `${FB}/control-smoldering` },
  ...["b1-2", "b2-1", "b3-1", "b4-1", "b5-1", "b6-1"].map((t) => ({ kind: "chair mock-up, fire barrier", page: `${FB}/${t}` })),
  { kind: "chair mock-up, fire barrier", page: "el/fcd/d2" },
];
const NIST_KINDS = ["chair", "sofa", "three stacked chairs", "chair mock-up, no barrier", "chair mock-up, cigarette ignition", "chair mock-up, fire barrier"];
const LEFT_OUT = [
  "Design Fires - Residential and Office Items, Test023 (Beanbag Chair F): expanded polystyrene beads in a bag, not an upholstered seat",
  "Room Flashover Experiments to Demonstrate the Effectiveness of Barrier Fabrics, Test_1 to Test_10: each sofa burnt in a furnished room, so the record is the room's",
  "Dry Tree Fire Hazard, WateredTree3 and Dry_Pine_Tree: a pine tree beside an upholstered chair and a shelf, so the record is not the chair's",
  "Transient Combustion Calorimetry, Test084 to Test086 (PatioChair): plastic chairs, not upholstered",
  "Fire Barriers in Full-Scale Chair Mock-ups, Propane Burner Check: the burner alone",
];
/** The projects' reports, as each project's page names them (DOIs checked on Crossref 2026-11-13). */
const NIST_REPORTS = {
  "Design Fires - Residential and Office Items": "Hamins, A., Di Cristina, G., Bundy, M., Chakalis, T., Fernandez, M., Kim, I., Ramesh, S. and Zammarano, M. (2025). The character of burning residential and office items. NIST Technical Note 2303. doi:10.6028/NIST.TN.2303",
  "Fire Barriers in Full-Scale Chair Mock-ups": "Thompson, A. L., Kim, I., Hamins, A., Bundy, M. and Zammarano, M. (2022). Performance and failure mechanism of fire barriers in full-scale chair mock-ups. Fire and Materials 46, 329-346. doi:10.1002/fam.3007",
  "(none named)": "D2's page names no project and no report; the database is cited for it",
};

/**
 * NIST's terms, as read on its own pages on 2026-11-13 (plan 15.6 asked that they be read in full before use).
 * The database's DOI resolves to its page; NIST's data record for it (data.nist.gov, ark:/88434/mds2-2314,
 * version 1.0.2) gives its licence as https://www.nist.gov/open/license, which leads to NIST's statements
 * page; the data are not Standard Reference Data, so its "Fair Use of Other NIST Data/Works" applies.
 */
const NIST_TERMS = {
  database: {
    url: "https://www.nist.gov/el/fcd",
    sentences: [
      "This Fire Calorimetry Database (FCD) contains results of fire experiments conducted at the NFRL.",
      "Certain commercial entities, equipment, or materials may be identified in this database to describe an experimental procedure or concept adequately. Such identification is not intended to imply recommendation or endorsement by the National Institute of Standards and Technology, nor is it intended to imply that the entities, materials, or equipment are necessarily the best available for the purpose.",
    ],
    citationBox: ["Bryant, R. and Bundy, M. The NIST 20 MW Calorimetry Measurement System for Large-Fire Research, Technical Note (NIST TN) 2077, 2019 https://doi.org/10.6028/NIST.TN.2077", "NIST Fire Calorimetry Database (FCD) https://doi.org/10.18434/mds2-2314"],
  },
  record: { url: "https://data.nist.gov/rmm/records?@id=ark:/88434/mds2-2314", fields: { license: "https://www.nist.gov/open/license", accessLevel: "public", version: "1.0.2" } },
  userGuide: {
    url: "https://www.nist.gov/document/fcd-user-guide-pdf (served as https://www.nist.gov/system/files/documents/2020/11/19/FCD_User_Guide_v4a.pdf, revised 10/23/2020)",
    sentences: [
      "There are multiple citations for experiments contained in this database. Each project has at least one report or journal article that describes the test methods, results and conclusions of the study. This reference should be cited when referring to experiments from a single project. The NIST Technical Note 2077 [1] should be cited when referencing the calorimetry methods or uncertainty analysis in this database. Finally, the NIST Fire Calorimetry Database [2] should be cited when using video, images, or data files or summary calculations found here.",
    ],
  },
  licence: {
    url: "https://www.nist.gov/open/license (served as https://www.nist.gov/open/copyright-fair-use-and-licensing-statements-srd-data-software-and-technical-series-publications), section Fair Use of Other NIST Data/Works",
    sentences: [
      "Data/works created by NIST employees that are not covered by the Standard Reference Data Act are subject to 17 U.S.C. §105 and generally are not subject to copyright protection within the United States.",
      "Distributions of non-SRD products should include the following statement:",
      "To the extent that NIST may hold copyright in countries other than the United States, you are hereby granted the non-exclusive irrevocable and unconditional right to print, publish, prepare derivative works and distribute the NIST data, in any medium, or authorize others to do so on your behalf, on a royalty-free basis throughout the world.",
      "You may improve, modify, and create derivative works of the data or any portion of the data, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the data and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the data: Data citation recommendations are provided at https://www.nist.gov/open/license .",
      "Permission to use this data is contingent upon your acceptance of the terms of this agreement and upon your providing appropriate acknowledgments of NIST’s creation of the data/work.",
    ],
  },
};
/** The statement NIST asks distributions of its non-SRD data to carry (its licence page, as read), and the change made here. */
const FCD_NOTICE =
  "Data from the NIST Fire Calorimetry Database (FCD), National Institute of Standards and Technology (Bundy, M. and Bryant, R. (2020), NIST Fire Calorimetry Database (FCD), doi:10.18434/mds2-2314, read 2026-11-13); calorimetry method: Bryant, R. and Bundy, M. (2019), The NIST 20 MW calorimetry measurement system for large-fire research, NIST TN 2077, doi:10.6028/NIST.TN.2077; each project's report under `references`. " +
  "NIST's statement: This data/work was created by employees of the National Institute of Standards and Technology (NIST), an agency of the Federal Government. Pursuant to title 17 United States Code Section 105, works of NIST employees are not subject to copyright protection in the United States. This data/work may be subject to foreign copyright. " +
  "The data/work is provided by NIST as a public service and is expressly provided “AS IS.” NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED OR STATUTORY, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST does not warrant or make any representations regarding the use of the data or the results thereof, including but not limited to the correctness, accuracy, reliability or usefulness of the data. NIST SHALL NOT BE LIABLE AND YOU HEREBY RELEASE NIST FROM LIABILITY FOR ANY INDIRECT, CONSEQUENTIAL, SPECIAL, OR INCIDENTAL DAMAGES (INCLUDING DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF BUSINESS INFORMATION, AND THE LIKE), WHETHER ARISING IN TORT, CONTRACT, OR OTHERWISE, ARISING FROM OR RELATING TO THE DATA (OR THE USE OF OR INABILITY TO USE THIS DATA), EVEN IF NIST HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. " +
  "Changed here (2026-11-13): each test's heat release channel (1 s) reduced to peaks, energies and fitted growth coefficients, and its page's initial and final mass to an energy per mass lost, by tools/calibration/heat-release-checks.mjs; the pages' own summaries are copied beside as published. The National Institute of Standards and Technology is the source of the data.";

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
async function read(file, repo = "fsri") {
  const f = await fetchFile(repo, file);
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

/**
 * The 15.3 window on a series [{t, q}] at t >= 0: first q > 25 kW to the first q >= 0.9 peak. A record with
 * no reading above 25 kW, or fewer than three readings from there to 90 % of the peak, has no window (`why`;
 * added 2026-11-13 for NIST's items, where two never reach 25 kW; every FSRI chair has a window).
 */
function window(pts) {
  let peak = pts[0];
  for (const p of pts) if (p.q > peak.q) peak = p;
  const i0 = pts.findIndex((p) => p.q > START_KW);
  const i1 = pts.findIndex((p) => p.q >= PEAK_FRACTION * peak.q);
  if (i0 < 0 || i1 < i0 + 2) return { peak, start: null, end: null, win: [], dips: [], why: i0 < 0 ? `no reading above ${START_KW} kW` : `fewer than three readings from the first above ${START_KW} kW to the first at ${PEAK_FRACTION * 100} % of the peak` };
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

/**
 * 15.3's reading of one item's record R ({t: s from ignition, q: kW}), shared by the FSRI chairs and NIST's
 * items: the 5 s means and the raw readings at t >= 0, each one's window and fit, the square-root line, and
 * the energy (kJ) of the 1 s readings from ignition to the record's end and to `cut` s (an event), with the
 * 5 s mean nearest `cut`.
 */
function itemCurve(R, cut = null) {
  const five = fiveSecond(R.t, R.q);
  const W = window(five);
  const raw = R.t.map((t, k) => ({ t, q: R.q[k] })).filter((p) => p.t >= 0 && Number.isFinite(p.q));
  const WR = window(raw);
  const fit = W.win.length ? fitAlpha(W.win) : null;
  const fitRaw = WR.win.length ? fitAlpha(WR.win) : null;
  const lin = W.win.length ? fitSqrt(W.win) : null;
  let E = 0, Ecut = 0; // kJ
  for (let k = 0; k < R.t.length; k += 1) {
    if (R.t[k] < 0 || !Number.isFinite(R.q[k])) continue;
    const dt = k + 1 < R.t.length ? R.t[k + 1] - R.t[k] : 1;
    E += R.q[k] * dt;
    if (cut != null && R.t[k] < cut) Ecut += R.q[k] * dt;
  }
  const atCut = cut != null ? five.reduce((a, p) => (Math.abs(p.t - cut) < Math.abs(a.t - cut) ? p : a), five[0]) : null;
  return { five, W, raw, WR, fit, fitRaw, lin, E, Ecut, atCut };
}

/** The fit's fields of a result row (the same for both parts). */
const fitFields = (C) =>
  C.fit
    ? {
        windowS: [C.W.start.t, C.W.end.t],
        windowKW: [r(C.W.start.q, 1), r(C.W.end.q, 1)],
        readingsAtOrBelow25kWInWindow: C.W.dips,
        alpha: r(C.fit.alpha, 5),
        t0S: r(C.fit.t0, 1),
        // t0 after the window's start: the readings before it are fitted as zero (a slower first stage the curve skips).
        t0AfterWindowStart: C.fit.t0 > C.W.start.t,
        n: C.fit.n,
        rmsKW: r(C.fit.rmsKW, 1),
        r2: r(C.fit.r2, 4),
        overMedium: r(C.fit.alpha / GROWTH_CLASSES.medium, 2),
        overFast: r(C.fit.alpha / GROWTH_CLASSES.fast, 2),
        place: placeOf(C.fit.alpha),
      }
    : { why: C.W.why };
const secondFields = (C) => ({
  raw1s: C.fitRaw ? { windowS: [C.WR.start.t, C.WR.end.t], alpha: r(C.fitRaw.alpha, 5), t0S: r(C.fitRaw.t0, 1), n: C.fitRaw.n, readingsAtOrBelow25kWInWindow: C.WR.dips.length } : { why: C.WR.why },
  sqrtLine5s: C.lin ? { alpha: r(C.lin.alpha, 5), t0S: r(C.lin.t0, 1) } : { why: C.W.why },
});

/** n, median (the mean of the middle two when n is even), min and max; with band, whether the median is inside 15.5's band. */
const summarise = (rows, key, d = 3, band = false) => {
  const v = rows.map(key).filter((x) => x != null);
  if (!v.length) return null;
  const med = median(v);
  return { n: v.length, median: r(med, d), min: r(Math.min(...v), d), max: r(Math.max(...v), d), ...(band ? { insideBand: med >= BAND[0] && med <= BAND[1] } : {}) };
};
/** Where the app's medium and fast curves fall against a summary's range. */
const where = (s) =>
  s && {
    ...s,
    ...Object.fromEntries(["medium", "fast"].map((k) => [k, GROWTH_CLASSES[k] < s.min ? "below the range" : GROWTH_CLASSES[k] > s.max ? "above the range" : "inside the range"])),
  };
/** The coefficient's three routes by kind, the same for both parts. */
const alphaSummaries = (rows) => ({
  alpha: where(summarise(rows, (c) => c.fit.alpha, 5)),
  alphaRaw1s: where(summarise(rows, (c) => c.secondRoutes.raw1s.alpha, 5)),
  alphaSqrtLine5s: where(summarise(rows, (c) => c.secondRoutes.sqrtLine5s.alpha, 5)),
});

const f = (x, d = 3) => (x == null ? "-" : Number(x).toFixed(d));
const rng = (s, d) => (s ? `median ${f(s.median, d)} (range ${f(s.min, d)} to ${f(s.max, d)})` : "-");

// ------------------------------------------------------------ UL FSRI (15.5 and 15.3's FSRI part)

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

async function fsriPart() {
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
    const C = itemCurve(R, supp ? supp.t : null);
    const mass = MASS[name] ?? null;
    chairs.push({
      id: name,
      kind,
      record: true,
      timeColumnVsIgnitionS: R.timeCheck,
      events: R.events.map((e) => `${e.e} @ ${e.t} s`),
      recordEndS: Math.max(...C.raw.map((p) => p.t)),
      peakKW: r(C.W.peak.q, 1),
      peakAtS: C.W.peak.t,
      rawPeakKW: r(C.WR.peak.q, 1),
      rawPeakAtS: C.WR.peak.t,
      energyMJ: r(C.E / 1000, 1),
      energyToSuppressionMJ: supp ? r(C.Ecut / 1000, 1) : null,
      hrrAtSuppressionKW: C.atCut ? r(C.atCut.q, 1) : null,
      massKg: mass,
      energyPerInitialMassMJkg: mass ? r(C.E / 1000 / mass, 2) : null,
      fit: fitFields(C),
      secondRoutes: secondFields(C),
    });
  }
  const byKind = {};
  for (const kind of ["RedAccent", "Overstuffed"]) {
    const rows = chairs.filter((c) => c.kind === kind && c.record);
    byKind[kind] = {
      n: rows.length,
      ...alphaSummaries(rows),
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
  for (const [kind, s] of Object.entries(byKind)) {
    console.log(`   ${kind} (n ${s.n}): alpha ${rng(s.alpha, 4)}; medium ${GROWTH_CLASSES.medium} ${s.alpha.medium}, fast ${GROWTH_CLASSES.fast} ${s.alpha.fast}`);
    console.log(`      raw 1 s alpha ${rng(s.alphaRaw1s, 4)} (medium ${s.alphaRaw1s.medium}, fast ${s.alphaRaw1s.fast}); sqrt-line alpha ${rng(s.alphaSqrtLine5s, 4)} (medium ${s.alphaSqrtLine5s.medium}, fast ${s.alphaSqrtLine5s.fast})`);
    console.log(`      peak ${rng(s.peakKW, 0)} kW; raw peak ${rng(s.rawPeakKW, 0)} kW; energy ${rng(s.energyMJ, 1)} MJ; per initial mass ${rng(s.energyPerInitialMassMJkg, 1)} MJ/kg`);
  }
  console.log(`   no record: ${check2.noRecord.join("; ") || "none"}`);
}

// ------------------------------------------------------------ NIST's Fire Calorimetry Database (15.3's NIST part)

const cellText = (s) => s.replace(/<[^>]+>/g, " ").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/&#0?39;/g, "'").replace(/&quot;/g, '"').replace(/\s+/g, " ").trim();
/** A number as a page prints it ("1,733"), or null for words ("Not measured", "below detection limit"). */
const pageNumber = (s) => {
  const v = s == null ? NaN : Number(String(s).replace(/,/g, ""));
  return Number.isFinite(v) ? v : null;
};

/** A test's page: its tables by caption ("Table 1" to "Table 4"), each a list of rows of cell text, and its CSV link. */
function readPage(html) {
  const tables = {};
  for (const t of html.matchAll(/<table\b[\s\S]*?<\/table>/gi)) {
    const cap = cellText(t[0].match(/<caption[^>]*>([\s\S]*?)<\/caption>/i)?.[1] ?? "");
    const key = cap.match(/^Table \d+/)?.[0];
    if (key) tables[key] = [...t[0].matchAll(/<tr\b[\s\S]*?<\/tr>/gi)].map((row) => [...row[0].matchAll(/<t[hd]\b[^>]*>([\s\S]*?)<\/t[hd]>/gi)].map((c) => cellText(c[1])));
  }
  const csv = html.match(/href="\/fcd-s3\?path=\/([^"]+?\.csv)"/i)?.[1] ?? null;
  const row = (table, label) => (tables[table] || []).find((x) => x[0].startsWith(label)) ?? null;
  return { tables, csv, row };
}

/** A database record (1 s): {t (s from ignition, as the file's Time column), q (kW), burner (kW)}. */
function nistRecord(text) {
  const lines = text.split(/\r?\n/).filter((l) => l.trim());
  const head = lines[0].split(",").map((s) => s.trim());
  const col = (prefix) => head.findIndex((h) => h.startsWith(prefix));
  // The burner channel: "Burner HRR (kW)" (2023 items) or "NG Burner HRR (kW)" (2018 mock-ups; their propane burner is not metered).
  const iT = col("Time"), iQ = col(HRR), iB = head.findIndex((h) => /burner hrr/i.test(h));
  if (iT < 0 || iQ < 0) throw new Error(`no "Time" or "${HRR}" column: ${head.join(", ")}`);
  const rows = lines.slice(1).map((l) => l.split(","));
  const v = (x) => (x == null || x.trim() === "" ? NaN : Number(x));
  return { head, t: rows.map((x) => Number(x[iT])), q: rows.map((x) => v(x[iQ])), burner: iB >= 0 ? rows.map((x) => v(x[iB])) : null };
}

async function nistRow(item) {
  const P = readPage(await read(item.page, "fcd-page"));
  const t1 = (label) => P.row("Table 1", label)?.[1] ?? null;
  const t2 = (label) => {
    const x = P.row("Table 2", label);
    return x ? { value: pageNumber(x[1]), printed: x[1], uc: pageNumber(x[2]), unit: x[3] } : null;
  };
  const t3 = (label) => pageNumber(P.row("Table 3", label)?.[1]);
  const events = (P.tables["Table 4"] || []).filter((x) => x.length >= 3 && pageNumber(x[1]) != null).map((x) => ({ e: x[2], t: r(pageNumber(x[1]) * 60, 1) }));
  const ign = events.find((e) => /^ignition$/i.test(e.e));
  const out = events.find((e) => /^fire out$/i.test(e.e));
  const base = { id: t1("Name"), kind: item.kind, project: t1("Project ID (s)") || "(none named)", specimen: t1("Specimen"), description: t1("Description"), ignition: t1("Ignition"), page: `https://www.nist.gov/${item.page}` };
  if (!P.csv) return { ...base, record: false, why: "the page links no CSV record" };
  let text;
  try {
    text = await read(P.csv, "fcd");
  } catch (e) {
    // D2's link answered 403 (Access Denied) from NIST's storage on 2026-11-13: a record the page names but does not serve.
    return { ...base, record: false, csv: `https://www.nist.gov/fcd-s3?path=/${P.csv}`, why: `the record the page links was not served: ${e.message.split(" for ")[0]}` };
  }
  const R = nistRecord(text);
  const C = itemCurve(R, out ? out.t : null);
  const mi = t3("Initial Specimen Mass"), mf = t3("Final Specimen Mass");
  const lost = mi != null && mf != null ? mi - mf : null;
  let Eb = 0; // kJ, the record's burner channel from ignition to its end (the ignition source, where it is metered)
  if (R.burner) for (let k = 0; k < R.t.length; k += 1) if (R.t[k] >= 0 && Number.isFinite(R.burner[k])) Eb += R.burner[k] * (k + 1 < R.t.length ? R.t[k + 1] - R.t[k] : 1);
  const page = { peakKW: t2("Peak Heat Release Rate"), timeToPeakMin: t2("Time to Peak"), thrMJ: t2("Total Heat Released, THR"), burnerThrMJ: t2("Natural Gas Burner Total Heat Released") ?? t2("Gas Burner Total Heat Released"), massLostKg: t2("Net Specimen Mass"), effectiveHeatOfCombustionMJkg: t2("Net Effective Heat of Combustion"), durationMin: t2("Test Duration"), hoodM: t3("Hood Size") };
  const oneMW = C.fit ? C.fit.t0 + Math.sqrt(1000 / C.fit.alpha) : null;
  return {
    ...base,
    record: true,
    csv: `https://www.nist.gov/fcd-s3?path=/${P.csv}`,
    columns: R.head,
    ignitionAtZero: ign ? ign.t === 0 : false,
    // The record's Time column against the page's events (minutes to 0.01): the page's Peak HRR event less the raw peak's time.
    peakEventLessRawPeakS: (() => {
      const pk = events.find((e) => /^peak hrr$/i.test(e.e));
      return pk ? r(pk.t - C.WR.peak.t, 1) : null;
    })(),
    events: events.map((e) => `${e.e} @ ${e.t} s`),
    recordS: [Math.min(...R.t), Math.max(...C.raw.map((p) => p.t))],
    peakKW: r(C.W.peak.q, 1),
    peakAtS: C.W.peak.t,
    rawPeakKW: r(C.WR.peak.q, 1),
    rawPeakAtS: C.WR.peak.t,
    energyMJ: r(C.E / 1000, 1),
    energyToFireOutMJ: out ? r(C.Ecut / 1000, 1) : null,
    hrrAtFireOutKW: C.atCut ? r(C.atCut.q, 1) : null,
    burnerChannelMJ: R.burner ? r(Eb / 1000, 2) : null,
    massInitialKg: mi,
    massFinalKg: mf,
    massLostKg: lost != null ? r(lost, 3) : null,
    effectiveHeatOfCombustionMJkg: lost > 0 ? r(C.E / 1000 / lost, 2) : null,
    effectiveHeatOfCombustionToFireOutMJkg: lost > 0 && out ? r(C.Ecut / 1000 / lost, 2) : null,
    fit: fitFields(C),
    secondRoutes: secondFields(C),
    beside: {
      note: "not pre-registered (as 15.6's note on the FSRI chairs): the fitted curve's time to 1 MW counted from ignition, t0 + sqrt(1000 / alpha); extrapolated where the item's 5 s peak is under 1 MW",
      oneMWFromIgnitionS: r(oneMW, 0),
      extrapolated: C.fit ? C.W.peak.q < 1000 : null,
    },
    pageSummary: page,
  };
}

async function nistPart() {
  const tests = [];
  for (const item of NIST_ITEMS) tests.push(await nistRow(item));
  const byKind = {};
  for (const kind of NIST_KINDS) {
    const rows = tests.filter((c) => c.kind === kind && c.record);
    const fitted = rows.filter((c) => c.fit.alpha != null);
    byKind[kind] = {
      n: rows.length,
      fitted: fitted.length,
      noFit: rows.filter((c) => c.fit.alpha == null).map((c) => `${c.id} (${c.fit.why})`),
      ...alphaSummaries(fitted),
      peakKW: summarise(rows, (c) => c.peakKW, 1),
      rawPeakKW: summarise(rows, (c) => c.rawPeakKW, 1),
      energyMJ: summarise(rows, (c) => c.energyMJ, 1),
      energyToFireOutMJ: summarise(rows, (c) => c.energyToFireOutMJ, 1),
      massLostKg: summarise(rows, (c) => c.massLostKg, 2),
      effectiveHeatOfCombustionMJkg: summarise(rows, (c) => c.effectiveHeatOfCombustionMJkg, 2),
      effectiveHeatOfCombustionToFireOutMJkg: summarise(rows, (c) => c.effectiveHeatOfCombustionToFireOutMJkg, 2),
      beside: { oneMWFromIgnitionS: summarise(fitted.filter((c) => !c.beside.extrapolated), (c) => c.beside.oneMWFromIgnitionS, 0), note: "items whose 5 s peak reaches 1 MW only" },
    };
  }
  const projects = [...new Set(tests.map((c) => c.project))];
  const doc = {
    generated: new Date().toISOString().slice(0, 10),
    tool: "tools/calibration/heat-release-checks.mjs --nist",
    plan: "wiki/research/fire-calibration-plan-2026-11-11.md, section 15.3 (the NIST part); no verdict word attaches (15.3)",
    data: { database: "NIST Fire Calorimetry Database (FCD)", doi: "10.18434/mds2-2314", page: "https://www.nist.gov/el/fcd", channel: `${HRR} (kW)`, read: "2026-11-13" },
    terms: NIST_TERMS,
    notice: FCD_NOTICE,
    references: Object.fromEntries(projects.map((p) => [p, NIST_REPORTS[p] ?? "no report found on the project's page"])),
    selection: {
      rule: "every single upholstered seat or sofa, and every full-scale upholstered chair mock-up, burnt alone under the NFRL's hood, with a 1 s record; chosen from the database's listing by specimen name and description before any record was opened",
      searched: "www.nist.gov/el/fcd/experiments?k=<word> for chair, sofa, couch, loveseat, recliner, armchair, upholstered, mockup, barrier, cushion, seat, furniture, foam (2026-11-13)",
      kinds: NIST_KINDS,
      leftOut: LEFT_OUT,
    },
    parameters: { startKW: START_KW, peakFraction: PEAK_FRACTION, meanStepS: STEP },
    method: {
      readings: `the suite's ${STEP} s means of the 1 s record (each the mean of the readings in (s - 2.5, s + 2.5], s = 0, 5, 10 ... s from ignition, the file's Time column), for the peak, the window and the fit; as the FSRI part`,
      window: `from the first 5 s mean above ${START_KW} kW (t >= 0) to the first at ${PEAK_FRACTION * 100} % of the 5 s peak, both included; as the FSRI part`,
      fit: "least squares of q = alpha max(0, t - t0)^2, t0 free (scanned in 0.5 s steps over [window start - 1200 s, window end), refined by golden section); as the FSRI part",
      energy: "the 1 s readings of the Heat Release Rate column summed from ignition to the record's end (kJ to MJ), and to the page's Fire Out event; no baseline taken off; as the FSRI part (which sums to its Suppression event)",
      mass: "the page's Table 3 initial and final specimen mass (gravimetric); the effective heat of combustion is the energy over their difference",
      pageSummary: "the page's Table 2 (and its hood size from Table 3), copied as published with its expanded uncertainty, for checking the reduction: the database sums from ignition to Fire Out",
      burnerChannel: "the record's Burner HRR (2023 items) or NG Burner HRR (2018 mock-ups) column summed like the energy, reported beside; the 2018 mock-ups' 18 kW propane ignition burner (80 s, about 1.4 MJ) is not metered and so stays in their Heat Release Rate, which is used as it is",
      timeCheck: "peakEventLessRawPeakS: the page's Peak HRR event (minutes to 0.01) less the record's raw peak time; it shows whether the record's Time column is the page's clock",
    },
    classes: { medium: GROWTH_CLASSES.medium, fast: GROWTH_CLASSES.fast, all: GROWTH_CLASSES, source: "topologicpy-web-frontend/src/core/fireClasses.js GROWTH_CLASSES" },
    tests,
    noRecord: tests.filter((c) => !c.record).map((c) => `${c.id} (${c.why})`),
    byKind,
    sources: sources.filter((s, q) => sources.findIndex((x) => x.path === s.path && x.repo === s.repo) === q),
  };
  mkdirSync(path.dirname(path.resolve(OUT)), { recursive: true });
  writeFileSync(path.resolve(OUT), JSON.stringify(doc, null, 1) + "\n");

  console.log("3. the design fire against NIST's upholstered items (15.3): 5 s means; alpha in kW/s^2; energy to the record's end (to Fire Out)");
  console.log("   test                 kind                          peak kW (raw)   energy MJ (to out)  page THR  lost kg  MJ/kg (page)   window s     alpha    t0 s    r2    raw 1 s  sqrt-line  1 MW s  place");
  for (const c of tests) {
    if (!c.record) {
      console.log(`   ${String(c.id).padEnd(20)} no record (${c.why})`);
      continue;
    }
    const F = c.fit;
    console.log(
      `   ${c.id.padEnd(20)} ${c.kind.padEnd(29)} ${f(c.peakKW, 0).padStart(5)} (${f(c.rawPeakKW, 0).padStart(5)})  ${f(c.energyMJ, 1).padStart(7)} (${f(c.energyToFireOutMJ, 1).padStart(7)})  ${f(c.pageSummary.thrMJ?.value, 1).padStart(7)}  ${f(c.massLostKg, 2).padStart(6)}  ${f(c.effectiveHeatOfCombustionMJkg, 1).padStart(5)} (${f(c.pageSummary.effectiveHeatOfCombustionMJkg?.value, 1).padStart(5)})  ` +
        (F.alpha != null
          ? `${`${F.windowS[0]}-${F.windowS[1]}`.padEnd(11)} ${f(F.alpha, 4)}  ${f(F.t0S, 1).padStart(6)}${F.t0AfterWindowStart ? "*" : " "} ${f(F.r2, 3)}  ${f(c.secondRoutes.raw1s.alpha, 4)}   ${f(c.secondRoutes.sqrtLine5s.alpha, 4)}    ${f(c.beside.oneMWFromIgnitionS, 0).padStart(5)}${c.beside.extrapolated ? "e" : " "}  ${F.place}`
          : `no fit: ${F.why}`),
    );
  }
  console.log("   (* t0 after the window's start: the readings before it are fitted as zero; e: 1 MW beyond the item's peak, extrapolated)");
  for (const [kind, s] of Object.entries(byKind)) {
    console.log(`   ${kind} (n ${s.n}, fitted ${s.fitted}): alpha ${rng(s.alpha, 4)}; medium ${GROWTH_CLASSES.medium} ${s.alpha?.medium ?? "-"}, fast ${GROWTH_CLASSES.fast} ${s.alpha?.fast ?? "-"}`);
    console.log(`      raw 1 s alpha ${rng(s.alphaRaw1s, 4)} (medium ${s.alphaRaw1s?.medium ?? "-"}, fast ${s.alphaRaw1s?.fast ?? "-"}); sqrt-line alpha ${rng(s.alphaSqrtLine5s, 4)} (medium ${s.alphaSqrtLine5s?.medium ?? "-"}, fast ${s.alphaSqrtLine5s?.fast ?? "-"})`);
    console.log(`      peak ${rng(s.peakKW, 0)} kW; raw peak ${rng(s.rawPeakKW, 0)} kW; energy ${rng(s.energyMJ, 1)} MJ; to Fire Out ${rng(s.energyToFireOutMJ, 1)} MJ; effective heat of combustion ${rng(s.effectiveHeatOfCombustionMJkg, 1)} MJ/kg (to Fire Out ${rng(s.effectiveHeatOfCombustionToFireOutMJkg, 1)})`);
    console.log(`      beside: 1 MW from ignition ${rng(s.beside.oneMWFromIgnitionS, 0)} s (items reaching 1 MW); medium ${f(Math.sqrt(1000 / GROWTH_CLASSES.medium), 0)} s, fast ${f(Math.sqrt(1000 / GROWTH_CLASSES.fast), 0)} s`);
    if (s.noFit.length) console.log(`      no fit: ${s.noFit.join("; ")}`);
  }
  console.log(`   no record: ${doc.noRecord.join("; ") || "none"}`);
}

if (NIST) await nistPart();
else await fsriPart();
console.log(`\nwritten: ${path.relative(ROOT, path.resolve(OUT)).split(path.sep).join("/")}`);
