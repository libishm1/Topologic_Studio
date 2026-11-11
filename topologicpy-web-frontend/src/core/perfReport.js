/**
 * Several fire scenarios in one report, for the performance-based check
 * (performanceCheck.js; the user's request of 2026-11-07 and
 * wiki/research/rset-components-2026-11-07.md 7.2).
 *
 * Each fire planned and checked adds an entry: its base case under the
 * criterion chosen and, when the plan has them, two sensitivity cases under
 * the sensitivity criterion (ASET / RSET at least 1.2, Singapore SFEG 2025
 * Table 10.4):
 * - the exit nearest the fire blocked (SFEG 2025 sensitivity test 5; PD
 *   7974-6 discounts the largest exit), from the plan's second deadline map;
 * - the first detector failing (SFEG 2025 test 6): detection when the smoke
 *   reaches the last of the ceiling detectors round the fire.
 * The report passes only if every case of every fire passes (PreMovement.lean,
 * report_pass_iff, on the worst margins).
 */

/** The worst of a list of statuses: fail, then cannot tell, then pass. */
export function worstStatus(list) {
  return list.includes("fail") ? "fail" : list.includes("cannot tell") ? "cannot tell" : list.length ? "pass" : "cannot tell";
}

/**
 * Detection as the performance check takes it, so the extinguisher attempt
 * can take the same: the plan's smoke detection (a Flamescapes run's
 * ceiling detectors, or head height for a run kept from before them) when
 * `fromRun` and the plan has one, else `typed`. `from` says which: "ceiling"
 * or "run" from the run, else "set", "not in the run" (the smoke never
 * reached a detector), "no run" (a plan of another fire model) or "no plan".
 * @returns {{seconds: number, from: string}}
 */
export function planDetection(plan, { fromRun, typed }) {
  if (!fromRun) return { seconds: typed, from: "set" };
  if (!plan) return { seconds: typed, from: "no plan" };
  if (Number.isFinite(plan.detection)) return { seconds: plan.detection, from: plan.detectionAt === "ceiling" ? "ceiling" : "run" };
  return { seconds: typed, from: /^flamescapes/.test(plan.model ?? "") ? "not in the run" : "no run" };
}

/**
 * The cases one plan gives.
 * @param {Function} check   performanceCheck
 * @param {object} graph
 * @param {object} plan      the forecast worker's result: deadlines, deadlinesBlocked, exitsLeft, blockedExit, detectionSlowest
 * @param {object} opts      performanceCheck's options for the base case (exits, detection, ...), and
 *                           criterion, sensitivity (rsetPresets.js RSET_CRITERIA entries), fromCeiling
 */
export function planCases(check, graph, plan, { criterion, sensitivity, fromCeiling = false, ...common }) {
  const cases = [
    {
      id: "base",
      label: "Base case",
      criterion: criterion.label,
      detection: common.detection,
      result: check(graph, plan.deadlines, { ...common, factor: criterion.factor, margin: criterion.margin }),
    },
  ];
  if (plan.deadlinesBlocked && plan.exitsLeft?.length) {
    cases.push({
      id: "exit-blocked",
      label: "The exit nearest the fire blocked (SFEG 2025 test 5)",
      criterion: sensitivity.label,
      detection: common.detection,
      result: check(graph, plan.deadlinesBlocked, { ...common, exits: plan.exitsLeft, factor: sensitivity.factor, margin: sensitivity.margin }),
    });
  }
  if (fromCeiling) {
    const slowest = plan.detectionSlowest;
    const label = "The first detector fails: the ceiling detector the smoke reaches last (SFEG 2025 test 6)";
    if (Number.isFinite(slowest)) {
      cases.push({
        id: "detector",
        label,
        criterion: sensitivity.label,
        detection: slowest,
        result: check(graph, plan.deadlines, { ...common, detection: slowest, factor: sensitivity.factor, margin: sensitivity.margin }),
      });
    } else {
      cases.push({ id: "detector", label, criterion: sensitivity.label, detection: null, result: null, note: "not every ceiling detector responded within the run: plan a longer one" });
    }
  }
  return cases;
}

/** A case's status: its result's, or cannot tell when it has none. */
export const caseStatus = (c) => c.result?.status ?? "cannot tell";

/** The report's verdict over every case of every fire. */
export function reportStatus(entries) {
  return worstStatus(entries.flatMap((e) => e.cases.map(caseStatus)));
}

/** One case in figures: the worst over its storeys. */
export function caseFigures(c) {
  const storeys = c.result?.storeys ?? [];
  const min = (key) => storeys.reduce((m, s) => Math.min(m, s[key]), Infinity);
  const max = (key) => storeys.reduce((m, s) => Math.max(m, s[key] ?? 0), 0);
  return {
    worstMargin: storeys.length ? min("worstMargin") : null,
    worstRatio: storeys.length ? min("worstRatio") : null,
    share: storeys.length ? min("share") : null,
    worstWait: c.result?.queues ? max("worstWait") : null,
    measured: c.result?.measured?.status ?? null,
  };
}

const escape = (text) => String(text).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[ch]);
const seconds = (x) => (x === null || x === undefined ? "" : x === -Infinity ? "no safe time" : Number.isFinite(x) ? `${Math.round(x)} s` : "");
const ratio = (x) => (Number.isFinite(x) ? x.toFixed(2) : "");
const percent = (x) => (Number.isFinite(x) ? `${(100 * x).toFixed(1)} %` : "");
const VERDICT = { pass: "Pass", fail: "Fail", "cannot tell": "Cannot tell" };

/**
 * The report as a standalone HTML page.
 * @param {Array} entries  {label, model, detectionFrom, cases}
 * @param {object} meta    title, model (file name), date (text), inputs: [[name, value, source]], limits: [text]
 */
export function reportHtml(entries, meta) {
  const verdict = reportStatus(entries);
  const rows = entries
    .flatMap((e) =>
      e.cases.map((c, k) => {
        const f = caseFigures(c);
        return `<tr><td>${k === 0 ? escape(e.label) : ""}</td><td>${escape(c.label)}<br><small>${escape(c.criterion)}</small></td><td>${seconds(c.detection)}</td><td class="${caseStatus(c).replace(" ", "-")}">${VERDICT[caseStatus(c)]}${c.note ? `<br><small>${escape(c.note)}</small>` : ""}</td><td>${seconds(f.worstMargin)}</td><td>${ratio(f.worstRatio)}</td><td>${percent(f.share)}</td><td>${f.worstWait === null ? "" : seconds(f.worstWait)}</td><td>${f.measured ? VERDICT[f.measured] : ""}</td></tr>`;
      }),
    )
    .join("\n");
  const details = entries
    .map((e) => {
      const parts = e.cases
        .filter((c) => c.result)
        .map((c) => {
          const items = c.result.storeys
            .map((s) => {
              const m = s.measured ? ` Measured spread: ${VERDICT[s.measured.status]}, ${percent(s.measured.share)} start in time.` : "";
              const q = s.worstWait !== null ? ` Queues: ${Math.round(s.occupants)} occupants; the last to start waits up to ${seconds(s.worstWait)} at doors.` : "";
              return `<li><strong>${escape(s.name)}</strong>: ${VERDICT[s.status]}. ${escape(s.text)}.${escape(m)}${escape(q)}</li>`;
            })
            .join("");
          return `<h4>${escape(c.label)}</h4><p>Occupants must be able to set off up to ${seconds(c.result.start)} after ignition (detection at ${seconds(c.detection)}, the alarm at ${seconds(c.result.cue)}). ${escape(c.criterion)}.</p><ul>${items}</ul>`;
        })
        .join("");
      return `<section><h3>${escape(e.label)}</h3><p>${escape(e.model)}; detection ${escape(e.detectionFrom)}.</p>${parts}</section>`;
    })
    .join("\n");
  const inputs = meta.inputs.map(([name, value, source]) => `<tr><th>${escape(name)}</th><td>${escape(value)}</td><td><small>${escape(source || "")}</small></td></tr>`).join("\n");
  const limits = meta.limits.map((t) => `<li>${escape(t)}</li>`).join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escape(meta.title)}</title>
<style>
body{font:15px/1.5 system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1b1f24;background:#fff}
table{border-collapse:collapse;width:100%;margin:1rem 0}th,td{border:1px solid #ccd3db;padding:.35rem .5rem;text-align:left;vertical-align:top}
th{background:#f2f4f7}small{color:#55606e}.fail{color:#a1260d;font-weight:600}.cannot-tell{color:#7a5a00;font-weight:600}.pass{color:#1c6b33;font-weight:600}
h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:2rem}h3{font-size:1.05rem}h4{font-size:.95rem;margin-bottom:.2rem}
</style></head><body>
<h1>${escape(meta.title)}</h1>
<p>${escape(meta.model)}. ${escape(meta.date)}. Verdict over ${entries.length} fire${entries.length === 1 ? "" : "s"}: <span class="${verdict.replace(" ", "-")}">${VERDICT[verdict]}</span>.</p>
<p>Performance-based egress check (ASET against RSET) from Topologic Studio's fire forecast: evidence for an alternative method under SBC 801-CR-2018 104.9, beside the prescriptive code check, not a replacement for it.</p>
<h2>Scenarios</h2>
<table><thead><tr><th>Fire</th><th>Case and criterion</th><th>Detection</th><th>Verdict</th><th>Worst margin</th><th>Least ASET / RSET</th><th>Start in time (least storey)</th><th>Longest queue</th><th>Measured spread</th></tr></thead>
<tbody>
${rows}
</tbody></table>
<h2>Inputs and sources</h2>
<table><tbody>
${inputs}
</tbody></table>
<h2>Each fire, storey by storey</h2>
${details}
<h2>What the check does not show</h2>
<ul>${limits}</ul>
</body></html>
`;
}
