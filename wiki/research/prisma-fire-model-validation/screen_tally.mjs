// Tallies the abstract screening (SCREENING_PROTOCOL.md): counts by relevance,
// included records (relevance 2 or 3), and agreement between the first screen
// and the independent second screen on its sample (Cohen's kappa, unweighted on
// include/exclude and linearly weighted on 0-3). Reads work/screen (git-ignored),
// writes screening_counts.json (ids and codes only, no abstracts).
// node screen_tally.mjs
import { readFileSync, writeFileSync, readdirSync } from "node:fs";
const dir = new URL("./work/screen/", import.meta.url);
const coded = readdirSync(dir).filter((f) => /^coded_\d+\.json$/.test(f)).sort();
const first = new Map();
const byBatch = {};
for (const f of coded) {
  const rows = JSON.parse(readFileSync(new URL(f, dir), "utf8"));
  byBatch[f] = [0, 0, 0, 0];
  for (const r of rows) { first.set(r.id, r); byBatch[f][r.relevance] += 1; }
}
const counts = [0, 0, 0, 0];
for (const r of first.values()) counts[r.relevance] += 1;
const second = JSON.parse(readFileSync(new URL("second_coded_00.json", dir), "utf8"));
const pairs = second.filter((r) => first.has(r.id)).map((r) => [first.get(r.id).relevance, r.relevance]);
function kappa(pairs, k, weight) {
  const n = pairs.length;
  const m = Array.from({ length: k }, () => new Array(k).fill(0));
  for (const [a, b] of pairs) m[a][b] += 1;
  const ra = m.map((row) => row.reduce((s, x) => s + x, 0));
  const cb = m[0].map((_, j) => m.reduce((s, row) => s + row[j], 0));
  let po = 0, pe = 0;
  for (let i = 0; i < k; i += 1) for (let j = 0; j < k; j += 1) {
    const w = weight(i, j);
    po += (w * m[i][j]) / n;
    pe += (w * ra[i] * cb[j]) / (n * n);
  }
  return { kappa: +((po - pe) / (1 - pe)).toFixed(3), observed: +po.toFixed(3), matrix: m };
}
const include = (r) => (r >= 2 ? 1 : 0);
const binary = kappa(pairs.map(([a, b]) => [include(a), include(b)]), 2, (i, j) => (i === j ? 1 : 0));
const weighted = kappa(pairs, 4, (i, j) => 1 - Math.abs(i - j) / 3);
const included = [...first.values()].filter((r) => r.relevance >= 2).map((r) => ({ id: r.id, relevance: r.relevance, quantities: r.quantities, study_type: r.study_type, links: r.links }));
const out = {
  generated: new Date().toISOString(),
  screened: first.size,
  byRelevance: { 0: counts[0], 1: counts[1], 2: counts[2], 3: counts[3] },
  byBatch,
  included: included.length,
  secondScreen: { sample: second.length, matched: pairs.length, includeKappa: binary, weightedKappa: weighted },
  includedRecords: included,
};
writeFileSync(new URL("./screening_counts.json", import.meta.url), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ screened: out.screened, byRelevance: out.byRelevance, included: out.included, includeKappa: binary.kappa, includeObserved: binary.observed, weightedKappa: weighted.kappa, matched: pairs.length, binaryMatrix: binary.matrix }));
