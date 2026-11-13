// Tallies the abstract screening (SCREENING_PROTOCOL.md): counts by relevance,
// included records (relevance 2 or 3), and agreement between the first screen
// and the independent second screen on its sample (Cohen's kappa, unweighted on
// include/exclude and linearly weighted on 0-3). Also (added 2026-11-12) the
// adjudication of the second-screen sample, the English-only eligibility
// exclusion after screening, and the final included counts. Reads work/screen
// and work/scored.json (git-ignored), writes screening_counts.json (ids and
// codes only, no abstracts).
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
const codes = (r) => ({ id: r.id, relevance: r.relevance, quantities: r.quantities, study_type: r.study_type, links: r.links });
const included = [...first.values()].filter((r) => r.relevance >= 2).map(codes);

// Added 2026-11-12, after the second screen and before the full-text stage.
// Adjudication of the second-screen sample by the conservative title and
// abstract rule: a record is included if either screener included it. A record
// the second screen adds takes the second screen's codes; one the second screen
// would drop keeps the first screen's.
const ADJUDICATION_RULE = "conservative title and abstract rule: a record is included if either screener coded it relevance 2 or 3";
const secondById = new Map(second.map((r) => [r.id, r]));
const addedBySecond = second.filter((r) => first.has(r.id) && first.get(r.id).relevance < 2 && r.relevance >= 2).map((r) => r.id);
const keptAgainstSecond = second.filter((r) => first.has(r.id) && first.get(r.id).relevance >= 2 && r.relevance < 2).map((r) => r.id);
const adjudicated = new Map(first);
for (const id of addedBySecond) adjudicated.set(id, secondById.get(id));
const adjCounts = [0, 0, 0, 0];
for (const r of adjudicated.values()) adjCounts[r.relevance] += 1;
const adjIncluded = [...adjudicated.values()].filter((r) => r.relevance >= 2);

// Eligibility after screening (PROTOCOL.md section 2, English only). Rule, set
// 2026-11-12 after screening: language is judged by the record's own title and
// abstract; the OpenAlex language tag decides only when those are not in
// English or are absent. Title and abstract each count as English only when
// written in English throughout: no letters of a non-Latin script, and more
// English than German or French function words. A bilingual abstract (another
// language beside the English) is therefore not "in English", and the tag
// decides. Reads title, abstract and tag from work/scored.json (git-ignored);
// writes only ids and the decision, never the text.
const scored = JSON.parse(readFileSync(new URL("./work/scored.json", import.meta.url), "utf8"));
const meta = new Map(scored.map((w) => [String(w.id).split("/").pop(), w]));
const language = (id) => (meta.get(id) || {}).language || "n/r";
// Scripts other than Latin (Greek is left out: its letters occur as symbols in English text).
const NON_LATIN = /[Ѐ-ӿ֐-׿؀-ۿऀ-ॿ฀-๿぀-ヿ㐀-鿿가-힣]/;
const STOP = {
  en: new Set("the of and in to is a for with was were by on are that this from as be an it at which".split(" ")),
  de: new Set("der die das und ist mit von den zu im für werden wird eine ein des auf sich nicht".split(" ")),
  fr: new Set("le la les des et est du une un dans pour sur par au aux sont que qui".split(" ")),
};
const stopCount = (text, lang) => (String(text).toLowerCase().match(/[a-zà-ÿß]+/g) || []).filter((w) => STOP[lang].has(w)).length;
function inEnglish(text) {
  if (!text || !String(text).trim()) return null; // absent
  if (NON_LATIN.test(text)) return false;
  // Any sentence written in German or French (a bilingual abstract) fails the test.
  for (const s of String(text).split(/(?<=[.!?])\s+/)) {
    const en = stopCount(s, "en"), de = stopCount(s, "de"), fr = stopCount(s, "fr");
    if (Math.max(de, fr) >= 2 && Math.max(de, fr) > en) return false;
  }
  const en = stopCount(text, "en");
  return en >= 1 && en > stopCount(text, "de") && en > stopCount(text, "fr");
}
function languageDecision(id) {
  const w = meta.get(id) || {};
  const title = inEnglish(w.title);
  const abstract = inEnglish(w.abstract);
  const tag = language(id);
  if (title === true && abstract === true) return { id, tag, title, abstract, decidedBy: "title and abstract", english: true };
  return { id, tag, title, abstract, decidedBy: "OpenAlex tag", english: tag === "en" };
}
const decisions = new Map(adjIncluded.map((r) => [r.id, languageDecision(r.id)]));
const notEnglish = adjIncluded.filter((r) => !decisions.get(r.id).english);
const finalIncluded = adjIncluded.filter((r) => decisions.get(r.id).english);
// Every record where the text and the tag disagree, or the tag decided against English.
const reviewed = adjIncluded.map((r) => decisions.get(r.id)).filter((d) => d.tag !== "en" || d.decidedBy === "OpenAlex tag");
const hasDoi = (id) => { const d = (meta.get(id) || {}).doi; return Boolean(d && d !== "None"); };
const tallyRel = (rows) => ({ 2: rows.filter((r) => r.relevance === 2).length, 3: rows.filter((r) => r.relevance === 3).length });

const out = {
  generated: new Date().toISOString(),
  screened: first.size,
  byRelevance: { 0: counts[0], 1: counts[1], 2: counts[2], 3: counts[3] },
  byBatch,
  included: included.length,
  secondScreen: { sample: second.length, matched: pairs.length, includeKappa: binary, weightedKappa: weighted },
  adjudication: {
    rule: ADJUDICATION_RULE,
    addedBySecondScreen: addedBySecond.map((id) => ({ id, first: first.get(id).relevance, second: secondById.get(id).relevance, adjudicated: adjudicated.get(id).relevance })),
    keptIncludedAgainstSecondScreen: keptAgainstSecond.map((id) => ({ id, first: first.get(id).relevance, second: secondById.get(id).relevance })),
    byRelevance: { 0: adjCounts[0], 1: adjCounts[1], 2: adjCounts[2], 3: adjCounts[3] },
    excludedAtScreening: adjCounts[0] + adjCounts[1],
    included: adjIncluded.length,
  },
  languageExclusion: {
    criterion: "PROTOCOL.md section 2 (English only), applied after screening",
    rule: "language is judged by the record's own title and abstract (each in English throughout); the OpenAlex language tag decides only when those are not in English or are absent",
    tagNotEnglish: adjIncluded.filter((r) => language(r.id) !== "en").map((r) => decisions.get(r.id)),
    decidedByTagWithEnglishTag: reviewed.filter((d) => d.tag === "en").length,
    excluded: notEnglish.map((r) => ({ id: r.id, language: language(r.id), relevance: r.relevance, decidedBy: decisions.get(r.id).decidedBy })),
    count: notEnglish.length,
  },
  final: {
    included: finalIncluded.length,
    byRelevance: tallyRel(finalIncluded),
    withDoi: finalIncluded.filter((r) => hasDoi(r.id)).length,
    withoutDoi: finalIncluded.filter((r) => !hasDoi(r.id)).length,
    coreWithDoi: finalIncluded.filter((r) => r.relevance === 3 && hasDoi(r.id)).length,
    coreWithoutDoi: finalIncluded.filter((r) => r.relevance === 3 && !hasDoi(r.id)).length,
  },
  includedRecords: included,
  finalIncludedRecords: finalIncluded.map(codes),
};
writeFileSync(new URL("./screening_counts.json", import.meta.url), JSON.stringify(out, null, 1));
console.log(JSON.stringify({ screened: out.screened, byRelevance: out.byRelevance, included: out.included, includeKappa: binary.kappa, includeObserved: binary.observed, weightedKappa: weighted.kappa, matched: pairs.length, binaryMatrix: binary.matrix }));
console.log(JSON.stringify({ adjudication: { ...out.adjudication, rule: undefined }, languageExclusion: out.languageExclusion, final: out.final }));
