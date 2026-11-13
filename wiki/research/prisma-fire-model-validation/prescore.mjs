/**
 * The FMV review's pre-score (PROTOCOL.md section 4, step 2): a weighted
 * regex score over concept groups in each record's title and abstract, with
 * an exclusion gate, and the threshold that leaves about `--target` records
 * for screening (ties at the threshold are kept). Writes the pool with a
 * `score` field and the counts; reads nothing but the harvest.
 *
 *   node prescore.mjs work/harvest/openalex_raw.json work/scored.json [--target 600]
 */
import { readFileSync, writeFileSync } from "node:fs";

const [input, output] = process.argv.slice(2, 4);
const at = process.argv.indexOf("--target");
const TARGET = at > 0 ? Number(process.argv[at + 1]) : 600;

const GROUPS = [
  { name: "model", weight: 5, re: /fire dynamics simulator|\bfds\b|\bcfast\b|zone model|fire models?\b|fire simulation|firefoam|computational fluid dynamics|\bcfd\b|large eddy simulation|\bles\b/i },
  { name: "validation", weight: 5, re: /validat|calibrat|comparison with (the )?(experiment|measurement)|compared with (the )?(experiment|measurement)|benchmark|model uncertainty|predictive capabilit/i },
  { name: "enclosure", weight: 3, re: /compartment|\broom\b|enclosure|corridor|building|atrium|multi-room/i },
  { name: "experiment", weight: 2, re: /experiment|full-scale|large-scale|measured|measurements|test series/i },
];
// Each quantity group found adds 3.
const QUANTITIES = [
  /hot gas layer|upper layer|smoke layer|layer temperature|layer height|layer interface|layer depth/i,
  /ceiling jet/i,
  /fire plume|buoyant plume|plume temperature|centreline|centerline/i,
  /oxygen|carbon dioxide|carbon monoxide|gas concentration|species/i,
  /door flow|doorway|vent flow|neutral plane|opening flow|steckler/i,
  /visibility|optical density|extinction coefficient|soot|smoke concentration|obscuration/i,
  /sprinkler|detector|activation time/i,
  /heat flux|wall temperature|surface temperature|heat loss/i,
];
// Out of scope unless the record is also about an enclosure (PROTOCOL.md section 2).
const EXCLUDE = /wildland|forest fire|wildfire|bushfire|internal combustion|combustor|gas turbine|engine cylinder|coal mine/i;

const pool = JSON.parse(readFileSync(input, "utf8"));
const all = Array.isArray(pool) ? pool : pool.works || Object.values(pool);
// PROTOCOL.md section 4, step 1: duplicates by normalised DOI, then by
// normalised title and year (OpenAlex ids are already unique); the first kept
// takes the others' queries.
const norm = (s) => (s || "").toLowerCase().replace(/<[^>]+>/g, "").replace(/[^a-z0-9]+/g, " ").trim();
const seen = new Map();
const records = [];
let duplicates = 0;
for (const r of all) {
  const doi = (r.doi || "").toLowerCase().replace(/^https?:\/\/(dx\.)?doi\.org\//, "");
  const keys = [doi && `doi:${doi}`, norm(r.title) && `t:${norm(r.title)}|${r.publication_year}`].filter(Boolean);
  const first = keys.map((k) => seen.get(k)).find(Boolean);
  if (first) {
    duplicates += 1;
    for (const q of r.queries || []) if (!first.queries.includes(q)) first.queries.push(q);
    continue;
  }
  for (const k of keys) seen.set(k, r);
  records.push(r);
}
for (const r of records) {
  const text = `${r.title || ""} ${r.abstract || ""}`;
  let score = 0;
  for (const g of GROUPS) if (g.re.test(text)) score += g.weight;
  for (const q of QUANTITIES) if (q.test(text)) score += 3;
  if (EXCLUDE.test(text) && !GROUPS[2].re.test(text)) score = 0;
  r.score = score;
}
const sorted = records.map((r) => r.score).sort((a, b) => b - a);
const threshold = sorted[Math.min(TARGET, sorted.length) - 1] ?? 0;
const kept = records.filter((r) => r.score >= threshold).length;
writeFileSync(output, JSON.stringify(records, null, 1));
const counts = { harvested: all.length, duplicates, records: records.length, excludedByGate: records.filter((r) => r.score === 0).length, target: TARGET, threshold, kept, belowThreshold: records.length - kept };
writeFileSync(output.replace(/\.json$/, "-counts.json"), JSON.stringify(counts, null, 1));
console.log(JSON.stringify(counts));
