// Calibrating how the people-leaving model reads the density that slows a step
// (core/crowd.js `densityReading`), by the rules of
// wiki/research/crowd-calibration-2026-11-24.md, which were fixed before this ran.
//
//   node tools/validation/crowd-calibrate.mjs [--out FILE.json]
//
// Calibration data: RiMEA Test 16's single-file percentiles only, through the
// harness's own RiMEA 16 case (crowd-validation.mjs --only fundamental1D), with
// the model's default options. Every one of the 36 combinations is run. The
// choice: the most points inside the experiments' 10 to 90 % band; among those
// the smallest J, the mean of ((v - p50) / h)^2 with h half the band; J within
// 5 % of the best is a tie, which goes to the reading nearest the current one.
// Nothing here looks at the held-out cases (RiMEA 4, RiMEA 12d, the rest of the harness).
// Writes benchmark/results/<--out> (default 2026-11-24-crowd-density-calibration.json).
import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "..", "..");
const harness = path.join(here, "crowd-validation.mjs");

// The candidates, the current values first (section 2 of the rules).
const CENTRES = ["ahead", "own"];
const RADII = [1.0, 1.5, 2.0];
const AREAS = ["all", "standing"];
const FLOORS = [0.15, 0.1, 0.05];
const CURRENT = { centre: "ahead", radius: 1.0, area: "all", floor: 0.15 };

const rows = [];
for (const centre of CENTRES) {
  for (const radius of RADII) {
    for (const area of AREAS) {
      for (const floor of FLOORS) {
        const reading = { centre, radius, area, floor };
        const out = execFileSync(process.execPath, [harness, "--no-write", "--only", "fundamental1D", "--print", "RiMEA-16", "--reading", JSON.stringify(reading)], { encoding: "utf8" });
        const result = JSON.parse(out.split(/\r?\n/).find((l) => l.startsWith("RESULT ")).slice(7));
        const points = result.measured;
        const inBand = points.filter((p) => p.inside_band).length;
        const J = points.reduce((s, p) => s + ((p.mean_speed_mps - p.p50) / ((p.p90 - p.p10) / 2)) ** 2, 0) / points.length;
        const changed = Object.keys(CURRENT).filter((k) => reading[k] !== CURRENT[k]).length;
        rows.push({ ...reading, points_in_band: inBand, of: points.length, J: Math.round(J * 1e4) / 1e4, changed, points });
        console.log(`${centre.padEnd(5)} R ${radius.toFixed(1)} ${area.padEnd(8)} floor ${floor.toFixed(2)}: ${inBand} of ${points.length} in band, J ${J.toFixed(4)}`);
      }
    }
  }
}

// The choice, exactly as the rules have it.
const most = Math.max(...rows.map((r) => r.points_in_band));
const top = rows.filter((r) => r.points_in_band === most);
const bestJ = Math.min(...top.map((r) => r.J));
const tied = top.filter((r) => r.J <= bestJ * 1.05);
tied.sort((a, b) => a.changed - b.changed || a.radius - b.radius || b.floor - a.floor || a.J - b.J);
const chosen = tied[0];
const current = rows.find((r) => r.changed === 0);
console.log(`\ncurrent: ${current.points_in_band} of ${current.of} in band, J ${current.J}`);
console.log(`chosen: ${JSON.stringify({ centre: chosen.centre, radius: chosen.radius, area: chosen.area, floor: chosen.floor })}: ${chosen.points_in_band} of ${chosen.of} in band, J ${chosen.J} (${tied.length} tied within 5 % of the best J ${bestJ})`);

const outArg = process.argv.indexOf("--out");
const outFile = path.join(root, "benchmark", "results", outArg > 0 ? process.argv[outArg + 1] : "2026-11-24-crowd-density-calibration.json");
writeFileSync(
  outFile,
  JSON.stringify(
    {
      date: "2026-11-24",
      rules: "wiki/research/crowd-calibration-2026-11-24.md",
      data: "RiMEA Test 16 single-file percentiles (tools/validation/rimea-test16-percentiles.json), 14 points",
      choice: { centre: chosen.centre, radius: chosen.radius, area: chosen.area, floor: chosen.floor, points_in_band: chosen.points_in_band, J: chosen.J },
      current: { points_in_band: current.points_in_band, J: current.J },
      tied: tied.map(({ points, ...r }) => r),
      combinations: rows,
    },
    null,
    1,
  ),
);
console.log(path.relative(root, outFile));
