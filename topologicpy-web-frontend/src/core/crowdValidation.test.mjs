// The people-leaving model's verification tests (tools/validation/crowd-validation.mjs:
// the RiMEA guideline's and NIST TN 1822's cases) run with the unit tests, so a change to
// crowd.js that breaks one shows at once. The known failures are findings about the
// model, recorded in wiki/research/crowd-vv-report-2026-11-24.md; another would be new.
// The cases run twice: as the app runs by default, and with the corridors' and stairs'
// flow capacity (`passageFlow`), which RiMEA 12b, NIST Verif.5.1 and the SFPE flows need.
import test from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const harness = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "..", "tools", "validation", "crowd-validation.mjs");

const run = (...flags) => {
  const out = execFileSync(process.execPath, [harness, "--quick", "--no-write", ...flags], { encoding: "utf8" });
  const line = out.split(/\r?\n/).find((l) => l.startsWith("SUMMARY "));
  assert.ok(line, "the harness ends with its summary");
  return JSON.parse(line.slice(8));
};

test("the RiMEA and NIST verification cases by default: the same passes, and only the known failures", () => {
  // Without a capacity for corridors and stairs: a corridor passes more than a door (RiMEA 12b), a stair
  // drawn as one file of nodes passes the same whatever its width (SFPE flows), the corridor does not
  // flow steadily before the stair (NIST Verif.5.1); and the single-file diagram is too fast (RiMEA 16).
  const summary = run();
  assert.deepEqual(summary.failed.sort(), ["NIST-Verif.5.1", "RiMEA-12b", "RiMEA-16", "SFPE-flows"]);
  assert.ok(summary.pass >= 22, `${summary.pass} pass`);
});

test("the RiMEA and NIST verification cases with passageFlow: only the single-file diagram fails", () => {
  const summary = run("--passage-flow");
  assert.deepEqual(summary.failed.sort(), ["RiMEA-16"]);
  assert.ok(summary.pass >= 25, `${summary.pass} pass`);
});

test("the calibrated density reading: the single-file diagram passes, and what it costs is on record", () => {
  // densityReading "calibrated" (off by default) was chosen on RiMEA 16's single-file data alone
  // (wiki/research/crowd-calibration-2026-11-24.md). RiMEA 16 then passes; but on the data held back
  // it failed its rules: without a capacity the queue before a corridor or a stair crawls, and with
  // passageFlow the 1 m stair passes a quarter less than the SFPE flow, so the SFPE flows fail.
  const plain = run("--reading", "calibrated");
  assert.deepEqual(plain.failed.sort(), ["NIST-Verif.5.1", "SFPE-flows"]);
  const capped = run("--reading", "calibrated", "--passage-flow");
  assert.deepEqual(capped.failed.sort(), ["SFPE-flows"]);
});
