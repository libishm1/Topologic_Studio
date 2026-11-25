# Rules table

Last updated: 2026-11-25. One row for each ruling the project works under. A rule is protected only when all three cells are filled: the code that enforces it, the test that proves the code does, and a file that shows it held. **An empty cell is written "none" and means the rule rests on someone remembering it.**

Paths are from the repository root; `src/` is `topologicpy-web-frontend/src/`. A cell marked "not looked up" was not checked when this table was written and counts as empty until it is.

## Results and claims

| # | Ruling | Code that enforces it | Test that proves it | Evidence |
| --- | --- | --- | --- | --- |
| R1 | "Calibrated" only when a quantity passes the band (0.8 to 1.25, scatter at most 0.35) on every reading; the strictest word is the one reported | `tools/calibration/validation-suite.mjs` with `tools/calibration/lib/stats.mjs` | not looked up | the result files' final words; the comparison script's recomputation of 1956 stored values (plan 16.58) |
| R2 | A rule for a comparison is written into the plan before the run it judges | none (commit order only) | none | plan sections 16.54 and 16.55 before 16.56 and 16.57; tool commits before result commits |
| R3 | A stored rerun must reproduce the saved result before anything new is read from it | `tools/calibration/tenability-species.mjs pairs` (branch `flamescapes/head-height-species`, not on main) | its own tests on that branch (37) | plan 16.57, the pairs check |
| R4 | The people model's failing test cases are exactly the known list | `tools/validation/crowd-validation.mjs` | `src/core/crowdValidation.test.mjs` | `benchmark/results/2026-11-25-crowd-validation-passage-flow.json` (PR 26, not yet merged) |
| R5 | A claim is made only if it has a row in `claims-traceability.md` | none | none | none |
| R6 | *Proposed, not a ruling of this project (the owner doubts it applies here):* a default value carries its source | none | none | none |
| R7 | The tenability formulas are the standard's (ISO 13571), not our own | `src/core/tenability.js` | `src/core/tenability.test.mjs` | none stored |
| R8 | The dose limit the user sets is the one every dose reading uses | `src/core/tenability.js` and its callers (PR 17) | not looked up | PR 17 |

## The app

| # | Ruling | Code that enforces it | Test that proves it | Evidence |
| --- | --- | --- | --- | --- |
| A1 | Heavy features are off by default behind a switch, and cost nothing when off | defaults in `src/state/useStudio.js` | none that lists the defaults; `src/core/performanceCheck.test.mjs` covers speed | none |
| A2 | An option is on by default only after every example building runs with it | none | a unit test of the stair deadlock (PR 24) | `wiki/issues/browser-session-2026-11-24.md` |
| A3 | Fires are drawn visibly but the physics is never inflated to show them | none | none | none |
| A4 | A browser claim needs console, errors, failed requests and screenshots recorded | none | none | `wiki/issues/browser-session-*.md` |
| A5 | The unit tests pass before a deploy | not looked up (the deploy workflow) | n/a | not looked up |

## The repository

| # | Ruling | Code that enforces it | Test that proves it | Evidence |
| --- | --- | --- | --- | --- |
| G1 | No em or en dashes in committed prose | none in the repository (only the plan's insert scripts, kept outside it) | none | none |
| G2 | No local paths in committed files | none in the repository (same scripts) | none | none |
| G3 | No tool-attribution or co-author lines in commits, merges or PR bodies | none | none | the commit log |
| G4 | Nobody but the owner commits to main; others open a PR | the host's permission settings | n/a | the merge log |
| G5 | Licensed standards, corpus models, RadCal and unpublished paper results stay out of the repository | none | none | none |
| G6 | Secrets (tokens, keys) are never written to a file, log or commit | none | none | none |

## What the empty cells say

- **R6 is a proposal, not a rule here.** Three values were named this week as having no usable source. Traced on 2026-11-25 (plan 16.59): heptane's CO yield of 0.008 is NIST SP 1013-1's and the value in FDS's own validation inputs; the 880 nm is the MIREX meter of Börger et al. 2024, recorded in this project's literature note; only the uncertainty 0.07 for a crossing time was borrowed, and the guide's nearest value is 0.12. So two of the three were sourced and misreported. The narrower point stands: a value should name where it came from at the place it is used. The owner decides whether that becomes a rule.
- **R5, A3, G1, G2, G5 and G6 have no code at all.** G1 and G2 are the cheapest to close: one script in the repository, run by the tests, that scans committed prose.
- **A1 has code but no test that pins the defaults,** so a default can change without any test failing.
- Seven of the 18 rules (R6 set aside) have some enforcing code; two (R4, R7) have both code and a test on main; R4 will have all three once PR 26 is merged.

## Keeping it

A ruling is added here the day it is made. When code or a test is added for a rule, its cell is filled in the same commit.
