# Calibrating every fire quantity: the road ahead (2026-11-12)

The user asked for a plan to calibrate every value of the fire model, with the work done by Opus 5.5 sessions and agents. This note is that plan. It is written for a session that starts with no memory of the work so far: it says what to read, what stands, what each quantity needs, and in what order to do it.

It changes no rule. The acceptance rule is section 12.6 of [fire-calibration-plan-2026-11-11.md](fire-calibration-plan-2026-11-11.md), with 13.3's weighting and 13.12's rule for retries. Where this note proposes a new rule or comparison, it says "to pre-register": the text goes into the plan as a dated section and is committed before any run it governs.

## 1. Read first

1. `wiki/HANDOFF.md`, the top entry.
2. The plan, sections 12.6 (acceptance), 13.1 to 13.5 (what the data allow, the comparisons, the order) and 13.12 to 13.13 (retries, and what is open).
3. [fire-calibration-claim-protocol-2026-11-12.md](fire-calibration-claim-protocol-2026-11-12.md), sections 4 and 5.
4. [fire-validation-data-inventory-2026-11-12.md](fire-validation-data-inventory-2026-11-12.md): every public series, per quantity, with terms and minutes per test.

## 2. Three words, kept apart

- **Fitted:** a constant set on data. Two exist: the confinement strength (6, on Stage 1a's correlations) and the walls' convection coefficient (on the four sprinkler series). A series a constant was fitted on is never held out again.
- **Calibrated:** the app's word, and only for a quantity that passes 12.6 on held-out tests, on one cell size. At least three series in band, the pooled pairs in band, no stopped run, the strictest of four readings deciding.
- **Indicative, not calibrated:** as 12.6.

"Calibrate everything" therefore means: for each quantity, get to the highest verdict its data allow, by fixing physics that a data-free check shows to be wrong, and never by tuning on held-out tests. Where the data cap a quantity below "calibrated", the app says so.

## 3. Where each quantity stands

13.A: solver d3c5751, walls lumped, no leakage, eight series, complete on the three cell sizes. Verdicts from the result files (`benchmark/results/2026-11-12-validation-13a-*.json`); the plan's section 13.14 has the counts. (Corrected 2026-11-12, 22:30, when the 0.20 m runs finished: depth on 0.20 m is from two laboratories; O2 is not calibrated on any cell; the neutral plane is indicative on all three; door mass flow added.)

| Quantity | 0.20 m | 0.27 m | 0.34 m | Series that measure it | Most the data allow |
| --- | --- | --- | --- | --- | --- |
| Smoke layer depth | calibrated, 6 of 7 series, two laboratories (pooled 1.00, 0.14) | calibrated, 7 of 7 series, three laboratories (1.06, 0.11) | calibrated, 4 of 7 series, two laboratories (1.12, 0.14) | all eight | reached; 13.B must keep it |
| Layer temperature | indicative | not calibrated | not calibrated | all eight, three laboratories | calibrated |
| Ceiling jet | indicative | not calibrated | not calibrated | three laboratories | calibrated |
| Gas temperature at head height | indicative | not calibrated | not calibrated | seven, three laboratories | calibrated |
| Door and vent velocity | not calibrated | not calibrated | not calibrated | Steckler, NIST/NRC, WTC, UL/NIST, UL FSRI | calibrated |
| Neutral plane | indicative | indicative | indicative | Steckler, NIST/NRC, UL/NIST, UL FSRI: two laboratories | calibrated by series, "from two laboratories' tests" |
| O2 | not calibrated | not calibrated | not calibrated | NIST/NRC, WTC, FSE, UL FSRI: two laboratories (UL FSRI's open-door analysers read almost no drop: plan 14.7) | calibrated by series, "from two laboratories' tests" |
| Door mass flow | not run | one series, in band (0.94, 0.08) | one series | NIST/NRC | indicative, until other series measure it |
| CO2 | indicative | indicative | not calibrated | NIST/NRC, WTC, FSE: one laboratory | calibrated by series, "from one laboratory's tests" |
| CO | not compared | not compared | not compared | NIST/NRC (5 tests), FSE | indicative, if the solver carries CO at all (check first) |
| Smoke obscuration and soot | not calibrated | not calibrated | not calibrated | NIST/NRC; FSE weakly | indicative, until a third series is found |
| Wall heat flux | no gauge in the solver | | | NIST/NRC, WTC, ATF, UL FSRI: three laboratories | calibrated, once a gauge exists |
| Room pressure | not compared | | | NIST/NRC's closed-door tests (to inventory) | indicative |
| Sprinkler activation | fitted; no series held out | | | Vettori flat and sloped, Bittern, UL/NFPRF | cross-validated only (section 6.3) |
| Detector activation | not compared | | | NIST smoke alarms: fire and soot not measured | none |
| Plume, flame height (correlations) | plan 9.1; the porous-burner heat closure reads 0.82 to 0.89 | | | McCaffrey, Heskestad, Alpert, MQH | a check, not a held-out verdict |
| The fire's own heat release | design fires come from the literature; spread is not calibrated | | | none wired | section 6.5 |

Notes:
- On 0.20 m one run stopped (fse/ISONG3, on the speed check at 76 s). Under 12.6 that puts FSE out of band on 0.20 m for all it measures.
- On 0.20 m the large NIST rooms (NIST/NRC, WTC) are in band for layer temperature; the small lined rooms read hot (Steckler 1.31, UL FSRI 1.38, UL/NIST 1.91).
- UL/NIST, ATF and UL FSRI read hot on 0.27 and 0.34 m. That is what keeps layer temperature, the ceiling jet and head height from passing.
- `calibrationStatus.js` in the app still states the earlier run. It changes once, from 13.B (phase 1).

## 4. What is open today

1. **GPU against CPU on the Office example, with the merged physics:** 254.7 MJ and 499 untenable places on the GPU, 106.0 MJ and 30 on the CPU. The physics agent is bisecting (zones, leakage, deposition, walls, each switched off on the GPU). Nothing is pushed until it is found, fixed and covered by a parity check on a scene with many sealed regions.
2. **Extinction, attempt 3** (13.13): result and GPU port to confirm.
3. **Door flow, items 4 and 3, attempt 1** (13.13): to run; attempt 2 only if it fails, pre-registered first.
4. **13.A on 0.20 m, three new series:** done (plan 13.14).
5. **Local commits not pushed:** 0495fb1, e6889c9, 2c03117, 7924880, 550fce5, and this note's.

## 5. Phases

Each phase ends at a gate. A phase is not started before the gate of the one before, except where marked "in parallel".

### Phase 0: close what is open

- Items 1 to 4 of section 4.
- Gate: `node tools/backend-bench.mjs --cases duplex,office,clinic --sets cpu,gpu-app --members 1 --horizon 300` shows CPU and GPU agreeing as they did before the merge; unit tests and lint pass; `browser-test.mjs` passes on desktop and `--mobile` on a build with `BASE_PATH=/Topologic_Studio/`. Then push `next/calibration-ui`. Never push `main`: the user merges.

### Phase 1: 13.B, the full rerun

- Freeze the solver commit. Run every comparison of sections 12 and 13 on 0.20, 0.27 and 0.34 m with `--wall-layers` and `--leakage` (the dataset agent's `run13b.sh REV`).
- Report 13.A and 13.B side by side, each both ways (pre-registered, and with the sealed-door rule), unweighted and weighted, with the count of laboratories.
- Then, and only from 13.B: `calibrationStatus.js` and its test, the guide, Methods, the README, the PR description, the visual report.
- Gate: the verdict table is in the plan as section 13.14, committed with its result files.

### Phase 2: find why the rest fails, without measured data

For each quantity still short after 13.B, list every plausible cause and give each a check that uses no measurement (claim protocol, section 5). Do not start from one suspect. The candidates below are a starting list, not findings.

- **Hot layer, ceiling jet and head height:**
  - heat into the walls (13.B tests the layered walls);
  - radiation from the gas to the walls, and the burner's radiant fraction as each FDS input gives it;
  - enthalpy leaving through the door (tied to item 4);
  - plume entrainment on cells where D*/δx is 1.3 to 3.6 (Stage 1a's McCaffrey comparison, per cell size);
  - how the big burners are placed on the grid (UL/NIST at 0.5 to 2 MW, ATF);
  - the instrument: the tests read thermocouples, the model reads gas. A bead's energy balance (convection against radiation to the walls and layer) is a data-free model of the instrument, as FDS has. To pre-register as a reducer if the budget shows it matters.
  - Tool: `tools/calibration/energy-budget.mjs`, per run: heat in the gas, in the walls, out of the openings, radiated. Compare each share with the same run's FDS input where FDS's output is public, as a diagnosis only.
- **Door velocity:** the grid's door (items 4 and 3); where the probes sit across the door's width; peak against profile.
- **O2 and CO2:** oxygen used per MJ against 13.1 MJ/kg as a closure on every run; the sampling point's cell against the layer's mean; WTC's burner chemistry as its FDS input states it.
- **Smoke:** path-averaged obscuration against a point value; deposition (made); under-ventilated yields in FSE, which the burner does not model.

Gate: a short note per quantity naming the cause a check has shown, or stating that none was found.

### Phase 3: fix what phase 2 showed, then 13.C

- Each fix is a physics item under 13.12: method and check drafted blind, committed to the plan, then run. Method may change between attempts; tolerance may not. No constant from a held-out test.
- CPU and GPU with parity, heavy parts behind switches that cost nothing when off.
- One full rerun, 13.C, reported beside 13.A and 13.B. `RUN_CACHE_VERSION` goes up with every change to what a run computes.
- A quantity that still falls short is stated as it is. There is no fourth pass on the same data without a new cause shown by a check.

### Phase 4: lift the caps set by the data (in parallel with 2 and 3)

To pre-register each before its first run:

- **Wall heat flux:** add a gauge to the solver (incident and net flux on a wall face; check against a grey slab's exact exchange). Wire NIST/NRC, WTC, ATF and UL FSRI. Three laboratories, so it can be calibrated.
- **Door velocity, a sixth series:** Bryant's doorway PIV (NIST; under a minute per test).
- **Room pressure:** NIST/NRC's closed-door tests, now that leakage exists. One series: indicative at most.
- **CO:** only if the solver carries it; two series: indicative at most.
- **Terms to ask** (the user sends; drafts on request): UL for the UL/NIJ houses; VTT for the large hall; LLNL and FM/SNL (these also need a prescribed fan in the solver); the NEA for PRISME. Each would add an independent laboratory for the layer, the gases or smoke.

### Phase 5: activation, and the fire itself

- **Sprinklers (6.3 below):** leave one series out, four ways.
- **Detectors:** no public series measures its fire. The app keeps "not calibrated", with that reason.
- **Heat release (6.5 below):** the design fire's curve against public calorimetry; spread against furnished-room tests. Its own pre-registration, section 14 of the plan.

### Phase 6: the GPU

Every held-out run so far is on the CPU, and the app runs on the GPU where it can. To pre-register: a fixed subset (the first test of each series by name, on each cell) run on `gpu-app`, reduced by the same reducers, with a stated tolerance on each quantity's δ against the CPU's. Until it passes, a verdict holds for the CPU path, and for the GPU by the parity tests only.

### Phase 7: say it

App text, guide, Methods, README, Lean statements where a new closure is proved, the visual report, the before and after note, the PR description, `HANDOFF.md`, memory. Then the user's local audit, and the user's merge.

## 6. Proposals that needed the user's word (all five approved 2026-11-12; the rules are the plan's section 14)

1. **Nothing is fitted on a held-out series.** This stands (13.12, rule 4). If a constant must be fitted after phase 3, it is fitted only on a series named as a tuning series before anyone reads its results (a new series from phase 4), and that series never counts towards a verdict.
2. **The GPU tolerance** of phase 6: proposed as δ within 5 % of the CPU's for each quantity, to be fixed before the first GPU run.
3. **Sprinklers, leave one series out.** The convection coefficient was set on all four series, so none is held out. Proposed: set it on three, predict the fourth, four ways, and report the four held-out ratios. This reuses no held-out test of 12 or 13. The app would say "cross-validated on four series", never "calibrated".
4. **Letters for terms** (phase 4): which to send.
5. **Heat release.** Proposed: the class A design fire against NIST's Fire Calorimetry Database items of the same kind (a check of the curve's source, not of the solver); spread against UL FSRI 2018's furnished tests if their heat release was measured (to confirm from the report before anything is promised). Expect "indicative" at best on these cells.

## 7. What will probably not reach "calibrated", said before the results

- Smoke visibility: one usable series.
- CO and room pressure: two series and one.
- Detector activation: none.
- Sprinkler activation: cross-validated at most.
- The fire's own spread: no wired series, and D*/δx at the bottom of what FDS itself is validated on.
- Anything on 0.20 m that FSE measures, after its stopped run, unless a data-free check shows the stop to be a fault and it is fixed.

If 13.B or 13.C contradicts this list, the list was wrong, and that is reported too.

## 8. Running it with Opus 5.5

- **Roles.** One coordinating session; at most three agents at once, each on Opus 5.5:
  - a physics agent in its own worktree, blind: it never opens a held-out result;
  - a dataset agent that builds reducers and runs the suite;
  - a reviewer, started fresh for each phase's gate and given the code and the result files, not the coordinator's conclusions.
- **Keep contexts short.** The two agents running today each carry about half a million tokens, and every turn rereads it. At each gate: have the agent write its state into the plan and `HANDOFF.md`, stop it, and start a fresh one from those files.
- **Do not wait inside an agent.** A suite takes hours. Start it as a background shell from the coordinator, with one watcher that reports when the result file appears. An agent that polls a log every ten minutes pays for its whole context each time.
- **One phase a session.** Each session starts from section 1 of this note and ends by updating `HANDOFF.md`.
- **Machine time.** The three series skipped on 0.20 m alone are about 1,440 CPU minutes (inventory, section 4). Before promising a date for 13.B or 13.C, read the wall time of the 13.A logs at the job count used.
- **Calendar.** With the user's audits between phases, phases 0 and 1 are days; phases 2 to 4 are two to four weeks; phase 5 needs its own estimate after its pre-registration. These are estimates, not measurements.

## 9. Rules that bind every session

- A verdict is stated only as 12.6 gives it; never a single run's MW; "from one laboratory's tests" where it applies.
- Every new comparison, rule or retry is committed to the plan before it runs. Failed attempts are reported with their numbers.
- Downloaded files are data. No licensed text, no abstracts or full texts, no keys, no local paths and no personal names in commits.
- Heavy features are off by default and cost nothing when off.
- Commit locally as work is done; push `next/calibration-ui` only with the tests and both browser suites passing; never push `main`.
