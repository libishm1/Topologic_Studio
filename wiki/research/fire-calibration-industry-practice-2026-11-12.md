# Fire calibration: what industry practice requires, and where the protocol stands (2026-11-12)

This note is the industry-practice section of the fire-model calibration review. It sets the project's claim protocol beside the standards and guides that govern the verification and validation of fire models. It gives what each requires, how each defines its terms, a gap table, and what "calibrated" can honestly mean for a coarse real-time solver.

It follows [fire-calibration-claim-protocol-2026-11-12.md](fire-calibration-claim-protocol-2026-11-12.md), [fire-calibration-plan-2026-11-11.md](fire-calibration-plan-2026-11-11.md) (sections 8 and 12) and [fire-calibration-literature-2026-11-12.md](fire-calibration-literature-2026-11-12.md). Writing it changed no code, ran no solver and moved no verdict. Section 7 lists every source and how it was checked. A source that could not be opened is named as such, and nothing is attributed to it.

## 1. What the project does

The solver is Flamescapes, run in the browser on cells of 0.20 to 0.45 m. Its claim rests on these rules:

- Statistics as the FDS Validation Guide computes them (section 4.4): the bias factor δ and the model's relative standard deviation σ̃M from the log ratios of model to measurement, with the experimental σ̃E of its Table 4.1. Each pair is a device's largest rise above ambient in a window, or Steckler's steady value.
- Bands set by this project (2026-11-08 plan, section 8): within the calibrated band at 0.8 ≤ δ ≤ 1.25 and σ̃M ≤ 0.35; indicative at 0.67 to 1.5 and σ̃M ≤ 0.5. The guide gives the statistics, not the bands.
- A quantity is "calibrated" on a cell size when at least three independent held-out datasets are in band and none is outside the indicative band. A dataset with a stopped run counts as outside. Where plan 12.6 and the protocol's section 4 differ, the stricter governs.
- Pre-registration, with dated addenda for every later change.
- One fitted parameter, the vorticity confinement, fitted only on McCaffrey's plume and Alpert's ceiling jet. It now stands at 3.
- A fault may be fixed only when a check that uses no validation data shows it. Every held-out comparison is then rerun, and both results are reported.

The held-out series are Steckler (NBS), NBS multi-room, NIST/NRC and WTC, with ATF's corridors added in the later reruns. Every test has a prescribed heat release. The suite runs on the CPU path on 0.20, 0.27 and 0.34 m cells.

The stated verdict (plan 12.8, carried by `topologicpy-web-frontend/src/core/calibrationStatus.js`):
- calibrated: the smoke layer's depth on 0.27 m cells (δ 1.05, σ̃M 0.15);
- indicative: the layer's temperature on 0.27 and 0.34 m, and the ceiling jet;
- not calibrated: O2, CO2 and the door's peak outflow;
- the neutral plane is in band, but in one dataset only;
- visibility is not compared.

The app's run card names these statuses. It quotes the bias and scatter of the calibrated quantity in words, and applies no uncertainty to a run's own values. It says that a run's untenable places come from visibility and gas temperature at head height, that the run reports no layer depth, and that it predicts its own heat release.

## 2. What the standards and guides require

### 2.1 ASTM E1355

The current edition is E1355-23, listed as active by ASTM. The FDS Validation Guide and NUREG-1824 Supplement 1 follow E1355-12. The CFAST guide cites E1355-04. The standard's full text was not read for this note. What follows comes from its publisher page and from the guides that apply it.

- Evaluation has four areas (section 1.2): define the model and the scenarios; verify the theoretical basis and assumptions; verify mathematical and numerical robustness; quantify the accuracy and uncertainty of results for similar scenarios.
- Validation for one application or scenario does not imply validation for another (5.1).
- Approving officials are to confirm that a model is used within its applicable limits and with acceptable accuracy (5.3.4).
- The guide is not an acceptance test (5.4). It states that it cannot be used to provide quantitative measures (1.4).
- Precision is the repeatability of a deterministic model. Accuracy is how well it replicates a real fire (5.5).
- Robustness is checked by analytical tests, code checking and numerical tests (as NUREG-1824 Supplement 1, section 5.2, lists them).
- Sensitivity may be studied by direct or finite-difference methods, by factorial or Latin hypercube designs, or by global and response-surface methods (as the CFAST guide, section 2.1, lists them).
- Validation calculations are blind, specified or open (as the FDS and CFAST guides give them; section 3).
- The NRC/EPRI team found that E1355 gives no criteria for judging predictive capability from V&V results, so it wrote its own (NUREG-1824, 2007, section 2.6.2).

### 2.2 ISO 16730-1 and ISO/TS 13447

ISO 16730-1:2015 sets a framework for verifying and validating any calculation method used in fire safety engineering. It is not model-specific. Its abstract lists:
- a process to show that equations and methods are implemented correctly (verification) and that the method represents the real world accurately (validation);
- documentation that shows the method's scientific and technical basis is adequate;
- requirements for the data against which predictions are checked;
- guidance for developers, users and those who assess results.

ISO lists it as confirmed in 2021 and under systematic review in 2026. Its body, with its definitions and procedural clauses, was not read for this note.

ISO/TS 13447:2013 gives guidance on zone models. Its abstract says it is not a basis for justifying any particular model or for regulation, and that users must understand a model's theory and judge the accuracy and validity of its results. Flamescapes is not a zone model, so it bears here only through that principle.

### 2.3 NUREG-1824, its Supplement 1, and NUREG-1934

NUREG-1824 (EPRI 1011999, 2007) verified and validated five models under ASTM E1355 against 26 experiments.
- Applicability is judged by normalized parameters. Its Table 2-4 lists them; its Table 2-5 gives their ranges in the test series.
- A scenario inside those ranges is covered by the results. Outside them, the study supports no validation determination.
- Two criteria graded each model and quantity: are the physics appropriate, and do the differences exceed the experimental and input uncertainty. The grades were GREEN, YELLOW+, YELLOW-, YELLOW and RED. The report states where engineering judgement entered.
- GREEN meant validated only within the parameter space of the test series.
- Models were modified and corrected during the study, after interpreting results, and the report says so (section 3.1.2).
- Three named peer reviewers read the draft, and it was noticed in the Federal Register for comment.

Supplement 1 (EPRI 3002002182, November 2016) extended this to 340 experiments from 16 series.
- It replaced the colour grades with δ and σ̃M, after McGrattan and Toman (2011).
- Its pairs are weighted so that each of 10 equal intervals of the measured range counts equally (section 6.1, note 7).
- σ̃E comes from its own analysis of each measurement (section 3.3).
- Its statistics apply only to the model versions evaluated. Users of newer versions must cite updated statistics (sections 1.3 and 7.1).
- Its Table 3-3 gives the normalized parameters and their ranges. Its Table 7-1 sets the 2007 ranges beside the new ones.
- δ and σ̃M are to be applied as uncertainty bounds and combined with input uncertainty, with a sensitivity analysis where inputs are uncertain (section 7.1, item 7).

NUREG-1934 (EPRI 1023259, November 2012) is the user's guide to that evidence.
- Its process has six steps: goals, scenarios, model selection, calculation, sensitivity and uncertainty, documentation.
- It names three kinds of uncertainty, after NUREG-1855: parameter, model and completeness.
- Its chapter 4 turns validation into a decision quantity. The true value is taken as normal with mean M/δ and standard deviation σ̃M M/δ, with every quantity as a rise above ambient. The probability that it exceeds a critical value follows from the complementary error function (Eq. 4-1). Section 4.3 works examples.
- Its Table 2-5 gives the normalized parameters and their ranges, derived from NUREG-1824's Table 2-4.
- Section 4.1 sorts a scenario as inside, partly inside or outside that space. Partly inside needs further justification. Outside needs independent justification.
- Section 2.3.7 allows a sensitivity analysis that moves inputs, in the unfavourable direction, back into the validated range.
- For FDS grids (section 2.3.5): D*/δx of 5 to 10 usually serves where gross smoke movement matters; at least three cells across any door, window or fire; differences between grids judged against the model's expected accuracy.
- Documentation should let someone else reproduce the results (section 2.6).
- The ACRS asked for two things in this guide: the ranges of normalized parameters in plant use, and model uncertainty as probability distributions (section 1.1).
- The guide went to a peer panel, through two Federal Register comment periods, and before the ACRS.

### 2.4 FDS Technical Reference Guide, Volume 3: Validation (NIST SP 1018-3, release 6.11.1)

- It takes its definitions from ASTM E1355 (Preface). Verification checks that the governing equations are solved correctly. Validation judges whether they model the physics, usually by comparison with experiment. Differences beyond numerical error and measurement uncertainty are attributed to the model's assumptions. Volume 2 holds verification.
- Validation has three parts: compare, quantify the differences given the uncertainties, and decide whether the model fits the use. The guide does the first two. The third is the end user's. To call FDS "validated" means the user has quantified its uncertainty for the application and judged it fit (chapter 1).
- All its calculations are "open" in E1355's sense: the results were known when they were run (section 1.1).
- Section 3.115 and Table 3.55 define a "parameter space" as the range of applicability. A scenario must have comparable physical parameters before the guide can be cited for it. Table 3.56 gives the numerical ones (D*, D*/δx, H/D*). Both physical and numerical parameters must be comparable.
- The metric is usually the peak of the time-averaged histories, as a rise above ambient. Peacock et al. (1999) are cited for other metrics (section 4.4).
- σ̃M may not fall below σ̃E, because a model cannot be shown more accurate than the data (section 4.4).
- Pairs are weighted by a regressogram of 10 equal bins, so that dense regions do not skew δ (section 4.4).
- The distribution of Eq. 4.23 assumes that the modeller has not adjusted inputs to offset a known bias in the output (section 4.4). Section 4.5 works a probability of exceedance.
- σ̃E combines the measurement's uncertainty with that of the inputs propagated through the model (Table 4.1).
- Averaged and far-field quantities are predicted better than point and near-field ones (section 4.6).
- The guide is regenerated at every minor release.
- Chapter 2 separates design applications, where the heat release is prescribed and the model predicts transport, from forensic reconstructions, where it is predicted. Most validation covers the first kind.

### 2.5 CFAST Verification and Validation Guide (NIST TN 1889v3, revision 7.7.7)

It uses the same E1355 definitions, the same blind, specified and open classes, and the same parameter-space table. Its section 2.1 reviews sensitivity studies of CFAST. In those, heat loss to the walls, the heat release and the ceiling height dominated the outputs.

### 2.6 SFPE Guidelines for Substantiating a Fire Model for a Given Application (2011)

The guide itself was not read. Its steps are taken from SFPE's catalogue page and from two papers by Hofmeister and Hill, who describe it:
1. Define the problem of interest: scope, influences, outputs.
2. Select a candidate model.
3. Verification and validation. If no existing study covers the application, review other data or commission tests. Do not apply a model blindly.
4. User effects: assumptions, inputs, the spatial domain and grid, the design fire, its location and suppression. The user should know how sensitive the results are to these.
5. Documentation, at the level the project, client or authority requires.

### 2.7 Other practice

- McGrattan and Toman (2011) split a prediction's uncertainty into input uncertainty and model uncertainty. They estimate the second from comparisons with measurements alone, as a bias factor and a relative standard deviation. They argue its simplicity suits approving authorities.
- Peacock et al. (1999) treat time histories as vectors. Functional analysis then defines lengths, angles and distances between curves, so comparisons become quantitative.
- Audouin et al. (2011) compared several field and zone models with a confined, ventilated pool fire in the OECD PRISME programme, over six quantities. They found that more than one metric is needed.
- Rein et al. (2009) gathered a priori predictions of Dalmarnock Fire Test One from seven teams. The spread between teams was much larger than the experimental error. Fire growth was poorly predicted.
- Jahn, Rein and Torero (2008) found FDS's fire growth in that test sensitive to the fire's location and area, its radiative fraction and the materials' thermal and ignition properties.

Not verified: BS 7974:2019 and PD 7974-1:2019. Only PD 7974-1's catalogue page was opened, and it says nothing on computer models. Nothing in this note rests on either.

## 3. How the sources define the terms

### 3.1 Definitions

| Term | What the sources say | This project |
| --- | --- | --- |
| Model evaluation | E1355, as the FDS and CFAST guides quote it: quantifying the accuracy of chosen results from a model for a specific use. It has two parts, verification and validation. | Not used as a term. |
| Verification | E1355 via FDS and CFAST guides: the equations are solved correctly; it does not show they are the right equations. ISO 16730-1: equations and methods are implemented correctly. NUREG-1824 Supplement 1: the model represents its developer's conceptual description; it was "built" correctly. | Checks that use no measured data: heat closure, sealed rooms, the oxygen-limited burner, the sweep-order symmetry, CPU and GPU step agreement. |
| Validation | E1355 via the guides: the equations are an appropriate model of the physics, usually shown by comparison with experiments. ISO 16730-1: an accurate representation of the real world. NUREG-1824 Supplement 1: the right model was built. FDS guide: compare, quantify, then let the end user decide fitness. | The held-out comparisons. |
| Calibration | Not defined in any document read. The FDS guide uses it for fitting inputs to measurements. It contrasts "true predictions" with "calibrations" (VTT's work). It calls one hose-stream case a calibration more than a validation. It notes that fitted heat release ramps make the fitted temperatures non-independent (smoke alarm tests). | A status label for held-out results in band (section 1). The confinement fit is a calibration in the guides' sense. |
| Model uncertainty | NUREG-1934 (after NUREG-1855): from idealised physics and numerical solution, estimated by V&V. FDS guide and McGrattan and Toman: δ and σ̃M from comparisons. | δ and σ̃M per quantity, dataset and cell. |
| Parameter and completeness uncertainty | NUREG-1934: parameter uncertainty comes from the inputs and is studied by sensitivity analysis. Completeness uncertainty comes from phenomena the model omits, and is covered indirectly by validation. | Not studied for inputs. The omissions are listed (literature note, section 4.8). |
| Experimental uncertainty | FDS guide Table 4.1 and NUREG-1824 Supplement 1, section 3.3: the measurement's own, plus the inputs' propagated through the model. | Table 4.1's values. The neutral plane's 0.05 is the project's assumption. |
| Precision, accuracy | E1355 5.5: repeatability of a deterministic model; agreement with a real fire. | Bit-for-bit repeats on the CPU path (plan 12.7); accuracy as δ and σ̃M. |
| Range of applicability, parameter space | FDS guide 3.115: the parameters of the experiments outline where the validation applies; physical and numerical parameters must both be comparable. NUREG-1824 and NUREG-1934: normalized parameters; inside, partly inside or outside. | Heat release, room volumes, openings, walls, times and D*/δx (plan 12.8). No non-dimensional groups but D*/δx. |
| Blind, specified, open | E1355 via the FDS guide. Blind: the user builds inputs from a basic description. Specified: full inputs given. Open: inputs and results known. | Open calculations, pre-registered before results were read. |
| User effects | SFPE step 4: uncertainty a user adds through assumptions, inputs and the spatial domain. | Not studied. |

### 3.2 The non-dimensional groups, and where the held-out series sit

The groups, as the FDS guide (section 3.115) and NUREG-1934 (section 2.3.6) define them:
- Fire Froude number, Q* = Q / (ρ∞ c_p T∞ √(gD) D²), with D the effective diameter √(4A/π) of the fire's base. Typical accidental fires have Q* near 1.
- Flame length ratio, L_f/H, with L_f = D (3.7 Q*^(2/5) - 1.02) after Heskestad. NUREG-1934 adds the fire's base height to L_f.
- Global equivalence ratio, φ: the fuel's mass flow over the oxygen's, over the stoichiometric ratio. It is written as Q / (13,100 kJ/kg × ṁ_O2), with ṁ_O2 = 0.23 × ½ A0 √H0 for natural ventilation and 0.23 ρ V̇ for mechanical. φ < 1 is well ventilated.
- Compartment aspect ratios, W/H and L/H.
- Ceiling jet distance ratio, r_cj/H, and radial distance ratio, r_rad/D.
- Numerical: D* = (Q / (ρ∞ c_p T∞ √g))^(2/5), the plume resolution index D*/δx, and H/D*.

The held-out series, as the FDS Validation Guide's Table 3.55 gives them for each series as a whole. The project's tests are the guide's or a subset of them, so its own ranges may be narrower. D*/δx at 0.27 m is from the literature note, section 2.3.

| Series | Q (kW) | Q* | L_f/H | φ | W/H | L/H | r_cj/H | D*/δx at 0.27 m (guide's runs) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Steckler | 31.6 to 158 | 0.8 to 3.8 | 0.3 to 0.7 | 0.0 to 0.6 | 1.3 | 1.3 | n/a | 0.9 to 1.7 (5 to 9) |
| NBS multi-room | 110 | 1.5 | 0.5 | 0.0 | 1.0 | 5.1 | n/a | 1.5 (4) |
| NIST/NRC | 350 to 2,200 | 0.3 to 2.0 | 0.3 to 1.0 | 0.0 to 0.3 | 1.9 | 5.7 | 0.3 to 2.1 | 2.5 to 5.0 (5 to 11) |
| WTC | 1,970 to 3,240 | 0.6 to 0.9 | 0.8 to 1.1 | 0.3 to 0.5 | 0.9 | 1.8 | 0.0 to 0.8 | 4.7 to 5.6 (9 to 11) |
| ATF corridors (later reruns only) | 50 to 500 | 0.3 to 3.3 | 0.3 to 0.9 | 0.0 to 0.1 | 0.8 | 7.1 | 0.8 to 6.0 | 1.1 to 2.7 (3 to 7) |

Over the four series of the stated verdict: Q* 0.3 to 3.8, L_f/H 0.3 to 1.1, φ 0.0 to 0.6, W/H 0.9 to 1.9, L/H 1.3 to 5.7, and D*/δx 0.9 to 5.6 on 0.27 m cells. For comparison:
- NUREG-1824 (2007), as NUREG-1934's Table 2-5 gives it: Q* 0.4 to 2.4, flame length ratio 0.2 to 1.0, φ 0.04 to 0.6, aspect ratio 0.6 to 5.7;
- NUREG-1824 Supplement 1, Table 7-1: Q* 0.2 to 9.1, flame length ratio 0.0 to 1.6, φ 0.0 to 0.6, aspect ratio 0.6 to 8.3.

All the held-out fires are well ventilated. The project's physical space sits inside the NRC's 2016 space. Its numerical resolution sits three to six times below the guide's own runs, and mostly below NUREG-1934's 5 to 10.

## 4. Gap table

"Met" means the project does what the source asks. "Partly" means it does some of it. Each step is the smallest that would close the gap. None of them changes the solver.

| # | Requirement (source) | What the project has | Met | Minimal step |
| --- | --- | --- | --- | --- |
| G1 | Define the use, and the quantities decisions rest on, before evaluating (E1355 1.2.1; SFPE step 1; NUREG-1934 step 1) | Plan 2026-11-11, section 1, maps app outputs to quantities. Tenability rests on head-height gas temperature and visibility. Neither has a held-out verdict. The calibrated quantity, layer depth, is not reported for a run. | Partly | Pre-register, before computing it, a point comparison of gas temperature at the trees' thermocouples between 1.5 and 2.0 m on the four series (σ̃E 0.07). The decision quantity then has its own δ and σ̃M. |
| G2 | Theoretical basis and assumptions documented (E1355 1.2.2; ISO 16730-1 documentation; NUREG-1824 Supplement 1, section 4) | Equations are in the source and in plan notes. The omissions are listed in the literature note, section 4.8. | Partly | One technical-reference note: governing equations, sub-models, the fitted parameter, default inputs and the known omissions, each with its code location. |
| G3 | Mathematical and numerical robustness (E1355 1.2.3: analytical tests, code checking, numerical tests; FDS keeps a separate Verification Guide) | Heat closure, the sealed-room test, the oxygen-limited burner test, the sweep-order symmetry check, the run-health guard, CPU and GPU single-step agreement, unit tests. | Met in substance; not collected | A verification record: each check, its analytic or conservation reference, its tolerance and its last result, run by one command before every held-out rerun. |
| G4 | Statistics as the guides compute them (FDS guide 4.4; NUREG-1824 Supplement 1, 6.1) | The same δ and σ̃M equations, σ̃M floored at σ̃E, rises above ambient, peak metric (`lib/stats.mjs`). No 10-bin regressogram weighting. | Partly | Add the weighting. Report weighted and unweighted statistics. Pre-register which one the verdict uses. |
| G5 | Experimental uncertainty for each quantity (FDS Table 4.1; NUREG-1824 Supplement 1, 3.3; ISO 16730-1 data requirements) | Table 4.1's values. The neutral plane's σ̃E 0.05 is the project's assumption, and is labelled so. | Mostly | Report the neutral-plane verdict also at σ̃E 0.08 and 0.10, to show whether it hinges on the assumption. |
| G6 | Acceptance criteria (E1355 sets none and is not an acceptance test; NUREG-1824 2007 wrote its own grades; the FDS guide leaves fitness to the user) | Project bands, pre-registered and labelled as the project's. | Met in kind | None for the rule. In Methods, give each band's basis: about twice FDS's transport scatter, and FDS's weakest outputs (plan 2026-11-08, section 8). |
| G7 | Number and independence of validation series (no standard sets a minimum; NUREG-1824: 26 experiments, then 340 from 16 series; the FDS guide has 18 series for layer temperature under natural ventilation) | Four series from NBS and NIST, one lineage. NIST/NRC and WTC share a laboratory and lead. ATF, a third laboratory, is in the later reruns. | Partly | Add one non-NIST laboratory already judged a good fit: the UL FSRI 2018 compartments (literature note, section 5), pre-registered. |
| G8 | Validation class and verifiable timing (E1355's blind, specified and open classes; no document read asks for pre-registration) | Open calculations, pre-registered with dated addenda. This exceeds the standards. Plan 12 says sections 12.1 to 12.6 were not committed before the first baseline run. The protocol records that the burner fault was found because a held-out run stopped. | Partly | The suite refuses to start unless the pre-registration file is committed, and writes its hash into the results file (it already records the git state). For the next new dataset, someone other than the modeller holds the measured file and runs the statistics, a specified calculation in E1355's sense. |
| G9 | Range of applicability by non-dimensional groups (FDS guide 3.115, Table 3.55; NUREG-1824 2007, Tables 2-4 and 2-5; Supplement 1, Tables 3-3 and 7-1; NUREG-1934, Table 2-5 and sections 2.3.7 and 4.1) | Heat release, room volumes, openings, walls, times and D*/δx (plan 12.8). The app checks peak kW and duration only (`COVERED`). No Q*, L_f/H, φ, W/H, L/H or r_cj/H. | Partly | Adopt section 3.2's ranges as the covered space. Compute the same groups for each app run from its scene and design fire. Label the run inside, partly inside or outside, as NUREG-1934 section 4.1 does. |
| G10 | Numerical resolution and grid sensitivity (FDS guide 3.115; NUREG-1934 2.3.5: D*/δx 5 to 10, three cells across openings and fire, grid differences judged against accuracy) | Verdicts per cell size; D*/δx reported; the sealed-door rule. On 0.27 m cells, D*/δx is 0.9 to 5.6, Steckler's doors are 0.9 to 3.7 cells wide, and ATF's burner is under two. | Partly; the guidance itself is not met | Say in each verdict that it lies below NUREG-1934's resolution guidance. Flag app runs with any opening or burner under three cells, not only under one. Tabulate each quantity's change in δ between 0.27 and 0.34 m beside its σ̃M. |
| G11 | Sensitivity analysis (E1355 via the CFAST guide 2.1; NUREG-1934 step 5; FDS guide chapter 1; Jahn et al. 2008) | The confinement sweep, which is a fit. Diagnostics `--walls stone` and `--fuel app`. The old transport. Three cell sizes. No input sensitivity. | Partly | One-at-a-time perturbations on one single-room and one corridor test on 0.27 m: heat release ±15 % (NBS's stated uncertainty), radiant fraction ±0.1, wall properties ×0.5 and ×2, confinement 2 and 4. Report each change in δ beside σ̃M. Nothing is set from it. |
| G12 | User effects (SFPE step 4; E1355's blind class; Rein et al. 2009) | The harness builds scenes from FDS inputs with the app's voxeliser and burner. The IFC route and a user's choices (fire location and class, door states, walls) are not exercised. | No | Rebuild two held-out tests, one Steckler door and one NIST/NRC test, through the app's IFC route as a user would. Report their δ beside the harness's. If two people can do it independently, report their spread. |
| G13 | Model uncertainty usable in decisions (NUREG-1934 chapter 4, Eq. 4-1; FDS guide 4.5; McGrattan and Toman 2011; the ACRS request) | Status labels only. The 2026-11-08 plan (sections 8 and 9) proposes corrected values and the band's unfavourable side. `calibrationStatus.js` holds δ and σ̃M only as text, for one quantity. | No | Store δ and σ̃M per quantity and cell in `calibrationStatus.js`. For a quantity with a verdict, show the probability that the true rise exceeds a threshold rise, by Eq. 4-1. Apply it to tenability once G1 gives head-height temperature a verdict. |
| G14 | Statistics tied to the version evaluated (NUREG-1824 Supplement 1, 1.3 and 7.1; FDS guide chapter 1) | SHA-256 digests of the solver and harness, `RUN_CACHE_VERSION`, a full rerun after each fix, both results reported. The suite runs the CPU path. The app also runs a WebGPU path. | Mostly | State that the verdict holds for the CPU path. Or rerun a held-out subset on the GPU path once the burner's rays are ported. |
| G15 | Independent review (NUREG-1824 2007: named peer reviewers and a Federal Register notice; NUREG-1934: a peer panel, two comment periods, the ACRS) | An internal multi-agent review (2026-11-12) that reproduced two transport faults. No external reviewer. | No | Send the protocol, plan section 12 and the results files to one external fire modeller. Record the comments and the responses as a dated addendum. |
| G16 | Documentation that lets someone else reproduce the results (NUREG-1934 2.6; ISO 16730-1; SFPE step 5) | Commands, pinned commits, data digests, results files, licence notices. | Met | Generate one per-quantity summary, a scatter plot and a table, from the results file, ordered by quantity as the FDS guide is. |
| G17 | More than one comparison metric (Audouin et al. 2011; Peacock et al. 1999; the FDS guide uses peaks and cites Peacock for others) | One metric per quantity: the peak rise, or Steckler's steady value. | Partly | Report a time-history distance after Peacock et al. for the layer and the ceiling jet, as a secondary statistic outside the verdict rule. |

Two observations come out of the table. First, the project already exceeds the documents read on two points: it pre-registers, and it fixes faults only on evidence from checks without measured data. Second, its largest gaps are not in the statistics. They lie where the evidence meets the app: the decision quantities (G1, G13), the space a run must fall in (G9, G10), and how users and inputs move the answer (G11, G12).

## 5. What "calibrated" can honestly mean

In the vocabulary of the documents read, the held-out comparisons are validation. "Calibration" means fitting a parameter or an input to measurements. In that sense the project has calibrated one thing, the confinement strength, against McCaffrey and Alpert. No document read defines "calibrated" as a status, and none sets acceptance bands. NUREG-1824 (2007) wrote its own grades and limited them to the parameter space of its tests. The FDS guide calls a model "validated" only after the end user has quantified its uncertainty for the application and judged it fit.

For a coarse real-time model, the label's honest content is narrow. For a named quantity, on a named cell size, for fires with a prescribed heat release, inside a stated range of Q*, L_f/H, φ, aspect ratios and D*/δx, the bias and scatter against a number of held-out series met a band the project set in advance. Two limits follow from the sources. The app's own runs predict their heat release, and the FDS guide and NUREG-1934 both separate predicted growth from prescribed transport. The resolution also sits below NUREG-1934's guidance, so any verdict belongs to the solver on that cell size, not to the method.

The wording the standards support is: "Compared with N measured test series for [quantity] on [cell] m cells: bias δ, relative standard deviation σ̃M (FDS Validation Guide method), for fires of prescribed heat release within [ranges]. Outside these ranges, no validation determination." If the project keeps "calibrated" as its label, the label should carry that definition wherever it appears. The confinement would then be described as calibrated against correlations, and the held-out results as validation within the project's band.

## 6. Rules kept

- Nothing here changes a verdict, a band or a pre-registered rule. The steps in section 4 are proposals.
- Each step that adds a comparison (G1, G5, G7, G8, G17) is to be pre-registered with its date before its results are read.
- No step sets a solver parameter from held-out data.

## 7. Sources

Standards and guides:

- ASTM International. *ASTM E1355-23, Standard Guide for Evaluating the Predictive Capability of Deterministic Fire Models.* West Conshohocken, PA, 2023. Publisher page `store.astm.org/e1355-23.html`, read 2026-11-12: status active; sections 1.1 to 1.4 and 5.1 to 5.5 as shown there. The full text was not read. The FDS guide and NUREG-1824 Supplement 1 cite E1355-12; the CFAST guide cites E1355-04.
- ISO. *ISO 16730-1:2015, Fire safety engineering. Procedures and requirements for verification and validation of calculation methods. Part 1: General.* Edition 1, 2015, ISO/TC 92/SC 4. ISO catalogue page `committee.iso.org/standard/62111.html`, read 2026-11-12 (www.iso.org refused the fetch): abstract, status, confirmed 2021, systematic review 2026. The body was not read.
- ISO. *ISO/TS 13447:2013, Fire safety engineering. Guidance for use of fire zone models.* Edition 1, 2013, ISO/TC 92/SC 4. ISO catalogue page `committee.iso.org/standard/53799.html`, read 2026-11-12: abstract, confirmed 2024. The body was not read.
- U.S. Nuclear Regulatory Commission and Electric Power Research Institute. *Verification and Validation of Selected Fire Models for Nuclear Power Plant Applications, Volume 1: Main Report.* NUREG-1824, EPRI 1011999, May 2007. NRC catalogue page (`nrc.gov/reading-rm/doc-collections/nuregs/staff/sr1824/v1/`) read through an Internet Archive snapshot of 2025-02-15, because nrc.gov refuses automated requests. The report (ML071650546) was read through the same archive: sections 2.6.1 to 2.6.2, 3.1.2 and 3.2, Tables 2-4 and 2-5, and the acknowledgements. No DOI.
- U.S. Nuclear Regulatory Commission and Electric Power Research Institute. *Verification and Validation of Selected Fire Models for Nuclear Power Plant Applications, Supplement 1.* NUREG-1824 Supplement 1, EPRI 3002002182, final report, November 2016. NRC catalogue page read through an Internet Archive snapshot of 2026-04-15. The report was read from a copy cached on 2026-11-08 whose title page matches: the abstract, sections 1.2 to 1.4, 3.3, 5.2, 6.1 and 7, and Tables 3-2, 3-3 and 7-1. No DOI.
- U.S. Nuclear Regulatory Commission and Electric Power Research Institute. *Nuclear Power Plant Fire Modeling Analysis Guidelines (NPP FIRE MAG).* NUREG-1934, EPRI 1023259, final report, November 2012. NRC catalogue page read through an Internet Archive snapshot of 2024-04-25. The report was read from a copy cached on 2026-11-08: the acknowledgements, section 1.1, sections 2.3.5 to 2.3.7, 2.5 and 2.6, chapter 4 (4.1 and 4.2), and Tables 2-5, 4-1 and 4-2. No DOI.
- McGrattan, K., Hostikka, S., Floyd, J., McDermott, R., Vanella, M., Mueller, E., Paul, C. *Fire Dynamics Simulator Technical Reference Guide, Volume 3: Validation.* NIST Special Publication 1018-3, sixth edition, July 10, 2026, revision FDS-6.11.1-0-gff928db. The release PDF was read: the Preface, chapter 1, the opening of chapter 2, section 3.115 with Tables 3.55 and 3.56, and sections 4.1, 4.2 and 4.4 to 4.6. Its printed DOI, 10.6028/NIST.SP.1018, is registered on Crossref to the 2006 version 4 guide, and 10.6028/NIST.SP.1018-3 is not registered (both checked on Crossref 2026-11-12). It is cited by number and revision.
- Peacock, R. D., Forney, G. P., Reneke, P. A. *CFAST, Consolidated Fire And Smoke Transport (Version 7), Volume 3: Verification and Validation Guide.* NIST Technical Note 1889v3, revision CFAST-7.7.7-0-gb43e74e0f, May 27, 2026. A copy cached on 2026-11-08 was read: the abstract, chapter 1, section 2.1 and the parameter-space section. Release 7.7.7 is listed on `github.com/firemodels/cfast/releases` (read 2026-11-12). Its printed DOI, 10.6028/NIST.TN.1889v3, is registered on Crossref (checked 2026-11-12) to the 2015 *Volume 3: Software Development and Model Evaluation Guide*, a different title, so it is cited by number and revision.
- Society of Fire Protection Engineers. *Guidelines for Substantiating a Fire Model for a Given Application.* SFPE Engineering Guide G.06, 2011. SFPE catalogue page `sfpe.connectedcommunity.org/publications/guides/firemodel`, read 2026-11-12: listed as "(2010)" in the title, publish date 2011, 71 pages. The number G.06 and the steps are from the two Hofmeister and Hill papers below. The guide itself was not read. NUREG-1934 cites it as 2010 and NUREG-1824 Supplement 1 as 2011.
- Hofmeister, C. E., Hill, S. M. *SFPE Guidelines for Substantiating a Fire Model for a Given Application.* Fire Protection Engineering Extra, Issue 54. SFPE, undated on the page. Read 2026-11-12 on `sfpe.connectedcommunity.org`.
- Hofmeister, C. E., Hill, S. M. (2011). *Case Study Example using SFPE Guidelines for Substantiating a Fire Model for a Given Application.* Proceedings, Fire and Evacuation Modeling Technical Conference 2011, Baltimore, MD, 15 to 16 August 2011. Read from `files.thunderheadeng.com/femtc/2011_d1-02-hofmeister-paper.pdf` (first page checked 2026-11-12). No DOI.

Papers, each DOI checked on Crossref on 2026-11-12, title and year matching:

- McGrattan, K., Toman, B. (2011). Quantifying the predictive uncertainty of complex numerical models. *Metrologia* 48(3), 173-180. https://doi.org/10.1088/0026-1394/48/3/011. Abstract read on the IOP page. The full text was not read; the method is as the FDS guide and NUREG-1824 Supplement 1 give it.
- Peacock, R. D., Reneke, P. A., Davis, W. D., Jones, W. W. (1999). Quantifying fire model evaluation using functional analysis. *Fire Safety Journal* 33(3), 167-184. https://doi.org/10.1016/S0379-7112(99)00029-6. Abstract read on NIST's publication page. The full text was not read.
- Audouin, L., Chandra, L., Consalvi, J.-L., Gay, L., Gorza, E., Hohm, V., Hostikka, S., Ito, T., Klein-Hessling, W., Lallemand, C., Magnusson, T., Noterman, N., Park, J. S., Peco, J., Rigollet, L., Suard, S., Van-Hees, P. (2011). Quantifying differences between computational results and measurements in the case of a large-scale well-confined fire scenario. *Nuclear Engineering and Design* 241(1), 18-31. https://doi.org/10.1016/j.nucengdes.2010.10.027. Abstract read on VTT's research portal. The full text was not read.
- Rein, G., Torero, J. L., Jahn, W., Stern-Gottfried, J., Ryder, N. L., Desanghere, S., Lázaro, M., Mowrer, F., Coles, A., Joyeux, D., Alvear, D., Capote, J. A., Jowsey, A., Abecassis-Empis, C., Reszka, P. (2009). Round-robin study of a priori modelling predictions of the Dalmarnock Fire Test One. *Fire Safety Journal* 44(4), 590-602. https://doi.org/10.1016/j.firesaf.2008.12.008. Abstract read on the Edinburgh Research Archive. The full text was not read.
- Jahn, W., Rein, G., Torero, J. L. (2008). The effect of model parameters on the simulation of fire dynamics. *Fire Safety Science* 9, 1341-1352. https://doi.org/10.3801/IAFSS.FSS.9-1341. The abstract was read.

Opened, but not used for any claim:

- BSI. *PD 7974-1:2019, Application of fire safety engineering principles to the design of buildings. Initiation and development of fire within the enclosure of origin (Sub-system 1).* Catalogue page on `knowledge.bsigroup.com`, read 2026-11-12: current, published 31 March 2019. It says nothing on computer models, and the document was not read. BS 7974:2019 was not opened.

The repository's own records: the claim protocol, plan sections 8 and 12, the literature note, `tools/calibration/lib/stats.mjs`, and `topologicpy-web-frontend/src/core/calibrationStatus.js`.
