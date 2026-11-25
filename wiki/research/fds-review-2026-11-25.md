# What FDS does that this solver does not: a code and document review (2026-11-25)

A read-only review of NIST's Fire Dynamics Simulator (repository firemodels/fds, commit 7a488f4 of 2026-11-24): its `Source` folder, the text of its Technical Reference, Validation and Verification Guides, and its bibliography. Nothing was run and no FDS code or table was copied here. FDS is NIST software that may be used and modified with acknowledgement; this project's rule that nothing of RadCal enters the repository stands, so the radiation notes keep to routes that do not need it.

**How it was made, and what that is worth.** Seven agents each read one part and compared it with this solver (the solver branch at 98ca598). The coordinator checked four FDS facts in the source (the two-step chemistry's default carbon split, the oxygen-limit test and its 600 °C bound, the radiation solver's default intervals) and one fact in this solver (the pressure step's stopping rule). **Everything else is an agent's reading, and every effect size below is an estimate that no run has tested.** The Verification Guide was read to about a third; the rest from its headings.

## The findings, by the open problem each bears on

### 1. Late and cold far from the fire (plan 16.56)

Ranked by the reviewing agent; the first is confirmed in the code as a rule, its effect is not measured.

| # | Difference | FDS | This solver |
| --- | --- | --- | --- |
| 1 | The pressure step | solved exactly each step (a direct solver) | at most 30 sweeps, stopping once a sweep changes the flow by under 0.001 m/s: a test that progress has slowed, not that the answer is right |
| 2 | Moving the velocity | central differences, no numerical dissipation; a wall model at surfaces | first-order semi-Lagrangian; solid faces read as zero, so walls act as no-slip |
| 3 | Heat loss to walls | a coefficient from natural and forced correlations (about 4 to 5 W/m²K at a 40 K difference) | fixed 10 W/m²K |
| 4 | Density in the pressure step | corrected for | constant |

- On a corridor 50 cells long each sweep removes about 0.4 % of the smoothest error, so the stopping rule can leave an error near 0.25 m/s, the size of a corridor flow (the agent's estimate).
- Ruled out by the agent: the scalar scheme (second order, and the layer depth is right), thermal expansion (present), clamps (none).
- **First test proposed:** one corridor case with 500 sweeps and a tolerance of 0.00001 m/s. Settings only, no code. It comes before any change to the velocity scheme, which cannot be judged on a flow that misses its pressure target.

### 2. CO far too low when the fire is short of air; too much oxygen used (plan 16.57)

- **FDS burns in two steps.** The fuel's carbon goes first to CO (two thirds by default) and soot; a second step burns those down to the final yields with the oxygen that is left. Short of air, the CO stays. This solver adds a fixed yield per kg burnt, whatever the oxygen.
- **FDS puts a flame out** when oxygen is under a limit that falls with temperature, and never above 600 °C. This solver burns until oxygen is essentially gone. The agent infers this is the main cause of the oxygen result at head height.
- **FDS reacts only the mixed part of a cell** each step (about 40 % in 0.1 s on 0.2 m cells, by the agent's arithmetic); this solver reacts the whole cell.
- **Smallest change:** one more transported quantity for the intermediate products (about a sixth more species transport), and an oxygen-limit test that stores nothing.

### 3. Gas radiation at 87 hours (plan 16.58)

- FDS solves radiation every third step and then only a fifth of its 100 directions: a fifteenth of the work per step. It uses one gray band, with the absorption coefficient tabulated once at the start.
- This solver, each step: 8 rays from every warm cell, 9 bands, and an iterative solve of the absorption in every cell touched.
- **Cheapest changes** (the agent's unmeasured estimates of the stage's speed-up): run the stage every third to fifth step (3 to 5 times); one gray coefficient tabulated from this project's own band tables (6 to 10 times); a radiation grid twice as coarse (about 16 times). The first two together would bring 87 hours to roughly 10 to 13.

### 4. What the comparison itself may be getting wrong

- **Thermocouples.** The measured temperatures are thermocouple beads. FDS compares through a bead model (1 mm, emissivity 0.85); this project compares with the raw gas temperature. In a hot layer facing cold walls a bead reads low: about 5 K at 130 °C, 30 K at 330 °C, 100 K at 630 °C as upper bounds (the agent's estimate). **Part of "reads 35 to 50 % high" may be the comparison.** Near 60 °C it is under 2 K, so closures are untouched.
- **Toxic dose.** FDS's CO term is 14 to 31 % above this project's between 100 and 5000 ppm; its CO₂ factor applies at any level, ours above 2 %; ours adds a small oxygen dose in clean air. FDS has no heat dose; ours is ISO 13571's.
- **Visibility** is the same in form and constants.
- **Soot.** FDS by default neither deposits nor ages soot, and this solver deposits, so deposition does not explain smoke arriving a third too soon.
- **Detectors.** FDS delays a smoke detector by the time smoke takes to enter it; ours has no lag.

### 5. Verification (the collaborator's "a lot of verification work")

FDS runs its whole verification suite nightly, each case passing on a stated tolerance against an exact or expected value. From this project's test names, three topics have no test: how accurately the flow carries a quantity, whether the flow conserves volume after the pressure step, and radiation geometry against exact answers. A first set of ten cases with closed-form answers, in the agent's order (tolerances are its proposals):

1. Isothermal through-flow: the divergence after the pressure step is within the solver's stated bound.
2. A block carried once across a box returns unchanged; mass within 0.1 %.
3. Solid-body rotation of a smooth profile on 16, 32 and 64 cells: the order of accuracy, recorded.
4. 16 kW into a sealed 8 m³ box: 24 kPa at 30 s.
5. Gas injected into a sealed box: +64,253 Pa, 1.365 kg/m³, 135.7 °C.
6. 1 kg of a species injected is 1 kg found; mass fractions sum to 1.
7. A reacting flow: each species' balance closes.
8. A fire in a room with adiabatic walls: enthalpy out equals heat released.
9. Radiation: the box's configuration factors, a bead between 20 and 500 °C walls (380.3 °C), a plane layer.
10. A steady insulated steel plate: the flux and each layer's temperature.

Cases 1 to 3 would also measure the two leading suspects of section 1 directly.

### 6. Data (the Validation Guide)

- **Smoke and gas at head height: not found.** No series in the guide's text has those instruments confirmed at 1.5 to 2.0 m. Reports to open first: NBSIR 88-3752 (NBS multi-room) and NIST TN 1455-1.
- **Far field:** DelCo Trainers (NIST TN 1953), NRCC Smoke Tower, UL/NIJ Houses, CSTB Tunnel; NIST Smoke Alarms only as ratios, its heat release being fitted.
- **Under-ventilated CO:** NIST FSE 2008, already held, is the only full-scale series.
- **FDS's own bias and scatter by quantity** is built with the guide and is not in the source; it has to be read from the published guide.
- The experimental uncertainties in the guide's text agree with the suite's and with plan 16.59.

### 7. Other differences (sprinklers, solids, pressure, stability)

- **Sprinklers:** no cooling of a link by water from an open head (later heads may open early); one drop size; drops fall straight down; spray cools gas only above 100 °C, so smoke at 60 to 100 °C is not cooled.
- **Growing a fire from materials:** no temperature profile inside the fuel, no char, one set of kinetics for every fuel. FDS's simplest route (a prescribed heat release per area once a surface reaches its ignition temperature, over a one-dimensional wall solver) is the nearest fit to what this solver already has.
- **Stability:** FDS redoes a step that breaks its limit and clips density while conserving mass; it never stops on a gas temperature. This solver checks between frames and stops at 3000 K. That bears on the ISOHept26 stop (plan 16.56).
- **Pressure zones:** this solver restarts zones at ambient when a door opens and ignores zone-to-zone leakage.

## A ranked list of what to do

Nothing here is started. Each solver item needs its rule fixed in the plan first.

| # | Action | Kind | Cost | Bears on |
| --- | --- | --- | --- | --- |
| 1 | The pressure-step test on one ATF and one NBS case | settings, two runs | hours | far field |
| 2 | Verification cases 1 to 3 | new tests | days | far field; verification |
| 3 | Score temperatures through a bead model, as a second reading beside the present one | post-processing, rule first | days | every temperature word |
| 4 | Oxygen-limit test in combustion | solver | days | oxygen at head height |
| 5 | Two-step CO | solver, one new field | a week or two | CO, toxic dose |
| 6 | Gas radiation every few steps, then one gray coefficient | solver | a week or two | head-height temperature; the 0.20 m cell |
| 7 | The other seven verification cases | new tests | a week or two | verification |
| 8 | Open the two reports for head-height smoke; add DelCo Trainers | data | days each | head-height smoke; far field |
| 9 | FDS's dose form as an option; detector entry lag | app | days | the egress reading |

## Limits

- Two agents read parts of `fire.f90`, `radi.f90`, `main.f90` and `cons.f90` from an older local copy while the download was briefly incomplete; they marked which. The GPU kernels of this solver were not reviewed.
- No claim here says FDS is right where this solver is wrong: FDS is the reference whose validation record is public, and the differences are where to look.
