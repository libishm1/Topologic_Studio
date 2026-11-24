# People leaving: calibrating the density reading (2026-11-24)

This note fixes the rules for calibrating how the people-leaving model (`topologicpy-web-frontend/src/core/crowd.js`) reads the density that slows a step. Sections 1 to 6 were written and committed before any code for it was changed and before any calibration run. Section 7, the results, was added afterwards and reports what came out, pass or fail.

It follows the discipline of the fire calibration: rules before runs, nothing tuned on held-out data, failures reported as measured.

## 1. The fault to be corrected

A step's speed is the person's own below 0.54 persons/m², and at most SFPE's k (1 - a D) above it. D is read as the other people within 1 m of the node ahead, over the area of all the nodes there.

That reading is low in a moving crowd, for two reasons seen in the code:
- The node ahead is free, or no one could step on to it, and the walker is left out of the count; but both cells stay in the area. In a single file at 1 person/m (2 persons/m² on a 0.5 m lattice) the reading is about 1.2 persons/m².
- People step into gaps, so the surroundings of the node ahead are thinner than the crowd round the walker.

Measured before any change (harness of 2026-11-24, default options):
- RiMEA Test 16, single file: 8 of 14 points inside the experiments' 10 to 90 % band. Too fast from 1.25 persons/m up: 0.75 m/s at 1.26 /m against a 90 % value of 0.66.
- RiMEA Test 4, 10 m wide corridor: 1.044 m/s at 1.21 persons/m² (SFPE 0.949), 0.767 m/s at 2.44 (SFPE 0.492, Weidmann 0.469).

## 2. Free parameters

Four, each with a short list of candidate values fixed here. The current values come first.

| Parameter | Candidates | Meaning |
| --- | --- | --- |
| Centre | the node ahead; the walker's own node | Where the disc the density is read over is centred |
| Radius | 1.0, 1.5, 2.0 m | The disc's radius |
| Area | all the nodes' cells in the disc; those cells less the walker's and the target's | The floor the count is divided by. The second is the floor the others could stand on: the walker's cell and the free cell ahead hold none of them |
| Speed floor | 0.15, 0.10, 0.05 m/s | The least speed of a step |

That is 2 x 3 x 2 x 3 = 36 combinations. All 36 are run; none is added after seeing results. If none does better than the current reading, the current reading stays.

Not free, and not touched: SFPE's k = 1.40 m/s, a = 0.266, the stair's k = 1.08 m/s, the free-walking threshold of 0.54 persons/m², the 1.9 persons/m² cap on the density in the queue before a door, the door and passage capacities, the walker being left out of the count, and the lattice.

## 3. Calibration data

Only RiMEA Test 16's single-file percentiles (`tools/validation/rimea-test16-percentiles.json`; from rimea.de, raw data Chraibi and Subaih, Zenodo doi:10.5281/zenodo.8378592): the 10, 50 and 90 % values of mean speed against line density.

The model's side is the harness's own RiMEA 16 case, unchanged: a 200 m file of nodes, seven set densities from 0.25 to 1.75 persons/m, two populations (1.0 m/s for all; the default 1.19 m/s, 25 % either way), three seeds, speeds read between 10 and 40 s in x 40 to 80 m. That gives 14 points. The runs use the model's default options, so `passageFlow` is off: the reading has to stand without the corridor capacity, which caps a single file on its own account.

## 4. Held-out data

Not used to choose any parameter, and looked at only after the choice is fixed:

1. **RiMEA Test 4**, the fundamental diagram in a 10 m wide corridor, against the SFPE relation (the model's own) and Weidmann's (1993) curve. This is the main held-out check: it is two-dimensional, and the calibration data is one-dimensional.
2. **RiMEA Test 12d**, the flow through bottlenecks 0.8, 1.0 and 1.2 m wide.
3. **Every other case of the harness**, with and without `passageFlow`: RiMEA 12b, NIST Verif.5.1, the SFPE flows, NIST Verif.5.2 and the rest.

Held-out data the plan of 2026-11-11 names that is not in the repository and is therefore not run here: the bottleneck experiments of Kretz et al. (2006) and of Hoogendoorn and Daamen (2005), the corridor experiments of Zhang et al. (2011), and the building evacuations of NIST TN 1822's Table 2. The Jülich bottleneck series (Seyfried et al. 2009), which the plan put on the calibration side, is not in the repository either and is not used.

## 5. Objective and choice

For each combination, over the 14 points:
- **N**, the number of points whose mean speed lies between the 10 and 90 % values at the measured density. This is the test's own criterion.
- **J**, the mean of ((v - p50) / h)², with h half the distance from the 10 to the 90 % value. It measures how far the speeds sit from the experiments' median, in units of the band.

The choice: the largest N; among those, the smallest J. Combinations whose J is within 5 % of the best count as tied, and a tie goes to the one nearest the current reading: fewest parameters changed, then the smaller radius, then the higher floor.

## 6. Pass rules for the held-out data

Fixed now, applied once, to the chosen combination:

1. **RiMEA 4.** At every measured density at or below 3 persons/m², the mean speed is within 15 % of the SFPE relation at that density. The same points against Weidmann's curve are reported beside it, with 25 % as the mark; that second comparison is reported, not required, since the model does not use Weidmann's relation. Above 3 persons/m² the speeds are reported only: there the two reference curves part (SFPE reaches zero at 3.76 persons/m²) and the lattice is full at 4.
2. **RiMEA 12d.** The flows stay at the door's SFPE capacity to within 2 % and rise with the width. Their ratio to the measured flows (about 1.9 persons/s for each metre) is reported; it is about one half by design and is not a criterion.
3. **The rest of the harness.** No case that passed before fails after, by default or with `passageFlow`.

What may be said afterwards:
- If RiMEA 16 passes with the chosen reading, the reading is *calibrated on single-file data*. That is a fit, not a validation.
- Only if rule 1 passes may the reading be said to *hold on held-out two-dimensional data*, and then only against reference curves, not against experiments.
- If rule 1 fails, the result is reported as a failure. The parameters are not chosen again in the light of it.

A new reading changes the model's default results, so it goes behind an option that is off by default. Section 7 says what the default should be.

## 7. Results (added after the runs)

Run: `node tools/validation/crowd-calibrate.mjs`, then the harness with `--reading calibrated`, by default and with `--passage-flow`. Files: `benchmark/results/2026-11-24-crowd-density-calibration.json` (all 36 combinations, point by point) and `2026-11-24-crowd-density-calibrated-default.json`, `2026-11-24-crowd-density-calibrated-passage-flow.json`.

One thing was not by the book: before the sweep, one combination (the area corrected, the rest as now) was run once to check that the option reached the model. Its result was seen. The choice below was then made by the rule of section 5 over all 36, by the script, with no hand in it.

### 7.1 The choice

Fourteen of the 36 combinations put all 14 points in the band. Among them one has the smallest J, with no tie inside 5 %:

| | Centre | Radius | Area | Floor | Points in band | J |
| --- | --- | --- | --- | --- | --- | --- |
| Current | node ahead | 1.0 m | all cells | 0.15 m/s | 8 of 14 | 1.348 |
| Chosen | node ahead | 2.0 m | cells the others could stand on | 0.15 m/s | 14 of 14 | 0.165 |

What the candidates showed:
- Leaving the walker's and the target's cells out of the area was enough at 1.0 m (14 of 14, J 0.297). A wider disc lowered J further.
- Centring on the walker's own node was worse in most combinations.
- A lower floor never helped. With the area corrected it hurt: at 0.05 m/s the dense file all but stops (2 of 14 at 1.0 m).

Single-file speeds, 1.0 m/s for all (the other population is within 0.02 m/s of it from 0.44 persons/m up):

| Density, persons/m | Before, m/s | After, m/s | Experiments' 10, 50, 90 % (at the density measured after) |
| --- | --- | --- | --- |
| 0.26 | 0.999 | 0.997 | 0.864, 1.165, 1.475 |
| 0.44 | 0.999 | 0.961 | 0.746, 1.111, 1.413 |
| 0.67 | 0.988 | 0.872 | 0.616, 0.994, 1.319 |
| 0.96 | 0.892 | 0.554 | 0.408, 0.680, 1.023 |
| 1.28 | 0.750 (outside) | 0.321 | 0.197, 0.402, 0.626 |
| 1.53 | 0.513 (outside) | 0.178 | 0.105, 0.260, 0.467 |
| 1.74 | 0.506 (outside) | 0.150 | 0.066, 0.190, 0.368 |

The chosen reading is in the code as `DENSITY_READING_CALIBRATED`, taken with `densityReading: "calibrated"`. It is calibrated on single-file data. Every speed after sits below the experiments' median, and the last is the floor itself.

### 7.2 The held-out data

**Rule 1, RiMEA 4: fails.**

| Measured density, persons/m² | Speed, m/s | SFPE | Difference | Weidmann | Difference |
| --- | --- | --- | --- | --- | --- |
| 0.65 | 1.149 | 1.159 | -1 % | 1.241 | -7 % |
| 1.16 | 1.023 | 0.969 | +6 % | 0.975 | +5 % |
| 2.27 | 0.702 | 0.556 | **+26 %** | 0.519 | **+35 %** |
| 3.28 (reported only) | 0.303 | 0.178 | | 0.274 | +11 % |
| 4.20 (reported only) | 0.126 | none | | 0.129 | -2 % |

At 2.27 persons/m² the speed is 26 % above the SFPE relation, outside the 15 % of the rule, and 35 % above Weidmann's, outside its 25 % mark. It is better than before (0.767 m/s at 2.44 persons/m², 56 % above SFPE and 64 % above Weidmann), but it does not pass. The reading that fits a single file still reads low in a wide crowd, where people step sideways into gaps.

**Rule 2, RiMEA 12d: passes.** 0.659, 0.922 and 1.186 persons/s at 0.8, 1.0 and 1.2 m, against door capacities of 0.658, 0.921 and 1.184; rising with the width. The ratios to the measured flows are 0.43, 0.49 and 0.52, as before.

**Rule 3, the rest of the harness: fails.**

| | Default reading | Calibrated reading |
| --- | --- | --- |
| By default: pass, fail | 22, 4 (RiMEA 12b, RiMEA 16, NIST Verif.5.1, SFPE flows) | 24, 2 (NIST Verif.5.1, SFPE flows) |
| With `passageFlow`: pass, fail | 25, 1 (RiMEA 16) | 25, 1 (SFPE flows) |

With `passageFlow` the SFPE flows passed before and fail after: the 1 m stair passes 0.534 persons/s where SFPE gives 0.71 (0.707 with the default reading). The 1.2 m and 2.4 m stairs, the corridor and the door keep their values.

By default no case that passed now fails, but two results that already failed get much worse, and they are the ones a user would meet:
- A stair drawn as one file of nodes passes 0.153 persons/s (0.542 before).
- A corridor 1 m wide passes 0.537 persons/s (1.188 before; SFPE 0.79). RiMEA 12b passes for that reason, not for a good one.
- NIST Verif.5.1: the last person is out at 502 s (214 s before; 80 s with `passageFlow` and the default reading).

The cause is the one found on 2026-11-11 as finding 3. A truer, higher reading brings back the crawl into a queue wherever the density there is not capped. Before a door it is capped; before a corridor or a stair it is capped only with `passageFlow`; and on a 1 m stair itself the cap of 1.9 persons/m² now holds the speed at its lowest, 0.53 m/s.

### 7.3 Verdict

- The reading is calibrated on single-file data: RiMEA 16 goes from 8 to 14 of 14.
- It is not validated. It failed two of the three rules set for the held-out data, and by those rules the parameters are not chosen again.
- **The default stays as it was.** `densityReading` is left out by default, and a run without it is exactly as before. The app does not offer the calibrated reading.
- The calibrated reading stays in the code as an option for analysis, with these results stated beside it.

### 7.4 What a next round would need

These are proposals, not results. Each needs its own rules written first.
- A cap on the queue's density before every passage, on by default, so that a truer reading cannot bring the crawl back. Until then no higher reading can be the default.
- Two-dimensional calibration data. A single file cannot tell the model how people step sideways into gaps. The corridor experiments of Zhang et al. (2011) would do, with RiMEA 4's curves and the bottleneck experiments kept back.
- For stairs, a reading along the stair (a headway) in place of the disc.
