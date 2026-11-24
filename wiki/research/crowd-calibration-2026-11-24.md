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

## 7. Results

Not yet run.
