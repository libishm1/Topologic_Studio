# Fire calibration: before and after, and the ways back (2026-11-12)

What the branch `next/calibration-ui` (pull request #2) changes in the fire, measured on the same inputs before and after, and how to undo it. Every number is from a result file named beside it. Commit hashes are as published; `commit-hash-map-2026-11-12.md` maps the ones cited before the history rewrite.

## 1. Heat and mass (no measured data)

`tools/calibration/heat-closure.mjs`, adiabatic walls. Old: the live site's transport (`benchmark/results/2026-11-12-heat-closure-old.json`). Final: d3c5751 (`benchmark/results/2026-11-12-heat-closure-o2.log`).

| Cells | Sealed box heat at 60 s, old | final | Burner room closure at 30 s, old | final |
| --- | --- | --- | --- | --- |
| 0.20 m | 1.611 | 1.042 | 3.401 | 0.972 |
| 0.27 m | 1.375 | 1.012 | 5.087 | 0.959 |
| 0.34 m | 1.138 | 1.001 | 6.754 | 0.958 |

A sealed box should keep 1.000 of its heat; a burner room should account for 0.9 to 1.0 of the heat released. The porous-burner case, burner cells that hold no heat of their own, still fails at 0.82 to 0.89, with gas near 2,600 K. No validation test and no design fire uses porous cells. A room's contents do when they burn: this is the open item of section 6.

## 2. Held-out measured fires

Plan section 12.10 has the full tables. Below, the stricter of the two pre-registered rules, on 0.27 m cells:

| Quantity | Before the burner fix (1fc44f7's solver) | Final (d3c5751) |
| --- | --- | --- |
| Smoke layer depth | indicative (ATF's two stopped runs count against it) | calibrated: δ 1.08, σ̃M 0.15, 5 of 5 series |
| Smoke layer temperature | not calibrated | not calibrated (indicative on 0.34 m) |
| Gas under the ceiling | not calibrated | not calibrated (indicative on 0.34 m) |
| O2, CO2, door outflow, neutral plane, smoke obscuration | not calibrated | not calibrated |

The verdicts on the earlier four-series run (62d786c, before ATF was added) were: depth calibrated (δ 1.05, σ̃M 0.15); temperature and ceiling jet indicative. Adding the ATF corridors left the depth calibrated, and moved temperature and the ceiling jet on 0.27 m to not calibrated, because ATF's trees near the fire read 1.9 to 2.6 times measured.

Sprinkler first activation (`benchmark/results/2026-11-12-sprinkler-o2-stats.txt`), with each test's measured fire: all four series are within the calibrated band, pooled δ 1.03 (σ̃M 0.16) over 29 tests. Before the burner fix the pooled figure was 1.04 (0.16). The walls' convection coefficient was set on these tests, so they are reported apart.

## 3. The example buildings

`tools/backend-bench.mjs`: class A design fire, 300 s, one run each. The scratch results are summarised here; the live solver is ea185d4.

| Building | Solver | CPU: MJ, peak MW, untenable nodes | GPU: MJ, peak MW, untenable nodes |
| --- | --- | --- | --- |
| Duplex | live | 109.4, 1.04, 12 | 112.3, 1.05, 15 |
| Duplex | final | 10.9, 0.30, 12 | 10.5, 0.27, 15 |
| Office | live | 129.1, 1.62, 450 | 194.5, 3.05, 760 |
| Office | final | 105.4, 1.04, 30 | 105.9, 1.05, 32 |
| Clinic | live | 105.3, 1.04, 85 | 105.4, 1.04, 81 |
| Clinic | final | 105.3, 1.04, 24 | 105.4, 1.04, 22 |

- **The CPU and the GPU now agree.** The live solver's Office fire read 1.6 MW on the CPU and 3.1 MW on the GPU.
- **Fewer places become untenable.** The live transport made heat, which spread to more places.
- **The Duplex's fire is oxygen-limited.** Its room runs out of oxygen, so the fire releases about a tenth of its design curve. The live solver kept releasing heat there with no oxygen left. The same test burner, in a sealed 3 × 3 × 2.5 m room, used about 42 % of the room's oxygen before its flame cells ran dry, which matches real fires going out at 12 to 15 % oxygen.
- **The pan fire.** The Clinic's unattended cooktop is a pan of oil. Before pools became design fires, it went out about 45 s in on the heat-conserving solver. Now it holds its steady pool burning rate, 19 kW, for the whole run. The live solver grew it to 500 kW with heat its transport made.

## 4. Speed

The final solver runs at about half the live one's speed, because of the cut steps and the flux transport. The GPU is still faster than real time on the examples: Office 2.65 times and Clinic 1.31 times, both measured with other jobs running. The CPU runs Office at 0.57 and Clinic at 0.51 times real time on its coarser grid, with the same caveat.

## 5. Ways back

1. **Before the merge:** nothing changes on the live site until the user merges pull request #2 by hand. The site stays at ea185d4.
2. **After the merge, all of it:** the tag `restore/live-2026-11-11` marks ea185d4. Push that commit to `main` (`git push origin restore/live-2026-11-11^{commit}:main --force-with-lease`, on the user's word only). GitHub Pages redeploys.
3. **After the merge, the fire only:** in the app, Fire solver "Previous" runs the site's solver as it was before 2026-11-12, with no redeploy.
4. **Settings:** the new settings have their own keys and an allow-list migration, so an older build ignores them. Stored GPU speeds are versioned (`gpuRate.v2`), so the old speeds are dropped, not misread.
5. **Backend:** Render is unchanged.

## 6. Open items

- Porous cells: the porous-burner closure check (section 1).
- Head-height temperature, which decides untenability, has no held-out verdict of its own; the industry-practice note's gap G1 gives the step.
- The FDS guide's 10-bin weighting (gap G4).
- A non-NIST dataset (gap G7).
- Smoke visibility needs two more held-out series; the PRISMA review ranks the candidates.
