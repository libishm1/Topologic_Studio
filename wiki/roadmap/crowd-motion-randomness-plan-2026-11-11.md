# Plan: randomness in how people move (2026-11-11)

The user: "Build the crowd randomness simulation as well, by scramble I meant people motion, not exactly doors randomness; write it as a plan we will build on it." This plan is what we build from, a stage at a time. Every mechanism comes from a paper; the table at the end says which papers were read for their numbers and which only had their DOI checked (read before building on them).

## Where the crowd stands (core/crowd.js, released d32f4d6)

People walk the walkable graph, one a node, at SFPE speeds (each person's unimpeded speed uniform within 25 % of 1.19 m/s, slowed by the density ahead), set off within 10 s, take the quickest exit (or the nearest), and step to the next node on their walk tree, or, when it is taken, to the free neighbour nearest the exit. Given a seed, every move is decided: the same seed gives the same crowd, and within a run nobody hesitates, wanders, speeds up or slows down except for the crowd round them.

## What "people motion randomness" means here

Not doors. Four ways real walkers vary, each measured or modelled in the literature:

1. **Speed wobbles in time.** A person's walking speed drifts about their own preferred speed and recovers, on a timescale of seconds.
2. **Steps are not always the best one.** People sometimes step sideways or hesitate; less so when they know the way and are in a hurry.
3. **Conflicts.** Two people going for the same space: sometimes neither gets it at once (the local pressure of a crowd).
4. **Following others.** People follow where others have gone (herding), which helps a little and harms a lot.

And three that are not motion but make a crowd vary: when people start (pre-movement), how fast each walks (between people), and which exit they choose.

## Mechanisms

### M1. Speed fluctuations (stage 1)

Corbetta, Lee, Benzi, Muntean and Toschi (2017, *Phys. Rev. E* 95:032316, doi:10.1103/PhysRevE.95.032316; read, arXiv:1610.07429), from 72,376 tracked walkers in a corridor: the speed `u` along the walk follows a Langevin equation in a double-well potential,

```
du = -4 alpha u (u^2 - u_p^2) dt + sigma_x dW,   alpha = 0.0625 m^-2 s, sigma_x = 0.16 m s^-3/2, u_p = 1.0 m/s
```

so about `u_p` the speed relaxes at `8 alpha u_p^2 = 0.5 /s` (a correlation time of about 2 s) with a stationary spread of `sigma_x / sqrt(2 x 0.5) = 0.16 m/s`, 16 % of `u_p`. Sideways the walker sways as a damped oscillator (`dv = (-2 beta y - 2 gamma v) dt + sigma_y dW`, beta 1.63 s m^-2, gamma 0.207 /s, sigma_y 0.16), about +-10 cm, and about one walker in 450 to 1,000 turns back.

**In Studio.** Each person's speed factor `f = u / u_p` follows the same equation scaled to their own unimpeded speed (their SFPE draw times f), integrated by Euler-Maruyama in steps of at most 0.1 s, kept above the unstable point (0.2 of u_p; turning back is left out: in an evacuation it is a different behaviour). The crowd's density slowing applies on top, as now. Sway is below the graph's 0.5 m spacing: stage 3 draws it.

### M2. Stochastic step choice: the floor field (stage 1)

Burstedde, Klauck, Schadschneider and Zittartz (2001, doi:10.1016/S0378-4371(01)00141-8) and Kirchner and Schadschneider (2002, doi:10.1016/S0378-4371(02)00857-9; read, cond-mat/0203461): each step a person picks a neighbouring cell (or stays) with

```
p_ij = N exp(k_D D_ij) exp(k_S S_ij) (1 - n_ij) xi_ij
```

`S` the static field (rising towards the exit, from the distance to it), `D` the dynamic field (M4), `n` occupation, `xi` 0 for walls, `N` normalising. Kirchner and Schadschneider ran `k_S` = 0.1, 0.4 and 10 (strong: nearly deterministic, "ordered"), at about 0.3 s a step.

**In Studio.** With `k_S` set, a person picks among staying and each free open neighbour (not into the fire, not a lift) with weight `exp(-k_S (d(u) - d(v)) / spacing)`, `d` the walk to their exit in metres, so `k_S` keeps its per-cell meaning on the 0.5 m graph. Levels offered: exact (as now, the limit `k_S` to infinity), 10 (their "ordered"), 2 and 0.4 (their intermediate). The deterministic limit is a theorem to prove (stage 4).

### M3. Friction in conflicts (stage 2)

Kirchner, Nishinari and Schadschneider (2003, *Phys. Rev. E* 67:056122, doi:10.1103/PhysRevE.67.056122; abstract read, cond-mat/0209383): when two or more people choose the same cell in one step, with probability `mu` none of them moves, else one at random; this changes how evacuation time depends on the other parameters, not only its size, and gives arching at exits as in granular flow.

**In Studio.** Needs a parallel update: every person chooses, then conflicts are resolved, then all move. The crowd now updates in sequence (nearest the exit first), so stage 2 adds a parallel mode beside it, not in place of it.

### M4. Herding: the dynamic floor field (stage 2)

Kirchner and Schadschneider (2002): a person leaving a cell adds one to `D` there; `D` diffuses with probability `alpha` and decays with `delta` (0.3 in most of their runs). Evacuation was fastest at a small non-zero `k_D` with intermediate `k_S` (0.4), slowest with strong herding and weak `k_S` (their panic regime); the ratio `k_D / k_S` acts as Helbing, Farkas and Vicsek's panic parameter (2000, *Nature* 407:487, doi:10.1038/35035023).

**In Studio.** `D` is the slime mould's trail (core/slimeEgress.js lays, spreads and fades one): the crowd lays it as it walks and senses it with weight `k_D`.

### M5. Pre-movement (stage 3)

When each person starts, from measured distributions: Lovreglio, Kuligowski, Gwynne and Boyce (2019, *Fire Safety J.* 105:107-128, doi:10.1016/j.firesaf.2018.12.009; to read) and Purser and Bensilum (2001, doi:10.1016/S0925-7535(00)00066-7; to read). Studio's performance check already has measured pre-movement presets (core/rsetPresets.js, PreMovement.lean): the crowd should draw from the same ones instead of 0 to 10 s.

### M6. Speeds between people (stage 3)

Bosina and Weidmann (2017, *Physica A* 468:1-29, doi:10.1016/j.physa.2016.09.044; to read): walking speeds aggregated from the literature, by group and setting, to replace the uniform 25 % spread with a measured distribution.

### M7. Exit choice (stage 3)

Lovreglio, Fonzone and dell'Olio (2016, *Transp. Res. A* 92:59-75, doi:10.1016/j.tra.2016.06.018; to read): a mixed logit for exit choice in evacuations. The quickest choice (crowd.js) becomes the mean of a logit over the exits' times, with the paper's coefficients.

### M8. Sway drawn (stage 3)

M1's sideways oscillator drawn on the figures (a few centimetres off the walk), for the look only.

## Many runs, not one

With randomness on, one run means little. Each setting runs several seeds (5 by default, 10 at most on a desktop, 3 on a phone) and the panel reports the median and the range of the last one out and of each exit's people; the view plays the first. The same rule as the Flamescapes repeat runs shown as a range.

## Validation

- **Off is exact.** With every mechanism off, every run is bit for bit today's (the crowd tests as they are).
- **Invariants on.** One person a node, no one into a place the fire has made untenable, everyone out who can reach an exit, every seed (tests over many seeds).
- **M1's statistics.** A long run of one person's speed factor: mean 1 within 2 %, spread 0.16 within 15 %, correlation at 2 s about 1/e.
- **M2's limit.** As `k_S` grows, the walks approach the exact ones; as it falls, evacuation times rise and spread (Kirchner and Schadschneider's trend with no herding).
- **Against experiments (stage 4).** The flow through a bottleneck by its width (Kretz, Grünebohm and Schreckenberg 2006, doi:10.1088/1742-5468/2006/10/P10014; to read) and speed against density (Seyfried, Steffen, Klingsch and Boltes 2005, doi:10.1088/1742-5468/2005/10/P10002; to read), with randomness on and off: randomness must not move the crowd off the SFPE values it is timed by.
- **Lean (stage 4).** The step probabilities are a distribution; their limit as `k_S` grows is the best step; M1's drift has stable points at `+-u_p` and the linearised rate `8 alpha u_p^2`; the parallel update with friction keeps one person a node.

## Stages

| Stage | What | Off by default |
| --- | --- | --- |
| 1 (built 2026-11-11) | M1 speed fluctuations, M2 stochastic step choice, many runs as a range; People leaving: "Walking speed varies (Corbetta et al. 2017)" and "Steps" (exact, mostly direct, wandering, disoriented), "Runs" | yes |
| 2 | M3 friction (parallel update beside the sequential one), M4 herding on the slime trail | yes |
| 3 | M5 pre-movement from the performance check's presets, M6 measured speeds, M7 logit exit choice, M8 sway drawn | yes |
| 4 | Validation against Kretz 2006 and Seyfried 2005; the Lean results above | |

## Stage 1 as built (2026-11-11)

`core/crowd.js` `fluctuate` and `kS` (WALK_FLUCTUATION, fluctuationDrift, STEP_RANDOMNESS), drawn from a random stream of their own so a run with both off is bit for bit the old one (a test). People leaving: "Walking speed varies over time", "Each step" (straight on, mostly direct kS 10, wandering kS 2, disoriented kS 0.4) and "Runs" (1, 3, 5 or 10 seeds; 3 at most on a phone): the view plays the first seed, the panel gives the last one out over all of them (lowest, median, highest).

Measured, one person at 0.6 m/s over three seeds in a corridor: the speed factor's mean 0.976 and spread 0.161 (six seeds). The mean sits just under 1 because Corbetta's potential is steeper above the preferred speed than below it. Last one out, three seeds each (s):

| | Office, two storeys | NIBS Clinic |
| --- | --- | --- |
| Exact | 65 to 73 | 283 to 350 |
| Speeds wobble | 66 to 73 | 305 to 335 |
| Steps, kS 10 | 66 to 73 | 426 to 452 |
| Steps, kS 2 | 75 to 102 | 492 to 515 |
| Steps, kS 0.4 | 484 to 543 | 1,748 to 1,784 (up to 6 still inside at 30 min) |
| Both, kS 2 | 85 to 99 | 462 to 528 |

In free flow kS 10 walks as the exact model does; in the Clinic's long queues people step aside and hesitate, as in Kirchner and Schadschneider's jams, which costs time. A run costs 0.15 to 0.5 s on the Clinic against 0.1 s exact.

## Sources and how far each was checked

| Source | DOI on Crossref | Read for its numbers |
| --- | --- | --- |
| Corbetta et al. 2017 | yes | yes (arXiv 1610.07429: equations, Table 1) |
| Kirchner and Schadschneider 2002 | yes | yes (cond-mat/0203461: Eq. 1, parameters, results) |
| Burstedde et al. 2001 | yes | abstract (cond-mat/0102397) |
| Kirchner, Nishinari and Schadschneider 2003 | yes | abstract (cond-mat/0209383) |
| Helbing, Farkas and Vicsek 2000 | yes | no |
| Helbing and Molnár 1995 (social force, doi:10.1103/PhysRevE.51.4282) | yes | no |
| Lovreglio et al. 2019; Purser and Bensilum 2001 | yes | no |
| Bosina and Weidmann 2017 | yes | no |
| Lovreglio, Fonzone and dell'Olio 2016 | yes | no |
| Haghani and Sarvi 2018 (review of empirical methods, doi:10.1016/j.trb.2017.06.017) | yes | no |
| Seyfried et al. 2005; Kretz et al. 2006 | yes | no |
