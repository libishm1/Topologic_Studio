# Why the learned escape fails near the fire, and a shielded learner that does not (2026-11-05)

Research note. Code: `topologicpy-web-frontend/src/core/rlSafe.js` (new), its tests `src/core/rlSafe.test.mjs`, the benchmark `tools/rl-bench.mjs`; proofs: `formal/fireegress/FireEgress/SafeEscape.lean`. Every number in sections 3, 4 and 7 is printed by `node tools/rl-bench.mjs --summary benchmark/results/2026-11-05-rl.json` (the table named in brackets, for example [Rates]); theorem counts come from `formal/fireegress/axioms-all-2026-11-05.txt`. Corpus models are anonymised as corpus-1 to corpus-6; their files and names are not in the repository.

## 1. Summary

The question was why Studio's Q-learning (after Jabi et al. 2019) fails so often when the fire is close, whether the defect lies in the training or in the model, and whether a better learned pathway is possible.

1. **Most failures near the fire are races that no route can win.** Over 848 runs on nine buildings, the app would show a path that does not reach the exit in 559; in 508 of them (90.9%) no escape to the chosen exit exists at all [Scenarios where an escape ... exists]. With the fire within two links of the shortest route and spreading at one or two links per move, an escape exists in 2 of 144 scenarios (1.4%, 95% interval 0.4 to 4.9%) [Rates]. The app's note says so, but the card presents it as a failed agent.
2. **When an escape exists, the shipped method fails in 15.0% of runs (51 of 340, 11.6 to 19.2%), and closeness does not raise this**: 12.5% with the fire within five links of the route, 15.7% farther away [Rates].
3. **The defect is in the training, not in the reward model.** In 0 of the 51 failures is the exact optimum of rl.js's own reward unable to escape; in 9 the learned policy itself does not escape (all 9 walk into the fire); in 42 (82.4%) the policy escapes but the path the app draws, a greedy walk that may not revisit a node, gets stuck [Where the shipped method's failures come from]. All 42 failing drawings come from policies that walk in loops [How the drawn path fails]. The loops are learned from exploration: with exploration set to zero the same learner fails in 0 of 74 checked scenarios, against 11 of 74 at the shipped 0.2, and more episodes make the drawing worse (43.2% at 4,000) [One knob at a time] (section 4.4).
4. **The model has defects too, but they are not what the user sees.** The node-only state of Jabi et al. (2019), which rl.js no longer uses, fails in 40.5% of the checked scenarios. The shipped reward's optimum escapes later than the earliest escape in 24 of 339 cases and in one case does not escape at all; a Lean counterexample shows why (`legacy_reward_prefers_staying`). And training towards one exit misses the 102 of 848 runs (12.0%) in which only another exit can still be reached [Rates].
5. **A better learned pathway is possible.** The new method learns on the time-expanded states (node, step) inside a shield computed from the fire's escape-deadline map, with values initialised from an admissible potential and an optimality certificate. It escapes in 340 of 340 scenarios to the same exit and 442 of 442 to any exit, every time in the earliest number of moves and certified (to any exit, 81.9% within 400 trials and all within 20,000), at a median of 1.8 ms against 30.4 ms for the shipped learner (section 7). Its safety and the certificate are proved in Lean (33 results, standard axioms only).

## 2. The method under study

rl.js is tabular Q-learning on the navigation graph. With a fire, the state is (move, step); the reward is -0.1 per median-length link (proportional to length), +10 on entering the exit and -10 on entering a node at or after its ignition step, both terminal; gamma is 0.99; the shaping potential is minus the fire-free distance to the exit, zero at terminals and at the step limit (QLearning.lean); untried moves start at their one-step shaped value; exploration is epsilon-greedy with epsilon 0.2; the step size is 0.5; the app runs 400 episodes of at most 200 steps (`useStudio.js`). The fire is the hop front from its origin divided by the fire speed (links per move). The card shows a greedy walk that never revisits a node, "Late-training success", the policy's return against the exact optimum of the same reward, and whether any escape exists (a breadth-first search).

Jabi et al. (2019) used the node alone as the state, 3,000 episodes of at most 100 steps and unstated rewards and rates (section 5.1). rl.js already corrected the state, the terminal potentials and the horizon (the 2026-10-27 review, findings W5 and LG6).

## 3. Experiment

**Corpus.** Six corpus models from `benchmark/manifest.json` (1,092 to 33,654 nodes) and the three shipped examples with exit doors (duplex apartment, office on two storeys, courtyard ring), built as `tools/benchmark.mjs` builds them (web-ifc extraction, the 0.5 m lattice, exits by `tools/exits.mjs`) [Models]. The buildingSMART sample scene has no exit doors and was left out.

**Scenarios.** Per model, four starts drawn from floor nodes at least ten links from an exit (seeded by the model's SHA-256); the exit is the nearest by walking distance, as the app's "nearest exit" picks it; the fire-free route is A*. For each start, one fire origin at hop distance 0, 1 to 2, 3 to 5, 6 to 10, 11 to 20 and more than 20 links from that route, each at fire speeds 0.25, 0.5, 1 and 2: 212 fires, 848 runs, 340 with an escape to the chosen exit and 442 with an escape to some exit.

**Methods compared on each run.** (a) The shipped `qLearningPath` at the app's defaults. (b) A copy of its training loop with one setting changed at a time (episodes 100, 1,000 or 4,000; epsilon 0.05, 0 or decaying from 1 to 0.05; step size 0.1 or 1; an optimistic prior; the node-only state; the deadline shield; step limits 100 and 400), run on the 74 speed-1 scenarios with an escape; the copy reproduced rl.js exactly in 848 of 848 runs [Checks]. (c) The exact optimum of the shipped reward (backward induction, as rl.js computes its "best return"), walked with ties to the first move; its value matched rl.js's in 848 of 848 runs. (d) The new method, to the chosen exit and to any exit, and a variant that masks only nodes untenable on arrival. (e) The exact earliest escape (breadth-first search).

**Classification of failures.** Given an escape: *model* if the optimum of the shipped reward does not escape; *training* if it does but the learned policy, walked with revisits allowed, does not; *drawing* if the policy escapes but the drawn no-revisit path does not.

**Machine.** One Node 24.21 process on an Intel Core i7-14700 (28 logical cores, one used), Windows 11; another training job shared the machine, so timings are indicative. Run with `node tools/rl-bench.mjs --prepare` then `node tools/rl-bench.mjs` (about 15 minutes).

## 4. Diagnosis

### 4.1 Feasibility first

How often an escape to the chosen exit exists falls steeply with closeness and speed [Scenarios where an escape ... exists]:

| Fire to route (links) | speed 0.25 | speed 0.5 | speed 1 | speed 2 |
|---|---|---|---|---|
| 0 | 19.4% | 8.3% | 0.0% | 0.0% |
| 1 to 2 | 27.8% | 16.7% | 5.6% | 0.0% |
| 3 to 5 | 66.7% | 36.1% | 19.4% | 0.0% |
| 6 to 10 | 88.9% | 69.4% | 44.4% | 0.0% |
| 11 to 20 | 97.1% | 82.9% | 62.9% | 8.6% |
| more than 20 | 100.0% | 97.0% | 81.8% | 45.5% |

At the app's default speed of one link per move the fire is as fast as the evacuee, and a fire on or next to the route leaves no escape in 70 of 72 such runs. Any learner "fails" there; the exact search proves that nothing could succeed. This, not learning, is most of what a user sees near the fire.

### 4.2 When an escape exists

The shipped method's drawn path fails in 51 of 340 runs (15.0%), with no trend in closeness: 9 of 72 (12.5%, 6.7 to 22.1%) with the fire within five links and 42 of 268 (15.7%, 11.8 to 20.5%) farther away [Rates]. Per distance and speed the cells hold 2 to 34 runs each and range from 0% to 26.7% without order [Shipped Q-learning: share ...]. Where the fire cuts the fire-free shortest route (22 runs) the policy itself fails more often, 18.2% against 1.6% elsewhere, which is the effect a user would attribute to closeness; but such runs are few once infeasible ones are set aside [by whether the fire cuts ...].

### 4.3 Where the failures come from

| Class | Runs | Share of the 51 failures |
|---|---|---|
| model: the shipped reward's optimum does not escape | 0 | 0% |
| training: the optimum escapes, the learned policy does not | 9 | 17.6% |
| drawing: the policy escapes, the drawn path does not | 42 | 82.4% |

[Where the shipped method's failures come from]. The one run whose reward optimum does not escape is one the learner happened to escape (section 4.5). Every training failure ends in the fire (9 burn, 0 at the step limit) [Objectives and lengths]. Along the walk of the shipped reward's optimal policy, a median of 50% of the (move, step) pairs had ever been updated by training when the policy failed, against 78% when it succeeded [Objectives and lengths]: a table indexed by the step is sparse after 400 episodes (a median of 2,108.5 entries [To any exit]) and does not carry what it learned at one step to the next.

### 4.4 The mechanism: a pessimistic table, exploration and the drawing rule

The 42 drawing failures end with the walk stuck (40) or in the fire (2); in all 42 the policy walk revisits a node (median 2 revisits), and so do 92 of the 289 successful runs; tied values play no part [How the drawn path fails]. The policy learns loops, and the drawing, which forbids revisits, leaves the trained path at the loop and enters untrained states, where the prior knows nothing of the fire.

The loops come from exploration replayed by a table that ranks trained moves above untried ones. rl.js starts an untried move at its one-step shaped value, about zero, while the shaped value of a move on an escaping trajectory is near +10 discounted; by Wiewiora's equivalence the initial table is pessimistic against escaping. Once an exploratory episode escapes, its moves, loops included, outrank every untried move at the same steps, and the greedy policy replays them. The knobs, each changed alone on the 74 speed-1 runs with an escape, agree [One knob at a time]:

| Variant | Drawn path fails | Policy walk fails | Shipped failures fixed | Shipped successes broken |
|---|---|---|---|---|
| shipped (400 episodes, epsilon 0.2, step size 0.5, 200 steps) | 14.9% | 5.4% | | |
| epsilon 0.05 | 1.4% | 0.0% | 10 of 11 | 0 |
| epsilon 0 | 0.0% | 0.0% | 11 of 11 | 0 |
| epsilon from 1 to 0.05 | 13.5% | 6.8% | 6 | 5 |
| 100 episodes | 12.2% | 9.5% | 5 | 3 |
| 1,000 episodes | 33.8% | 2.7% | 6 | 20 |
| 4,000 episodes | 43.2% | 1.4% | 5 | 26 |
| step size 0.1 | 8.1% | 8.1% | 7 | 2 |
| step size 1 | 32.4% | 2.7% | 6 | 19 |
| optimistic prior (+10 on every untried move) | 77.0% | 66.2% | 2 | 48 |
| node-only state (Jabi et al. 2019) | 40.5% | 43.2% | 4 | 23 |
| deadline shield | 17.6% | 0.0% | 7 | 9 |
| step limit 100 or 400 | 14.9% | 5.4% | 0 | 0 |

More training improves the policy (its walk fails in 1.4% after 4,000 episodes) but worsens the drawing (43.2%): more exploratory escapes, more loops. Without exploration the learner follows its prior down the fire-free distance gradient and learns only where the fire stops it, so it learns no loops and fails in none of these runs. A uniformly optimistic prior is far worse, because a table indexed by the step has too many untried entries to exhaust in 400 episodes and the greedy walk keeps leaving what it learned. The deadline shield (section 6.2) makes every policy walk escape, but the drawing rule still fails 17.6% of runs, since it can strand the walk where the shield leaves only visited nodes. What the data point to is a prior that ranks moves by their true cost when the fire does not interfere, explored without noise; the new method's potential is exactly that, and needs no learning at all in 335 of 340 runs (section 7).

### 4.5 Defects of the model

1. **The state (Jabi et al. 2019).** With the node alone as the state the same learner fails in 40.5% of the 74 checked scenarios and its policy in 43.2% [One knob at a time]. Lean proves that no policy on the node alone can be right from every start (`node_policy_fails`). rl.js already uses (move, step).
2. **The reward.** The optimum of the shipped reward escapes in 339 of the 340 runs where an escape exists, but later than the earliest escape in 24 of those 339 (7.1%, 4.8 to 10.3%) [Rates], since it minimises discounted length, not time. In one run on corpus-4 (an escape of 122 moves at fire speed 0.25) it does not escape at all: its value, -2.28, is that of walking to and fro over short links until the step limit, which beats escaping, -4.21 (the learned policy, which did escape; that run's row in the results file). An escape over m median-length links returns -10 + 20 x 0.99^(m-1), below zero beyond 70 moves; Lean proves the comparison for this case (`legacy_reward_prefers_staying`). Shaping cannot repair this: it preserves the optimum, and the optimum is wrong. The defect grows with the step limit, which the page allows up to 1,000.
3. **One start, one exit.** In 102 of 848 runs (12.0%, 10.0 to 14.4%) the chosen exit is cut off but another exit can still be reached [Rates]; a learner trained towards one exit cannot find those escapes.
4. **No guarantee.** Burning is a penalty, not a constraint, so nothing stops a learned policy from walking into the fire; 9 of 340 policies do.

### 4.6 Which dominates

For what the user sees, infeasibility dominates: 508 of the 559 shown failures have no escape. Among runs with an escape, training defects dominate: all 51 failures are training or drawing failures, against 0 caused by the reward's optimum, and setting epsilon to zero removes every one in the checked subset. Among model defects, the single exit matters most (102 runs), then the reward's late optima (24), then its non-escaping optimum (1). The node-only state of the original method would dominate everything (40.5%), but it is already gone.

## 5. Literature

Every DOI below was resolved at Crossref on 2026-11-05 (the record's title, authors, year and venue matched), except where marked: "no DOI" means the publisher assigned none and the official URL was checked instead; Jansen et al. (2020) has a DataCite DOI, which resolves at doi.org but not at Crossref. Text from papers and web pages was treated as data. No PDF is kept in the repository.

### 5.1 Reinforcement learning for evacuation and egress

Jabi et al. (2019) is the method Studio's RL follows (full text read; pages are the printed folios). Its environment-based state is "the index into the list of vertices" (p. 87), with no time index, while the fire is a precomputed list of room temperatures per time step (p. 89). It trains 3,000 episodes of at most 100 steps (pp. 89 to 90), states no numeric reward, discount, learning rate or exploration rate, and evaluates by average-reward curves only, noting that "no validation method has been pursued" (p. 92). Studio's rl.js already departs from it in three ways that matter here: the state is (node, step) when there is a fire, the reward is stated (+10, -10, -0.1 per median link, gamma 0.99), and the exact optimum of that reward is computed and shown. The Crossref record of the Blucher DOI gives pages 85 to 96; the printed paper and the second DOI (10.52842/conf.ecaade.2019.2.085) give 85 to 94.

Later work moved to deep networks and larger settings. Sharma et al. (2021) transfer a tabular Q-matrix learned on the fire-free graph into a deep Q-network trained under a spreading fire, in buildings of up to 91 rooms; this is the closest precedent for using fire-free knowledge as a prior, which the method below also does, but with a lower bound rather than a learned table. Xu et al. (2021) learn multi-exit evacuation with Rainbow DQN and exits that open on a schedule, without fire. Rosa et al. (2023) learn fire escape routes in school buildings with transfer learning. Fang et al. (2026) learn a path planner whose reward is the fractional effective dose of toxic gases and report about 62% less exposure than Dijkstra; whether that Dijkstra used time-dependent costs is not stated in the abstract, so the gain may come from hazard awareness rather than from learning. Multi-agent work covers pedestrian groups (Martinez-Gil et al. 2014), evacuation-guidance robots (Zhao et al. 2025) and congestion-aware wildfire routing with a fire mask (Raei et al. 2026); PyroRL (Guman et al. 2024) is an open wildfire evacuation environment. None of these gives a safety guarantee, and none compares with the exact optimum of its own model, which for a known deterministic fire is computable.

### 5.2 Time-dependent and time-expanded shortest paths

The time-expanded network goes back to dynamic flows (Ford and Fulkerson 1958) and was applied to building evacuation by Chalmet et al. (1982) and Choi et al. (1988); Skutella (2009) is the standard introduction to flows over time, and Lu et al. (2005) avoid the expanded network's size with capacity time series. For a single evacuee, Cooke and Halsey (1966) gave the first recursion with time-dependent transit times, Dreyfus (1969) the label-setting form, Orda and Rom (1990) the role of waiting, and Kaufman and Smith (1993) the FIFO condition under which the time-dependent fastest path costs no more than a static search. Fire-aware forecasts appear in Guan et al. (2023), a bi-objective dynamic Dijkstra for multi-exit building fires, and Shih and Tsai (2023), who plan the safest path on a predicted fire. Studio's forecast router (forecast.js) and its deadline map (deadline.js) are of this kind and are proved exact in TimeDependent.lean, Forecast.lean and Deadline.lean. The hop-clock fire of the RL step is a FIFO network of this kind too, which is why its earliest escape is one breadth-first search.

### 5.3 Potential-based shaping and its limits

Ng, Harada and Russell (1999) show that shaping rewards of the form gamma Phi(s') - Phi(s) leave the optimal policies unchanged, and that no other additive shaping does so for every MDP. Wiewiora (2003) shows that such shaping is the same as initialising the Q-table with Phi. Devlin and Kudenko (2012) extend the invariance to potentials that change during learning, Harutyunyan et al. (2015) turn arbitrary advice into a dynamic potential, and Eck et al. (2016) study shaping for finite-horizon planning. Grzes (2017) gives the condition that matters for egress: in episodic tasks the invariance holds only if Phi is zero wherever an episode can stop, including the step limit. Studio's rl.js was corrected on this point on 2026-10-28 (QLearning.lean). Two limits remain. Shaping changes the speed of learning, not what is learned: it cannot repair a reward whose optimum is wrong (section 4.5). And Wiewiora's equivalence cuts both ways: rl.js's prior, the one-step shaped value, amounts to a pessimistic initial table, because the +10 for escaping is far above it (section 4.4).

### 5.4 Safe reinforcement learning with shields

A shield (Bloem et al. 2015) is a runtime enforcer synthesised from a safety specification; Alshiekh et al. (2018) put one between a learner and its environment, so that only safe actions are taken, during training and after it, and give conditions under which learning still converges. Jansen et al. (2020) compute probabilistic shields by model checking, Carr et al. (2023) shield under partial observability, Huang and Ontañón (2022) analyse invalid-action masking, and García and Fernández (2015), Krasowski et al. (2023) and Gu et al. (2024) survey the field. The common weak point is the model the shield is computed from. Here it is the fire forecast itself, which the learner already uses for its rewards, so a shield costs nothing in assumptions. The shield must, however, look ahead: masking a move into a node that burns on arrival is not enough (section 6.2).

### 5.5 Value iteration on finite-horizon problems

With a known model and a finite horizon, backward induction gives the exact optimal non-stationary policy (Bellman 1957; Puterman 1994). The state must carry the time when the hazard does (Pardo et al. 2018), and Q-learning's convergence (Watkins and Dayan 1992) is an asymptotic statement that says nothing about a run of 400 episodes. Egress with absorbing exits is a stochastic shortest-path problem in the sense of Bertsekas and Tsitsiklis (1991) when the fire is uncertain, and a deterministic one here.

### 5.6 Curriculum learning

Curricula order training from easy to hard (Bengio et al. 2009; Narvekar et al. 2020; Soviany et al. 2022); reverse curricula start next to the goal and move outward (Florensa et al. 2017). For egress the backward pass of the learner below does the same work within one trial: values learned near the exit are carried back to the start at once.

### 5.7 Methods that use the model

Dyna (Sutton 1990) and prioritised sweeping (Moore and Atkeson 1993) add planning updates from a model to Q-learning and need far fewer real steps. Learning real-time search (LRTA*, Korf 1990, doi:10.1016/0004-3702(90)90054-4) and trial-based real-time dynamic programming (RTDP, Barto, Bradtke and Singh 1995, doi:10.1016/0004-3702(94)00011-O) learn values by greedy trials from an admissible initial estimate and converge to the optimum on the states that matter, for deterministic problems in finitely many trials. Deep Q-networks (Mnih et al. 2015), value iteration networks (Tamar et al. 2017) and graph neural network policies (Almasan et al. 2022) pay off when a policy must transfer across many buildings or the state is too large for a table; on one known building graph with a precomputed fire they add variance, not accuracy (Moerland et al. 2023 survey the trade-off).

## 6. The method: shielded learning on the time-expanded graph

The module is `topologicpy-web-frontend/src/core/rlSafe.js`; its tests are `src/core/rlSafe.test.mjs`. It keeps the problem of rl.js (the graph, the hop clock, the fire's hop front over the fire speed, the step limit) and changes four things.

### 6.1 The objective

The goal of an evacuee is lexicographic: escape if any escape exists, and then as early as possible. The new method states it directly. A move costs one step, an exit is worth nothing more, and there is no discount; a move into a node after it closes is not a cost but is excluded (section 6.2). The value of a state (node, step) is then the number of steps to the earliest safe exit, and finite-horizon value iteration on the time-expanded graph computes it exactly (`escapeValues`; Lean `vi_eq_earliest`). Ties between equally early routes go to the one with the shorter fire-free walk to an exit, so drawn routes stay short. This removes the reward defect of section 4.5 by construction.

### 6.2 The shield from the deadline map

`hopDeadlines` computes, for every node, the latest step D(v) at which an evacuee can stand there and still reach an exit safely by the step limit: D(x) = open(x) at an exit and D(v) = min(open(v), max over open links (v, w) of D(w) - 1) elsewhere, with open(v) the last step at which v may be entered. This is the max-min recursion of deadline.js on the hop clock; with whole-step labels it is set exits outward with one bucket per step, in O(nodes + links + H). The shield allows the move into w at step t only if t + 1 <= D(w).

Proved in Lean (`SafeEscape.lean`, section 8): any solution of the recursion is exactly the deadline (`deadline_iff`), so an allowed move is safe on arrival and keeps an escape (`shield_safe`, `shield_keeps`); a move is masked only if no escape goes through it, so the shield removes nothing a learner could need (`shield_complete`); some move is allowed until an exit is reached (`shield_nonempty`); and every walk inside the shield, however its moves are chosen, reaches an exit by the step limit without entering a closed node, whenever the start has an escape at all (`shield_reaches_exit`). Success therefore no longer depends on training: a random policy inside the shield escapes.

A shield that masks only moves into nodes untenable on arrival is weaker. A node can be tenable on arrival and sealed one step later; `arrival_mask_traps` is a three-node example in which the arrival mask lets the agent into a pocket with no allowed move while an escape exists, and the deadline shield masks the move into the pocket.

### 6.3 The learner

The learner is learning real-time search (LRTA*, Korf 1990; trial-based RTDP, Barto, Bradtke and Singh 1995) on the states (node, step), inside the shield:

1. Each state's value is a lower bound on its steps to safety. An unvisited state starts at the hop distance to the nearest exit through nodes the deadline map keeps (D >= 0) and to exits that stay usable (D >= 1). It is a lower bound because every safe escape stays on such nodes (`consistent_admissible`); it is sharper than the fire-free distance, because it ignores exits the fire has already sealed. By Wiewiora's equivalence this initial table is potential-based shaping with that potential (`potential_init_eq_shaping`), so it is shaping from the deadline map, in the form in which its effect on learning can be proved.
2. A trial walks from (start, 0), at each state moving to the allowed neighbour with the least 1 + V(w, t + 1) (ties to the shorter fire-free walk) and first raising V(v, t) to that least value. After the trial the same update is applied backward along its path, so that what was learned near its end reaches the start at once.
3. The exact value never falls with a later step (leaving later never escapes sooner; `val_mono_time`), so a bound learned at (v, t) also holds at every later step of v. The table keeps each node's bounds as a non-decreasing step function of the step (`StepTable`), which shares one update across all later steps (`raise_admissible`). This is the generalisation across time that a table indexed by (node, step) otherwise lacks, and it is sound for this problem class only because the fire never recedes.
4. Learning stops on a certificate: a trial that takes no more steps than V(start, 0) had when it began is an earliest escape, because V never exceeds the exact value (`lrta_admissible`, `certificate_optimal`). A trial that changes no value passes the test (`trial_no_change`), and values are whole numbers below a cap, so only finitely many trials can fail it (`bounded_raises`). If the trial budget runs out first, the result says the route is not certified, and how many steps it takes against the earliest escape.

The values are indexed by the step, not by the node alone, because a node can need different moves at different steps; `node_policy_fails` is a seven-node example in which no policy on the node alone escapes earliest from one start and escapes at all from another, while a table on (node, step) does both. One table serves every start: the bounds are about states, not about the start, so learning from several starts shares its work (tested), and with all the building's exits as targets the result is a policy towards any exit.

### 6.4 What is new, and how strong the claim is

Each ingredient is known: time-expanded states (Pardo et al. 2018; already in rl.js), shields (Alshiekh et al. 2018), potential-based shaping and its equivalence to initialisation (Ng et al. 1999; Wiewiora 2003), and LRTA* and RTDP (Korf 1990; Barto et al. 1995). What we believe is new, within the literature searched in section 5, is the combination for fire egress and three specific points:

1. a shield computed as the exact deadline map of a known spreading fire, proved to be both sound and complete for escape (it never admits a doomed move and never masks a viable one), so that success is guaranteed by construction whenever an escape exists, during training as well as after it;
2. sharing learned bounds across later steps, which is sound because the hazard is monotone in time and which removes much of the cost of indexing the table by the step;
3. an optimality certificate that the learner checks itself, so a result is either proved earliest or reported as not certified, with machine-checked proofs (Lean 4, Mathlib) of every statement above.

Relative to Jabi et al. (2019), the method changes the state (node and step instead of node), the objective (earliest safe escape instead of an unspecified reward), the safety (a proved shield instead of an episode-ending penalty), the evaluation (against the exact optimum, with a certificate) and the scope (any exit, a table shared by every start).

The claim is moderate. For a known deterministic fire the exact planner (`earliestSafeEscape`, one breadth-first search) is faster than any learner and needs no training; the learner's value is that it is a learned method whose every answer is safe and whose optimality is checked, which is what the Simulate step's RL card sets out to show. Nothing here is claimed for an uncertain fire, for crowds, or for a policy that transfers between buildings.

## 7. Old against new

On the 340 runs with an escape to the chosen exit, same exit, same fire, same step limit [Old against new]:

| Method | Escapes | Earliest escape | Median extra moves when escaping | Median run time |
|---|---|---|---|---|
| shipped Q-learning, 400 episodes | 85.0% | 30.6% | 2 | 30.4 ms |
| arrival mask and learning real-time search | 100.0% | 100.0% | 0 | not timed |
| deadline shield and learning real-time search (new) | 100.0% | 100.0% | 0 | 1.8 ms |
| exact earliest escape, breadth first | 100.0% | 100.0% | 0 | 0.105 ms |

The shipped learner's escapes are the earliest in 104 of its 289 successes; its objective is discounted length, not time, and its loops add moves. The new method's routes are a median 1.2% longer than the fire-free shortest route, against 5.0% for the breadth-first route, which takes the first earliest route it finds [Objectives and lengths].

The learner usually needs no learning at all: with one exit, 335 of 340 runs are certified on the first trial and stored no value (median 0 values, largest 356, against a median of 2,108.5 table entries for the shipped learner); the slowest needed 391 trials [To any exit]. With every exit as a target, the potential can point at an exit that the fire seals before this start can reach it, and learning takes longer: 81.9% of the 442 runs are certified within 400 trials and all 442 within 20,000 (median 1 trial, 90th percentile 1,150, largest 12,501; median 2.6 ms, largest 2,639.7 ms). Sharing bounds across later steps halves the mean number of trials (467.6 against 1,058.9) and certifies 100.0% instead of 98.4% within 20,000 [To any exit]. The backward pass matters little with one exit (certified 99.4% without it within 400 trials). Masking only nodes untenable on arrival trapped the first trial in 5 of 340 runs (1.5%); the deadline shield never did, as proved.

Run time per model, median over its runs [Run time per model]:

| Model | Nodes | Shipped, 400 episodes with its optimum | New, learning | New, whole request | Exact escape | Deadline map | Escape field, every start |
|---|---|---|---|---|---|---|---|
| corpus-1 | 1,092 | 11.9 | 0.5 | 1 | 0.068 | 0.15 | 6.6 |
| corpus-2 | 5,106 | 23 | 2.04 | 3.3 | 0.096 | 0.5 | 28.3 |
| corpus-3 | 4,384 | 33.8 | 1.83 | 3.4 | 0.196 | 0.47 | 15.3 |
| corpus-4 | 12,461 | 83.8 | 4.88 | 11 | 0.213 | 1.21 | 22.2 |
| corpus-5 | 17,470 | 141 | 6.72 | 16.8 | 0.385 | 2.62 | 48 |
| corpus-6 | 33,654 | 168.9 | 19.47 | 27.4 | 0.269 | 9.43 | 139.6 |
| duplex-apartment | 1,540 | 11.4 | 0.61 | 1.1 | 0.054 | 0.22 | 7.3 |
| office-two-storey | 3,374 | 26.4 | 1 | 2.2 | 0.113 | 0.43 | 12.7 |
| courtyard-ring | 650 | 11 | 0.25 | 0.5 | 0.05 | 0.13 | 1.3 |

All times in milliseconds.

The new learner is 9 to 44 times faster than the shipped one on these models, and the whole request (deadline map, learning and the exact check) stays under 30 ms at the median on the largest. The exact escape is faster still, under 0.4 ms: for a known fire, planning is the cheaper route to the same answer, and the learner's role is to be a learned method whose every answer is safe and whose optimality it proves. The escape field for every start at once, by backward induction, costs at most 140 ms at the median, on the 34,000-node model.

## 8. What is proved in Lean

`formal/fireegress/FireEgress/SafeEscape.lean`, 33 results, built with the rest of the project on 2026-11-05 (28 modules, 1,297 theorems and lemmas, 1,284 axiom prints, every one on `propext`, `Classical.choice` and `Quot.sound` only, no `sorry`, no warnings; `formal/fireegress/axioms-all-2026-11-05.txt`). The model is the code's hop clock: a move at step t into w is safe when t + 1 <= open(w), exits are terminal, H is the step limit.

| Claim | Theorems |
|---|---|
| Finite-horizon value iteration on the time-expanded graph returns the earliest safe escape, and is infinite exactly when none is left | `bi_spec`, `vi_le`, `vi_esc`, `vi_top_iff`, `vi_eq_earliest` |
| Leaving later never escapes sooner | `esc_mono`, `val_mono_time` |
| The max-min recursion has one solution, the deadline: t <= D(v) iff v can be entered at t with an escape left | `deadline_iff`, `deadline_le_H` |
| The shield never admits a move that is unsafe on arrival or that loses the escape, never masks a move an escape uses, always leaves a move, and every shielded walk reaches an exit by H when the start has an escape | `shield_iff`, `shield_safe`, `shield_keeps`, `shield_complete`, `shield_nonempty`, `shield_reaches_exit` |
| Learned values never exceed the exact ones, also when a bound is shared with every later step | `consistent_admissible`, `lrta_admissible`, `raise_admissible` |
| The certificate is sound, a trial that changes nothing passes it, and only finitely many trials can fail it | `walk_esc`, `certificate_optimal`, `trial_no_change`, `bounded_raises` |
| Initialising with a potential is potential-based shaping of the costs | `potential_init_eq_shaping` |
| Counterexamples: no node-only policy is right from every start; the arrival mask can trap; the shipped reward can prefer staying inside | `node_policy_fails`, `arrival_mask_traps`, `G3_no_escape_from_p`, `legacy_reward_prefers_staying` |
| Unfolding lemmas | `esc_zero`, `esc_time`, `esc_inv`, `bi_exit`, `bi_zero`, `bi_succ` |

Potential-based shaping leaving the optimal policy unchanged on a time-expanded problem with zero terminal potential is `QLearning.shaped_dag_eq` (proved on 2026-10-28) with states (node, step); under the shield every terminal is an exit, where the potential is zero. Convergence of tabular Q-learning on such problems is `QLearning.qlearning_tendsto_dag`; it is asymptotic, which is why the new method uses a learner with a finite certificate instead.

Not proved: the bucket algorithm of `hopDeadlines` and the loops of the learner as programs (the tests check their outputs against the proved statements on 40 to 300 random cases per statement), the metre tie-breaking, floating point, and any number of trials for a given building.

## 9. Recommendations

1. Use the shielded learner as the Simulate step's learned escape, towards every exit of the building by default, and report moves against the earliest escape and whether the result is certified, instead of the discounted return.
2. Present "no escape exists" as the outcome of the scenario, not as a failure of the agent, since it is 90.9% of what users see as failure; say near the fire-speed field that at one link per move a fire on or beside the route leaves no escape in almost every case.
3. If the Q-learning card stays, as the method of Jabi et al. (2019) for comparison, draw the policy walk rather than the no-revisit walk, or lower epsilon to 0.05: either removes most of its failures in this study (drawing failures 42 of 340; epsilon 0.05 leaves 1 of 74 checked scenarios).
4. Longer step limits allow longer escapes, and for those the shipped reward's optimum need not escape at all (section 4.5); under the new objective this cannot happen.

## 10. Limitations and open risks

1. The fire is the hop front at a constant speed in links per move, as in rl.js; the shield is as safe as that forecast. Under an uncertain fire the shield should come from a conservative forecast (Surrogate.lean gives one for the hop surrogate), and nothing here covers stochastic spread.
2. One seed per run. The rates come with Wilson intervals over runs, not over seeds; the shipped learner's outcome on a single run depends on its seed.
3. Nine buildings, four starts each; the knob study used the 74 speed-1 runs with an escape. The no-trend result for closeness rests on 72 runs within five links.
4. Timings were taken while another job used the machine.
5. The certificate needs the trial budget: with every exit, 18.1% of runs needed more than 400 trials and one needed 12,501 (2.6 s). The default budget is 20,000 trials; the result says when it is not certified.
6. The claims of novelty (section 6.4) rest on the searches of section 5, not on an exhaustive review.

## 11. References

Every DOI was resolved at Crossref on 2026-11-05, except the one marked "not in Crossref" (a DataCite DOI, resolved at doi.org). Author lists are as Crossref gives them, the first three and "et al."

- Almasan, Suárez-Varela, Rusek et al. (2022). Deep reinforcement learning meets graph neural networks: Exploring a routing optimization use case. Computer Communications, 196, 184-194. doi:10.1016/j.comcom.2022.09.029
- Alshiekh, Bloem, Ehlers et al. (2018). Safe Reinforcement Learning via Shielding. Proceedings of the AAAI Conference on Artificial Intelligence, 32(1). doi:10.1609/aaai.v32i1.11797
- Barto, Bradtke, Singh (1995). Learning to act using real-time dynamic programming. Artificial Intelligence, 72(1-2), 81-138. doi:10.1016/0004-3702(94)00011-O
- Bellman (1957). Dynamic Programming. Princeton University Press. doi:10.1515/9781400835386 (the 2010 reprint; the 1957 edition has none)
- Bengio, Louradour, Collobert et al. (2009). Curriculum learning. Proceedings of the 26th Annual International Conference on Machine Learning (ICML 2009), 41-48. doi:10.1145/1553374.1553380
- Bertsekas, Tsitsiklis (1991). An Analysis of Stochastic Shortest Path Problems. Mathematics of Operations Research, 16(3), 580-595. doi:10.1287/moor.16.3.580
- Bloem, Könighofer, Könighofer et al. (2015). Shield Synthesis: Runtime Enforcement for Reactive Systems. Tools and Algorithms for the Construction and Analysis of Systems (TACAS 2015), Lecture Notes in Computer Science, 533-548. doi:10.1007/978-3-662-46681-0_51
- Carr, Jansen, Junges et al. (2023). Safe Reinforcement Learning via Shielding under Partial Observability. Proceedings of the AAAI Conference on Artificial Intelligence, 37(12), 14748-14756. doi:10.1609/aaai.v37i12.26723
- Chalmet, Francis, Saunders (1982). Network Models for Building Evacuation. Management Science, 28(1), 86-105. doi:10.1287/mnsc.28.1.86
- Choi, Hamacher, Tufekci (1988). Modeling of building evacuation problems by network flows with side constraints. European Journal of Operational Research, 35(1), 98-110. doi:10.1016/0377-2217(88)90382-7
- Cooke, Halsey (1966). The shortest route through a network with time-dependent internodal transit times. Journal of Mathematical Analysis and Applications, 14(3), 493-498. doi:10.1016/0022-247X(66)90009-6
- Devlin, Kudenko (2012). Dynamic potential-based reward shaping. Proceedings of the 11th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2012), IFAAMAS, 433-440. No DOI; https://eprints.whiterose.ac.uk/75121/
- Dreyfus (1969). An Appraisal of Some Shortest-Path Algorithms. Operations Research, 17(3), 395-412. doi:10.1287/opre.17.3.395
- Eck, Soh, Devlin et al. (2016). Potential-based reward shaping for finite horizon online POMDP planning. Autonomous Agents and Multi-Agent Systems, 30(3), 403-445. doi:10.1007/s10458-015-9292-6
- Fang, Tam, Lovreglio et al. (2026). Development of a tenability-based path planning model for building fire evacuations using reinforcement learning. Journal of Building Engineering, 118, 115132. doi:10.1016/j.jobe.2025.115132
- Florensa, Held, Wulfmeier et al. (2017). Reverse Curriculum Generation for Reinforcement Learning. Proceedings of the 1st Annual Conference on Robot Learning (CoRL 2017), PMLR, 78, 482-495. No DOI; https://proceedings.mlr.press/v78/florensa17a.html
- Ford, Fulkerson (1958). Constructing Maximal Dynamic Flows from Static Flows. Operations Research, 6(3), 419-433. doi:10.1287/opre.6.3.419
- García, Fernández (2015). A Comprehensive Survey on Safe Reinforcement Learning. Journal of Machine Learning Research, 16(42), 1437-1480. No DOI; https://jmlr.org/papers/v16/garcia15a.html
- Grzes (2017). Reward Shaping in Episodic Reinforcement Learning. Proceedings of the 16th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2017), IFAAMAS, 565-573. No DOI; https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf
- Gu, Yang, Du et al. (2024). A Review of Safe Reinforcement Learning: Methods, Theories, and Applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12), 11216-11235. doi:10.1109/TPAMI.2024.3457538
- Guan, Hou, Yu et al. (2023). Dynamic Evacuation Path Planning for Multi-Exit Building Fire: Bi-Objective Model and Algorithm. Fire Technology, 59(5), 2853-2876. doi:10.1007/s10694-023-01448-x
- Guman, O’Brien, Pondoc et al. (2024). PyroRL: A Reinforcement Learning Environment for Wildfire Evacuation. Journal of Open Source Software, 9(101), 6739. doi:10.21105/joss.06739
- Harutyunyan, Devlin, Vrancx et al. (2015). Expressing Arbitrary Reward Functions as Potential-Based Advice. Proceedings of the AAAI Conference on Artificial Intelligence, 29(1). doi:10.1609/aaai.v29i1.9628
- Huang, Ontañón (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. The International FLAIRS Conference Proceedings, 35. doi:10.32473/flairs.v35i.130584
- Jabi, Chatzivasileiadi, Wardhana, Lannon, Aish (2019). The synergy of non-manifold topology and reinforcement learning for fire egress. Proceedings of eCAADe 37 / SIGraDi 23, vol. 2, Porto, 85-94 (printed folios; Crossref gives 85-96 for this DOI). doi:10.5151/proceedings-ecaadesigradi2019_671; also doi:10.52842/conf.ecaade.2019.2.085 (85-94).
- Jansen, Könighofer, Junges et al. (2020). Safe Reinforcement Learning Using Probabilistic Shields (Invited Paper). 31st International Conference on Concurrency Theory (CONCUR 2020), LIPIcs, 171, 3:1-3:16. doi:10.4230/LIPIcs.CONCUR.2020.3 (not in Crossref)
- Kaufman, Smith (1993). Fastest paths in time-dependent networks for intelligent vehicle-highway systems application. IVHS Journal, 1(1), 1-11. doi:10.1080/10248079308903779
- Korf (1990). Real-time heuristic search. Artificial Intelligence, 42(2-3), 189-211. doi:10.1016/0004-3702(90)90054-4
- Krasowski, Thumm, Müller et al. (2023). Provably Safe Reinforcement Learning: Conceptual Analysis, Survey, and Benchmarking. Transactions on Machine Learning Research (TMLR). No DOI; https://openreview.net/forum?id=mcN0ezbnzO
- Lu, George, Shekhar (2005). Capacity Constrained Routing Algorithms for Evacuation Planning: A Summary of Results. Advances in Spatial and Temporal Databases (SSTD 2005), Lecture Notes in Computer Science, 291-307. doi:10.1007/11535331_17
- Martinez-Gil, Lozano, Fernández (2014). MARL-Ped: A multi-agent reinforcement learning based framework to simulate pedestrian groups. Simulation Modelling Practice and Theory, 47, 259-275. doi:10.1016/j.simpat.2014.06.005
- Mnih, Kavukcuoglu, Silver et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540), 529-533. doi:10.1038/nature14236
- Moerland, Broekens, Plaat et al. (2023). Model-based Reinforcement Learning: A Survey. Foundations and Trends in Machine Learning, 16(1), 1-118. doi:10.1561/2200000086
- Moore, Atkeson (1993). Prioritized sweeping: Reinforcement learning with less data and less time. Machine Learning, 13(1), 103-130. doi:10.1007/BF00993104
- Narvekar, Peng, Leonetti et al. (2020). Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey. Journal of Machine Learning Research, 21(181), 1-50. No DOI; https://jmlr.org/papers/v21/20-212.html
- Ng, Harada, Russell (1999). Policy invariance under reward transformations: Theory and application to reward shaping. Proceedings of the 16th International Conference on Machine Learning (ICML 1999), 278-287. No DOI; https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
- Orda, Rom (1990). Shortest-path and minimum-delay algorithms in networks with time-dependent edge-length. Journal of the ACM, 37(3), 607-625. doi:10.1145/79147.214078
- Pardo, Tavakoli, Levdik et al. (2018). Time Limits in Reinforcement Learning. Proceedings of the 35th International Conference on Machine Learning, PMLR, 80, 4045-4054. No DOI; https://proceedings.mlr.press/v80/pardo18a.html
- Puterman (1994). Markov Decision Processes. Wiley (Wiley Series in Probability and Statistics). doi:10.1002/9780470316887
- Raei, Safarzadeh, Wang (2026). Congestion-aware Multi-agent Reinforcement Learning for Wildfire Evacuation Routing. The International Archives of the Photogrammetry, Remote Sensing and Spatial Information Sciences, XLIX-M-1-2026, 41-50. doi:10.5194/isprs-archives-xlix-m-1-2026-41-2026
- Rosa, Falqueiro, Bonacin et al. (2023). EvacuAI: An Analysis of Escape Routes in Indoor Environments with the Aid of Reinforcement Learning. Sensors, 23(21), 8892. doi:10.3390/s23218892
- Sharma, Andersen, Granmo et al. (2021). Deep Q-Learning With Q-Matrix Transfer Learning for Novel Fire Evacuation Environment. IEEE Transactions on Systems, Man, and Cybernetics: Systems, 51(12), 7363-7381. doi:10.1109/TSMC.2020.2967936
- Shih, Tsai (2023). Safest-path planning approach for indoor fire evacuation. International Journal of Disaster Risk Reduction, 93, 103760. doi:10.1016/j.ijdrr.2023.103760
- Skutella (2009). An Introduction to Network Flows over Time. Research Trends in Combinatorial Optimization (Springer), 451-482. doi:10.1007/978-3-540-76796-1_21
- Soviany, Ionescu, Rota et al. (2022). Curriculum Learning: A Survey. International Journal of Computer Vision, 130(6), 1526-1565. doi:10.1007/s11263-022-01611-x
- Sutton (1990). Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming. Machine Learning Proceedings 1990 (Morgan Kaufmann / Elsevier), 216-224. doi:10.1016/B978-1-55860-141-3.50030-4
- Tamar, Wu, Thomas et al. (2017). Value Iteration Networks. Proceedings of the 26th International Joint Conference on Artificial Intelligence (IJCAI 2017); abridged from NIPS 2016 (Advances in Neural Information Processing Systems 29), 4949-4953. doi:10.24963/ijcai.2017/700
- Watkins, Dayan (1992). Q-learning. Machine Learning, 8(3-4), 279-292. doi:10.1007/BF00992698
- Wiewiora (2003). Potential-Based Shaping and Q-Value Initialization are Equivalent. Journal of Artificial Intelligence Research, 19, 205-208. doi:10.1613/jair.1190
- Xu, Huang, Mango et al. (2021). Simulating multi‐exit evacuation using deep reinforcement learning. Transactions in GIS, 25(3), 1542-1564. doi:10.1111/tgis.12738
- Zhao, Liang, Ma et al. (2025). Adversarial Reinforcement Learning for Enhanced Decision-Making of Evacuation Guidance Robots in Intelligent Fire Scenarios. IEEE Transactions on Computational Social Systems, 12(5), 2030-2046. doi:10.1109/TCSS.2024.3502420
