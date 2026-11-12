# The shortest walks and the slime mould's desire lines: why they differ (2026-11-12)

The user saw that the egress map's lines (the walk from every room to its nearest exit) and the slime mould's desire lines are not the same, and asked why. This note measures where they part, then switches the slime's differences off one at a time to find what causes each. Tool: `tools/egress-vs-slime.mjs`; comparison: `topologicpy-web-frontend/src/core/egressCompare.js`; results: `benchmark/results/2026-11-12-egress-vs-slime.json` (3 seeds a variant, no fire).

## 1. What each one answers

- **The egress map** (`core/egressMap.js`) asks: from each room's farthest place, what is the shortest walk to the nearest exit? It runs one multi-source Dijkstra from all exits. Each room counts once, alone; nobody gets in anyone's way. Shortest walks pull tight round corners.
- **The slime mould** (`core/slimeEgress.js`, Jones's Physarum agents) asks: if people kept leaving from everywhere on the floor, where would the escape traffic run?
  - 2,000 agents start anywhere on the floor, and one that leaves starts again somewhere else, so traffic is weighted by floor area.
  - Each agent senses 3 links ahead and 60° to either side, and turns 60° at a time. It follows the walk to the exits plus the trail the others left (a full tube is worth 3 m of walk).
  - Only one agent fits on a node.
  - The desire lines are the trail averaged over the run's second half.

So one is a route per room, and the other is a traffic map.

## 2. Where they part

Within 1 m:

| Building | Slime trail on a shortest walk | Shortest walks under a desire line (weight ≥ 0.25) | Exit use apart (0 to 1) | At a wall: slime trail / shortest walks |
| --- | --- | --- | --- | --- |
| Duplex (20 rooms, 6 exits) | 0.70 | 0.16 | 0.64 | 0.20 / 0.42 |
| NIBS Office (98 rooms, 8 exits) | 0.54 | 0.35 | 0.86 | 0.15 / 0.24 |
| NIBS Clinic (260 rooms, 19 exits) | 0.67 | 0.33 | 0.73 | 0.24 / 0.31 |

- **Most of the slime's traffic runs on shortest walks.** 54 % to 70 % of its trail lies within 1 m of one.
- **The slime draws only the busy walks.** Only 16 % to 35 % of the shortest walks carry a desire line. The others are walks from rooms few people leave from: on a traffic map they stay faint.
- **They leave by different exits.** "Exit use apart" is the total variation between two splits: each exit's share of the floor it is nearest for, and its share of the slime's traffic. At 0.64 to 0.86, much of the slime leaves by exits that are not the nearest for where it started.
- **The slime keeps off walls.** 24 % to 42 % of the shortest walks' nodes are next to a wall, against 15 % to 24 % of the slime's trail.

## 3. What causes each difference

The slime again, with one difference switched off at a time (means of 3 seeds):

| Variant | Duplex: on shortest, exits apart, at walls | Office | Clinic |
| --- | --- | --- | --- |
| As the app runs it | 0.70, 0.64, 0.20 | 0.54, 0.86, 0.15 | 0.67, 0.73, 0.24 |
| No trail in what they follow | 0.70, 0.64, 0.20 | 0.54, 0.86, 0.14 | 0.65, 0.73, 0.24 |
| Sensors one link out | 0.75, 0.42, 0.23 | 0.60, 0.48, 0.16 | 0.71, 0.46, 0.25 |
| A tenth of the agents | 0.75, 0.64, 0.39 | 0.53, 0.87, 0.24 | 0.78, 0.73, 0.32 |
| All three off | 0.72, 0.42, 0.43 | 0.62, 0.48, 0.22 | 0.74, 0.49, 0.34 |

- **The exits: the sensors' reach.**
  - Sensing 3 links ahead and 60° to either side, an agent reads the walk field coarsely. Near a junction or a door it often turns towards the wrong exit, and a door narrower than its sensor fan reads as a wall.
  - With sensors one link out, the exit split moves 35 % to 45 % of the way towards the nearest-exit split in all three buildings.
- **The walls: crowding.**
  - With one agent a node, the lane along a wall and round a corner, where the shortest walks run, fills up, and the others pass beside it.
  - With a tenth of the agents, the slime's share at walls matches the shortest walks' almost exactly (0.39 against 0.42, 0.24 against 0.24, 0.32 against 0.31).
- **The trail, the "desire" itself: almost nothing here.** At its default weight (a full tube worth 3 m of walk), following the trail moves no reading by more than 0.02. On these floors the walk field dominates. A stronger trail would merge routes into trunks, as desire lines in grass do.
- **What remains with all three off is the agents' own movement.** The exit split is still 0.42 to 0.49 apart: they turn 60° at a time and take a random turn 2 % of the time, so they do not run down the gradient exactly.

## 4. How to read the two together

- The egress map is the code check's question: every room's own worst walk.
- The slime is a picture of where escape traffic concentrates, given crowding and imperfect wayfinding. That makes it a better guide to where corridors and doors carry the most people than to which exit is nearest.
- Where the two agree, both the geometry and the traffic point the same way.
- Where the slime leaves by another exit than the nearest, look at that junction or door: it is where coarse wayfinding and crowding turn people.

Neither is calibrated against people. Crowd validation (`wiki/research/crowd-validation-plan-2026-11-11.md`) covers the crowd model, not the slime.
