# Plan: desire lines of fire egress (2026-11-11)

The user: "The whole idea was how the slime mould algorithm defines desire lines in egress and can make sense of the chaos"; "a medial axis or skeletal algorithm of the slime mould spread, to make sense of what direction the slime mould is suggesting for egress from floor spaces to the exits, the medial axis of the rooms to the exits, in contrast with the yellow slime mould"; "fire and smoke as a retardant for the slime mould is something we have to model to get the fire egress desire lines"; "I had envisioned the slime mould as informing the people leaving, in all exits"; and to take cues from desire lines in parks and forest trails, mosh pits, and robotics (SLAM and path planning), "which are abstractions of the math behind some of the algorithms in nature and people movement".

This plan builds on the slime mould (src/core/slimeEgress.js, released d32f4d6) and the plan for randomness in people's motion ([crowd-motion-randomness-plan-2026-11-11.md](crowd-motion-randomness-plan-2026-11-11.md)).

## The idea in one paragraph

The slime mould's trail is a field over the floor: where escape traffic concentrates. Its **desire lines** are that field's ridges, the lines the tubes run along, each with a direction (towards the exit the tube leads to). The floor's **medial axis** is the ridge of a different field, the clearance from walls: the line through each room and corridor furthest from both sides, the path a robot planner takes to keep clear of obstacles. Drawn together, the two say where the building asks people to go (the medial axis, geometry alone) and where the slime mould, drawn by the exits and kept from the fire, says they will go (the desire lines). Fire and smoke act on the slime as a retardant, so the desire lines move as the fire grows; and the trail, laid by the slime, becomes a field the people leaving follow, as walkers follow a worn path across a park.

## Sources (DOIs checked on Crossref; "read" means the model's equations were read)

| Source | What it gives | Read |
| --- | --- | --- |
| Helbing, Keltsch and Molnár 1997, *Nature* 388:47-50, doi:10.1038/40353 (arXiv cond-mat/9805158) | Human trail systems: the ground's comfort `dG/dt = (G0 - G)/T + I (1 - G/Gmax) sum delta(r - r_a)`, the trail potential `V = integral exp(-abs(r - r_a)/sigma) G`, direction `e = (d - r + grad V) / norm`; one parameter `kappa = I T / sigma^2`: large gives a minimal way system, small direct paths; real ones a compromise of directness and comfort | yes |
| Helbing, Schweitzer and Keltsch 1997, *Phys. Rev. E* 56:2527, doi:10.1103/PhysRevE.56.2527 | The active walker model for human and animal trails | DOI only |
| Silverberg, Bierbaum, Sethna and Cohen 2013, *Phys. Rev. Lett.* 110:228701, doi:10.1103/PhysRevLett.110.228701 (arXiv 1302.1886) | Mosh and circle pits: soft-body repulsion, self-propulsion `mu (v0 - v)`, flocking `alpha` (alignment within 4 radii), Gaussian noise; noise dominant gives the gas-like mosh pit, flocking dominant the vortex circle pit | yes |
| Choset and Burdick 2000, *Int. J. Robotics Research* 19:96-125, doi:10.1177/02783640022066770 | The generalized Voronoi graph: the points equidistant from two or more obstacles, a roadmap of maximum clearance (the medial axis of the free space) | DOI only |
| Khatib 1986, *Int. J. Robotics Research* 5:90-98, doi:10.1177/027836498600500106 | Artificial potential fields: attraction to the goal, repulsion from obstacles; the floor field in robotics | DOI only |
| Tero et al. 2010, *Science* 327:439, doi:10.1126/science.1177894; Bonifaci et al. 2012, doi:10.1016/j.jtbi.2012.06.017 | Slime mould networks as transport design; convergence to shortest paths | DOI only (earlier) |
| Fridolf, Ronchi and Nilsson 2013, *Fire Safety J.* 59:8-21, doi:10.1016/j.firesaf.2013.03.007 | Walking speed and exit choice in smoke: the retardant on people | DOI only |
| Kirchner and Schadschneider 2002, doi:10.1016/S0378-4371(02)00857-9 | The dynamic floor field, herding `k_D` | yes (earlier) |

SLAM (simultaneous localisation and mapping) is how a robot builds the map these planners run on; Studio's map is the IFC, so SLAM's part here is the planning half (Voronoi roadmaps, potential fields) and, later, scanned buildings with no model.

## Stages

### D1. Skeletons: the medial axis and the slime's desire lines (built; see below)

- **Clearance.** Each walkable node's walk to the nearest boundary (a node missing a lattice neighbour: a wall, a doorway's jamb, the floor's edge), by a multi-source Dijkstra from the boundary (the robotics "brushfire").
- **Medial axis.** The nodes on clearance's ridge: a node whose clearance is no less than both neighbours along some lattice direction and more than at least one, with clearance at least 0.5 m (thin corridors and doorways keep their centre line). Within a doorway the axis passes through the door. Drawn in cyan.
- **The medial way out.** From each room (IfcSpace, or each part of the floor) the cheapest walk to an exit where a link costs its length times `1 + beta / clearance`: the robot planner's maximum-clearance route, which keeps to the medial axis.
- **Desire lines.** The ridge of the slime's averaged trail (the same ridge test on the weights, above a share of the 99th percentile), each segment pointing down the walk to its exit. Drawn in a contrasting colour (magenta) over the yellow weights, with the direction shown.
- **What it says.** Where the desire lines leave the medial axis, the slime (drawn by the exits, kept from the fire) and the building's geometry disagree: a corner cut, a room crossed, a wide hall used along one side.

### Built so far (2026-11-11, not released)

D1 and the first half of D2 are built (src/core/skeleton.js, src/core/slimeEgress.js, workers/slime.worker.js; Route step, "Slime mould": "Desire lines" and "Medial axis of the floor", both off until asked; "The fire": as if none, held back (soft), walled off (hard)). Two definitions changed from those above once measured:

- **The medial axis is the lambda-medial axis** (Chazal and Lieutier 2005, *Graphical Models* 67:304-331, doi:10.1016/j.gmod.2005.01.002, DOI checked): the brushfire carries each node's nearest boundary node, and a node is on the axis where its and a neighbour's nearest boundary nodes lie at least 1 m apart (the clearer of the two). The ridge test above marked plateaus too: 12,231 links on the NIBS Office and 21,317 on the Clinic, so every line was "near the axis". Now 7,176 and 10,677: one line through each room and corridor, with the rectangle's branches to its corners.
- **Desire lines are the slime's ways out from the whole floor**, not the ridge of its trail (which broke into fragments): one walk from the exits where a link costs its length times 1 + 2 (1 - weight) (times 1 + 20 x smoke with the soft fire), and the way out traced from one node in each 2 m square of each floor; a link's flow is the ways through it, and the ways bundle into trunks along the tubes, as Helbing et al.'s trails do. Drawn dark over the yellow, thicker with flow, with arrows to the exits; links fewer than two ways pass are left out.
- **The soft fire** (the user: "a softer version of the slime mould that avoids the areas that are untenable; it never functions if there's a fire"): the hard version walled off every place the fire made untenable during its whole run, and a fire that reaches the whole building left nothing (with a stand-in fire spreading 0.5 m/s, all 17,470 NIBS Office and 25,407 Clinic places walled, no agent placed). The soft one keeps the fire's clock (a step 0.42 s), its smoke rising over the 30 s before a place is untenable: the way out round it costs 1 + 20 x smoke a link, the place repels, and its trail fades by a further half at full smoke; it is never walled off. With the same fire: 2,000 agents grown, routes found, 39 % (Office) and 74 % (Clinic) of the trail laid in smoke (the fire covers most of both within the run).
- Measured, no fire: 71 % (NIBS Office) and 80 % (Clinic) of the desire lines lie within half a metre of the medial axis; the slime's run 0.6 to 0.9 s, the skeletons 20 to 40 ms. Without a fire the slime's run is bit for bit the released one (a check on all 17,470 NIBS Office weights).
- **Gaps 1 to 4 below (built, not released).** "Bottlenecks", "Dead ends" and "Branches it gave up", each off until asked, with a line of readings under the notice. With a Flamescapes run the soft fire's smoke is that run's visibility at head height, frame by frame (clear at 30 m, untenable at 10 m), and each agent's sensors reach less far in it; graph fires keep the 30 s ramp. The desire lines now carry people (one per 9 m^2 of floor, as the performance check) rather than floor samples. A passage's width is the IFC door's (declared, else measured) or twice the largest clearance within two links; a dead end's depth is the Gromov product `(dA + dB - dAB) / 2` over the walks to the two nearest exits. Measured on the slime's default run, no fire:

| Model | People | Worst bottleneck | Longest one way out | Floor beyond 6.1 m / 15.2 m | Given up |
| --- | --- | --- | --- | --- | --- |
| NIBS Office | 481 | 202 through a 0.92 m door, 250 s | 41.4 m | 3,494 / 1,780 m^2 | 163 m^2 |
| NIBS Clinic | 697 | 264 through a 0.92 m door, 326 s | 103.5 m | 4,772 / 3,489 m^2 | 146 m^2 |
| Office, two storeys | 92 | 51 through a 0.9 m door, 65 s | 23.0 m | 435 / 373 m^2 | 11 m^2 |

  The readings take 9 to 61 ms. The dead-end areas are large because the walk before a second way out is measured from every floor node, rooms included, and from parts of the graph that reach one exit only. IBC 1020.5's limit is for corridors; a room's single way out is held instead to the common path of egress travel (IBC 2021 Table 1006.2.1; for Group B, 100 ft or 30.5 m sprinklered, and 75 ft or 22.9 m unsprinklered above 30 occupants, from secondary sources, to check against the code's text). Next: corridors told from rooms by the IfcSpace names, rooms read against the common-path limit for the performance check's occupancy, and 1020.5's exception for dead ends shorter than 2.5 times their width.

### D2. Fire and smoke as a retardant (the soft fire and Flamescapes smoke built)

- The slime run keeps the fire's clock: a step is the time an agent takes to walk a link (0.5 m at 1.19 m/s, about 0.42 s), so 400 steps are about 170 s of fire.
- A place is a wall from its untenable time on (as now, but in time, not all at once).
- Before it, smoke retards: the trail there fades faster and the agents sense it as repelling, rising over the `lead` seconds before the place becomes untenable (a proxy for every fire model), or, with a Flamescapes run, from its smoke: the visibility and the extinction coefficient at head height in each frame, as Fridolf et al. relate walking speed to smoke.
- The desire lines are then drawn at chosen moments of the fire, to show them moving.

### D3. The slime informs the people leaving

- The crowd senses the slime's trail as the dynamic floor field (`k_D`, Kirchner and Schadschneider), so people follow the desire lines, through every exit (or only the exits chosen: the option "Only the exit picked" shows what losing the others costs).
- Helbing et al.'s ground comfort `G` as the alternative: the crowd wears its own trail as it goes (`kappa` from direct paths to a minimal way system).
- Measured: the exits' loads and the last one out with and without, over several seeds.

### D4. Dense crowds: mosh pits and crowd turbulence (research, off the main path)

- Silverberg et al.'s model on the grid (src/core/physarum.js's grid and seeded agents as the base): repulsion, propulsion, flocking, noise. Where the crowd at a bottleneck becomes as dense as a pit, whether its motion turns gas-like or vortex-like.
- Studied on its own; not a default in the building's evacuation.

### D5. Trails over terrain

- Helbing et al.'s trail model on an open plan with obstacles and slopes (a park, a forest trail), with the slime mould on the same plan: whether the slime's desire lines and the walkers' trails agree. The grid model (physarum.js) already runs on images, so a site's terrain can be read as its substrate.

### D6. Validation

- The slime's desire lines against the crowd's footfall (where the people leaving actually walked), per building and per fire.
- Helbing et al.'s figures: a park with entrances at its corners gives their published way systems as `kappa` grows.
- Lean: the clearance and medial-axis definitions (a ridge node has clearance at least its neighbours' along a direction), the retardant's monotonicity (more smoke never makes a place more attractive).

## Gaps found against the literature (2026-11-11)

The user shared a summary of slime-mould egress work (it reads as an AI's summary; checked below) and asked what the implementation misses. Ranked, with what each would take:

| # | Gap | Now | Stage |
| --- | --- | --- | --- |
| 1 | **Bottlenecks.** Tube thickness is flux; the desire lines' flow counts 2 m floor samples, not people, and is never set against what a door or corridor passes. | People-weighted flow over each link's clear width (a door's from the IFC, a corridor's twice its clearance) against the specific flow of 1.3 persons/s per metre: the seconds each would hold everyone the slime sends through it. | built, not released |
| 2 | **Dead ends.** Physarum explores and prunes; the weights average only the run's second half, so the explored-then-abandoned branches (the misleading corridors) are thrown away. Nor does the code check limit dead ends. | The slime's abandoned branches (much visited early, little trail late); and the walk before a second, separate way out opens (the shared part of the routes to the two nearest exits), against IBC 2021 1020.5's dead-end limits of 20 ft (6.1 m), 50 ft (15.2 m) sprinklered for groups B, E, F, I-1, M, R-1, R-2, S and U, none where the dead end is shorter than 2.5 times its width (from an ICC interpretation and two summaries; ICC's text is behind a login, to check against it). | built, not released (rooms and the 2.5-width exception next) |
| 3 | **Real smoke.** The soft fire ramps over 30 s before a place is untenable, a stand-in; debris and structural blockage are not modelled. | A Flamescapes run's visibility at head height, frame by frame, as the retardant (graph fires keep the ramp). | built, not released |
| 4 | **Sightlines.** The sensors look 1.5 m ahead whatever the smoke. | The sensor offset shortens with the visibility where each agent stands (with 3). Walking speed in smoke (Fridolf, Ronchi and Nilsson 2013) for the people leaving, to read first. | sensors built, not released; walking speed next |
| 5 | **Where people start.** One person per 9 m^2 everywhere, for the slime and the people leaving. | Occupant load by room use (IfcSpace names and the code's load factors). | next |
| 6 | **Assembly areas.** Food only at the exit doors; an exit into a closed courtyard counts as one onto the street. | Exits weighed by what lies beyond (the gathering places, the site). | next |
| 7 | **Tubes that thicken and wither.** Agents leave trails, not conducting tubes. | Tero's flux-conductivity model (Bonifaci et al.'s convergence) on the graph, and Physarum user-equilibrium assignment (Xu, Jiang and Shou 2018, *Applied Mathematical Modelling*; Liu, Zhang and Deng 2014, arXiv 1406.2128) to balance the exits. | research |
| 8 | **Prior work.** Kalogeiton, Papadopoulos, Georgilas, Sirakoulis and Adamatzky's slime-mould crowd model (2015: *Int. J. General Systems* 44:354-391, doi:10.1080/03081079.2014.997527, checked; and *Studies in Computational Intelligence* 600:123-151) is the nearest; they check it against published fundamental diagrams, as ours has not been. | Read, reproduce a case, and the fundamental diagram for the people leaving. | research |
| 9 | **A physical test.** | Export the plan, exits and occupied zones for a printed agar maze (inoculation at the occupied zones, oats at the exits, light for the fire) to set the organism's network against the digital one. | research |
| 10 | **Export.** | The desire lines, medial axis and weights out to Grasshopper and Rhino. | next |
| 11 | **Seeds and time.** One seed; one average. | Several seeds, the lines that persist; snapshots on the fire's clock. | next |
| 12 | **The slime guiding the crowd** (D3). | As planned. | next |

Checked from the summary: paper 1 is Kalogeiton et al. (above), whose claim is agreement with published fundamental diagrams, not that it "accurately predicts frantic human path choices"; paper 2 ("Physarum-based Emergence Evacuation Plan") was not found, the nearest being the Physarum user-equilibrium traffic papers in item 7; paper 3 is Tan, Wang, Cao and Xu 2023 (*Heliyon*), a physical slime network among Wuhan's hospitals that did better than an origin-destination network on global efficiency, city-scale, not building egress. That the slime's networks predict where people go is this plan's hypothesis to test (items 8 and 9), not established.

## What it is and is not

Desire lines, the medial axis and the slime's tubes are readings of the building and the fire, for design: where the escape concentrates, where it departs from the geometry. They are not timed evacuation results; the forecast and the people leaving are.
