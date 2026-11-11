# Plan: the evacuation distance map and the wheelchair accessibility audit (2026-11-11)

The user pointed at two of Florent Poux's kits in [florentPoux/3d-data-materials](https://github.com/florentPoux/3d-data-materials) (code MIT; read 2026-11-11), `evacuation-distance-map` and `wheelchair-accessibility-audit`, with their outputs run locally, and asked: are we already addressing these evacuation plans, and which algorithms would we rather use? Plan and audit first; nothing is built here.

The kits work on classified point clouds of S3DIS (Armeni et al., CVPR 2016), whose terms limit use to non-commercial research and forbid redistribution. Their outputs (evacuation routes, metrics, issue lists) are derived from it and stay out of this repository; the numbers quoted below are from the kits' own READMEs.

## What the kits do

**Evacuation distance map** (S3DIS Area 1, 1.38 M points, about 53 s on a CPU). Points get roles from their classes (floor, wall, door, obstacle); a 5 cm walkable grid with clearance to each obstacle's edge, doors open at their leaf only; eight hand-picked exits; one multi-source Dijkstra from all exits over a 16-neighbour grid (8 neighbours read 2.3 % long on average and 5.5 % at most against 16; 32 neighbours differ from 16 by under 1 %); a route from each space's farthest point (33 spaces, the farthest 23.4 m, the mean 8.2 m); isolines every 5 m; one-way floor split into common paths in rooms and dead-end corridors in circulation, the corridors from graph bridges (7 over 2 m, the longest 6.1 m, the longest common path 8.4 m); and a what-if sweep, each exit blocked in turn, then a cascade blocking the worst in order (with B, E and G lost the farthest walk goes from 27.1 to 50.0 m), 36 solves.

**Wheelchair accessibility audit** (S3DIS Area 2, 4.03 M points, 3.7 s). A walkable grid; the clear width everywhere from a distance transform read face to face; where a wheelchair's centre can stand; each of 38 doors measured between its jambs; turning space by a morphological opening with a 1.50 m element (a half-size element would accept 43 % more floor); reach as connected components from the entrances with rooms sealed at gaps a walker cannot pass (8 of 26 rooms by wheelchair, 23 on foot); a numbered issue list (passage too narrow, door too narrow, stairs only, no turning space, not measured); and an "act" step, one added ramp taking the reach from 8 to 16 rooms. Profile values: passage 0.90 m, door 0.80 m, turning 1.50 m, body 0.45 m; the README cites no standard.

## Audit: are we already addressing them?

| Kit feature | Studio now | Gap |
| --- | --- | --- |
| Distance to the nearest exit everywhere | Yes: the code check's multi-source Dijkstra per storey (codeCheck.js), drawn as distance to spare against the limit ("Show travel") | Our lattice is 0.5 m with 8 neighbours: up to 8.24 % long (proved, Eikonal.lean `oct_12_5`); the kit's 5 cm, 16 neighbours is within 2.75 % |
| Exits | Doors out of the building (exitDoors `final`), stairs as exits on upper storeys | None: ours come from the IFC, the kit's are picked by hand |
| A route from every space's farthest point | No: the longest walk per storey, and a route from a chosen start | Per-room worst point and its route, from the same field |
| Catchment of each exit | Partly: the crowd's people per exit, the slime's flux per exit | Floor area and rooms per exit, from the field |
| Isolines | No | Contours of the field every 5 m (drawing only) |
| Dead ends and common path | Partly (dc2229c): the walk before a second way out (Gromov product), against IBC 1020.5, rooms included | Corridors told from rooms; a dead-end corridor as an object (mouth, tip, length, width) from graph bridges; 1020.5's 2.5 x width exception; rooms against Table 1006.2.1's common path |
| Exit blocked, what-if | Partly: the performance check's "exit blocked" case (one plan made with that switch); the crowd's "the exit you picked" | A cheap sweep of every exit (N-1) and a worst-first cascade, the farthest walk against the limit each time |
| Clear width | Coarse: the brushfire clearance on the 0.5 m lattice (skeleton.js), used for bottlenecks | 0.5 m steps cannot tell 0.80 from 0.90 m: a fine raster is needed |
| Door widths | Yes for exit doors: the IFC's widths against 1010.1.1 (0.80 m SBC, 0.813 m IBC) | Every door, against the accessible-route width |
| Turning space, wheelchair reach, issues, act | No | All of it |
| Point-cloud input | No: Studio reads IFC | Scan to the same raster (with a collaborator's scan-to-BIM work) |

So the evacuation map is largely there, at a coarser metric and with three missing readings (per-room routes, catchments, the exit sweep); the accessibility audit is not there at all.

## Algorithms I would use

1. **Keep the graph for routing; add a fine raster per storey for measuring.** The graph's Dijkstra and A* carry the Lean proofs, the fire, the crowd and the slime; replacing it would cost all of that. Widths, turning space and an accurate distance need a raster: each storey's floor at 5 to 10 cm, occupied where the walls' sections at body height (graph `sections`), columns and furniture footprints stand, doors open at their IFC leaf width.
2. **Exact Euclidean distance transform** on that raster (Felzenszwalb and Huttenlocher 2012, *Theory of Computing* 8:415-428, doi:10.4086/toc.2012.v008a019, DOI checked): linear time, exact. Clear width is twice it along its ridge (the medial axis we already compute, now at 5 cm); turning space is the opening with a disc, which on an exact transform is the centres at least 0.76 m clear dilated by 0.76 m: no structuring element to rasterise, and no half-size mistake.
3. **Distance field: 16-neighbour Dijkstra on the raster**, with Fast Marching (Sethian 1996, *PNAS* 93:1591-1595, doi:10.1073/pnas.93.4.1591, DOI checked) as the check. The 16-neighbour error has a clean bound, 1 / cos(13.28 deg) - 1 = 2.75 %, the same argument as the 8-neighbour 8.24 % already in Eikonal.lean, so the proof extends; Fast Marching is isotropic and takes a speed field directly, which is where walking speed in smoke (gap 4 of the desire-lines plan) belongs later. Storeys join through the graph's stair links.
4. **Dead-end corridors from bridges** (Tarjan 1974, *Information Processing Letters* 2:160-161, doi:10.1016/0020-0190(74)90003-9, DOI checked) on the walkable graph within circulation spaces, told from rooms by IfcSpace names (corridor, lobby, hall, circulation) and, without names, by shape (long and narrow on the raster). A corridor's mouth is its bridge to the rest; its length is read from the mouth as 1020.5 measures it, and its width from the transform for the 2.5 x width exception. Rooms take the common path against Table 1006.2.1.
5. **Exit sweep and cascade**: one multi-source solve per configuration (tens of milliseconds on our graphs), each exit blocked in turn, then the worst first; the farthest walk, the share over the limit and the rooms rerouted each time.
6. **Wheelchair reach**: the same raster thresholded at the profile (centre space where the transform is at least half the passage width), stairs removed, doors below the door width closed, connected components from the building's accessible entrances; rooms reached are IfcSpaces. Profiles from ICC A117.1-2017 and the 2010 ADA Standards (accessible route 36 in, 915 mm; door 32 in, 815 mm clear; turning circle 60 in, 1525 mm; sections 403.5.1, 404.2.3 and 304.3.1, to check against the texts) and ISO 21542 as the international set; the Saudi requirement to be read from SBC 201 before any claim.
7. **Act step**: greedy, as the kit's: each candidate (a ramp beside a short flight, a door widened, a lift added) scored by the rooms it newly connects, the best taken, again; reported as options, not a design.
8. **Accessible means of egress** (IBC 2021 1009): the link between the two kits. From every room a wheelchair reaches, an accessible route to an exit, an area of refuge or an elevator; and later wheelchair users in the crowd, who cannot take the stairs and wait at refuge.

## Stages

- **E1. Readings on the graph we have (cheap).** Per-room worst points and routes, exit catchments, isolines, the exit sweep and cascade. Off until asked; tens of milliseconds.
- **E2. Corridors and rooms.** Bridges, circulation by IfcSpace names, dead-end corridors with mouth, tip, length and width, the 2.5 x width exception, rooms against the common-path limit. Closes the desire-lines plan's dead-end follow-up.
- **E3. The fine raster.** Per storey in a worker, off by default, sized to the device (10 cm on phones): exact transform, widths, 16-neighbour field, Fast Marching check. Port the kits' known-answer tests (synthetic corridors, doors and floors; MIT) and measure against them.
- **A1. The wheelchair audit.** Profiles, reach, the issue list on the plan and as CSV, the act step.
- **A2. Accessible means of egress**, then wheelchair users in the crowd.
- **S1. Scan input, with a collaborator.** A classified point cloud (S3DIS's 13 classes) onto the same raster, so a scanned building gets the same map and audit, and an as-built scan can be set against its design IFC.

## Checks

- Lean: the 16-neighbour bound; the transform's opening equals the disc test; a bridge's removal disconnects its tip.
- Against the kits: on their synthetic tests, the same answers; on S3DIS (local only, never committed), the farthest walk 23.4 m and mean 8.2 m on Area 1, and 8 of 26 rooms on Area 2, once S1 exists.
- Speed: the Vienna model on the web with everything off as now; each reading timed when on.

## What it is and is not

The distance map and the dead ends are code readings of the geometry, as the code check's. The audit flags where a profile fails; it is not a certification, and the values must be checked against the code that applies (SBC 201 for Saudi Arabia).
