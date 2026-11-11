# Jones's Physarum models as pseudocode, and a slime mould egress (2026-11-11)

The user asked whether Jeff Jones's multi-agent *Physarum polycephalum* models can inform the egress algorithms, "with weights for egress routes and against fire nodes", then to port them: convert the code to pseudocode, augment it with modern methods, and build it. The user had the author's example sketches (Processing; shared privately in 2022, not in this repository). This note restates the models as pseudocode from those sketches and the published papers; the code in `src/core/physarum.js` and `src/core/slimeEgress.js` is written from this note, not from the sketches.

## Sources (DOIs checked on Crossref, 2026-11-11)

- Jones, J. (2010). Characteristics of pattern formation and evolution in approximations of *Physarum* transport networks. *Artificial Life* 16(2): 127-153. doi:10.1162/artl.2010.16.2.16202. The agent model.
- Jones, J. (2011). Influences on the formation and evolution of *Physarum polycephalum* inspired emergent transport networks. *Natural Computing* 10: 1345-1369. doi:10.1007/s11047-010-9223-z. Growth and shrinkage of the population, nutrients.
- Jones, J. (2015). *From Pattern Formation to Material Computation: Multi-agent Modelling of Physarum Polycephalum*. Springer. doi:10.1007/978-3-319-16823-4.
- Jones, J. and Saeed, M. (2007). Image enhancement: an emergent pattern formation approach via decentralised multi-agent systems. *Multiagent and Grid Systems* 3(1). doi:10.3233/mgs-2007-3108. Section 6.4: shortest paths through a maze by chemotactic agents on a diffusing field, adapting when a wall is removed.
- Steinbock, O., Tóth, Á. and Showalter, K. (1995). Navigating complex labyrinths: optimal paths from chemical waves. *Science* 267: 868-871. doi:10.1126/science.267.5199.868. The chemical-wave root of the maze method.
- Tero, A., Kobayashi, R. and Nakagaki, T. (2007). A mathematical model for adaptive transport network in path finding by true slime mold. *J. Theor. Biol.* 244: 553-564. doi:10.1016/j.jtbi.2006.07.015. The flow-conductivity model.
- Bonifaci, V., Mehlhorn, K. and Varma, G. (2012). Physarum can compute shortest paths. *J. Theor. Biol.* 309: 121-133. doi:10.1016/j.jtbi.2012.06.017. Convergence of Tero's model to the shortest path.
- Burstedde, C. et al. (2001), doi:10.1016/S0378-4371(01)00141-8, and Kirchner, A. and Schadschneider, A. (2002), doi:10.1016/S0378-4371(02)00857-9: the floor-field pedestrian model, whose dynamic field is a trail laid and followed as the agents' is.

## 1. The network model (Jones 2010, 2011)

State: a grid of cells, each with a substrate value (a wall, a food source, free floor), a trail value (chemoattractant) and at most one agent. An agent has a real position, the integer cell it occupies, and a heading.

```
parameters
  SO  sensor offset (cells)          SA  sensor angle (deg)       RA  rotation angle (deg)
  SS  step size (cells)              depT trail deposited per successful step
  pCD probability of a random turn   damp fraction of trail lost per step
  food: value projected each step at food cells (suppressed while agents cover them)
  adaptive: GW, Gmin, Gmax (growth window, counts); SW, Smin, Smax (shrinkage window); test every N steps

init: place agents on free cells (not wall, not occupied), random headings

each step t:
  project food: for each food cell f
      v = (an agent within its 3x3) ? suppressedValue : foodValue
      add v to trail over f's 3x3
  for each agent, in a fresh random order:                       # motor stage
      if random() < pCD: heading = random angle; continue
      target = position + SS (cos heading, sin heading)
      at a border: reflect the heading (or wrap)
      cell = round(target)
      if cell is a wall or occupied:
          heading = random angle; position = centre of current cell   # blocked: no deposit
      else:
          move to cell; trail[cell] += depT
          if adaptive and t mod N == 0: mark for division when the agents in the
              GW x GW window round it number in (Gmin, Gmax]
  for each agent, in a fresh random order:                       # sensory stage
      if adaptive and t mod N == 0: mark for death when the agents in the
          SW x SW window number outside [Smin, Smax]
      F, FL, FR = trail at position + SO (cos, sin)(heading + 0, -SA, +SA)   # -1 outside the grid
      if any sensor is outside the grid: heading += RA
      elif F > FL and F > FR: keep heading
      elif F < FL and F < FR: heading += (FL < FR ? +RA : -RA)
      elif FL < FR: heading += RA
      elif FR < FL: heading -= RA
  diffuse: trail = (1 - damp) * mean3x3(trail); trail = 0 on walls (absorbed at the border)
  divide: each marked agent buds a new one into a random free cell of its 3x3 (not a wall)
  die: remove each marked agent
```

Defaults in the author's map example: SO 3, SA 60, RA 60, SS 1, depT 5, pCD 0, damp 0.1, food 10 (2.5 while covered), adaptive with N 5, GW 9 (0, 10], SW 5 [0, 23]. Larger SO with smaller SA and RA (for example SO 9, SA 22.5, RA 45) give coarser, smoother networks (Jones 2010).

## 2. The erosion model (the author's erosion example)

The same agents, but they wear a substrate down instead of laying trail, and seek the low.

```
as the network model, with
  depT = 0                                   # no trail
  each step: trail += projectScale * substrate   (scale 0.01); then diffuse as above
  sensing inverted: turn towards the LOWEST of F, FL, FR (each comparison reversed)
  each successful step: substrate[cell] = max(0, substrate[cell] - erodeBy)   (erodeBy 10)
  with probability pTP (0.01) an agent jumps to a random free cell instead of moving
```

Worn cells project less, so the agents gather there and wear them more: channels deepen by positive feedback, as desire lines do in grass.

## 3. Chemotactic shortest paths (Jones and Saeed 2007, 6.4)

```
each step:
  chemical[exit] += source; chemical = (1 - damp) * mean3x3(chemical) over free cells
  each agent: face the neighbour cell with the most chemical (ties at random), step if free
  each agent that moved: trail[cell] += depT
  trail *= (1 - erode)                       # the selection pressure
the persistent trail is the shortest path from the agents' start to the exit; open a
wall and a shorter path takes over, the old one fades
```

## 4. Modern augmentations

1. **On the walkable graph, not a picture.** The building's walkable graph is already a lattice about 0.5 m apart with walls as blocked links, storeys and stairs (lattice.js). The agents live on its nodes; a sensor is the node reached by marching SO steps from the agent, at each step along the link best aligned with the sensor's direction, stopping at a wall (a sensor that hits a wall reads -1, as one outside the grid does). Stairs are links like any other, so the slime mould climbs between storeys.
2. **An attractant that never fades to nothing.** A diffusing source decays exponentially, so far from an exit the field is flat and the agents wander (Jones notes the plateau). The exits' attractant is seeded from the walking distance to them instead (a geodesic field, the static floor field of Burstedde et al.): `A(v) = exp(-d(v) / lambda)`, which keeps a gradient across a whole building. The agents' trail is laid and diffused on top, as in Jones's model.
3. **Fire as a repellent and a wall.** A place the fire has made untenable is a wall: no agent enters it and trail there is absorbed, as the maze's walls absorb chemical. Places near it carry a repellent `R(v) = exp(-dFire(v) / mu)` from the walking distance to the untenable places, which the agents sense as negative. As the fire grows the tubes through it fade and others take over, as the maze path does when a wall moves.
4. **A flux, not a crowd.** Agents start on the occupied floor and, when they reach an exit, start again from a random occupied place (Jones's teleport), so the trail measures how much escape traffic each place carries.
5. **Weights.** The trail on each node, normalised, is the egress weight: where the slime mould's tubes are. The route from a start follows the steepest ascent of `A + w T - r R` (attractant, trail, repellent) to an exit, and is reported against the shortest walk.
6. **Seeded, deterministic, typed arrays.** One seed gives one run (mulberry32), as the crowd's does; the state is typed arrays, run in a worker; off by default, a chunk loaded only when asked for.
7. **Later.** WebGPU compute for millions of agents on the grid model; Tero's flow-conductivity model on the graph, which Bonifaci et al. proved converges to the shortest path, as a deterministic counterpart that could be stated in Lean.

## 5. Checked against the code, and proved

The ports were checked line by line against the author's sketches (not in this repository) and corrected where they differed: a step that rounds back into the agent's own cell is blocked (the code's occupancy test counts the agent itself); the sensors are measured from the agent's cell, not its real position; without wrap-around the diffusion zeroes the border ring; the adaptive tests fall on steps 5, 10, ... and not within 5 cells of the edge; in the erosion model a random turn is drawn before a jump, a jump takes a random heading and lands only on a cell the agent may start on (`startOn`: the code's start colour, 0), and the grid wraps round. On the walkable graph, walls absorb trail as the grid's do (divisor 9 on floors), SA and RA are 60 degrees and the exits are food (10, or 2.5 while covered).

`formal/fireegress/FireEgress/Physarum.lean` proves 23 results on the code's rules: the diffusion's total on a torus is exactly (1 - damp) times the old, never negative, under the old maximum, and with absorbing border and walls falls at least as fast; the turn keeps a strictly strongest front, never turns away from a strictly stronger side, and turns left on a tie (the paper says at random); moving agents one at a time in any order, and teleporting them, keeps one agent a cell and none on a wall (so none in the fire, on the graph); erosion stays in [0, s], takes exactly e when it can, and keeps a deeper channel deeper.

## 6. What it is and is not

The slime mould's routes are emergent and stochastic; they approximate shortest and safe paths and show where escape traffic concentrates. They are not an evacuation model to quote: the forecast's earliest safe route (forecast.js) is exact for one person, and the crowd (crowd.js) times people by the SFPE values. The weights are for seeing which corridors and stairs the building's escape leans on.
