# Browser core and the lattice graph (2026-10-27)

Goal from the user: run the core features in the browser so the site works
from GitHub Pages alone, make it fast, ship example IFCs with the site, and fix
"the grid extracted from the IFC has diagonals".

## Diagnosis: why the graph has diagonals

Measured on the user's model (`Building-Architecture.ifc`, buildingSMART's
"ifc silly sample scene") and on the Duplex:

| | silly scene | Duplex |
| --- | --- | --- |
| nodes / edges | 101 / 707 | 1,765 / 10,441 |
| mean degree | 14.0 | 11.8 |
| edges at an odd angle (not 0/45/90/135 within 5 deg) | 54% | 57% |
| edges longer than 1 m | 45% | 18% |
| nearest-neighbour spacing, median (setting 0.5 m) | 0.47 m | 0.19 m |

Two causes, both in the pipeline rather than the model:

1. **Sampling is per triangle.** `sampler.worker.js` `sampleMesh` lays a
   barycentric lattice inside each triangle. IFC slabs are tessellated into
   arbitrary, often long thin triangles, so each triangle gets its own sheared
   pattern and spacing, and shared edges are sampled twice. Voxel and column
   de-duplication thin the cloud but keep the irregular positions.
2. **Linking is k-nearest.** The server links each point to its 12 nearest
   neighbours within 2.25 m (`geometry/adjacency.py`). On an irregular cloud
   that is every angle; on a sparse one it reaches far, knight's-move points.

## Fix

One lattice for the whole building, aligned to it:

- Orientation: the dominant wall direction (length-weighted angle mod 90).
- Sample every walkable triangle at the shared lattice positions inside it
  (point-in-triangle in plan, height interpolated). Each (i, j) column keeps
  the top of each height cluster, as before.
- Link lattice neighbours only: the 4 axis neighbours, and the 4 diagonals
  unless "rectilinear", with no corner cutting (a diagonal needs both axis
  neighbours). Floor links obey `max_edge_rise`; links touching a stair obey a
  gradient limit instead, so a flight is walkable at any lattice spacing.
- Walls block lattice edges that cross them; edges within a door's half-width
  of a door centre are exempt. Door nodes are the lattice nodes nearest each
  door, so no off-lattice points are added.

## Browser core

Built in plain JavaScript in a worker, not Pyodide: Pyodide ran the Python
unchanged at about 2x native, but costs ~20 MB and ~3.5 s per visit
(measured in Chrome, scratch benchmark 2026-10-27), and needed four int32 fixes
(`np.repeat` with int64 counts) to run at all.

Order of work, each step verified and committed:

1. Lattice graph builder in JS (`src/core/`), unit-tested with `node --test`.
   The browser builds the graph; when a backend is reachable the graph is
   uploaded (`POST /api/graph/import`) so server-only features keep working.
2. Example IFCs (`tools/make_example_ifcs.py`, CC0, ours) in
   `public/examples/`, with an "Examples" picker. Not shipped: the Duplex and
   the buildingSMART sample, whose licences could not be confirmed.
3. Route (A*), components, hazard weights in JS.
4. Alternatives, fire (radial, flood, temperature), RL in JS.
5. The backend becomes optional: TopologicPy engine, Navigation cross-check,
   topology JSON. The site works without it.

## Results on real models (tools/lattice-batch.mjs)

Iterated against 16 models: the Duplex, the buildingSMART sample scene, our two
examples, and 12 web-ifc test models (ThatOpen/engine_web-ifc
`tests/ifcfiles/public`, downloaded to `C:\Users\lmurugesan\ifc-samples` for
testing only; their licences vary, so they are not shipped). "Largest" is the
share of nodes in the biggest connected piece with walls on.

| Model | Start of loop | Now | Notes |
| --- | --- | --- | --- |
| Duplex | 50% (k-NN said 94%) | 51% + 49% | Two mirror-image dwellings, each with its own doors: correct. k-NN's 94% leaked through the party wall. |
| FZK-Haus (KIT) | 6 parts, 51% | 1 part, 100% | Stair grid at half spacing, section walls. |
| Office A (NIBS) | 20 parts, 53%, 0 door nodes | 4 parts, 96% | No floor slabs at door level: floors come only from IfcSpace bottoms. Ceilings (IfcCovering CEILING) had been read as floors. |
| Dental clinic | 59 parts, 12% | 18 parts, 69% + 23% | |
| DigitalHub | 42 parts, 50% | 17 parts, 63% + 36% | |
| C20 Institute (KIT) | 11 parts, 74% | 6 parts, 75% | |
| HouseZ | 19 parts, 52% | 17 parts, 37% | Split-level basement (floors at -5.85/-5.64/-5.30) and a top floor no stair reaches. Not fixed. |
| kleine Wohnung | 14 parts, 68% | 19 parts, 26% + 18% | Several flats; partly legitimate. |
| buildingSMART sample | 1 part (707 edges, 54% odd angles) | 1 part, 316 lattice edges | |
| office / courtyard examples | | 1 part each | Courtyard grid follows its 30 degree turn. |

What changed, each found on a specific model:

1. Stair nodes on a half-spacing sub-grid; stair links may climb a gradient
   plus one riser (0.30 m). FZK-Haus's 1 m stair had one column and broke.
2. Door bridging on every door, both sides (Duplex back door).
3. Fragments under 8 nodes with no door or stair are pruned (slab edges
   outside the facade).
4. IfcSpace bottoms are walkable ("LOD1 inside the walls"); coverings count as
   floor only when typed FLOORING (Office A).
5. Obstacles are sections of the real geometry 0.75 m above each floor
   (walls, windows, railings, columns, curtain walls, members, plates), with a
   segment grid index. Openings without an IfcDoor stay open; L-shaped and
   curved walls are right. Dental clinic build 2.3 s -> 0.6 s with the index.
6. Roof exclusion measured from the whole model's top, only with floor 2 m
   below; the old rule emptied single-storey models.

Still open: Mini Project produces 0 nodes (24 floor meshes, no walls);
IfcOpenHouse only 9; HouseZ split levels need ramps/steps between floor
heights (a "step" link up to ~0.6 m where a door joins two levels?).

## Open items

- TopologicPy `Navigation` does not join the generated office's stairs to its
  floors (4 parts), although the stairs have 1 m landings; Duplex stairs do
  join. Unexplained; check `IFC._NavigationExtract` stair rules.
