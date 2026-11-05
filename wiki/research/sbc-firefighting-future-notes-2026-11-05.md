# Future notes: Saudi Building Code checks, fire hoses and firefighter access (2026-11-05)

Status: notes for later work, not a design. No limit below is quoted from a code. Each value marked "from the code" must be read from the current Saudi Building Code edition (SBC 201, the general building code, and SBC 801, the fire code) and cited with its clause before any check uses it. Companion note: `fire-suppression-note-2026-11-05.md` (sprinklers, activation, spray).

## What the app already has to build on

- **Graph quantities:** the walkable graph gives every node's walking distance to the nearest exit door (the deadline map and `earliestArrival` in `src/core/forecast.js` and `deadline.js`), the stairs and doors, and the exits read from the IFC's own flags.
- **Elements and openings:** element identification (`src/core/identify.js`) and the Flamescapes openings (doors open or closed, windows, air terminals).
- **Fire results:** for each node, when it becomes untenable (heat, the ISO 13571 CO dose, visibility), which gives an available-time measure to set against the walking time.

## 1. SBC egress checks (a "Code check" view)

Each check is a measurement the graph already supports, compared with a limit from the code that depends on the occupancy group and on whether the building is sprinklered:

- **Exit access travel distance:** the walking distance from each node to the nearest exit, against the maximum from the code. The view should colour the nodes over the limit and give the worst room. The graph's distance is a walking path, which is what the code measures; it is not straight-line distance.
- **Common path of egress travel and dead ends:** the distance until two separate routes to two exits become available, and corridor dead-end lengths, against the limits from the code. Needs two-route search (`alternatives.js` has k shortest paths).
- **Number of exits and their separation:** the exits a storey or room has, against the number the code requires for its occupant load. Two exits must be separated by at least the fraction of the area's diagonal that the code sets.
- **Occupant load and exit capacity:**
  - the occupant load is each IfcSpace's area divided by the code's load factor for its use; the use comes from the space's name or the `Pset_SpaceCommon` / `OccupancyType` properties where the exporter writes them;
  - the exit width required is that load times the code's width per occupant, which differs for stairs and level components;
  - the door widths are already read (`ifcSemantics.js`, `widthM`).
- **Stairs:** riser, going and width against the code, where the IFC carries stair properties (`Pset_StairCommon`, `Pset_StairFlightCommon`).

The output would be a report card per check (pass, fail, cannot tell), each with the clause it applies, and a JSON export alongside the TopologicPy export. "Cannot tell" must be explicit wherever the model lacks the data, such as occupancy or a door width.

## 2. Fire hoses and standpipes

- **In the IFC:**
  - `IfcFireSuppressionTerminal` with PredefinedType HOSEREEL, FIREHYDRANT or BREECHINGINLET (IFC4), with its type object;
  - in IFC2x3, `IfcFlowTerminal` typed by `IfcFireSuppressionTerminalType`.

  The element identification rules should name these too. A check of the local corpus is needed: few architectural models carry them.
- **Coverage check:** a hose reaches the points within its hose length along the walking graph, plus a jet allowance; both lengths come from the code or the product data. A multi-source shortest path from every hose cabinet gives each node's covered/not covered state. Nodes out of reach are the result. This is the same computation as the exit distance, with the cabinets as sources.
- **With the fire:** a hose cabinet whose surroundings become untenable before the fire service would reach it is of little use. The node's untenable time from the Flamescapes run answers this.
- **Manual suppression in the solver:** the suppression note's step 4 (one discharge at a point and time, by agent and fire class).

## 3. Firefighter access points

- **What to read:**
  - fire service access doors and openings, the fire command centre, firefighting lifts and stairs;
  - in practice these are mostly named elements or spaces (a door or room named for the fire service), occasionally `IfcTransportElement` with a firefighting property, and rarely anything typed;
  - the site's access road is usually outside the IFC.
- **Checks:**
  - the walking distance from each access point to every storey through the firefighting stair;
  - how many exits and stairs the fire service shares with evacuees;
  - whether the access route stays tenable during the fire (untenable times along it).
- **Input:** a manual "access point" picker, like the fire origin, is needed for models that do not mark them.

## 4. Doors in the graph fire models

The Flamescapes solver now closes doors for the smoke. The three graph fire models (heat diffusion, spread along links, distance bands) still treat every door link alike. A closed door could hold a link for its fire resistance period, the rating coming from `Pset_DoorCommon.FireRating` where present, so a fire behind a closed rated door reaches the next room later. This needs the rating's meaning (integrity and insulation minutes) from the code and from EN 13501-2 or the national equivalent.

## Sources to read before any of this is built

- **The Saudi Building Code:** SBC 201 and SBC 801, the current edition, from the Saudi Building Code National Committee. Each limit is to be cited by clause. The editions and their English texts need checking.
- **The IFC4 schema:** `IfcFireSuppressionTerminalTypeEnum`, `Pset_SpaceCommon`, `Pset_DoorCommon` (FireRating, FireExit) and `Pset_StairCommon`.
- **ISO 13571 and the SFPE Handbook chapters** already used for tenability, to set the available time against the required egress time.
