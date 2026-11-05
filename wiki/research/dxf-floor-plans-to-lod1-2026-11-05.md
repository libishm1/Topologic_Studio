# Floor-plan drawings to LOD1 in the browser: the DXF reader and the drafting rules

Date: 2026-11-05
Status: core and benchmark. New files only: `topologicpy-web-frontend/src/core/drawingToLod1.js` and its tests, `tools/dxf-lod1-check.mjs`; numbers in [`benchmark/results/2026-11-05-dxf-lod1.json`](../../benchmark/results/2026-11-05-dxf-lod1.json). Nothing is wired into the app yet.
Related: the CM-iTAD lab's LOD1 pipeline (a private project kept outside the repository; notebook `LOD1-00-DraftingRooms`, `lod1/drafting.py`), whose method this follows; `src/core/lod1.js`, the model it feeds.

Conventions. The lab's two buildings are real and private: this note and the results file hold aggregate counts only, never a drawing, a name or a per-room result. They are called A (a 4-storey residence, 54 written rooms on 4 plans) and B (a hotel, 194 written rooms on 7 plans, the typical floor counted once), as in the results file.

---

## 1. The DXF reader: an own reader, not `dxf-parser`

| | `dxf-parser` 1.1.2 (npm) | own reader (`readDxf`) |
|---|---|---|
| Licence | MIT | the repository's |
| Maintenance | last release 2021-11-12, one maintainer | here |
| Size | 345 kB unpacked (70 files, ESM and CommonJS builds) plus `loglevel` (115 kB) | about 450 lines inside `drawingToLod1.js` |
| LINE, LWPOLYLINE (bulges), POLYLINE + VERTEX, ARC, CIRCLE, TEXT, MTEXT, `$INSUNITS` | parsed as records | parsed and converted |
| INSERT with nested BLOCKS | the INSERT and the block's records; exploding is left to the caller (insert point, rotation, scales, base point, column and row arrays, nesting) | exploded at any depth, transforms composed, mirrored inserts kept anticlockwise |
| Object coordinate systems (extrusion 0, 0, -1: mirrored geometry) | not applied | applied |
| Layer 0 inside blocks, BYLAYER and BYBLOCK linetypes | not resolved | resolved (`layer`, `linetype`), and the lab's `A>B` key (`path`) |
| MTEXT | raw, with its formatting codes | formatting removed; the text's centre estimated from its attachment point |
| Time on the lab's sheets (5.8 to 13.4 MB) | 153 ms (9 MB) and 261 ms (13.4 MB), unexploded | 60 to 140 ms for the hotel's sheets, 250 to 390 ms for the 13.4 MB sheet, exploded |

The drafting needs exactly what `dxf-parser` leaves out: every block exploded in world coordinates, the layer and linetype each part is drawn with, and plain texts with a position. Using it would save only the group-code tokenizer, about 150 lines, and would add an unmaintained dependency. So no dependency was added. If one is added later, it should be loaded by a dynamic `import()` when a DXF is opened.

`decodeDxfBytes` turns the file's bytes into text: UTF-8 from AutoCAD 2007 (`AC1021`) on, else the header's `$DWGCODEPAGE` (e.g. Windows-1256 for Arabic) through `TextDecoder`. Binary DXF and DWG are refused with a message: save them as ASCII DXF.

Two facts about real sheets shaped the reader. The lab's sheets were converted by Rhino, which writes every arc as a polyline, so door swings are found by fitting circles to polylines as well as from ARC entities. And `$INSUNITS` is unreliable: every hotel sheet declares metres, but two are drawn in millimetres. The scale is therefore read from the drawing (section 2.1).

## 2. The method

`draftRooms(entities, opts)` is a pure function, safe in a Web Worker. It uses a uniform spatial hash for noding, snapping, ray casts and point location; nothing in its hot paths is worse than O(n log n). Defaults are in brackets.

1. **Walls.** The linework of the wall layers and window layers (windows close the facade), in metres, less:
   - the parts of door blocks;
   - fixture and furniture blocks (a block-name pattern in English and Arabic; the lab's `not_wall`);
   - dashed and dotted lines. A long dashed line is a projection: the floor below or a beam overhead. A straight dashed line 0.5 to 2.5 m long is the head of an opening, and is bridged like a door (step 2).
2. **Doors.** A door block (name pattern) or a swing on a door layer gives a hinge (the arc's centre) and two chords. The chord the drawing draws along is the open leaf, sampled at five points within 3 cm or 6 % of the radius. The other chord is the opening. With no leaf drawn, the chord parallel to more wall near the hinge is taken. Two leaves closing on one point are a double door. Folding and sliding doors (no swing) take their block's long axis. Each opening is bridged jamb to jamb: rays along the chord find the jambs (up to 0.6 m), and rays across from just inside the wall find both faces. A face is the last of the lines up to 12 cm apart: plaster, finish and core. The bridge is drawn on both faces and the centre line, so the doorway becomes a piece of wall.
3. **Outline.** The walls are rasterised (10 cm), closed by 1 m (openings up to 2 m bridged, as the lab's buffer and its inverse), filled, and pushed out by one pixel so the outline never runs along the facade. The outline's boundary bounds rooms where a facade is left open.
4. **Arrangement.** Endpoints within `snap` [3 cm] are joined. A dangling end goes onto a segment within `snap`, or else along its own line up to `extend` [30 cm] to the first line it meets. Then every crossing, touch and collinear overlap is noded, dangling edges are pruned, and the faces are traced from half-edges sorted by angle. A part's outer cycle becomes a hole of the face around it. A face is a room when its area is at least `minArea` [0.8 m²] and its width (2 A / P) at least `minWidth` [0.35 m], and it is not a long strip (width under 0.6 m and 4πA/P² under 0.1). Thin strips touching the outline are the outside.
5. **Names.** Texts inside a room are its labels. Levels, areas, dimensions, notes and single characters (grid bubbles) are not. Names typed in the old Arabic keyboard fonts are decoded (`decode`: `auto` when a text holds `Hg`, `all`, `none`). Classes come from a word list in English, Arabic and transliteration (the lab's general vocabulary), restricted to the hotel schedule's 13 classes. Unnamed rooms with a sanitary fixture are Toilet; long narrow ones are Corridor.
6. **Split by names.** A room holding two different names more than 1.5 m apart is cut along the shortest admissible line (0.3 to `maxCut` [3.5 m]) from a re-entrant corner (a wall end): to another corner, or along a wall's own line to the far side. The cut becomes an opening.
7. **Merge across openings.** An opening head between a named room and a small unnamed space (up to `mergeArea` [8 m²]) is dropped and the faces are traced again, when that space opens onto no other named room. This catches an entrance lobby or a wardrobe recess that serves one room. A hall serving several rooms stays.
8. **Grow to the centre-lines.** Every edge moves out by half the thickness of the wall behind it. Rays at 20, 50 and 80 % of the edge cross the wall faces, plaster lines included, to the next room or the outside, up to `maxWall` [0.8 m]. Neighbouring edges meet at their lines' crossing, or step where they run straight on. Holes under 2 m² (columns) are filled. Neighbours come from the same rays.
9. **Doors to rooms.** Probes every 5 cm up to 1 m on each side of the opening give the room, the outside, or "unknown" (inside walls).

`draftSummary` and `toLod1Model` are described in the module. The summary holds rooms with id, area, label, labels, centroid, neighbours and doors. The model is lod1.js's: rooms with ids `<level>:<room>`, doors placed on the shared boundary by `doorOnBoundary`, exits on the entrance level, and stair and lift links by plan overlap.

## 3. Results against the lab's written rooms

Each written room is matched to the draft room that overlaps it best, by IoU sampled on a 5 cm grid: **found** IoU ≥ 0.8 (the notebook's "found"), **matched** ≥ 0.5 (the notebook's "found or close"), **near** 0.3 to 0.5. Aided runs use the schedule's reading settings: wall and door layers per sheet, block patterns, decoding. Unaided runs use `suggestLayers` on the sheet alone (layers, door blocks, scale), with only the sheet's position given, as a user picking the sheet region would.

| | written | notebook 00: found / with close | aided: found / matched / near | unaided: found / matched / near | extra draft rooms (aided / unaided) |
|---|---|---|---|---|---|
| A, residence | 54 | 30 / 43 | **42 / 49 / 4** | 38 / 45 / 6 | 6 / 6 |
| B, hotel (7 plans) | 194 | 20 / 65 | **57 / 135 / 28** | 53 / 135 / 25 | 147 / 193 |

What each step is worth (aided, found / matched):

| change | A | B |
|---|---|---|
| all steps | 42 / 49 | 57 / 135 |
| no door blocks or swings bridged | 25 / 42 | 15 / 56 |
| no growing to centre-lines | 37 / 48 | 31 / 131 |
| no splitting by names | 42 / 49 | 55 / 127 |
| no outline closing | 42 / 49 | 56 / 131 |
| snap 1 cm, extend 10 cm | 42 / 49 | 43 / 106 |
| snap 6 cm, extend 50 cm | 43 / 49 | 55 / 129 |
| suggested layers (with the schedule's scale) | 38 / 45 | 53 / 135 |

Earlier stages of this work locate the rest. Taking dashed lines as walls, as the notebook does, cut A's rooms along the floor below's outline: 30 found, the notebook's number. Dropping all dashed lines lost the hotel's opening heads: B's matched rooms fell from 81 to 60, until short dashed lines were bridged as openings. Before the leaf test sampled along the chord, a leaf drawn as a thin rectangle in pieces was missed, the wrong chord was bridged, and 46 doors on the typical floor alone had one room on both sides.

**Why the notebook found fewer.**
- Its barriers were both chords of every swing, the open leaf too, found only as polylines of 8 or more points on the listed door layers.
- Its reader turns a top-level block's parts into paths; a nested INSERT appears to have none, so doors inside nested blocks were skipped.
- A closed chord that stops at a door frame leaves a gap that a 3 cm buffer does not close.
- All of these leak rooms into each other: 99 of the hotel's 194 rooms were "merged".
- It took dashed projection lines as walls.
- It kept wall strips thicker than 24 cm as spaces of their own.

**Why the hotel stays hard.** Its sheets are electrical as-built drawings:
- They leave out partitions: a cafeteria, its pantry and a stair are one space on the ground floor.
- Spaces bounded only by columns (an open-air arcade, a ramp) are not traced at all.
- The written schedule merges and splits by judgement: a guest room keeps its entry; a core is split into stair, lift and lobby.
- Site and excavation walls on wall layers make large extra spaces around the basements.

The hotel's 147 extra rooms are mostly closets, entries and service spaces that the schedule folds into rooms. No door to the outside was found on its sheets: the entrances have no door blocks.

**Time** (Intel i7-14700, Node 24, one thread, one run):
- reading: 60 to 140 ms per hotel sheet (5.8 to 9.7 MB), 250 to 390 ms for the 13.4 MB sheet holding all of A's plans;
- drafting a storey: 12 to 100 ms aided for A, 35 to 280 ms for B; `suggestLayers` adds up to 200 ms;
- `npm test`: 9 synthetic tests in 0.2 s.

## 4. Failure modes left

1. Partitions the drawing does not draw. The split by names needs a wall end to cut from.
2. Spaces bounded by columns or grid lines only: arcades, open parking, ramps.
3. The schedule's judgement: entries and wardrobes in rooms, cores split, rooms joined for LOD1.
4. Linework beyond the building on wall layers (site walls, excavation lines): large unnamed extra rooms.
5. Doors with no block, no swing and no dashed head: rooms leak, and exits are not found.
6. A short dashed line that is not an opening (a beam over a corridor) cuts a space in two.
7. Unaided only:
   - wall layers named outside the word lists and drawn without wall pairs;
   - names in the Arabic keyboard font without `Hg` stay encoded;
   - a sheet of several drawings needs its region picked.
8. Walls drawn only as hatches (HATCH is not read) or as single lines with no pair: these work as barriers, but give no thickness to grow by.
9. Stair and lift links by plan overlap only; the flight, landing and direction are not read.

## 5. What the review UI needs

The draft is a proposal. The user must confirm, in order:
1. the sheet region for each storey (a DXF often holds several plans) and the scale;
2. the layers: suggestions pre-ticked, with the linework highlighted on hover;
3. a common point for stacking the storeys;
4. levels (floor level and height).

Then the rooms, with the issues list as the guide:
- no-label, several-labels, large-room, small-room;
- door-inside-room, door-one-side, names-outside-rooms.

The interactions that matter most, by how often the benchmark needed them:
1. **merge** (select two or more rooms): over-segmentation is the largest error by count, 147 extras in the hotel;
2. **split** by a line snapped to wall ends, the cut becoming a door or an opening: for missing partitions;
3. **rename and set class** (13 fixed classes; AI suggestions accepted or rejected per room);
4. **set storey and levels**, and mark doors as exits: none were found on the hotel's sheets;
5. delete rooms (site spaces), draw a room for column-bounded spaces, and add or remove doors.

Edits should be kept by room position, so that re-drafting with other settings does not lose them.
