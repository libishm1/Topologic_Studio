# IFC handling in Topologic Studio Next: a survey against current practice

Items 1 to 5 and 7 have since been built (2026-10-28); section 1 describes the code before them.

Date: 2026-10-27
Status: survey and measurements. No repository code was changed; this file is the only edit.
Scope: the browser IFC path (`topologicpy-web-frontend/src/viewer/ViewerManager.js`, `src/viewer/categories.js`, `src/lib/fragmentCache.js`, `src/core/lattice.js`, `src/workers/sampler.worker.js`, `src/App.jsx`, `vite.config.js`), the Node tools (`tools/ifc-extract.mjs`, `tools/exits.mjs`, `tools/deadline-check.mjs`, `tools/make_example_ifcs.py`) and the backend IFC routes (`topologicpy-web-backend/app/routers/ifc.py`, `app/legacy/contract.py`), on branch `next/browser-core`.
Library versions read: web-ifc 0.0.77, @thatopen/components 3.4.8, @thatopen/fragments 3.4.7, IfcOpenShell 0.8.5 (for cross-checks only).
Builds on: [../roadmap/review-2026-10-27.md](../roadmap/review-2026-10-27.md), [gap-analysis-2026-10-27/deadline-crosscheck.json](gap-analysis-2026-10-27/deadline-crosscheck.json).

Line numbers are those of the working tree on 2026-10-27. `src/App.jsx`, `src/viewer/ViewerManager.js` (after line 874) and `tools/deadline-check.mjs` had uncommitted edits by another session while this survey ran, so their line numbers may drift; function names are given alongside.

Test set. The 19 rows of `deadline-crosscheck.json` contain 18 distinct files: `Building-Architecture.ifc` is byte-identical to `public/examples/buildingsmart-sample-scene.ifc`. All numbers below are over these 18 files (12 in `C:\Users\lmurugesan\ifc-samples`, 4 in `public/examples`, 2 in Downloads). Eight are IFC2X3 and ten IFC4; none is IFC4X3. Fourteen contain at least one IfcDoor.

Method. Every number in this note comes from scripts run with web-ifc 0.0.77 in Node, kept in the session scratchpad (`ifc-survey/semantics.mjs`, `exits.mjs`, `errors.mjs`, `frag.mjs`, `storeys.mjs`, `sidecar.mjs`, `extra.mjs`, `area_wi.mjs`, `area_ios.py`). Browser timings were not measured. buildingSMART's HTML documentation refused automated retrieval (HTTP 403), so property definitions were read from the buildingSMART property set templates shipped with IfcOpenShell 0.8.5 [S5]; entity page URLs are given for reference.

---

## 1. Summary

Studio's IFC path is geometry-first. web-ifc tessellates the model, That Open converts it to fragments, and the walkable lattice is built from slab, stair, door, wall and space triangles. The geometric side is sound on these 18 files: length units are normalised to metres, the model is moved to the origin, every element with a representation produces a mesh, and every door is placed on the graph. The weak side is semantics. The browser reads one IFC attribute in total (`PredefinedType` on coverings, `categories.js:124`), so it cannot tell an exit door from an internal one, a roof from a floor, or one storey from the next except by geometry.

The measurements say this matters.

1. Exits. Pset_DoorCommon flags exits on 5 of the 18 files. Adding IfcRelSpaceBoundary (a door whose boundary is marked EXTERNAL) raises that to 8 of the 14 files with doors, with no false positive against the Pset flags. The geometric fallback used by the Node tools today ("a door with walkable floor on one side only", `tools/exits.mjs`) recovers 27 of the 37 Pset-flagged exits on the five flagged files (73%), and 12 of its 39 picks are not flagged (precision 69%). In the browser there is no exit identification at all: the user picks one exit from a list labelled only by coordinates (`App.jsx:1069-1084`).
2. Roofs. The geometric roof rule (`lattice.js:433-453`) removes only surfaces within 1 m of the model top. On C20-Institute 991 lattice nodes (7.1%) and on ISSUE_159 848 nodes (13.9%) stand on slabs typed `ROOF`, which the browser could exclude with a helper it already has.
3. Storeys. Section-cut levels come from slab geometry (`App.jsx:247`) and collapse to a single level on three files whose floors are modelled as spaces or coverings (dental clinic, Office A, example), although the files declare 4, 3 and 2 storeys.
4. Robustness. A cached fragments file that fails to load is not re-converted (`ViewerManager.js:464-473`), and the cache key ignores library versions. An empty graph is announced as "Graph ready" with 0 nodes. Four files emit between 10 and 1,236 web-ifc geometry errors that the importer silences; a cross-check against IfcOpenShell found no loss in egress-relevant geometry.
5. Performance. IFC to fragments conversion runs on the main thread (0.9 to 2.6 s in Node for 3.9 to 20.3 MB files). Parsing itself is fast (0 to 144 ms per file for `OpenModel`), so multithreaded WASM would buy little.

The eight changes ranked in section 3 address these in order of impact per effort. The first, a semantic sidecar read with web-ifc beside the fragments conversion, costs at most 184 ms per file in Node and feeds exits, roofs, storeys and door attributes at once.

---

## 2. Findings

Impact is on the egress analysis (H high, M medium, L low). Effort is S (under a day), M (a few days), L (a week or more). "Verifiable" says whether the local test models can show the change worked.

| # | Area | Current behaviour (file:line) | Best practice (source) | Gap | Impact | Effort | Verifiable locally |
|---|---|---|---|---|---|---|---|
| 1 | Length units | web-ifc scales geometry by IfcUnitAssignment; lattice thresholds are in metres (`lattice.js:394-409`). Measured: 7 of 18 files are in millimetres, all come out in metres (extents 5 to 162 m). | Read IfcUnitAssignment and convert to SI [S10]. | None for geometry. | none | none | yes (done) |
| 2 | Units for storey elevations | That Open's `getUnitsFactor` (fragments `index.mjs:12394-12420`) knows METRE, FOOT (case-sensitive name) and the MILLI, CENTI, DECI prefixes; no INCH, no general IfcConversionBasedUnit factor. | Use `ConversionFactor` of IfcConversionBasedUnit [S10]. | Imperial files would get wrong absolute storey elevations if storeys are used (strategy 5). | L | S | no (no imperial file) |
| 3 | Large coordinates | `COORDINATE_TO_ORIGIN = true` (`ViewerManager.js:135`); the coordination matrix is read and sent to the server (`ViewerManager.js:745-752`). Largest shift in the set: 505.8 m vertical (ISSUE_159). Fragments drops items beyond `distanceThreshold` 1e5 m (`index.mjs:40035`). | Work near a false origin to keep single-precision geometry exact [S11, S13]. | None observed. No file with kilometre-scale eastings in the set. | L | none | partly |
| 4 | Georeferencing, true north | Not read. Fragments metadata already carries IfcProjectedCRS and IfcMapConversion (`index.mjs:37781-37857`). 3 of 18 files have IfcMapConversion; FZK-Haus has true north at -50 degrees. | IfcMapConversion / IfcProjectedCRS (IFC4), ePset_MapConversion (IFC2X3) [S10, S11, S12]. | Results cannot be placed on a map or aligned with site data. Not needed for routing. | L | S | yes (3 files) |
| 5 | Schema coverage | web-ifc 0.0.77 supports IFC2X3, IFC4, IFC4X3 (`ifc-schema.d.ts:1189-1193`). `categories.js:12-27` covers the deprecated *StandardCase classes too. No IFC4X3 file in the test set. | IFC4X3 deprecates the StandardCase classes [S8]. | Untested on IFC4X3. | L | S | no (add a buildingSMART IFC4X3 sample) |
| 6 | IFC2X3 door types | That Open's default importer keeps IfcDoorType but not IfcDoorStyle (absent from `ifcClasses`, fragments `index.mjs:39534-39860`; only in the category map at line 908). | IFC2X3 type data lives on IfcDoorStyle; IFC4 replaced it by IfcDoorType [S8]. | Type-level Pset_DoorCommon of IFC2X3 files is unreachable from fragments. Measured: Wellness has 4 doors whose IsExternal exists only on the style (all FALSE); no file has TRUE only on the type. | L | S | yes |
| 7 | Property precedence | `tools/ifc-extract.mjs:44-45` ORs instance and type values (`entry.IsExternal ||= ...`). | Pset_DoorCommon is `PSET_TYPEDRIVENOVERRIDE`: the occurrence value overrides the type value [S5]. | An occurrence set to FALSE under a TRUE type would still be an exit. FM_ARC has the reverse case on 8 doors (type FALSE, occurrence TRUE), which the OR happens to get right. | M | S | yes (FM_ARC; plus a synthetic file) |
| 8 | Storeys and levels | Storey ids are collected (`categories.js:26`) and only counted. Section levels come from upward slab faces (`App.jsx:247`, `lattice.js:877-911`). Measured: levels match declared storeys within 0.3 m on 9 files, but give one level for dental (4 storeys), Office A (3) and example (2). | Use IfcBuildingStorey with its absolute elevation; fragments already computes it (`replaceStoreyElevation`, `index.mjs:40002`) [S10]. | Section cut and per-storey reporting are wrong on 3 of 18 files. | M | S | yes |
| 9 | Elements outside storeys | Not checked; the graph is geometric, so unaffected. Measured: HouseZ has 4/15 doors, 42/140 walls, 10/18 slabs, 6/10 stairs not reachable from any storey by containment or aggregation; buildingSMART scene 2/3 slabs. | IfcRelContainedInSpatialStructure / IfcRelAggregates [S10]. | Storey-based features need a height fallback. | L | S | yes |
| 10 | Multi-building, multi-site | Single model, `autoCoordinate = false` (`ViewerManager.js:444-446`, components `index.mjs:11500`). Two files have two IfcSite. | Federate with a shared coordination matrix. | No federation of architecture plus structure. | L | M | partly |
| 11 | Exit identification | Browser: none; one user-picked exit (`App.jsx:1069-1084`); the in-progress forecast code passes `exits: [snap(graph, points.exit)]`. Node: Pset flags, else one-sided door, else room on one side (`tools/exits.mjs`). | Pset_DoorCommon.FireExit / IsExternal [S5, S6]; IfcRelSpaceBoundary.InternalOrExternalBoundary = EXTERNAL [S7]; semantic models for evacuation checking [S21, S22, S23]. | Space boundaries unused; geometric fallback 73% recall, 69% precision against Pset flags. | H | M | yes |
| 12 | Door operation, width, swing | Not read. Measured: OverallWidth on every door in 13 of 14 door-bearing files; OperationType on 11 (one REVOLVING door in C20, one DOUBLE_DOOR_SLIDING in FZK). | IfcDoor.OperationType / IfcDoorTypeOperationEnum [S8]; capacity from effective width in the hydraulic model [S24]. | No door capacity or door-type rule; revolving doors would be treated as ordinary exits. | M | S | yes |
| 13 | Openings and fills | Not read; door voids are found geometrically from the 0.75 m section (`lattice.js:418-431`). IfcRelFillsElement present for most doors in 11 files. | IfcRelVoidsElement / IfcRelFillsElement [S10]. | Host wall of a door unknown. Host-wall IsExternal is over-inclusive (27 doors on FM_ARC against 8 Pset exits; 34 of 47 on ISSUE_159), so it is a tie-breaker at most. | L | S | yes |
| 14 | Spaces and adjacency | IfcSpace bottoms are sampled as floor (`lattice.js:469`); spaces hidden in the view (`ViewerManager.js:489-499`). Space boundaries unused. 7 files have IfcRelSpaceBoundary (81 to 3,124 relations). | Room adjacency through doors from space boundaries [S7, S23]. | No room graph, no room-level results. | M | M | yes (7 files) |
| 15 | Zones, compartments, fire ratings | Not read. Measured: IfcZone in 1 file; Pset_WallCommon.FireRating on 2 files (38 and 40 walls); Pset_DoorCommon.FireRating present on 6 files. | IfcZone / IfcSpatialZone, Pset_SpaceFireSafetyRequirements [S5]. | Fire model ignores compartment boundaries. Data too sparse in the set to rely on. | L | M | barely |
| 16 | Stairs and ramps | Geometry only (`categories.js:18`); one walking speed. Pset_StairFlightCommon (RiserHeight, TreadLength, NumberOfRiser) on 6 files. | Stair speed depends on riser and tread [S24]. | Timed egress overestimates stair speed. | M | M | yes (6 files) |
| 17 | Roof slabs | Geometric rule: surfaces within 1 m of the model top (`lattice.js:433-453`). Measured: ROOF-typed slabs carry 991 nodes on C20 (13,984 to 12,993; parts 6 to 4) and 848 on ISSUE_159 (6,101 to 5,253; parts 35 to 34). | IfcSlab.PredefinedType ROOF and IfcRoof aggregation [S9]. | Lower roofs and terraces enter the walkable graph. | M | S | yes |
| 18 | Coverings, railings, plates | FLOORING coverings join floors (`ViewerManager.js:567-571`); railings, columns, windows, members, curtain walls are obstacles (`categories.js:25`). | As implemented. | None. | none | none | yes |
| 19 | Occupant load | Not read. Measured: Pset_SpaceOccupancyRequirements absent from all 18 files. | OccupancyNumber / AreaPerOccupant [S5]. | Must fall back to area-based loads. | L | S | no data |
| 20 | Cache correctness | Key is the IFC SHA-256 only (`fragmentCache.js:65-82`, `DB_VERSION = 1` at line 15). A failing cached load throws to the user with no re-conversion (`ViewerManager.js:464-473`). No `navigator.storage.persist()`. | Version cache entries with the producing library and settings; persistent storage [S4, S20]. | After a fragments upgrade or an importer-settings change the user gets stale or unreadable cached models. | M | S | in the browser suite |
| 21 | Silent geometry errors | The importer sets web-ifc to `LOG_LEVEL_OFF` (fragments `index.mjs:37271, 38370`). Measured: FM_ARC 1,236, ISSUE_159 1,074, ISSUE_021 81, Office A 10 error messages. Per-element mesh area matches IfcOpenShell within 10% except 15 walls (FM_ARC 8, ISSUE_159 7) whose window openings web-ifc left uncut; none hosts a door. | Report geometry problems to the user; validate [S16]. | Not visible; no measured egress impact on these files. | L | S | yes |
| 22 | Empty or degenerate models | 0 nodes is reported as "Graph ready" with a success toast (`App.jsx:430-436`; `graphIsSplit` is false for 0 components, `core/graph.js:63`). Causes in the set: no doors (example, buildingSMART scene: a graph but no exit), structural-only (ISSUE_021, a Tekla Structures file whose 24 slabs have 0.4 m2 of upward area in total, confirmed with IfcOpenShell; Building-Structural, no slabs). | Pre-flight content checks, IDS [S15, S16]. | User is not told why. | M | S | yes |
| 23 | File formats | `.ifc` only (`App.jsx:191-196`, `ui/Viewport.jsx:122`). IFCZIP rejected. | IFC-SPF, IFC-ZIP and IFC-XML are the standard exchange formats [S14]. | Zipped deliveries must be unzipped by hand. | L | S | no (no ifczip; can be made) |
| 24 | Size and memory | No client-side size guard. Server caps uploads at 256 MiB (`config.py:66`). web-ifc defaults `MEMORY_LIMIT` 2 GiB, `TAPE_SIZE` 64 MiB, `CIRCLE_SEGMENTS` 12 (`web-ifc-api.js:71501-71505`). Measured in Node: conversion of the 20.3 MB file raised RSS by 264 MB; the importer opens the model twice (geometry and properties processors). | Warn before loading files likely to exhaust a 32-bit WASM heap; offer a server path. | Large files fail with an engine error rather than an explanation. | L | S | partly (largest file 20.3 MB) |
| 25 | Validation | None before analysis. | IDS 1.0 (buildingSMART standard since 1 June 2024) checked with ifctester; buildingSMART validation service [S15, S16]. | Missing egress data is discovered late or never. | M | M | yes |
| 26 | Dead setting | `ifcLoader.settings.webIfc.OPTIMIZE_PROFILES = true` (`ViewerManager.js:136`). web-ifc 0.0.77 `LoaderSettings` has no such key (`web-ifc-api.d.ts:37-50`); That Open comments it out (`index.mjs:11364`). | Remove. | Misleading only. | none | S | n/a |
| 27 | Conversion thread | `IfcLoader.load` runs `new FRAGS.IfcImporter().process()` on the calling thread (components `index.mjs:11497-11512`). Measured in Node: 0.9 s (Office A, 3.9 MB) to 2.6 s (Wellness, 20.3 MB); output fragments 0.24 to 1.66 MB, about ten times smaller than the IFC. | Convert in a worker or once on a server; That Open recommends converting once and reusing `.frag` [S4]; transfer buffers [S19]. | Tab freezes during conversion; cancellation only between phases (`ViewerManager.js:434, 475`). | M | M | timing yes, UI in browser |
| 28 | Multithreaded WASM | Off by design (`vite.config.js:35-46`, which documents the crash of the pthread build). GitHub Pages cannot send COOP/COEP. Measured `OpenModel`: 0 to 144 ms per file. | Threads need cross-origin isolation [S17]; on Pages only via a service-worker shim [S18]. | Small gain, real risk. Leave off. | L | M | n/a |
| 29 | Loading only needed types | Browser converts all classes (needed for display) and extracts per category (`ViewerManager.js:573-580`). Node tool streams only graph types (`ifc-extract.mjs:75`): 1.1 to 23 times faster than streaming all meshes. | Stream by type where display is not needed [S1]. | None for the browser; an analysis-only path could skip display classes. | L | M | yes |
| 30 | Disposal | `clearModel` disposes the fragments model (`ViewerManager.js:538-553`); the importer cleans its own web-ifc instance. | `CloseModel` / `Dispose` [S1]. | None. | none | none | browser |
| 31 | Element identity | Door meshes carry `itemId` (`categories.js:72-76`, `lattice.js:864-865`). Fragments local ids are the IFC express ids (`index.mjs:37389-37392`). | Join on express id or GlobalId. | None: semantics read with web-ifc join directly with lattice door ids. | none | none | yes |
| 32 | Example files | The generator writes IfcDoor without Pset_DoorCommon, openings or spaces (`tools/make_example_ifcs.py:155-165`). | Pset_DoorCommon on doors [S5]. | Two shipped examples cannot demonstrate semantic exits. | L | S | yes |
| 33 | Other production practice | No element inspection on click, no highlight of doors by GUID, no BCF export of model problems, no saved analysis keyed by GlobalId. | BCF and IDS round trips with the authoring tool [S15, S21]. | Findings cannot be sent back to the modeller. | L | M | partly |

---

## 3. The eight strategies to implement, ranked by impact per effort

### 3.1 A semantic sidecar and an exit cascade (impact H, effort M)

What. Read the IFC once more with web-ifc, without geometry, and return a small JSON sidecar: schema, length unit, storeys, and for each door its express id, GlobalId, Name, storey, OperationType, OverallWidth, Pset_DoorCommon IsExternal and FireExit (occurrence overriding type), and its space boundaries. Exits follow a cascade, each exit recording its source: FireExit or IsExternal TRUE; else a door space boundary with `InternalOrExternalBoundary = EXTERNAL`; else the existing geometric rules in `tools/exits.mjs`. An explicit IsExternal FALSE counts as no information when no door in the file is TRUE (ISSUE_159 sets FALSE on all 47 doors, tested_sample on its only door).

Why fragments alone will not do. The default importer keeps IfcRelDefinesByProperties and IfcRelDefinesByType (fragments `index.mjs:39940-39985`) but not IfcRelSpaceBoundary, and fragments stores relations as item links, so the boundary's `InternalOrExternalBoundary` attribute would be lost even if the relation were added through `instanceCallback`. IfcDoorStyle is also not imported (finding 6). A web-ifc pass has neither limitation, and its express ids are the fragments local ids, so no GUID map is needed.

Where.
- New pure module `topologicpy-web-frontend/src/core/ifcSemantics.js` taking `(api, modelID, WEBIFC)`; a working prototype is `ifc-survey/sidecar.mjs` in the session scratchpad. It builds three maps in one pass each: IfcRelDefinesByProperties (object to property sets), IfcRelDefinesByType (object to type, then `HasPropertySets`), IfcRelSpaceBoundary (element to spaces and the EXTERNAL flag). APIs: `GetLineIDsWithType`, `GetLine`, `GetGuidFromExpressId`, `CloseModel`.
- Browser: run it in the import worker of strategy 3.7. Until that exists, call it after conversion in `ViewerManager.loadIfc` (after line 473) on `this.ifcLoader.webIfc` opened with `readIfcFile(bytes)`, then `CloseModel`. Cache the JSON next to the fragments under the same key (`fragmentCache.js`).
- Graph: pass `exitIds` into `buildLatticeGraph` and mark exit door nodes in `doorReport` (`lattice.js:628-674`); `App.jsx` `doorList` (1069-1084) labels doors by Name, storey and exit source; the Route step and the forecast default to the full exit set instead of one picked exit.
- Node: replace `doorFlags` in `tools/ifc-extract.mjs:35-64` with the shared module, which also fixes the OR merge (finding 7); `tools/exits.mjs` takes the cascade.
- Examples: write Pset_DoorCommon (IsExternal, FireExit on the entrance doors) in `tools/make_example_ifcs.py:155-165`.

Cost measured in Node: `OpenModel` 0 to 144 ms plus 0 to 115 ms of reading per file, at most 184 ms in total (dental clinic, 254 doors).

Test across the 18 files (extend `tools/deadline-check.mjs`, record `exitSource` per file):
- semantic exits on 8 files (dental 18, FM_ARC 8, Office A 6, Duplex 4, Wellness 1 by Pset; FZK-Haus 2, ISSUE_005 2, C20 1 by space boundary), against 5 today;
- on the four files that have both, the space-boundary set is a subset of the Pset set (12 of 18, 8 of 8, 6 of 6, 4 of 4);
- after the generator change, courtyard-ring and office-two-storey also report Pset exits (10 of 18);
- a unit test on a small IfcOpenShell-made file where the type says TRUE and the occurrence FALSE expects no exit.

### 3.2 Exclude ROOF slabs by type (impact M, effort S)

What. Remove slabs whose `PredefinedType` is ROOF, and slabs aggregated into an IfcRoof, from the floor group before sampling; keep the geometric rule for untyped files; offer a toggle for accessible roofs.

Where. `ViewerManager.extractEgressGeometry` (559-591): `const roofs = await withPredefinedType(this.model, ids.floors, "ROOF")` using the existing helper (`categories.js:124-135`); IfcRoof children through `model.getItemsChildren` (IfcRelAggregates is imported by default). Return them as a `roofs` group so the worker can add them on request. Mirror it in `tools/ifc-extract.mjs` (`GROUPS.floors`, line 17) with `GetLine(...).PredefinedType`.

Test (`tools/lattice-batch.mjs` over the 18 files): C20 13,984 to 12,993 nodes and 6 to 4 parts; ISSUE_159 6,101 to 5,253 nodes and 35 to 34 parts; FZK-Haus, ISSUE_005, FM_ARC, Duplex, IfcOpenHouse and the buildingSMART scene unchanged (their ROOF slabs are already removed by slope or by the 1 m rule).

### 3.3 Pre-flight report and actionable empty-graph messages (impact M, effort S)

What. On load, show what the file offers for egress and what is missing: schema, originating application, length unit, storeys, counts of slabs, spaces, doors, stairs, exits by source, elements outside storeys, and the web-ifc error count. Replace "Graph ready, 0 nodes" with a warning that names the cause.

Where. Data from the sidecar (3.1), `collectCategoryIds` and `model.getMetadata()` (fragments stores schema, FILE_NAME strings and CRS, `index.mjs:37766-37779`). UI in `ui/Panels.jsx` `ModelPanel`. In `App.jsx` `buildGraph` (toast at 430-436), branch on `stats.nodes === 0`. Optional backend endpoint `POST /api/ifc/validate` in `routers/ifc.py` running ifctester against a short Studio egress IDS (IfcDoor requires Pset_DoorCommon.IsExternal; IfcBuildingStorey present; IfcSlab PredefinedType set; IfcSpace present), returning JSON or BCF [S15, S16].

Test. A table-driven Node test over the 18 files expecting one warning code each where applicable: no IfcDoor (example, buildingSMART scene), structural-only with no walkable floor (ISSUE_021, Building-Structural), floors only as spaces or coverings (dental, Office A), doors without exit flags (HouseZ, ISSUE_159, IfcOpenHouse, tested_sample, and the two generated examples until 3.1 changes them), elements outside storeys (HouseZ, buildingSMART scene).

### 3.4 Version the fragments cache and recover from a bad entry (impact M, effort S)

What. Key cache entries by the IFC hash plus the fragments version, the web-ifc version and a hash of the importer settings (classes and relations added, `COORDINATE_TO_ORIGIN`); on a failed cached load, delete the entry and convert again; request persistent storage once. This is a precondition for 3.1, 3.2 and 3.7, each of which changes what a cached model must contain.

Where. `src/lib/fragmentCache.js` (`hashFile`, 65-82; `readCached`, 84-95): a `cacheKey(sha, meta)` helper, with versions written by `scripts/sync-assets.mjs`, which already copies the matching worker and WASM. `ViewerManager.loadIfc` (464-473): `try { fragments.core.load(...) } catch { await evict(key); /* fall through to conversion */ }`. `navigator.storage.persist()` after the first successful write [S20].

Test. Node unit test of the key function (different versions give different keys). Browser suite step: write a corrupt buffer under the Duplex's key, load the Duplex, expect a conversion and a success toast instead of "Could not load the IFC file".

### 3.5 Storey-aware levels, door grouping and per-storey results (impact M, effort S)

What. Take levels from IfcBuildingStorey and fall back to slab geometry only when storeys are missing or unusable; group the door list and the results by storey.

Where. `model.getItemsData(ids.storeys, { attributesDefault: false, attributes: ["Name", "Elevation"] })`. With the importer's default `replaceStoreyElevation` the Elevation is absolute and in metres (`index.mjs:40002`); in the viewer frame the level is that elevation plus the y translation of the coordination matrix the app already reads (`ViewerManager.js:745-752`). Merge in `App.jsx` at the `setLevels(floorLevels(...))` call (247). For elements outside any storey (HouseZ) assign by height. Check imperial units first (finding 2).

Test. The scratch script `ifc-survey/storeys.mjs` computes both: dental gives 4 levels (-1.0, 0.0, 4.57, 9.25 m) instead of [2.15]; Office A 3 (0.0, 4.27, 7.92) instead of [2.13]; example 2 (0.13, 3.27) instead of [0.06]; FZK-Haus, C20, HouseZ, office-two-storey, Wellness and the buildingSMART scene agree with the geometry within 0.3 m.

### 3.6 Door egress attributes on the graph (impact M, effort S to M)

What. Attach OverallWidth, OperationType and Name to door nodes. Use width as a capacity for timed egress (flow per unit effective width, as in the hydraulic model [S24]); flag revolving, sliding and folding doors as exits needing the user's confirmation rather than silently accepting them; compare OverallWidth with the width the lattice measures from the door mesh (`doorWaypoints`, `lattice.js:821-868`) to catch doors whose geometry and attributes disagree.

Where. Sidecar (3.1) to `doorReport` by `itemId`; `core/deadline.js` and `core/forecast.js` read capacities when present.

Test. OverallWidth present on every door of 13 of the 14 door-bearing files (HouseZ has none) and OperationType on 11; C20's single REVOLVING door is flagged. Acceptance to measure: mesh width within 0.2 m of OverallWidth on at least 90% of doors per file, with the outliers listed.

### 3.7 Convert and read semantics in a dedicated worker (impact M, effort M)

What. Move `IfcImporter.process` and the sidecar off the main thread, transfer the result back, and load it through the path that cached models already use.

Where. New `src/workers/ifc-import.worker.js`: `import { IfcImporter } from "@thatopen/fragments"` and `import * as WEBIFC from "web-ifc"`; set `importer.wasm = { path: WASM_PATH, absolute: true }` and `importer.webIfcSettings = { COORDINATE_TO_ORIGIN: true }`; `const frag = await importer.process({ bytes, raw: false, progressCallback: (p, info) => postMessage({ p, info }) })`; then a fresh `WEBIFC.IfcAPI`, `Init(undefined, true)` (force single thread, `web-ifc-api.d.ts:212`), `OpenModel`, `ifcSemantics`, `CloseModel`; `postMessage({ frag, semantics }, [frag.buffer])`. Send the IFC buffer as a transferable after hashing [S19]. In `ViewerManager.loadIfc` replace `this.ifcLoader.load` (444-449) by the worker call followed by `this.fragments.core.load(frag, { modelId })` as at 466-471. Cancel with `worker.terminate()` on `signal`.

Test. Browser suite on Wellness (20.3 MB): a `PerformanceObserver` for `longtask` sees no task over 200 ms during conversion (today the conversion is one or more multi-second tasks; 2.6 s in Node), and cancelling mid-conversion leaves the viewer usable. Node: the worker module's conversion output loads with `FragmentsModels` and gives the same category counts as today for all 18 files.

### 3.8 A room graph from IfcSpace and door space boundaries (impact M, effort M)

What. One node per IfcSpace, one edge per door that bounds two spaces, exit edges for EXTERNAL door boundaries; lattice nodes assigned to rooms by footprint. This is the two-graph representation proposed in the review (section 6, item 4) and in the BIM navigation literature [S23]; it gives room-level results (rooms with no route out), a much smaller RL state space, and a consistency check between the IFC's declared adjacency and the lattice's geometric connectivity.

Where. Sidecar (3.1) adds spaces (id, Name, storey) and door-to-spaces; new `src/core/rooms.js`; room assignment from the IfcSpace bottoms the lattice already samples (`lattice.js:469`).

Test. On the 7 files with space boundaries (FZK-Haus, ISSUE_005, C20, dental, FM_ARC, Office A, Duplex): every pair of spaces joined by a door in the IFC lies in one lattice component with walls on; mismatches are listed per file as model or lattice defects.

Not ranked in the top eight: IFCZIP support (S, low impact; unzip in the import worker with a small inflate library), georeferencing display (S, low), stair speed from Pset_StairFlightCommon (M, useful only once 3.6 brings capacities and timing), IDS validation beyond the minimal rule set of 3.3 (M), multithreaded WASM (finding 28: leave off).

---

## 4. Checked and found fine

- Length units. Seven of the 18 files are in millimetres; web-ifc returns metres for all of them (model extents 5 to 162 m), so the lattice's metre thresholds hold.
- Origin shift. `COORDINATE_TO_ORIGIN` is on and the coordination matrix is used for the server cross-check (`ViewerManager.js:135, 745-752`). The largest shift in the set (505.8 m vertical, ISSUE_159) is removed; single-precision positions near the origin are adequate at these extents.
- Geometry completeness. Every wall, slab, door, stair, stair flight, ramp, space and covering with a representation produced a mesh: 0 failures among 4,364 elements across the 18 files.
- Geometry agreement. Per-element mesh area from web-ifc matches IfcOpenShell 0.8.5 within 10% for all 1,526 elements of FZK-Haus, Office A, FM_ARC and ISSUE_159, except 15 walls with uncut window openings; no affected wall hosts a door. The web-ifc error messages on these files therefore did not cost egress geometry.
- ISSUE_021 is not a geometry failure. Its 24 slabs have 0.4 m2 of upward area in total in both web-ifc and IfcOpenShell: it is a structural model, which a pre-flight message should say (3.3).
- Doors on the graph. Every door in every file gets a graph node (`doorReport` node index at least 0 for all doors).
- Doorways without IfcDoor. The section at 0.75 m finds voids whether or not a door fills them (`lattice.js:418-431`).
- Floors from spaces and coverings. IfcSpace bottoms (Office A, dental) and FLOORING coverings are walkable; ceilings are excluded (`ViewerManager.js:567-571`, `categories.js:14-16`).
- Element identity. Fragments local ids equal IFC express ids (`index.mjs:37389-37392`), so door ids join with semantic data directly.
- Worker hand-off. Geometry goes to the sampler worker as transferable buffers and the graph comes back the same way (`App.jsx` `buildGraph`, `workers/sampler.worker.js:48-55`).
- Fragments cache. Converted models are cached in IndexedDB with LRU eviction under a 512 MB budget (`fragmentCache.js:19-21`), as That Open recommends [S4]; only the key and the failure path need work (3.4).
- Self-hosted assets. WASM and the fragments worker are served under the GitHub Pages base path, with a CDN fallback only for the worker (`ViewerManager.js:27-36`, `resolveWorkerUrl` at 1150).
- Single-threaded WASM. The decision and its reason are documented (`vite.config.js:35-46`); measured parse times (at most 144 ms for `OpenModel`) leave little for threads to gain.
- Disposal. Loading a new model disposes the previous fragments model and its overlays (`ViewerManager.js:538-553`).
- Backend IFC import. `/upload-ifc` uses IfcOpenShell with world coordinates (`legacy/contract.py:639`), which returns SI units; `/api/ifc/navigation-check` enforces the 256 MiB limit (`routers/ifc.py:306-309`, `config.py:66`).

Not verified here: IFC4X3 files, imperial-unit files, IFCZIP, files above 20.3 MB, kilometre-scale georeferenced coordinates, and any browser timing.

---

## 5. Sources

[S1] ThatOpen, engine_web-ifc 0.0.77, `web-ifc-api.d.ts` lines 37-50 (LoaderSettings) and 212 (`Init(customLocateFileHandler, forceSingleThread)`); defaults in `web-ifc-api.js` lines 71501-71505. https://github.com/ThatOpen/engine_web-ifc

[S2] ThatOpen, engine_components 3.4.8, `dist/index.mjs` lines 11353-11366 (IfcFragmentSettings) and 11497-11512 (`IfcLoader.load`). https://github.com/ThatOpen/engine_components

[S3] ThatOpen, engine_fragment 3.4.7, `dist/index.mjs`: IfcImporter defaults 39897-40035, default classes 39534-39860, log level 37271 and 38370, units 12394-12420, CRS metadata 37781-37857, local ids 37389-37392. https://github.com/ThatOpen/engine_fragment

[S4] That Open documentation, IfcLoader tutorial ("do that conversion once, save the resulting .frag file"). https://docs.thatopen.com/Tutorials/Components/Core/IfcLoader

[S5] buildingSMART property set templates for IFC4X3 as distributed with IfcOpenShell 0.8.5, `ifcopenshell/util/schema/Pset_IFC4X3.ifc` (Pset_DoorCommon with template type PSET_TYPEDRIVENOVERRIDE and the IsExternal, FireExit, HandicapAccessible, FireRating definitions; Pset_StairFlightCommon; Pset_SpaceFireSafetyRequirements; Pset_SpaceOccupancyRequirements). https://github.com/IfcOpenShell/IfcOpenShell/tree/v0.8.0/src/ifcopenshell-python/ifcopenshell/util/schema

[S6] buildingSMART, IFC 4.3, Pset_DoorCommon. https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/Pset_DoorCommon.htm

[S7] buildingSMART, IFC4, IfcRelSpaceBoundary. https://standards.buildingsmart.org/IFC/RELEASE/IFC4/FINAL/HTML/schema/ifcproductextension/lexical/ifcrelspaceboundary.htm ; IFC 4.3 space boundary concept. https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/concepts/Object_Connectivity/Space_Boundaries/Space_Boundaries_1st_Level/content.html

[S8] buildingSMART, IFC 4.3, IfcDoor and IfcDoorTypeOperationEnum. https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcDoor.htm ; https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcDoorTypeOperationEnum.htm ; IFC4 IfcDoorStyle (deprecated). https://standards.buildingsmart.org/IFC/RELEASE/IFC4/ADD2_TC1/HTML/schema/ifcarchitecturedomain/lexical/ifcdoorstyle.htm

[S9] buildingSMART, IFC 4.3, IfcSlabTypeEnum. https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcSlabTypeEnum.htm

[S10] buildingSMART, IFC 4.3, IfcUnitAssignment, IfcBuildingStorey, IfcRelContainedInSpatialStructure, IfcRelFillsElement, IfcMapConversion. https://ifc43-docs.standards.buildingsmart.org/IFC/RELEASE/IFC4x3/HTML/lexical/IfcUnitAssignment.htm ; .../IfcBuildingStorey.htm ; .../IfcRelContainedInSpatialStructure.htm ; .../IfcRelFillsElement.htm ; .../IfcMapConversion.htm

[S11] buildingSMART International and buildingSMART Australasia, User Guide for Geo-referencing in IFC, version 2.0 (2020). https://ucm.buildingsmart.org/use-case-details/2047/en

[S12] Noardo, F., et al. (2020). Tools for BIM-GIS integration (IFC georeferencing and conversions): results from the GeoBIM benchmark 2019. ISPRS International Journal of Geo-Information 9(9), 502. https://doi.org/10.3390/ijgi9090502

[S13] Bonsai (formerly BlenderBIM) documentation, Georeferencing (false origin and single-precision limits). https://docs.bonsaibim.org/guides/authoring/georeferencing.html

[S14] buildingSMART, IFC formats (IFC-SPF, IFC-XML, IFC-ZIP). https://technical.buildingsmart.org/standards/ifc/ifc-formats/

[S15] buildingSMART, Information Delivery Specification (IDS 1.0, approved 1 June 2024). https://www.buildingsmart.org/standards/bsi-standards/information-delivery-specification-ids/ ; https://github.com/buildingSMART/IDS

[S16] IfcOpenShell, ifctester (IDS checking). https://pypi.org/project/ifctester/ ; buildingSMART validation service. https://validate.buildingsmart.org/

[S17] MDN, SharedArrayBuffer security requirements. https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer#security_requirements ; web.dev, Making your website cross-origin isolated using COOP and COEP. https://web.dev/coop-coep/

[S18] GitHub community discussion 13309, custom COOP/COEP headers on GitHub Pages. https://github.com/orgs/community/discussions/13309 ; coi-serviceworker. https://github.com/gzuidhof/coi-serviceworker

[S19] MDN, Transferable objects. https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Transferable_objects

[S20] MDN, Storage quotas and eviction criteria; StorageManager.persist(). https://developer.mozilla.org/en-US/docs/Web/API/Storage_API/Storage_quotas_and_eviction_criteria ; https://developer.mozilla.org/en-US/docs/Web/API/StorageManager/persist

[S21] Eastman, C., Lee, J., Jeong, Y., Lee, J. (2009). Automatic rule-based checking of building designs. Automation in Construction 18(8), 1011-1033. https://doi.org/10.1016/j.autcon.2009.07.002

[S22] Choi, J., Choi, J., Kim, I. (2014). Development of BIM-based evacuation regulation checking system for high-rise and complex buildings. Automation in Construction 46, 38-49. https://doi.org/10.1016/j.autcon.2013.12.005

[S23] Isikdag, U., Zlatanova, S., Underwood, J. (2013). A BIM-Oriented Model for supporting indoor navigation requirements. Computers, Environment and Urban Systems 41, 112-123. https://doi.org/10.1016/j.compenvurbsys.2013.05.001

[S24] Gwynne, S. M. V., Rosenbaum, E. R. (2016). Employing the hydraulic model in assessing emergency movement. In SFPE Handbook of Fire Protection Engineering, 5th ed., Springer, pp. 2115-2151; as cited and discussed in Fire Technology (2022). https://link.springer.com/article/10.1007/s10694-022-01244-z
