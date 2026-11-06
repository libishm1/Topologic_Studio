# Handoff: Topologic Studio Next (2026-11-06, afternoon)

## State

- **Live:** https://libishm1.github.io/Topologic_Studio/ is `origin/main` = `next/browser-core` at 7e009e6 (the exits out of the building and rerouting to any of them), with README commits fbdaed5 and 7130fac, which need no deploy.
  - **Live check of 7e009e6:** 0 failed steps, with Render awake; before it, the suite passed on the dev server and on the Pages build, desktop and mobile.
  - **Live check of 3f9a4f2:** 0 failed steps, with Render awake.
  - **Deployed 2026-11-06:**
    - 34d036b, the TU Wien Vienna example;
    - 17cf2ff, the fix for the blank band;
    - d0d1749, field notes and hints, and the doors switch;
    - 8852fe2, the human check (Turnstile), which stays off until its keys are set;
    - 3f9a4f2, the measures.
  - **Live check of 4cfcefb:** 0 failed steps.
    - The rooms (LOD1) step passed live: the plan opened in 0.3 s and its graph joined both storeys.
    - **The first attempt** failed only the TopologicPy cross-check: the Render backend did not answer `/health` for about ten minutes.
    - **Once it answered,** the rerun passed in full; the cross-check took 69 s.
  - **Deployed this evening:**
    - da7064f, "Describe this model";
    - d9afc1c, the code check (live check 0 failed steps);
    - 96170e4, the Worker's room suggestions (the page's client only; the Worker is not deployed);
    - eea6d1d, rooms (LOD1) models;
    - 4cfcefb, the drawing core (no screen yet).
  - **Deploy route:** fast-forward `origin/main` to `next/browser-core` (`git merge-base --is-ancestor origin/main HEAD`, then `git push origin HEAD:main`). The local `main` is a stale ref.
  - **Watching the deploy:** `gh` is not installed. Read the runs from the public API at `https://api.github.com/repos/libishm1/Topologic_Studio/actions/runs?per_page=4`.
- **Uncommitted on purpose:**
  - `wiki/research/free-inference-gpu-options-2026-11-05.md`: its drafts speak in the user's voice and the repository is public. The user decides.
- **The drawing-to-LOD1 core is committed but has no screen yet:**
  - `src/core/drawingToLod1.js`, its 9 tests, `tools/dxf-lod1-check.mjs`;
  - the method, results and failure modes: `wiki/research/dxf-floor-plans-to-lod1-2026-11-05.md`.
- **Gates before any deploy:**
  - `npm test` in `topologicpy-web-frontend`: 259 tests;
  - `npx eslint .`;
  - the Pages build: `MSYS_NO_PATHCONV=1 BASE_PATH=/Topologic_Studio/ npm run build` from Git Bash;
  - `node tools/browser-test.mjs --url http://localhost:4173/Topologic_Studio/`, desktop then `--mobile`, against `vite preview` on that sub-path;
  - the live check after the deploy;
  - the Worker's own tests: `node --test test/index.test.mjs` in `cloudflare/describe-worker` (8 tests);
  - after any solver change, `node tools/flamescapes-gpu-check.mjs --phase full --quick` (CPU and GPU agreement, now including soot, CO2 and O2) and `--phase timing`.
- **Lean:** unchanged this evening; 28 modules, 1297 theorems.

## Resume here (2026-11-06)

0. **In order, as the user asked on 2026-11-06:**
   1. **Cloudflare: the user's steps, then the deploy.**
      - **Plugin:** Cloudflare's Claude Code plugin is installed for the user (`claude plugin marketplace add cloudflare/skills`, `claude plugin install cloudflare@cloudflare`), as the user asked from developers.cloudflare.com/agent-setup. It was reviewed first: it is from the Cloudflare organisation, has no hooks, and has one MCP server, `https://mcp.cloudflare.com/mcp`. The optional `cf` CLI was not installed: the page says to ask first.
      - **Needs the user:**
        - run `/reload-plugins`;
        - approve Cloudflare's OAuth on first use;
        - or run `npx wrangler login`.

        Never sign in for them.
      - **Then:**
        - deploy the Worker (`npx wrangler deploy` in `cloudflare/describe-worker`, or through the MCP);
        - set the repository variable `DESCRIBE_URL`;
        - create a Turnstile widget (Managed, hostname `libishm1.github.io`);
        - `npx wrangler secret put TURNSTILE_SECRET`;
        - set the repository variable `TURNSTILE_SITE_KEY`;
        - rerun Deploy Frontend.

        The Worker's README has the steps.
   2. **Smoke and gas measures:** done (3f9a4f2). See This round and `wiki/research/tenability-index-2026-11-06.md`.
   2b. **Asked for during the video (2026-11-06), not built:**
      - **Shelter areas outside the building:** identify safe assembly areas outside when there is a fire. Read the guidance first (distance from the facade, clear of exits and fire-service access, upwind), then propose the method before building it.
      - **"Nearest exit" stopped at a flat's own door in the Vienna model:** fixed this evening (see This round). The code check still counts every flagged exit door; whether it should count only the exits out of the building (with stairs as the upper storeys' exits) is open.
      - **Ignition by context:** the video picked the fire's context (cooking, electrical, gas) by hand from the ranked list. A "Context" choice in Likely ignition points would make this one click.
   3. **Questions answered but not decided:**
      - fire replay from a cache (IndexedDB, keyed by model, graph and fire inputs);
      - door states: an "as built" mode from SelfClosing and "-C" fire ratings, a seeded random mode, and later an ensemble.
   4. **Training** (IFCNet retraining, on Nebius, Vast.ai or a free offer): later, by the user's choice.
      - **Wanted in the data:** the user asked to use "all of these models", the list on github.com/bimdata/BIMData-Research-and-Development, page `pages/IFC_FILES.md`. It holds 100 IFC files (3.7 GB, 40 models) from seven sources: DURAARK at TIB, the Open IFC Model Repository, BIMcollab examples, Schependomlaan, IFCCompressor, NIBS common BIM files and markusplassen.no. The repository itself has no licence and holds screenshots only.
      - **Check each source's licence first.**
        - `NBS_Lakeside` is an NBS model: excluded by the user's standing rule.
        - The `Autodesk_*` models need Autodesk's terms read first.
        - The Open IFC Model Repository states no licences.
        - DURAARK is licensed dataset by dataset.
        - Several models are already in the local corpus: Duplex, FZK-Haus, Smiley West, HiTOS.
      - **Also wanted: Sand3D** (Sandflow, sandflow.ai/sand3d; the user asked 2026-11-06).
        - **What it is:** about 100K single 3D objects, each written as a Python program in the Sand3D language that builds a mesh (for glTF, Blender, three.js): furniture, lighting, vehicles, toys, garden and outdoor, sports, articulated objects (10K+ with 36K joints) and mechanical parts. No buildings, rooms or IFC. Counts are Sandflow's own.
        - **Where:** five Hugging Face datasets, `Sandflow/sand3d-objaverse` (66,887), `-cad` (24,495), `-articraft` (9,209), `-simready` (2,334), `-automate` (3,169), all tagged CC BY 4.0; the library is Apache-2.0 (github.com/Sandflow-ai/sand3d, created 2026-11-03). A model is "release soon" per the page.
        - **Licence, read on all five cards:** the programs, renders and USD files are CC BY 4.0. Each record is derived from a third-party object whose own licence also applies, and its per-row attribution must be kept (`source_license`, `attribution` columns). Upstream: Objaverse CC BY 4.0 or CC0 per object (Objaverse as a whole ODC-By), NVIDIA SimReady CC BY 4.0, Articraft CC BY 4.0 ("Articraft Data, Matt Zhou et al."), CADFS CC BY 4.0, AutoMate CC0. None is non-commercial or share-alike.
        - **Autodesk:** 16,766 of the CAD objects derive from Zero-To-CAD (ADSKAILab, Autodesk's AI lab). Its card was read: Apache-2.0, not gated, no further terms, so it passes the user's rule; keep the Apache notice and a statement of changes.
        - **Use here:** furniture and fixtures for the ignition-relevant furniture classes (IFCNet phase 1), as extra meshes or point clouds beside the IFC elements. Filter on each row's `f_score`, the pipeline's own measure of how close a program's shape is to its source. Keep the attribution columns with any derived set.

Items 1 to 4 below are unchanged from the night before.


The user accepted three steps for floor-plan drawings to LOD1. Step 1 is half-built, step 2 is built but not on screen, and step 3 is deferred.

1. **Drawings to LOD1 (step 1).**
   - **Done:**
     - the rooms model and its import, live (eea6d1d);
     - the drafting core, with its match rates checked here (This round).
   - **Next: the review screen.** Pipeline: `decodeDxfBytes`, `readDxf`, `suggestLayers`, `draftRooms`, in a worker or a dynamic import.
   - **What the user confirms per storey, in order:**
     1. the sheet region and scale (one DXF often holds several plans);
     2. the wall layers (suggestions pre-ticked);
     3. a common point to stack the storeys;
     4. floor level and height.
   - **Room review works from the draft's issues list.** The edits needed, most needed first:
     1. merge;
     2. split by a line snapped to wall ends;
     3. rename and set class;
     4. mark exits (the hotel's sheets have no entrance door blocks);
     5. delete or draw rooms, and add or remove doors.
   - **Keying edits:** key them by room position, so a re-draft keeps them. `keepLines: true` returns the linework to draw under the draft.
   - **End:** `toLod1Model([{level, draft}], levels)`, then `openLod1(model, {name})` in App.jsx, the entry point the JSON import uses.
   - **Merging needs a polygon union.** It is simple where drafted rooms share edges, since they are faces of one planar arrangement. It is not simple after the rooms have grown to the wall centre lines.
2. **AI room suggestions (step 2).**
   - **Built:** the Worker endpoint and the page's `suggestRoomsInCloud` (96170e4).
   - **To do:** show them in the review screen.
   - **Deploying the Worker needs the user's own Cloudflare account:** `npx wrangler login`, then `npx wrangler deploy` in `cloudflare/describe-worker`, then the repository variable `DESCRIBE_URL`. See that folder's README. Do not create accounts or sign in for the user.
3. **A fine-tuned vision model for scanned plans (step 3).** Deferred by agreement. It is worth doing only if many scanned plans must be handled. Ask the user before starting.
4. **IFCNet retraining:** paused by the user ("use the IFCNet weights themselves"). The shipped weights stay.
   - **To resume:** start from `STATUS.md` in the IFCNet working folder outside OneDrive, and the phase 1 plan in `wiki/research/ifcnet-furniture-study-plan-2026-11-05.md`.
   - **Before a paper:** read Utkucu et al. 2024 (doi:10.1016/j.aei.2024.102503) in full. It is the closest precedent for fusing names, rules and geometry, so the claim must stay narrow: ignition-relevant furniture sub-types, leave-buildings-out, in the browser, open weights with provenance.

Before any deploy: the gates above, then the live check, then update this file.

## This round (2026-11-06, evening)

- **README animations (fbdaed5, live):** two GIFs of the Vienna runs (doors open with the heat view; the gas fire with the tenability index), made from the video's frames with `tools/frames-gif.mjs`. The encoder moved to `tools/gif.mjs`, shared with `tools/fire-gif.mjs`. A doors-closed GIF was dropped: filmed from below, it did not read.
- **Exits out of the building, and rerouting to any of them** (the user asked for the route to try the next feasible exit when a stairwell is untenable, using fire escapes but "not every external door", so that the egress makes sense).
  - **The rule** (`exitDoors` in `core/ifcSemantics.js`, field `final`):
    - where the model says which doors are external (`external`: IsExternal TRUE or an EXTERNAL space boundary, read whatever FireExit says), the exits out are the external doors at street level (within 1.5 m of the lowest), and the fire exits at any level that are external and open onto no walkable floor;
    - an external door above street level with floor beyond it (a balcony or terrace) is left out, and so is a flat's front door flagged FireExit;
    - parts of the graph none of these reaches keep their exits; a model with no external doors keeps all its exit doors.
  - **Vienna:** 35 exit doors, 21 external, two exits out: the street door, and a FireExit door at the back of the first floor that opens onto no floor (most likely the hillside at grade; not checked against drawings). From the top floor the nearest is now that fire exit (33.6 m); the street door is 53.8 m.
  - **Vienna has one stairwell.** Its four IfcStairs are the four flights, and both exits are at its foot, 1.5 m apart, so no fire there makes a sensible switch between them from the upper floors. The user was told; the switching GIF uses the Office example (two stairs, an exit at each end).
  - **Used by:** the nearest exit, hazard-aware rerouting (`runFire({exits})`, new setting **Reroute towards**, default any exit), the forecast's "All N exits out of the building" and the agent's "any exit". The code check is unchanged.
  - **Search:** `astarToAny` in `core/route.js`, A* to the cheapest of several ends (heuristic: distance to the nearest). `astar` itself is untouched: a version that took a list was about 7% slower on one end (median 0.439 against 0.409 ms on a 150×150 lattice), so the two are kept apart.
  - **On screen:** orange rings at the exits (`ViewerManager.setExitDoors`), a legend entry, and an "Exit: another exit" row on the reroute card.
  - **Cache:** `IMPORT_SETTINGS` is 3, so cached models are read again for the new door field.
  - **Tests:** 259 unit tests (new: `astarToAny`, rerouting to another exit, the exits rule, the door's external flag), and three new browser-suite checks on Vienna (two exits out, two rings, the nearest from the top floor is the fire exit).
- **The rerouting GIF** (`docs/images/office-reroute.gif`, 1.3 MB): the Office example, a person upstairs at the east end, a Flamescapes class B fire at the foot of the east stair (8.6 MW peak), **Impassable above** 60 °C. The route takes the east stair until step 20 (100 s), then turns to the west stair and exit. Captured from the dev app with a caption injected into the page; the scripts are in the session scratchpad (`reroute/capture-reroute.mjs`). The switch came at step 20 in all five runs; the peak varied from 8.6 to 9.0 MW, as the GPU sometimes restarts on a coarser grid. The README quotes no single run's MW (Open 11): its Vienna captions were changed to match after fbdaed5 had quoted 13.3, 3.0 and 10.9 MW.

## This round (2026-11-06, afternoon)

- **A video of the Vienna model** (the user's request): 3 min 22 s, captioned, 1280×720 H.264.
  - **Where:** in OneDrive under `CM-iTAD/topologic_webapp/videos/2026-11-06 Vienna fire escape/`, full quality (48 MB) and phone (21 MB). Not in the repository.
  - **What it shows:**
    - the model, the graph and the ranked ignition points;
    - an electrical fault (class C) in a first-floor living-kitchen, 300 s with smoke and embers. Doors open: 13.3 MW peak, about 5 000 embers, 13 spot fires, and the route replanned at step 55. Doors closed: 3.0 MW;
    - a gas fire (class B) at another flat's hob: 10.9 MW;
    - escapes from the top floor (53.8 m down the stairwell to the street door) and from the fire's floor (29.5 m);
    - the heat, smoke and tenability views.
  - **How it was made:** Playwright driving the dev app, then a canvas page recorded with MediaRecorder (MP4). The scripts are in the session scratchpad: `video/capture.mjs`, `compose.mjs`, `probe.mjs`. A temporary dev-only hook exposing the exits was removed afterwards; the tree is clean.
  - **What it taught about the fires:** an electrical or gas fire in the services rooms stays small (26 to 122 kW), as there is little to burn. A fire in a furnished living-kitchen grows to megawatts within 300 s.

- **The human check** (8852fe2): Cloudflare Turnstile in front of the Worker.
  - **On the page:** `lib/turnstile.js` loads Turnstile only when "Describe this model" is set to Cloudflare and the site has a site key. It shows one slim strip, the panel's width and in the app's theme, under the send button. It checks at once and gives a fresh token for each request.
  - **In the Worker:** it checks the token with `siteverify` when it has `TURNSTILE_SECRET`, and refuses with 403 otherwise.
  - **Tested:** with a stand-in for `siteverify`, and end to end on localhost with Cloudflare's test keys.
- **Smoke, gas and tenability measures** (3f9a4f2):
  - **Readout:** the solver reads soot (mg/m³), CO2 and O2 (%) at every node, on the CPU and the GPU.
    - **Agreement:** CPU and GPU correlate 0.90 to 0.93 on them.
    - **Speed:** unchanged. On the GPU, 3.04 against 3.06 ms a step, and the worker run takes 1.9 s both times. The CPU is within noise.
    - **Memory switch:** "Record soot, CO₂ and O₂ too". A run past 64 MB of frames keeps the basic three.
  - **Show on the graph:**
    - Heat;
    - Smoke: visibility, soot, extinction, obscuration or optical density per metre;
    - Gases: CO, CO2, O2;
    - Tenability.
  - **The tenability index** is the largest share of a limit reached, after ISO 13571 cl. 4.1. Its four parts:
    - gas temperature;
    - visibility, 0 in clear air;
    - the asphyxiant dose: CO / 35 000 a minute, times exp(CO2/5) above 2 % CO2, plus Purser's low-O2 dose as in FDS;
    - the convected-heat dose, T^3.4 / 5e7 a minute.

    Both doses are measured against FED 0.3.
  - **The planner** closes nodes on the same four. No closure changed on the five examples within 30 s.
  - **Exports:** the TopologicPy JSON carries the series, the index, its doses and the time each node reached 1. A new CSV gives one row per node per frame, gzipped.
  - **Sources and what was read directly:** in the research note and on the Methods page. The ISO 13571:2012 constants were read in the 2007 text; confirm them against a full 2012 copy before a publication relies on them.

## This round (2026-11-06, morning)

- **Field notes and hints** (d0d1749, built to the plan and the user's answers):
  - **Field notes:** 12 questions marked found from the app's own results; "Field notes N/12" in the header; the list in a dialog.
  - **Hints:** one at a time at the foot of the panel. They lead past the model to the graph, route, alternatives, code check, fire, Flamescapes, closed doors, forecast plan and agent. On by default, with a switch and "Show closed hints again".
  - **Guided runs:** each ends on its note. "The way out" reports this run's own walk.
  - **Files:** `src/core/fieldNotes.js` (data, tested), `src/state/useFieldNotes.js` (localStorage), `src/ui/FieldNotes.jsx`.
- **The doors switch** (d0d1749):
  - **The problem:** it was locked for a whole fire run and its playback, with no explanation.
  - **The fix:** it now always sets the next run's doors. While a Flamescapes fire plays with the other setting, it offers "Restart the fire with the doors ...". The fire model choices still lock, and now say so.

- **The TU Wien escape-route model is a site example** (34d036b), by the user's decision:
  - **Name:** "Vienna apartment building (TU Wien)". The user called it the office building, but its record describes a residential building; the mixed-use custom model, which has offices, stays local.
  - **Licence:** CC BY 4.0, credited in `public/examples/index.json`, on the welcome screen and in the README.
  - **Unchanged file:** SHA-256 759c5e5f… as in the record, kept byte for byte by a `.gitattributes` rule (`-text`).
  - **Opening:** the build converts its 13.5 MB to 862 KB of fragments, so it opens in about a second with no IFC reader.
  - **In the app:**
    - graph: 6,795 nodes, 95% in one part;
    - exit doors read from the file: 35 of 81;
    - the code check passes on every storey.
  - **Browser suite:** section 8 checks it opens and reads its exits, and that its graph's largest part holds at least 90%.
- **The blank band after clicking a switch** (17cf2ff, reported by the user):
  - **Cause:** a switch's hidden checkbox was positioned against the page, not its switch. For switches low in a panel (Flamescapes, the code check's sprinklers) it sat below the window. Focusing it on a click scrolled the whole page up by as much as 340 px, though `body` has `overflow: hidden`.
  - **Fix:**
    - the checkbox is inside its switch;
    - `#root` is `position: relative; overflow: clip`, so the page has nothing to scroll.
  - **Browser suite:** flips the sprinklers switch and checks the page has not moved.
  - **The same band** showed in the 2026-11-05 screenshots of the hotel fire run and was wrongly blamed on the test script.

## This round (2026-11-05, late evening)

- **"Describe this model"** (da7064f, live):
  - **What it does:** describes the model from the app's own computed facts (`src/core/describeFacts.js`).
  - **Where it runs:**
    - on the device with Chrome's Prompt API (Chrome 148 or later, desktop);
    - or through the Cloudflare Worker in `cloudflare/describe-worker` (Workers AI, `@cf/openai/gpt-oss-20b`, free allowance, no retention by Cloudflare's terms). The Worker is offered only when the site is built with `DESCRIBE_URL`.
  - **Logo:** the engine's logo is off.
- **Code check** (d9afc1c, live; `src/core/codeCheck.js`; Route step, "Code check"):
  - **What it checks, per storey:**
    - the longest walk to an exit or stair;
    - exits against the occupant load;
    - exit separation;
    - exit door widths;
    - high-rise.
  - **Values:** IBC 2021, occupancy group B (offices), each labelled to be confirmed against SBC 201. The SBC's own text is not open. Sources are in `wiki/research/tall-office-sbc-egress-2026-11-05.md`.
  - **Display:** a toggle colours the graph by the distance to spare.
  - **On the TU Wien test models** (CC BY 4.0, local only): the real-world model passes; the custom model fails on exits and their separation.
- **The Worker's room suggestions** (96170e4): `POST /rooms`.
  - **Input:** one storey's draft summary.
  - **Output:** suggested merges, and names with classes.
  - **Checks:** only suggestions that name known rooms survive. Merged rooms must touch, each room gets at most one merge and one name, and classes come from the LOD1 set.
  - **Cost:** about 100 to 200 neurons a storey.
- **Rooms (LOD1) models in the app** (eea6d1d, live):
  - **The core** (`src/core/lod1.js`, written by an agent, checked here) reads:
    - a TopologicPy LOD1 export with its circulation and adjacency graphs;
    - a GeoJSON plan, whose schema is in `wiki/concepts/lod1-models.md`;
    - IfcSpace volumes.

    It converts them to the egress geometry an IFC gives, and also writes the GeoJSON plan.
  - **The app** (Model step, "Rooms models (LOD1)"; or drop the JSON files together on the viewport):
    - opens the model with no IFC;
    - draws it as rooms tinted by class, with walls, slabs, stairs, and doors (exits green) (`ViewerManager.loadLod1`);
    - lists rooms, storeys and warnings;
    - exports the GeoJSON plan.

    The graph, routes, code check, both fire models and the TopologicPy export run on it unchanged. Identification and the TopologicPy cross-check need an IFC and say so.
  - **The lab's hotel model** (private, local only; not in the repository): 586 rooms on 15 storeys.
    - In the dev app it opened in 0.5 s, with 1 warning (one opening whose rooms do not touch).
    - Graph: 47,305 nodes in 0.6 s, 100% reachable, every storey joined.
    - A Flamescapes fire ran 180 s in 58.5 s on the GPU, with no console errors.
    - Headless (`tools/lod1-check.mjs`, rerun here): the route from the top-floor guest room is 143.1 m, and the GeoJSON round trip is identical.
  - **The new block** (also private): 54 rooms on 4 storeys, all joined.
  - **IfcSpace fast path:** within about 6% of the full IFC's route on four public models, with 3 to 5 times faster extraction.
  - **Rooms without IfcSpace:** traced from the graph's regions, they matched too few rooms to ship (12 of 21 on the Duplex).
  - **The browser suite** gains step 9: a two-storey GeoJSON plan opens, lists its rooms, exports, and builds a graph joined through its stair.
    - **A false failure, fixed:** the step's first version failed because its wait function used a closure variable. The function runs inside the page, where the variable does not exist, so the wait failed at once. The step now passes the value in.
- **Drawings to LOD1, the core** (4cfcefb; `src/core/drawingToLod1.js`, written by an agent, checked here):
  - **Reader:** its own DXF reader, with no dependency. It explodes nested blocks and handles mirrored entities, code pages, and Rhino's arcs exported as polylines.
  - **Pipeline:** it suggests layers, bridges door openings, builds the planar arrangement, cuts rooms by their names and grows them to the walls' centre lines. `toLod1Model` and `draftSummary` (for the Worker) are included.
  - **Results against the lab's checked rooms**, rerun here: rooms found at IoU 0.8 or more, then matched at 0.5 or more.

    | building | rooms | lab's drafting | with the lab's layer settings | layers suggested by the core |
    |---|---|---|---|---|
    | residence (new block) | 54 | 30 found, 43 matched | 42 found, 49 matched, 6 extra | 38 found, 45 matched |
    | hotel, 7 sheets | 194 | 20 found, 65 matched | 57 found, 135 matched, 147 extra | 53 found, 135 matched, 193 extra |

  - **Speed:** each sheet is read in 0.1 to 0.4 s and a storey is drafted in 12 to 280 ms.
  - **Failure modes:**
    - partitions the drawing omits;
    - spaces bounded only by columns;
    - site walls on wall layers;
    - no entrance doors on the hotel's sheets;
    - hatch-only walls.

## This round (2026-11-05, afternoon and evening)

- **Identification** (study 4ef5e1e; app 8ad8b3d):
  - **Rules:** `src/core/identify.js` names elements by IFC class, else name, else box. Accuracy 0.86 and macro-F1 0.83 on 6,066 typed elements from 45 models; 0.55 to 0.60 on 159 hand-labelled proxies. The full study is `wiki/research/ifc-identification-benchmark-2026-11-05.md`.
  - **IFCNet:** the DGCNN is ported to ONNX, exactly and pruned: 1.45 MB, and 22 to 45 times faster with no answer changed. It reproduces the paper's balanced accuracy (0.799 against 0.791). It adds almost nothing to the rules.
  - **Shipped weights:** by the user's decision the weights are in the public build (`public/models/ifcnet-dgcnn-2048.onnx`, `NOTICE-ifcnet.md`). IFCNet's code is MIT; the weights state no licence, and they come down on request.
  - **The Model step:** "What the elements are" shows the rules at once; "Identify by shape" runs IFCNet on generic elements the name could not place and keeps its answer at 90% or more. onnxruntime-web 1.30.0 loads from jsDelivr on first use.
  - **LLMs:** Qwen 0.5 to 0.6B in the browser scored below the rules and made them worse when asked to correct them.
- **Flamescapes:**
  - **Liquid fires** (classes B and K) never lit; fixed with the pool burning-rate cap (0e9c2f1; Babrauskas 1983 via NUREG-1805).
  - **GPU-lost fallback:** a GPU lost mid-run restarts the fire on the CPU (fac59bb).
  - **Smoke cloud and embers** (4f0f82a): the solver's soot is drawn as a smoke cloud (`smokeCloud`, both backends); the run card says "Embers: off" or "on: N thrown, M spot fires".
  - **Speed** (4f0f82a): on this machine's integrated GPU a 120 s fire went from 72 s to 25 to 29 s.
    - The view draws 15 times a second while the GPU computes (`ViewerManager.setDrawRate`).
    - The GPU grid is sized to its measured speed in fixed levels (`flamescapes.worker.js`, `GPU_STEP_MS` 33). The speed is kept per GPU in localStorage.
    - "Grid on the GPU" offers speed-sized or finest.
    - One CPU core: 2.8 to 2.9 times real time on the Duplex (`benchmark/results/2026-11-05-fire.json`).
  - **Grid dependence (important):** the same Duplex sofa fire peaked at 0.09 MW on 0.20 m, 2.5 MW on 0.23 m, 6.1 MW on 0.27 m and 0.75 MW on the CPU's 0.34 m. The Methods page and the panel say to compare runs on one grid. See Open 11.
  - **Openings** (00ec1a4, 49ae6e5):
    - doors open or closed: a switch beside the fire class, for the smoke only; routes still use every door;
    - windows closed or open;
    - ventilation off or running: every air terminal swaps gas with one well-mixed duct volume at an assumed 100 L/s a terminal (`ducts` on both backends; `DuctMixing.lean` proves conservation and that values stay within range).

    Checked on the CC BY TalTech DS3 model: 60 of 82 terminals inside the clipped grid.
- **Export for TopologicPy (JSON)** (49ae6e5, `src/core/topologicExport.js`): in the Graph step and under a finished fire. It holds the graph in IFC world coordinates at floor level, with per-node fire time series and untenable times, edges with lengths, and the run's settings. topologicpy 0.9.86 `Graph.ByJSONFile` reads the Duplex export with all 1,597 nodes and 4,597 edges.
- **Guided runs** (49ae6e5, `src/core/guided.js`): three presets on the welcome screen run the steps themselves and stop at the result (*The way out*; *A fire on the route*; *Smoke behind closed doors*).
- **Notes for later:**
  - `wiki/research/fire-suppression-note-2026-11-05.md`: sprinklers, RTI activation, Fire-X's droplet evaporation, manual agents;
  - `wiki/research/sbc-firefighting-future-notes-2026-11-05.md`: SBC egress checks, hose coverage, firefighter access, doors in the graph fire models. No code value is quoted.
- **Tools:**
  - `tools/fire-gif.mjs` records the Flamescapes fire as a GIF with no encoder to install (`--origin`, `--smoke-at`);
  - `tools/flamescapes-gpu-check.mjs` takes `--class` and `--ducts`.

## Flamescapes, fire classes and the ignition study (2026-11-05)

- Papers: Flamescapes (Wrede, Wagner et al. 2026, ACM TOG 45(6) 246; DOI printed as 10.1145/3842518, not yet registered at Crossref on 2026-11-05) with its supplement, and Fire-X (Wrede et al. 2025, doi:10.1145/3763338), both CC BY 4.0, transcribed in `wiki/sources/papers/` with attribution and equation catalogues (Flamescapes 20 equations and 217 parameters, Fire-X 32 and 133). The PDFs are in the ignored `reference/papers/`. Their GitHub repositories (helgewrede/flamescapes, helgewrede/firex) hold project pages only, so no code was reused; saskaZs/Fire-Simulation is a particle visual with no licence, used for no code.
- Solver: `src/core/voxelize.js`, `flamescapes.js`, `flamescapesRun.js` follow Algorithm 1 on a grid built from the IFC (walls, slabs, roofs, stairs, obstacles; doors open; contents the paper's statistical fire load, Eqs. 3 to 5). The header of `flamescapes.js` lists what is the paper's and what is changed for a coarse CPU grid: exact integration of each reaction and two-body exchange over a step, a thermally thick surface layer (1 kg/m^2), sub-grid ember spot ignition by excess heat only, the first item alight at the start, thin walls snapped to the cell holding most of them, solid-aware trilinear advection. Constants are the paper's Tables 3 to 5 where it gives them, Fire-X's 417 kJ per mol O2, and the fire class's fuel for the first item.
- App: an optional toggle in the Simulate step (off by default) adds "Flamescapes: fire and smoke" and "... and embers". `workers/flamescapes.worker.js` sizes the grid to the device (`lib/hardware.js`) and streams frames without keeping them; the fire plays them as they arrive, reroutes on the gas temperature at head height, and the plan reuses the run (`forecastFromFrames` in `forecast.js`: heat FED plus ISO 13571 CO FED; smoke closes a node below the visibility limit; the plan warns when the grid covers only part of the graph). Heat or smoke view; glowing flame and ember particles in persistent layers.
- Fire classes (`src/core/fireClasses.js`): NFPA 10 A, B, C, D, K and ISO 3941:2026 L, each value with its source key; CO2 yields from each fuel's carbon balance; suitable and unsuitable extinguishers shown in the panel. Values come from the review `wiki/research/ignition-sources-2026-11-05.md` and `.json` (ignition sources, design fires, IFC mapping, DOIs checked at Crossref).
- Likely ignition points (`src/core/ignition.js`, catalogue generated by `tools/ignition-catalog.mjs` into `ignitionCatalog.js` and `ignitionClasses.js`): elements and rooms ranked by share of fire deaths times match confidence over candidates; one click sets the fire origin and its class. Lithium-ion scenarios are class L.
- Lean: FireX.lean, Flamescapes.lean, FlamescapesPort.lean (seventh batch; 26 modules, more than 1,230 results, standard axioms only). The proofs confirmed the transcribers' suspected typos (none refuted) and found six defects in the port, fixed in 30c019a; class D's heat capacity followed in 0d0c676.
- Architecture review (2026-11-05, 16 findings): all fixed in 7df6041. The two criticals were walls acting as an air source in advection and a clip that dropped long walls crossing a clipped storey; each has a regression test.
- Speed (`tools/fire-bench.mjs`, one CPU core, quiet machine, rerun 2026-11-05): Flamescapes 34 to 36 ms per step on the Duplex (26k cells, 0.37 m), 2.8 to 2.9 times real time; advection and projection are most of it. The graph models take 0.1 to 7 ms. GPU timings: see This round.
- Tests: 207 committed unit and property tests (2026-11-05 evening); the browser suite runs a 30 s Flamescapes fire with embers (15 to 23 s).
- Live check: 0 failed steps when the Render backend is awake (see State).
- README (2026-10-29): rewritten for new users, with a tour of the four steps on the Duplex, the other examples, a phone view and the Methods page. The screenshots in `docs/images/` come from `node tools/readme-shots.mjs` against a running dev server (or any build with `?debug`), using only the shipped examples; re-run it after a UI change. The RL shot's success rate varies between runs, so the README quotes none.

## Loading speed (2026-10-29)

Opening an IFC on a phone was slow: every step of a first open waited on the one before it, web-ifc was downloaded and started twice, and the examples were converted on the phone like any other file.

- The import worker reads only what the page uses from the fragments (`src/core/importSettings.js`): property sets, quantities, materials, type objects and their relations are left out (ifcSemantics reads them from the IFC). Conversion 40% faster on the Duplex and 26% on Office A at 4x CPU; fragments a third the size. `IMPORT_SETTINGS` is 2, so models cached before are converted again once.
- The same worker then reads the egress semantics with its already-loaded web-ifc and posts them after the fragments; `semantics.worker.js` (419 KB gzipped, a second web-ifc) is gone. The semantics are cached with the fragments, so reopening a cached model starts no web-ifc at all (Duplex reopen 0.22 s).
- Pointing at or tapping a file button starts the import worker at once and fetches web-ifc's WASM and the fragments worker in parallel, instead of one after another after the file arrives.
- The examples are converted when the site is built (`scripts/prepare-examples.mjs`, run by `npm run build`; output in `public/examples/prepared/`, not committed). An example whose converter string matches the page's opens from its fragments (Duplex: 81 KB instead of a 487 KB gzipped IFC plus 1.1 MB of web-ifc) with no IFC worker; the IFC is fetched only if the server cross-check needs it. Otherwise the example opens from its IFC as before.
- The viewer's chunks (three, the That Open engine, ViewerManager) get `modulepreload` links (a plugin in `vite.config.js`), so they download while the page script runs instead of after React renders.
- Measured on an emulated phone (390 x 844, main thread 4x slower, 4G at 9 Mbps and 85 ms, cold cache; median of 3), deployed build against this one, Duplex: tapping the example until the model is usable 4.08 s to 0.97 s; picking the file 2.94 s to 1.20 s; page start 2.44 s to 2.24 s; reopening from the cache 0.67 s to 0.69 s. Chrome will not slow workers, so on a real phone the conversion share is larger in both builds.
- Equivalence: the Model and Graph steps report identical counts, nodes, links, doors and reachability in both builds on the Duplex example and on AC20-FZK-Haus, Office A, example.ifc, AC-20 Smiley West (10 buildings), 4351, the dental clinic, HHS Office and S_Office (stair parts); every one opens faster (desktop 0.2 to 0.9 s). The browser suite adds a step that the Duplex example opens with no IFC worker and still reports 4 of 14 exit doors.
- Not done, possible next: a service worker so repeat visits skip the network (GitHub Pages caches assets for only 10 minutes); geometry conversion itself is now most of a first open of a user's own file.

## What the 2026-10-28 round added

- Two-zone fire model on the IfcSpace room graph (gap LG8): `src/core/rooms.js`, `src/core/zone.js`, `tools/zone-check.mjs`, offered in Simulate when a model has rooms.
- Surrogates with error bounds (LG7): `src/core/surrogate.js`.
- Vertical cores: `src/core/cores.js` and the lattice. Stair parts aggregated into IfcStair are read (S_Office); a stair without treads (an IfcStair with no steps, a room named as a stair) is walked along an assumed dog-leg line; a core is linked wherever its two floors are not already joined there (Smiley West's treads join neither floor); lifts and escalators (IfcTransportElement, rooms and proxies named as lifts) join floors only with "Use lifts and escalators" on. `tools/core-check.mjs` reproduces the comparisons, including a treads-removed experiment.
- Exits in every large part of the graph (`exitsPerPart` in `src/core/ifcSemantics.js`): AdvancedProject 1 exit to 18, canLeave 0.005 to 0.149.
- Review fixes C1, M1 to M11, K1, K2, m1 to m12, w1 to w4 (error boundary, plan staleness and cancellation, worker failure handling, RL in a worker, capped deadline-map work, colour ownership, accessibility).
- Lean: 23 modules, more than 990 theorems and lemmas, standard axioms only; new Surrogate.lean, Zone.lean, Cores.lean (README sixth batch). The Methods dialog renders 21 properties (84 TeX lines, all render in KaTeX).
- Benchmark (gap LG5): `node tools/benchmark.mjs` over 38 public models listed with source and SHA-256 in `benchmark/manifest.json`; speed, accuracy against an exact free-plane geodesic (`tools/geodesic.mjs`), exit-rule precision and recall, deadline checks, and a stress suite with 15 known-bug regressions, each shown to fail on the parent of its fix. Three result-preserving speed-ups (deadline map 1.3 to 1.6 times, RL optimum 1.5 to 6.6 times, A* 1.4 to 2.2 times); outputs identical on the corpus (`tools/output-hash.mjs`, and every non-timing field of `tools/forecast-check.mjs`). Results in `benchmark/results/`.
- Research: `wiki/research/open-source-evacuation-study-2026-10-28.md` (JuPedSim, Vadere, FDS+Evac, CFAST, Menge, MomenTUM from source; Pathfinder and buildingEXODUS from documents only), ranked gaps G1 to G9 and which RiMEA and NIST TN 1822 tests the model can run today.
- Test corpus: 11 openly licensed models in `%USERPROFILE%\ifc-samples\more2\` with `SOURCES.md` (local testing only; HHS Office is no-derivatives).

## Open

1. Robustness bugs the stress suite (`node tools/benchmark.mjs --stress`; see `benchmark/README.md`) marks open: an infinite vertex in a floor triangle makes the lattice build hang, and in a wall empties the graph; millimetre-scaled geometry crashes the lattice build (`Invalid array length`, it samples the whole grid before coarsening); broken or non-IFC files give raw web-ifc WebAssembly traps instead of a readable error, three taking 5 to 14 s to fail.
2. Gap G1 from the study: stairs are walked at level speed (reference tools use 0.45 to 0.88 of it), so stair times are 13% to 50% short. Needs a per-link pace used by every router (route.js, forecast.js, deadline.js, rl.js, alternatives.js, fire.js) and a Lean restatement with link times scaled.
3. Gaps G2 to G5 from the study: walking slowed in smoke, toxic gases and radiant heat in the dose, door capacity and queues, pre-movement distributions (pairs with the deadline map).
4. HITOS (`more2\osb-testfiles-hitos`) joins its floors only with lifts on; its stair cores link but its floors stay apart. Not yet investigated.
5. Lift rides cost their rise at walking speed with no wait; assumed stairs have no known width.
6. Novelty of C1 is provisional until Chen et al. (2023, https://doi.org/10.1016/j.ress.2023.109571) and Tabirca et al. (2015, https://doi.org/10.1016/j.procs.2015.05.425) are read in full.
7. Not proved: floating point for the zone model (tested), the zone Euler step's accuracy, componentLabels' correctness, the deadline map's distance below the exact deadline (up to 7.8 s in tests).
8. Occupant simulation add-on, asked for by the user (2026-11-05): many occupants moving at once, each with a pre-movement time, walking speed slowed on stairs (gap G1) and in smoke (the Flamescapes visibility, gap G2), door queues with a specific flow per metre of width (gap G4), and a running FED from heat and CO; route choice by shortest path, by re-planning on the fire as it is, by the forecast's earliest safe route, or by a learned Q-policy (today's Q-learning trains one start and one exit, so a per-exit or per-room policy is needed). Outputs: evacuation curve, RSET, incapacitated count, animated occupants. Needs a source review for each parameter before code.
9. Fire classes: the parameters marked `provisional` in `fireClasses.js` (B and C ignition temperatures for embers, K's vapour latent heat, L's fuel and agents) have no source yet.
10. Flamescapes in the browser is not calibrated against room fire tests; the review's room-scale data (NIST furniture calorimetry, cooktop fires) are the natural first comparisons. Indoors at 0.37 m the embers add little: the Duplex fire lights no spot fire in 120 s.
11. **Grid dependence of the Flamescapes fire.**
    - **The finding:** peak heat release moves by two orders of magnitude between 0.20 and 0.34 m cells, and not in one direction. The 0.23 m run also settles to a constant 460 kW tail.
    - **Before any quantitative use:** a grid convergence study (same scene, 0.15 to 0.4 m, CPU and GPU), and a check of the tail (the pilot, or a steady burn of the contents).
    - **Speed-sized grid noise:** sizing the GPU grid by speed makes the grid device-dependent; the stored speed keeps it fixed per browser. Fresh browsers picked 110k to 200k cells in three sessions, because the speed measurement is noisy while other jobs run.
12. **Ventilation is a bound, not a model.**
    - **Missing inputs:** system membership (IfcRelAssignsToGroup with IfcDistributionSystem) and supply, return and exhaust roles are not read; duct geometry is unused; 100 L/s a terminal is an assumption.
    - **Next:** read the systems where the IFC has them (DS3 has 4), make exhaust terminals a sink, and source the flows.
13. **Doors in the graph fire models,** SBC egress checks, fire hose coverage and firefighter access: see `wiki/research/sbc-firefighting-future-notes-2026-11-05.md`. Suppression: `fire-suppression-note-2026-11-05.md`.
14. **"Describe this model":** built and live (da7064f). The Worker is not deployed until the user sets it up with their own Cloudflare account.
15. **IFCNet:** retraining paused by the user (late evening).
    - the weights' licence (ask the authors);
    - the retraining study (agent phase 1 pilot; the full campaign needs approval);
    - the shape classifier adds almost nothing to the rules today.
16. **Learned escape:** done.
    - **The new method:** the shielded learner (`rlSafe.js`, `SafeEscape.lean`, `wiki/research/rl-egress-study-2026-11-05.md`) is the default.
    - **Still open:**
      - The shield uses the RL step's own hop-front fire. Covering the temperature, zone and Flamescapes forecasts needs a seconds-based version (deadline.js has one).
      - Certification with many exits can take up to 2.6 s.
      - rl.js still draws its policy with a no-revisit walk, which caused most of its misses. Drawing the policy walk instead cuts them from 15.0% to 2.6%.
    - **Item 8:** its per-room, any-exit policy now exists as `escapeField`.
17. **The CPU solver's advection** is half of each step, at about 730k trilinear samples. Any speed-up must stay bit-identical: vorticity reads the faces between solids, so no face can be skipped. Micro-optimisations, such as the exact own-component shortcut, are possible.
18. **The guided run "A fire on the route"** uses heat diffusion, which reads faintly. Spread-along-links, or the forecast plan, may show the rerouting better.
19. **Resolved, test only:** step 9's first version failed on a broken wait in the test itself (This round). The app opened the plan every time.
20. **The code check covers offices only** (IBC group B), so a hotel (group R-1) or a mixed building needs its own values.
    - **On the hotel:** it reports exits as close as 2.3 to 5.8 m apart on several storeys. The cause is not yet established, either the stairs' real layout or how stair entrances are grouped. Check before quoting it.
    - **Not checked:** the common path of travel, dead ends, stair widths and capacity, enclosure.
21. **The IfcSpace fast path is not in the app.** `lod1FromIfcSpaces` works headless.
    - **What it would give:** for very large IFCs it could skip fragments conversion: read spaces, doors and stairs in the import worker and draw prisms, as in `tools/lod1-check.mjs`.
    - **Frame:** do not mix its frame with a fragments model of the same file.
22. **Drawings to LOD1:** see Resume here, item 1. Even the improved core finds only 57 of the hotel's 194 rooms at IoU 0.8, with 147 extra rooms, so the review screen is not optional.
