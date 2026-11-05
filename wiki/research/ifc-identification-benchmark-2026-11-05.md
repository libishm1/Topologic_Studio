# What is this element? Rules, a geometry classifier, embeddings and small LLMs for identifying IFC elements in the browser

Date: 2026-11-05
Status: benchmark and prototype. New files only: `src/core/identify.js` and its tests, `src/core/identifyGeometry.js`, `src/workers/identify.worker.js`, `tools/identify-bench.mjs`, `tools/identify-bench.html`, `tools/identify-ifcnet.py`; numbers in [`benchmark/results/2026-11-05-identify.json`](../../benchmark/results/2026-11-05-identify.json). Nothing is wired into the app yet, and no model weights are in the repository.
Related: [ignition-sources-2026-11-05.md](ignition-sources-2026-11-05.md) (its corpus check found that exporters rarely type appliances and furniture; `ignition.js` matches them by name), [ifc-handling-survey-2026-10-27.md](ifc-handling-survey-2026-10-27.md).

Conventions. Third-party IFC models and IFCNet's data and weights were used locally only: this note and the results file hold aggregate numbers, never a file, a name or a per-element result. "Hard subset" labels were assigned by hand by the author of this note (Claude), from the element's name alone; they are judgement, not ground truth from the files. Every DOI was resolved at Crossref on 2026-11-05 (arXiv's DOI is registered with DataCite and was resolved there). Licences were read on the source pages named in Table 1.

---

## 0. Summary

The question was whether an algorithm, a ported geometry classifier, an embedding model or a small LLM in the browser best tells what an IFC element is when its class is withheld. On 6,066 typed elements from 45 models and 159 hand-labelled proxies, the rules win on every subset, at no download and about 6 µs an element; no model alone comes close, and adding IFCNet's shape where the rules fall back to the box gains at most two points on typed elements and nothing on proxies.

| Method | Typed elements: accuracy / macro-F1 | Proxies (hand-labelled) | Time an element | Download |
|---|---|---|---|---|
| Rules (`identify.js`) | 0.86 / 0.83 | 0.60 / 0.54 | 6 µs | none |
| IFCNet DGCNN, ported and pruned, WASM | 0.41 / 0.35 (0.61 / 0.58 on the 11 labels it has) | 0.28 / 0.20 | 0.21 s | 1.45 MB model, 3.1 MB runtime |
| all-MiniLM-L6-v2 int8, nearest label | 0.45 / 0.41 | 0.33 / 0.31 | 31 ms | 23 MB, 4.3 MB runtime |
| Qwen3-0.6B q4f16, calibrated, WebGPU (best LLM; sample of 160 and 80) | 0.47 / 0.42 (rules 0.82 / 0.82 on the same) | 0.23 / 0.35 (rules 0.63 / 0.54) | 2.2 s | 570 MB, 4.3 MB runtime |

IFCNet (Emunds et al. 2021) was ported to ONNX and made 22 to 45 times faster and five times smaller without changing one answer, by an exact rewrite of its edge convolutions and by pruning the channels training had left dead; the port reproduces the paper's accuracy on IFCNet's own test data, and the browser reproduces the port's answers. Its code is MIT, but its weights carry no stated licence and are kept out of the repository; the integration (`identifyGeometry.js`, a lazily loaded worker) takes a model URL and is off by default. Small LLMs are slow (2 to 6 s an element), large (0.5 to 0.8 GB), at or below the embeddings' accuracy even after calibration, and make the rules' answers worse when asked to correct them; one quantised build gave nonsense on this machine's GPU. Recommendation: ship `identify.js`; keep the shape classifier opt-in until the weights are licensed; keep LLMs, if at all, for describing a model in words, not for identifying its elements.

---

## 1. Survey: parts that could identify building elements in a browser

Table 1. Candidates, as found on 2026-11-05. Sizes are the files a browser would download; "brotli" is the transfer size from jsDelivr.

| Part | What it does | Licence (code; weights) | Download | Runtime | What porting takes |
|---|---|---|---|---|---|
| IFCNet DGCNN (Emunds et al. 2021, arXiv:2106.09712, DOI 10.48550/arXiv.2106.09712; code RWTH-E3D/ifcnet-models) | Point-cloud classifier, 20 IFC classes (IFCNetCore) | Code MIT (LICENSE: "this software and associated documentation files", 2021, Christoph Emunds). Weights: no licence stated (see 1.1) | Checkpoint 10.0 MB zip (21.7 MB with optimiser state); ported here to 1.45 MB fp32, 0.74 MB fp16, 0.39 MB int8 weights | onnxruntime-web 1.30.0 (MIT): WASM 14.2 MB, 3.07 MB brotli; WebGPU build 28.3 MB, 5.48 MB brotli | Done here (section 3.2): ONNX opset 17, exact rewrite and pruning, quantisation |
| IFCNet MeshNet (same) | Mesh classifier, same 20 classes | As above | 45 MB zip | onnxruntime-web | Needs MeshLab quadric decimation to 2,048 faces and face-neighbour lists before inference: a mesh simplifier in the browser |
| IFCNet MVCNN (same) | 12 rendered views, VGG-11 | As above | 1.46 GB zip | onnxruntime-web, WebGPU | Rendering 12 views per element; far too large |
| SpaRSE-BIM (Emunds et al. 2022, Advanced Engineering Informatics 53, 101641, DOI 10.1016/j.aei.2022.101641; branch `sparse-bim`) | Sparse 3D CNN on voxelised geometry; trained on IFCNet and BIMGEOM | Code MIT (same LICENSE); no pretrained weights linked | n/a | MinkowskiEngine (CUDA sparse convolutions) | Sparse convolutions have no ONNX operator: a re-implementation, and training |
| BIMGEOM / Collins et al. 2021 (EC3, DOI 10.35490/ec3.2021.168) | Graph CNN on triangle meshes; dataset from industry and academia | No public weights found | n/a | n/a | Not portable without weights |
| Qwen2.5-0.5B-Instruct (onnx-community build) | Small instruction-tuned LLM | Apache-2.0 (Qwen model card; the ONNX repository states none and inherits it) | q4f16 483 MB, int8 512 MB, q4 786 MB, fp16 997 MB | Transformers.js 3.8.1 (Apache-2.0, 0.89 MB) on WebGPU or WASM | None: runs as published |
| Qwen2.5-1.5B-Instruct | As above, larger | Apache-2.0 | q4f16 1.22 GB, int8 1.58 GB, q4 1.79 GB | As above | None in principle; here the q4f16 build aborted in Chrome (section 4.3) |
| Qwen2.5-3B-Instruct | As above | "other": the Qwen Research License, not Apache-2.0 | n/a | n/a | Excluded on licence |
| Qwen3-0.6B (onnx-community Qwen3-0.6B-ONNX) | Small LLM with a thinking mode | Apache-2.0 | q4f16 570 MB, int8 618 MB, q4 919 MB | As above | None (thinking turned off with an empty think block) |
| all-MiniLM-L6-v2 (Xenova build) | English sentence embeddings, 384-d | Apache-2.0 | fp32 90 MB, int8 23 MB | Transformers.js | None |
| multilingual-e5-small (Xenova build) | Multilingual sentence embeddings | MIT | int8 118 MB, fp32 470 MB | Transformers.js | None |
| bge-small-en-v1.5 (Xenova build) | English sentence embeddings | MIT | int8 34 MB | Transformers.js | None (not benchmarked: English only, like MiniLM) |
| IFC-specific text classifiers | Element names to IFC or bSDD classes | None with published weights found; the closest, a prototype that does nearest-label search with a multilingual MiniLM, has no verified licence | n/a | n/a | Method (c) below is the same idea |

### 1.1 Do IFCNet's weights come with the MIT licence?

Not clearly. The repository's LICENSE grants rights to "this software and associated documentation files" and the README links the pretrained models without stating any terms for them; the repository has no releases (GitHub lists none; one tag, v0.1, without assets), and the weights are not in it. They are served from ifcnet.e3d.rwth-aachen.de as `dgcnn_model.zip` and siblings, and that site, which also hosts the dataset, states no licence for either. The weights are a product of the dataset, which the paper describes as "around 1000 IFC models ... collected from real-world projects, student works and online sources" (section 3), with no terms. The server sends no CORS header, so a page on GitHub Pages cannot fetch the weights at run time either, and what it would fetch is a pickled PyTorch checkpoint, not an ONNX model.

So, as the user decided: the weights stay out of the repository and the public build. `identifyGeometry.js` has no default model URL; it takes one (an ONNX file built locally by `tools/identify-ifcnet.py` from weights the user downloaded, or a published copy if the authors license one), and without it the name rules' answer stands. The two ways to ship it are (a) to ask the authors (Christoph Emunds, E3D, RWTH Aachen) to state a licence for the weights, ideally MIT like the code, or (b) to train the same network on openly licensed geometry (the export script takes any checkpoint of this architecture).

---

## 2. Task and data

The task: given what the app reads for one element (Name, ObjectType, LongName, the PredefinedType text, its storey and room names, its bounding box and its height above its storey's floor) but not its IFC class, name its class from 18 labels: wall, slab, roof, door, window, stair (stairs, ramps and their landings), railing, column, beam, covering (ceilings, floor finishes, cladding), furniture, appliance, sanitary, heater, lighting, mep (distribution and services), space, other.

Ground truth is the element's own class and PredefinedType (its type's when its own is unset), mapped by `labelForClass` in `identify.js` (Table 2).

Table 2. IFC classes and labels.

| Label | IFC classes (IFC2x3 and IFC4) |
|---|---|
| wall | IfcWall, IfcWallStandardCase, IfcWallElementedCase, IfcCurtainWall |
| slab | IfcSlab, IfcSlabStandardCase, IfcSlabElementedCase (not ROOF or LANDING) |
| roof | IfcRoof; IfcSlab ROOF |
| door | IfcDoor, IfcDoorStandardCase |
| window | IfcWindow, IfcWindowStandardCase |
| stair | IfcStair, IfcStairFlight, IfcRamp, IfcRampFlight; IfcSlab LANDING |
| railing | IfcRailing |
| column | IfcColumn, IfcColumnStandardCase |
| beam | IfcBeam, IfcBeamStandardCase |
| covering | IfcCovering |
| furniture | IfcFurniture, IfcFurnishingElement, IfcSystemFurnitureElement |
| appliance | IfcElectricAppliance, IfcAudioVisualAppliance, IfcCommunicationsAppliance |
| sanitary | IfcSanitaryTerminal |
| heater | IfcSpaceHeater |
| lighting | IfcLightFixture, IfcLamp |
| mep | Segments, fittings, terminals, valves, dampers, outlets, switching and protective devices, boards, sensors, alarms, pumps, fans, boilers, tanks and the other distribution classes (65 in all; list in `identify.js`) |
| space | IfcSpace |
| other | Every other element class: IfcMember, IfcPlate, IfcFooting, IfcBuildingElementPart, IfcDiscreteAccessory, IfcTransportElement, IfcShadingDevice, civil and geographic elements |
| (none: hard subset) | IfcBuildingElementProxy, IfcFlowTerminal, IfcDistributionElement, IfcDistributionFlowElement |

Corpus: the 45 distinct building models of the local corpus (the benchmark manifest's 38 models less stress-only files and two duplicates, the `more2` folder's ten models, and the four shipped examples), 66,645 elements, median 401 per model. Read with web-ifc 0.0.77 (Node), geometry in the viewer frame.

- Primary subset: every typed element, at most 25 per label per model (drawn at random with a fixed seed, so big models do not dominate): 6,066 elements from 45 models. By label: wall 848, slab 573, roof 226, door 677, window 616, stair 297, railing 318, column 362, beam 289, covering 283, furniture 394, heater 25, lighting 17, mep 100, space 472, other 569; appliance and sanitary 0, because no model types them (the ignition review found the same).
- Hard subset: proxies and generic elements, the kind that matter in practice. From the 5,855 generic elements, one per distinct name per model, at most 12 per model (179 names); the author labelled 159 from the name alone and left out 20 whose names say nothing ("Object-017", codes). By label: other 66 (entourage people, trees, site marks, text), mep 30, sanitary 28, appliance 10, beam 6, slab 6, lighting 5, furniture 4, door, stair, railing and heater 1 each; from 24 models; IfcBuildingElementProxy 112, IfcFlowTerminal 46, IfcDistributionElement 1.
- LLM sample: up to 20 per label of the primary subset (317) and the whole hard subset (159), for the slow methods; every method is also scored on it.

Box features, computed as the app can compute them (`boxOf` in `identify.js`): the two horizontal extents along the element's own principal axes (a wall at 30 degrees is still thin), the height, and the bottom's height above the storey floor, the floor being the median bottom of the storey's doors (else walls). The room is the smallest IfcSpace whose box holds the element's centre.

---

## 3. Methods

All methods ran in the same headless Chrome on the same records (`tools/identify-bench.mjs run`, page `tools/identify-bench.html`), one fresh page per method, with the browser's memory sampled during each.

### 3.1 Rules (`src/core/identify.js`)

In order: the class when the caller passes a specific one (the app will; the benchmark does not); a LongName (only spatial elements carry one: a room); the PredefinedType through an enumeration table; the name (Name and ObjectType) through an ordered list of multilingual patterns; the box. Name patterns are English, German, Norwegian, Dutch, French, Spanish, Italian and a few Arabic words, with the `(?:^|[^a-z])` boundary the ignition review found necessary for Revit's "M_" names, and no leading boundary for German and Norwegian so compounds match (Innenwand, Yttervegg). The order puts the specific before the generic: parts named after their whole first (a stair's stringer, a curtain wall's mullion), sanitary fittings before services (a "Water Closet - Flush Valve" is a WC), furniture with doors and drawers before doors, everything before wall and slab, and weak structural words last. Sub-types come from the ignition catalogue's name patterns, so a cooktop and a refrigerator are both "appliance" with different `subtype`.

Development was honest about what it saw. Version 1 was written from general BIM vocabulary before any corpus name was read. Version 2 was tuned on half the models (23, the "dev" split) and checked on the other 22; the hard subset's names had been seen by then, so version 2's hard-subset score is optimistic and version 1's is the pre-registered one. Two bugs found by the unit tests were fixed after that: the German "Tor" (gate) matched the end of "radiator", "elevator" and "monitor" because it had no leading boundary, and "tank" in "Flush Tank" sent WCs to services.

### 3.2 IFCNet's DGCNN, ported (`tools/identify-ifcnet.py`, `src/core/identifyGeometry.js`, `src/workers/identify.worker.js`)

Port. The reference is IFCNet's own graph and checkpoint in CPU PyTorch 2.15. Input as IFCNet's preprocessing makes it: the element's triangles in IFC's Z-up frame (the viewer's (x, y, z) is IFC's (x, -z, y)), centred on the mean of the mesh's referenced vertices as stored and scaled so the farthest vertex is at 1, then 2,048 points drawn uniformly by area (seeded). IFCNet's OBJ files repeat each vertex per face, and pymesh averaged them so: against IFCNet's own point clouds this centre matches to a median 0.0003 of the unit sphere, against 0.005 for the mean of distinct vertices.

Exact rewrite. DGCNN's EdgeConv applies a 1x1 convolution to [x_j - x_i, x_i] for the 40 neighbours j of each point i, then batch norm, LeakyReLU and a max over j. With batch norm folded into the weights the pre-activation is A x_j + D x_i + b, and LeakyReLU is increasing, so max_j LeakyReLU(A x_j + D x_i + b) = LeakyReLU(max_j (A x_j) + D x_i + b): A x is computed once per point, not once per edge, and the 40-wide edge tensor is never built.

Exact pruning. Weight decay left many channels dead: their folded weights are all below 1e-6 (batch-norm scales of zero), so they output a constant. A constant channel adds nothing to neighbour distances, passes through the max unchanged and pools to itself, so its effect folds into the next layer's bias. Kept: 33 of 64, 53 of 64, 74 of 128 and 52 of 256 channels in the four EdgeConvs, 166 of 1,024 in the embedding, 507 of 512 in the first dense layer; parameters fall from 1.80 M to 0.36 M. The folded weights also held 0.9 M subnormal numbers, which put CPUs on a slow path (on one CPU thread the unpruned graph took 1.18 s at 1,024 points with them, and 0.18 s with the same weights stored as int8, which rounds them away); they are flushed to zero.

Then: ONNX opset 17 with static shapes, onnx-simplifier, and three precisions: fp32; fp16 (onnxconverter-common, fp32 inputs and outputs); int8 dynamic (onnxruntime `quantize_dynamic`, MatMul weights); and int8 weights with fp32 arithmetic ("w8": per-channel symmetric int8 initialisers behind DequantizeLinear). Point counts 512, 1,024 and 2,048, and at 512 and 1,024 also k scaled down (10, 20) to keep the neighbourhood's size; batches of 1 and 8.

IFCNet's 20 classes map onto 11 of the 18 labels (`IFCNET_LABELS` in `identifyGeometry.js`: wall, slab, door, stair, railing, beam, furniture, sanitary, heater, lighting, mep, and IfcPlate to other); it cannot say roof, window, column, covering, appliance or space.

### 3.3 Embeddings, nearest label

The element's text (Name, ObjectType, a meaningful PredefinedType, LongName) and a one-line description of each label ("a sanitary fitting: toilet, washbasin, sink, bath or shower") are embedded with mean pooling and normalised; the label with the highest cosine wins. all-MiniLM-L6-v2 in fp32 and int8, on WASM and WebGPU; multilingual-e5-small in int8 with its "query: " prefix.

### 3.4 Small LLMs, constrained

Transformers.js 3.8.1 from jsDelivr, models served locally (the download was timed separately). A short fixed prompt: the element's facts, then "Which one is it: wall, slab, ..., other? (covering = ceiling or floor finish; services = pipes, ducts, ...) Answer with one word.", and the assistant's turn starts "Answer:". Every label is one token with its leading space in Qwen's vocabulary ("services" stands for mep, "room" for space), so one forward pass gives the answer: the label token with the highest logit. The output is therefore always a label, greedy, with no generation. Free generation with parsing was not used: the constrained readout costs one forward pass and can only return a label. Qwen3 gets an empty think block. Decoding speed is measured apart (24 greedy tokens).

### 3.5 Hybrids

Rules then IFCNet (`combine` in `identifyGeometry.js`): the shape replaces the rules' answer only when the rules fell back to the box or found nothing, the shape's top class is not "other" and its probability is at least a threshold (0.5 to 0.9 tried). Rules then LLM: the prompt also carries "A rule-based guess says: X. Keep it unless it is wrong."

---

## 4. Results

Machine: Intel Core i7-14700 (20 cores, 28 threads), 32 GB, Intel UHD Graphics 770 (WebGPU adapter "intel gen-12lp", shader-f16), Windows 11, Chrome 156 headless, Node 24. Another user's six-process CPU job ran on the machine throughout, so CPU timings vary by up to a factor of two between runs; the tables give medians of 20 timed runs after three warm-up runs, and a range where a variant was timed in two runs. Memory is the growth of the browser's process tree (working set) during a method, sampled every 1.5 s.

### 4.1 The port of IFCNet works, and the optimisations keep it exact

On a stratified sample of IFCNetCore's own test split (20 objects per class, 394 in all), IFCNet's graph in PyTorch scores accuracy 0.799, balanced accuracy 0.799 and weighted F1 0.797. The paper reports balanced accuracy 0.791 and F1 0.822 for DGCNN on the whole split (Table 2 there), so checkpoint, class order and input preparation are right. The rewritten and pruned ONNX graph gives the same answer as IFCNet's graph on all 394 objects and on all 400 sampled corpus elements; logits differ by at most 0.16, from near-ties in the neighbour search, whose order depends on rounding (the unpruned rewrite in PyTorch differed on one of the 400).

Table 3. IFCNet variants. "Changed" is the share of answers that differ from IFCNet's own graph at 2,048 points (IFCNet test sample; corpus sample of 400). "Fixtures" compares the browser with onnxruntime in Python on 16 corpus clouds: answers the same, and the largest logit difference. WASM is onnxruntime-web 1.30.0 on one thread with SIMD.

| Variant | Size | Balanced accuracy | Changed | Largest drop in one class's recall | Fixtures (WASM) | WASM, 1 thread | WebGPU |
|---|---|---|---|---|---|---|---|
| IFCNet's graph, fp32, 2,048 points | 7.24 MB | 0.799 | reference | reference | n/a | 5.6 to 11.5 s | n/a |
| Rewritten, unpruned, fp32, 1,024 | 7.22 MB | 0.719 | 23.6 %; 33.3 % | 0.45 | 16 of 16, 0.10 | 188 to 190 ms | n/a |
| Rewritten and pruned, fp32, 2,048 | 1.45 MB | 0.799 | 0 %; 0 % | 0 | 16 of 16, 0.054 | 254 to 259 ms | 168 ms (16 of 16, 0.053) |
| Pruned, fp16, 2,048 | 0.74 MB | 0.794 | 1.8 %; 4.3 % | 0.05 (furniture) | 16 of 16, 0.064 | 315 to 392 ms | 149 ms (16 of 16, 0.56) |
| Pruned, int8 weights, fp32 arithmetic, 2,048 | 0.39 MB | 0.801 | 7.1 %; 10.0 % | 0.06 (stair) | 16 of 16, 0.64 | 325 to 375 ms | n/a |
| Pruned, int8 dynamic, 2,048 | 0.38 MB | 0.792 | 10.7 %; 10.7 % | 0.20 (space heater) | 16 of 16, 0.045 | 239 to 288 ms | n/a |
| Pruned, fp32, 1,024 | 1.45 MB | 0.719 | 23.6 %; 33.3 % | 0.45 (air terminal) | 16 of 16, 0.10 | 94 to 97 ms | 53 ms |
| Pruned, fp32, 1,024, k = 20 | 1.45 MB | 0.705 | 23.4 %; 32.0 % | 0.50 | 15 of 16, 0.27 | 73 ms | n/a |
| Pruned, fp32, 512 | 1.45 MB | 0.388 | 61.7 %; 69.0 % | 0.90 | 15 of 16, 0.10 | 44 to 49 ms | 28 to 44 ms |
| Pruned, fp32, 512, k = 10 | 1.45 MB | 0.362 | 64.5 %; 70.8 % | 1.00 | 16 of 16, 0.018 | 18 to 19 ms | n/a |

What each step bought, at 2,048 points on one WASM thread: the rewrite and the pruning together took an element from 5.6 to 11.5 s to 0.26 s (22 to 45 times), and the file from 7.2 to 1.45 MB, with no answer changed. Quantisation shrank the file a further two to four times but did not make it faster in WASM (the arithmetic left after pruning is small; int8 adds quantise and dequantise steps, fp16 adds casts), and it changed 2 to 11 % of the answers. The point count cannot come down: IFCNet was trained on 2,048 points, and its balanced accuracy falls to 0.72 at 1,024 and 0.39 at 512, with fewer neighbours (k scaled with the points) no better. Batches of eight cost 332 ms an element against 254 to 259 ms singly, so the worker runs one element at a time. Four WASM threads ran the 1,024-point model 2.4 times faster (37 to 41 ms against 94 to 97 ms); at 2,048 points two, four and eight threads took 142, 84 and 57 ms against 254 to 259 ms on one, with the same answers and logits as one thread (16 of 16, 0.054), but threads need cross-origin isolation, which GitHub Pages cannot provide. WebGPU on this integrated GPU was 1.5 times faster than one WASM thread at 2,048 points and needs a runtime almost twice as large (5.5 MB against 3.1 MB compressed); fp16 and int8-weight variants on WebGPU no longer reproduce the CPU's logits (largest differences 0.56 to 0.75, and at 1,024 points 9 to 11 of 16 answers the same).

Chosen: the pruned fp32 model at 2,048 points, 1.45 MB, on one WASM thread. It is the only variant that gives IFCNet's own answers on both test sets and the same logits on CPU, WASM and WebGPU within 0.054, and no other variant is faster in WASM; the 1.06 MB that fp16 or int8 would save is a third of the runtime's own download.

Functional checks of the chosen variant through the app's own module and worker (`identifyGeometry.js`, `identify.worker.js`, onnxruntime-web from jsDelivr): no NaN; 1.2 s to load the runtime, the model and the session; 211 ms an element over all 6,225 elements; the page's longest pause while the worker ran was 32 ms (a 0 ms timer loop), so it does not block the page; the browser grew by 176 MB. Failure paths, each with a message and the rules' answer kept: no model configured ("No geometry model is configured ..."), a model URL that answers 404 ("The geometry model did not download (404 Not Found)"), and a runtime that cannot be fetched, as offline ("Could not load onnxruntime-web ... Are you offline?").

### 4.2 Which method identifies elements best

Table 4. Accuracy and macro-F1 (accuracy / macro-F1) on the primary subset (6,066 typed elements), on the LLM sample's primary part (317, up to 20 per label) and on the hard subset (159 proxies and generic elements, labelled by hand); time, download and memory in the browser.

| Method | Primary (6,066) | LLM sample, primary (317) | Hard (159) | Time an element | Median model (401 elements) | Memory added |
|---|---|---|---|---|---|---|
| Rules (`identify.js`) | 0.86 / 0.83 | 0.83 / 0.83 | 0.60 / 0.54 | 6 µs | 2 ms | under 20 MB |
| Rules, first version (pre-registered) | 0.79 / 0.77 | 0.77 / 0.77 | 0.55 / 0.49 | 5 µs | 2 ms | under 20 MB |
| IFCNet DGCNN, pruned fp32, app worker (WASM) | 0.41 / 0.35 | 0.42 / 0.39 | 0.28 / 0.20 | 211 ms | 1.4 min | 176 MB |
| all-MiniLM-L6-v2 fp32, WASM | 0.44 / 0.40 | 0.48 / 0.47 | 0.33 / 0.31 | 49 ms | 20 s | 466 MB |
| all-MiniLM-L6-v2 int8, WASM | 0.45 / 0.41 | 0.50 / 0.49 | 0.33 / 0.31 | 31 ms | 12 s | 230 MB |
| all-MiniLM-L6-v2 fp32, WebGPU | 0.44 / 0.40 | 0.48 / 0.47 | 0.33 / 0.31 | 6.6 ms | 3 s | 1,473 MB |
| multilingual-e5-small int8, WASM | 0.37 / 0.38 | 0.44 / 0.43 | 0.31 / 0.28 | 51 ms | 20 s | 491 MB |
| Rules, then IFCNet (shape probability at least 0.6) | 0.853 / 0.830 | 0.849 / 0.852 | 0.585 / 0.530 | rules, plus IFCNet on 540 elements | | |
| Rules, then IFCNet (at least 0.8) | 0.862 / 0.831 | 0.830 / 0.832 | 0.597 / 0.543 | rules, plus IFCNet on 197 elements | | |
| Best LLM, Qwen3-0.6B q4f16 calibrated, WebGPU (Table 6) | n/a | 0.47 / 0.42 (160) | 0.23 / 0.35 (80) | 2.2 s | 15 min | not measured |

Downloads (model; runtime, compressed): rules nothing; IFCNet 1.45 MB and 3.1 MB; MiniLM 90 MB (fp32) or 23 MB (int8), e5-small 118 MB, and about 4.3 MB of Transformers.js and its onnxruntime build; the LLMs 0.48 to 0.79 GB and the same 4.3 MB. All but the WebGPU rows work without WebGPU. The MiniLM fp32 rows give identical answers on WASM and WebGPU, and int8 is as accurate as fp32 at a quarter of the size and 1.6 times the speed.

Per-label F1 of the rules on the primary subset: slab 0.96, space 0.96, door 0.94, covering 0.91, roof 0.90, wall 0.88, window 0.88, column 0.87, beam 0.82, railing 0.80, stair 0.79, other 0.73, furniture 0.73, mep 0.60, lighting 0.49, heater 1.00 (25 elements). On the hard subset: appliance 0.95, sanitary 0.91, beam 0.71, mep 0.69, lighting 0.67, door 0.67, other 0.62, and furniture 0.07 and slab 0.24, where the box rule calls small floor-standing things (entourage people, trees, bottles) furniture.

IFCNet on the corpus. Over all labels it is right on 41 % of the primary subset (macro-F1 0.35), because six labels are outside its classes; on the 11 labels it covers it reaches accuracy 0.61 and macro-F1 0.58 (4,107 elements). Its strengths are shapes: doors (F1 0.95 on the covered labels), furniture 0.76, stairs 0.75, walls 0.74, slabs 0.71; it is weak on services (0.14) and on "other" (0.08), which it cannot express. On the hard subset it is right on 28 % (30 % on the covered labels), with lighting 0.80, sanitary 0.64 and mep 0.44 its only useful classes. Its training data are single entities from about a thousand models, and the corpus's proxies (people, trees, site marks, parking bays) are nothing it was trained on.

Rules and IFCNet together. Letting the shape replace the rules' answer only where the rules fell back to the box or found nothing changes little, and never helps on the proxies. On the typed elements (rules 0.860 / 0.829) the pair scores 0.844 / 0.823 at a shape probability of 0.5, 0.853 / 0.830 at 0.6, 0.862 / 0.831 at 0.8 and 0.860 / 0.829 at 0.9; on the LLM sample, which weights the rare labels up, 0.6 gains two points (0.849 / 0.852 against 0.826 / 0.828) by turning lights and services that the box rule called furniture into lighting and mep; on the hard subset (rules 0.604 / 0.544) it scores 0.579 / 0.518, 0.585 / 0.530, 0.597 / 0.543 and 0.604 / 0.544. The gains are within a few elements, the losses are on the elements that matter in practice, and the shape costs 0.21 s an element against 6 µs.

Egress-critical labels (Table 5): the rules confuse doors, stairs, walls and slabs with each other in 37 of 2,395 cases (1.5 %), and lose a further 211 to other labels, mostly stairs whose flights and landings are named only by a code, and walls named as cladding or parapets. IFCNet, on the same elements, confuses them in 104 cases and loses 505.

Table 5. Rules on the primary subset: rows are the truth, columns the answer.

| Truth \ answer | door | stair | wall | slab | other labels |
|---|---|---|---|---|---|
| door (677) | 645 | 0 | 8 | 0 | 24 |
| stair (297) | 0 | 203 | 0 | 13 | 81 |
| wall (848) | 14 | 2 | 733 | 0 | 99 |
| slab (573) | 0 | 0 | 0 | 566 | 7 |

### 4.3 The small LLMs

Table 6. Qwen builds in Transformers.js 3.8.1, constrained to one label. "Raw" takes the label token with the highest logit; "calibrated" first divides out the label prior that the prompt alone gives (the same prompt with every field blank; contextual calibration, one extra forward pass per model, not per element). Samples are spread evenly over the LLM sample; n is primary / hard. Load is from the local server (the network download comes on top, section 4.4).

| Build | n | Primary sample | Hard | Time an element | Prefill tokens/s | Decode tokens/s | Load | Model download | Rules on the same elements (primary; hard) |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B q4, WebGPU | 160 / 80 | 0.34 / 0.31 | 0.04 / 0.09 | 2.4 s | 78 | 6.7 | 11.5 s | 786 MB | 0.82 / 0.82; 0.63 / 0.54 |
| same, calibrated | 160 / 80 | 0.39 / 0.34 | 0.14 / 0.25 | 2.4 s | | | | | as above |
| Qwen3-0.6B q4f16, WebGPU | 160 / 80 | 0.23 / 0.22 | 0.04 / 0.15 | 2.2 s | 89 | 20.3 | 5.1 s | 570 MB | as above |
| same, calibrated | 160 / 80 | 0.47 / 0.42 | 0.23 / 0.35 | 2.2 s | | | | | as above |
| Qwen2.5-0.5B q4, WASM | 27 / 13 | 0.18 / 0.13 | 0.00 / 0.00 | 5.7 s | 33 | 1.8 | 5.3 s | 786 MB | 0.78 / 0.77; 0.46 / 0.47 |
| Qwen2.5-0.5B int8, WASM | 27 / 13 | 0.11 / 0.04 | 0.00 / 0.00 | 4.7 s | 40 | 6.4 | 4.6 s | 512 MB | as above |
| Rules, then Qwen2.5-0.5B q4 corrects, WebGPU | 160 / 80 | 0.56 / 0.48 | 0.11 / 0.24 | 2.4 s | 84 | 6.8 | 6.8 s | 786 MB | 0.82 / 0.82; 0.63 / 0.54 |
| same, calibrated | 160 / 80 | 0.64 / 0.57 | 0.34 / 0.36 | 2.4 s | | | | | as above |
| Qwen2.5-0.5B q4f16, WebGPU (broken backend, see below) | 317 / 159 | 0.08 / 0.04 | 0.02 / 0.03 | 1.9 s | 101 | 6.1 | 6.0 s | 483 MB | 0.83 / 0.83; 0.60 / 0.54 |
| Qwen2.5-1.5B q4f16 | n/a | aborted | | | | | | 1.22 GB | |

On the same 160 and 80 elements, IFCNet scores 0.40 / 0.38 and 0.31 / 0.26, and MiniLM int8 0.54 / 0.52 and 0.40 / 0.36.

Every LLM is far below the rules on the same elements (Table 6, last column), and below the embedding models too. The models answered plain questions correctly when their backend worked ("What is the capital of France?": "Paris"; "Basic Wall:Exterior - Brick", wall, door or window: "Wall"), so the weakness is the task, not the plumbing: with eighteen labels and a record of names, codes and sizes, a model under a billion parameters picks a few favourite labels (wall and roof for Qwen2.5-0.5B, whatever the element). Calibration roughly doubles Qwen3-0.6B's accuracy and still leaves it at little more than half the rules'. Given the rules' answer as a hint, the LLM mostly makes it worse (Qwen2.5-0.5B, shown the rules' answer, scores 0.56 on the sample where the rules alone score 0.82, and 0.64 calibrated: it overturns right answers more often than wrong ones).

Backends. Qwen2.5-0.5B's q4f16 build (fp16 activations) produced nonsense on this machine's integrated GPU ("What is the capital of France?" answered "Hello, welcome to the discussion of this topic"), while its q4 build (4-bit weights, fp32 activations) on the same GPU, and every build on WASM, answered correctly; its q4f16 row in Table 6 is therefore a broken backend, not the model, and is shown only as a warning that a WebGPU build must be checked on the user's GPU before its answers are trusted. Qwen3-0.6B's q4f16 build worked. Qwen2.5-1.5B's q4f16 build (1.22 GB) aborted in Chrome on both WebGPU and WASM ("RuntimeError: Aborted()"), and its q4 build (1.79 GB) could not be downloaded in the session (Hugging Face file downloads stalled from mid-afternoon), so no model above 0.6 B was measured; Qwen2.5-3B is excluded on licence anyway.

Speed. Each element costs one forward pass over a prompt of about 190 tokens: 2.2 to 2.4 s on the GPU (prefill 78 to 89 tokens/s), 4.7 to 5.7 s on WASM (33 to 40 tokens/s), that is 15 to 16 minutes for a median model of 401 elements on the GPU and about 35 minutes on WASM. Most of the GPU time is the export's full logits tensor (every position times 151,936 vocabulary entries, about 110 MB) copied back to the CPU per element; an export that keeps only the last position's logits, and a cached key-value prefix for the fixed instructions, would each cut it, but neither is in the published builds. Free generation runs at 5.8 to 6.8 tokens/s for Qwen2.5-0.5B on the GPU, 20 tokens/s for Qwen3-0.6B, and 1.8 tokens/s for the q4 build on WASM. The one memory reading that succeeded (Qwen2.5-0.5B, WebGPU, 12 elements) grew the browser by 5.3 GB; later readings failed when the Windows process query timed out under load.

### 4.4 Downloads and first load

Measured from this machine on 2026-11-05 (Hugging Face at 10 to 18 MB/s): Qwen2.5-0.5B q4f16 483 MB in 44 s, q4 786 MB in 66 s, int8 512 MB in 43 s; Qwen3-0.6B q4f16 570 MB in 49 s; Qwen2.5-1.5B q4f16 1.22 GB in 61 s; all-MiniLM-L6-v2 int8 23 MB in 3 s, fp32 90 MB in 9 s; multilingual-e5-small int8 118 MB in 11 s. After the download, building the session took 5 to 9 s for the LLMs, 1.4 to 2.3 s for the embedding models, and 1.2 s for IFCNet (runtime, model and session from scratch). The runtimes: onnxruntime-web 1.30.0 WASM 3.1 MB compressed (14.3 MB raw); its WebGPU build 5.5 MB (28.3 MB); Transformers.js 3.8.1 0.9 MB raw plus the onnxruntime-web WebGPU build it loads (the onnxruntime-web 1.22 development build it pins: 21.6 MB raw, 4.1 MB compressed).

---

## 5. Recommendation

Ship the rules, and only the rules, as the default. `identify.js` costs nothing to download, about 6 µs an element (2 ms for a median model), works on every browser, and is the most accurate method on every subset: accuracy 0.86 and macro-F1 0.83 on the typed elements (0.87 and 0.87 on the models it was not tuned on), 0.60 and 0.54 on the hand-labelled proxies. The best model-based method on the typed elements, IFCNet's shape classifier, scores 0.41 (0.61 on the labels it can say), the embeddings 0.37 to 0.45, the LLMs 0.34 to 0.47 on the sample, and none of them adds more than a point or two when combined with the rules, and nothing on the proxies. The rules also tell a cooktop from a refrigerator (the ignition catalogue's sub-types), which no shape classifier here can.

How the app would use it (changes for the main thread, not made here):

- `ViewerManager`: an `identifyRecords()` beside `ignitionData()`, returning for every element the record `identify()` takes: `type` from `_category`, Name, ObjectType, LongName, PredefinedType or OperationType (as `elementRecords` reads them), and, for generic elements only, `dims` and `above` from `boxOf(extractMeshes(...))` and the storey's floor (the median bottom of its doors). Typed elements need no geometry: `identify()` returns their class's label at once.
- Model step: `countLabels(records)` gives the counts per label, typed elements by class and proxies by name or box; show proxies identified by name or box apart ("12 proxies read as sanitary fittings, 3 as appliances").
- `ignition.js`: no change is needed for matching (it already reads names with the same catalogue); `identify()`'s label and `subtype` can feed `rankIgnitionPoints` the proxies that only the box rule recognises, as candidates at a confidence below a name (0.4 here against 0.7).

IFCNet stays an opt-in, and off. The integration is built and tested (`identifyGeometry.js`, `identify.worker.js`): lazily loaded in its own worker, never in the main bundle, onnxruntime-web fetched from jsDelivr only when it runs, the rules' answer kept when anything fails. To turn it on the app passes `createGeometryIdentifier({ modelUrl })` a model it may serve; the first use then downloads 1.45 MB of model and 3.1 MB of runtime (cached by the browser afterwards), starts in about 1.2 s, and classifies at 0.21 to 0.26 s an element on one thread, about 16 s for the 75 proxies of the median model that has any, and 7 minutes for the worst (2,102). It is not worth shipping on this evidence (at a shape probability of 0.8 it adds 0.002 to the rules on typed elements and nothing on proxies), and it cannot be shipped until IFCNet's authors state a licence for the weights (section 1.1). `combine` defaults to 0.9, where it is harmless.

What the LLM is still good for: not identification, on this evidence, at the sizes a browser can carry. It could write a plain-language description of a model from facts the app already computes (counts per label, storeys, exits, the ranked ignition points), which this benchmark did not score; at 6 to 20 tokens a second on an integrated GPU a 100-word description (about 130 tokens) takes 7 to 22 s after a 0.5 to 0.8 GB download, so it belongs behind an explicit "Describe this model" action, WebGPU only, with the build checked on the user's GPU first (the q4f16 failure above).

Decisions for the main thread:

1. Wire `identify.js` into the Model step and ignition candidates as above (no new dependency, no download).
2. Whether to write to IFCNet's authors for a licence on the weights, or to drop the geometry path; and whether to keep `identifyGeometry.js` and its worker in the tree until then (they cost nothing until called).
3. Whether a "Describe this model" LLM feature is wanted at all, given its download and speed.

---

## 6. Limits

- The hard subset's labels are the author's, from names alone, and 20 of 179 sampled names were too vague to label; the subset therefore favours methods that read names, and it says nothing about the proxies whose names are empty or codes, the very cases a shape classifier is for. Its 159 elements come from 24 models and are dominated by "other" (66).
- The rules were tuned on half the models (the held-out half scores accuracy 0.874, macro-F1 0.865, against 0.848 and 0.809 on the tuning half), but their final hard-subset score (0.604) was measured after its names had been seen; the pre-registered first version scored 0.553.
- The primary subset has no typed appliance or sanitary fitting: no model in the corpus types them, so those labels are tested only on the hard subset.
- PredefinedType is an input, as the brief asks, and it gives doors, windows, slabs and coverings away when the exporter set it; the hard subset's elements rarely have one.
- IFCNet's numbers on its own test split use a stratified sample of 394 of its 2,379 test objects, because IFCNet's original graph needs about half a second an object on this CPU.
- One hand label was changed before any model was scored on the hard subset, but after the first rules were: a mixer tap, first labelled sanitary, became mep, as IFC types taps (IfcValve MIXING).
- No LLM above 0.6 B parameters was measured (section 4.3), and the LLM samples are 160 and 80 elements on the GPU and 27 and 13 on WASM.
- Memory readings failed for the LLM runs and the last rules run (the Windows process query timed out under load); the rules' figure is from the first run.
- Timings carry the load of another job on the same machine (section 4).

---

## 7. Reproducing

The corpus is listed in `benchmark/manifest.json` (download it to `~/ifc-samples`); IFCNet's `dgcnn_model.zip` and `IFCNetCorePly.zip` come from ifcnet.e3d.rwth-aachen.de; the Hugging Face models from the repositories in Table 1. Everything they produce stays outside the repository.

1. `python tools/identify-ifcnet.py export --weights DIR --out ONNX` (CPU PyTorch, onnx, onnxruntime, onnxsim, onnxconverter-common in a venv outside OneDrive), then `verify --ply PLY --corpus DATA/points.bin` after step 2.
2. `node tools/identify-bench.mjs extract --data DATA` (the dataset), hand labels in `DATA/hard-labels.json`, then `prepare --data DATA` and `python tools/identify-ifcnet.py fixtures --corpus DATA/fixture-points.bin`.
3. `node tools/identify-bench.mjs run --data DATA --models HF --onnx ONNX --ort ORT --methods rules,ifcnet,embed,llm,hybrid` (ORT: a folder with onnxruntime-web 1.30.0's `dist` files); `--methods llmsanity` checks an LLM build on the local GPU first. The numbers go to `benchmark/results/2026-11-05-identify.json`.
4. Unit tests: `node --test src/core/identify.test.mjs` in `topologicpy-web-frontend` (no download).
