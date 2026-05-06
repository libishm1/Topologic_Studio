# PDF Literature Summaries

This page summarizes all local PDFs that were found and parsed. Summaries are paraphrased from extracted metadata and first pages; full text was not copied into the wiki.

## A Dynamic Approach For Evacuees' Distribution And Optimal Routing In Hazardous Environments

File: `A dynamic approach for optimal routing in hazardous environments considering evacuees' distribution.pdf`

- Pages: 19
- Authors from first page: Pawel Boguslawski, Lamine Mahdjoubi, Vadim Zverovich, Fodil Fadli.
- Core idea: emergency routing should not rely only on static hazard scenarios. The paper compares static, semi-dynamic, and dynamic approaches where occupant density and hazard influence are updated over time and used for route calculation.
- Relevance to Topologic Studio: the current system already models dynamic hazard effects through streamed fire state and dynamic rerouting. It does not yet model occupant density, congestion, or density databases. This paper supports a future `density_weight` or crowd-coupled routing layer.
- Current adoption: partial. The project has dynamic fire/rerouting but not occupant distribution.
- Future extraction target: encode density as a node attribute, add time-indexed occupancy to `LAST_GRAPHS["ifc"]`, and make edge cost combine distance, hazard, and density.

## Analytic Prioritization Of Indoor Routes For Search And Rescue Operations In Hazardous Environments

File: `Computer aided Civil Eng - 2017 - Zverovich - Analytic Prioritization of Indoor Routes for Search and Rescue Operations in.pdf`

- Pages: 21
- Title metadata: "Analytic Prioritization of Indoor Routes for Search and Rescue Operations in Hazardous Environments".
- Core idea: route choice can be treated as a multi-criteria prioritization problem, integrating AHP, hazard proximity, travel distance/time, route complexity, adapted Duckham-Kulik logic, Dijkstra, and binary search.
- Relevance to Topologic Studio: the current implementation uses shortest path, wall-aware filtering, and a hazard-weighted edge cost during temperature simulation. It does not yet expose AHP weights, route complexity, signage complexity, number of turns, or SAR-specific route priorities.
- Current adoption: conceptual, not full algorithmic reproduction.
- Future extraction target: add route complexity features such as turns, stairs traversed, door count, path width estimates, and hazard proximity index, then expose a multi-objective route mode.

## The Synergy Of Non-Manifold Topology And Reinforcement Learning For Fire Egress

Files:

- `eCAADe_2019_asSubmitted_ORCA.pdf`
- `ecaadesigradi2019_671.pdf`
- `TopologicStudio/the synergy of non manifold topology.pdf`

The first two PDFs contain extractable text and appear to be versions of the same paper. The `TopologicStudio/the synergy of non manifold topology.pdf` file has metadata pointing to `ecaadesigradi2019_671.pdf` but no extractable text from the first pages; it should be treated as a duplicate or scanned copy until OCR/manual inspection confirms otherwise.

- Pages: 12, 10, and 10 respectively.
- Authors from extractable versions: Wassim Jabi, Aikaterini Chatzivasileiadi, Nicholas Mario Wardhana, Simon Lannon, Robert Aish.
- Core idea: non-manifold topology can provide a graph-like spatial representation for fire egress experiments, and reinforcement learning can learn escape policies in dynamic fire scenarios.
- Relevance to Topologic Studio: this is the closest conceptual ancestor. The current system uses TopologicPy graph objects, graph fire propagation, and a tabular Q-learning endpoint.
- Current adoption: partial. Topologic Studio adopts the NMT/graph/RL framing, but the current browser workflow uses IFC-derived sampled graph nodes rather than a manually authored simplified NMT game environment.
- Current mismatch: the README claims RL path training is done, and the backend exposes `/rl/train`. The frontend request currently sends `mode: graphMode`; for IFC workflows this may use `cell` or `wire` instead of `ifc` unless verified. Treat IFC-specific RL as needing validation.
- Future extraction target: reproduce the paper's state/action/reward assumptions explicitly in docs and tests, then compare one-exit and multi-exit cases.

## Emergency Response In Complex Buildings: Automated Selection Of Safest And Balanced Routes

File: `Safest and Balanced Routes - ver11 - double-column.pdf`

- Pages: 16
- Authors from first page: V. Zverovich, L. Mahdjoubi, P. Boguslawski, F. Fadli, H. Barki.
- Core idea: safest and balanced routes combine Dijkstra-style search with hazard proximity numbers, hazard propagation coefficient, proximity index, and multi-attribute decision-making.
- Relevance to Topologic Studio: the current `path_alpha` hazard weight is a simple scalar version of balanced routing. It penalizes high-temperature edges during dynamic rerouting.
- Current adoption: simplified. The system does not yet compute formal proximity indices or multi-epicenter hazard models.
- Future extraction target: represent each fire source as an epicenter, compute per-edge proximity and route-level proximity, and expose "shortest", "safest", and "balanced" route modes.

## A Framework For Two-Way Coupled Fire And Egress Modelling In Complex Buildings Using BIM

File: `Thesis_IMFSE_-_Muhammad_Ridha_Tantowi.pdf`

- Pages: 144
- Author metadata: Muhammad Ridha Tantowi.
- Core idea from first pages: fire and egress models can be coupled through BIM so fire conditions and occupant movement affect each other.
- Relevance to Topologic Studio: the project currently has one-way coupling from graph fire state into route recomputation. It does not yet feed evacuation/crowd state back into fire or smoke conditions.
- Current adoption: conceptual roadmap only.
- Future extraction target: define coupling boundary objects, exchange fields, simulation clock, hazard fields, occupant density fields, and validation scenarios.

## Two-Graph Building Interior Representation For Emergency Response Applications

File: `TWO-GRAPH_BUILDING_INTERIOR_REPRESENTATION_FOR_EME.pdf`

- Pages: 6
- Authors from first page: P. Boguslawski, L. Mahdjoubi, V. Zverovich, F. Fadli.
- Core idea: a full 3D topological model and navigable network can support indoor navigation, safe routes, hazard spread, and emergency response. The paper emphasizes a two-graph representation and variable-density navigable networks.
- Relevance to Topologic Studio: directly relevant to the current distinction between building geometry/topology and navigation graph overlays.
- Current adoption: partial. The current system builds a navigation graph from IFC slabs/stairs/doors/walls but does not yet maintain a separate semantic dual graph with rich room/door/corridor relationships.
- Future extraction target: add explicit room/cell nodes and a dual navigation graph where rooms, doors, stairs, and corridors are first-class graph entities rather than only sampled point nodes.

## Literature-To-System Mapping

| Research theme | Local reference | Current adoption | Gap |
|---|---|---|---|
| Non-manifold topology + RL | Jabi et al. fire egress papers | TopologicPy imports, graph fire, Q-learning endpoint | IFC RL behavior needs validation; state model is not documented in tests |
| Dynamic hazard routing | Boguslawski dynamic routing paper | SSE fire and dynamic path rerouting | No occupant density or temporal density database |
| Safest/balanced routes | Zverovich safest/balanced paper | Hazard-weighted cost scalar | No formal proximity index or AHP weighting |
| SAR route prioritization | Zverovich 2017 paper | Dijkstra and hazard weighting | No route complexity metrics |
| BIM/fire/egress coupling | Tantowi thesis | BIM/IFC input and fire-to-route coupling | No two-way fire-egress coupling |
| Two-graph interiors | Boguslawski two-graph paper | IFC geometry -> sampled navigation graph | No explicit room/door dual graph |

