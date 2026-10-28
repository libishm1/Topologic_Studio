# Research gap analysis and the features it led to (2026-10-27)

This note synthesises the gap analysis of Topologic Studio's references and methods, the novelty ("scoop") check of six candidate methods, the impact analysis that chose what to build, and the verification of what was built. The data files beside it are the evidence: `gap-matrix.json` (118 references by theme, eight ranked literature gaps), `queries.json` and `recent-papers.json` (OpenAlex searches for 2021 to 2026 work), `scoop-check.json` (372 logged queries, forward-citation traces, scores), and the cross-model results `deadline-crosscheck.json`, `forecast-crosscheck.json` and `forecast-crosscheck-semantic-exits.json`.

## Method

The references of the fire-egress methods study were grouped into 20 themes and coded for what each paper does and where it stops. Recent work was searched in OpenAlex with 17 logged queries (13 searches and four page checks) and screened at title and abstract level. Every DOI cited here was checked against Crossref for title and year (56 in recent-papers.json and 40 in the gap matrix; in the scoop check 74 on Crossref and two on DataCite). The scoop check scored six candidate methods for novelty, feasibility, research value and value to users, each out of five. Paper text was treated as data throughout. The searches have known limits: OpenAlex full-text matching inflates hit counts, preprints are under-represented, ten recent papers, mostly Elsevier, had no abstract available (one was recovered), and Semantic Scholar refused most requests without an API key, so its queries should be rerun before any submission.

## The gaps

| Rank | Gap | Confidence | Studio before | Studio now |
|---|---|---|---|---|
| LG1 | Routing that reads a hazard forecast at the evacuee's own arrival time on each link, with the exactness of FIFO time-dependent shortest paths | medium to high | FIFO lemmas in Lean; hop-count escape search | Earliest safe arrival on a metric clock by time-dependent Dijkstra, in the app |
| LG2 | Fractional effective dose as an exact route objective or constraint | medium | FED equations verified and dimension-checked; dynamic programme specified | Exact time-dose trade-off of convective-heat FED, least-dose route, dose-limited deadlines, in the app |
| LG3 | Machine-checked proofs in fire safety and egress | medium to high | Lean proofs of A*, Dijkstra, FIFO, shaping, 90 equations | Extended to Yen, time-dependent Dijkstra, minimum dose, Q-learning, the eikonal front and rounding |
| LG4 | A client-side pipeline from IFC to hazard-aware routing | medium | Pipeline without published timings or physical hazard | Timings measured across 29 models; hazard is still an index |
| LG5 | Benchmarks and error metrics for IFC-derived navigation graphs | medium | Metrication bound, TopologicPy cross-check | Cross-model harnesses over 29 public models; not yet a published benchmark |
| LG6 | RL results without the exact optimum as a baseline | medium | Exact escape search and shortest length only | Policy return reported beside the exact optimal return |
| LG7 | Routing-ready surrogates with error bounds | medium | None | None; the forecast routers accept any forecast whose heat never falls |
| LG8 | Lightweight IFC-generated fire models coupled to routing | medium | Two-zone model designed, not built | Unchanged |

## Impact analysis: what to build

Each gap was weighed by the value of a feature to users, how far existing proofs and code carry it, and the novelty the scoop check found. The first two gaps share one foundation, a fire forecast on a metric clock, so they were built together; the method the scoop check recommends (C1, below) is computed from the same foundation, and so were added at small cost. LG6 needed one backward pass over the reinforcement-learning problem Studio already defines. LG7 and LG8 need a physical fire model and were deferred, as was publishing LG5 as a benchmark. The IFC-handling survey (`../ifc-handling-survey-2026-10-27.md`) ranked eight improvements to how models are read; items 1 to 5 and 7 were built, because exit identification turned out to be the weakest link in the cross-model results.

## The recommended method: escape-deadline maps (C1)

The scoop check scored the candidates as follows (novelty, feasibility, research value, user value; total out of 20).

| Candidate | N | F | R | U | Total |
|---|---|---|---|---|---|
| C1 Escape-deadline maps | 4 | 4 | 4 | 5 | 17 |
| C5 Proof-assistant methodology | 3 | 4 | 3 | 2 | 12 |
| C2 Pareto time and dose routing | 2 | 3 | 2 | 2 | 9 |
| C3 Two-zone model coupled to routing | 1 | 2 | 2 | 4 | 9 |
| C4 Hierarchical room and aperture routing | 1 | 3 | 2 | 3 | 9 |
| C6 Any-angle paths with clearance | 1 | 3 | 1 | 2 | 7 |

C2 has clear precedents in two-attribute fire egress by dynamic programming (Kostreva et al., 1991, https://doi.org/10.3801/iafss.fss.3-805) and exact Pareto routes under an exposure limit (Bi et al., 2024, https://doi.org/10.3390/app14177521), so Studio's time-dose trade-off is a feature rather than a contribution. C1 computes, for every node of the building graph, a departure time never later than the latest departure from which an occupant can still reach an exit with a fractional effective dose at or below a limit, under a growing fire; for the chosen start that departure is bracketed by bisection on an exact search; and the properties are machine-checked. No searched work does all of this. Map-based ASET/RSET evaluates tenability point by point from simulation output (Schröder et al., 2020, https://doi.org/10.1016/j.firesaf.2020.103154). Wildfire trigger buffers compute a latest time to leave at community scale against a single arrival time and without dose (Cova et al., 2005, https://doi.org/10.1111/j.1467-9671.2005.00237.x). Per-node safety under deterministic spread was computed for sensor guidance without dose or proof (Tabirca et al., 2015, https://doi.org/10.1016/j.procs.2015.05.425), and a latest-departure route exists for mine flooding without dose (Du et al., 2020, https://doi.org/10.1002/cpe.6125). The claim therefore rests on the combination and stays provisional until Chen et al. (2023, https://doi.org/10.1016/j.ress.2023.109571), known only by its title, and Tabirca et al. (2015), read only as an abstract, have been read in full.

## What was built and how it was verified

The forecast runs the fire model to its horizon and keeps each node's heat as its running maximum, so heat never falls; under that condition every link is FIFO and every link's dose is non-decreasing in its departure time. The earliest safe arrival is time-dependent Dijkstra; the time-dose trade-off is a label-setting search whose first point is the earliest safe route and whose last is the least-dose route; the start's dose-limited deadline is a bisection on that exact search; and the map over all nodes is the minimum-dose backward recursion on a time grid with link times rounded up, which can only make it earlier than the exact deadline. On 400 random graphs the trade-off equals brute force over every simple route, and at 551 nodes of 25 random lattices the map's deadline works in the exact model and is never later than the exact one. Across 29 distinct public IFC models (`forecast-crosscheck-semantic-exits.json`), the 23 with a walkable graph and doors pass every check for both fire models: the router agrees with the first point of the trade-off from every sampled start, and every sampled map deadline works and is never late. Measured in Node with the browser code, planning took at most 38 ms and the map at most 6.4 s on graphs of up to 64,000 nodes. Two Lean modules were written against this code (`formal/fireegress/FireEgress/Forecast.lean`, 87 results; `Deadline.lean`, 91), on top of time-dependent Dijkstra and minimum dose; the full build of 20 modules has 838 results, all on Lean's standard axioms. They prove that the routers meet the earlier modules' conditions, that the trade-off is complete and sound, that the deadline is exact without a dose limit and bracketed with one, and that the map is never later than the exact deadline in exact arithmetic. The proofs and their property tests also found four faults in the code, each now fixed and kept as a regression test: two routes tied in arrival time could both stay on the trade-off, so the "fastest" route could be beaten on dose; the map rounded a link down when its time was above a whole number of grid steps by at most 1e-9 of a step; and in doubles, leaving at exactly the hard deadline can pass, which the bisection had assumed could not happen.

## Limits

The fire models are graph relaxations whose heat is an index, not a measured gas temperature, so every dose here ranks routes under the model and is not a measured FED. The seconds per fire step set the clock and are the user's modelling choice. The deadline map is never later than the exact deadline, but it was up to 7.8 s earlier in tests (C20-Institute), and no proof bounds this gap. Exit doors come from the model's own flags on 15 of the 23 models; the other eight rely on a geometric rule whose precision was 69% against the flags on five models.
