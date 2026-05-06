# Research Tracks

## Fire Egress Validation

- Compare graph route output against manually annotated exit routes.
- Record whether routes cross walls, skip doors, or jump floors.
- Add per-floor route diagnostics.
- Add fixture IFCs with known floor, door, stair, and wall counts.

## Dynamic Hazard Routing

- Compare static Dijkstra, wall-aware Dijkstra, and TopologicPy hazard-weighted routing.
- Track when path changes occur during temperature spread.
- Add route cost components: geometric length, hazard penalty, blocked-edge count.

## Reinforcement Learning

- Store reward curves and convergence summaries.
- Compare learned path to hazard-weighted shortest path.
- Add deterministic seeds for reproducibility.
- Consider graph neural or policy-gradient methods only after tabular baseline metrics are reliable.

## Knowledge Graph Documentation

- Use the current wiki graph as the navigation ontology.
- Add evidence links for every future algorithm claim.
- OCR scanned papers before extracting research conclusions.

