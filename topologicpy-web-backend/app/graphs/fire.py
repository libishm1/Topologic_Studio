"""Fire spread models over a navigation graph.

Three models, all operating on node indices:

``radial``
    Distance-banded ignition from the seed. Cheap, deterministic, good enough
    for a demo timeline.
``flood``
    Breadth-first spread along graph edges, so walls and closed doors shape
    the front.
``temperature``
    The diffusion model from the eCAADe 2019 work: each node relaxes toward the
    mean of its neighbours. Classic ran this as a Python double loop over every
    node and neighbour, every step. Here one step is a handful of numpy ops on
    the CSR arrays, which is what makes a live stream affordable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

from ..store import NavGraph

AMBIENT_C = 20.0
FIRE_C = 120.0


@dataclass
class FireOptions:
    model: str = "radial"
    max_steps: int = 60
    ambient: float = AMBIENT_C
    fire: float = FIRE_C
    heat_transfer: float = 1.20
    blocked: Optional[np.ndarray] = None


def radial_timeline(
    graph: NavGraph, start: int, max_steps: int = 60, step_size: Optional[float] = None
) -> List[List[int]]:
    """Ignition bands by straight-line distance from the seed."""
    if graph.node_count == 0 or not (0 <= start < graph.node_count):
        return []
    step = step_size or graph.step_size()
    step = step if step > 1e-6 else 1.0
    distances = np.linalg.norm(graph.points - graph.points[start], axis=1)
    bands = np.floor(distances / step).astype(np.int64)
    limit = min(int(bands.max()), max(0, max_steps - 1))
    return [np.flatnonzero(bands == b).tolist() for b in range(limit + 1)]


def flood_timeline(
    graph: NavGraph,
    start: int,
    max_steps: int = 60,
    blocked: Optional[np.ndarray] = None,
) -> List[List[int]]:
    """Breadth-first spread along edges."""
    n = graph.node_count
    if n == 0 or not (0 <= start < n):
        return []
    indptr, neighbours, edge_ids = graph.csr()
    seen = np.zeros(n, dtype=bool)
    seen[start] = True
    frontier = [start]
    timeline = [[start]]
    for _ in range(max(0, max_steps - 1)):
        nxt: List[int] = []
        for node in frontier:
            for k in range(indptr[node], indptr[node + 1]):
                if blocked is not None and blocked[int(edge_ids[k])]:
                    continue
                nbr = int(neighbours[k])
                if not seen[nbr]:
                    seen[nbr] = True
                    nxt.append(nbr)
        if not nxt:
            break
        timeline.append(nxt)
        frontier = nxt
    return timeline


def ignition_times(timeline: List[List[int]]) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for step, nodes in enumerate(timeline):
        for node in nodes:
            out.setdefault(int(node), step)
    return out


def temperature_steps(
    graph: NavGraph,
    start: int,
    options: FireOptions,
) -> Iterator[np.ndarray]:
    """Yield the node temperature field, one array per step.

    Vectorised over CSR: neighbour means come from ``np.add.reduceat`` on the
    sorted neighbour list, so a step costs one pass over the edges rather than
    a Python loop per node.
    """
    n = graph.node_count
    if n == 0 or not (0 <= start < n):
        return

    indptr, neighbours, edge_ids = graph.csr()
    degree = np.diff(indptr)
    has_neighbours = degree > 0

    # reduceat needs a start offset per node; nodes with no neighbours are
    # masked out afterwards rather than special-cased inside the loop.
    offsets = indptr[:-1].copy()
    offsets[~has_neighbours] = 0

    active = np.ones(len(neighbours), dtype=bool)
    if options.blocked is not None and len(neighbours):
        active = ~options.blocked[edge_ids]

    temps = np.full(n, float(options.ambient), dtype=np.float64)
    temps[start] = float(options.fire)

    safe_degree = np.where(has_neighbours, degree, 1).astype(np.float64)

    for _ in range(max(1, options.max_steps)):
        yield temps.copy()

        gathered = np.where(active, temps[neighbours], 0.0)
        if len(gathered):
            sums = np.add.reduceat(gathered, offsets)
            counts = np.add.reduceat(active.astype(np.float64), offsets)
        else:
            sums = np.zeros(n, dtype=np.float64)
            counts = np.zeros(n, dtype=np.float64)
        sums = np.where(has_neighbours, sums, 0.0)
        counts = np.where(counts > 0, counts, 1.0)

        neighbour_mean = sums / counts
        delta = neighbour_mean - temps
        # Heat only flows into a cooler cell in this simplified model.
        gain = np.where(delta > 0, delta * options.heat_transfer / safe_degree, 0.0)
        temps = temps + np.where(has_neighbours, gain, 0.0)
        temps[start] = float(options.fire)
        np.clip(temps, options.ambient, options.fire, out=temps)


def hot_nodes(temperatures: np.ndarray, threshold: float = 30.0) -> List[int]:
    return np.flatnonzero(temperatures > threshold).tolist()


def temperature_payload(
    temperatures: np.ndarray, threshold: float = 21.0
) -> Dict[str, float]:
    """Only send nodes that are meaningfully above ambient.

    Classic serialised the full temperature dict every step, which on a
    3000-node graph is most of the payload and most of the JSON parse cost on
    the client for values that are all still ambient.
    """
    indices = np.flatnonzero(temperatures > threshold)
    return {str(int(i)): round(float(temperatures[i]), 2) for i in indices}
