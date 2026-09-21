"""Benchmark TopologicPy's TGraph against the legacy Graph and our native A*.

Background: on topologicpy 0.9.64 the legacy ``Graph.ShortestPath`` cost ~1.4 s
per query on a 2,650-node egress graph versus ~14 ms for the built-in A*, which
is why the fast engine is the default. 0.9.7x ships ``TGraph``, a different
class with an index-based constructor (``ByEdgeIndexPairs``) and a ``useNumba``
option on ``ShortestPath``. If that is fast, the recommendation should change.

Run:
    .venv-next/Scripts/python tools/bench_tgraph.py [nodes_per_side]
"""
from __future__ import annotations

import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "topologicpy-web-backend"))

import numpy as np  # noqa: E402


def grid(side: int):
    """A side x side lattice: the shape of one storey of a navigation graph."""
    coords = []
    index = {}
    for i in range(side):
        for j in range(side):
            index[(i, j)] = len(coords)
            coords.append([float(i), 0.0, float(j)])
    edges = []
    for i in range(side):
        for j in range(side):
            for di, dj in ((1, 0), (0, 1)):
                other = (i + di, j + dj)
                if other in index:
                    edges.append([index[(i, j)], index[other]])
    return np.asarray(coords, dtype=np.float32), np.asarray(edges, dtype=np.int32), index


def timed(label, fn, repeats=1):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    best = min(times)
    unit = f"{best * 1000:.1f} ms" if best < 1 else f"{best:.2f} s"
    print(f"  {label:<52} {unit:>10}")
    return out, best


def main() -> None:
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 51
    coords, edges, index = grid(side)
    n, m = len(coords), len(edges)
    src, dst = index[(0, 0)], index[(side - 1, side - 1)]
    print(f"\nGrid {side}x{side}: {n} nodes, {m} edges   (egress graphs are ~2.6k / ~17k)\n")

    import topologicpy

    print(f"topologicpy {topologicpy.__version__}\n")

    # ---------------------------------------------------------- native A*
    from app.graphs.pathfinding import astar
    from app.store import NavGraph

    nav = NavGraph(
        points=coords,
        edges=edges,
        kinds=np.zeros(n, dtype=np.int8),
        up_axis="y",
    )
    nav.csr()  # warm the CSR build so we time the search, not the indexing
    print("built-in engine")
    result, fast_t = timed("astar", lambda: astar(nav, src, dst), repeats=3)
    print(f"    -> found={result.found} cost={result.cost:.2f} hops={len(result.node_ids)}\n")

    # ------------------------------------------------------------- TGraph
    from topologicpy.TGraph import TGraph

    print("TGraph (0.9.7x)")
    tg, _ = timed(
        "ByEdgeIndexPairs (build)",
        lambda: TGraph.ByEdgeIndexPairs(
            coords.tolist(), edges.tolist(), silent=True
        ),
    )
    if tg is None:
        print("    -> ByEdgeIndexPairs returned None; trying ByMeshData")
        tg, _ = timed(
            "ByMeshData (build)",
            lambda: TGraph.ByMeshData(coords.tolist(), edges.tolist()),
        )
    if tg is None:
        print("    -> could not build a TGraph; skipping")
        return

    for numba in (False, True):
        for astar_on in (False, True):
            label = f"ShortestPath useNumba={numba} useAStar={astar_on}"
            try:
                out, t = timed(
                    label,
                    lambda nb=numba, a=astar_on: TGraph.ShortestPath(
                        tg, src, dst, useNumba=nb, useAStar=a,
                        returnVertices=True, silent=True,
                    ),
                    repeats=2,
                )
                hops = len(out[1]) if isinstance(out, (list, tuple)) and len(out) > 1 else "?"
                print(f"    -> {type(out).__name__}, vertices={hops}")
            except Exception as exc:
                print(f"    -> FAILED: {type(exc).__name__}: {exc}")

    # ------------------------------------------------- legacy Graph, for scale
    print("\nlegacy Graph (for comparison)")
    from topologicpy.Dictionary import Dictionary
    from topologicpy.Graph import Graph

    vd = [Dictionary.ByKeysValues(["node_id"], [i]) for i in range(n)]
    ed = [Dictionary.ByKeysValues(["cost", "Length"], [1.0, 1.0]) for _ in edges]
    g, _ = timed(
        "ByMeshData(ontology=False) (build)",
        lambda: Graph.ByMeshData(
            coords.tolist(), edges.tolist(),
            vertexDictionaries=vd, edgeDictionaries=ed, ontology=False,
        ),
    )
    if g is not None:
        from topologicpy.Topology import Topology

        lookup = {}
        for v in Graph.Vertices(g):
            d = Topology.Dictionary(v)
            if d is not None:
                nid = Dictionary.ValueAtKey(d, "node_id")
                if nid is not None:
                    lookup[int(nid)] = v
        if src in lookup and dst in lookup:
            timed(
                "ShortestPath useAStar=True",
                lambda: Graph.ShortestPath(
                    g, lookup[src], lookup[dst], edgeKey="cost",
                    useAStar=True, silent=True,
                ),
            )

    print(f"\nReference: the built-in engine did it in {fast_t * 1000:.1f} ms\n")


if __name__ == "__main__":
    main()
