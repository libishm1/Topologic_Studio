"""Alternative routes: Yen's k shortest paths and spatially distinct routes."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from app.graphs import alternatives
from app.graphs.topologic_engine import available as topologic_available
from app.graphs.topologic_engine import has_k_shortest_paths
from app.store import NavGraph


def lattice(cells, diagonal=False):
    """A unit lattice over the given (i, j) cells, z up."""
    cells = sorted(cells)
    index = {c: n for n, c in enumerate(cells)}
    steps = [(1, 0), (0, 1)] + ([(1, 1), (1, -1)] if diagonal else [])
    edges = [
        [index[c], index[(c[0] + di, c[1] + dj)]]
        for c in cells
        for di, dj in steps
        if (c[0] + di, c[1] + dj) in index
    ]
    return NavGraph(
        points=np.array([[i, j, 0.0] for i, j in cells], dtype=np.float32),
        edges=np.array(edges, dtype=np.int32),
        kinds=np.zeros(len(cells), dtype=np.int8),
        up_axis="z",
        meta={"index": index},
    )


RING_BLOCK = {(i, j) for i in range(3, 9) for j in range(2, 6)}
RING_CELLS = [(i, j) for i in range(12) for j in range(8) if (i, j) not in RING_BLOCK]


@pytest.fixture
def ring():
    """A 12x8 floor with a solid block in the middle: two ways round it.

    Over the top is 15 m, under the bottom 17 m. Each way is a corridor two
    rows (1 m) wide, so a route along the other row is the same corridor.
    """
    return lattice(RING_CELLS)


@pytest.fixture
def corridor():
    """One straight corridor, three cells wide: no second way through."""
    return lattice([(i, j) for i in range(20) for j in range(3)])


def simple_path_costs(graph, start, end, weights):
    """Every simple path's cost, by brute force. Tiny graphs only."""
    indptr, neighbours, edge_ids = graph.csr()
    costs = []

    def walk(node, seen, cost):
        if node == end:
            costs.append(cost)
            return
        for k in range(indptr[node], indptr[node + 1]):
            nxt = int(neighbours[k])
            if nxt not in seen:
                walk(nxt, seen | {nxt}, cost + weights[edge_ids[k]])

    walk(start, {start}, 0.0)
    return sorted(costs)


class TestYen:
    def test_matches_brute_force_enumeration(self):
        graph = lattice([(i, j) for i in range(4) for j in range(3)])
        # At least the unit edge length, as hazard weights always are: A*'s
        # Euclidean heuristic is only admissible on such weights.
        weights = np.random.default_rng(7).uniform(1.0, 3.0, graph.edge_count)
        start, end = graph.meta["index"][(0, 0)], graph.meta["index"][(3, 2)]

        routes, notes = alternatives.alternative_routes(
            graph, start, end, k=8, method="yen", weights=weights
        )
        expected = simple_path_costs(graph, start, end, weights)[:8]
        assert [r.cost for r in routes] == pytest.approx(expected)
        assert notes == []
        for r in routes:
            assert len(set(r.path.node_ids)) == len(r.path.node_ids), "Yen paths are loopless"
        assert len({tuple(r.path.node_ids) for r in routes}) == len(routes)

    def test_never_uses_a_blocked_edge(self, ring):
        index = ring.meta["index"]
        start, end = index[(0, 4)], index[(11, 4)]
        blocked = np.zeros(ring.edge_count, dtype=bool)
        top = {index[(5, 6)], index[(6, 6)]}
        for e, (a, b) in enumerate(ring.edges):
            blocked[e] = {int(a), int(b)} == top

        routes, _ = alternatives.alternative_routes(
            ring, start, end, k=4, method="yen", blocked=blocked
        )
        assert routes
        for r in routes:
            pairs = {frozenset(p) for p in zip(r.path.node_ids, r.path.node_ids[1:])}
            assert frozenset(top) not in pairs

    def test_lattice_ties_make_near_duplicates(self, corridor):
        """Why "distinct" is the default: on a lattice, Yen's runners-up tie."""
        index = corridor.meta["index"]
        routes, _ = alternatives.alternative_routes(
            corridor, index[(0, 1)], index[(19, 1)], k=4, method="yen"
        )
        assert len(routes) == 4
        assert routes[1].stretch < 1.11
        assert all(r.overlap > 0.9 for r in routes[1:])

    @pytest.mark.skipif(not topologic_available(), reason="topologicpy not usable")
    def test_topologicpy_engine_says_which_search_ran(self, ring):
        index = ring.meta["index"]
        routes, notes = alternatives.alternative_routes(
            ring, index[(0, 4)], index[(11, 4)], k=3, method="yen", engine="topologicpy"
        )
        assert len(routes) == 3
        if has_k_shortest_paths():
            assert routes[0].path.engine == "topologicpy"
        else:
            assert any("0.9.87" in n for n in notes)

    @pytest.mark.skipif(not has_k_shortest_paths(), reason="needs topologicpy >= 0.9.87")
    def test_tgraph_k_shortest_paths_agrees_with_the_built_in_search(self):
        graph = lattice([(i, j) for i in range(4) for j in range(3)])
        weights = np.random.default_rng(3).uniform(1.0, 3.0, graph.edge_count)
        start, end = graph.meta["index"][(0, 0)], graph.meta["index"][(3, 2)]
        fast, _ = alternatives.alternative_routes(graph, start, end, k=6, method="yen", weights=weights)
        topo, notes = alternatives.alternative_routes(
            graph, start, end, k=6, method="yen", weights=weights, engine="topologicpy"
        )
        assert notes == []
        assert [r.path.engine for r in topo] == ["topologicpy"] * 6
        assert [r.cost for r in topo] == pytest.approx([r.cost for r in fast])


class TestDistinct:
    def test_finds_both_ways_round_the_block(self, ring):
        index = ring.meta["index"]
        routes, notes = alternatives.alternative_routes(
            ring, index[(0, 4)], index[(11, 4)], k=2, method="distinct"
        )
        assert len(routes) == 2, notes
        assert routes[0].cost == pytest.approx(15.0)
        assert routes[1].cost == pytest.approx(17.0)
        assert routes[1].stretch == pytest.approx(17 / 15)
        assert routes[1].overlap <= 0.5
        # One goes over the block, the other under it.
        ys = [max(p[1] for p in r.path.points) for r in routes]
        assert ys[0] >= 6 and min(p[1] for p in routes[1].path.points) <= 1

    def test_a_single_corridor_has_one_way_through(self, corridor):
        index = corridor.meta["index"]
        routes, notes = alternatives.alternative_routes(
            corridor, index[(0, 1)], index[(19, 1)], k=3, method="distinct"
        )
        assert len(routes) == 1
        assert any("Found 1 of 3" in n for n in notes)

    def test_stretch_limit_is_respected(self, ring):
        index = ring.meta["index"]
        routes, _ = alternatives.alternative_routes(
            ring, index[(0, 4)], index[(11, 4)], k=2, method="distinct", max_stretch=1.1
        )
        assert len(routes) == 1, "the 17 m route is 1.13x the shortest"

    def test_costs_are_reported_unpenalised(self, ring):
        index = ring.meta["index"]
        routes, _ = alternatives.alternative_routes(
            ring, index[(0, 4)], index[(11, 4)], k=2, method="distinct"
        )
        for r in routes:
            length = sum(
                float(np.linalg.norm(np.subtract(a, b)))
                for a, b in zip(r.path.points, r.path.points[1:])
            )
            assert r.cost == pytest.approx(length)

    def test_unreachable_exit_returns_nothing(self):
        graph = lattice([(0, 0), (1, 0), (5, 0), (6, 0)])
        routes, notes = alternatives.alternative_routes(graph, 0, 3, k=2)
        assert routes == []
        assert notes


class TestOverlapMeasure:
    def test_parallel_routes_half_a_metre_apart_share_space(self):
        a = np.array([[x, 0.0, 0.0] for x in range(10)], dtype=float)
        b = a + [0.0, 0.5, 0.0]
        assert alternatives._shared_share(b, alternatives._samples(a, 0.25), 1.0) == 1.0

    def test_routes_three_metres_apart_do_not(self):
        a = np.array([[x, 0.0, 0.0] for x in range(10)], dtype=float)
        b = a + [0.0, 3.0, 0.0]
        assert alternatives._shared_share(b, alternatives._samples(a, 0.25), 1.0) == 0.0


class TestEndpoint:
    def _ring_graph(self, client):
        points = [[i, j, 0.0] for i, j in RING_CELLS]
        body = client.post(
            "/api/ifc/graph",
            json={
                "floor_points": points,
                # 1.05 m links: the four lattice neighbours, no diagonals.
                "options": {"up_axis": "z", "agent_height": 0.0, "max_edge_floor": 1.05},
            },
        ).json()
        return body["graph_id"]

    def test_returns_ranked_distinct_routes(self, client):
        res = client.post(
            "/api/ifc/path/alternatives",
            json={
                "graph_id": self._ring_graph(client),
                "start_point": [0, 4, 0],
                "end_point": [11, 4, 0],
                "k": 3,
            },
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["method"] == "distinct"
        assert [r["rank"] for r in body["routes"]] == [1, 2]
        assert [r["length"] for r in body["routes"]] == pytest.approx([15.0, 17.0])
        assert body["routes"][1]["overlap"] <= 0.5
        assert "Found 2 of 3" in body["note"]

    def test_yen_method(self, client):
        body = client.post(
            "/api/ifc/path/alternatives",
            json={
                "graph_id": self._ring_graph(client),
                "start_point": [0, 4, 0],
                "end_point": [11, 4, 0],
                "k": 3,
                "method": "yen",
            },
        ).json()
        costs = [r["cost"] for r in body["routes"]]
        assert len(costs) == 3 and costs == sorted(costs)
        assert costs[0] == pytest.approx(15.0)


def test_lattice_helper_is_connected():
    graph = lattice(list(itertools.product(range(3), range(3))))
    assert graph.components()[0] == 1
