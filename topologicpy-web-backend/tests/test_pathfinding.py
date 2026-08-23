"""Pathfinding, fire spread, RL and engine parity."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.graphs import fire as fire_model
from app.graphs import pathfinding
from app.graphs.rl import q_learning_path
from app.graphs.topologic_engine import available as topologic_available
from app.store import GraphStore, NavGraph


class TestAStar:
    def test_finds_the_manhattan_optimum(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(11, 11)]
        result = pathfinding.astar(grid_graph, start, end)
        assert result.found
        # Unit grid, no diagonals: the optimum is 22 steps of length 1.
        assert result.cost == pytest.approx(22.0)
        assert len(result.node_ids) == 23

    def test_same_node(self, grid_graph):
        result = pathfinding.astar(grid_graph, 5, 5)
        assert result.found and result.cost == 0.0

    def test_out_of_range_endpoint(self, grid_graph):
        assert not pathfinding.astar(grid_graph, 0, 9999).found

    def test_disconnected_graph_reports_no_path(self):
        graph = NavGraph(
            points=np.array([[0, 0, 0], [1, 0, 0], [50, 0, 0]], dtype=np.float32),
            edges=np.array([[0, 1]], dtype=np.int32),
            kinds=np.zeros(3, dtype=np.int8),
        )
        assert not pathfinding.astar(graph, 0, 2).found

    def test_blocked_edges_force_a_detour(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(0, 2)]
        direct = pathfinding.astar(grid_graph, start, end)
        assert direct.cost == pytest.approx(2.0)

        blocked = np.zeros(grid_graph.edge_count, dtype=bool)
        for eid in range(grid_graph.edge_count):
            a, b = grid_graph.edges[eid]
            pair = {int(a), int(b)}
            if pair == {index[(0, 0)], index[(0, 1)]} or pair == {index[(0, 1)], index[(0, 2)]}:
                blocked[eid] = True
        detour = pathfinding.astar(grid_graph, start, end, blocked=blocked)
        assert detour.found
        assert detour.cost > direct.cost

    def test_hazard_weighting_steers_the_route(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(4, 0)]

        temps = np.full(grid_graph.node_count, 20.0)
        # Make the straight run along y=0 scorching.
        for x in range(5):
            temps[index[(x, 0)]] = 120.0

        cool = pathfinding.astar(grid_graph, start, end)
        assert [int(i) for i in cool.node_ids] == [index[(x, 0)] for x in range(5)]

        weights = pathfinding.hazard_weights(grid_graph, temps, alpha=5.0)
        hot = pathfinding.astar(grid_graph, start, end, weights=weights)
        assert hot.found
        # The route should now leave the hot row rather than run straight down it.
        assert [int(i) for i in hot.node_ids] != [int(i) for i in cool.node_ids]

    def test_hazard_weights_are_identity_without_temperatures(self, grid_graph):
        weights = pathfinding.hazard_weights(grid_graph, None, alpha=5.0)
        assert np.allclose(weights, grid_graph.edge_lengths)


class TestMasksAndComponents:
    def test_lethality_threshold_blocks_hot_edges(self, grid_graph):
        temps = np.full(grid_graph.node_count, 20.0)
        temps[: grid_graph.node_count // 2] = 200.0
        mask = pathfinding.blocked_mask(grid_graph, False, temps, lethality_threshold=100.0)
        assert mask is not None and mask.any()

    def test_largest_component_ignores_orphans(self):
        graph = NavGraph(
            points=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [90, 0, 0]], dtype=np.float32
            ),
            edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
            kinds=np.zeros(4, dtype=np.int8),
        )
        component = pathfinding.largest_component(graph)
        assert component.tolist() == [True, True, True, False]

    def test_endpoint_snapping_skips_orphan_nodes(self):
        """A pick near a floating sample must not strand the search."""
        graph = NavGraph(
            points=np.array(
                [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2.05, 0, 0]], dtype=np.float32
            ),
            edges=np.array([[0, 1], [1, 2]], dtype=np.int32),
            kinds=np.zeros(4, dtype=np.int8),
        )
        component = pathfinding.largest_component(graph)
        # Node 3 is nearest but disconnected; resolution must return node 2.
        assert graph.nearest([2.04, 0, 0]) == 3
        assert pathfinding.resolve_endpoint(graph, [2.04, 0, 0], restrict=component) == 2

    def test_resolve_endpoint_accepts_legacy_ids(self, grid_graph):
        assert pathfinding.resolve_endpoint(grid_graph, None, "ifc_7") == 7
        assert pathfinding.resolve_endpoint(grid_graph, None, "ifc_99999") is None


class TestCsr:
    def test_csr_is_symmetric(self, grid_graph):
        indptr, neighbours, edge_ids = grid_graph.csr()
        assert len(neighbours) == 2 * grid_graph.edge_count
        assert indptr[-1] == len(neighbours)
        for node in range(grid_graph.node_count):
            for k in range(indptr[node], indptr[node + 1]):
                nbr = int(neighbours[k])
                back = neighbours[indptr[nbr] : indptr[nbr + 1]]
                assert node in back.tolist()

    def test_isolated_nodes_have_empty_rows(self):
        graph = NavGraph(
            points=np.array([[0, 0, 0], [1, 0, 0], [9, 9, 9]], dtype=np.float32),
            edges=np.array([[0, 1]], dtype=np.int32),
            kinds=np.zeros(3, dtype=np.int8),
        )
        indptr, _, _ = graph.csr()
        assert indptr[3] - indptr[2] == 0


class TestFire:
    def test_radial_bands_start_at_the_seed(self, grid_graph):
        timeline = fire_model.radial_timeline(grid_graph, 0, max_steps=50)
        assert timeline and timeline[0] == [0]

    def test_flood_reaches_the_whole_component(self, grid_graph):
        timeline = fire_model.flood_timeline(grid_graph, 0, max_steps=100)
        assert sum(len(step) for step in timeline) == grid_graph.node_count

    def test_flood_respects_blocked_edges(self, grid_graph):
        blocked = np.ones(grid_graph.edge_count, dtype=bool)
        timeline = fire_model.flood_timeline(grid_graph, 0, 50, blocked)
        assert timeline == [[0]]

    def test_temperature_diffuses_and_stays_bounded(self, grid_graph):
        options = fire_model.FireOptions(model="temperature", max_steps=25)
        steps = list(fire_model.temperature_steps(grid_graph, 0, options))
        assert len(steps) == 25
        first, last = steps[0], steps[-1]
        assert first[0] == pytest.approx(120.0)
        # The seed stays pinned, the rest warms up, nothing exceeds the source.
        assert last.max() <= 120.0 + 1e-6
        assert last.min() >= 20.0 - 1e-6
        assert last.mean() > first.mean()

    def test_temperature_payload_is_sparse(self, grid_graph):
        options = fire_model.FireOptions(model="temperature", max_steps=2)
        first = next(iter(fire_model.temperature_steps(grid_graph, 0, options)))
        payload = fire_model.temperature_payload(first)
        # Only the ignition node is above ambient on step 0.
        assert list(payload) == ["0"]

    def test_ignition_times_are_first_touch(self):
        assert fire_model.ignition_times([[0], [1, 2], [2, 3]]) == {0: 0, 1: 1, 2: 1, 3: 2}


class TestRL:
    def test_reaches_the_exit_on_an_open_grid(self, grid_graph):
        index = grid_graph.meta["index"]
        result = q_learning_path(
            grid_graph, index[(0, 0)], index[(2, 2)], episodes=600, max_steps=60, seed=7
        )
        assert result.path[0] == index[(0, 0)]
        # Whether it converges is stochastic; the contract is that it never
        # emits a repeated node, which Classic's rollout did.
        assert len(set(result.path)) == len(result.path)

    def test_is_reproducible_under_a_seed(self, grid_graph):
        index = grid_graph.meta["index"]
        kwargs = dict(episodes=120, max_steps=40, seed=42)
        a = q_learning_path(grid_graph, index[(0, 0)], index[(3, 3)], **kwargs)
        b = q_learning_path(grid_graph, index[(0, 0)], index[(3, 3)], **kwargs)
        assert a.path == b.path

    def test_degenerate_endpoints(self, grid_graph):
        assert q_learning_path(grid_graph, 3, 3, episodes=5).path == [3]
        assert q_learning_path(grid_graph, 0, 99999, episodes=5).path == []


class TestGraphStore:
    def test_round_trip(self, grid_graph):
        store = GraphStore(capacity=4, ttl=1000)
        gid = store.put(grid_graph)
        assert store.get(gid) is grid_graph
        assert store.get(None, "ifc") is grid_graph

    def test_lru_eviction(self, grid_graph):
        store = GraphStore(capacity=2, ttl=1000)
        ids = [store.put(grid_graph) for _ in range(3)]
        assert store.get(ids[0]) is None
        assert store.get(ids[2]) is not None

    def test_drop(self, grid_graph):
        store = GraphStore(capacity=4, ttl=1000)
        gid = store.put(grid_graph)
        assert store.drop(gid) is True
        assert store.get(gid) is None

    def test_isolation_between_graphs(self, grid_graph):
        """Two builds must not clobber each other, which Classic allowed."""
        store = GraphStore(capacity=4, ttl=1000)
        other = NavGraph(
            points=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
            edges=np.array([[0, 1]], dtype=np.int32),
            kinds=np.zeros(2, dtype=np.int8),
        )
        a = store.put(grid_graph)
        b = store.put(other)
        assert store.get(a).node_count == grid_graph.node_count
        assert store.get(b).node_count == 2


@pytest.mark.skipif(not topologic_available(), reason="topologicpy backend unavailable")
class TestTopologicParity:
    def test_engines_agree_on_cost(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(5, 5)]
        fast = pathfinding.shortest_path(grid_graph, start, end, engine="fast")
        topo = pathfinding.shortest_path(
            grid_graph, start, end, engine="topologicpy", allow_fallback=False
        )
        assert fast.found and topo.found
        assert topo.cost == pytest.approx(fast.cost, rel=1e-6)

    def test_topologic_honours_blocked_edges(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(0, 2)]
        blocked = np.zeros(grid_graph.edge_count, dtype=bool)
        for eid in range(grid_graph.edge_count):
            pair = {int(grid_graph.edges[eid, 0]), int(grid_graph.edges[eid, 1])}
            if pair == {index[(0, 0)], index[(0, 1)]}:
                blocked[eid] = True
        result = pathfinding.shortest_path(
            grid_graph, start, end, engine="topologicpy", blocked=blocked, allow_fallback=False
        )
        assert result.found
        assert index[(0, 1)] not in result.node_ids[:2]

    def test_topologic_applies_hazard_weights(self, grid_graph):
        index = grid_graph.meta["index"]
        start, end = index[(0, 0)], index[(4, 0)]
        temps = np.full(grid_graph.node_count, 20.0)
        for x in range(5):
            temps[index[(x, 0)]] = 120.0
        weights = pathfinding.hazard_weights(grid_graph, temps, alpha=5.0)
        result = pathfinding.shortest_path(
            grid_graph, start, end, engine="topologicpy", weights=weights, allow_fallback=False
        )
        assert result.found
        straight = [index[(x, 0)] for x in range(5)]
        assert result.node_ids != straight

    def test_falls_back_to_fast_when_asked(self, grid_graph):
        result = pathfinding.shortest_path(grid_graph, 0, 0, engine="topologicpy")
        assert result.found
