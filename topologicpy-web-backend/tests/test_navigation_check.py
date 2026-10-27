"""Cross-checking Studio's graph against TopologicPy's Navigation graph."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.graphs import navigation_check
from app.graphs.navigation_check import NavigationGraph, align, compare, covered
from app.graphs.topologic_engine import available as topologic_available
from app.store import NavGraph

# Searched upward rather than at a fixed depth: inside the Docker image the
# tests sit at /app/tests, which has only three parents, and parents[3] raised
# IndexError at collection.
DUPLEX = next(
    (
        parent / "Ifc2x3_Duplex_Architecture.ifc"
        for parent in Path(__file__).resolve().parents
        if (parent / "Ifc2x3_Duplex_Architecture.ifc").exists()
    ),
    Path("Ifc2x3_Duplex_Architecture.ifc.missing"),
)
needs_navigation = pytest.mark.skipif(
    not (topologic_available() and navigation_check.available()),
    reason="topologicpy.Navigation not usable",
)


def two_floors(step=0.5):
    """IFC-frame points: two 6x4 m floors, 3.1 m apart, Z up."""
    xs, ys = np.arange(0, 6.01, step), np.arange(0, 4.01, step)
    floor = np.array([[x, y, 0.0] for x in xs for y in ys])
    return np.vstack([floor, floor + [0.0, 0.0, 3.1]])


def as_studio(ifc_points, offset, up_axis="y", lift=0.75):
    """What the viewer would hold: turned, shifted, lifted by the agent height."""
    turned = ifc_points @ navigation_check.TURNS[up_axis].T
    lifted = turned.copy()
    lifted[:, 1 if up_axis == "y" else 2] += lift
    return lifted + offset


class TestAlign:
    def test_recovers_the_offset(self):
        nav = two_floors()
        offset = np.array([-2.36, -2.51, -11.64])
        studio = as_studio(nav, offset)
        studio += np.random.default_rng(0).normal(0, 0.03, studio.shape)

        fit = align(nav, studio, "y")
        assert fit.offset == pytest.approx(offset + [0, 0.75, 0], abs=0.05)
        assert fit.median < 0.08

    def test_only_the_coordination_matrix_resolves_a_lattice_ambiguity(self):
        """Why the browser sends its matrix.

        Navigation covers only part of Studio's floor here. Slid by a grid
        step, that part still lands on Studio points, so no fit can tell the
        true offset from a wrong one. The viewer's matrix can.
        """
        offset = np.array([4.0, 1.0, -3.0])
        studio = as_studio(two_floors(), offset)
        nav = two_floors()
        nav = nav[nav[:, 0] > 1.0]

        fitted = align(nav, studio, "y")
        exact = navigation_check.from_coordination(
            translation_matrix(offset), nav, studio, "y", lift=0.75
        )
        assert fitted.median == pytest.approx(0.0, abs=1e-9)
        assert exact.median == pytest.approx(0.0, abs=1e-9)
        assert exact.offset == pytest.approx(offset + [0, 0.75, 0])
        assert exact.method == "coordination" and fitted.method == "fitted"

    def test_coordination_matrix_must_be_sixteen_numbers(self):
        with pytest.raises(ValueError):
            navigation_check.from_coordination([1, 0, 0], two_floors(), two_floors(), "y")

    def test_rejects_an_unknown_up_axis(self):
        with pytest.raises(ValueError):
            align(two_floors(), two_floors(), "x")


def translation_matrix(offset):
    """A three.js Matrix4's elements (column-major) for a pure translation."""
    m = np.eye(4)
    m[:3, 3] = offset
    return m.T.reshape(-1).tolist()


def test_covered_uses_the_radius():
    points = np.array([[0.0, 0, 0], [5.0, 0, 0]])
    others = np.array([[0.3, 0, 0]])
    assert covered(points, others, 0.5).tolist() == [True, False]
    assert covered(points, np.zeros((0, 3)), 0.5).tolist() == [False, False]


@needs_navigation
class TestCompare:
    def _fixture(self, drop_upper_studio=False):
        from topologicpy.TGraph import TGraph

        step = 0.5
        nav_points = two_floors(step)
        n_floor = len(nav_points) // 2
        tree_edges = []
        # Lattice links on each floor, plus one stair link between floors.
        lookup = {tuple(np.round(p, 3)): i for i, p in enumerate(nav_points)}
        for i, p in enumerate(nav_points):
            for d in ((step, 0, 0), (0, step, 0)):
                j = lookup.get(tuple(np.round(p + d, 3)))
                if j is not None:
                    tree_edges.append([i, j])
        tree_edges.append([lookup[(6.0, 4.0, 0.0)], lookup[(6.0, 4.0, 3.1)]])
        edges = np.array(tree_edges)
        nav = NavigationGraph(
            points=nav_points,
            edges=edges,
            kinds=["floor"] * len(nav_points),
            storey_of=["L1"] * n_floor + ["L2"] * n_floor,
            storeys=[
                {"id": "L1", "name": "Level 1", "elevation": 0.0},
                {"id": "L2", "name": "Level 2", "elevation": 3.1},
                {"id": "R", "name": "Roof", "elevation": 6.0},
            ],
            tgraph=TGraph.ByMeshData(nav_points.tolist(), edges.tolist()),
            parse_ms=0.0,
            graph_ms=0.0,
        )
        offset = np.array([1.0, -2.5, 4.0])
        studio = as_studio(nav_points, offset)
        if drop_upper_studio:
            studio = studio[:n_floor]
        graph = NavGraph(
            points=studio.astype(np.float32),
            edges=np.zeros((0, 2), dtype=np.int32),
            kinds=np.zeros(len(studio), dtype=np.int8),
            up_axis="y",
            meta={"agent_height": 0.75},
        )
        return graph, nav, translation_matrix(offset)

    def test_identical_space_is_fully_covered(self):
        graph, nav, matrix = self._fixture()
        report = compare(graph, nav, 0.5, coordination=matrix)
        assert report["alignment"] == "coordination"
        assert report["residual_median"] == pytest.approx(0.0, abs=1e-6)
        assert report["navigation_covered"] == 1.0
        assert report["studio_covered"] == 1.0
        assert report["notes"] == []
        assert [s["name"] for s in report["storeys"]] == ["Level 1", "Level 2"]
        assert [s["studio_nodes"] for s in report["storeys"]] == [117, 117]

    def test_a_missing_storey_shows_up_per_storey(self):
        graph, nav, matrix = self._fixture(drop_upper_studio=True)
        report = compare(graph, nav, 0.5, coordination=matrix)
        by_name = {s["name"]: s for s in report["storeys"]}
        assert by_name["Level 1"]["navigation_covered"] == 1.0
        assert by_name["Level 2"]["navigation_covered"] == 0.0
        assert by_name["Level 2"]["studio_nodes"] == 0
        assert report["navigation_covered"] == pytest.approx(0.5)

    def test_without_the_matrix_it_fits_and_says_so(self):
        graph, nav, _ = self._fixture()
        report = compare(graph, nav, 0.5)
        assert report["alignment"] == "fitted"
        assert any("No coordination matrix" in n for n in report["notes"])

    def test_routes_on_both_graphs(self):
        graph, nav, matrix = self._fixture()
        view = graph.points.astype(float)
        start, end = view[0].tolist(), view[len(view) // 2 + 5].tolist()
        report = compare(
            graph, nav, 0.5, studio_route=[start, end],
            start_point=start, end_point=end, coordination=matrix,
        )
        route = report["route"]
        assert route["found"], route["note"]
        assert route["points"][0] == pytest.approx(start, abs=1e-3)
        assert route["points"][-1] == pytest.approx(end, abs=1e-3)
        assert route["navigation_length"] > 3.1

    def test_disconnected_storeys_are_reported_as_such(self):
        graph, nav, matrix = self._fixture()
        nav.edges = nav.edges[:-1]  # drop the stair link
        from topologicpy.TGraph import TGraph

        nav.tgraph = TGraph.ByMeshData(nav.points.tolist(), nav.edges.tolist())
        view = graph.points.astype(float)
        report = compare(
            graph, nav, 0.5, start_point=view[0].tolist(), end_point=view[-1].tolist(),
            coordination=matrix,
        )
        assert not report["route"]["found"]
        assert "not connected" in report["route"]["note"]


@needs_navigation
@pytest.mark.skipif(not DUPLEX.exists(), reason="Duplex IFC not present")
class TestDuplex:
    def test_navigation_builds_both_levels(self):
        nav, _ = navigation_check.navigation_graph(DUPLEX.read_bytes(), 1.0)
        names = {s["name"] for s in nav.storeys}
        assert {"Level 1", "Level 2"} <= names
        assert len(nav.points) > 500
        assert "tread" in nav.kinds
        again, cached = navigation_check.navigation_graph(DUPLEX.read_bytes(), 1.0)
        assert cached and again is nav

    def test_endpoint_reports_coverage(self, client):
        built = client.post(
            "/api/ifc/graph",
            json={
                "floor_points": [[x * 0.5, 0.0, -y * 0.5] for x in range(16) for y in range(40)],
                "options": {"up_axis": "y", "agent_height": 0.75, "max_edge_floor": 0.8},
            },
        ).json()
        res = client.post(
            "/api/ifc/navigation-check",
            data={"graph_id": built["graph_id"], "spacing": "1.0"},
            files={"file": ("duplex.ifc", DUPLEX.read_bytes(), "application/octet-stream")},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["navigation_nodes"] > 500
        assert 0.0 <= body["studio_covered"] <= 1.0
        assert body["nodes_b64"] and body["edges_b64"]


class TestEndpointValidation:
    def _graph(self, client):
        return client.post(
            "/api/ifc/graph",
            json={
                "floor_points": [[x * 0.75, y * 0.75, 0.0] for x in range(6) for y in range(6)],
                "options": {"up_axis": "z", "agent_height": 0.0, "max_edge_floor": 0.8},
            },
        ).json()["graph_id"]

    def test_rejects_a_non_ifc_file(self, client):
        res = client.post(
            "/api/ifc/navigation-check",
            data={"graph_id": self._graph(client)},
            files={"file": ("plan.txt", b"hello", "text/plain")},
        )
        assert res.status_code == 400

    def test_rejects_a_malformed_point(self, client):
        res = client.post(
            "/api/ifc/navigation-check",
            data={"graph_id": self._graph(client), "start_point": json.dumps("north")},
            files={"file": ("a.ifc", b"ISO-10303-21;", "application/octet-stream")},
        )
        assert res.status_code == 400

    @needs_navigation
    def test_unreadable_ifc_is_a_422(self, client):
        res = client.post(
            "/api/ifc/navigation-check",
            data={"graph_id": self._graph(client)},
            files={"file": ("broken.ifc", b"ISO-10303-21; not really", "application/octet-stream")},
        )
        assert res.status_code == 422
        assert "TopologicPy" in res.json()["detail"]
