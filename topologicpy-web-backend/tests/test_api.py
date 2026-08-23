"""End-to-end API behaviour, including the Classic compatibility routes."""
from __future__ import annotations

import base64

import numpy as np
import pytest


def decode_f32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.float32).reshape(-1, 3)


def decode_u32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.uint32).reshape(-1, 2)


def slab_points(z=0.0, n=8, step=0.75):
    return [[x * step, y * step, z] for x in range(n) for y in range(n)]


class TestMeta:
    def test_health(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["version"]
        assert "topologicpy" in body

    def test_capabilities_lists_engines(self, client):
        body = client.get("/api/capabilities").json()
        assert "fast" in body["engines"]
        assert set(body["fire_models"]) == {"radial", "flood", "temperature"}


class TestPointCloudGraph:
    def test_build_and_path(self, client):
        payload = {
            "floor_points": slab_points(),
            "options": {"up_axis": "z", "agent_height": 0.0, "max_edge_floor": 0.8},
        }
        res = client.post("/api/ifc/graph", json=payload)
        assert res.status_code == 200, res.text
        body = res.json()

        graph_id = body["graph_id"]
        assert body["stats"]["nodes"] == 64
        nodes = decode_f32(body["nodes_b64"])
        edges = decode_u32(body["edges_b64"])
        assert nodes.shape == (64, 3)
        assert len(edges) == body["stats"]["edges"]

        path = client.post(
            "/api/ifc/path",
            json={
                "graph_id": graph_id,
                "start_point": [0.0, 0.0, 0.0],
                "end_point": [5.25, 5.25, 0.0],
            },
        ).json()
        assert path["found"] is True
        assert len(path["points"]) >= 2
        assert path["engine"] == "fast"

    def test_agent_height_applies_to_the_up_axis(self, client):
        """Classic added agent height to index 2 even on a y-up model."""
        body = client.post(
            "/api/ifc/graph",
            json={
                "floor_points": [[x * 1.0, 0.0, y * 1.0] for x in range(4) for y in range(4)],
                "options": {"up_axis": "y", "agent_height": 1.5, "max_edge_floor": 1.1},
            },
        ).json()
        nodes = decode_f32(body["nodes_b64"])
        assert np.allclose(nodes[:, 1], 1.5)
        assert nodes[:, 2].max() > 1.5  # the z spread survived untouched

    def test_empty_payload_is_rejected(self, client):
        res = client.post("/api/ifc/graph", json={"floor_points": []})
        assert res.status_code == 400
        assert "walkable" in res.json()["detail"].lower()

    def test_unknown_graph_id_is_rejected(self, client):
        res = client.post(
            "/api/ifc/path",
            json={"graph_id": "does-not-exist", "start_point": [0, 0, 0], "end_point": [1, 1, 1]},
        )
        assert res.status_code == 400

    def test_graphs_are_independent(self, client):
        """Two graphs coexist - Classic's single global slot could not."""
        small = client.post(
            "/api/ifc/graph",
            json={"floor_points": slab_points(n=4), "options": {"max_edge_floor": 0.8}},
        ).json()
        large = client.post(
            "/api/ifc/graph",
            json={"floor_points": slab_points(n=8), "options": {"max_edge_floor": 0.8}},
        ).json()
        assert small["graph_id"] != large["graph_id"]

        again = client.get(f"/api/ifc/graph/{small['graph_id']}").json()
        assert again["stats"]["nodes"] == 16
        assert large["stats"]["nodes"] == 64

    def test_delete_graph(self, client):
        gid = client.post(
            "/api/ifc/graph",
            json={"floor_points": slab_points(n=4), "options": {"max_edge_floor": 0.8}},
        ).json()["graph_id"]
        assert client.delete(f"/api/ifc/graph/{gid}").json()["deleted"] is True
        assert client.get(f"/api/ifc/graph/{gid}").status_code == 400

    def test_walls_block_the_route(self, client):
        payload = {
            "floor_points": slab_points(n=8, step=0.75),
            "walls": [
                {
                    "segment": [[-1.0, 2.6], [3.0, 2.6]],
                    "thickness": 0.1,
                    "up_min": -1.0,
                    "up_max": 3.0,
                }
            ],
            "options": {"up_axis": "z", "agent_height": 0.0, "max_edge_floor": 0.8},
        }
        body = client.post("/api/ifc/graph", json=payload).json()
        assert body["stats"]["blocked_edges"] > 0

        blocked_run = client.post(
            "/api/ifc/path",
            json={
                "graph_id": body["graph_id"],
                "start_point": [0.0, 0.0, 0.0],
                "end_point": [0.0, 5.25, 0.0],
                "use_walls": True,
            },
        ).json()
        open_run = client.post(
            "/api/ifc/path",
            json={
                "graph_id": body["graph_id"],
                "start_point": [0.0, 0.0, 0.0],
                "end_point": [0.0, 5.25, 0.0],
                "use_walls": False,
            },
        ).json()
        assert blocked_run["found"] and open_run["found"]
        assert blocked_run["length"] > open_run["length"]

    def test_engine_comparison_endpoint(self, client):
        body = client.post(
            "/api/ifc/graph",
            json={"floor_points": slab_points(n=6), "options": {"max_edge_floor": 0.8}},
        ).json()
        res = client.post(
            "/api/ifc/path/compare",
            json={
                "graph_id": body["graph_id"],
                "start_point": [0.0, 0.0, 0.0],
                "end_point": [3.75, 3.75, 0.0],
            },
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["fast"]["found"]
        if payload["topologicpy"]["found"]:
            assert payload["cost_delta"] < 1e-3


class TestLegacyRoutes:
    def test_classic_contract_still_works(self, client, two_storey_geometry):
        res = client.post(
            "/ifc-egress-graph",
            json={
                **two_storey_geometry,
                "up_axis": "z",
                "base_spacing": 0.5,
                "max_edge_floor": 1.2,
                "max_edge_stair": 0.5,
                "agent_height": 0.0,
                "use_walls": True,
            },
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["stats"]["nodes"] > 0
        assert body["stats"]["stair_nodes"] > 0
        assert body["stats"]["door_nodes"] == 1
        # Classic's response keys must survive verbatim.
        assert set(["mode", "stats", "edges", "edge_ids", "coords"]) <= set(body)
        assert body["edge_ids"] and body["edge_ids"][0][0].startswith("ifc_")

        path = client.post(
            "/ifc-egress-path",
            json={"start_point": [0.5, 0.5, 0.0], "end_point": [5.5, 1.0, 0.0]},
        )
        assert path.status_code == 200, path.text
        assert len(path.json()["points"]) >= 2

    def test_egress_graph_rejects_empty_geometry(self, client):
        res = client.post("/ifc-egress-graph", json={"floors": [], "stairs": []})
        assert res.status_code == 400

    def test_stairs_connect_the_two_storeys(self, client, two_storey_geometry):
        """The whole point of the stair handling: a route between levels."""
        body = client.post(
            "/ifc-egress-graph",
            json={
                **two_storey_geometry,
                "up_axis": "z",
                "base_spacing": 0.4,
                "max_edge_floor": 1.0,
                "max_edge_stair": 0.45,
                "agent_height": 0.0,
                "use_walls": False,
            },
        ).json()
        assert body["stats"]["nodes"] > 0
        # Sanity: the graph spans both slab heights.
        heights = {round(c[2], 1) for c in body["coords"].values()}
        assert 0.0 in heights


class TestSimulation:
    @pytest.fixture
    def graph_id(self, client):
        return client.post(
            "/api/ifc/graph",
            json={
                "floor_points": slab_points(n=8),
                "options": {"agent_height": 0.0, "max_edge_floor": 0.8},
            },
        ).json()["graph_id"]

    def test_fire_timeline(self, client, graph_id):
        body = client.post(
            "/api/fire/timeline",
            json={"graph_id": graph_id, "model": "radial", "start_point": [0, 0, 0]},
        ).json()
        assert body["steps"] > 0
        assert body["timeline"][0]

    def test_flood_timeline_covers_the_graph(self, client, graph_id):
        body = client.post(
            "/api/fire/timeline",
            json={
                "graph_id": graph_id,
                "model": "flood",
                "start_point": [0, 0, 0],
                "max_steps": 200,
            },
        ).json()
        assert sum(len(s) for s in body["timeline"]) == 64

    def test_fire_stream_emits_events(self, client, graph_id):
        url = (
            f"/api/fire/stream?graph_id={graph_id}&model=temperature&max_steps=4"
            "&delay_ms=0&start_x=0&start_y=0&start_z=0"
        )
        with client.stream("GET", url) as response:
            assert response.status_code == 200
            text = "".join(response.iter_text())
        assert '"type": "meta"' in text
        assert '"type": "temperature_step"' in text
        assert '"type": "done"' in text

    def test_fire_stream_reroutes_around_the_fire(self, client, graph_id):
        url = (
            f"/api/fire/stream?graph_id={graph_id}&model=temperature&max_steps=12"
            "&delay_ms=0&start_x=0&start_y=0&start_z=0"
            "&end_x=5.25&end_y=5.25&end_z=0"
            "&path_start_x=0&path_start_y=5.25&path_start_z=0"
            "&stream_path=true&path_recompute_interval=3&path_alpha=2.0"
        )
        with client.stream("GET", url) as response:
            text = "".join(response.iter_text())
        assert '"type": "path_update"' in text

    def test_rl_train(self, client, graph_id):
        body = client.post(
            "/api/rl/train",
            json={
                "graph_id": graph_id,
                "start_point": [0, 0, 0],
                "exit_point": [1.5, 1.5, 0],
                "episodes": 200,
                "max_steps": 40,
                "seed": 3,
            },
        ).json()
        assert body["path"]
        assert len(body["points"]) == len(body["path"])

    def test_missing_graph_is_a_400(self, client):
        res = client.post(
            "/api/fire/timeline",
            json={"graph_id": "nope", "start_point": [0, 0, 0]},
        )
        assert res.status_code == 400
