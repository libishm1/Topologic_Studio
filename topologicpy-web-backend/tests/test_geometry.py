"""Geometry pipeline: sampling, adjacency, obstacles."""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.geometry.adjacency import (
    KIND_DOOR,
    KIND_FLOOR,
    KIND_STAIR,
    build_adjacency,
    dedupe_edges,
    snap_to_grid,
)
from app.geometry.common import axis_index, horizontal_axes
from app.geometry.obstacles import (
    WallField,
    door_positions_from_geometry,
    edge_block_mask,
    segments_intersect_2d,
    wall_segments_from_geometry,
)
from app.geometry.sampling import collapse_columns, decimate, sample_walkable_points


class TestAxes:
    def test_axis_index(self):
        assert axis_index("x") == 0
        assert axis_index("y") == 1
        assert axis_index("z") == 2
        assert axis_index("nonsense") == 2

    def test_horizontal_axes_excludes_up(self):
        assert horizontal_axes("y") == (0, 2)
        assert horizontal_axes("z") == (0, 1)


class TestSampling:
    def test_flat_slab_is_sampled(self):
        verts = [0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0]
        indices = [0, 1, 2, 0, 2, 3]
        pts = sample_walkable_points(verts, indices, spacing=1.0, up_axis="z")
        assert len(pts) > 4
        assert np.allclose(pts[:, 2], 0.0)

    def test_vertical_wall_is_rejected(self):
        # A wall in the xz plane: its normal is horizontal, so nothing walkable.
        verts = [0, 0, 0, 4, 0, 0, 4, 0, 3, 0, 0, 3]
        indices = [0, 1, 2, 0, 2, 3]
        pts = sample_walkable_points(verts, indices, spacing=0.5, up_axis="z", max_slope_deg=10)
        assert len(pts) == 0

    def test_slope_tolerance_is_respected(self):
        # A 30 degree ramp in the xz plane.
        rise = math.tan(math.radians(30)) * 4
        verts = [0, 0, 0, 4, 0, rise, 4, 4, rise, 0, 4, 0]
        indices = [0, 1, 2, 0, 2, 3]
        assert len(sample_walkable_points(verts, indices, spacing=1.0, max_slope_deg=10)) == 0
        assert len(sample_walkable_points(verts, indices, spacing=1.0, max_slope_deg=45)) > 0

    def test_roof_exclusion(self):
        verts = [0, 0, 10, 4, 0, 10, 4, 4, 10, 0, 4, 10]
        indices = [0, 1, 2, 0, 2, 3]
        assert len(sample_walkable_points(verts, indices, spacing=1.0, exclude_above=9.0)) == 0
        assert len(sample_walkable_points(verts, indices, spacing=1.0, exclude_above=11.0)) > 0

    def test_max_points_is_honoured(self):
        verts = [0, 0, 0, 40, 0, 0, 40, 40, 0, 0, 40, 0]
        indices = [0, 1, 2, 0, 2, 3]
        pts = sample_walkable_points(verts, indices, spacing=0.2, max_points=50)
        assert len(pts) <= 50

    def test_degenerate_input_is_safe(self):
        assert len(sample_walkable_points([], [])) == 0
        assert len(sample_walkable_points([0, 0, 0], [0, 1, 2])) == 0
        # Indices pointing past the vertex buffer must not raise.
        assert len(sample_walkable_points([0, 0, 0, 1, 0, 0, 1, 1, 0], [0, 1, 99])) == 0

    def test_up_axis_y(self):
        # Slab lying in the xz plane, y up. Wound so the normal points +y;
        # sampling only accepts upward-facing surfaces.
        verts = [0, 0, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4]
        up_wound = [0, 2, 1, 0, 3, 2]
        assert len(sample_walkable_points(verts, up_wound, spacing=1.0, up_axis="y")) > 0
        # The same slab is a wall when z is treated as up.
        assert len(sample_walkable_points(verts, up_wound, spacing=1.0, up_axis="z")) == 0
        # Reversing the winding makes it a soffit, which is not walkable.
        down_wound = [0, 1, 2, 0, 2, 3]
        assert len(sample_walkable_points(verts, down_wound, spacing=1.0, up_axis="y")) == 0

    def test_collapse_columns_keeps_the_walking_surface(self):
        """A slab is sampled top and underside; only the top is walkable."""
        pts = np.array(
            [
                [0.0, 0.0, 0.00],  # slab underside
                [0.0, 0.0, 0.15],  # slab top  <- keep
                [1.0, 0.0, 0.00],
                [1.0, 0.0, 0.15],  # <- keep
            ],
            dtype=np.float32,
        )
        out = collapse_columns(pts, cell=0.5, up_axis="z", gap=0.9)
        assert len(out) == 2
        assert np.allclose(sorted(out[:, 2]), [0.15, 0.15])

    def test_collapse_columns_keeps_separate_storeys(self):
        pts = np.array(
            [
                [0.0, 0.0, 0.00],
                [0.0, 0.0, 0.15],
                [0.0, 0.0, 3.00],  # storey above
                [0.0, 0.0, 3.15],
            ],
            dtype=np.float32,
        )
        out = collapse_columns(pts, cell=0.5, up_axis="z", gap=0.9)
        assert len(out) == 2
        assert np.allclose(sorted(out[:, 2]), [0.15, 3.15])

    def test_collapse_columns_respects_up_axis(self):
        pts = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.15, 0.0]], dtype=np.float32
        )
        out = collapse_columns(pts, cell=0.5, up_axis="y", gap=0.9)
        assert len(out) == 1
        assert out[0][1] == pytest.approx(0.15)

    def test_decimate_removes_duplicates(self):
        pts = np.array([[0, 0, 0], [0.01, 0, 0], [5, 5, 5]], dtype=np.float32)
        assert len(decimate(pts, 0.5)) == 2


class TestAdjacency:
    def test_unit_grid_connects_neighbours(self):
        pts = np.array(
            [[x, y, 0.0] for x in range(4) for y in range(4)], dtype=np.float32
        )
        kinds = np.zeros(len(pts), dtype=np.int8)
        edges = build_adjacency(pts, kinds, max_edge_floor=1.01, max_edge_stair=0.4)
        # 4x4 grid: 12 horizontal + 12 vertical links, no diagonals at r=1.01.
        assert len(edges) == 24
        assert (edges[:, 0] < edges[:, 1]).all()

    def test_floor_links_stay_level(self):
        """Two stacked sheets must not brace into a space-frame truss."""
        lower = [[x * 0.5, y * 0.5, 0.0] for x in range(5) for y in range(5)]
        upper = [[x * 0.5, y * 0.5, 0.15] for x in range(5) for y in range(5)]
        pts = np.asarray(lower + upper, dtype=np.float32)
        kinds = np.zeros(len(pts), dtype=np.int8)

        braced = build_adjacency(
            pts, kinds, max_edge_floor=1.0, max_edge_stair=0.4, max_edge_rise=5.0
        )
        flat = build_adjacency(
            pts, kinds, max_edge_floor=1.0, max_edge_stair=0.4, max_edge_rise=0.05
        )

        def rises(edges):
            return np.abs(pts[edges[:, 1], 2] - pts[edges[:, 0], 2])

        assert (rises(braced) > 0.05).any(), "control: bracing exists without the limit"
        assert not (rises(flat) > 0.05).any(), "no sloped floor links survive"
        assert len(flat) < len(braced)

    def test_rectilinear_rejects_diagonals(self):
        pts = np.array(
            [[x, y, 0.0] for x in range(4) for y in range(4)], dtype=np.float32
        )
        kinds = np.zeros(len(pts), dtype=np.int8)
        loose = build_adjacency(pts, kinds, max_edge_floor=1.5, max_edge_stair=0.4)
        strict = build_adjacency(
            pts, kinds, max_edge_floor=1.5, max_edge_stair=0.4, rectilinear=True
        )
        assert len(strict) < len(loose)
        assert len(strict) == 24

    def test_stair_and_floor_are_bridged_at_landings(self):
        floor = [[x * 0.5, 0.0, 0.0] for x in range(6)]
        stair = [[2.5, 0.0, 0.2 * (i + 1)] for i in range(8)]
        pts = np.asarray(stair + floor, dtype=np.float32)
        kinds = np.asarray([KIND_STAIR] * len(stair) + [KIND_FLOOR] * len(floor), dtype=np.int8)
        edges = build_adjacency(pts, kinds, max_edge_floor=0.6, max_edge_stair=0.25)

        stair_ids = set(range(len(stair)))
        crossing = [
            e for e in edges
            if (int(e[0]) in stair_ids) != (int(e[1]) in stair_ids)
        ]
        assert crossing, "stairs must connect to the slab at their landings"

    def test_doors_are_linked_to_walkable_points(self):
        floor = [[x * 0.5, 0.0, 0.0] for x in range(6)]
        door = [[1.25, 0.0, 0.0]]
        pts = np.asarray(floor + door, dtype=np.float32)
        kinds = np.asarray([KIND_FLOOR] * len(floor) + [KIND_DOOR], dtype=np.int8)
        edges = build_adjacency(pts, kinds, max_edge_floor=0.6, max_edge_stair=0.4)
        door_index = len(floor)
        touching = [e for e in edges if door_index in (int(e[0]), int(e[1]))]
        assert touching

    def test_dedupe_drops_self_loops_and_repeats(self):
        raw = np.array([[1, 0], [0, 1], [2, 2], [3, 4]], dtype=np.int32)
        out = dedupe_edges(raw)
        assert out.tolist() == [[0, 1], [3, 4]]

    def test_empty_input(self):
        assert len(build_adjacency(np.zeros((0, 3), np.float32), np.zeros(0, np.int8), 1, 1)) == 0

    def test_snap_to_grid_produces_cardinal_links(self):
        pts = np.array([[0.02, 0.0, 0.0], [1.01, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
        kinds = np.zeros(3, dtype=np.int8)
        out_pts, out_kinds, edges = snap_to_grid(pts, kinds, cell_size=1.0, up_axis="z")
        assert len(out_pts) == 3
        assert len(edges) == 2

    def test_snap_to_grid_bridges_gaps(self):
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=np.float32)
        kinds = np.zeros(2, dtype=np.int8)
        out_pts, _, edges = snap_to_grid(pts, kinds, cell_size=1.0, up_axis="z", max_gap=4)
        assert len(out_pts) == 4  # two originals plus two bridged cells
        assert len(edges) == 3


class TestObstacles:
    def test_segment_intersection(self):
        assert segments_intersect_2d((0, 0), (2, 0), (1, -1), (1, 1))
        assert not segments_intersect_2d((0, 0), (1, 0), (2, -1), (2, 1))
        # Touching at an endpoint is not a crossing.
        assert not segments_intersect_2d((0, 0), (1, 0), (1, 0), (1, 1))

    def test_wall_centreline_follows_longest_span(self):
        geom = {
            "vertices": [
                0, 0, 0, 6, 0, 0, 6, 0.2, 0, 0, 0.2, 0,
                0, 0, 3, 6, 0, 3, 6, 0.2, 3, 0, 0.2, 3,
            ]
        }
        walls = wall_segments_from_geometry([geom], up_axis="z")
        assert len(walls) == 1
        (x0, y0), (x1, y1) = walls[0]["segment"]
        assert x0 == pytest.approx(0.0) and x1 == pytest.approx(6.0)
        assert y0 == pytest.approx(0.1) and y1 == pytest.approx(0.1)
        assert walls[0]["up_min"] == pytest.approx(0.0)
        assert walls[0]["up_max"] == pytest.approx(3.0)

    def test_wall_field_blocks_crossing_edge(self):
        walls = WallField(
            [{"segment": ((0.0, 1.0), (4.0, 1.0)), "thickness": 0.1, "up_min": 0.0, "up_max": 3.0}],
            up_axis="z",
        )
        assert walls.blocks([2.0, 0.0, 0.5], [2.0, 2.0, 0.5])
        assert not walls.blocks([0.5, 0.0, 0.5], [1.5, 0.0, 0.5])

    def test_wall_field_respects_vertical_extent(self):
        walls = WallField(
            [{"segment": ((0.0, 1.0), (4.0, 1.0)), "thickness": 0.1, "up_min": 0.0, "up_max": 1.0}],
            up_axis="z",
        )
        # An edge on the storey above passes over the wall.
        assert not walls.blocks([2.0, 0.0, 5.0], [2.0, 2.0, 5.0])

    def test_door_position_is_at_the_threshold(self):
        geom = {"vertices": [2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 2.1, 2.0, 0.0, 2.1]}
        pos = door_positions_from_geometry([geom], up_axis="z")
        assert len(pos) == 1
        assert pos[0][0] == pytest.approx(2.5)
        assert pos[0][2] == pytest.approx(0.0)

    def test_door_edges_are_exempt_from_blocking(self):
        pts = np.array([[2.0, 0.0, 0.5], [2.0, 2.0, 0.5]], dtype=np.float32)
        edges = np.array([[0, 1]], dtype=np.int32)
        walls = WallField(
            [{"segment": ((0.0, 1.0), (4.0, 1.0)), "thickness": 0.1, "up_min": 0.0, "up_max": 3.0}],
            up_axis="z",
        )
        assert edge_block_mask(pts, edges, walls).tolist() == [True]
        assert edge_block_mask(pts, edges, walls, exempt=np.array([0])).tolist() == [False]


class TestFaceOrientation:
    """A slab's underside and a ceiling are horizontal but not walkable."""

    #: A unit square in the xy plane, wound counter-clockwise so its normal is +z.
    UP_FACING = ([0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0], [0, 1, 2, 0, 2, 3])
    #: The same square wound the other way: normal is -z.
    DOWN_FACING = ([0, 0, 0, 4, 0, 0, 4, 4, 0, 0, 4, 0], [0, 2, 1, 0, 3, 2])

    def test_upward_face_is_walkable(self):
        verts, indices = self.UP_FACING
        assert len(sample_walkable_points(verts, indices, spacing=1.0, up_axis="z")) > 0

    def test_downward_face_is_rejected(self):
        verts, indices = self.DOWN_FACING
        pts = sample_walkable_points(verts, indices, spacing=1.0, up_axis="z")
        assert len(pts) == 0, "a ceiling or slab underside must not be walkable"

    def test_fallback_accepts_either_winding(self):
        verts, indices = self.DOWN_FACING
        pts = sample_walkable_points(
            verts, indices, spacing=1.0, up_axis="z", require_upward=False
        )
        assert len(pts) > 0
