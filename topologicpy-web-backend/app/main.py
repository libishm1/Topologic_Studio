# To start the backend:
# cd C:\Users\sarwj\OneDrive - Cardiff University\Documents\GitHub\TopologicStudio\topologicpy-web-backend
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# To start the front end:
# cd C:\Users\sarwj\OneDrive - Cardiff University\Documents\GitHub\TopologicStudio\topologicpy-web-frontend
# npm run dev

# topologicpy-web-backend/app/main.py
import sys
sys.path.append("C:/Users/sarwj/OneDrive - Cardiff University/Documents/GitHub/topologicpy/src")
from topologicpy.Vertex import Vertex
from topologicpy.Edge import Edge
from topologicpy.Wire import Wire
from topologicpy.Face import Face
from topologicpy.Shell import Shell
from topologicpy.Cluster import Cluster
from topologicpy.Topology import Topology
from topologicpy.Graph import Graph
from topologicpy.Dictionary import Dictionary
from topologicpy.Color import Color

from .schemas import TopologyPayload
from .utils import summarize_topology
from collections import defaultdict


# backend/main.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import math
import json
import time
import random


app = FastAPI(
    title="TopologicPy Web Backend",
    version="0.1.0",
)

cors_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
extra_origins = os.getenv("CORS_ORIGINS", "")
if extra_origins:
    cors_origins.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

LAST_GRAPHS = {"wire": None, "cell": None, "ifc": None}

def _vertex_coord_map(vertices):
    coords = {}
    for v in vertices or []:
        uid = v.get("uid") or v.get("uuid")
        coord = v.get("coordinates") or v.get("Coordinates")
        if uid and coord and len(coord) >= 3:
            coords[uid] = coord[:3]
    return coords

def _build_adjacency(edges, vertex_ids, allowed_kinds=None):
    adj = {uid: set() for uid in vertex_ids}
    for e in edges or []:
        if allowed_kinds is not None:
            kind = (e.get("dictionary") or {}).get("edgeKind")
            if kind not in allowed_kinds:
                continue
        verts = e.get("vertices") or []
        if len(verts) < 2:
            continue
        v0, v1 = verts[0], verts[1]
        if v0 in adj and v1 in adj:
            adj[v0].add(v1)
            adj[v1].add(v0)
    return {k: list(v) for k, v in adj.items()}

def _nearest_node_id(coords_map, point, allowed=None):
    if not coords_map or not point or len(point) < 3:
        return None
    px, py, pz = point[0], point[1], point[2]
    best = None
    best_d = 1e18
    allowed_set = set(allowed) if allowed is not None else None

    # First pass: try to find nodes on the same floor (within 0.2m Y-distance)
    # Y is the vertical axis in this IFC model
    for uid, coord in coords_map.items():
        if allowed_set is not None and uid not in allowed_set:
            continue
        x, y, z = coord
        y_diff = abs(y - py)
        if y_diff <= 0.2:  # Same floor level (tighter tolerance for better accuracy)
            xz_d = (x - px) ** 2 + (z - pz) ** 2
            if xz_d < best_d:
                best = uid
                best_d = xz_d

    # If we found a node on the same floor, return it
    if best is not None:
        return best

    # Second pass: if no nodes on same floor, find nearest in 3D space
    for uid, coord in coords_map.items():
        if allowed_set is not None and uid not in allowed_set:
            continue
        x, y, z = coord
        d = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2
        if d < best_d:
            best = uid
            best_d = d
    return best

def _resolve_start_id(graph, start_id=None, start_point=None):
    adj = graph.get("adjacency") or {}
    if start_id and start_id in adj:
        return start_id
    if start_point:
        coords = graph.get("coords", {})
        return _nearest_node_id(coords, start_point, allowed=adj.keys())
    return None


def _default_start_id(graph):
    adj = graph.get("adjacency") or {}
    if not adj:
        return None
    coords = graph.get("coords") or {}
    if coords:
        xs = [c[0] for c in coords.values()]
        ys = [c[1] for c in coords.values()]
        zs = [c[2] for c in coords.values()]
        center = [
            (min(xs) + max(xs)) / 2,
            (min(ys) + max(ys)) / 2,
            (min(zs) + max(zs)) / 2,
        ]
        return _nearest_node_id(coords, center, allowed=adj.keys())
    return next(iter(adj.keys()), None)

def _estimate_step_size(coords, adjacency=None):
    if coords and adjacency:
        min_dist = None
        for node, nbrs in adjacency.items():
            if node not in coords:
                continue
            x0, y0, z0 = coords[node]
            for nbr in nbrs:
                if nbr not in coords:
                    continue
                x1, y1, z1 = coords[nbr]
                d = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
                if d <= 0:
                    continue
                if min_dist is None or d < min_dist:
                    min_dist = d
        if min_dist:
            return min_dist
    return 1.0


def _axis_index(up_axis: str) -> int:
    if up_axis == "y":
        return 1
    if up_axis == "x":
        return 0
    return 2


def _triangle_normal(v1, v2, v3):
    ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
    bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
    nx = ay * bz - az * by
    ny = az * bx - ax * bz
    nz = ax * by - ay * bx
    mag = (nx * nx + ny * ny + nz * nz) ** 0.5
    if mag == 0:
        return 0.0, 0.0, 0.0
    return nx / mag, ny / mag, nz / mag


def _sample_walkable_points(vertices, indices, normals, spacing, max_slope_deg, up_axis="z", max_points=20000):
    if not vertices or not indices:
        return []
    up_idx = _axis_index(up_axis)
    min_up = math.cos(math.radians(90 - max_slope_deg))
    points = []
    for i in range(0, len(indices), 3):
        i1 = indices[i] * 3
        i2 = indices[i + 1] * 3
        i3 = indices[i + 2] * 3
        if i3 + 2 >= len(vertices):
            continue
        v1 = [vertices[i1], vertices[i1 + 1], vertices[i1 + 2]]
        v2 = [vertices[i2], vertices[i2 + 1], vertices[i2 + 2]]
        v3 = [vertices[i3], vertices[i3 + 1], vertices[i3 + 2]]
        if normals and i1 + 2 < len(normals):
            n = [normals[i1], normals[i1 + 1], normals[i1 + 2]]
        else:
            n = _triangle_normal(v1, v2, v3)
        if n[up_idx] < min_up:
            continue
        ax, ay, az = v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]
        bx, by, bz = v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]
        cx = ay * bz - az * by
        cy = az * bx - ax * bz
        cz = ax * by - ay * bx
        area = 0.5 * (cx * cx + cy * cy + cz * cz) ** 0.5
        if area <= 1e-6:
            continue
        samples = max(1, int(math.ceil((area ** 0.5) / max(spacing, 1e-3))))
        for u in range(samples + 1):
            for v in range(samples + 1 - u):
                alpha = u / samples
                beta = v / samples
                gamma = 1 - alpha - beta
                if gamma < 0:
                    continue
                px = alpha * v1[0] + beta * v2[0] + gamma * v3[0]
                py = alpha * v1[1] + beta * v2[1] + gamma * v3[1]
                pz = alpha * v1[2] + beta * v2[2] + gamma * v3[2]
                points.append([px, py, pz])
                if len(points) >= max_points:
                    return points
    return points


def _build_point_adjacency(points, max_dist):
    if not points:
        return {}, {}
    inv = 1.0 / max_dist if max_dist > 0 else 1.0
    buckets = defaultdict(list)
    for idx, p in enumerate(points):
        key = (int(p[0] * inv), int(p[1] * inv), int(p[2] * inv))
        buckets[key].append(idx)
    adjacency = {idx: [] for idx in range(len(points))}
    for idx, p in enumerate(points):
        bx, by, bz = int(p[0] * inv), int(p[1] * inv), int(p[2] * inv)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((bx + dx, by + dy, bz + dz), []):
                        if j <= idx:
                            continue
                        q = points[j]
                        d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5
                        if d <= max_dist:
                            adjacency[idx].append(j)
                            adjacency[j].append(idx)
    coords = {f"ifc_{i}": points[i] for i in range(len(points))}
    adj_named = {f"ifc_{i}": [f"ifc_{j}" for j in nbrs] for i, nbrs in adjacency.items()}
    return coords, adj_named


def _build_point_adjacency_hybrid(points, num_stair_points, max_dist_stair, max_dist_floor):
    """
    Build adjacency with different max distances for stair points vs floor points.
    First num_stair_points are stair points, rest are floor points.
    Stair-to-stair connections use max_dist_stair (short for tread-to-tread).
    Floor-to-floor connections use max_dist_floor (longer for open spaces).
    Stair-to-floor connections use max(max_dist_stair, max_dist_floor) for transitions.
    """
    if not points:
        return {}, {}

    # Use the larger distance for spatial hashing to capture all potential connections
    inv = 1.0 / max(max_dist_stair, max_dist_floor) if max(max_dist_stair, max_dist_floor) > 0 else 1.0
    buckets = defaultdict(list)
    for idx, p in enumerate(points):
        key = (int(p[0] * inv), int(p[1] * inv), int(p[2] * inv))
        buckets[key].append(idx)

    adjacency = {idx: [] for idx in range(len(points))}
    for idx, p in enumerate(points):
        is_stair_i = idx < num_stair_points
        bx, by, bz = int(p[0] * inv), int(p[1] * inv), int(p[2] * inv)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in buckets.get((bx + dx, by + dy, bz + dz), []):
                        if j <= idx:
                            continue

                        is_stair_j = j < num_stair_points
                        q = points[j]
                        d = ((q[0] - p[0]) ** 2 + (q[1] - p[1]) ** 2 + (q[2] - p[2]) ** 2) ** 0.5

                        # Determine max distance based on point types
                        if is_stair_i and is_stair_j:
                            # Stair-to-stair: use short distance for tread-to-tread
                            max_d = max_dist_stair
                        elif not is_stair_i and not is_stair_j:
                            # Floor-to-floor: use longer distance
                            max_d = max_dist_floor
                        else:
                            # Stair-to-floor transition: use longer distance to allow connection
                            max_d = max(max_dist_stair, max_dist_floor)

                        if d <= max_d:
                            adjacency[idx].append(j)
                            adjacency[j].append(idx)

    coords = {f"ifc_{i}": points[i] for i in range(len(points))}
    adj_named = {f"ifc_{i}": [f"ifc_{j}" for j in nbrs] for i, nbrs in adjacency.items()}

    # Force-connect stairs to floors: find stair endpoints and connect to nearest floor points
    # This ensures floors and stairs form a connected graph even if spatial hashing misses connections
    forced_connections = 0
    if num_stair_points > 0 and num_stair_points < len(points):
        stair_indices = list(range(num_stair_points))
        floor_indices = list(range(num_stair_points, len(points)))

        # Find stair points at top and bottom (extreme Y values - vertical axis)
        stair_y_vals = [(i, points[i][1]) for i in stair_indices]
        stair_y_vals.sort(key=lambda x: x[1])

        # Bottom 10% and top 10% of stair points are likely endpoints
        num_endpoints = max(2, len(stair_indices) // 10)
        bottom_stairs = [idx for idx, _ in stair_y_vals[:num_endpoints]]
        top_stairs = [idx for idx, _ in stair_y_vals[-num_endpoints:]]

        # Connect each stair endpoint to nearest 3 floor points within 2.0 meters
        for stair_idx in bottom_stairs + top_stairs:
            sx, sy, sz = points[stair_idx]
            distances = []
            for floor_idx in floor_indices:
                fx, fy, fz = points[floor_idx]
                # Only consider floor points within reasonable Y distance (same level)
                y_diff = abs(fy - sy)
                if y_diff > 1.0:  # Different floor level, skip
                    continue
                d = ((fx - sx) ** 2 + (fy - sy) ** 2 + (fz - sz) ** 2) ** 0.5
                if d <= 2.0:  # Within 2.0 meters
                    distances.append((d, floor_idx))

            # Connect to 3 nearest floor points
            distances.sort()
            for _, floor_idx in distances[:3]:
                stair_uid = f"ifc_{stair_idx}"
                floor_uid = f"ifc_{floor_idx}"
                if floor_uid not in adj_named.get(stair_uid, []):
                    adj_named.setdefault(stair_uid, []).append(floor_uid)
                    adj_named.setdefault(floor_uid, []).append(stair_uid)
                    forced_connections += 1

    return coords, adj_named


def _shortest_path_ids(adjacency, coords, start_id, end_id):
    if start_id not in adjacency or end_id not in adjacency:
        return []
    import heapq
    dist = {start_id: 0.0}
    prev = {}
    heap = [(0.0, start_id)]
    while heap:
        d, node = heapq.heappop(heap)
        if node == end_id:
            break
        if d != dist.get(node, 0.0):
            continue
        for nbr in adjacency.get(node, []):
            if nbr not in coords:
                continue
            x0, y0, z0 = coords[node]
            x1, y1, z1 = coords[nbr]
            w = ((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2) ** 0.5
            nd = d + w
            if nd < dist.get(nbr, 1e18):
                dist[nbr] = nd
                prev[nbr] = node
                heapq.heappush(heap, (nd, nbr))
    if end_id not in dist:
        return []
    path = [end_id]
    cur = end_id
    while cur != start_id:
        cur = prev.get(cur)
        if cur is None:
            break
        path.append(cur)
    path.reverse()
    return path


def _compute_radial_timeline(coords, start_id, end_id=None, max_steps=60, step_size=None, allowed=None):
    if not coords or start_id not in coords:
        return [], {}
    if not step_size or step_size <= 0:
        step_size = 1.0
    allowed_set = set(allowed) if allowed is not None else None
    sx, sy, sz = coords[start_id]
    dist_map = {}
    for uid, coord in coords.items():
        if allowed_set is not None and uid not in allowed_set:
            continue
        x, y, z = coord
        dist_map[uid] = ((x - sx) ** 2 + (y - sy) ** 2 + (z - sz) ** 2) ** 0.5
    if not dist_map:
        return [], {}
    max_dist = max(dist_map.values())
    if end_id and end_id in coords:
        ex, ey, ez = coords[end_id]
        end_dist = ((ex - sx) ** 2 + (ey - sy) ** 2 + (ez - sz) ** 2) ** 0.5
        max_dist = min(max_dist, end_dist)
    max_bucket = max(0, int(math.ceil(max_dist / step_size)))
    max_bucket = min(max_bucket, max_steps - 1)
    buckets = {}
    for uid, dist in dist_map.items():
        bucket = int(dist / step_size)
        if bucket > max_bucket:
            continue
        buckets.setdefault(bucket, []).append(uid)
    timeline = []
    ignite_time = {}
    for bucket in range(max_bucket + 1):
        nodes = buckets.get(bucket, [])
        timeline.append(nodes)
        for n in nodes:
            ignite_time[n] = bucket
    return timeline, ignite_time


def _compute_fire_timeline(adjacency, coords, start_id, max_steps=60, radial=False, end_id=None, step_size=None):
    if radial:
        allowed = adjacency.keys() if adjacency else None
        return _compute_radial_timeline(coords, start_id, end_id, max_steps, step_size, allowed=allowed)
    if not adjacency or start_id not in adjacency:
        return [], {}
    visited = {start_id}
    current = [start_id]
    timeline = [current]
    steps = 0
    while current and steps < max_steps:
        nxt = []
        for node in current:
            for nbr in adjacency.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    nxt.append(nbr)
        if not nxt:
            break
        timeline.append(nxt)
        current = nxt
        steps += 1
    ignite_time = {}
    for step, nodes in enumerate(timeline):
        for n in nodes:
            ignite_time[n] = step
    return timeline, ignite_time


def _compute_temperature_fire_spread(adjacency, coords, start_id, max_steps=60,
                                     ambient_temp=20.0, fire_temp=120.0, heat_transfer_rate=1.20):
    """
    Compute fire spread using temperature-based model (based on eCAADe 2019 paper).
    Each cell has a temperature that increases based on adjacent cells' temperatures.

    Args:
        adjacency: Graph adjacency dict
        coords: Node coordinates dict
        start_id: Starting fire node
        max_steps: Maximum simulation steps
        ambient_temp: Initial ambient temperature (°C)
        fire_temp: Initial fire source temperature (°C)
        heat_transfer_rate: Heat transfer coefficient

    Returns:
        List of temperature dictionaries for each timestep
    """
    if not adjacency or start_id not in adjacency:
        return []

    # Initialize all cells with ambient temperature
    all_nodes = list(adjacency.keys())
    temperatures = {node: ambient_temp for node in all_nodes}
    temperatures[start_id] = fire_temp  # Set fire source

    # Store temperature history for visualization
    temp_timeline = []

    for step in range(max_steps):
        # Record current temperatures
        temp_timeline.append(temperatures.copy())

        # Calculate new temperatures for next step
        new_temperatures = temperatures.copy()

        for node in all_nodes:
            if node == start_id:
                # Fire source maintains high temperature
                new_temperatures[node] = fire_temp
                continue

            # Get adjacent cells
            neighbors = adjacency.get(node, [])
            if not neighbors:
                continue

            # Calculate heat transfer from neighbors
            neighbor_temps = [temperatures[nbr] for nbr in neighbors]
            avg_neighbor_temp = sum(neighbor_temps) / len(neighbor_temps)

            # Heat transfer formula: increase temperature based on neighbor average
            temp_diff = avg_neighbor_temp - temperatures[node]
            if temp_diff > 0:
                # Only increase temperature, not decrease (simplified model)
                new_temperatures[node] += temp_diff * heat_transfer_rate / len(neighbors)

        temperatures = new_temperatures

    return temp_timeline


def _build_cell_grid(bounds, cell_size=1.0, max_cells=2000):
    minx = bounds.get("minx")
    maxx = bounds.get("maxx")
    miny = bounds.get("miny")
    maxy = bounds.get("maxy")
    minz = bounds.get("minz")
    maxz = bounds.get("maxz")
    if minx is None or maxx is None or not math.isfinite(minx) or not math.isfinite(maxx):
        return [], {}, None
    span_x = maxx - minx
    span_y = maxy - miny
    span_z = maxz - minz
    if span_x <= 0 or span_y <= 0 or span_z <= 0:
        return [], {}, None
    nx = max(1, int(span_x / cell_size))
    ny = max(1, int(span_y / cell_size))
    nz = max(1, int(span_z / cell_size))
    total = nx * ny * nz
    if total > max_cells:
        scale = (max_cells / total) ** (1.0 / 3.0)
        nx = max(1, int(nx * scale))
        ny = max(1, int(ny * scale))
        nz = max(1, int(nz * scale))
    dx = span_x / nx
    dy = span_y / ny
    dz = span_z / nz
    step = min(dx, dy, dz)

    nodes = []
    node_index = {}
    for k in range(nz):
        z = minz + (k + 0.5) * dz
        for j in range(ny):
            y = miny + (j + 0.5) * dy
            for i in range(nx):
                x = minx + (i + 0.5) * dx
                uid = f"cell_{i}_{j}_{k}"
                nodes.append({
                    "id": uid,
                    "center": [x, y, z],
                    "minx": x - dx * 0.5,
                    "maxx": x + dx * 0.5,
                    "miny": y - dy * 0.5,
                    "maxy": y + dy * 0.5,
                    "z_min": z - dz * 0.5,
                    "z_max": z + dz * 0.5,
                })
                node_index[(i, j, k)] = uid

    adjacency = {n["id"]: [] for n in nodes}
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                uid = node_index[(i, j, k)]
                for di, dj, dk in [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]:
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz:
                        adjacency[uid].append(node_index[(ni, nj, nk)])

    return nodes, adjacency, step

def _q_learning_path(adjacency, start_id, exit_id, ignite_time=None, episodes=200, max_steps=200):
    if not adjacency or start_id not in adjacency or exit_id not in adjacency:
        return []
    q = defaultdict(lambda: defaultdict(float))
    epsilon = 0.2
    alpha = 0.5
    gamma = 0.9
    for _ in range(max(1, episodes)):
        state = start_id
        t = 0
        for _ in range(max_steps):
            neighbors = adjacency.get(state, [])
            if not neighbors:
                break
            if random.random() < epsilon:
                nxt = random.choice(neighbors)
            else:
                nxt = max(neighbors, key=lambda n: q[state][n])
            reward = -0.1
            done = False
            if ignite_time and t >= ignite_time.get(nxt, 1e9):
                reward = -10.0
                done = True
            if nxt == exit_id:
                reward = 10.0
                done = True
            max_next = max(q[nxt].values()) if q[nxt] else 0.0
            q[state][nxt] = (1 - alpha) * q[state][nxt] + alpha * (reward + gamma * max_next)
            state = nxt
            t += 1
            if done:
                break
    # greedy rollout
    path = [start_id]
    state = start_id
    visited = {start_id}
    for _ in range(max_steps):
        neighbors = adjacency.get(state, [])
        if not neighbors:
            break
        nxt = max(neighbors, key=lambda n: q[state][n])
        path.append(nxt)
        if nxt == exit_id or nxt in visited:
            break
        visited.add(nxt)
        state = nxt
    return path

def _sse_event(payload):
    return f"data: {json.dumps(payload)}\n\n"


class FireSimRequest(BaseModel):
    mode: str = "wire"
    start_id: Optional[str] = None
    end_id: Optional[str] = None
    start_point: Optional[List[float]] = None
    end_point: Optional[List[float]] = None
    max_steps: int = 60
    precompute: bool = True
    radial: bool = True
    delay_ms: int = 200

class RLRequest(BaseModel):
    mode: str = "wire"
    start_id: Optional[str] = None
    exit_id: Optional[str] = None
    start_point: Optional[List[float]] = None
    exit_point: Optional[List[float]] = None
    episodes: int = 200
    max_steps: int = 200
    use_fire: bool = True


class IfcGeometry(BaseModel):
    expressID: int
    vertices: List[float]
    indices: List[int]
    normals: Optional[List[float]] = None


class IfcEgressRequest(BaseModel):
    floors: List[IfcGeometry] = []
    stairs: List[IfcGeometry] = []
    agent_height: float = 0.75
    base_spacing: float = 0.5
    stair_multiplier: float = 0.5
    max_edge_length: float = 1.5
    max_edge_floor: Optional[float] = None
    max_edge_stair: Optional[float] = None
    up_axis: str = "z"
    max_points: int = 20000


class IfcEgressPathRequest(BaseModel):
    start_point: Optional[List[float]] = None
    end_point: Optional[List[float]] = None


@app.post("/ifc-egress-graph")
def ifc_egress_graph(req: IfcEgressRequest):
    """Build navigation graph using simplified point sampling with strict limits"""
    floors = req.floors or []
    stairs = req.stairs or []
    if not floors and not stairs:
        raise HTTPException(status_code=400, detail="No IFC floor/stair geometry provided.")

    # Use simple point sampling with STRICT limits
    base_spacing = max(req.base_spacing, 1.5)  # Minimum 1.5m spacing
    stair_spacing = max(base_spacing * 0.3, 0.3)  # VERY dense spacing for stairs (0.3m)

    # Use slider values if provided, otherwise use calculated defaults
    max_edge_floor = (
        req.max_edge_floor if req.max_edge_floor is not None else base_spacing * 1.5
    )  # Default 2.25m or slider value
    max_edge_stair = (
        req.max_edge_stair if req.max_edge_stair is not None else 0.4
    )  # Default 0.4m or slider value
    max_total_points = 3000  # Increased limit to accommodate both dense stair sampling AND floor grid

    stair_points = []
    floor_points = []

    # Sample stairs FIRST (prioritize connectivity between floors)
    for geom in stairs:
        if len(stair_points) >= max_total_points:
            break
        pts = _sample_walkable_points(
            geom.vertices,
            geom.indices,
            geom.normals,
            stair_spacing,
            45,  # Higher angle for stairs
            up_axis=req.up_axis,
            max_points=300,  # MUCH higher limit per stair (300 points) for multi-level connectivity
        )
        stair_points.extend(pts[:300])

    # Sample floors with remaining budget
    remaining_budget = max_total_points - len(stair_points)
    points_per_floor = max(100, remaining_budget // max(1, len(floors)))  # Minimum 100 points per floor for visibility

    for geom in floors:
        if len(floor_points) >= remaining_budget:
            break
        pts = _sample_walkable_points(
            geom.vertices,
            geom.indices,
            geom.normals,
            base_spacing,
            10,  # Low angle tolerance
            up_axis=req.up_axis,
            max_points=points_per_floor,
        )
        floor_points.extend(pts[:points_per_floor])

    # Combine all points
    all_points = stair_points + floor_points
    all_points = all_points[:max_total_points]

    if not all_points:
        raise HTTPException(status_code=400, detail="No walkable points extracted.")

    num_stair_points = min(len(stair_points), len(all_points))

    # Add agent height
    all_points = [[p[0], p[1], p[2] + req.agent_height] for p in all_points]

    # Build adjacency with SEPARATE edge limits for stairs vs floors
    coords, adjacency = _build_point_adjacency_hybrid(all_points, num_stair_points, max_edge_stair, max_edge_floor)

    LAST_GRAPHS["ifc"] = {
        "coords": coords,
        "adjacency": adjacency,
        "step": _estimate_step_size(coords, adjacency),
    }

    # Build edge list for visualization
    edge_list = []
    for node_id, neighbors in adjacency.items():
        for neighbor_id in neighbors:
            if node_id < neighbor_id:
                edge_list.append([coords[node_id], coords[neighbor_id]])

    return {
        "mode": "ifc",
        "stats": {
            "nodes": len(coords),
            "edges": len(edge_list),
        },
        "edges": edge_list,
    }


@app.post("/ifc-egress-path")
def ifc_egress_path(req: IfcEgressPathRequest):
    graph = LAST_GRAPHS.get("ifc") if LAST_GRAPHS else None
    if not graph or not graph.get("adjacency"):
        raise HTTPException(status_code=400, detail="No IFC egress graph available. Build it first.")
    start_id = _resolve_start_id(graph, None, req.start_point)
    end_id = _resolve_start_id(graph, None, req.end_point)
    if not start_id or not end_id:
        raise HTTPException(status_code=400, detail="Invalid start or end point.")

    path_ids = _shortest_path_ids(graph["adjacency"], graph["coords"], start_id, end_id)
    if not path_ids:
        raise HTTPException(status_code=404, detail="No path found between start and end points.")
    points = [graph["coords"][pid] for pid in path_ids if pid in graph["coords"]]
    return {
        "mode": "ifc",
        "points": points,
    }

@app.post("/fire-sim")
def fire_sim(req: FireSimRequest):
    graph = LAST_GRAPHS.get(req.mode) if LAST_GRAPHS else None
    if not graph or not graph.get("adjacency"):
        raise HTTPException(status_code=400, detail="No graph available. Load an IFC file first.")
    start_id = _resolve_start_id(graph, req.start_id, req.start_point)
    end_id = _resolve_start_id(graph, req.end_id, req.end_point)
    if not start_id:
        start_id = _default_start_id(graph)
    if not start_id:
        raise HTTPException(status_code=400, detail="Invalid start point or start id.")
    step_size = graph.get("step") or _estimate_step_size(graph.get("coords"), graph.get("adjacency"))
    timeline, _ = _compute_fire_timeline(graph["adjacency"], graph.get("coords"), start_id, req.max_steps, req.radial, end_id, step_size)
    return {
        "mode": req.mode,
        "start_id": start_id,
        "timeline": timeline,
        "cell_bboxes": graph.get("bboxes", []),
    }

@app.get("/fire-sim/stream")
def fire_sim_stream(
    mode: str = "wire",
    start_id: Optional[str] = None,
    end_id: Optional[str] = None,
    start_x: Optional[float] = None,
    start_y: Optional[float] = None,
    start_z: Optional[float] = None,
    end_x: Optional[float] = None,
    end_y: Optional[float] = None,
    end_z: Optional[float] = None,
    max_steps: int = 60,
    precompute: bool = True,
    radial: bool = True,
    delay_ms: int = 200,
):
    graph = LAST_GRAPHS.get(mode) if LAST_GRAPHS else None
    if not graph or not graph.get("adjacency"):
        raise HTTPException(status_code=400, detail="No graph available. Load an IFC file first.")
    start_point = None
    end_point = None
    if start_x is not None and start_y is not None and start_z is not None:
        start_point = [start_x, start_y, start_z]
    if end_x is not None and end_y is not None and end_z is not None:
        end_point = [end_x, end_y, end_z]
    start_id = _resolve_start_id(graph, start_id, start_point)
    end_id = _resolve_start_id(graph, end_id, end_point)
    if not start_id:
        start_id = _default_start_id(graph)
    if not start_id:
        raise HTTPException(status_code=400, detail="Invalid start point or start id.")

    def gen():
        if mode == "cell" and graph.get("bboxes"):
            yield _sse_event({"type": "meta", "cell_bboxes": graph.get("bboxes", [])})
        if precompute:
            step_size = graph.get("step") or _estimate_step_size(graph.get("coords"), graph.get("adjacency"))
            timeline, _ = _compute_fire_timeline(graph["adjacency"], graph.get("coords"), start_id, max_steps, radial, end_id, step_size)
            for step, nodes in enumerate(timeline):
                yield _sse_event({"type": "step", "step": step, "nodes": nodes})
                time.sleep(max(delay_ms, 0) / 1000.0)
        else:
            visited = {start_id}
            current = [start_id]
            step = 0
            yield _sse_event({"type": "step", "step": step, "nodes": current})
            while current and step < max_steps:
                nxt = []
                for node in current:
                    for nbr in graph["adjacency"].get(node, []):
                        if nbr not in visited:
                            visited.add(nbr)
                            nxt.append(nbr)
                if not nxt:
                    break
                step += 1
                yield _sse_event({"type": "step", "step": step, "nodes": nxt})
                time.sleep(max(delay_ms, 0) / 1000.0)
                current = nxt
        yield _sse_event({"type": "done"})

    return StreamingResponse(gen(), media_type="text/event-stream")

@app.post("/rl/train")
def rl_train(req: RLRequest):
    graph = LAST_GRAPHS.get(req.mode) if LAST_GRAPHS else None
    if not graph or not graph.get("adjacency"):
        raise HTTPException(status_code=400, detail="No graph available. Load an IFC file first.")
    start_id = _resolve_start_id(graph, req.start_id, req.start_point)
    exit_id = _resolve_start_id(graph, req.exit_id, req.exit_point)
    if not start_id or not exit_id:
        raise HTTPException(status_code=400, detail="Invalid start or exit.")
    ignite_time = None
    if req.use_fire:
        _, ignite_time = _compute_fire_timeline(graph["adjacency"], graph.get("coords"), start_id, req.max_steps, False, None, None)
    path = _q_learning_path(graph["adjacency"], start_id, exit_id, ignite_time, req.episodes, req.max_steps)
    return {
        "mode": req.mode,
        "start_id": start_id,
        "exit_id": exit_id,
        "path": path,
        "cell_bboxes": graph.get("bboxes", []),
    }


@app.get("/graph-meta")
def graph_meta(mode: str = "wire"):
    graph = LAST_GRAPHS.get(mode) if LAST_GRAPHS else None
    if not graph:
        raise HTTPException(status_code=400, detail="No graph available. Load an IFC file first.")
    return {
        "mode": mode,
        "cell_bboxes": graph.get("bboxes", []),
    }


class RawTopologyItem(BaseModel):
    type: str
    uid: str
    dictionary: Dict[str, Any] = {}
    apertures: List[Any] = []
    coordinates: Optional[List[float]] = None  # Vertex
    vertices: Optional[List[str]] = None       # Edge
    edges: Optional[List[str]] = None          # Wire
    wires: Optional[List[str]] = None          # Face
    faces: Optional[List[str]] = None          # Shell
    shells: Optional[List[str]] = None         # Cell
    cells: Optional[List[str]] = None          # CellComplex

class ProcessedVertex(BaseModel):
    uid: str
    x: float
    y: float
    z: float
    dictionary: Dict[str, Any]

class ProcessedFace(BaseModel):
    uid: str
    triangles: List[List[List[float]]]  # [[[x,y,z], ...], ...]
    dictionary: Dict[str, Any]

class ParentsMap(BaseModel):
    # parents[uuid] = {"Cluster": ..., "CellComplex": ..., ...}
    parents: Dict[str, Dict[str, Optional[str]]]

class UploadResponse(BaseModel):
    vertices: List[ProcessedVertex]
    faces: List[ProcessedFace]
    parents: ParentsMap
    raw: List[Dict[str, Any]]  # pass-through for sidebar details

LEVELS = [
    "Cluster",
    "CellComplex",
    "Cell",
    "Shell",
    "Face",
    "Wire",
    "Edge",
    "Vertex",
]

def json_by_cluster(cluster):
    import json
    j_vertices = []
    j_edges = []
    j_wires = []
    j_faces = []
    j_shells = []
    j_cells = []
    j_cellComplexes = []
    vertices = Topology.Vertices(cluster)
    edges = Topology.Edges(cluster)
    wires = Topology.Wires(cluster)
    faces = Topology.Faces(cluster)
    shells = Topology.Shells(cluster)
    cells = Topology.Cells(cluster)
    cellComplexes = Topology.CellComplexes(cluster)

    for i, t in enumerate(vertices):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "v"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(edges):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "e"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(wires):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "w"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(faces):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "f"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(shells):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "s"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(cells):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "c"+str(i))
        t = Topology.SetDictionary(t, t_d)
    for i, t in enumerate(cellComplexes):
        t_d = Topology.Dictionary(t)
        t_d = Dictionary.SetValueAtKey(t_d, "uid", "x"+str(i))
        t = Topology.SetDictionary(t, t_d)


    for i, v in enumerate(vertices):
        uid = "v"+str(i)
        coords = Vertex.Coordinates(v, mantissa=4)
        t_d = Topology.Dictionary(v)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid"], ["Vertex", uid])
        v = Topology.SetDictionary(v, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="edge")
        p_edges = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="wire")
        p_wires = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="face")
        p_faces = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="shell")
        p_shells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="cell")
        p_cells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(v, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Vertex",
            "uid": uid,
            "coordinates": coords,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": p_edges,
                "wires": p_wires,
                "faces": p_faces,
                "shells": p_shells,
                "cells": p_cells,
                "cellComplexes": p_cellComplexes
            }
        }
        j_vertices.append(final_dic)
    
    for i, e in enumerate(edges):
        uid = "e"+str(i)
        v_uids = [Dictionary.ValueAtKey(Topology.Dictionary(v), "uid") for v in Topology.Vertices(e)]
        t_d = Topology.Dictionary(e)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "vertices"], ["Edge", uid, v_uids])
        e = Topology.SetDictionary(e, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(e, hostTopology=cluster, topologyType="wire")
        p_wires = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(e, hostTopology=cluster, topologyType="face")
        p_faces = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(e, hostTopology=cluster, topologyType="shell")
        p_shells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(e, hostTopology=cluster, topologyType="cell")
        p_cells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(e, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Edge",
            "uid": uid,
            "vertices": v_uids,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": p_wires,
                "faces": p_faces,
                "shells": p_shells,
                "cells": p_cells,
                "cellComplexes": p_cellComplexes
            }
        }
        j_edges.append(final_dic)

    for i, w in enumerate(wires):
        uid = "w"+str(i)
        e_uids = [Dictionary.ValueAtKey(Topology.Dictionary(e), "uid") for e in Topology.Edges(w)]
        t_d = Topology.Dictionary(w)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "edges"], ["Wire", uid, e_uids])
        w = Topology.SetDictionary(w, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(w, hostTopology=cluster, topologyType="face")
        p_faces = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(w, hostTopology=cluster, topologyType="shell")
        p_shells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(w, hostTopology=cluster, topologyType="cell")
        p_cells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(w, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Wire",
            "uid": uid,
            "edges": e_uids,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": [],
                "faces": p_faces,
                "shells": p_shells,
                "cells": p_cells,
                "cellComplexes": p_cellComplexes
            }
        }
        j_wires.append(final_dic)
    for i, f in enumerate(faces):
        triangles = Face.Triangulate(f)
        tri_coords = []
        for triangle in triangles:
            verts = [Vertex.Coordinates(v) for v in Topology.Vertices(triangle)]
            tri_coords.append(verts)
        uid = "f"+str(i)
        t_d = Topology.Dictionary(f)
        w_uids = [Dictionary.ValueAtKey(Topology.Dictionary(w), "uid") for w in Topology.Wires(f)]
        opacity = Dictionary.ValueAtKey(t_d, "opacity", 0.4)
        color = Dictionary.ValueAtKey(t_d, "color", "white")
        color = Color.AnyToHex(color)
        color = color.lower()
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "color", "wires"], ["Face", uid, color, w_uids])
        f = Topology.SetDictionary(f, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(f, hostTopology=cluster, topologyType="shell")
        p_shells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(f, hostTopology=cluster, topologyType="cell")
        p_cells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(f, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Face",
            "uuid": uid,
            "triangles": tri_coords,
            "wires": w_uids,
            "color": color,
            "opacity": opacity,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": [],
                "faces": [],
                "shells": p_shells,
                "cells": p_cells,
                "cellComplexes": p_cellComplexes
            }
        }
        j_faces.append(final_dic)
    
    for i, s in enumerate(shells):
        uid = "s"+str(i)
        f_uids = [Dictionary.ValueAtKey(Topology.Dictionary(f), "uid") for f in Topology.Faces(s)]
        t_d = Topology.Dictionary(s)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "faces"], ["Shell", uid, f_uids])
        s = Topology.SetDictionary(s, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(s, hostTopology=cluster, topologyType="cell")
        p_cells = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        sup_tops = Topology.SuperTopologies(s, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Shell",
            "uid": uid,
            "faces": f_uids,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": [],
                "faces": [],
                "shells": [],
                "cells": p_cells,
                "cellComplexes": p_cellComplexes
            }
        }
        j_shells.append(final_dic)

    for i, c in enumerate(cells):
        uid = "c"+str(i)
        s_uids = [Dictionary.ValueAtKey(Topology.Dictionary(s), "uid") for s in Topology.Shells(c)]
        t_d = Topology.Dictionary(c)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "shells"], ["Cell", uid, s_uids])
        c = Topology.SetDictionary(c, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        sup_tops = Topology.SuperTopologies(c, hostTopology=cluster, topologyType="cellComplex")
        p_cellComplexes = [Dictionary.ValueAtKey(Topology.Dictionary(t), "uid") for t in sup_tops]
        final_dic = {
            "type": "Cell",
            "uid": uid,
            "shells": s_uids,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": [],
                "faces": [],
                "shells": [],
                "cells": [],
                "cellComplexes": p_cellComplexes
            }
        }
        j_cells.append(final_dic)
    
    for i, x in enumerate(cellComplexes):
        uid = "x"+str(i)
        c_uids = [Dictionary.ValueAtKey(Topology.Dictionary(c), "uid") for c in Topology.Cells(x)]
        t_d = Topology.Dictionary(x)
        t_d = Dictionary.SetValuesAtKeys(t_d, ["type", "uid", "cells"], ["CellComplex", uid, c_uids])
        x = Topology.SetDictionary(x, t_d)
        py_d = Dictionary.PythonDictionary(t_d)
        final_dic = {
            "type": "CellComplex",
            "uid": uid,
            "cells": c_uids,
            "dictionary": py_d,
            "parents": {
                "vertices": [],
                "edges": [],
                "wires": [],
                "faces": [],
                "shells": [],
                "cells": [],
                "cellComplexes": []
            }
        }
        j_cellComplexes.append(final_dic)
    raw = j_vertices+j_edges+j_wires+j_faces+j_shells+j_cells+j_cellComplexes
    return_dic = {
        "vertices": j_vertices,
        "edges": j_edges,
        "faces": j_faces,
        "raw": raw
        }
    return return_dic

from typing import Any, Union, List
from fastapi import Body

@app.post("/upload-topology")
async def upload_topology(payload: Union[List[Any], dict] = Body(...)):
    """
    Accept either:
    - a pre-converted viewer contract dict (has vertices/faces keys)
    - the original TopologicPy JSON export (list of dicts)
    Anything else is rejected with a helpful error instead of 500s.
    """
    # Already in viewer contract form
    if isinstance(payload, dict) and "vertices" in payload and "faces" in payload:
        return payload

    # Clearly not a TopologicPy export (e.g., Blender/Sverchok graph JSON)
    if isinstance(payload, dict) and "export_version" in payload and "main_tree" in payload:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported JSON format: looks like a Blender/Sverchok graph. "
                "Please upload a TopologicPy topology export or a pre-converted viewer contract."
            ),
        )

    if not isinstance(payload, list):
        raise HTTPException(
            status_code=400,
            detail="Unsupported JSON format: expected a list of TopologicPy topology dictionaries.",
        )

    try:
        topologies = Topology.ByJSONDictionary(
            jsonDictionary=payload,
            tolerance=0.0001,
            silent=False,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Failed to parse TopologicPy JSON: {exc}",
        )

    cluster = Cluster.ByTopologies(topologies)
    contract = json_by_cluster(cluster)
    return contract

# @app.post("/upload-topology")
# async def upload_topology(payload: dict):
#     import json
#     # payload is your original TopologicPy JSON
#     topologies = Topology.ByJSONDictionary(jsonDictionary = payload, tolerance = 0.0001, silent = False)
#     cluster = Cluster.ByTopologies(topologies)
#     result = json_by_cluster(cluster)
#     print(result[:100])
#     return result

# @app.post("/upload-topology", response_model=UploadResponse)
# def upload_topology(items: List[RawTopologyItem]):
#     by_uuid: Dict[str, RawTopologyItem] = {item.uuid: item for item in items}

#     # --- Coordinates for vertices ---
#     vertex_coords: Dict[str, List[float]] = {}
#     processed_vertices: List[ProcessedVertex] = []
#     for item in items:
#         if item.type == "Vertex" and item.coordinates:
#             x, y, z = item.coordinates
#             vertex_coords[item.uuid] = [x, y, z]
#             processed_vertices.append(
#                 ProcessedVertex(
#                     uuid=item.uuid,
#                     x=x,
#                     y=y,
#                     z=z,
#                     dictionary=item.dictionary or {},
#                 )
#             )

#     # --- Build basic parent maps (one level at a time) ---
#     parent_face_to_shell: Dict[str, str] = {}
#     parent_shell_to_cell: Dict[str, str] = {}
#     parent_cell_to_cc: Dict[str, str] = {}
#     parent_wire_to_face: Dict[str, str] = {}
#     parent_edge_to_wire: Dict[str, str] = {}
#     parent_vertex_to_edge: Dict[str, str] = {}
#     cluster_of: Dict[str, str] = {}

#     # cluster from dictionary["__cluster__"]
#     for item in items:
#         cl_id = item.dictionary.get("__cluster__")
#         if cl_id:
#             cluster_of[item.uuid] = cl_id

#     # Shell -> faces
#     for item in items:
#         if item.type == "Shell" and item.faces:
#             for f_uuid in item.faces:
#                 parent_face_to_shell[f_uuid] = item.uuid

#     # Cell -> shells
#     for item in items:
#         if item.type == "Cell" and item.shells:
#             for s_uuid in item.shells:
#                 parent_shell_to_cell[s_uuid] = item.uuid

#     # CellComplex -> cells
#     for item in items:
#         if item.type == "CellComplex" and item.cells:
#             for c_uuid in item.cells:
#                 parent_cell_to_cc[c_uuid] = item.uuid

#     # Face -> wires
#     for item in items:
#         if item.type == "Face" and item.wires:
#             for w_uuid in item.wires:
#                 parent_wire_to_face[w_uuid] = item.uuid

#     # Wire -> edges
#     for item in items:
#         if item.type == "Wire" and item.edges:
#             for e_uuid in item.edges:
#                 parent_edge_to_wire[e_uuid] = item.uuid

#     # Edge -> vertices
#     for item in items:
#         if item.type == "Edge" and item.vertices:
#             v0, v1 = item.vertices
#             parent_vertex_to_edge[v0] = item.uuid
#             parent_vertex_to_edge[v1] = item.uuid

#     # --- Build a unified parents map for each uuid ---
#     parents: Dict[str, Dict[str, Optional[str]]] = {
#         uuid: {lvl: None for lvl in LEVELS} for uuid in by_uuid.keys()
#     }

#     # Fill direct parents
#     for f_uuid, s_uuid in parent_face_to_shell.items():
#         parents[f_uuid]["Shell"] = s_uuid
#     for s_uuid, c_uuid in parent_shell_to_cell.items():
#         parents[s_uuid]["Cell"] = c_uuid
#     for c_uuid, cc_uuid in parent_cell_to_cc.items():
#         parents[c_uuid]["CellComplex"] = cc_uuid
#     for w_uuid, f_uuid in parent_wire_to_face.items():
#         parents[w_uuid]["Face"] = f_uuid
#     for e_uuid, w_uuid in parent_edge_to_wire.items():
#         parents[e_uuid]["Wire"] = w_uuid
#     for v_uuid, e_uuid in parent_vertex_to_edge.items():
#         parents[v_uuid]["Edge"] = e_uuid
#     for uuid, cl_id in cluster_of.items():
#         parents[uuid]["Cluster"] = cl_id

#     # Self as own level
#     for uuid, item in by_uuid.items():
#         typ = item.type
#         if typ in LEVELS:
#             parents[uuid][typ] = uuid
#         # Cluster ids themselves: treat id string as "Cluster" root
#         cl_id = item.dictionary.get("__cluster__")
#         if cl_id:
#             # create entry for cluster id if not present
#             if cl_id not in parents:
#                 parents[cl_id] = {lvl: None for lvl in LEVELS}
#                 parents[cl_id]["Cluster"] = cl_id

#     # --- Faces -> triangles (outer wire only, no holes yet) ---
#     processed_faces: List[ProcessedFace] = []

#     def wire_vertices_order(wire_uuid: str) -> List[List[float]]:
#         wire = by_uuid.get(wire_uuid)
#         if not wire or not wire.edges:
#             return []
#         edges = [by_uuid[eid] for eid in wire.edges if eid in by_uuid]

#         # assume edges list already describes loop; walk them
#         if not edges:
#             return []
#         first_edge = edges[0]
#         v0, v1 = first_edge.vertices
#         coords: List[List[float]] = []
#         coords.append(vertex_coords.get(v0, [0, 0, 0]))
#         current = v1
#         for edge in edges[1:]:
#             a, b = edge.vertices
#             if a == current:
#                 nxt = b
#             elif b == current:
#                 nxt = a
#             else:
#                 # fallback if ordering is weird
#                 nxt = a
#             coords.append(vertex_coords.get(nxt, [0, 0, 0]))
#             current = nxt
#         return coords

#     def triangulate_fan(points: List[List[float]]) -> List[List[List[float]]]:
#         # very simple fan triangulation (assumes convex-ish, no holes)
#         if len(points) < 3:
#             return []
#         tris: List[List[List[float]]] = []
#         p0 = points[0]
#         for i in range(1, len(points) - 1):
#             tris.append([p0, points[i], points[i + 1]])
#         return tris

#     for item in items:
#         if item.type != "Face" or not item.wires:
#             continue
#         outer_wire_uuid = item.wires[0]  # ignore inner wires (holes) for now
#         ring = wire_vertices_order(outer_wire_uuid)
#         tris = triangulate_fan(ring)
#         processed_faces.append(
#             ProcessedFace(
#                 uuid=item.uuid,
#                 triangles=tris,
#                 dictionary=item.dictionary or {},
#             )
#         )

#     return UploadResponse(
#         vertices=processed_vertices,
#         faces=processed_faces,
#         parents=ParentsMap(parents=parents),
#         raw=[i.dict() for i in items],
#     )





@app.post("/upload-ifc")
async def upload_ifc(include_path: bool = False, tilt_min: float = 0.3, max_z_span: float = 1.0, min_floor_area: float = 9.0, file: UploadFile = File(...)):
    """
    Accept IFC upload and return a viewer contract.
    - include_path=True will also generate slab grid wires and a simple egress path.
    """
    filename = file.filename or "uploaded.ifc"
    if not filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Only .ifc files are supported.")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file provided.")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ifc")
    try:
        tmp.write(data)
        tmp.flush()
        tmp.close()
        return convert_ifc_to_contract(tmp.name, include_path=include_path, tilt_min=tilt_min, max_z_span=max_z_span, min_floor_area=min_floor_area)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


def convert_ifc_to_contract(file_path: str, include_path: bool = False, tilt_min: float = 0.3, max_z_span: float = 1.0, min_floor_area: float = 9.0):
    """
    Parse IFC with ifcopenshell.geom and return viewer contract.
    If include_path is True, generate:
      - grid wires on each floor/slab element
      - a simple multi-floor egress polyline (red, thick)
    """
    try:
        import ifcopenshell
        import ifcopenshell.geom
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"IFC library unavailable: {exc}")

    try:
        model = ifcopenshell.open(file_path)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Failed to read IFC: {exc}")

    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)

    vertices = []
    vertex_index = {}
    faces = []
    raw = []
    edges = []
    vertex_uid_seq = 0
    face_uid_seq = 0
    edge_uid_seq = 0

    floor_bboxes = []
    floor_bboxes_hint = []
    stair_bboxes_hint = []
    cell_nodes = []
    cell_adjacency = {}
    cell_bboxes = []
    all_bbox = {
        "minx": math.inf,
        "maxx": -math.inf,
        "miny": math.inf,
        "maxy": -math.inf,
        "minz": math.inf,
        "maxz": -math.inf,
    }
    # Geometry-first stair detection buffers and thresholds
    tread_candidates = []
    HORIZONTAL_DOT = 0.95
    TREAD_AREA_MIN = 0.25
    TREAD_AREA_MAX = 3.00
    RISER_MIN = 0.12
    RISER_MAX = 0.22
    MIN_STEPS = 4
    MAX_TREAD_WIDTH = 1.8
    DELTA_Z_TOL = 0.04
    XY_OVERLAP_MIN = 0.25

    def add_vertex(x, y, z):
        nonlocal vertex_uid_seq, all_bbox
        key = (round(x, 6), round(y, 6), round(z, 6))
        if key in vertex_index:
            return vertex_index[key]
        uid = f"v{vertex_uid_seq}"
        vertex_uid_seq += 1
        vertex_index[key] = uid
        vertices.append({
            "type": "Vertex",
            "uid": uid,
            "coordinates": [x, y, z],
            "dictionary": {"uid": uid},
        })
        all_bbox["minx"] = min(all_bbox["minx"], x)
        all_bbox["maxx"] = max(all_bbox["maxx"], x)
        all_bbox["miny"] = min(all_bbox["miny"], y)
        all_bbox["maxy"] = max(all_bbox["maxy"], y)
        all_bbox["minz"] = min(all_bbox["minz"], z)
        all_bbox["maxz"] = max(all_bbox["maxz"], z)
        return uid

    def add_edge(v0, v1, color="red", width=2, kind=None):
        nonlocal edge_uid_seq
        uid = f"e{edge_uid_seq}"
        edge_uid_seq += 1
        edge_dict = {"edgeColor": color, "edgeWidth": width}
        if kind:
            edge_dict["edgeKind"] = kind
        edges.append({
            "type": "Edge",
            "uid": uid,
            "vertices": [v0, v1],
            "dictionary": edge_dict,
        })
        return uid

    def is_floor(product) -> bool:
        try:
            ptype = product.is_a()
        except Exception:
            ptype = ""
        name = getattr(product, "Name", "") or ""
        upper = str(name).upper()
        return (
            "SLAB" in upper
            or "FLOOR" in upper
            or "SLAB" in ptype.upper()
            or "FLOOR" in ptype.upper()
        )

    def _overlap_xy(a, b):
        dx = min(a["maxx"], b["maxx"]) - max(a["minx"], b["minx"])
        dy = min(a["maxy"], b["maxy"]) - max(a["miny"], b["miny"])
        return dx, dy

    



    def detect_stairs_from_treads(cands):
        if not cands:
            return []
        filtered = []
        for c in cands:
            width = max(c["maxx"] - c["minx"], c["maxy"] - c["miny"])
            if width <= MAX_TREAD_WIDTH and TREAD_AREA_MIN <= c.get("area", 0) <= TREAD_AREA_MAX:
                filtered.append(c)
        if len(filtered) < MIN_STEPS:
            return []
        filtered.sort(key=lambda x: x["z"])
        chains = []
        visited = set()
        for i, base in enumerate(filtered):
            if i in visited:
                continue
            chain = [i]
            visited.add(i)
            current = i
            while True:
                best = None
                best_dz = 1e9
                cz = filtered[current]["z"]
                for j in range(current + 1, len(filtered)):
                    if j in visited:
                        continue
                    dz = filtered[j]["z"] - cz
                    if dz < RISER_MIN:
                        continue
                    if dz > RISER_MAX:
                        break
                    dx, dy = _overlap_xy(filtered[current], filtered[j])
                    if dx < XY_OVERLAP_MIN or dy < XY_OVERLAP_MIN:
                        continue
                    if dz < best_dz:
                        best = j
                        best_dz = dz
                if best is None:
                    break
                chain.append(best)
                visited.add(best)
                current = best
            if len(chain) >= MIN_STEPS:
                chains.append(chain)
        # drop very shallow stacks (likely noise)
        filtered_chains = []
        for ch in chains:
            zs_tmp = [filtered[idx]["z"] for idx in ch]
            if (max(zs_tmp) - min(zs_tmp)) >= 0.6:
                filtered_chains.append(ch)
        chains = filtered_chains

        results = []
        for chain in chains:
            zs = [filtered[idx]["z"] for idx in chain]
            dzs = [zs[k+1] - zs[k] for k in range(len(zs)-1)]
            if dzs and (max(dzs) - min(dzs)) > DELTA_Z_TOL:
                continue
            minx = min(filtered[idx]["minx"] for idx in chain)
            maxx = max(filtered[idx]["maxx"] for idx in chain)
            miny = min(filtered[idx]["miny"] for idx in chain)
            maxy = max(filtered[idx]["maxy"] for idx in chain)
            z_min = min(filtered[idx].get("z", 0.0) for idx in chain)
            z_max = max(filtered[idx].get("z", 0.0) for idx in chain)
            pts = [
                (minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy)
            ]
            results.append({
                "minx": minx,
                "maxx": maxx,
                "miny": miny,
                "maxy": maxy,
                "z": sum(zs) / len(zs),
                "z_min": z_min,
                "z_max": z_max,
                "pts": pts,
                "is_stair": True,
            })
        return results

    for product in model.by_type("IfcProduct"):
        if not getattr(product, "Representation", None):
            continue
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
        except Exception:
            continue
        geom = shape.geometry
        verts = getattr(geom, "verts", None) or getattr(geom, "vertices", None)
        inds = getattr(geom, "faces", None) or getattr(geom, "indices", None)
        name_upper = str(getattr(product, "Name", "") or "").upper()
        otype_upper = str(getattr(product, "ObjectType", "") or "").upper()
        try:
            ptype_upper = product.is_a().upper()
        except Exception:
            ptype_upper = ""
        hint_floor = any(w in name_upper for w in ["FLOOR", "SLAB"]) or any(w in otype_upper for w in ["FLOOR", "SLAB"]) or ("FLOOR" in ptype_upper or "SLAB" in ptype_upper)
        hint_stair = any(w in name_upper for w in ["STAIR", "STAIRS"]) or any(w in otype_upper for w in ["STAIR", "STAIRS"]) or ("STAIR" in ptype_upper)

        pxs, pys, pzs = [], [], []
        normals = []

        for i in range(0, len(inds), 3):
            tri_idxs = inds[i : i + 3]
            tri_coords = []
            for idx in tri_idxs:
                base = idx * 3
                x, y, z = verts[base : base + 3]
                add_vertex(x, y, z)
                tri_coords.append([x, y, z])
                pxs.append(x)
                pys.append(y)
                pzs.append(z)
            if len(tri_coords) == 3:
                (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = tri_coords
                ux, uy, uz = x2 - x1, y2 - y1, z2 - z1
                vx, vy, vz = x3 - x1, y3 - y1, z3 - z1
                cx, cy, cz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
                norm = math.sqrt(cx * cx + cy * cy + cz * cz)
                if norm > 1e-6:
                    nx, ny, nz = cx / norm, cy / norm, cz / norm
                    normals.append((nx, ny, nz))
                    if include_path and abs(nz) >= HORIZONTAL_DOT and len(tri_coords) == 3:
                        area = 0.5 * norm
                        if TREAD_AREA_MIN <= area <= TREAD_AREA_MAX:
                            xs = [p[0] for p in tri_coords]
                            ys = [p[1] for p in tri_coords]
                            zs_ = [p[2] for p in tri_coords]
                            w_x = max(xs) - min(xs)
                            w_y = max(ys) - min(ys)
                            if max(w_x, w_y) <= MAX_TREAD_WIDTH:
                                cxc = sum(xs) / 3.0
                                cyc = sum(ys) / 3.0
                                czc = sum(zs_) / 3.0
                                tread_candidates.append({
                                    "minx": min(xs),
                                    "maxx": max(xs),
                                    "miny": min(ys),
                                    "maxy": max(ys),
                                    "z": czc,
                                    "area": area,
                                })
            faces.append({
                "type": "Face",
                "uid": f"f{face_uid_seq}",
                "triangles": [tri_coords],
                "dictionary": {
                    "ifc_guid": getattr(product, "GlobalId", None),
                    "ifc_name": getattr(product, "Name", None),
                    "ifc_type": product.is_a() if hasattr(product, "is_a") else None,
                },
            })
            face_uid_seq += 1

        raw.append({
            "ifc_guid": getattr(product, "GlobalId", None),
            "ifc_name": getattr(product, "Name", None),
            "ifc_type": product.is_a() if hasattr(product, "is_a") else None,
        })

        if include_path and (is_floor(product) or hint_stair) and pxs and pys and pzs:
            span_x = max(pxs) - min(pxs)
            span_y = max(pys) - min(pys)
            area = span_x * span_y
            z_span = max(pzs) - min(pzs)
            avg_abs_nz = sum(abs(n[2]) for n in normals) / len(normals) if normals else 1.0
            heuristic_stair = (1.0 <= z_span <= 4.0) and max(span_x, span_y) >= 1.0 and 0.2 <= avg_abs_nz <= 0.8 and (z_span / max(span_x, span_y) >= 0.2)
            is_stair_flag = bool(hint_stair or heuristic_stair)
            if area >= min_floor_area and (avg_abs_nz >= tilt_min or z_span <= max_z_span):
                pts2d = list({(round(x, 4), round(y, 4)) for x, y in zip(pxs, pys)})
                record = {
                    "minx": min(pxs),
                    "maxx": max(pxs),
                    "miny": min(pys),
                    "maxy": max(pys),
                    "z": sum(pzs) / len(pzs),
                    "z_min": min(pzs),
                    "z_max": max(pzs),
                    "pts": pts2d,
                    "is_stair": is_stair_flag,
                }
                floor_bboxes.append(record)
                if hint_floor:
                    floor_bboxes_hint.append(dict(record))
                if is_stair_flag:
                    stair_bboxes_hint.append(dict(record))

    # build coarse cell graph (simple mode)
    cell_nodes, cell_adjacency, cell_step = _build_cell_grid(all_bbox)

    if include_path:
        # geometry-first stair detection from tread candidates (no IFC annotations needed)
        detected_stairs = detect_stairs_from_treads(tread_candidates)
        if detected_stairs:
            floor_bboxes.extend(detected_stairs)

    if include_path:
        # Prefer named floors if provided; otherwise use geometric detection
        if floor_bboxes_hint:
            floor_bboxes = floor_bboxes_hint
        if stair_bboxes_hint:
            floor_bboxes = floor_bboxes + stair_bboxes_hint

    if include_path and floor_bboxes:
        # merge nearby floors (mezzanines) to reduce duplicate grids
        floor_bboxes.sort(key=lambda b: b["z"])
        merged = []
        MIN_GAP = 1.5  # meters: floors closer than this merge together

        for bbox in floor_bboxes:
            record = dict(bbox)
            record["count"] = 1
            if not merged:
                merged.append(record)
                continue
            last = merged[-1]
            if abs(record["z"] - last["z"]) <= MIN_GAP and bool(record.get("is_stair")) == bool(last.get("is_stair")):
                last["minx"] = min(last["minx"], record["minx"])
                last["maxx"] = max(last["maxx"], record["maxx"])
                last["miny"] = min(last["miny"], record["miny"])
                last["maxy"] = max(last["maxy"], record["maxy"])
                last["z_min"] = min(last.get("z_min", last["z"]), record.get("z_min", record["z"]))
                last["z_max"] = max(last.get("z_max", last["z"]), record.get("z_max", record["z"]))
                pts = last.get("pts", []) + record.get("pts", [])
                last["pts"] = pts
                last["is_stair"] = last.get("is_stair") or record.get("is_stair")
                last["count"] += 1
                last["z"] = (last["z"] * (last["count"] - 1) + record["z"]) / last["count"]
            else:
                merged.append(record)
        floor_bboxes = []
        for m in merged:
            m.pop("count", None)
            floor_bboxes.append(m)

        # precompute medial axis (approx.) per floor using PCA of footprint points

        for bbox in floor_bboxes:
            pts = bbox.get("pts") or []
            if len(pts) >= 2:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                sx = sy = sxy = 0.0
                for x, y in pts:
                    dx = x - cx
                    dy = y - cy
                    sx += dx * dx
                    sy += dy * dy
                    sxy += dx * dy
                n = len(pts)
                sx /= n; sy /= n; sxy /= n
                trace = sx + sy
                det = sx * sy - sxy * sxy
                tmp = (trace * trace) / 4 - det
                lam = trace / 2 + (tmp ** 0.5 if tmp > 0 else 0)
                if abs(sxy) < 1e-6 and abs(lam - sx) < 1e-9:
                    dir_vec = (1.0, 0.0)
                else:
                    dir_vec = (sxy, lam - sx)
                norm = (dir_vec[0] ** 2 + dir_vec[1] ** 2) ** 0.5
                if norm < 1e-9:
                    dir_vec = (1.0, 0.0)
                    norm = 1.0
                dir_vec = (dir_vec[0] / norm, dir_vec[1] / norm)

                # project points on axis to find span
                min_t = 1e18
                max_t = -1e18
                for x, y in pts:
                    t = (x - cx) * dir_vec[0] + (y - cy) * dir_vec[1]
                    if t < min_t:
                        min_t = t
                    if t > max_t:
                        max_t = t
                start_xy = (cx + dir_vec[0] * min_t, cy + dir_vec[1] * min_t)
                end_xy = (cx + dir_vec[0] * max_t, cy + dir_vec[1] * max_t)
                z = bbox["z"] + 0.1
                bbox["medial_start"] = (start_xy[0], start_xy[1], z)
                bbox["medial_end"] = (end_xy[0], end_xy[1], z)

        # Split floors and stairs for path generation
        floors_only = [b for b in floor_bboxes if not b.get("is_stair")]
        stairs_only = [b for b in floor_bboxes if b.get("is_stair")]
        if not floors_only:
            floors_only = list(floor_bboxes)

        # Add stair connectors
        stair_connectors = []
        for st in stairs_only:
            if st.get("z_min") is None or st.get("z_max") is None:
                continue
            cx = 0.5 * (st["minx"] + st["maxx"])
            cy = 0.5 * (st["miny"] + st["maxy"])
            lower_uid = add_vertex(cx, cy, st["z_min"] + 0.05)
            upper_uid = add_vertex(cx, cy, st["z_max"] - 0.05)
            add_edge(lower_uid, upper_uid, color="red", width=4, kind="stair")
            stair_connectors.append({
                "cx": cx,
                "cy": cy,
                "z_min": st["z_min"],
                "z_max": st["z_max"],
                "lower_uid": lower_uid,
                "upper_uid": upper_uid,
            })


        for bbox in floor_bboxes:
            span_x = bbox["maxx"] - bbox["minx"]
            span_y = bbox["maxy"] - bbox["miny"]
            step = max(min(span_x, span_y) / (6.0 if bbox.get("is_stair") else 12.0), 0.5 if bbox.get("is_stair") else 0.75)
            z = bbox["z"] + 0.05
            x_values = []
            y_values = []
            n_x = max(2, int(span_x / step) + 1)
            n_y = max(2, int(span_y / step) + 1)
            for i in range(n_x + 1):
                x_values.append(bbox["minx"] + i * step)
            for j in range(n_y + 1):
                y_values.append(bbox["miny"] + j * step)

            grid_color = "#88bbff" if bbox.get("is_stair") else "#8888ff"
            for x in x_values:
                v0 = add_vertex(x, bbox["miny"], z)
                v1 = add_vertex(x, bbox["maxy"], z)
                add_edge(v0, v1, color=grid_color, width=1, kind="grid")
            for y in y_values:
                v0 = add_vertex(bbox["minx"], y, z)
                v1 = add_vertex(bbox["maxx"], y, z)
                add_edge(v0, v1, color=grid_color, width=1, kind="grid")
            # For stairs, add a simple diagonal guide along the run
            if bbox.get("is_stair") and bbox.get("z_max") is not None:
                diag_start = add_vertex(bbox["minx"], bbox["miny"], bbox.get("z_min", bbox["z"]))
                diag_end = add_vertex(bbox["maxx"], bbox["maxy"], bbox.get("z_max", bbox["z"]))
                add_edge(diag_start, diag_end, color="red", width=2, kind="stair_diag")

        # Build path using stairs when available
        path_points = []

        def floor_center(bb):
            return (
                0.5 * (bb["minx"] + bb["maxx"]),
                0.5 * (bb["miny"] + bb["maxy"]),
                bb["z"] + 0.1,
            )

        floor_sorted = sorted(floors_only, key=lambda b: b["z"])
        if stair_connectors and len(floor_sorted) >= 2:
            path_points.append(floor_center(floor_sorted[0]))
            for i in range(len(floor_sorted) - 1):
                f0 = floor_sorted[i]
                f1 = floor_sorted[i + 1]
                mid_z = 0.5 * (f0["z"] + f1["z"])
                stair = min(stair_connectors, key=lambda s: abs(mid_z - 0.5 * (s["z_min"] + s["z_max"])))
                path_points.append((stair["cx"], stair["cy"], stair["z_min"] + 0.05))
                path_points.append((stair["cx"], stair["cy"], stair["z_max"] - 0.05))
                path_points.append(floor_center(f1))
        else:
            # fallback: use medial start/end per floor
            for bbox in floor_bboxes:
                start = bbox.get("medial_start")
                end = bbox.get("medial_end")
                if start and end:
                    path_points.extend([start, end])
                else:
                    cx = 0.5 * (bbox["minx"] + bbox["maxx"])
                    cy = 0.5 * (bbox["miny"] + bbox["maxy"])
                    cz = bbox["z"] + 0.1
                    path_points.append((cx, cy, cz))

        if len(path_points) >= 2:
            prev_uid = add_vertex(*path_points[0])
            for pt in path_points[1:]:
                cur_uid = add_vertex(*pt)
                add_edge(prev_uid, cur_uid, color="red", width=6, kind="path")
                prev_uid = cur_uid


    global LAST_GRAPHS
    vertex_coords = _vertex_coord_map(vertices)
    wire_adjacency = _build_adjacency(edges, vertex_coords.keys(), {"grid", "stair", "stair_diag"})
    cell_coords = {n["id"]: n["center"] for n in cell_nodes}
    LAST_GRAPHS = {
        "wire": {
            "adjacency": wire_adjacency,
            "coords": vertex_coords,
            "edges": edges,
            "step": _estimate_step_size(vertex_coords, wire_adjacency),
        },
        "cell": {
            "adjacency": cell_adjacency,
            "coords": cell_coords,
            "bboxes": cell_nodes,
            "step": cell_step,
        },
    }

    if not faces:
        raise HTTPException(status_code=422, detail="IFC parsed but no renderable geometry found.")

    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "raw": raw,
    }
