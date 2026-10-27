"""Request and response models for the Next API.

Two request shapes reach the graph builder:

``PointCloudRequest``  (preferred, ``POST /api/ifc/graph``)
    The browser has already parsed the IFC into fragments, so it samples
    walkable points in a worker and sends a few thousand of them. Payloads
    drop from tens of megabytes of triangles to a couple of hundred kilobytes.

``IfcEgressRequest``  (legacy, ``POST /ifc-egress-graph``)
    Raw triangle soup, exactly as the Classic frontend sends it. Kept so the
    Next backend can serve the Classic frontend unchanged, which is what makes
    an A/B comparison on one machine possible.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

UpAxis = Literal["x", "y", "z"]
Engine = Literal["fast", "topologicpy"]


class GraphOptions(BaseModel):
    up_axis: UpAxis = "z"
    agent_height: float = Field(0.75, ge=0.0, le=3.0)
    max_edge_floor: float = Field(2.25, gt=0.0, le=50.0)
    max_edge_stair: float = Field(0.4, gt=0.0, le=10.0)
    #: Largest height difference a floor-to-floor link may span. Keeps the floor
    #: mesh flat instead of bracing nearby surfaces into a space-frame truss.
    max_edge_rise: float = Field(0.35, gt=0.0, le=5.0)
    #: Height difference still treated as the same slab when collapsing a
    #: vertical column down to its walking surface.
    column_gap: float = Field(0.9, gt=0.0, le=5.0)
    use_walls: bool = True
    rectilinear: bool = False
    grid_snap: bool = False
    grid_cell_size: Optional[float] = Field(None, gt=0.0, le=10.0)
    decimate: float = Field(0.0, ge=0.0, le=5.0)
    #: Neighbour cap per node. Keeps the graph sparse without coarsening
    #: the sampling, which is what keeps doorways navigable.
    max_degree: int = Field(12, ge=2, le=64)
    max_points: int = Field(20000, ge=16, le=400000)


class WallSegment(BaseModel):
    """A wall reduced to a 2D centreline plus its vertical extent."""

    segment: List[List[float]]
    thickness: float = 0.1
    up_min: float = -1e9
    up_max: float = 1e9


class PointCloudRequest(BaseModel):
    """Pre-sampled navigation points, produced by the browser worker."""

    floor_points: List[List[float]] = []
    stair_points: List[List[float]] = []
    door_points: List[List[float]] = []
    walls: List[WallSegment] = []
    options: GraphOptions = Field(default_factory=GraphOptions)
    graph_id: Optional[str] = None


class GraphImportRequest(BaseModel):
    """A graph the browser already built (src/core/lattice.js), for the
    features that still run here. Arrays are base64, as in GraphResponse."""

    #: float32, 3 per node.
    nodes_b64: str
    #: uint32, 2 per edge.
    edges_b64: str
    #: uint8, one kind per node (0 floor, 1 stair, 2 door).
    kinds_b64: str
    #: uint8, one per edge: 1 where a wall blocks it.
    blocked_b64: Optional[str] = None
    up_axis: UpAxis = "y"
    agent_height: float = Field(0.75, ge=0.0, le=3.0)
    spacing: Optional[float] = None
    angle_deg: Optional[float] = None
    wall_segments: int = 0


class IfcGeometry(BaseModel):
    expressID: int = 0
    vertices: List[float] = []
    indices: List[int] = []
    normals: Optional[List[float]] = None


class IfcEgressRequest(BaseModel):
    """Classic triangle-soup contract. Field names are deliberately unchanged."""

    floors: List[IfcGeometry] = []
    stairs: List[IfcGeometry] = []
    doors: List[IfcGeometry] = []
    walls: List[IfcGeometry] = []
    use_walls: bool = True
    agent_height: float = 0.75
    base_spacing: float = 0.5
    stair_multiplier: float = 0.5
    max_edge_length: float = 1.5
    max_edge_floor: Optional[float] = None
    max_edge_stair: Optional[float] = None
    up_axis: str = "z"
    max_points: int = 20000
    rectilinear: bool = False
    grid_snap: bool = False
    grid_cell_size: Optional[float] = None


class GraphStats(BaseModel):
    nodes: int
    edges: int
    floor_nodes: int = 0
    stair_nodes: int = 0
    door_nodes: int = 0
    wall_segments: int = 0
    blocked_edges: int = 0
    components: int = 1
    largest_component: int = 0


class GraphResponse(BaseModel):
    graph_id: str
    mode: str = "ifc"
    stats: GraphStats
    up_axis: str = "z"
    bounds: List[List[float]] = []
    #: base64 float32, 3 values per node.
    nodes_b64: Optional[str] = None
    #: base64 uint32, 2 values per edge.
    edges_b64: Optional[str] = None
    #: base64 uint8, one node kind per node (0 floor, 1 stair, 2 door).
    kinds_b64: Optional[str] = None
    timings: Dict[str, float] = {}


class PathRequest(BaseModel):
    graph_id: Optional[str] = None
    mode: str = "ifc"
    start_point: Optional[List[float]] = None
    end_point: Optional[List[float]] = None
    start_id: Optional[str] = None
    end_id: Optional[str] = None
    engine: Engine = "fast"
    use_walls: bool = True
    alpha: float = Field(0.0, ge=0.0, le=10.0)
    lethality_threshold: Optional[float] = None
    temperatures: Optional[Dict[str, float]] = None


class PathResponse(BaseModel):
    mode: str = "ifc"
    graph_id: Optional[str] = None
    found: bool
    points: List[List[float]] = []
    node_ids: List[int] = []
    cost: float = 0.0
    length: float = 0.0
    engine: str = "fast"
    fallback_from: Optional[str] = None
    note: Optional[str] = None


class ComparePathResponse(BaseModel):
    """Side-by-side output of both engines on one graph, for parity checks."""

    graph_id: Optional[str] = None
    fast: PathResponse
    topologicpy: PathResponse
    same_route: bool
    cost_delta: float
    fast_ms: float
    topologicpy_ms: float


class AlternativesRequest(PathRequest):
    """Several routes between one start and exit. See ``graphs/alternatives``."""

    k: int = Field(3, ge=1, le=8)
    #: ``distinct``: routes through different space. ``yen``: the k shortest
    #: loopless paths, TGraph.KShortestPaths' semantics.
    method: Literal["distinct", "yen"] = "distinct"
    #: Largest share of a route that may run within ``separation`` of a route
    #: already kept (distinct only).
    max_overlap: float = Field(0.5, ge=0.0, le=1.0)
    #: Longest route kept, as a multiple of the shortest cost (distinct only).
    max_stretch: float = Field(1.5, ge=1.0, le=10.0)
    #: Metres within which two routes count as sharing space.
    separation: float = Field(1.5, gt=0.0, le=20.0)


class AlternativeRoute(PathResponse):
    rank: int = 1
    #: Cost over the best route's cost.
    stretch: float = 1.0
    #: Share of the length within ``separation`` of an earlier route.
    overlap: float = 0.0


class AlternativesResponse(BaseModel):
    graph_id: Optional[str] = None
    method: str
    engine: str
    requested: int
    routes: List[AlternativeRoute] = []
    ms: float = 0.0
    note: Optional[str] = None


class NavigationStorey(BaseModel):
    id: str
    name: str
    elevation: float
    navigation_nodes: int
    studio_nodes: int
    navigation_covered: Optional[float] = None
    studio_covered: Optional[float] = None


class NavigationRoute(BaseModel):
    found: bool
    studio_length: Optional[float] = None
    navigation_length: Optional[float] = None
    #: Navigation's route length over Studio's.
    ratio: Optional[float] = None
    points: List[List[float]] = []
    ms: float = 0.0
    note: Optional[str] = None


class NavigationCheckResponse(BaseModel):
    """Studio's graph against TopologicPy's ``Navigation`` graph of the same IFC."""

    graph_id: Optional[str] = None
    spacing: float
    #: Distance within which a node counts as covered by the other graph.
    radius: float
    navigation_nodes: int
    navigation_edges: int
    navigation_components: int
    navigation_kinds: Dict[str, int] = {}
    studio_nodes: int
    studio_edges: int
    #: "coordination": the viewer's own matrix, exact. "fitted": ICP fallback.
    alignment: str = "fitted"
    #: Translation carrying Navigation's turned points onto Studio's.
    offset: List[float]
    residual_median: float
    residual_p90: float
    #: Share of Navigation's nodes with a Studio node within ``radius``.
    navigation_covered: float
    #: Share of Studio's nodes with a Navigation node within ``radius``.
    studio_covered: float
    studio_covered_by_kind: Dict[str, float] = {}
    storeys: List[NavigationStorey] = []
    route: Optional[NavigationRoute] = None
    #: Navigation's graph in the viewer frame, encoded like GraphResponse.
    nodes_b64: Optional[str] = None
    edges_b64: Optional[str] = None
    timings: Dict[str, float] = {}
    cached: bool = False
    notes: List[str] = []


class FireSimRequest(BaseModel):
    graph_id: Optional[str] = None
    mode: str = "ifc"
    model: Literal["radial", "flood", "temperature"] = "radial"
    start_id: Optional[str] = None
    end_id: Optional[str] = None
    start_point: Optional[List[float]] = None
    end_point: Optional[List[float]] = None
    max_steps: int = Field(60, ge=1, le=2000)
    delay_ms: int = Field(200, ge=0, le=10000)
    use_walls: bool = True


class RLRequest(BaseModel):
    graph_id: Optional[str] = None
    mode: str = "ifc"
    start_id: Optional[str] = None
    exit_id: Optional[str] = None
    start_point: Optional[List[float]] = None
    exit_point: Optional[List[float]] = None
    #: Where the fire starts. Classic (and Next until 2026-10) seeded it at the
    #: evacuee's own start node; without an origin the policy trains fire-free.
    fire_id: Optional[str] = None
    fire_point: Optional[List[float]] = None
    episodes: int = Field(200, ge=1, le=20000)
    max_steps: int = Field(200, ge=2, le=5000)
    use_fire: bool = True
    #: Fire hops per evacuee move. 1.0 is Classic: the front moves exactly as
    #: fast as the agent, which on a real floor plan usually leaves no escape.
    fire_speed: float = Field(1.0, gt=0.0, le=10.0)
    use_walls: bool = True
    epsilon: float = Field(0.2, ge=0.0, le=1.0)
    alpha: float = Field(0.5, gt=0.0, le=1.0)
    gamma: float = Field(0.99, gt=0.0, le=1.0)
    shaping: bool = True
    seed: Optional[int] = None


class RLResponse(BaseModel):
    graph_id: Optional[str] = None
    mode: str = "ifc"
    path: List[int] = []
    points: List[List[float]] = []
    reached_exit: bool = False
    episodes: int = 0
    burned: bool = False
    reward: float = 0.0
    length: float = 0.0
    optimal_length: Optional[float] = None
    success_rate: float = 0.0
    curve: List[float] = []
    escape_steps: Optional[int] = None
    fire_start_id: Optional[int] = None
    train_ms: float = 0.0
    note: Optional[str] = None


class IfcEgressPathRequest(BaseModel):
    """Classic path contract."""

    start_point: Optional[List[float]] = None
    end_point: Optional[List[float]] = None
    graph_id: Optional[str] = None
    engine: Optional[Engine] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    topologicpy: Dict[str, Any] = {}
    ifcopenshell: Optional[str] = None
    graphs: Dict[str, Any] = {}
