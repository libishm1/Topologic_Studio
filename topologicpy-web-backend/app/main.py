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
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import math


app = FastAPI(
    title="TopologicPy Web Backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

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
    all_bbox = {
        "minx": math.inf,
        "maxx": -math.inf,
        "miny": math.inf,
        "maxy": -math.inf,
        "minz": math.inf,
        "maxz": -math.inf,
    }

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

    def add_edge(v0, v1, color="red", width=2):
        nonlocal edge_uid_seq
        uid = f"e{edge_uid_seq}"
        edge_uid_seq += 1
        edges.append({
            "type": "Edge",
            "uid": uid,
            "vertices": [v0, v1],
            "dictionary": {"edgeColor": color, "edgeWidth": width},
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
        if not verts or not inds:
            continue

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
                    normals.append((cx / norm, cy / norm, cz / norm))
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

        if include_path and is_floor(product) and pxs and pys and pzs:
            span_x = max(pxs) - min(pxs)
            span_y = max(pys) - min(pys)
            area = span_x * span_y
            z_span = max(pzs) - min(pzs)
            avg_abs_nz = sum(abs(n[2]) for n in normals) / len(normals) if normals else 1.0
            if area >= min_floor_area and (avg_abs_nz >= tilt_min or z_span <= max_z_span):
                pts2d = list({(round(x, 4), round(y, 4)) for x, y in zip(pxs, pys)})
                floor_bboxes.append(
                    {
                        "minx": min(pxs),
                        "maxx": max(pxs),
                        "miny": min(pys),
                        "maxy": max(pys),
                        "z": sum(pzs) / len(pzs),
                        "pts": pts2d,
                    }
                )

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
            if abs(record["z"] - last["z"]) <= MIN_GAP:
                last["minx"] = min(last["minx"], record["minx"])
                last["maxx"] = max(last["maxx"], record["maxx"])
                last["miny"] = min(last["miny"], record["miny"])
                last["maxy"] = max(last["maxy"], record["maxy"])
                pts = last.get("pts", []) + record.get("pts", [])
                last["pts"] = pts
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

        for bbox in floor_bboxes:
            span_x = bbox["maxx"] - bbox["minx"]
            span_y = bbox["maxy"] - bbox["miny"]
            step = max(min(span_x, span_y) / 12.0, 0.75)
            z = bbox["z"] + 0.05
            x_values = []
            y_values = []
            n_x = max(2, int(span_x / step) + 1)
            n_y = max(2, int(span_y / step) + 1)
            for i in range(n_x + 1):
                x_values.append(bbox["minx"] + i * step)
            for j in range(n_y + 1):
                y_values.append(bbox["miny"] + j * step)

            for x in x_values:
                v0 = add_vertex(x, bbox["miny"], z)
                v1 = add_vertex(x, bbox["maxy"], z)
                add_edge(v0, v1, color="#8888ff", width=1)
            for y in y_values:
                v0 = add_vertex(bbox["minx"], y, z)
                v1 = add_vertex(bbox["maxx"], y, z)
                add_edge(v0, v1, color="#8888ff", width=1)

        # Build path along medial axes instead of simple centroids
        path_points = []
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
                add_edge(prev_uid, cur_uid, color="red", width=6)
                prev_uid = cur_uid


    if not faces:
        raise HTTPException(status_code=422, detail="IFC parsed but no renderable geometry found.")

    return {
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "raw": raw,
    }
