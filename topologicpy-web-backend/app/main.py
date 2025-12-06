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
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="TopologicPy Web Backend",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # fine for local dev
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
    - the original TopologicPy export (list of dicts), OR
    - a pre-converted 'contract' dict with vertices/edges/faces/etc.

    You will use your TopologicPy helper to turn 'payload'
    into the viewer contract JSON.
    """
    # CASE 1: payload is already in viewer-contract form (like contract.json)
    # if isinstance(payload, dict) and "vertices" in payload and "faces" in payload:
    #     # You generated the contract on the TopologicPy side already
    #     return payload

    # CASE 2: payload is original TopologicPy JSON export (list of dicts)
    # Call your existing TopologicPy conversion here.
    # I'm using a placeholder function name; replace with your real one.
    topologies = Topology.ByJSONDictionary(jsonDictionary = payload, tolerance = 0.0001, silent = False)
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
