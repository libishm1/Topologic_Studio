# topologicpy-web-backend/app/utils.py
from .schemas import TopologyPayload


def summarize_topology(topology: TopologyPayload) -> dict:
    """Return simple metadata about the uploaded topology."""
    num_vertices = len(topology.vertices)
    num_faces = len(topology.faces)

    face_ids = [f.id for f in topology.faces]

    return {
        "num_vertices": num_vertices,
        "num_faces": num_faces,
        "face_ids": face_ids,
    }
