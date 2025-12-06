# topologicpy-web-backend/app/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Vertex(BaseModel):
    id: str
    x: float
    y: float
    z: float
    data: Optional[Dict[str, Any]] = None


class Face(BaseModel):
    id: str
    vertex_ids: List[str]
    data: Optional[Dict[str, Any]] = None


class TopologyPayload(BaseModel):
    vertices: List[Vertex]
    faces: List[Face]
