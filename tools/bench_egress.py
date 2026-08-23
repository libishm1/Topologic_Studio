"""Benchmark the IFC egress graph pipeline on a real IFC model.

Extracts the same floor/stair/door/wall triangle payload the frontend sends,
then times the graph build. Point it at either backend:

    ../.venv-next/Scripts/python.exe tools/bench_egress.py path/to/model.ifc
    ../.venv-next/Scripts/python.exe tools/bench_egress.py model.ifc --classic <dir>

The payload is cached next to the model so both runs measure the same input.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SLAB_TYPES = ("IfcSlab", "IfcSlabStandardCase", "IfcSlabElementedCase", "IfcCovering")
STAIR_TYPES = ("IfcStair", "IfcStairFlight")
DOOR_TYPES = ("IfcDoor",)
WALL_TYPES = ("IfcWall", "IfcWallStandardCase")


def extract_payload(ifc_path: Path, cache: Path | None = None) -> dict:
    if cache and cache.exists():
        print(f"[bench] reusing cached payload {cache.name}")
        return json.loads(cache.read_text())

    import ifcopenshell
    import ifcopenshell.geom

    print(f"[bench] parsing {ifc_path.name} with ifcopenshell ...")
    t0 = time.perf_counter()
    model = ifcopenshell.open(str(ifc_path))
    print(f"[bench]   open: {time.perf_counter() - t0:.2f}s  schema={model.schema}")

    settings = ifcopenshell.geom.settings()
    try:
        settings.set(settings.USE_WORLD_COORDS, True)
    except Exception:
        # ifcopenshell 0.8 renamed the setting API.
        try:
            settings.set("use-world-coords", True)
        except Exception:
            pass

    def collect(types):
        out = []
        for type_name in types:
            try:
                elements = model.by_type(type_name)
            except RuntimeError:
                # Not every entity exists in every schema (IFC2X3 has no
                # IfcSlabStandardCase, for instance).
                continue
            for element in elements:
                try:
                    shape = ifcopenshell.geom.create_shape(settings, element)
                except Exception:
                    continue
                geometry = shape.geometry
                verts = list(geometry.verts)
                faces = list(geometry.faces)
                if len(verts) < 9 or len(faces) < 3:
                    continue
                out.append({"expressID": element.id(), "vertices": verts, "indices": faces})
        return out

    t0 = time.perf_counter()
    payload = {
        "floors": collect(SLAB_TYPES),
        "stairs": collect(STAIR_TYPES),
        "doors": collect(DOOR_TYPES),
        "walls": collect(WALL_TYPES),
    }
    print(f"[bench]   tessellate: {time.perf_counter() - t0:.2f}s")

    for key, items in payload.items():
        tris = sum(len(g["indices"]) // 3 for g in items)
        print(f"[bench]   {key:7}: {len(items):4} elements, {tris:7} triangles")

    if cache:
        cache.write_text(json.dumps(payload))
        size = cache.stat().st_size / 1e6
        print(f"[bench]   cached payload: {size:.1f} MB of JSON")
    return payload


def run_backend(app_dir: Path, payload: dict, label: str, repeats: int = 3) -> None:
    sys.path.insert(0, str(app_dir))
    for name in list(sys.modules):
        if name == "app" or name.startswith("app."):
            del sys.modules[name]

    from fastapi.testclient import TestClient

    from app.main import app  # noqa: E402

    body = {
        **payload,
        "up_axis": "z",
        "base_spacing": 0.5,
        "max_edge_floor": 2.25,
        "max_edge_stair": 0.4,
        "agent_height": 0.75,
        "use_walls": True,
        "max_points": 20000,
    }
    raw = json.dumps(body)
    print(f"\n[{label}] request payload: {len(raw) / 1e6:.1f} MB")

    with TestClient(app) as client:
        times = []
        result = None
        for i in range(repeats):
            t0 = time.perf_counter()
            response = client.post("/ifc-egress-graph", json=body)
            dt = time.perf_counter() - t0
            if response.status_code != 200:
                print(f"[{label}] FAILED {response.status_code}: {response.text[:300]}")
                return
            times.append(dt)
            result = response.json()
            print(f"[{label}]   run {i + 1}: {dt:.3f}s")

        stats = result.get("stats", {})
        print(
            f"[{label}] median {statistics.median(times):.3f}s  "
            f"nodes={stats.get('nodes')} edges={stats.get('edges')}"
        )
        print(f"[{label}] response size: {len(response.content) / 1e6:.1f} MB")

        # Time a path query on the resulting graph.
        coords = result.get("coords") or {}
        if len(coords) >= 2:
            keys = list(coords)
            start, end = coords[keys[0]], coords[keys[-1]]
            path_times = []
            for _ in range(5):
                t0 = time.perf_counter()
                pr = client.post(
                    "/ifc-egress-path", json={"start_point": start, "end_point": end}
                )
                path_times.append(time.perf_counter() - t0)
            found = pr.status_code == 200
            print(
                f"[{label}] path query median {statistics.median(path_times) * 1000:.1f}ms "
                f"(found={found})"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ifc", type=Path)
    parser.add_argument("--next", dest="next_dir", type=Path, default=None)
    parser.add_argument("--classic", dest="classic_dir", type=Path, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache", type=Path, default=None)
    args = parser.parse_args()

    payload = extract_payload(args.ifc, args.cache)

    if args.classic_dir:
        run_backend(args.classic_dir, payload, "classic", args.repeats)
    target = args.next_dir or Path(__file__).resolve().parents[1] / "topologicpy-web-backend"
    run_backend(target, payload, "next", args.repeats)


if __name__ == "__main__":
    main()
