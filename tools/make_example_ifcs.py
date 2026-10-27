"""Generate the example IFC models shipped with the site.

The examples are built here from code rather than taken from elsewhere so the
site can publish them: they are original, CC0, and every dimension is known,
which also makes them useful as test models.

    python tools/make_example_ifcs.py [out_dir]

Writes into topologicpy-web-frontend/public/examples/ by default, plus an
index.json the app reads to list them.

Models:

* office-two-storey: 30 x 12 m, two storeys 3 m apart, a central corridor with
  five rooms either side, a stair at each end (in the north-west and
  north-east rooms) and an exit door at each end of the ground corridor.
* courtyard-ring: 16 x 16 m, one storey, a 2.4 m corridor running round a
  closed core, one exit. Every start has two ways out, one each way round.
  Built rotated 30 degrees off the world axes, so the sampling grid has to
  follow the building rather than the world.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import ifcopenshell
import ifcopenshell.api as api
import ifcopenshell.api.aggregate  # noqa: F401  (submodules register on import)
import ifcopenshell.api.context  # noqa: F401
import ifcopenshell.api.geometry  # noqa: F401
import ifcopenshell.api.project  # noqa: F401
import ifcopenshell.api.root  # noqa: F401
import ifcopenshell.api.spatial  # noqa: F401
import ifcopenshell.api.unit  # noqa: F401
from ifcopenshell.util.shape_builder import ShapeBuilder

SLAB = 0.2
WALL = 0.2
DOOR_W = 0.9
DOOR_H = 2.1


class Model:
    """A thin builder over ifcopenshell.api. Plan coordinates in metres."""

    def __init__(self, name: str, rotation_deg: float = 0.0):
        self.f = api.project.create_file(version="IFC4")
        self.project = api.root.create_entity(self.f, ifc_class="IfcProject", name=name)
        units = [
            api.unit.add_si_unit(self.f, unit_type=t)
            for t in ("LENGTHUNIT", "AREAUNIT", "VOLUMEUNIT")
        ]
        api.unit.assign_unit(self.f, units=units)
        model = api.context.add_context(self.f, context_type="Model")
        self.body = api.context.add_context(
            self.f, context_type="Model", context_identifier="Body",
            target_view="MODEL_VIEW", parent=model,
        )
        self.site = api.root.create_entity(self.f, ifc_class="IfcSite", name="Site")
        self.building = api.root.create_entity(self.f, ifc_class="IfcBuilding", name=name)
        api.aggregate.assign_object(self.f, products=[self.site], relating_object=self.project)
        api.aggregate.assign_object(self.f, products=[self.building], relating_object=self.site)
        self.theta = math.radians(rotation_deg)
        self.builder = ShapeBuilder(self.f)
        self.counts = {"walls": 0, "doors": 0, "slabs": 0, "stairs": 0}

    # -------------------------------------------------------------- placing

    def _matrix(self, x, y, z, angle=0.0):
        """Plan position and heading, turned by the model's rotation."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        gx, gy = c * x - s * y, s * x + c * y
        a = angle + self.theta
        m = np.eye(4)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = math.cos(a), -math.sin(a), math.sin(a), math.cos(a)
        m[:3, 3] = (gx, gy, z)
        return m

    def _place(self, product, storey, x, y, z, angle=0.0, contain=True):
        if contain:
            api.spatial.assign_container(self.f, products=[product], relating_structure=storey)
        api.geometry.edit_object_placement(self.f, product=product, matrix=self._matrix(x, y, z, angle))

    # ------------------------------------------------------------ elements

    def storey(self, name: str, elevation: float):
        st = api.root.create_entity(self.f, ifc_class="IfcBuildingStorey", name=name)
        st.Elevation = elevation
        api.aggregate.assign_object(self.f, products=[st], relating_object=self.building)
        api.geometry.edit_object_placement(self.f, product=st, matrix=self._matrix(0, 0, elevation))
        st.Elevation = elevation
        return st

    def slab(self, storey, x0, y0, x1, y1, top: float):
        """A rectangular floor slab whose top face is at `top`."""
        slab = api.root.create_entity(self.f, ifc_class="IfcSlab", predefined_type="FLOOR", name="Floor")
        rep = api.geometry.add_slab_representation(
            self.f, context=self.body, depth=SLAB,
            polyline=[(0, 0), (x1 - x0, 0), (x1 - x0, y1 - y0), (0, y1 - y0)],
        )
        api.geometry.assign_representation(self.f, product=slab, representation=rep)
        self._place(slab, storey, x0, y0, top - SLAB)
        self.counts["slabs"] += 1

    def floor_with_holes(self, storey, x0, y0, x1, y1, top, holes=()):
        """Cover a rectangle with slabs, leaving rectangular holes open.

        Cuts the plan into strips at every hole edge along x, and within each
        strip into pieces along y, keeping the pieces outside every hole.
        """
        xs = sorted({x0, x1, *[h[0] for h in holes], *[h[2] for h in holes]})
        for xa, xb in zip(xs, xs[1:]):
            mid = 0.5 * (xa + xb)
            cuts = sorted(
                {y0, y1, *[v for h in holes if h[0] <= mid <= h[2] for v in (h[1], h[3])]}
            )
            for ya, yb in zip(cuts, cuts[1:]):
                ym = 0.5 * (ya + yb)
                inside = any(h[0] <= mid <= h[2] and h[1] <= ym <= h[3] for h in holes)
                if not inside and xb - xa > 1e-6 and yb - ya > 1e-6:
                    self.slab(storey, xa, ya, xb, yb, top)

    def wall(self, storey, p0, p1, base: float, height: float):
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        if length < 0.05:
            return
        wall = api.root.create_entity(self.f, ifc_class="IfcWall", name="Wall")
        rep = api.geometry.add_wall_representation(
            self.f, context=self.body, length=length, height=height,
            thickness=WALL, offset=-WALL / 2,
        )
        api.geometry.assign_representation(self.f, product=wall, representation=rep)
        self._place(wall, storey, p0[0], p0[1], base, math.atan2(dy, dx))
        self.counts["walls"] += 1

    def wall_with_doors(self, storey, p0, p1, base, height, door_at=()):
        """A wall split around door openings; `door_at` are distances along it."""
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        cursor = 0.0
        for centre in sorted(door_at):
            a, b = centre - DOOR_W / 2, centre + DOOR_W / 2
            self.wall(storey, (p0[0] + ux * cursor, p0[1] + uy * cursor),
                      (p0[0] + ux * a, p0[1] + uy * a), base, height)
            self.door(storey, (p0[0] + ux * a, p0[1] + uy * a), math.atan2(dy, dx), base)
            cursor = b
        self.wall(storey, (p0[0] + ux * cursor, p0[1] + uy * cursor), p1, base, height)

    def door(self, storey, at, angle, base):
        door = api.root.create_entity(self.f, ifc_class="IfcDoor", name="Door")
        door.OverallWidth, door.OverallHeight = DOOR_W, DOOR_H
        rep = api.geometry.add_door_representation(
            self.f, context=self.body, overall_height=DOOR_H, overall_width=DOOR_W,
        )
        api.geometry.assign_representation(self.f, product=door, representation=rep)
        # Door geometry starts at its local origin; centre it in the wall's thickness.
        nx, ny = -math.sin(angle), math.cos(angle)
        self._place(door, storey, at[0] - nx * WALL / 2, at[1] - ny * WALL / 2, base, angle)
        self.counts["doors"] += 1

    def stair(self, storey, start, heading, width, rise, risers=16, going=0.28):
        """A straight flight of solid steps from `start`, climbing `rise`.

        Each step is a block whose top is a horizontal tread; the last tread is
        one riser below the floor it arrives at, as on a real stair.
        """
        riser = rise / risers
        items = []
        for i in range(1, risers):
            items.append(
                self.builder.extrude(
                    self.builder.rectangle(size=(going, width)),
                    magnitude=riser * i,
                    position=((i - 1) * going, -width / 2, 0.0),
                )
            )
        rep = self.builder.get_representation(self.body, items)
        stair = api.root.create_entity(self.f, ifc_class="IfcStair", predefined_type="STRAIGHT_RUN_STAIR", name="Stair")
        flight = api.root.create_entity(self.f, ifc_class="IfcStairFlight", name="Flight")
        flight.NumberOfRisers, flight.NumberOfTreads = risers, risers - 1
        flight.RiserHeight, flight.TreadLength = riser, going
        api.geometry.assign_representation(self.f, product=flight, representation=rep)
        api.spatial.assign_container(self.f, products=[stair], relating_structure=storey)
        api.aggregate.assign_object(self.f, products=[flight], relating_object=stair)
        api.geometry.edit_object_placement(
            self.f, product=stair, matrix=self._matrix(start[0], start[1], storey.Elevation, heading)
        )
        api.geometry.edit_object_placement(
            self.f, product=flight, matrix=self._matrix(start[0], start[1], storey.Elevation, heading)
        )
        self.counts["stairs"] += 1
        return (risers - 1) * going  # run length

    def write(self, path: Path):
        self.f.write(str(path))


# ------------------------------------------------------------------ models


def office_two_storey() -> Model:
    m = Model("Office, two storeys")
    length, depth, height, storey_h = 30.0, 12.0, 2.8, 3.0
    corridor = (5.0, 7.0)
    rooms = [(x, x + 6.0) for x in range(0, 30, 6)]
    # Each stair room is 6 m long: a 1 m landing, a 3.75 m flight (15 treads of
    # 0.25 m, risers of 0.1875 m), and a 1 m landing at the top. Without the
    # landings the flight started against a wall and could not be stepped onto.
    going = 0.25
    run = 15 * going
    stair_w = 1.2
    west = dict(start=(1.1, 11.0), heading=0.0, going=going)
    east = dict(start=(28.9, 11.0), heading=math.pi, going=going)
    holes = [(1.0, 10.2, 1.1 + run, 11.8), (28.9 - run, 10.2, 29.0, 11.8)]

    for level, name in enumerate(("Level 1", "Level 2")):
        top = level * storey_h
        st = m.storey(name, top)
        m.floor_with_holes(st, 0, 0, length, depth, top, holes if level == 1 else ())
        # Outer walls; the ground corridor ends are exits.
        ends = [corridor[0] + 1.0] if level == 0 else []
        m.wall_with_doors(st, (0, 0), (0, depth), top, height, door_at=ends)
        m.wall_with_doors(st, (length, 0), (length, depth), top, height, door_at=ends)
        m.wall(st, (0, 0), (length, 0), top, height)
        m.wall(st, (0, depth), (length, depth), top, height)
        # Corridor walls with a door into every room, and the partitions.
        doors = [0.5 * (a + b) for a, b in rooms]
        m.wall_with_doors(st, (0, corridor[0]), (length, corridor[0]), top, height, door_at=doors)
        m.wall_with_doors(st, (0, corridor[1]), (length, corridor[1]), top, height, door_at=doors)
        for _, b in rooms[:-1]:
            m.wall(st, (b, 0), (b, corridor[0]), top, height)
            m.wall(st, (b, corridor[1]), (b, depth), top, height)
        if level == 0:
            m.stair(st, rise=storey_h, width=stair_w, **west)
            m.stair(st, rise=storey_h, width=stair_w, **east)
    return m


def courtyard_ring() -> Model:
    m = Model("Courtyard ring", rotation_deg=30.0)
    size, ring, height = 16.0, 2.4, 2.8
    st = m.storey("Level 1", 0.0)
    a, b = ring, size - ring
    # The corridor is the ring; the core is closed (no doors), so it is not floor.
    m.floor_with_holes(st, 0, 0, size, size, 0.0, holes=[(a, a, b, b)])
    m.wall(st, (0, 0), (size, 0), 0.0, height)
    m.wall(st, (0, size), (size, size), 0.0, height)
    m.wall(st, (0, 0), (0, size), 0.0, height)
    m.wall_with_doors(st, (size, 0), (size, size), 0.0, height, door_at=[size / 2])
    for p0, p1 in (((a, a), (b, a)), ((b, a), (b, b)), ((b, b), (a, b)), ((a, b), (a, a))):
        m.wall(st, p0, p1, 0.0, height)
    return m


MODELS = {
    "office-two-storey": (
        office_two_storey,
        "Two storeys, a corridor with ten rooms per floor, a stair and an exit at each end.",
    ),
    "courtyard-ring": (
        courtyard_ring,
        "A corridor round a closed core with one exit: every start has two ways out. Rotated 30 degrees.",
    ),
}


# Third-party models shipped alongside ours, by the user's decision
# (2026-10-27), with their sources credited. Listed only if the file is present.
EXTERNAL = [
    {
        "id": "duplex-apartment",
        "name": "Duplex apartment",
        "file": "duplex-apartment.ifc",
        "description": "Two mirror-image dwellings over two storeys, each with its own stair and front door.",
        "credit": "Duplex Apartment, Common Building Information Model Files, National Institute of "
        "Building Sciences (NIBS) buildingSMART alliance and U.S. Army ERDC, 2011.",
        "source": "https://wbdg.org/node/9165",
        "licence": "Publisher states no licence; shown with credit.",
    },
    {
        "id": "buildingsmart-sample-scene",
        "name": "buildingSMART sample scene",
        "file": "buildingsmart-sample-scene.ifc",
        "description": "buildingSMART's small IFC4 sample building: three slabs, four walls, two spaces.",
        "credit": "\"ifc silly sample scene\", buildingSMART International, Sample-Test-Files (PCERT sample scene).",
        "source": "https://github.com/buildingSMART/Sample-Test-Files",
        "licence": "No licence file in the repository; shown with credit.",
    },
]


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for entry in EXTERNAL:
        path = out_dir / entry["file"]
        if path.exists():
            index.append({**entry, "bytes": path.stat().st_size})
    for slug, (make, description) in MODELS.items():
        model = make()
        path = out_dir / f"{slug}.ifc"
        model.write(path)
        index.append(
            {
                "id": slug,
                "name": model.project.Name,
                "file": path.name,
                "description": description,
                "bytes": path.stat().st_size,
                "credit": "Generated for Topologic Studio by tools/make_example_ifcs.py.",
                "licence": "CC0-1.0",
                **model.counts,
            }
        )
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KiB {model.counts}")
    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    default = Path(__file__).resolve().parents[1] / "topologicpy-web-frontend" / "public" / "examples"
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
