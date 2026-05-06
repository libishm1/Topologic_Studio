# IFC Lite Profile

The term "IFC Lite" should be treated as an internal lightweight profile unless a specific external standard or library is selected later. The current official IFC standard tracked by buildingSMART is IFC `4.3.2.0`; IFC `5` is under development.

## Why A Lite Profile Helps

The current egress workflow does not need full IFC semantics. It needs a reliable subset:

- Walkable horizontal or stair surfaces.
- Vertical circulation.
- Door/opening positions.
- Wall/obstacle segments.
- Storey or level hints.
- Coordinate transform and units.

An internal lite profile can make the graph builder faster, testable, and easier to port.

## Proposed Minimal Contract

```json
{
  "units": "m",
  "up_axis": "y",
  "floors": [],
  "stairs": [],
  "doors": [],
  "walls": [],
  "levels": [],
  "transform": null,
  "source_ifc_schema": "IFC4X3_ADD2"
}
```

Each geometry item should include:

- `expressID`
- `type`
- `vertices`
- `indices`
- `normals`
- optional `bbox`
- optional `level_id`

## Integration Path

1. Introduce a conversion function in `IFCViewer.jsx` that emits the lite profile before posting.
2. Add backend `IfcLiteProfile` Pydantic models.
3. Make `/ifc-egress-graph` accept either current arrays or the profile.
4. Add a validation-only endpoint before graph generation.
5. Cache extracted lite profiles or fragments so reloads do not reparse IFC every time.

## Compatibility Notes

- buildingSMART documents IFC as schema plus documentation, property/quantity sets, and exchange mechanisms.
- The recommended exchange file for IFC2x3, IFC4, and IFC4.3 remains `.ifc` STEP Physical File Format.
- The app should not claim compliance for a lite profile unless it validates against official IFC requirements or a defined MVD.

