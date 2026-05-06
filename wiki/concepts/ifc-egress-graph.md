# IFC Egress Graph Implementation

This compatibility page has been split into implementation slices.

Use these pages instead:

- [Backend IFC egress graph](../backend/ifc-egress-graph.md)
- [IFC egress API](../api/ifc-egress.md)
- [Frontend IFC viewer](../frontend/ifc-viewer.md)
- [Frontend app orchestration](../frontend/app-orchestration.md)
- [Open risks](../issues/open-risks.md)

The key implementation path is:

```text
IFCViewer.jsx -> App.jsx -> POST /ifc-egress-graph -> LAST_GRAPHS["ifc"] -> POST /ifc-egress-path
```

