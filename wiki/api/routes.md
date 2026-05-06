# API Routes

Base URL is controlled by the frontend `VITE_API_BASE` environment variable. Local default in `src/App.jsx` is `http://localhost:8000`.

| Route | Method | Requires prior state | Returns |
|---|---|---|---|
| `/health` | GET | No | `{ "status": "ok" }` |
| `/upload-topology` | POST | No | Viewer contract |
| `/upload-ifc` | POST multipart | No | Viewer contract |
| `/ifc-egress-graph` | POST JSON | Browser-extracted IFC geometry | IFC graph contract and stores `LAST_GRAPHS.ifc` |
| `/ifc-egress-path` | POST JSON | `/ifc-egress-graph` | Coordinate path |
| `/fire-sim` | POST JSON | Any active graph for selected mode | Precomputed timeline |
| `/fire-sim/stream` | GET query | Any active graph for selected mode | SSE stream |
| `/rl/train` | POST JSON | Any active graph for selected mode | Node ID path |
| `/graph-meta` | GET query | Active graph for selected mode | Cell metadata |

Route details are split by family:

- [IFC egress](ifc-egress.md)
- [Fire and RL](fire-rl.md)
- [Upload contracts](upload-contracts.md)

