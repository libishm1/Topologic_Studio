# Topologic Studio: the app

The browser app: React and Vite, the That Open engine for the viewer, and
web-ifc for reading IFC. What it does, with screenshots, is in the
[main README](../README.md).

Needs Node.js 20.11 or later.

```bash
npm ci                     # also copies web-ifc's WASM and the fragments worker into public/
npm run dev                # http://localhost:5173
npm test                   # unit and property tests of src/core
npm run lint
npm run build              # converts the examples, then builds dist/
npm run prepare-examples   # converts the examples only, for npm run dev
```

The app looks for the optional server at `http://localhost:8000`. To use
another address, copy `.env.example` to `.env` and set `VITE_API_BASE`. For a
build served from a sub-path, as on GitHub Pages, set `BASE_PATH` (for
example `BASE_PATH=/Topologic_Studio/`).

| Folder | What is in it |
| --- | --- |
| `src/core/` | Routing, the walkable graph, stairs and cores, fire models, forecast and deadline map, Q-learning, IFC semantics; each with its tests |
| `src/workers/` | The IFC import, graph, forecast and RL workers |
| `src/viewer/` | `ViewerManager`, which owns the 3D scene |
| `src/ui/` | The four step panels, the viewport overlays and the Methods page |
| `src/lib/` | The server client and the IndexedDB model cache |
| `public/examples/` | The example IFCs and their credits (`index.json`) |
| `scripts/` | Asset sync after install, and the build-time example conversion |
