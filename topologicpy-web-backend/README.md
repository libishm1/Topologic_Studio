# Topologic Studio: the optional server

A FastAPI service around TopologicPy. The app works without it; it adds the
TopologicPy `TGraph` route engine, the Navigation cross-check and the
Topology JSON viewer. See the [main README](../README.md) for what runs where
and for the API table.

Needs Python 3.12. From the repository root:

```bash
python -m venv .venv-next
.venv-next/Scripts/python -m pip install -r topologicpy-web-backend/requirements-dev.txt   # bin/python on macOS and Linux

cd topologicpy-web-backend
../.venv-next/Scripts/python -m uvicorn app.main:app --reload --port 8000
../.venv-next/Scripts/python -m pytest tests -q
```

<http://localhost:8000/api/capabilities> should report `"usable": true`, and
<http://localhost:8000/docs> lists every route. If you run it on another port,
set `VITE_API_BASE` in `topologicpy-web-frontend/.env`.

`Dockerfile` is the image Render runs (`render.yaml` at the repository root).
Allowed browser origins are set with `CORS_ORIGINS`.
