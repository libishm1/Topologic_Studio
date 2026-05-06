# Render Backend

Backend deployment is described by root `render.yaml` and `topologicpy-web-backend/Dockerfile`.

## `render.yaml`

```yaml
services:
  - type: web
    name: topologicstudio-backend
    runtime: docker
    rootDir: topologicpy-web-backend
    autoDeploy: true
    healthCheckPath: /health
    envVars:
      - key: CORS_ORIGINS
        value: https://libishm1.github.io
```

## Dockerfile

The backend Dockerfile:

- Starts from `python:3.11-slim`.
- Installs `requirements.txt`.
- Installs `ifcopenshell topologicpy` explicitly after requirements.
- Copies `app/`.
- Starts Uvicorn on `0.0.0.0:8000`.

## Important Notes

- `requirements.txt` lists `topologicpy` but not `ifcopenshell`; Docker adds `ifcopenshell` explicitly.
- If local and deployed behavior diverge for `/upload-ifc`, check whether `ifcopenshell` is installed in the local venv.
- The health check path is `/health`, which currently imports and returns successfully.
- CORS is narrow in Render: currently only `https://libishm1.github.io` is added through env vars. Add the full GitHub Pages origin if it changes.

## Smoke Test

Local import route listing succeeded:

```text
/health GET
/ifc-egress-graph POST
/ifc-egress-path POST
/fire-sim POST
/fire-sim/stream GET
/rl/train POST
/graph-meta GET
/upload-topology POST
/upload-ifc POST
```

