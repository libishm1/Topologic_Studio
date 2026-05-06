# Deployment And GitHub Pages

## Deployment Model

The deployment is split:

- Frontend: static Vite build deployed to GitHub Pages.
- Backend: FastAPI/Uvicorn service deployed to Render with Docker.

The frontend must know the backend URL at build time via `VITE_API_BASE`. The backend must allow the frontend origin through `CORS_ORIGINS`.

## Vite Configuration

Current `vite.config.js`:

```js
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || '/',
})
```

Context7 Vite notes:

- Vite exposes client-side env variables through `import.meta.env`.
- By default, only variables prefixed with `VITE_` are exposed to client code.
- For GitHub Pages under `https://<username>.github.io/<repo>/`, Vite `base` should be `'/<repo>/'`.
- `vite build` creates production output and `vite preview` can locally preview it.

Local adoption:

- GitHub Actions sets `BASE_PATH: '/${{ github.event.repository.name }}/'`.
- `App.jsx` reads `import.meta.env.VITE_API_BASE || "http://localhost:8000"`.
- Workflow sets `VITE_API_BASE` from repository Actions variable or falls back to `https://topologicstudio-backend.onrender.com`.

## GitHub Actions Workflow

Workflow: `.github/workflows/deploy-frontend.yml`

Trigger:

- push to `main`
- frontend files or workflow changed

Steps:

1. Checkout.
2. Setup Node 20.
3. `npm ci`.
4. Build with `BASE_PATH` and `VITE_API_BASE`.
5. Upload Pages artifact from `topologicpy-web-frontend/dist`.
6. Deploy through `actions/deploy-pages@v4`.

Current improvement: `VITE_API_BASE` is no longer hardcoded only to Render. It can be set through GitHub repository variable `VITE_API_BASE`.

## Render Backend

`render.yaml` defines:

- type: web
- runtime: docker
- rootDir: `topologicpy-web-backend`
- healthCheckPath: `/health`
- `CORS_ORIGINS=https://libishm1.github.io`

Backend `Dockerfile`:

- base: `python:3.11-slim`
- installs `requirements.txt`
- additionally installs `ifcopenshell topologicpy`
- runs `uvicorn app.main:app --host 0.0.0.0 --port 8000`

## Deployment Checklist

1. Backend Render service builds successfully.
2. Backend `/health` returns `{"status":"ok"}`.
3. Render `CORS_ORIGINS` includes the exact GitHub Pages origin.
4. GitHub repository variable `VITE_API_BASE` points to Render backend.
5. GitHub Pages Source is set to GitHub Actions.
6. Vite `base` is set to `/<repo>/` during CI.
7. Browser console confirms API calls go to Render, not localhost.
8. IFC upload and egress graph work in production.

## Current Deployment Risks

- Frontend build output includes a very large JS chunk around 5.2 MB minified, mostly from 3D/IFC dependencies. This can slow first load on GitHub Pages.
- Lint fails, though build succeeds.
- Backend graph state is process-local. Render restarts or multiple instances will lose graph state.
- If Render cold-starts, frontend API calls may appear stalled.
- `VITE_WEBIFC_WASM_PATH` defaults to CDN. If CDN access is blocked, IFC loading fails unless WASM assets are hosted with the app.

