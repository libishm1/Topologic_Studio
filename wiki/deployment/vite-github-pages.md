# Vite And GitHub Pages

Frontend deployment is defined in `.github/workflows/deploy-frontend.yml` and `topologicpy-web-frontend/vite.config.js`.

## Current Vite Config

```js
export default defineConfig({
  plugins: [react()],
  base: process.env.BASE_PATH || '/',
})
```

The workflow sets:

- `BASE_PATH: '/${{ github.event.repository.name }}/'`
- `VITE_API_BASE: ${{ vars.VITE_API_BASE || 'https://topologicstudio-backend.onrender.com' }}`

This is the correct idea for GitHub Pages under a repository subpath. Context7 Vite docs confirm that `base` controls rewritten asset paths for nested deployments and that `import.meta.env` is the runtime-facing environment mechanism.

## Workflow Shape

The current workflow:

1. Runs on pushes to `main` affecting frontend files or the workflow.
2. Uses Node `20`.
3. Runs `npm ci`.
4. Runs `npm run build`.
5. Uploads `topologicpy-web-frontend/dist` through GitHub Pages artifact.
6. Deploys with `actions/deploy-pages@v4`.

## Local Build Command

For verification without touching app output, build into the wiki:

```powershell
cd "TopologicStudio/topologicpy-web-frontend"
$env:PATH=(Resolve-Path ..\node-v24.11.1-win-x64).Path + ';' + $env:PATH
npm.cmd run build -- --outDir ../wiki/verification/frontend-dist --emptyOutDir
```

Latest run succeeded. See [verification](../verification/README.md).

## Current Deployment Risks

- ESLint reports `process` as undefined in `vite.config.js`. Build still succeeds, but lint fails.
- The production JS bundle is over 5 MB before gzip, mainly due to IFC/3D libraries and worker payloads.
- If the repo name changes, GitHub Pages base path changes automatically because `BASE_PATH` uses the repository name.
- If `VITE_API_BASE` is not configured in repository variables, the workflow falls back to the Render backend URL.

## References

- Vite build docs: https://github.com/vitejs/vite/blob/main/docs/guide/build.md
- Vite static deploy docs: https://github.com/vitejs/vite/blob/main/docs/guide/static-deploy.md

