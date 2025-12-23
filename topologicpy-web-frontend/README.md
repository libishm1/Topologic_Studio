# React + Vite

## Local development (Windows)

Start the frontend with the bundled Node runtime (no admin install needed):

```powershell
cd "C:/Users/lmurugesan/OneDrive - Alfaisal University/CM-iTAD/topologic_webapp/TopologicStudio/topologicpy-web-frontend"
$nodeDir = "C:/Users/lmurugesan/OneDrive - Alfaisal University/CM-iTAD/topologic_webapp/TopologicStudio/node-v24.11.1-win-x64"
$env:Path = "$nodeDir;$env:Path"
& "$nodeDir\npm.cmd" run dev -- --host 0.0.0.0 --port 5173
```

If the backend is running on a different port (not 8000), update `API_BASE` in `src/App.jsx`.

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend using TypeScript with type-aware lint rules enabled. Check out the [TS template](https://github.com/vitejs/vite/tree/main/packages/create-vite/template-react-ts) for information on how to integrate TypeScript and [`typescript-eslint`](https://typescript-eslint.io) in your project.
