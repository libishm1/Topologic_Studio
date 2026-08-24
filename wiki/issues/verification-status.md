# Verification Status

Last checked: 2026-08-24 (Next line, browser verified). Sections below dated 2026-05-06 describe the Classic line.

## Required Browser Evidence For Future Runs

Any future browser verification must record:

- headless or headful mode
- tested URL
- browser console summary
- page errors
- failed requests
- API statuses
- screenshot paths
- human review decision if visual or research-validity judgment was needed

Use [headless browser console testing](../testing/headless-browser-console-testing.md).

## Backend Import And Routes

Command:

```powershell
cd TopologicStudio\topologicpy-web-backend
.\.venv\Scripts\python.exe -c "from app.main import app; print(app.title, app.version); [print(r.path, sorted(list(getattr(r,'methods',[])))) for r in app.routes if hasattr(r,'methods')]"
```

Result:

- Import succeeded.
- App title/version: `TopologicPy Web Backend 0.1.0`.
- Expected routes are registered.

## Backend Versions

Command:

```powershell
.\.venv\Scripts\python.exe -c "import fastapi, uvicorn, pydantic, topologicpy; ..."
```

Result:

- FastAPI `0.124.0`
- Uvicorn `0.38.0`
- Pydantic `2.12.5`
- TopologicPy `0.8.93`

## Frontend Lint

Command:

```powershell
cd TopologicStudio\topologicpy-web-frontend
$env:PATH=(Resolve-Path ..\node-v24.11.1-win-x64).Path + ';' + $env:PATH
npm.cmd run lint
```

Result:

- Failed with 9 errors and 4 warnings.
- See [open risks](open-risks.md).

## Frontend Build

Command:

```powershell
npm.cmd run build -- --outDir ../wiki/verification/frontend-dist --emptyOutDir
```

Result:

- Build succeeded.
- 89 modules transformed.
- Output written to `wiki/verification/frontend-dist/`.
- Large chunk warning remains.

## PDF Parse

Command:

```powershell
python -c "import fitz; ..."
```

Result:

- `the synergy of non manifold topology.pdf` has 10 pages and 0 extractable text characters.
- OCR required.


---

# Next Line — Browser Verification, 2026-08-24

Branch `next/studio-2`. Full session log:
[browser-session-2026-08-24.md](browser-session-2026-08-24.md).

Required evidence, per the checklist above:

- **Mode**: headless Chrome 151, installed browser driven by playwright-core
  (`tools/browser-test.mjs`). `--headed` available.
- **URL**: `http://127.0.0.1:5173`, backend `http://127.0.0.1:8000`.
- **Console summary**: 0 errors, 0 warnings after fixes (4 bugs found and fixed
  during the session).
- **Page errors**: 0.
- **Failed requests**: 0.
- **API statuses**: all 2xx. `GET /api/capabilities` reports
  `topologicpy 0.9.64`, `topologic_core installed`, `usable: true`,
  engines `["fast", "topologicpy"]`.
- **Screenshots**: `tools/shots/` (gitignored, regenerate with
  `npm run test:browser`).
- **Human review**: two decisions outstanding, see below.

## Commands

```powershell
cd TopologicStudio-Next	opologicpy-web-backend
..\.venv-next\Scripts\python.exe -m uvicorn app.main:app --port 8000

cd TopologicStudio-Next	opologicpy-web-frontend
npm.cmd run dev

cd TopologicStudio-Next
npm.cmd run test:browser
```

## Results

- Browser harness: 21 / 21 checks pass.
- Backend unit tests: 73 / 73 pass.
- Pipeline smoke test (`npm run test:pipeline`): all checks pass.
- Frontend lint: clean (was 9 errors / 4 warnings on Classic).
- Frontend build: clean. First paint is 83 KB gzipped; Three and That Open are
  lazy chunks.

## Versions

- FastAPI 0.141.1, Uvicorn 0.52.4, Pydantic 2.13.4
- TopologicPy 0.9.64 + topologic-core 8.0.4, IfcOpenShell 0.8.5
- @thatopen/components 3.4.8, @thatopen/fragments 3.4.7, web-ifc 0.0.77,
  three 0.185.1

## Human decisions still needed

1. **Door waypoint modelling.** One waypoint per door opening is now produced
   (was one per mesh). Whether that is right for egress capacity analysis is a
   research call.
2. **Silent pick snapping.** Picks snap to the nearest graph node with no
   warning when the click lands far from walkable space.
