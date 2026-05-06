# Headless Browser Console Testing

Use this page when an agent needs to test the frontend autonomously and read the browser console.

## Goal

Run a browser against the local Vite app, capture console output, page errors, failed requests, and screenshots, then update known-bug documentation.

## Preconditions

- Backend running on `http://localhost:8000`.
- Frontend running on `http://localhost:5173`.
- If global Node/npm is unavailable, prepend the bundled Node path.

```powershell
cd "C:\Users\lmurugesan\OneDrive - Alfaisal University\CM-iTAD\topologic_webapp\TopologicStudio\topologicpy-web-frontend"
$env:PATH=(Resolve-Path ..\node-v24.11.1-win-x64).Path + ';' + $env:PATH
npm.cmd run dev -- --host 0.0.0.0 --port 5173
```

## Preferred Agent Workflow

1. Start or confirm backend.
2. Start or confirm frontend.
3. Open the page with Playwright or equivalent browser automation.
4. Capture:
   - `console` messages
   - `pageerror`
   - failed `request`
   - response status for API calls
   - screenshot after each major step
5. Execute one focused workflow.
6. Write findings into `wiki/issues/`.
7. Ask a human only for visual or domain-judgment decisions.

## Playwright Console Harness

Use [playwright-console-harness.md](playwright-console-harness.md) for the one-off script pattern. Keep this workflow page loaded for the process and load the harness only when writing or running browser automation.

## IFC Workflow Test

For `Ifc2x3_Duplex_Architecture.ifc`:

1. Load page.
2. Upload the IFC through the `Load IFC` input.
3. Wait for viewer ready or processing badge to clear.
4. Capture console and screenshot.
5. Trigger egress graph build if the UI control is available.
6. Capture API response status and graph stats.
7. Pick start and exit only if stable selectors or raycast coordinates are available.
8. If path appears, capture screenshot and response.

## Human Checkpoints

Ask the human to review:

- Is the IFC upright and scaled correctly?
- Does the graph overlay cover walkable floors and stairs without obvious floating noise?
- Does the selected route make spatial sense?
- Does fire visualization make the graph easier to understand?
- Is an observed console warning acceptable or should it become a tracked bug?

## Failure Output Format

When a test fails, record:

```text
Workflow:
Environment:
Exact command:
Console errors:
Failed requests:
Screenshot path:
Expected:
Actual:
Suspected files:
Next action:
Human decision needed:
```
