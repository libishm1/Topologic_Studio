# Updating GitHub Pages — options and consequences

Date: 2026-09-21
Status: **decision pending, nothing public has been changed**
Asked for by: Libish

---

## What is live right now

| Piece | Where | What it actually is |
| --- | --- | --- |
| Public site | https://libishm1.github.io/Topologic_Studio/ | **Classic** frontend. Page title is still `topologicpy-web-frontend`. |
| Public API | https://topologicstudio-backend.onrender.com | **Classic** backend. `/health` → 200, `/api/capabilities` → 404. |
| Build trigger | `.github/workflows/deploy-frontend.yml` | Fires on push to **`main`**, path `topologicpy-web-frontend/**`. |
| Build config | same workflow | `BASE_PATH=/Topologic_Studio/`, `VITE_API_BASE=https://topologicstudio-backend.onrender.com`. |

Render's free tier spins down when idle. The first request after a quiet period
can fail outright before the service wakes; the second succeeds. That is worth
knowing before reading anything into a single failed health check.

## Why this is not a one-line change

Three facts make "update the Pages site" bigger than it sounds.

**1. Pages builds from `main`, and the two lines have diverged.**

```
next/studio-2  is 10 commits ahead of main
main           is 18 commits ahead of next/studio-2
```

A `git merge-tree` dry run reports a conflict. `main` carries work that
`claude/ifc-port-baseline` (the base of the Next line) never had — the Codex
wiki/HITL documentation merge among it. So merging is a real review, not a
fast-forward.

**2. The Next frontend cannot run against the Classic backend.**

Next calls `/api/ifc/graph`, `/api/ifc/path`, `/api/fire/stream`,
`/api/capabilities`. Classic serves none of them. Publishing the Next frontend
against the current `VITE_API_BASE` gives a site that loads, renders its shell,
and then 404s on every action. The frontend and backend have to move together.

**3. The backends are deliberately separate services.**

| | Classic | Next |
| --- | --- | --- |
| Render service | `topologicstudio-backend` | `topologicstudio-next-backend` |
| `autoDeploy` | `true` | `false` |

Next was given its own blueprint precisely so the two never share a process,
an environment or a rollback. Deploying Next means **standing up a second
Render service**, not replacing the first. That is a new (possibly billable)
service on your account, and only you can create it.

---

## The options

### A. Separate preview deployment — lowest risk

Publish the Next line to its own URL and leave the public site exactly as it is.

- New workflow on `next/studio-2` deploying to a second Pages target (a
  `gh-pages` branch, or a separate repo).
- Stand up `topologicstudio-next-backend` on Render from the existing
  blueprint; point the preview build's `VITE_API_BASE` at it.
- Add that preview origin to the Next backend's `CORS_ORIGINS`.

**Cost:** a second Render service. **Risk to the live site:** none.
**Gives you:** somewhere to show and test the Next line publicly before
committing to it.

### B. Merge Next into `main` and make it the public site

- Resolve the merge against the 18 commits on `main`.
- Stand up the Next backend on Render **in the same change**, and repoint
  `VITE_API_BASE`.
- Re-run the full suite against the deployed pair, not just locally.

**Cost:** the merge review, plus a second Render service.
**Risk:** this is the merge decision deliberately deferred in August. If it goes
wrong the public demo is broken until it is reverted. It should not happen the
day before you need to show anything.

### C. Only bump Classic to the latest TopologicPy

Leave the architecture alone. Update `topologicpy` on `main` and let the
existing workflow redeploy.

**Careful — this is not a version bump.** Classic pins `topologicpy` (currently
resolving to 0.8.93) and 0.9 split the native backend into a separate
`topologic-core` distribution. Classic's `requirements.txt` does not list it,
so a naive bump produces a service that imports fine and then fails on the
first geometry call. Classic also calls the legacy `Graph` API throughout,
which still exists in 0.9.71 but is the slow path.

Doable, but it is a change to the **demo-safe** line, which the whole two-line
split exists to protect.

### D. Change nothing public yet

The Next line is pushed and reproducible locally via `studio`. Revisit when
there is a reason to publish.

---

## Recommendation

**A, and not yet B.** The Next line has never been used by a human — every
interaction so far has been scripted — and it has only ever been run against
one IFC model on one machine. A preview URL gets it in front of people without
putting the public demo behind an untested merge.

Before B, the open items from
[handoff-2026-08-24.md](handoff-2026-08-24.md) still stand: a human driving the
app, a second and third model, and CI.

## What I have not done

No workflow was added or edited, nothing was merged, no Render service was
created, and the live site and API are untouched. The only thing pushed is the
TGraph engine work on `next/studio-2`.
