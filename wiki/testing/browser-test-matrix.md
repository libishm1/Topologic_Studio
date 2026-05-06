# Browser Test Matrix

Use this as the compact matrix for autonomous plus human-in-the-loop testing.

| Workflow | Autonomous checks | Browser evidence | Human checkpoint | Bug docs |
|---|---|---|---|---|
| App boot | Page loads, no fatal console/page errors | console, pageerror, screenshot | Visual layout is acceptable | `verification-status.md` or `open-risks.md` |
| Backend health | `/health` returns 200 | response status | None unless public backend differs | `verification-status.md` |
| IFC load | IFC file accepted, viewer not blank | console, screenshot | Model orientation and scale | `open-risks.md` |
| IFC graph | `/ifc-egress-graph` succeeds, stats nonzero | response, screenshot | Graph density/plausibility | `known-bugs-fixed.md` or `open-risks.md` |
| IFC path | `/ifc-egress-path` returns at least two points | response, screenshot | Route plausibility | issue page plus template |
| Fire stream | SSE emits expected event types | console, network, event log | Fire colors communicate risk | `verification-status.md` |
| Dynamic path | `path_update` events appear | event log, screenshot | Reroute makes spatial sense | `open-risks.md` |
| RL train | `/rl/train` returns path or clear error | response | Path usefulness | issue page |
| Deployment preview | Built app loads with correct base path | console, request status | Public URL works | deployment issue/risk |

