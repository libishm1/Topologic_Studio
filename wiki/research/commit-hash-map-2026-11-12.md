# Commit hashes before and after the 2026-11-12 history rewrite

Before this branch was first pushed, its local history (every commit after `ae8191b`, the parent of the first rewritten commit, on `origin/main` at `ea185d4`) was rewritten to remove a collaborator's name from two planning notes, at the user's request: unpublished names stay out of the repository. Only three lines changed, in `wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md` and `wiki/HANDOFF.md`; every commit keeps its message, author and dates. Every hash changed, so a hash cited in a file or commit message written before the rewrite (the claim protocol's addenda, the validation plan, the result files' `git.head`, commit messages) is the old one: this table maps it to the commit as published.

| Before | After | Committed (+0300) | Subject |
| --- | --- | --- | --- |
| `ae8191b` | `ae8191b` | 2026-11-11 11:43 | Handoff: released ea185d4 (describer facts, atria), live check passed |
| `0346857` | `b99c597` | 2026-11-11 15:33 | Plan: evacuation distance map and wheelchair accessibility audit |
| `f7d631d` | `1c3911a` | 2026-11-11 16:44 | Egress map, fine measuring grid and wheelchair audit; crowd validated |
| `772bb69` | `50a5e0e` | 2026-11-11 18:53 | Fire solver: run-health guard, cut steps, confinement fitted (A to E) |
| `3b8ecbd` | `e9d3786` | 2026-11-11 18:53 | Interface audit acted on; a stopped fire run says why |
| `d340ac7` | `e2af789` | 2026-11-11 21:20 | Guide names every panel control; ten panel inconsistencies fixed |
| `7e2e9f5` | `62d786c` | 2026-11-11 22:31 | Fire solver: heat-conserving transport (F) on CPU and GPU; confinement refitted |
| `2c75f3a` | `06095a8` | 2026-11-11 23:33 | Site examples on the old and the final solver; stale fallbacks tidied |
| `d9160e1` | `d4add45` | 2026-11-12 00:26 | Claim protocol: dated addendum before the held-out verdict |
| `0afce90` | `277e1e3` | 2026-11-12 00:28 | Claim protocol: runs whose door the grid seals are not represented |
| `abd9754` | `3d39c13` | 2026-11-12 00:28 | Claim protocol: the addenda's times as their commits record them |
| `41415f7` | `cdd2789` | 2026-11-12 00:45 | Claim protocol: the WTC tree fault, recorded before its rerun |
| `c70b40d` | `130d9a1` | 2026-11-12 01:15 | The app says what the fire is validated for, run by run |
| `e81f59f` | `b984d63` | 2026-11-12 01:32 | Fire model validation review: protocol and queries, before the harvest |
| `74e259c` | `dd7c318` | 2026-11-12 01:33 | Fire model validation review: screening protocol and pre-score, before screening |
| `f1551d7` | `7a1fd06` | 2026-11-12 01:36 | Fire model validation review: harvest logged, deduplicated and pre-scored |
| `888395d` | `0d968c4` | 2026-11-12 01:40 | A calibrated grid option, and help to run the fire on a laptop's faster GPU |
| `6dce48f` | `fc89492` | 2026-11-12 02:17 | Held-out validation harness, data and section 12 (pre-registration for the final run) |
| `2ac6141` | `37fd403` | 2026-11-12 02:21 | Fire solver: the review's two transport faults fixed on the CPU |
| `0c62bd9` | `1fc44f7` | 2026-11-12 02:21 | Validation 12.9 amendment 8: each test burner's radiative fraction and yields from its FDS input |
| `04c5d54` | `c1117b0` | 2026-11-12 02:27 | Handoff: overnight state of the fire calibration (2026-11-12, 02:30) |
| `6773eae` | `be47cf4` | 2026-11-12 06:15 | Fire solver on the GPU follows the CPU fixes; the app offers a solver choice |
| `6ee8c93` | `d3c5751` | 2026-11-12 06:53 | Fire: a burner's fuel burns only with oxygen; pools are design fires |
| `7292e28` | `8247756` | 2026-11-12 06:54 | Claim protocol: dated addendum before the oxygen-limited rerun's verdict |
| `51674ec` | `42b7b8a` | 2026-11-12 06:54 | Claim protocol: the new addendum's time as its commit records it |
| `a2f15d4` | `75c1758` | 2026-11-12 07:16 | Lean: the heat-conserving transport and the oxygen-limited burner |
| `39785f3` | `0ada5a4` | 2026-11-12 07:18 | Fire model validation review: title and abstract screening tallied |
| `6516edb` | `cb36da0` | 2026-11-12 07:22 | GPU speed kept per GPU is the whole run's; sprinkler statistics on 6ee8c93 |
| `e3e9e2b` | `1b34ad3` | 2026-11-12 07:22 | Fire calibration: literature study of the quantities short of calibrated |
| `fc1d3d2` | `62f7cb4` | 2026-11-12 07:34 | Fire model validation review: included DOIs verified double-blind |
| `c7856f0` | `2559aa4` | 2026-11-12 07:38 | Fire model validation review: the OpenAlex refusal was ours, not the key's |
| `1557692` | `c4821e5` | 2026-11-12 07:44 | Fire calibration: what industry practice requires, and the gaps |
| `60b04a9` | `e810d88` | 2026-11-12 08:01 | Fire model validation review: synthesis and PRISMA 2020 flow |
| `ed64296` | `ca84c69` | 2026-11-12 09:01 | The app states the final held-out verdict (6ee8c93) |
| `b0991ec` | `f9e20fa` | 2026-11-12 11:12 | Validation 12.10: the final held-out results, before and after the burner fix |

**2026-11-13.** At the user's request the name is restored, in the same two lines of `wiki/roadmap/evacuation-map-and-accessibility-audit-plan-2026-11-11.md`, by a new commit on top of the published history. The hashes above stay as published; nothing was rewritten again.
