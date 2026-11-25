# Release scorecard

Last updated: 2026-11-25. The release counted here is **the version the two papers will describe** (abstract due 2026-12-15, tool paper due 2027-01-15). If the release is meant to be something else, the list changes and the states stay.

**States.** Not started. Built: the code or the result exists. Tested: an automated test or a recorded check covers it. Owner's check done: the owner has looked at it in the running app or read the result, and said so. An item counts as finished only in the last state.

| # | Item | State | What the next state needs |
| --- | --- | --- | --- |
| 1 | Evacuees stand outside after leaving; those who cannot leave are reported | tested | the owner's look in the live app |
| 2 | Corridor and stair capacity, on by default | tested | the owner's look on a large building (Vienna) |
| 3 | Clock stops at its total; "Slower in smoke" shown only with a fire | built | a browser check, then the owner's look |
| 4 | Pre-movement time and walking speed in smoke (options) | tested | pre-movement in the panel; the owner's look |
| 5 | People model verification, 28 of 29, in the test suite | tested | a stored result file at that count; the owner's read of the report |
| 6 | People model against a real evacuation | not started | a data set and a protocol |
| 7 | Dose limit setting applied everywhere | built | a test named for it; the owner's look |
| 8 | Guide text consistent with the results | built | footers' date; a rerun of the guide check; the owner's read |
| 9 | Fire: smoke layer depth and neutral plane, the standing of 2026-11-13 | tested | the owner's read of the plan's tables |
| 10 | Fire: head-height temperature in band | built (gas radiation, not on the site) | a pass on its rule; it does not pass today |
| 11 | Fire: times to 60 °C and to the heat dose | tested, and failing | the corridor far-field item, which needs the owner's go |
| 12 | Fire: smoke and gases at head height | not started | a data set that measures them there |
| 13 | Fire: CO when the fire is short of air | not started | a proposal fixed before any run |
| 14 | Sprinkler timing | tested | the leave-one-laboratory-out rerun; the owner's read |
| 15 | GPU and CPU solvers agree | built (measured before 2026-11-12) | one comparison on the current solver |
| 16 | Claims table, rules table, open-items register | built | the owner's read; the first weekly review |
| 17 | Paper abstract (2026-12-15) | not started in this repository | a draft, kept out of the public repository |
| 18 | Tool paper (2027-01-15) | not started in this repository | a draft, kept out of the public repository |

## Count

| State | Items |
| --- | --- |
| Not started | 5 |
| Built | 6 |
| Tested | 7 (one of them tested and failing) |
| Owner's check done | 0 |

**How close:** 13 of 18 items exist in some form, 7 are under a test or a recorded check, and none has the owner's check recorded. Two of the items that matter most for the papers' claims (10 and 11) have been measured and do not pass, so more testing will not finish them: they need solver work.

## Keeping it

Reviewed with the open-items register each week. A state moves only on evidence: a commit for built, a test or a session record for tested, the owner's word for the last.
