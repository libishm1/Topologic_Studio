# This week's faults, by cause (2026-11-24 and 2026-11-25)

A count of the faults found or fixed in the two days' work, each put under one cause. The list was made from the week's handoff notes, PRs and plan sections, so a fault nobody wrote down is not in it. Limits of the models found by comparison (the late far field, dense flow too fast) are findings, not faults, and are left out. The sorting is one person's judgement; several faults could sit under two causes and were put under the nearer one.

## The count

| Cause | Faults | Share |
| --- | --- | --- |
| B. A check made against a simplified case instead of the real thing | 7 | 30 % |
| A. A value or fact taken from elsewhere, or from memory, without its source | 6 | 26 % |
| D. A setting or a piece of interface logic with no test | 4 | 17 % |
| C. Text or evidence left stale after the code changed | 3 | 13 % |
| E. A test that fails by chance or by machine load | 2 | 9 % |
| F. Tooling | 1 | 4 % |
| Total | 23 | |

**The two causes guessed beforehand, A and B, account for 13 of 23 (57 %).** That is a majority, not "most by far": a third cause, things with no test at all (D), and a fourth, stale text (C), make up most of the rest.

## The faults

**B. Checked against a simplified case (7)**
1. Evacuees drawn back inside the building: the gathering place was never checked on a real plan (PR 18).
2. Occupants who reach no exit not reported; an upper exit's people drawn at ground level: found on Vienna, not on the test rooms (PR 19).
3. The capacity option deadlocked on stairs in three real buildings; the harness's rooms are simple and passed (PR 24).
4. Two test cases were run with deviations from their text (an exit closing at 10 s for 1 s; people not packed from one side) and counted as passes (PR 21).
5. KIT Smiley West's stair flights are cross-linked: real geometry no test has (open).
6. The suite's dead-channel rule assumes a reading under 1 % is a dead sensor; in one test it is real (plan 16.57).
7. The smoke reducer assumed a thermocouple at the probe's own point; the data have it 2 cm away (plan 16.57).

**A. Taken without its source (6)**
8. Heptane's CO yield of 0.008 is the source's upper bound, used as the value (plan 16.55, 16.57).
9. "880 nm" for the smoke meter: no instrument checked is at that wavelength; origin not found (plan 16.55).
10. The saved file for 0.27 m named from memory, wrongly (plan 16.57).
11. IMO Test 10's placement of people written from memory, wrongly (PR 21).
12. The stair and door node labels swapped in a diagnosis: the constants were assumed, not read.
13. The deck said the people model had been compared with no test cases; a harness already existed.

**D. No test (4)**
14. The dose-limit setting was not wired to the standing doses (PR 17).
15. The least step speed lifted a slow walker in smoke (PR 24).
16. The people clock ran past its total (PR 25).
17. "Slower in smoke" was offered with no fire to make smoke (PR 25).

**C. Stale text or evidence (3)**
18. Eighteen inconsistencies in the guide (PR 15).
19. Stale text inside the app (PR 16).
20. The stored people-model result file says 25 pass where the claim is 28 (open item 19).

**E. Tests that fail by chance or load (2)**
21. A population test failed by sampling chance in quick mode (fixed with fixed seeds).
22. A timing test fails only on a loaded machine (open).

**F. Tooling (1)**
23. A merge call failed silently and was only caught by reading the remote.

## What each cause asks for

| Cause | The cheapest guard | Rule it would protect (rules table) |
| --- | --- | --- |
| B | Every option and every drawing change is run on the seven example buildings before merge, by a script, not by hand | A2 |
| A | Every default and every fuel property names its source in the code, and a test fails where one is missing | R6 |
| D | A test that pins the default settings and one for each setting's wiring | A1, R8 |
| C | The guide check and the result files regenerated in the same PR as the change they describe | R4, R5 |

Guards for B and A would have caught 13 of the 23. The guard for B is the larger piece of work; the guard for A is a day.
