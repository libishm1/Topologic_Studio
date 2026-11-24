# Browser session, 2026-11-24: the two new People leaving toggles

Headless Chrome (playwright-core) against main at pull request 23, before the stair deadlock fix and with both toggles off by default. No backend running.

## What was tested

- **Corridors and stairs limit the flow**, on the Duplex and the two-storey Office: toggle on, then off again.
- **Slower in smoke**, on the Duplex, after a Flamescapes run on the GPU (WebGPU, integrated graphics).

## Results

| Model | Toggle | Off | On |
| --- | --- | --- | --- |
| Duplex | Flow limit | 44 people, last out at 29 s | 44 people, last out at 30 s |
| Office, two storeys | Flow limit | 92 people, last out at 65 s | 92 people, last out at 66 s |
| Duplex, fire of 180 s, leaving at 120 s | Slower in smoke | last out at 148 s, 8 cut off by the fire | last out at 157 s, 8 cut off by the fire |
| Duplex, fire of 30 s, leaving at once | Slower in smoke | last out at 29 s | last out at 29 s (no smoke under 10 m visibility before everyone is out) |

- Switching a toggle off again gave the first numbers back in every case.
- No page errors. The only console errors were the refused requests to the backend, which was not running.
- Vienna with the flow limit on showed "75 still inside when the walk stopped": the stair deadlock, fixed afterwards (a stair that fills while its foot is held up empties again).

## Findings

| Kind | Finding | Status |
| --- | --- | --- |
| functional | With the flow limit on, people stayed on a stair for good in Vienna, the NIBS Office and KIT Smiley West | Fixed the same day, with a unit test |
| visual | With the fire, the people clock can read past its total ("167 s of 158 s") | Open |
| visual | The clock's total is rounded up and the notice to the nearest second ("of 96 s" beside "out at 95 s") | Open |
| functional | "Slower in smoke" is shown whenever people leave with the fire, also with a fire model that has no smoke; it then does nothing | Open; its hint says it needs a Flamescapes run |

## Not tested

- The flow limit on by default, in a browser (the default was switched on after this session).
- The CPU path of the fire solver (the GPU was available).
- The flow limit together with a fire.
- The NIBS Office and KIT Smiley West in a browser after the fix (checked headless only: everyone leaves).
