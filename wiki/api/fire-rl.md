# Fire And RL API

## `POST /fire-sim`

Used for precomputed fire playback.

Request fields:

- `mode`: `wire`, `cell`, or `ifc`.
- `start_id`, `end_id`: optional node IDs.
- `start_point`, `end_point`: optional coordinates.
- `max_steps`: default `60`.
- `precompute`: default `true`.
- `radial`: default `true`.
- `delay_ms`: default `200`, mostly relevant to stream route.

Response:

- `mode`
- `start_id`
- `timeline`: array of node ID arrays
- `cell_bboxes`: optional metadata for cell mode

## `GET /fire-sim/stream`

Streams Server-Sent Events with `text/event-stream`.

Important query params:

- `mode`
- `max_steps`
- `precompute`
- `radial`
- `delay_ms`
- `use_temperature`
- `stream_path`
- `path_recompute_interval`
- `path_alpha`
- `path_lethality_threshold`
- `start_x`, `start_y`, `start_z`
- `end_x`, `end_y`, `end_z`

Event types:

- `meta`: currently cell bounding boxes.
- `step`: node IDs for non-temperature spread.
- `temperature_step`: node temperature dictionary.
- `path_update`: dynamic path coordinates, cost, changed flag.
- `done`: stream complete.

## `POST /rl/train`

Request fields:

- `mode`
- `start_id`, `exit_id`
- `start_point`, `exit_point`
- `episodes`: default `200`
- `max_steps`: default `200`
- `use_fire`: default `true`

Response fields:

- `mode`
- `start_id`
- `exit_id`
- `path`: node ID sequence
- `cell_bboxes`

