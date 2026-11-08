# Map-view contract, version `mapview/0` (draft)

The data a fleet supplies to a read-only map viewer: the map a robot builds, where the robot is,
what is drawn on top, and what is happening. `contract.py` beside this file implements it, and
the tests in `tests/test_mapview_contract.py` hold it to this text.

**Status: experimental, outside the Swarmada contract.** `mapview/0` is a sidecar format of this
adapter, not part of the Swarmada contract: it is not in the Fleet Adapter protocol, the CRDs or
the normative RFC text, and nothing in the control plane reads it. It has its own version string
and changes on its own schedule. A change that would make a `mapview/0` reader misread a message
gets a new version string, and every `mapview/0` reader refuses that string (see
[Versions](#versions)).

What a consumer may rely on while it is experimental:

- Every message carries `contractVersion`, and a reader that refuses an unknown version, as
  [Versions](#versions) says, never misreads a message from a later one.
- The format has no command message: nothing a consumer receives can, or asks it to, move a robot.
- Frames are never mixed: a writer names a new frame for each SLAM session, and the frame rules
  below hold.
- A `guided:` source is operator-driven, and a `simulated: true` session involves no real robot.

What a consumer must not rely on: the fields, rates and limits of `mapview/0` staying as they are,
any version being supported after a later one appears, or this format being promoted to a stable
version.

The key words MUST, MUST NOT, SHOULD and MAY are used as in RFC 2119.

## Transport

- One JSON object per message, UTF-8, one message per line, each line ending in `\n` (JSON
  Lines). The same lines are a file, the body of an HTTP `POST` to the map-view service
  (`Content-Type: application/x-ndjson`), and the `data:` of one server-sent event each.
- A line MUST NOT be longer than 8 MiB (8,388,608 characters). `NaN` and `Infinity` are not
  numbers in this contract.
- The contract has no command message. A reader MUST NOT send anything to a robot, and a
  viewer connects to the map-view service, never to a robot's network address.

## Every message

| Field | Type | Meaning |
|---|---|---|
| `contractVersion` | string | `"mapview/0"`. Read first; see [Versions](#versions). |
| `kind` | string | `MapSnapshot`, `PoseSample`, `Overlays` or `SessionStatus`. |
| `robot` | string | A local alias: a DNS label (lowercase letters, digits and `-`, 1 to 63 characters, starting and ending with a letter or digit). No dot, colon or space, so an address, a host name or a MAC cannot be sent as one. |
| `frame` | string | The coordinate frame, 1 to 128 characters from `A-Z a-z 0-9 _ @ . / -`. For a SLAM session it is `map@<session>`. |
| `stamp` | number | When the data was observed, in seconds since the Unix epoch, on the writer's clock. |

A reader MUST ignore a field it does not know, and MUST refuse a message that lacks a required
field or carries a required field of the wrong type.

## Frames and units

- Lengths are metres. Angles are degrees, counter-clockwise positive, measured from the frame's
  +x axis, as in ROS (REP 103).
- A map's rows run along +y and its columns along +x; the grid is not rotated in its frame.
- Every SLAM session is its own frame, even on one robot, until something aligns two of them.
  Nothing in `mapview/0` aligns frames. A writer names a new frame for each SLAM session, and a
  resumed session (SLAM started from a saved pose graph) MAY keep the frame it resumed.
- A viewer locks to the frame of the first message it accepts. A `SessionStatus` with `phase:
  started` in another frame begins a new view: the viewer discards everything it holds and locks
  to the new frame. Any other message in another frame is refused and counted. A viewer MUST NOT
  draw data from two frames in one view.

## `MapSnapshot`: the occupancy grid

| Field | Type | Meaning |
|---|---|---|
| `seq` | integer, at least 0 | Increases with each snapshot of the session. A reader keeps the snapshot with the highest `seq`; an older one never replaces a newer one. |
| `resolution` | number, above 0 | Metres per cell. |
| `origin` | `[x, y]` | The frame position, in metres, of the outer corner of cell (0, 0). |
| `width`, `height` | integers, at least 1 | Cells. `width` x `height` MUST NOT exceed 16,777,216 (4096 x 4096); a reader refuses a larger grid from these two fields, before decoding anything. |
| `cells` | string | Base64 of a PNG: 8-bit grayscale, not interlaced, every row with filter type 0, `width` x `height` pixels. Pixel value = occupancy + 1: 0 is unknown (-1), 1 is free (0), 101 is occupied (100). **Row 0 of the PNG is row 0 of the grid**, the bottom row, at `origin`, as in a ROS `OccupancyGrid`. |
| `sha256` | string | SHA-256, 64 lowercase hex digits, of the raw pixel values (row-major from row 0, before PNG). A reader checks it and refuses a snapshot whose cells do not match. |

## `PoseSample`: where the robot is

| Field | Type | Meaning |
|---|---|---|
| `x`, `y` | number | Metres, in `frame`. |
| `yaw_deg` | number | Heading in degrees, counter-clockwise positive. Any finite value; a reader normalises it. |
| `source` | string, 1 to 32 characters | What the pose comes from: `slam` (map to base_footprint through TF), `odometry`, or `sim` for a simulated robot. |

## `Overlays`: what is drawn on top

The latest `Overlays` replaces the previous one whole.

| Field | Type | Meaning |
|---|---|---|
| `keepouts` | list | Each `{name, grid}`. `grid` carries the `MapSnapshot` grid fields (`resolution`, `origin`, `width`, `height`, `cells`, `sha256`), checked the same way; a cell of 100 (pixel 101) is kept out, and every other value is not. |
| `zones` | list | Each `{name, polygon}`, `polygon` being at least 3 points `[x, y]` in `frame`. |
| `target` | object or null | `{label, x, y}`, and optionally `state`: where the robot is to go next, a guided session's next mark or a navigate-handover's target. `label` is 1 to 32 characters. |

`target.state` is **optional**. A writer that knows who is driving to the target says so; a
target without it is drawn as before. When present it is one of:

| `state` | Colour | Means |
|---|---|---|
| `robot` | yellow | The robot drives itself to the target (Nav2); or Nav2 declined and the robot is being brought to a confirmed rest. Teleop reaches nothing. |
| `operator` | red | The operator drives (teleop), under the action's lease. For a navigate-handover, written only after the robot was confirmed at rest; a manual-drive's circle is `operator` from its start. |
| `reached` | green | The action ended Succeeded: the robot is at the target. |
| `stopped` | grey | Nothing may drive: a stop of any scope, a lease that ended, or the action ended Failed. Not left for this target; only a new action, with its own target, starts again at `robot`. |
| `done` | none | The last target was reached; no target is left. A viewer shows "DONE". |

- The writer derives `state` from the action as the adapter runs it, never from the pose: a
  reader MUST NOT infer it from where the robot is drawn.
- `robot` and `operator` are never both true: one action, one lease holder, one state.
- A stop wins at once: `stopped` follows the stop on the writer's next look (0.1 s), whatever
  the action's half.
- A reader that knows the field refuses any other value of it. A reader that does not know the
  field ignores it, as it ignores any field it does not know, and still reads `{label, x, y}`.

`target.baton` is **optional** too: who holds the robot, from the adapter's own record, written
by a navigate-handover's or a manual-drive's feed. When present it is an object:

| Field | Type | Meaning |
|---|---|---|
| `operation` | string | `navigate-handover` or `manual-drive`. |
| `action` | string | The FleetAction's name, whole (`label` may be cut at 32 characters). |
| `gate` | string | The teleop gate now: `armed` (open for this action, forwarding nothing: Nav2 drives, or a declined goal waits for rest), `open` (the operator's teleop reaches the base) or `closed`. |
| `phase` | string | The action's own phase in the adapter: `Navigating` (Nav2 drives), `Declined` (Nav2 declined; the base is held while rest, or the operator's ready, is confirmed), `Running` (the operator drives), `Succeeded`, `Failed`. |
| `gateLog` | list | Every gate change for this action, oldest first, as the gate recorded it under the adapter's lock: `{seq, stamp, gate, reason}`, `stamp` the writer's wall clock. At most the newest 16. |
| `turn` | object | `{startedAt, endedAt, outcome, message}`: when the action's lease was granted; when the writer saw the action end and its gate close (null while it runs); `succeeded`, `failed` or `stopped` (null while it runs); the action's own message. |
| `circle` | object or null | The adapter's last in-circle / still judgement: `{distance_m, radius_m, inside, still_s, settle_s}`. Null until it has judged a pose: while Nav2 drives it judges none (Nav2 says when it arrives). |

- Every field is the adapter's, never inferred from the pose or from a message: a reader MUST
  NOT derive the gate from where the robot is drawn or from `SessionStatus.text`.
- A new `Overlays` is written whenever the baton changes, so each gate change and each sampled
  tick is in the feed with its time.

## `SessionStatus`: what is happening

| Field | Type | Meaning |
|---|---|---|
| `source` | string | What produced the session (below). |
| `phase` | string | `started`, `running` or `stopped`. |
| `text` | string, at most 200 characters | One line for people. |
| `area_free_m2` | number or null | Free area of the latest map, in square metres. |
| `simulated` | boolean, optional | `true` when no real robot is involved. Absent means `false`. A viewer that shows a simulated session says so. |
| `progress_pct` | integer 0 to 100, or null; optional | The FleetAction's own progress, where it reports one. |

`source`, and who moves the robot:

| `source` | Who moves the robot | A viewer says |
|---|---|---|
| `guided:<label>` | An operator drives it (teleop). | operator-driven. A viewer MUST NOT describe a guided session as autonomous. |
| `replay:<label>` | Nobody: a stored run played back. | a replay. |
| A FleetAction operation, such as `explore-floor` | The robot, inside the bound the FleetAction names. | the operation's name. |

## Rates

What a writer sends while a session runs:

| Message | When |
|---|---|
| `SessionStatus` | At start (`started`), when the phase or text changes (`running`), and at the end (`stopped`). |
| `Overlays` | Once at start, before or with the first `MapSnapshot`, and whenever an overlay changes. |
| `MapSnapshot` | When the map has changed, checked at most every 5 s; and at least every 30 s even when it has not (a repeat carries a new `seq` and the same `sha256`). |
| `PoseSample` | 2 Hz (every 0.5 s); at most 10 Hz. |

A writer that cannot reach its reader MAY drop messages; it drops the oldest `PoseSample`s
first and keeps the latest `MapSnapshot`, `Overlays` and `SessionStatus`.

## Staleness

A consumer judges freshness by when the latest message of each kind **arrived**, on its own
clock. It MUST NOT compare `stamp` with its own clock: the writer's clock and the reader's
differ. In a replay, the replay's clock (the stamps played back) is the consumer's clock.

| What | Fresh | Stale | How it is shown |
|---|---|---|---|
| Pose | A `PoseSample` arrived within 2 s | No `PoseSample` for more than 2 s; **lost** after 10 s | Stale: the robot is drawn hollow and grey, with its age ("pose 4 s old"). Lost: the heading arrow is removed and the last position stays as a hollow marker. |
| Map | A `MapSnapshot` arrived within 35 s | None for more than 35 s (the 30 s repeat was missed) | The map is drawn dimmed, with the time since the last snapshot. Separately, the viewer shows how long ago the map last changed; an unchanged map is not stale. |
| Session | | | After `phase: stopped`, nothing is expected: no staleness is shown, and the viewer says the session ended. |

## Versions

- A reader reads `contractVersion` first. If it is not a version the reader implements, the
  reader refuses the whole message without reading any other field, counts it apart from other
  refusals, and shows the count and the version string it saw. It keeps accepting messages in
  the versions it knows.
- A reader MUST NOT guess at, or partly read, a message in an unknown version, and MUST NOT
  treat `mapview/1` as `mapview/0`.
- The map-view service refuses an unknown version at ingest and forwards nothing from it.

## Refusals

A reader refuses, and counts, a message that is not JSON, is over the line limit, lacks a
required field, has a field of the wrong type or a number that is not finite, names an unknown
`kind`, carries a grid whose size, PNG or `sha256` does not check, or arrives in another frame
(see [Frames and units](#frames-and-units)). A refused message is counted, never raised to the
person viewing, and never stops the reader.

## Privacy

- A writer MUST NOT put a site value in a message: no address, place name, MAC or network name.
- A map shows the layout of someone's home or site. The map-view service listens only on
  loopback unless it is told otherwise, and nothing is public unless its owner exports it.

## Changes

Within `mapview/0`. An optional field a reader may ignore keeps the version string; a change that
would make a `mapview/0` reader misread a message gets a new one.

- 2026-10-05: `Overlays.target.state`, optional, added (`robot`, `operator`, `reached`,
  `stopped`, `done`), written by a navigate-handover's feed. A reader of the earlier text ignores
  it and reads `target` as before.
- 2026-10-10: `Overlays.target.baton`, optional, added (the gate's record, the turn's start and
  end, the in-circle / still judgement), and a manual-drive's feed: source `guided:manual-drive`,
  its circle `operator` from the start (a manual-drive opens its gate to the operator in the
  lease step; it takes no rest confirmation). A reader of the earlier text ignores `baton` and
  reads `target` as before.
