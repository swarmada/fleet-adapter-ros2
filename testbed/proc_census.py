#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Teardown census: is each launched process group RUNNING, DEFUNCT, or ABSENT?

WHY THIS EXISTS
----------------
The robot sessions launch each stack with `setsid nohup ... &` from an `mrun` shell that
exits at once, and tear it down with `kill -INT -<pgid>`, then `-TERM`. Every group leader
is then left `<defunct>`: reparented to the MentorPi container's PID 1, which does not
reap. The teardown at the end of the 2026-09-18 R1 interval run printed

    container 0   slam_toolbox 0   bridge 0   bag 0   nav2 nodes 0

while PGIDs 19687, 21716, 21747 and 28433 each still held a `Zs [ros2] <defunct>` leader.
The rest-channel run later that same day printed `none left` over PGID 24063's defunct
leader. Both checks counted by NAME, and a defunct process has lost its argv: `ps` shows
`[ros2] <defunct>`, so no pattern naming what it ran can match it. A count by name
therefore answers 0 for a group that is reaped, for a group that is defunct, and -- if
the name changed -- for a group that is still running. Three different states, one
number.

This is a whole class of defect in this testbed: a check that reports a count it cannot
back.
The fix is not "also count zombies". It is a verdict with three values, and every one of
them is printed:

    RUNNING   at least one member is not a zombie        -- teardown did not finish
    DEFUNCT   members exist, and every one is a zombie   -- torn down, not reaped
    ABSENT    no process in the table has this PGID      -- gone

A PGID survives the zombie state, so a PGID query sees all three. A NAME query cannot
see DEFUNCT, and says so on every line instead of printing ABSENT. And every defunct row
in the table is listed at the end whether or not anything asked about it, so a zombie is
never invisible because nobody knew its PGID.

USAGE
-----
The table is read from stdin, so the check runs on the Mac against the robot's process
table and needs nothing installed on the robot:

    ~/mrun 'ps -eo pid,ppid,pgid,stat,args' | \\
        python3 testbed/proc_census.py --pgid 21716 --pgid 21747 --name slam_toolbox

Exit 0 = every queried group ABSENT and no name matched a live process. 1 = something is
RUNNING. 3 = nothing running, but a queried group is DEFUNCT. 2 = the table could not be
read. Parsing is pure, so every verdict is testable without a robot.

The LAST line printed is the verdict, `PASS  teardown clean` or `FAIL  TEARDOWN NOT CLEAN`
naming each group that is not ABSENT. An exit code alone is a line in a log:
the robot runbooks call this at the end of every teardown, and the operator has to be able
to see the failure without knowing the codes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

RUNNING, DEFUNCT, ABSENT = "RUNNING", "DEFUNCT", "ABSENT"
COLUMNS = ["PID", "PPID", "PGID", "STAT"]  # then ARGS/CMD/COMMAND, which may contain spaces
PS_COMMAND = "ps -eo pid,ppid,pgid,stat,args"


@dataclass(frozen=True)
class Proc:
    pid: int
    ppid: int
    pgid: int
    stat: str
    args: str

    @property
    def defunct(self) -> bool:
        # procps and BSD ps both put the state letter first; Z is a zombie.
        return self.stat.startswith("Z")


@dataclass(frozen=True)
class Verdict:
    target: str
    state: str
    running: tuple[Proc, ...]
    defunct: tuple[Proc, ...]
    note: str = ""


class TableError(ValueError):
    pass


def parse_ps(text: str) -> list[Proc]:
    """Rows of `ps -eo pid,ppid,pgid,stat,args`. Refuses anything else rather than guess:
    a census read from the wrong columns would print confident nonsense."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise TableError("empty process table")
    head = lines[0].split()
    if head[:4] != COLUMNS or len(head) != 5:
        raise TableError(f"header {lines[0].strip()!r} is not `{PS_COMMAND}`")
    procs = []
    for ln in lines[1:]:
        parts = ln.split(None, 4)
        if len(parts) < 4:
            raise TableError(f"cannot read row {ln!r}")
        try:
            pid, ppid, pgid = (int(p) for p in parts[:3])
        except ValueError as exc:
            raise TableError(f"cannot read row {ln!r}") from exc
        procs.append(Proc(pid, ppid, pgid, parts[3], parts[4] if len(parts) > 4 else ""))
    return procs


def _verdict(target: str, members: list[Proc], note: str = "") -> Verdict:
    running = tuple(p for p in members if not p.defunct)
    defunct = tuple(p for p in members if p.defunct)
    state = RUNNING if running else DEFUNCT if defunct else ABSENT
    return Verdict(target, state, running, defunct, note)


def by_pgid(procs: list[Proc], pgid: int) -> Verdict:
    return _verdict(f"PGID {pgid}", [p for p in procs if p.pgid == pgid])


NAME_BLIND = ("a name cannot match a defunct process -- its argv is gone; "
              "query its PGID, and see the defunct census below")


def by_name(procs: list[Proc], pattern: str) -> Verdict:
    """What `pgrep -f` answers, with the part it cannot answer said out loud.

    Zombies are excluded from the match on purpose: `[ros2] <defunct>` matching `ros2`
    would be a coincidence of the comm field, not evidence of which group it was. So a
    name query reports RUNNING or ABSENT-FROM-LIVE-PROCESSES, never plain ABSENT."""
    rx = re.compile(pattern)
    live = [p for p in procs if not p.defunct and rx.search(p.args)]
    v = _verdict(f"name /{pattern}/", live, NAME_BLIND)
    if v.state == ABSENT:
        return Verdict(v.target, "NO LIVE MATCH", (), (), NAME_BLIND)
    return v


def _row(p: Proc) -> str:
    return f"{p.pid:>7} ppid {p.ppid:<7} pgid {p.pgid:<7} {p.stat:<5} {p.args}"


def render(verdicts: list[Verdict], procs: list[Proc]) -> str:
    out = []
    for v in verdicts:
        out.append(f"{v.target:<24} {v.state:<13} running {len(v.running)}  "
                   f"defunct {len(v.defunct)}")
        out += [f"    running  {_row(p)}" for p in v.running]
        out += [f"    defunct  {_row(p)}" for p in v.defunct]
        if v.note:
            out.append(f"    note: {v.note}")
    zombies = [p for p in procs if p.defunct]
    out.append(f"defunct census: {len(zombies)} defunct row(s) in the whole table")
    out += [f"    {_row(p)}" for p in zombies]
    return "\n".join(out)


def verdict_line(verdicts: list[Verdict]) -> str:
    """The one line a teardown is judged by. A DEFUNCT group fails it as surely as a
    RUNNING one: torn down is not gone, and "none left" over a zombie is the defect
    this census exists for."""
    code = exit_code(verdicts)
    if code == 0:
        return f"PASS  teardown clean: {', '.join(v.target for v in verdicts)} -- exit 0"
    why = {RUNNING: "still running: teardown did not finish",
           DEFUNCT: "dead but not reaped: torn down, NOT gone"}
    bad = [f"{v.target} {v.state} ({why[v.state]})" for v in verdicts if v.state in why]
    return (f"FAIL  TEARDOWN NOT CLEAN: {'; '.join(bad)} -- exit {code}. "
            f"Do not record this teardown as clean.")


def exit_code(verdicts: list[Verdict]) -> int:
    if any(v.state == RUNNING for v in verdicts):
        return 1
    if any(v.state == DEFUNCT for v in verdicts):
        return 3
    return 0


def main(argv: list[str] | None = None, stdin=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pgid", type=int, action="append", default=[],
                    help="a process group that was launched and should now be gone")
    ap.add_argument("--name", action="append", default=[],
                    help="a pgrep -f style pattern; cannot see defunct processes")
    args = ap.parse_args(argv)
    if not args.pgid and not args.name:
        ap.error("give at least one --pgid or --name")
    try:
        procs = parse_ps((stdin or sys.stdin).read())
    except TableError as exc:
        print(f"proc_census: {exc}", file=sys.stderr)
        print("FAIL  TEARDOWN NOT VERIFIED: the process table could not be read -- exit 2")
        return 2
    verdicts = ([by_pgid(procs, g) for g in args.pgid]
                + [by_name(procs, n) for n in args.name])
    print(render(verdicts, procs))
    print(verdict_line(verdicts))
    return exit_code(verdicts)


if __name__ == "__main__":
    sys.exit(main())
