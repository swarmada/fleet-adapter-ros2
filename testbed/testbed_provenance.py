#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
# SPDX-License-Identifier: Apache-2.0

"""Identify the apparatus that produced a number. Nothing else.

WHY THIS EXISTS
----------------
The testbed is untracked by design, and that design is not challenged here: it is a
measuring instrument, and an instrument that lives under version control alongside the
thing it measures invites being tweaked until a failing run passes. What went wrong is
narrower. "Untracked" was allowed to mean "unversioned, uncompared, and existing in three
mutually inconsistent copies" -- so `Makefile.demo` (which defines the targets named in
the recorded demo casts) exists on exactly one disk, `BUILD ?=` was lost from two
Makefiles, and NO TOOL IN EITHER REPOSITORY COULD SEE ANY OF IT.

Every number in `validation/` is quoted as reproducible via a named command. That premise
requires the command to still exist and still mean what it meant. Re-derivation is this
project's standing rule, and a number whose apparatus cannot be identified cannot be
re-derived.

So: give the apparatus an identity, print it beside every number, and make divergence a
thing you can check rather than a thing you discover three workspaces later.

WHAT IS FINGERPRINTED
---------------------
Every file in the testbed directory that DETERMINES BEHAVIOUR: the Makefiles, the compose
files, the Dockerfile, nav2_params.yaml, and every .py and .sh probe. Sorted by name,
each hashed, and the list of (name, hash) pairs hashed again -- so the digest changes if
any harness file changes, is added, or is REMOVED. Removal is the case this is for;
a per-file hash list alone would not have caught a missing `Makefile.demo`.

Generated artefacts are excluded and named in the output rather than silently skipped:
`map/` (deterministic output of make_map.py, which IS fingerprinted), `__pycache__`,
`_to_delete`, and `conformance-*.json` run outputs.

This file fingerprints ITSELF. Adding it changed every digest once, on purpose: a
fingerprinter outside its own fingerprint is a fingerprinter anyone can edit.

WHAT IS NOT COVERED, and it matters
-----------------------------------
The container image. The harness runs inside `swarmada/ros2-testbed:jazzy`, and the
Dockerfile is fingerprinted but the BUILT IMAGE is not -- an apt mirror moving under a
`ros-jazzy-nav2-bringup` install produces a different Nav2 from an identical Dockerfile.
`Makefile.r2 r2-stamp` records the image ID alongside the digest for that reason; it is
host-side information and cannot be read from inside the container.

USE
---
    python3 testbed_provenance.py                 # the one-line stamp
    python3 testbed_provenance.py --manifest      # sha256sum -c compatible manifest

and, from a measurement script:

    from testbed_provenance import stamp
    print(f"testbed: {stamp()}")

To check a workspace against a recorded session:

    cd <the testbed named in that session's results document>
    sha256sum -c <that session's recorded manifest>
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Behaviour-determining files. Matched against the file NAME, in this directory only --
# the testbed is flat by design and a recursive walk would pull in map/ and __pycache__.
HARNESS_SUFFIXES = (".py", ".sh", ".yml", ".yaml")
HARNESS_NAMES = ("Dockerfile",)
HARNESS_PREFIXES = ("Makefile",)
# Excluded, and listed in the stamp rather than dropped in silence.
EXCLUDED_DIRS = ("map", "__pycache__", "_to_delete")
EXCLUDED_PREFIXES = ("conformance-",)


def _is_harness(name: str) -> bool:
    if name.startswith(EXCLUDED_PREFIXES):
        return False
    if name in HARNESS_NAMES or name.startswith(HARNESS_PREFIXES):
        return True
    return name.endswith(HARNESS_SUFFIXES)


def harness_files(root: str = HERE) -> list[str]:
    """Sorted names of every behaviour-determining file directly in `root`."""
    return sorted(
        n for n in os.listdir(root)
        if os.path.isfile(os.path.join(root, n)) and _is_harness(n)
    )


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def file_hashes(root: str = HERE) -> list[tuple[str, str]]:
    return [(n, _sha256(os.path.join(root, n))) for n in harness_files(root)]


def digest(root: str = HERE) -> str:
    """One hash over the whole apparatus.

    Hashing "name + hash" per line, rather than concatenating file contents, is what
    makes a REMOVED file change the digest, which is the failure mode. Concatenated
    contents would be blind to it.
    """
    pairs = file_hashes(root)
    joined = "".join(f"{n}\0{h}\n" for n, h in pairs)
    return hashlib.sha256(joined.encode()).hexdigest()


def stamp(root: str = HERE) -> str:
    """The one line that goes beside every number."""
    pairs = file_hashes(root)
    return (f"{digest(root)[:12]} ({len(pairs)} harness files, "
            f"{os.path.basename(os.path.dirname(os.path.abspath(root)))}/"
            f"{os.path.basename(os.path.abspath(root))})")


def changed_since(before: str, root: str = HERE) -> str | None:
    """Was the instrument edited WHILE it was measuring?

    Every script stamps at the start and again beside its results. `testbed/` is
    bind-mounted into the container, so an edit lands live -- and a harness edited during
    its own run is the exact thing the standing rule ("do not adapt the instrument to
    what it measures") exists to prevent. It happened once during this file's own
    development and was caught only because the two stamps were read side by side.
    Returns a warning line, or None when the apparatus held still.
    """
    now = stamp(root)
    if now == before:
        return None
    return (f"WARNING: the testbed CHANGED DURING THIS RUN -- started {before}, "
            f"ended {now}. These numbers were taken through two different instruments "
            f"and must not be quoted.")


def manifest(root: str = HERE) -> str:
    """`sha256sum -c` compatible, with the roll-up digest and the exclusions as
    comments. `sha256sum -c` ignores lines it cannot parse, so the header travels with
    the manifest instead of in a separate file that gets separated from it."""
    pairs = file_hashes(root)
    lines = [
        "# swarmada testbed manifest",
        f"# digest      {digest(root)}",
        f"# files       {len(pairs)}",
        f"# generated   {os.path.abspath(root)}",
        f"# excluded    dirs={','.join(EXCLUDED_DIRS)} "
        f"prefixes={','.join(EXCLUDED_PREFIXES)}",
        "# check with  cd <testbed> && sha256sum -c <this file>",
    ]
    lines += [f"{h}  {n}" for n, h in pairs]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=HERE, help="testbed directory (default: this one)")
    ap.add_argument("--manifest", action="store_true",
                    help="print a sha256sum -c compatible manifest")
    ap.add_argument("--digest", action="store_true", help="print the full digest only")
    args = ap.parse_args()

    if args.manifest:
        sys.stdout.write(manifest(args.root))
    elif args.digest:
        print(digest(args.root))
    else:
        print(stamp(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
