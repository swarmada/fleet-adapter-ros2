#!/usr/bin/env python3
# Copyright 2026 The Swarmada Authors <maintainers@swarmada.io>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check this repository's public text: no handles, no names nobody else can reach.

VENDORED, NOT CALLED. The control plane's `scripts/check-error-strings.py` is the parent
of this file and the reason it has this name. It could not be pointed at this repository:

  * it walks `*.go` only, and there is no Go here -- every finding below lives in Python,
    YAML, a Dockerfile, two Makefiles, a shell script or Markdown. Run as-is against these
    trees it reports "no violations" over any amount of anything, which is worse than
    no gate: it is a green light with nothing behind it.
  * it has no rule about handles at all. That rule is this repository's, and the
    core repository is read-only from here.

So this is a copy that walks what is actually here and carries the rules that are actually
wanted. The parent's voice rules come with it, re-aimed at the constructors a Python
message reaches an operator through, so the vendoring does not quietly drop them.

WHAT IS CHECKED, and why each one is about a reader rather than about tidiness:

  1. A TRACKER HANDLE names a record only the project can open. "necessary rather than
     defensive (TRK-0000)" tells a reader that a reason exists and that they may not have
     it. The reason was nearly always already in the sentence, and where it was not, the
     handle was carrying it alone. Filenames count: `trk0000_bringup_evidence.py` reaches
     anyone who lists the directory.
  2. A PATH ON SOMEONE'S MACHINE, or into a tree that is not published, is an instruction
     that cannot be followed -- `sha256sum -c ~/<someone>/<unpublished-dir>/...` as the way
     to verify a run.
  3. THE NAME OF AN UNPUBLISHED REPOSITORY OR TREE, with or without a path around it.
     "the fix landed there, not in this repo" is a forward reference to a clone URL that
     does not exist. The names are DATA, in `scripts/public-surface-names.txt`, because
     this file is scanned too and cannot hold the strings it forbids; see that file for
     how each entry is spelled and why. A reader who needs the distinction gets the
     published name -- the core repository, the adapter registry -- or the reason itself.
  4. A WORK-SESSION HANDLE: a short uppercase prefix and one or two digits, the shape this
     project's prompts and track reports are named in. "HWn's teardown", "an earlier RCn
     census" tells a reader that a run happened and that its report is somewhere they
     cannot look. What the handle stood for -- a date, a plant, an earlier run of this
     same check -- is what they needed, and is nearly always shorter than the sentence
     built around the handle. Shape alone cannot separate these from published
     identifiers of the same shape, so the published ones are ALLOWED BY NAME in the same
     data file, each with the document that publishes it. Two exclusions are structural
     rather than listed: a token carrying a hyphenated prefix (`FA-Rn`, `FZ-Rn`, `DD-Sn`)
     is a published requirement identifier, and a token of three or more digits is a
     version or a model number, not a handle.
  5. A LINK OR TRAILER THAT POINTS AT A WORK SESSION, a conversation, or an unpublished
     artifact. It resolves to a sign-in page for everyone outside the account that made
     it, which is worse than citing nothing: it reads as provenance a reviewer could
     check. Data, in the same file and bracketed the same way. The attribution trailers
     are deliberately NOT covered -- `Co-Authored-By` and `Signed-off-by` name a person
     or a model, not a place a reader is being sent.
  6. The parent's VOICE rules, on strings an operator reads while something is broken:
     apology, marketing register, first person, "simply", an exclamation point.

GENERATED FILES ARE EXEMPT BY PATH, NOT BY LUCK. `GENERATED_EXEMPT` below names each
generated file and says why it is exempt, and the checker prints the exemptions it
applied on every clean run. This is deliberate. The protobuf compiler emits an import of
a third-party module whose path contains one of the words rule 3's neighbour forbids;
the file says DO NOT EDIT at the top and is regenerated from the proto, so a finding
there is not actionable. Leaving those files out of the scanned trees would have the same
effect and record nothing -- the next person to widen the trees would re-discover the
finding and have to re-decide it. A generated file NOT on this list is scanned like any
other file, so a new one forces the decision rather than inheriting it.

THIS FILE IS SCANNED TOO, and so is its data file. `make check-error-strings` includes
`scripts/`, and there is no self-exemption: a gate that passes itself by standing outside
its own trees is the shape of defect it exists to catch. Every example above is a neutral
placeholder for that reason.

Usage: python3 scripts/check-error-strings.py [path ...]   (default: .)
       python3 scripts/check-error-strings.py -            (read the text on stdin)
Exit:  0 clean, 1 violations found, 2 the checker could not run.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".ruff_cache",
             ".pytest_cache", "dist", "build", "map"}
SKIP_SUFFIXES = {".pyc", ".png", ".pgm", ".jpg", ".pdf", ".gz", ".whl", ".so"}
# Every text file that carries prose a reader outside the project may reach.
TEXT_SUFFIXES = {".py", ".sh", ".yml", ".yaml", ".md", ".txt", ".toml", ".cfg", ".json"}
TEXT_NAMES_RE = re.compile(r"^(Dockerfile|Makefile)(\..+)?$")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Exempt from the CONTENT rules, by path, each with the reason. The FILENAME rule still
# applies: a generated file may not be named after a handle either. Keys are repository
# relative. See the module docstring for why this is a list and not a suffix pattern.
GENERATED_EXEMPT = {
    "fleet_adapter/v1/fleet_adapter_pb2.py":
        "protobuf compiler output, marked DO NOT EDIT and regenerated from "
        "fleet_adapter/v1/fleet_adapter.proto. Its line 11 imports a third-party module "
        "whose dotted path carries a word rule 3's neighbour forbids; the path belongs to "
        "the protobuf runtime, not to this project, and editing it would be undone by the "
        "next regeneration",
    "fleet_adapter/v1/fleet_adapter_pb2_grpc.py":
        "grpc plugin output, from the same proto and regenerated with it. Exempt on the "
        "same ground, and listed rather than inferred so that both halves of the generated "
        "pair are a decision",
}

# ── the data the rules are made of ───────────────────────────────────────────

DATA_FILE = Path(__file__).resolve().with_name("public-surface-names.txt")

# Not a pipe: half the patterns in the data file are alternations, and a pipe separator
# would cut the first branch off every one of them and silently narrow the rule.
SEPARATOR = " :: "


class DataError(Exception):
    """The checker cannot run: its data file is missing, empty or unreadable."""


def load_sections(path: Path) -> dict[str, list[tuple[str, str]]]:
    """Read `[section]` / `<regex> :: <why>` lines. Raises DataError rather than
    returning an empty section, which would turn a rule off without saying so."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DataError(f"cannot read {path}: {exc}") from exc

    sections: dict[str, list[tuple[str, str]]] = {}
    current: str | None = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
            continue
        if current is None:
            raise DataError(f"{path}:{lineno}: an entry before any section heading")
        pattern, sep, why = line.partition(SEPARATOR)
        if not sep:
            raise DataError(f"{path}:{lineno}: an entry with no {SEPARATOR.strip()} "
                            f"before its reason")
        pattern, why = pattern.strip(), why.strip()
        try:
            re.compile(pattern)
        except re.error as exc:
            raise DataError(f"{path}:{lineno}: {pattern!r} is not a regex: {exc}") from exc
        sections[current].append((pattern, why))

    for required in ("unpublished-trees", "unreachable-links", "public-handles"):
        if not sections.get(required):
            raise DataError(f"{path}: section [{required}] is missing or empty")
    return sections


# ── the public-surface rules ─────────────────────────────────────────────────

TRACKER = re.compile(r"\b(?:ITEM|ISSUE|TICKET|BUG)[-_ ]?\d{3,4}\b", re.I)

# THE TWO WORDS ARE SPELLED AS CHARACTER CLASSES, here and below. `[i]nternal` matches the
# same text the bare word would and costs nothing at runtime, but neither word ever appears
# in this file -- which is what lets this file sit inside the trees it scans. A gate that
# has to be exempted from its own rule in order to state it is not a gate. The data file
# spells its entries the same way, for the same reason.
_WORD_A = r"[i]nternal"
_WORD_B = r"[p]rivate"

# A developer's own machine, or a directory this repository does not publish.
UNPUBLISHED_PATH = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+"), "names a path on a developer's machine"),
    (re.compile(rf"\b(?:{_WORD_A}|{_WORD_B})-[A-Za-z0-9_-]+/?"),
     "names a directory this repository does not publish"),
]

# THE FLAT RULE, and it is flat on purpose. Either word, in any use: naming a thing, or as
# an ordinary adjective. The maintainer's ruling of 2026-09-21 is that neither is published,
# so there is no "but it is only an adjective" branch to argue over at review time. What is
# meant is always sayable another way -- "built-in", "own", "in-process", "unexported" --
# and the replacement is more precise than the word it replaces in nearly every case.
FORBIDDEN_WORD = re.compile(rf"\b(?:{_WORD_A}|{_WORD_B})\b", re.I)

# A work-session handle: one to four capitals, one or two digits, an optional lowercase
# suffix letter (the shape a follow-on prompt is named in). The lookbehind drops a token
# that carries a hyphenated capital prefix, which is how every published requirement
# identifier in this contract is written; the lookahead keeps the match to a whole token,
# so a GPU part number with a trailing letter is not half-matched. Three or more digits is
# a version or a model number and never reaches here.
SESSION_HANDLE = re.compile(r"(?<![A-Za-z0-9-])([A-Z]{1,4}[0-9]{1,2}[a-z]?)(?![A-Za-z0-9])")
SESSION_HANDLE_ANY_CASE = re.compile(SESSION_HANDLE.pattern, re.I)

# ── the parent's voice rules ─────────────────────────────────────────────────

# Python's equivalents of the parent's message constructors: what an operator reads.
CONSTRUCTOR = re.compile(
    r"\b(?:print|raise|warn|warning|error|exception|critical|info|debug|fail|"
    r"assert_never|ArgsError|RuntimeError|ValueError|SystemExit)\b"
    r"|\blog(?:ger|ging)?\.[A-Za-z]+"
    r"|\bap\.error\b")

LITERAL = re.compile(r'"(?:[^"\\\n]|\\.)*"' + r"|'(?:[^'\\\n]|\\.)*'")

BANNED = [
    (re.compile(r"\b(oops|whoops|uh-oh)\b", re.I),
     "performs an emotion at an operator who is debugging"),
    (re.compile(r"\b(sorry|apolog\w+|unfortunately)\b", re.I),
     "apologises; a message apologises for nothing"),
    (re.compile(r"\bplease\b", re.I), "is politeness in a machine message"),
    (re.compile(r"\b(we|we're|we've|our|us)\b", re.I),
     "gives a machine message a corporate voice"),
    (re.compile(r"\b(simply|just|easily|obviously|trivially)\b", re.I),
     "asserts that something is easy"),
    (re.compile(r"\b(magic|magical|awesome|great news|something went wrong)\b", re.I),
     "is marketing register or names nothing"),
    # `!=` and f-string conversions (`{x!r}`) are syntax, not punctuation aimed at a reader.
    (re.compile(r"!(?![=rsa])"), "uses an exclamation point"),
]


def iter_text_files(roots: list[str]):
    for root in roots:
        base = Path(root)
        paths = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in paths:
            if not path.is_file() or SKIP_DIRS & set(path.parts):
                continue
            if path.suffix in SKIP_SUFFIXES:
                continue
            if path.suffix in TEXT_SUFFIXES or TEXT_NAMES_RE.match(path.name):
                yield path


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def check_name(path: Path, is_public_handle) -> list[str]:
    hit = TRACKER.search(path.name) or re.search(r"\bitem[-_]?\d{3,4}", path.name, re.I)
    if hit:
        return [f"{path}: the FILENAME cites a tracker item ({hit.group(0)!r})\n"
                f"    rename it for what the file does; a directory listing carries this too"]
    # Case-INSENSITIVE here, where the content rule is not, and the asymmetry is the
    # point: filenames are lowercase by convention, so an uppercase-only rule could never
    # see a handle in one. In prose the same relaxation would match `v1` in a package
    # path and `rc1_rf2o` inside a verbatim capture, neither of which is a handle. The
    # tracker rule above already splits the same way.
    for match in SESSION_HANDLE_ANY_CASE.finditer(path.stem):
        if not is_public_handle(match.group(1).upper()):
            return [f"{path}: the FILENAME carries a work-session handle "
                    f"({match.group(1)!r})\n"
                    f"    name it for what it holds; a directory listing carries this too"]
    return []


def check_text(path: Path, lines: list[str], named, is_public_handle) -> list[str]:
    found: list[str] = []
    for lineno, line in enumerate(lines, 1):
        where = f"{path}:{lineno}"

        hit = TRACKER.search(line)
        if hit:
            found.append(f"{where}: {line.strip()[:100]}\n"
                         f"    {hit.group(0)!r} cites a tracker item, which a reader "
                         f"outside the project cannot open")

        for pattern, reason in UNPUBLISHED_PATH:
            hit = pattern.search(line)
            if hit:
                found.append(f"{where}: {line.strip()[:100]}\n"
                             f"    {hit.group(0)!r} {reason}")

        for pattern, why in named:
            hit = pattern.search(line)
            if hit:
                found.append(f"{where}: {line.strip()[:100]}\n"
                             f"    {hit.group(0)!r} {why}")

        hit = FORBIDDEN_WORD.search(line)
        if hit:
            found.append(f"{where}: {line.strip()[:100]}\n"
                         f"    {hit.group(0)!r} is a word this repository does not publish, "
                         f"in any use; say what is meant instead")

        for match in SESSION_HANDLE.finditer(line):
            handle = match.group(1)
            if is_public_handle(handle):
                continue
            found.append(f"{where}: {line.strip()[:100]}\n"
                         f"    {handle!r} is a work-session handle: it names a prompt or a "
                         f"track report nobody outside the project can read. Say what it "
                         f"stood for -- the date, the plant, the run")

        if path.suffix == ".py" and CONSTRUCTOR.search(line):
            for literal in LITERAL.findall(line):
                for pattern, reason in BANNED:
                    hit = pattern.search(literal)
                    if hit:
                        found.append(f"{where}: {literal.strip()[:100]}\n"
                                     f"    {hit.group(0)!r} {reason}")
                        break
    return found


def main() -> int:
    try:
        sections = load_sections(DATA_FILE)
    except DataError as exc:
        print(f"check-error-strings cannot run: {exc}", file=sys.stderr)
        return 2

    named = [(re.compile(pattern), why)
             for key in ("unpublished-trees", "unreachable-links")
             for pattern, why in sections[key]]
    allowed = re.compile("|".join(f"(?:{p})" for p, _ in sections["public-handles"]))

    def is_public_handle(token: str) -> bool:
        return allowed.fullmatch(token) is not None

    args = sys.argv[1:] or ["."]
    findings: list[str] = []
    exempted: list[str] = []

    if args == ["-"]:
        text = sys.stdin.read().splitlines()
        findings.extend(check_text(Path("<stdin>"), text, named, is_public_handle))
    else:
        for path in iter_text_files(args):
            findings.extend(check_name(path, is_public_handle))
            reason = GENERATED_EXEMPT.get(repo_relative(path))
            if reason is not None:
                exempted.append(f"  {repo_relative(path)}\n      {reason}")
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            findings.extend(check_text(path, lines, named, is_public_handle))

    if findings:
        print("Public text carrying something a reader outside the project cannot use:\n")
        for finding in findings:
            print(finding)
        print(f"\n{len(findings)} finding(s). Replace a handle with the reason it stood "
              f"for -- not by deleting\nthe sentence around it, which is where the reason "
              f"usually already was.")
        return 1

    print("check-error-strings: no violations")
    if exempted:
        print(f"{len(exempted)} generated file(s) exempt from the content rules by path:")
        for line in exempted:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
