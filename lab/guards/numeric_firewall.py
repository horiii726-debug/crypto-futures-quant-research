#!/usr/bin/env python3
"""PreToolUse(Write|Edit) hook: block untagged numbers written into reports/.

R1  Every numeric claim in reports/ must carry an [EXP-xxxxxx] tag.

Reads a JSON hook event on stdin:
    {"tool_input": {"file_path": "...", "content": "..."}}          (Write)
    {"tool_input": {"file_path": "...", "new_string": "..."}}       (Edit)
Exit 0 = allow, exit 2 = block.

Heuristic: split prose into lines. A line is a "numeric claim" if it contains a
digit that is not part of an obvious non-metric token (dates, list markers,
section numbers, code spans, tables of contents, URLs, EXP ids themselves).
Every numeric-claim line must contain [EXP-xxxxxx] or an explicit
[NO-METRIC] / [SPEC] / [PLAN] escape used for structural text.
"""
import json
import re
import sys

EXP_TAG = re.compile(r"\[EXP-[0-9a-fA-F]{6}\]")
ESCAPE_TAG = re.compile(r"\[(NO-METRIC|SPEC|PLAN|TBD|NULL)\]")
# tokens that carry digits but are not metrics
BENIGN = re.compile(
    r"""(
        \b\d{4}-\d{2}-\d{2}\b            # ISO date
      | \bR\d+\b                          # rule refs R1..R10
      | \bG\d+(\.\d+)?\b                  # gate refs G0..G10
      | \bL[1-6]\b                        # ladder levels
      | \bS\d\b                           # stage refs S0..S3
      | \bF-1\b                           # feasibility table name
      | ^\s{0,3}\d+[.)]\s                 # ordered list marker
      | \#+\s*\d                          # markdown heading number
      | \bv?\d+\.\d+\.\d+\b               # version strings
      | https?://\S+                      # urls
      | \bEXP-[0-9a-fA-F]{6}\b            # exp ids
      | \b[A-Z]{2,4}-[0-9a-f]{6,10}\b     # ledger ids
    )""",
    re.VERBOSE,
)
DIGIT = re.compile(r"\d")


def is_reports_path(p: str) -> bool:
    p = (p or "").replace("\\", "/")
    return "/reports/" in p or p.startswith("reports/") or p == "reports"


def offending_lines(text: str):
    out = []
    in_code = False
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not DIGIT.search(line):
            continue
        if EXP_TAG.search(line) or ESCAPE_TAG.search(line):
            continue
        scrubbed = BENIGN.sub("", line)
        if DIGIT.search(scrubbed):
            out.append((i, line.rstrip()))
    return out


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return 0
    ti = event.get("tool_input") or {}
    path = ti.get("file_path", "")
    if not is_reports_path(path):
        return 0
    content = ti.get("content")
    if content is None:
        content = ti.get("new_string", "")
    bad = offending_lines(content or "")
    if bad:
        sys.stderr.write(
            "BLOCKED (R1): numeric claim(s) in reports/ without an [EXP-xxxxxx] tag:\n"
        )
        for ln, txt in bad[:20]:
            sys.stderr.write(f"  L{ln}: {txt}\n")
        sys.stderr.write(
            "Tag each metric with its ledger experiment id, or mark structural "
            "text with [SPEC]/[PLAN]/[NO-METRIC].\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
