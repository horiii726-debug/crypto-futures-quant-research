#!/usr/bin/env python3
"""PreToolUse(Bash) hook: block any shell command that touches data/test/.

R2  data/test/ is sealed. One look per family, only via guards/unseal_test.py.

Reads a JSON hook event on stdin: {"tool_input": {"command": "..."}}.
Exit 0 = allow, exit 2 = block.
"""
import json
import re
import sys

# Path fragments that indicate the sealed partition. Matches ./data/test/,
# data/test/, /root/crypto-lab/data/test/, quoted or not.
SEALED = re.compile(r"(^|[\s\"'=(/])data/test(/|\"|'|\s|$)")

# The only sanctioned tool that may name the path.
ALLOW_TOOLS = ("guards/unseal_test.py", "lab/guards/unseal_test.py")


def main() -> int:
    raw = sys.stdin.read()
    try:
        event = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        # Cannot parse -> fail closed only if the literal path appears.
        event = {"tool_input": {"command": raw}}

    cmd = (event.get("tool_input") or {}).get("command", "") or ""

    if any(tok in cmd for tok in ALLOW_TOOLS):
        return 0

    if SEALED.search(cmd) or "data/test" in cmd.replace("\\", "/"):
        sys.stderr.write(
            "BLOCKED (R2): command references data/test/. The test partition is "
            "sealed. Use `python3 lab/guards/unseal_test.py` (one look per family, "
            "logged to ledger, no --force).\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
