#!/usr/bin/env python3
"""Detect [EXP-xxxxxx] ids cited in reports/ (or anywhere) that do not exist
in the ledger.

R1  Every numeric claim in reports/ carries a real ledger tag.
Pairs with numeric_firewall.py: the firewall enforces that a tag is present,
this enforces that the tag resolves.

Usage:
    python3 lab/guards/ledger_checkpoint.py [paths...]        # default: reports/
Exit 0 = all cited ids resolve, exit 2 = dangling id found.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP_RE = re.compile(r"\[?(EXP-[0-9a-fA-F]{6})\]?")


def main(argv) -> int:
    sys.path.insert(0, str(ROOT))
    from lab.ledger import Ledger

    L = Ledger()
    targets = [Path(p) for p in argv] or [ROOT / "reports"]

    cited: dict[str, list[str]] = {}
    for base in targets:
        files = [base] if base.is_file() else list(base.rglob("*.md")) + list(base.rglob("*.txt"))
        for f in files:
            try:
                txt = f.read_text()
            except OSError:
                continue
            for m in EXP_RE.finditer(txt):
                cited.setdefault(m.group(1), []).append(str(f.relative_to(ROOT)))

    dangling = {eid: locs for eid, locs in cited.items() if not L.has_experiment(eid)}
    if dangling:
        sys.stderr.write("DANGLING EXP ids cited but absent from ledger (R1):\n")
        for eid, locs in dangling.items():
            sys.stderr.write(f"  {eid}  in  {', '.join(sorted(set(locs)))}\n")
        return 2

    print(f"ledger_checkpoint OK: {len(cited)} distinct EXP id(s) cited, all resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
