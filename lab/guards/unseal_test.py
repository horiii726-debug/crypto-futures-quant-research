#!/usr/bin/env python3
"""The ONLY sanctioned path to the sealed test partition.

R2  data/test/ is sealed. One look per family, only through this script.
R3  The look is recorded to the ledger; DSR and multiplicity accounting see it.
R8  Requires a human sign-off token.

There is deliberately NO --force flag and the look CANNOT be repeated: the
ledger's test_looks table raises if a family already has a row.

Usage:
    python3 lab/guards/unseal_test.py \
        --family F_XSEC --exp EXP-abc123 \
        --approved-by "<human token / name>" \
        --note "final OOS read for surviving candidate"

On success it prints the unseal token and the list of files under data/test/.
Reading the files themselves is up to the caller (experiment-runner only).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIR = ROOT / "data" / "test"


def dir_sha256(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(d).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Unseal data/test/ once per family.")
    ap.add_argument("--family", required=True)
    ap.add_argument("--exp", required=True, help="EXP-xxxxxx that consumes the look")
    ap.add_argument("--approved-by", required=True, help="human sign-off token (R8)")
    ap.add_argument("--note", default="")
    # No --force. Intentionally absent.
    args = ap.parse_args()

    if not args.approved_by.strip():
        sys.stderr.write("REFUSED: empty --approved-by. R8 requires human sign-off.\n")
        return 2

    sys.path.insert(0, str(ROOT))
    from lab.ledger import Ledger

    L = Ledger()
    if not L.has_experiment(args.exp):
        sys.stderr.write(
            f"REFUSED: {args.exp} is not in the ledger. Register the experiment "
            f"before consuming a test look (R1/R3).\n"
        )
        return 2

    if L.test_looks_used(args.family) >= 1:
        sys.stderr.write(
            f"REFUSED (R2): family {args.family} has already spent its single test "
            f"look. This cannot be repeated and there is no override.\n"
        )
        return 2

    sha = dir_sha256(TEST_DIR) if TEST_DIR.exists() else "EMPTY"
    try:
        tl = L.record_test_look(args.family, args.exp, sha, args.approved_by, args.note)
    except PermissionError as e:
        sys.stderr.write(f"REFUSED: {e}\n")
        return 2

    print(f"UNSEALED family={args.family} look_id={tl} dataset_sha={sha}")
    files = sorted(p.relative_to(ROOT).as_posix()
                   for p in TEST_DIR.rglob("*") if p.is_file())
    for f in files:
        print("  " + f)
    if not files:
        print("  (test partition is empty - nothing downloaded yet, S1 not run)")
    print("This look is now permanently recorded. The family cannot look again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
