import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lab.ledger import Ledger
from lab.research import run_micro


def main():
    L = Ledger()
    r = run_micro.run(L, max_full_per_family=45)
    out = {
        "survivors": [s.get("name", s.get("feature")) for s in r["survivors"]],
        "n_full_results": len(r["results"]),
        "family_full_trials": r["family_full_trials"],
        "screen_evals": r["screen_evals"],
        "verdicts": r["verdicts"],
    }
    (ROOT / "reports" / "MICRO_RESULT.json").write_text(json.dumps(out, indent=1, default=float))
    print("MICRO DONE survivors=", len(r["survivors"]))


if __name__ == "__main__":
    main()
