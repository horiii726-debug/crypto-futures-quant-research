---
description: Write a lesson with root cause after a candidate fails
---

A candidate just failed or was vetoed. Failures are never deleted (R6).

1. Identify the subject (HYP-... / EXP-...) and the gate that stopped it.
2. Root-cause it. Not "it didn't work" - *why*:
   - wrong mechanism? mechanism right but effect too small (UNDERPOWERED)?
   - leakage found by red-team? cost floor too high for the horizon?
   - overfit (PBO high / DSR low given the ledger trial count)?
   - regime dependence? survivorship artefact?
3. Decide the ladder level for the next attempt (L1..L6). Never sideways.
   If the family is out of budget: write a BOUND, mark it FROZEN.
4. Write it:
   `python3 -c "from lab.ledger import Ledger; L=Ledger(); L.add_lesson(root_cause='...', lesson='...', family='F_XXX', subject_id='HYP-...', ladder_level='L3')"`
5. If a bound: `L.add_bound(family, objective, bound_stmt, evidence_exp_ids)`
   with explicit numbers (tagged) describing what is now ruled out.
6. The theorist MUST see this lesson before the next proposal in this family.
