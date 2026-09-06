---
description: One full research cycle across every division
---

**director** drives; every division acts in order. No number is produced by an
LLM - all metrics come from ledger experiments (R1).

1. **director**: pick family + objective. Check `config/budgets.yaml`. Query
   lessons. Refuse if the family is FROZEN or out of budget.
2. **librarian**: sources for the objective. Identifiers must resolve.
   Falsification search. UNRETRIEVED papers are not summarised.
3. **theorist**: hypothesis record (validate against
   `schemas/hypothesis.schema.json`). MUST have queried lessons.
4. **data-warden** (G1, veto): partitions, hashes, survivorship, quality gate.
   May declare UNDERPOWERED.
5. **research-coder** (G2): implement the feature + the pre-registered grid.
   `kernels.py` IDENTICAL; `leak_canary` clean.
6. **experiment-runner** (G3, G3.5): `run_grid` on train/valid - every fit
   logged before its result. No test look yet.
7. **validator** (G4, G5, G6, veto): power, purged CV, DSR (ledger trial
   count), PBO, MCS/SPA/DM, walk-forward.
8. **red-team** (G9, veto): placebo, leakage hunt, cost stress, surrogate.
9. **risk-officer** (G8, veto): drawdown, tail, P(breach), P(ruin), Monte
   Carlo, crash-correlation stress.
10. **execution-analyst** (G7, veto): cost floor, F-1 feasibility.
11. If all gates PASS on `valid`: **experiment-runner** unseals `test` once for
    this family (human sign-off), runs the frozen config, logs it.
12. **director**: if any FAIL/VETO -> `/postmortem` then climb the ladder.
    A VETO cannot be overturned.
