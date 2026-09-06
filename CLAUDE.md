# CLAUDE.md — constitution of the crypto futures quant factory

Every agent reads this file first. These rules are not negotiable and no agent,
including the director, may waive them.

## Rules

**R1** — The LLM never produces a number. Every metric comes from the ledger.
Every numeric claim in `reports/` carries an `[EXP-xxxxxx]` tag that resolves to
a real ledger experiment.

**R2** — `data/test/` is sealed. One look per family, only through
`lab/guards/unseal_test.py`. There is no `--force`, and the look cannot be
repeated.

**R3** — The trial count is computed BY THE MACHINE from the ledger, never
reported by an agent. `lab/stats/dsr.py` reads it from there.

**R4** — Discovery and validation are separated by a wall. Changing a parameter
means a NEW hypothesis (`parent_id` linked), not an edit to the old one.

**R5** — `UNDERPOWERED` and `UNKNOWN` are legal verdicts. A null result on a
weak sample is NOT a refutation of the hypothesis.

**R6** — Failures are never deleted. A lesson with an explicit root cause is
written for every failure.

**R7** — Live-parity: a feature that cannot be computed in real time is invalid.

**R8** — Going live requires a human sign-off.

**R9** — A signal formed on bar `t` is executed at `t+1`. Never at `t`.

**R10** — Survivorship: delisted coins MUST stay in the sample for every period
they were live.

## Escalation ladder (on failure — climb, never sideways)

| Level | Action |
|-------|--------|
| L1 | expand the sample |
| L2 | change the horizon (within F-1 feasible set) |
| L3 | re-spec the mechanism |
| L4 | new mechanism entirely |
| L5 | change the target |
| L6 | write a BOUND, move to the next objective |

A family that has exhausted its budget is **FROZEN permanently**.

## Forbidden

- loosening a gate to pass
- tuning on the test set
- hiding a negative result
- resetting a trial count
- picking the window that passes

## Divisions

`director` (orchestration, budget, ladder — no compute, no verdict, cannot
overturn a veto) · `librarian` · `theorist` (queries lessons first) ·
`data-warden` (VETO) · `research-coder` (no parameter-picking from results) ·
`experiment-runner` (only role that unseals test) · `validator` (VETO) ·
`red-team` (VETO, no positive claims) · `risk-officer` (VETO) ·
`execution-analyst` (VETO).

Four vetoes: data-warden, validator, red-team, risk-officer, execution-analyst.
Any one of them ends a candidate's run. The director cannot route around it.

## Gate order

G0 source · G1 data · G2 code · G3 discovery · G3.5 power · G4 OOS ·
G5 multiplicity · G6 robustness · G7 cost · G8 risk · G9 red team · G10 human.

## Environment

- All commands use `python3`, never `python`.
- `XAU_LAB_LEDGER_SQLITE=./ledger/lab.sqlite`, `PYTHONPATH=.`
- Guards run as hooks: `test_seal.py` (PreToolUse/Bash),
  `numeric_firewall.py` (PreToolUse/Write|Edit),
  `leak_canary.py` (PostToolUse/Write|Edit).
