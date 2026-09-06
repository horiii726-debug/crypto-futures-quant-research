# crypto-lab

A crypto-futures quantitative research factory built around one idea: **the
process, not the researcher, decides what is real.** The LLM agents orchestrate
and reason; every number comes from an append-only ledger of experiments.

## Layout

```
CLAUDE.md          constitution - every agent reads it first
ROADMAP.md         S0 → S1 → S2 → S3 stages
config/            gates, budgets, venue (null on purpose), universe, risk
schemas/           hypothesis + verdict JSON schemas
lab/
  ledger.py        append-only SQLite; UPDATE/DELETE rejected by triggers
  make_synth.py    synthetic multi-coin OHLCV (vol clustering, jumps,
                   8h funding, crash-correlation surge); --edge 0 / 0.35
  guards/          test_seal, numeric_firewall, unseal_test, leak_canary,
                   ledger_checkpoint
  stats/           dsr (trial count from ledger), pbo (CSCV), mcs (+SPA+DM),
                   power (overlap+autocorr), surrogate (block-bootstrap shuffle)
  engine/          kernels (numba 3-tier, bit-identical), labels (triple
                   barrier + uniqueness), cv (purged k-fold + embargo + CPCV),
                   backtest (portfolio, cost-aware, t→t+1), signals, runner
  features/  data/  rag/
data/{raw,processed,features,train,valid,test}   test/ is SEALED
ledger/  reports/  paper/  live/  archive/
tests/test_lab.py  10 mandatory checks
.claude/           10 tool-scoped agents, 3 commands, settings + hooks
```

## Quick start

```bash
export PYTHONPATH=. XAU_LAB_LEDGER_SQLITE=./ledger/lab.sqlite
python3 tests/test_lab.py                 # 10/10 must pass
python3 lab/engine/kernels.py             # numba IDENTICAL
python3 tests/calibration.py              # edge0 not a candidate; edge0.35 is
```

## Rules that matter most

R1 no LLM numbers · R2 sealed test · R3 machine trial count · R9 t→t+1 ·
R10 survivorship. See `CLAUDE.md`.

Status: **S0 complete.** S1/S2 await explicit confirmation.
