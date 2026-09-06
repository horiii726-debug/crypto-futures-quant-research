# S0 — Foundation verification

[NO-METRIC] Status recorded on stage completion. Figures below are test-suite
diagnostics, not research metrics; research metrics only ever come from ledger
experiments tagged [EXP-xxxxxx] (R1).

## Verification gates — all green

| Check | Command | Result |
|-------|---------|--------|
| Test suite | `python3 tests/test_lab.py` | [SPEC] 10 / 10 passed |
| Kernel identity | `python3 lab/engine/kernels.py` | numba IDENTICAL (3 tiers: python / numpy / numba) |
| Test seal | `echo '{...cat data/test/x...}' \| python3 lab/guards/test_seal.py` | exit [SPEC] 2 (blocked) |
| Calibration | `python3 tests/calibration.py` | PASS |
| Venue block | `run_backtest` with `venue.yaml` costs | `VenueNotConfigured` raised — pipeline blocked on purpose |

## Calibration — control vs planted

| Dataset | Verdict | Why |
|---------|---------|-----|
| control, `--edge 0` | NOT A CANDIDATE | net Sharpe < 0; loses ~transaction cost; does not clear surrogate q99 noise ceiling; DSR ~ 0 |
| planted, `--edge 0.35` | CANDIDATE | net Sharpe > 0; clears noise ceiling; DSR ≥ family threshold; trial count read from ledger |

No leak: the zero-edge control does not pass as a candidate.

## venue.yaml

[SPEC] status: `UNCONFIGURED`. Every required field (`maker_fee`, `taker_fee`,
funding interval / cap, tick size, min notional, leverage tiers, liquidation
rule, `slippage_bps`) is `null`. This blocks the backtest deliberately until S1
fills them from the official Binance USDT-M spec.

## Notes / limitations of the scaffold

- Guards are wired as Claude Code hooks in `.claude/settings.json`; they enforce
  only inside a session that loads that file.
- `lab/stats` implementations (DSR deflation benchmark, SPA recentring, MCS
  range statistic) are working reference implementations calibrated by their
  selftests; they should get a second review before any live decision.
- `lab/features/{price,xsec,flow,funding}.py` are on the signal whitelist but
  not yet written — that is S3 work. The calibration pipeline uses the built-in
  causal primitives in `lab/engine/signals.py`.
- No market data downloaded; no literature retrieved (S0 constraint).
