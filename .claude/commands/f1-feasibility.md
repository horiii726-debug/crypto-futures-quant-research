---
description: Build the F-1 table of feasible horizons from measured costs
---

Run the **execution-analyst** to produce `reports/F1_CRYPTO.md`.

Steps:
1. Verify `config/venue.yaml` status is `configured` (no null required field).
   If still null, STOP - report that the pipeline is correctly blocked.
2. Measure the cost floor per horizon from `data/raw` bookTicker + aggTrades:
   fee + funding-over-hold + slippage + latency cost.
3. For horizons 5m, 15m, 1h, 4h, 1d, 3d compute required IC and required hit
   rate to clear the cost floor at the universe's median liquidity.
4. Every number tagged `[EXP-xxxxxx]` from a real ledger experiment.
5. Mark each horizon FEASIBLE or INFEASIBLE. INFEASIBLE horizons are closed:
   record a bound in the ledger.
6. Print the table and the list of feasible horizons.
