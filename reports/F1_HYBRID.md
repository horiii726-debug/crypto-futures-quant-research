# F-1 (maker/hybrid execution) — measured costs

[SPEC] Universe `prio50_2y`. Taker round-trip 11.71 bps vs **hybrid round-trip 8.53 bps** (limit-at-touch, p_fill and adverse selection simulated on the real aggTrades tape, taker fallback after the wait). NO 100% maker-fill assumption.

| horizon | taker floor (bps) | hybrid floor (bps) | disp D (bps) | req IC taker | req IC hybrid | FEASIBLE taker | FEASIBLE hybrid |
|---|---|---|---|---|---|---|---|
| 5m | 12.21 | 9.04 | 17.4 | 0.8404 | 0.6217 | no | no |
| 15m | 12.23 | 9.05 | 30.2 | 0.4857 | 0.3594 | no | no |
| 1h | 12.28 | 9.1 | 60.4 | 0.2439 | 0.1808 | no | no |
| 4h | 12.5 | 9.32 | 122.6 | 0.1224 | 0.0913 | no | no |
| 1d | 13.96 | 10.79 | 313.4 | 0.0535 | 0.0413 | no | **yes** |
| 3d | 17.47 | 14.3 | 566.8 | 0.037 | 0.0303 | yes | **yes** |

**Feasible (taker-only):** 3d
**Feasible (hybrid):** 1d, 3d
**Reopened by hybrid execution:** 1d