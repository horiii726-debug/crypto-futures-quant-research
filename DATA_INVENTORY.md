# DATA INVENTORY

**9.5 GB on disk, none of it committed** (`.gitignore`) — GitHub caps files at
100 MB and the whole set is regenerable for free from `data.binance.vision`
(no API key) and the Bybit public API. The downloader for every dataset below
is in this repo.

## Price

| dataset | coverage | period | size | built by |
|---|---|---|---|---|
| `data/processed/klines_1m/` | **55 coins**, 1-minute OHLCV | 2023-09 → 2025-08 | 2.4 GB | `lab/data/binance_vision.py` |
| `data/processed/long_panel/` | **62 coins**, 1h OHLCV, delisted retained | **2020-01 → 2025-06 (5.49 y)** | 96 MB | `lab/data/long_panel.py` |
| `data/{train,valid,test}/prio50_2y/1h/` | 55 coins, 1h research panel (sealed test split) | 2023-09 → 2025-08 | — | `lab/research/main_s1_s7.py` |
| `data/processed/quarterly/` | BTC/ETH front-quarterly futures + annualised basis | 2021-02 → 2025-06 | 3.1 MB | `lab/data/quarterly.py` |

## TAPE (aggTrades → microstructure bars)

| | |
|---|---|
| location | `data/processed/tape/{COIN}_{3,5,15}m.parquet` |
| coverage | **20 coins**, 2023-09 → 2025-08 (2 y), 50 files, 836 MB |
| coins | BTC ETH BNB SOL XRP ADA DOGE AVAX LINK LTC UNI NEAR ARB ZEC SUI AAVE INJ WLD TAO 1000PEPE |
| 16 columns | `ofi_usd` `buy_frac` `aggr_imb` `vwap` `kyle_lambda` `roll_spread_bps` `trade_sign_ac1` `big_trade_imb` `realised_vol` `notional` `n_trades` + OHLC |
| built by | `lab/data/tape.py` + `rustkernels/src/tape.rs` (~30× faster) |
| raw | aggTrades zips deleted after processing (~31 GB) — regenerate with `lab/data/_mass_download.py tape` |

## DOM (bookDepth → order-book bars)

| | |
|---|---|
| processed | `data/processed/bookdepth_bars/{COIN}_{5,15}m.parquet` — **18 coins**, 2023-01 → 2024-05, 36 files, 259 MB |
| 8 columns | `depth_imb_1pct` `depth_imb_5pct` `imb_vol` `book_slope` `total_depth` `microprice_tilt` `depth_withdraw` `n_snap` |
| daily | `data/processed/bookdepth_daily/` — 30 coins |
| **raw ladder** | `data/raw/data/futures/um/daily/bookDepth/` — **7 245 zips**, 28 coins × ~260 days, ±1%…±5% cumulative bands, 30 s snapshots |
| ladder summary | `data/processed/book_ladder.json` — per-coin median depth at each band (used for walk-the-book cost) |
| built by | `lab/data/bookdepth.py` · `lab/data/book_ladder.py` · `rustkernels/src/book.rs` |
| limitation | bookDepth carries **bands, not per-level L2** — no touch quote, no queue position |

## Open Interest

| dataset | coverage | period | source |
|---|---|---|---|
| `data/processed/bybit_oi/` | **41 coins**, hourly OI | 2023-08 → 2025-08 | Bybit public API — `lab/data/_collect_bybit_oi.py` |
| `data/processed/binance_oi_hist/` | **23 coins**, hourly OI **in USD** | **2020-09 → 2023-10** | Binance vision `metrics` — `lab/data/binance_metrics.py` |
| `data/processed/multi_venue/` | 23 coins, normalised close/volume/OI, one UTC grid | 2021-01 → 2023-09 | `lab/data/multi_venue.py` |

## Funding / Basis

| dataset | coverage | period | source |
|---|---|---|---|
| `data/processed/basis/` | **55 coins** — `funding` `premium` `mark_close` `index_close` `perp_close` | 2023-09 → 2025-08 | `lab/data/binance_vision.py` |
| `data/processed/bybit_funding/` | 55 coins, 8h funding | ~2 y | `lab/data/_collect_bybit_funding.py` |

## Liquidation

| | |
|---|---|
| processed | `data/processed/liqsnap/{BTCUSD_PERP,ETHUSD_PERP}.parquet` — hourly `liq_long_usd` `liq_short_usd` `liq_total_usd` `liq_imb` |
| period | 2023-06 → 2024-09 |
| raw | 899 zips, `data/raw/data/futures/cm/daily/liquidationSnapshot/` |
| built by | `lab/data/liqsnap.py` |
| limitation | Binance publishes coin-M `liquidationSnapshot` for **BTC and ETH only** → 2 correlated series, structurally underpowered |

## Execution calibration (small, **committed**)

| file | what |
|---|---|
| `data/processed/exec_calibration.json` | per-coin `p_fill`, adverse selection, queue depth — simulated on the real trade tape |
| `data/processed/exec_summary.json` | hybrid round-trip 8.53 bps mean |
| `data/processed/spread_measured.json` / `spread_extra.json` / `aggtrades_micro.json` | tick-level Roll half-spread, 25 coins |
| `data/processed/bookdepth_features.json` · `bookticker_stats.json` | near-touch depth, L1 quoted spread (5 coins) |
| `data/raw/_spec/*.json` | exchangeInfo, fundingInfo, leverage brackets, universe pool |

## Regenerate everything

```bash
export PYTHONPATH=.
python3 lab/data/long_panel.py          # 5.49y price panel (62 coins)
python3 lab/data/binance_metrics.py     # OI in USD back to 2020-09
python3 lab/data/multi_venue.py         # normalise + Bybit-vs-Binance OI validation
python3 lab/data/_mass_download.py tape # aggTrades -> tape bars (download, process, delete raw)
python3 lab/data/_mass_download.py book # bookDepth -> DOM bars
python3 lab/data/book_ladder.py         # ±1..±5% depth ladder for walk-the-book cost
python3 lab/data/liqsnap.py             # coin-M liquidation bars
python3 lab/data/quarterly.py           # front-quarterly futures basis
python3 lab/exec/universe.py            # rolling liquidity filter
```

All free. No API key needed for historical data (keys are only for live trading,
see `config/secrets.example.yaml`).

## Known data gaps (these are what bound the research)

| missing | consequence |
|---|---|
| **live L2 order book** (per-level, touch, queue) | every microstructure bound says "the edge is at the touch, we cannot see it" |
| bookTicker ends ~2024-04, bookDepth ends ~2024-05 | DOM research window is only ~1.4 y |
| Binance REST OI history ~30 days | long OI history only exists via the `metrics` bulk files (2020-09+) |
| coin-M liquidation = BTC/ETH only | F_LIQD structurally underpowered |
| no options / vol surface | the one major mechanism class never tested |
| no on-chain / stablecoin supply | F069 not testable |
