# Binance Data Collector (Final Version) — README

This document explains what changed from the **previous** (spot-kline + futures depth WS + realtime liquidation only) collector to the **current final version** (futures-optimized + stablecoin filtering + liquidation history bootstrap + ±5% liquidity via REST).

---

## What changed (high level)

### 1) Universe selection (Top N /USDT, stablecoins excluded)
- The collector now builds its symbol list from `/ticker/24hr` **USDT pairs only** and excludes stablecoin base assets (USDC, DAI, FDUSD, etc.).
- The symbol list can **auto-refresh** (hourly), and matrices are rebuilt while preserving overlapping symbol history.

**Why:** stablecoins don’t matter for pump/dump detection, and you want a rolling “top N” universe.

---

### 2) Futures-first architecture (optional: futures-only klines)
- You can run klines from **futures endpoint** instead of spot (`wss://fstream.binance.com/...`) while keeping the same kline parsing logic.
- This lets you combine price candles + futures-only signals (funding/OI/liq) using one market source.

**Why:** futures markets are usually more liquid, and liquidation/funding/OI are critical for your detection system.

---

### 3) Added “full futures metrics” collection
A new futures metrics collector was added (hybrid REST + WebSocket):
- Funding rate + next funding time
- Mark price + index price + premium
- Open interest
- Long/short ratios (accounts, positions, global)
- Taker buy/sell ratio
- 24h ticker stats (price change %, volume, quote volume)

**Optimization included:**
- Switched some futures streams to **aggregate streams** (1 stream for all symbols) to reduce stream usage.

---

### 4) Liquidation system upgraded (history + realtime)
Previously: realtime liquidation stream only.

Now:
- A history loader downloads **last 7 days** at startup (from Binance daily files), parses them, and merges into the same in-memory history used by the realtime stream.
- Realtime continues via WebSocket.
- A daily updater checks for the next daily file and merges it (rolling window).

**Why:** you requested “bootstrap 7 days, then continue realtime, then reconcile when daily file arrives.”

---

### 5) Open orders / order book requirement implemented as ±5% liquidity
Your requirement was: “all open orders for each symbol within ±5% of price, updated ~15s (or 60s if rate-limited).”

**Final implementation approach:**
- Instead of maintaining 100–165 full order books by WebSocket (heavy), we added a `DepthLiquidityCollector` that:
  - uses REST depth snapshots (e.g., `limit=1000`)
  - calculates total **bid liquidity** and **ask liquidity** inside \([mid*0.95, mid*1.05]\)
  - writes results into `MarketMatrix` fields:
    - `liquidity_bid_5pct`
    - `liquidity_ask_5pct`

**Why:** it exactly matches the “±5% range” requirement and avoids 100–165 extra depth streams.

---

### 6) Removed old FuturesDepthManager (optional cleanup)
The old approach (`FuturesDepthManager`) maintained `depth20@100ms` books via WS.
- That is “fast but shallow” (top 20 levels rarely covers ±5%).
- After implementing `DepthLiquidityCollector`, you can safely remove it if not used elsewhere.

**Important:** if you removed `FuturesDepthManager`, you must also update any accessor that referenced `self.futures_depth` (e.g., `get_depth_liquidity_range`) to read from `MarketMatrix` instead.

---

## Current architecture (components)

### Symbol Manager
- Fetches top N USDT pairs by quoteVolume.
- Excludes stablecoin base assets.
- Hourly refresh (optional), rebuilds matrices preserving existing symbols’ history.

### MarketMatrix (core store)
For each timeframe:
- Stores OHLCV arrays (fixed depth)
- Stores indicator buffers (RSI, MFI, ADX, vol z-score)
- Stores futures metrics arrays
- Stores liquidity arrays (`liquidity_bid_5pct`, `liquidity_ask_5pct`)
- Exposes `get_analysis()` to return one fast dictionary snapshot for the bot layer.

### FuturesDataCollector
- WebSocket aggregate streams for:
  - mark price / funding
  - 24h ticker stats
- REST periodic calls for:
  - OI
  - ratios
  - taker ratio

### LiquidationMonitor + LiquidationHistoryLoader
- WebSocket realtime liquidations (`!forceOrder@arr`)
- Startup bootstrap: last 7 days from daily files
- Daily reconcile task
- Rolling windows for 1h / 24h stats (and/or your chosen retention)

### DepthLiquidityCollector (±5%)
- REST depth snapshots (rotating through symbols)
- Computes liquidity inside ±5% range of mid
- Writes liquidity into matrices

---

## Migration notes (previous → current)

### Persistence (“Why is it still 100 coins?”)
- If `.npz` state exists, startup may restore old symbols (e.g., 100).
- Use one of:
  - Delete `./data_storage/matrix_*.npz`
  - Run with `FORCE_REFRESH=1`

### Naming collisions (method vs attribute)
- Do NOT rename `get_top_100_coins()` to `top_coins()` because `self.top_coins` is already a list.
- Use names like: `fetch_top_coins()` / `refresh_top_coins()`.

---

## How to change the number of coins

Change only the slice in the symbol fetcher:

```py
new_top_coins = [x["symbol"] for x in usdt_pairs[:165]]
