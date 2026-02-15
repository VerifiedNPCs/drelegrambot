# Data_Collector.py
import asyncio
import json
import os
import gc
import logging
import psutil
import aiohttp
import numpy as np
import traceback
import time
import threading
from datetime import datetime
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict 
import zipfile
import csv
from io import BytesIO, StringIO
from datetime import datetime, timedelta
from config import config, setup_logging

logger = setup_logging("DataCollector")
STABLECOINS = {
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD', 
    'USDD', 'USDP', 'FRAX', 'USDB', 'USDE'
}
# ==============================================================================
# ⚡ NUMBA JIT ENGINE (Zero-Allocation Optimization)
# ==============================================================================
def env_flag(name: str, default: str = "true") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

@njit(parallel=True, fastmath=True, cache=True)
def calc_indicators_inplace(
    closes, volumes, highs, lows, 
    out_rsi, out_mfi, out_adx, out_vol_z, out_atr,
    period_rsi=14, period_mfi=14, period_adx=14, period_atr=14
):
    """
    Writes results directly into pre-allocated output arrays.
    Prevents memory allocation churn and GC pauses.
    """
    n_symbols, n_cols = closes.shape

    for i in prange(n_symbols):
        row_c = closes[i]
        row_v = volumes[i]
        row_h = highs[i]
        row_l = lows[i]

        if row_c[-1] == 0: continue

        # Find first valid data point
        valid_start = 0
        for k in range(n_cols):
            if row_c[k] != 0:
                valid_start = k
                break

        if n_cols - valid_start < 30: continue

        # --- 1. RSI Calculation ---
        avg_gain = 0.0
        avg_loss = 0.0

        for j in range(valid_start + 1, valid_start + period_rsi + 1):
            if j >= n_cols: break
            change = row_c[j] - row_c[j - 1]
            if change > 0: avg_gain += change
            else: avg_loss -= change

        avg_gain /= period_rsi
        avg_loss /= period_rsi

        for j in range(valid_start + period_rsi + 1, n_cols):
            change = row_c[j] - row_c[j - 1]
            if change > 0:
                avg_gain = (avg_gain * (period_rsi - 1) + change) / period_rsi
                avg_loss = (avg_loss * (period_rsi - 1)) / period_rsi
            else:
                avg_gain = (avg_gain * (period_rsi - 1)) / period_rsi
                avg_loss = (avg_loss * (period_rsi - 1) - change) / period_rsi

        if avg_loss == 0:
            out_rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out_rsi[i] = 100.0 - (100.0 / (1.0 + rs))

        # --- 2. MFI Calculation ---
        pos_flow = 0.0
        neg_flow = 0.0
        start_mfi = max(valid_start + 1, n_cols - period_mfi)

        for j in range(start_mfi, n_cols):
            tp_curr = (row_h[j] + row_l[j] + row_c[j]) / 3.0
            tp_prev = (row_h[j - 1] + row_l[j - 1] + row_c[j - 1]) / 3.0
            raw_flow = tp_curr * row_v[j]

            if tp_curr > tp_prev: pos_flow += raw_flow
            elif tp_curr < tp_prev: neg_flow += raw_flow

        if neg_flow == 0:
            out_mfi[i] = 100.0
        else:
            mfi_ratio = pos_flow / neg_flow
            out_mfi[i] = 100.0 - (100.0 / (1.0 + mfi_ratio))

        # --- 3. ADX Calculation ---
        tr_sum = 0.0
        dm_pos_sum = 0.0
        dm_neg_sum = 0.0
        start_adx = max(valid_start + 1, n_cols - period_adx)
        
        for j in range(start_adx, n_cols):
            up = row_h[j] - row_h[j - 1]
            down = row_l[j - 1] - row_l[j]
            pdm = up if (up > down and up > 0) else 0.0
            ndm = down if (down > up and down > 0) else 0.0
            tr = max(row_h[j] - row_l[j], abs(row_h[j] - row_c[j - 1]), abs(row_l[j] - row_c[j - 1]))
            
            tr_sum += tr
            dm_pos_sum += pdm
            dm_neg_sum += ndm

        if tr_sum > 0:
            pdi = 100 * dm_pos_sum / tr_sum
            ndi = 100 * dm_neg_sum / tr_sum
            sum_di = pdi + ndi
            out_adx[i] = 100 * abs(pdi - ndi) / sum_di if sum_di > 0 else 0.0
        else:
             out_adx[i] = 0.0

        # --- 4. Volume Z-Score ---
        count = 20
        start_v = max(valid_start, n_cols - count)
        v_sum = 0.0
        v_sq_sum = 0.0
        actual_count = 0

        for j in range(start_v, n_cols - 1): 
            val = row_v[j]
            v_sum += val
            v_sq_sum += val * val
            actual_count += 1

        if actual_count > 5:
            v_mean = v_sum / actual_count
            v_var = (v_sq_sum / actual_count) - (v_mean * v_mean)
            v_std = np.sqrt(v_var) if v_var > 0 else 1.0
            out_vol_z[i] = (row_v[-1] - v_mean) / v_std
        else:
            out_vol_z[i] = 0.0

        # --- 5. ATR Calculation (Wilder's Smoothing) ---
        # First, calculate the initial SMA for the first period
        start_atr = valid_start + 1
        atr_val = 0.0
        
        # 1. Calculate SMA for the first 'period_atr' elements to seed the RMA
        if n_cols - start_atr > period_atr:
            tr_sum = 0.0
            for j in range(start_atr, start_atr + period_atr):
                high = row_h[j]
                low = row_l[j]
                prev_close = row_c[j-1]
                
                hl = high - low
                hc = abs(high - prev_close)
                lc = abs(low - prev_close)
                
                tr = hl
                if hc > tr: tr = hc
                if lc > tr: tr = lc
                tr_sum += tr
            
            atr_val = tr_sum / period_atr

            # 2. Apply Wilder's Smoothing (RMA) for the rest
            for j in range(start_atr + period_atr, n_cols):
                high = row_h[j]
                low = row_l[j]
                prev_close = row_c[j-1]
                
                hl = high - low
                hc = abs(high - prev_close)
                lc = abs(low - prev_close)
                
                tr = hl
                if hc > tr: tr = hc
                if lc > tr: tr = lc
                
                # ATR = ((Prior ATR * (n-1)) + Current TR) / n
                atr_val = ((atr_val * (period_atr - 1)) + tr) / period_atr
        
        out_atr[i] = atr_val



# ==============================================================================
# 💀 LIQUIDATION MONITOR (Full History & Stats)
# ==============================================================================
class LiquidationHistoryLoader:
    def __init__(self, liq_monitor: 'LiquidationMonitor', symbols: List[str]):
        self.liq_monitor = liq_monitor
        self.symbols = set(s.upper() for s in symbols)
        self.base_url = "https://data.binance.vision/data/futures/um/daily/liquidationSnapshot"
        self.running = True
        self.last_loaded_date = None
         
    
    async def download_and_parse_day(self, session: aiohttp.ClientSession, date_str: str) -> int:
        loaded_count = 0

        for symbol in self.symbols:
            try:
                filename = f"{symbol}-liquidationSnapshot-{date_str}.zip"
                url = f"{self.base_url}/{symbol}/{filename}"

                async with session.get(url, timeout=30) as resp:
                    if resp.status == 404:
                        continue
                    if resp.status != 200:
                        logger.warning(f"Failed to download {symbol} liq data for {date_str}: HTTP {resp.status}")
                        continue

                    zip_data = await resp.read()

                with zipfile.ZipFile(BytesIO(zip_data)) as z:
                    csv_filename = f"{symbol}-liquidationSnapshot-{date_str}.csv"
                    if csv_filename not in z.namelist():
                        continue

                    with z.open(csv_filename) as csvfile:
                        csv_content = csvfile.read().decode('utf-8')
                        reader = csv.DictReader(StringIO(csv_content))

                        for row in reader:
                            try:
                                ts = float(row['time']) / 1000.0
                                sym = row['symbol']
                                side = row['side']
                                qty = float(row['original_quantity'])
                                price = float(row['price'])
                                usd_val = qty * price

                                with self.liq_monitor._lock:
                                    self.liq_monitor.history.append((ts, sym, side, usd_val, price))

                                loaded_count += 1
                            except (KeyError, ValueError, TypeError):
                                continue

            except asyncio.TimeoutError:
                logger.warning(f"Timeout downloading {symbol} liq data for {date_str}")
            except Exception as e:
                logger.debug(f"Error loading {symbol} liq for {date_str}: {e}")

        return loaded_count



    async def load_last_7_days(self):
        '''
        Downloads last 7 days of liquidation history at startup.
        '''
        logger.info("💀 Loading 7-day liquidation history...")

        today = datetime.utcnow()
        total_loaded = 0

        async with aiohttp.ClientSession() as session:
            for days_ago in range(7, 0, -1):  # 7 days ago to yesterday
                target_date = today - timedelta(days=days_ago)
                date_str = target_date.strftime("%Y-%m-%d")

                logger.info(f"📥 Downloading liquidations for {date_str}...")      
                count = await self.download_and_parse_day(session, date_str)       
                total_loaded += count

                # Small delay to avoid rate limits
                await asyncio.sleep(0.5)

        # After loading, recalculate stats from scratch
        await self.liq_monitor._cleanup_loop_once()

        logger.info(f"✅ Loaded {total_loaded:,} historical liquidations (7 days)")

        # Set last loaded date to yesterday
        self.last_loaded_date = (today - timedelta(days=1)).strftime("%Y-%m-%d")   


    async def daily_update_task(self):
        '''
        Checks daily at 00:05 UTC if new historical file is available.
        Downloads it and merges with existing data.
        '''
        logger.info("💀 Daily liquidation update task started (checks at 00:05 UTC)")

        while self.running:
            try:
                now = datetime.utcnow()

                # Calculate seconds until next 00:05 UTC
                tomorrow = now + timedelta(days=1)
                next_check = tomorrow.replace(hour=0, minute=5, second=0, microsecond=0)

                # If we're past 00:05 today, check today's file
                today_check = now.replace(hour=0, minute=5, second=0, microsecond=0)
                if now > today_check:
                    next_check = today_check

                # Calculate wait time
                wait_seconds = (next_check - now).total_seconds()

                if wait_seconds > 0:
                    logger.debug(f"Next liq history update in {wait_seconds/3600:.1f} hours")
                    await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour max
                    continue

                # Time to check for new file
                yesterday = now - timedelta(days=1)
                date_str = yesterday.strftime("%Y-%m-%d")

                # Skip if already loaded
                if date_str == self.last_loaded_date:
                    await asyncio.sleep(3600)  # Check again in 1 hour
                    continue

                logger.info(f"📥 Checking for new liquidation file: {date_str}")   

                async with aiohttp.ClientSession() as session:
                    count = await self.download_and_parse_day(session, date_str)   

                if count > 0:
                    logger.info(f"✅ Loaded {count:,} new liquidations from {date_str}")
                    self.last_loaded_date = date_str

                    # Trigger cleanup to recalculate windows
                    await self.liq_monitor._cleanup_loop_once()
                else:
                    logger.info(f"No new liquidation data found for {date_str}")   

                # Wait 1 hour before checking again
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Daily liq update error: {e}")
                await asyncio.sleep(300)  # Retry in 5 min


    async def start(self):
        '''
        Main entry point: Load history, then start daily updater.
        '''
        await self.load_last_7_days()
        await self.daily_update_task()
class LiquidationMonitor:
    def __init__(self):
        self.url = "wss://fstream.binance.com/ws/!forceOrder@arr"
        self.running = True
        self._lock = threading.Lock()
        
        # History: Stores raw events for the last 24h
        # Format: deque([(timestamp, symbol, side, usd_value, price), ...])
        from collections import deque
        self.history = deque()
        
        # Fast Lookup Cache: { "BTCUSDT": { "1h_long": 0.0, "1h_short": 0.0, ... } }
        self.symbol_stats = defaultdict(lambda: {
            "1h_long": 0.0, "1h_short": 0.0, 
            "24h_long": 0.0, "24h_short": 0.0,
            "last_price": 0.0
        })

    async def start(self):
        """Main asyncio loop to connect and listen."""
        logger.info("💀 Liquidation Monitor Started")
        
        # Start cleanup task in background
        asyncio.create_task(self._cleanup_loop())

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(self.url) as ws:
                        logger.info("💀 Liq Stream Connected")
                        async for msg in ws:
                            if not self.running: break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    self._process_event(data)
                                except Exception:
                                    pass
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break
            except Exception as e:
                logger.error(f"Liq Stream Disconnected: {e}")
                await asyncio.sleep(5)
    async def _cleanup_loop_once(self):
        '''
        Manual trigger for cleanup (used by history loader).
        Same logic as _cleanup_loop but runs once.
        '''
        now = time.time()
        cutoff_24h = now - 86400
        cutoff_1h = now - 3600

        with self._lock:
            # 1. Prune old history
            while self.history and self.history[0][0] < cutoff_24h:
                self.history.popleft()

            # 2. Re-calculate ALL stats from scratch
            new_stats = defaultdict(lambda: {
                "1h_long": 0.0, "1h_short": 0.0,
                "24h_long": 0.0, "24h_short": 0.0,
                "last_price": 0.0
            })

            # Copy last prices from old stats
            for sym, old_s in self.symbol_stats.items():
                new_stats[sym]["last_price"] = old_s["last_price"]

            for ts, sym, side, usd, price in self.history:
                s = new_stats[sym]
                if side == 'SELL':
                    s["24h_long"] += usd
                    if ts > cutoff_1h: s["1h_long"] += usd
                else:
                    s["24h_short"] += usd
                    if ts > cutoff_1h: s["1h_short"] += usd

            self.symbol_stats = new_stats

        logger.debug(f"Liq stats recalculated: {len(self.history)} events in memory")

    def _process_event(self, data):
        payload = data.get('o', {})
        if not payload:
            return

        try:
            symbol = payload.get('s')
            side = payload.get('S')  # SELL = Long liq, BUY = Short liq
            qty = float(payload.get('q', 0.0))
            price = float(payload.get('p', 0.0))
            ts = float(payload.get('T', time.time() * 1000)) / 1000.0

            if not symbol or side not in ('BUY', 'SELL'):
                return

            usd_val = qty * price

            with self._lock:
                self.history.append((ts, symbol, side, usd_val, price))

                stats = self.symbol_stats[symbol]
                stats["last_price"] = price

                if side == 'SELL':
                    stats["24h_long"] += usd_val
                    stats["1h_long"] += usd_val
                else:
                    stats["24h_short"] += usd_val
                    stats["1h_short"] += usd_val

        except Exception:
            pass


    async def _cleanup_loop(self):
        """Runs every minute to remove old data (>24h) and re-calc windows."""
        while self.running:
            await asyncio.sleep(60)
            
            now = time.time()
            cutoff_24h = now - 86400
            cutoff_1h = now - 3600
            
            with self._lock:
                # 1. Prune old history
                while self.history and self.history[0][0] < cutoff_24h:
                    self.history.popleft()
                
                # 2. Re-calculate ALL stats from scratch to ensure accuracy
                # (This prevents "drift" where subtractions might be slightly off over time)
                new_stats = defaultdict(lambda: {
                    "1h_long": 0.0, "1h_short": 0.0, 
                    "24h_long": 0.0, "24h_short": 0.0,
                    "last_price": 0.0
                })
                
                # Copy last prices from old stats to avoid losing them if no liqs in 1 min
                for sym, old_s in self.symbol_stats.items():
                    new_stats[sym]["last_price"] = old_s["last_price"]

                for ts, sym, side, usd, price in self.history:
                    s = new_stats[sym]
                    if side == 'SELL':
                        s["24h_long"] += usd
                        if ts > cutoff_1h: s["1h_long"] += usd
                    else:
                        s["24h_short"] += usd
                        if ts > cutoff_1h: s["1h_short"] += usd
                
                self.symbol_stats = new_stats

    def get_liquidation_data(self, symbol: str) -> dict:
        """
        PUBLIC METHOD: Called by Telegram Bot / Image Generator.
        Returns clean dictionary with 0.0 if no data exists.
        """
        with self._lock:
            s = self.symbol_stats.get(symbol, {})
            return {
                "1h_long": s.get("1h_long", 0.0),
                "1h_short": s.get("1h_short", 0.0),
                "24h_long": s.get("24h_long", 0.0),
                "24h_short": s.get("24h_short", 0.0),
                "last_liq_price": s.get("last_price", 0.0)
            }

# ==============================================================================
# 🧠 MARKET MATRIX (Memory Safe)
# ==============================================================================

class MarketMatrix:
    def __init__(self, symbols, timeframe, max_candles=1000):
        self.symbols = symbols
        self.timeframe = timeframe
        self.n_symbols = len(symbols)
        self.max_candles = max_candles
        self.symbol_map = {sym: i for i, sym in enumerate(symbols)}
        
        # --- Memory Allocations ---
        # 1. OHLCV (Standard Candles)
        self.closes = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.volumes = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.highs = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.lows = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.timestamps = np.zeros((self.n_symbols, max_candles), dtype=np.uint64)
        self.last_update_ts = np.zeros(self.n_symbols, dtype=np.uint64)

        # 2. Indicators (Computed In-Place)
        self.rsi = np.full(self.n_symbols, 50.0, dtype=np.float32)
        self.mfi = np.full(self.n_symbols, 50.0, dtype=np.float32)
        self.adx = np.zeros(self.n_symbols, dtype=np.float32)
        self.atr = np.zeros(self.n_symbols, dtype=np.float32)
        self.vol_z = np.zeros(self.n_symbols, dtype=np.float32)

        # 3. Futures Data (Global Metrics)
        self.funding_rate = np.zeros(self.n_symbols, dtype=np.float32)
        self.next_funding_time = np.zeros(self.n_symbols, dtype=np.uint64)
        self.open_interest = np.zeros(self.n_symbols, dtype=np.float32)
        self.long_short_ratio_accounts = np.zeros(self.n_symbols, dtype=np.float32)
        self.long_short_ratio_positions = np.zeros(self.n_symbols, dtype=np.float32)
        self.long_short_ratio_global = np.zeros(self.n_symbols, dtype=np.float32)
        self.taker_buy_sell_ratio = np.zeros(self.n_symbols, dtype=np.float32)

        # 4. Mark Price & Index Price
        self.mark_price = np.zeros(self.n_symbols, dtype=np.float32)
        self.index_price = np.zeros(self.n_symbols, dtype=np.float32)
        self.premium_index = np.zeros(self.n_symbols, dtype=np.float32)

        # 5. 24h Stats
        self.price_change_24h = np.zeros(self.n_symbols, dtype=np.float32)
        self.volume_24h = np.zeros(self.n_symbols, dtype=np.float32)
        self.turnover_24h = np.zeros(self.n_symbols, dtype=np.float32)

        # 6. Liquidity Depth (-5% / +5%)
        self.liquidity_bid_5pct = np.zeros(self.n_symbols, dtype=np.float32)
        self.liquidity_ask_5pct = np.zeros(self.n_symbols, dtype=np.float32)
        self.liquidity_bid_bins = np.zeros((self.n_symbols, 20), dtype=np.float32)
        self.liquidity_ask_bins = np.zeros((self.n_symbols, 20), dtype=np.float32)

        self.lock = threading.Lock()
        self.last_calc_time = time.time()

    def update_candle(self, symbol, close, volume, high, low, ts_ms: int, is_closed: bool):
        idx = self.symbol_map.get(symbol)
        if idx is None: return

        # Constants for gap detection
        # (Move these to class constants if you prefer)
        TF_MS = {
            '1m': 60000, '5m': 300000, '15m': 900000, 
            '1h': 3600000, '4h': 14400000, '1d': 86400000
        }
        interval = TF_MS.get(self.timeframe, 60000)
        pos = self.max_candles - 1

        with self.lock:
            # 1. Gap Detection Logic
            last_ts = int(self.timestamps[idx, pos])
            
            if last_ts > 0 and ts_ms > (last_ts + interval * 1.5):
                # Detected a gap larger than 1.5x interval
                missed_count = int((ts_ms - last_ts) // interval) - 1
                
                # Limit fills to 5 candles max to prevent freezing
                fill_count = min(missed_count, 5)

                if fill_count > 0:
                    prev_close = self.closes[idx, pos]
                    
                    # Fill gaps with flat candles (Doji)
                    for k in range(fill_count):
                        # Shift history left
                        self.closes[idx, :-1] = self.closes[idx, 1:]
                        self.volumes[idx, :-1] = self.volumes[idx, 1:]
                        self.highs[idx, :-1] = self.highs[idx, 1:]
                        self.lows[idx, :-1] = self.lows[idx, 1:]
                        self.timestamps[idx, :-1] = self.timestamps[idx, 1:]
                        
                        # Insert Ghost Candle at end
                        ghost_ts = last_ts + (interval * (k + 1))
                        self.closes[idx, pos] = prev_close
                        self.volumes[idx, pos] = 0.0  # Zero volume for gaps
                        self.highs[idx, pos] = prev_close
                        self.lows[idx, pos] = prev_close
                        self.timestamps[idx, pos] = ghost_ts

            # 2. Standard Update Logic
            self.closes[idx, pos] = close
            self.volumes[idx, pos] = volume
            self.highs[idx, pos] = high
            self.lows[idx, pos] = low
            
            if ts_ms:
                self.timestamps[idx, pos] = np.uint64(ts_ms)
                self.last_update_ts[idx] = np.uint64(ts_ms)

            if is_closed:
                # Shift for next candle
                self.closes[idx, :-1] = self.closes[idx, 1:]
                self.volumes[idx, :-1] = self.volumes[idx, 1:]
                self.highs[idx, :-1] = self.highs[idx, 1:]
                self.lows[idx, :-1] = self.lows[idx, 1:]
                self.timestamps[idx, :-1] = self.timestamps[idx, 1:]

                # Initialize new candle with current values
                self.closes[idx, pos] = close
                self.volumes[idx, pos] = 0.0
                self.highs[idx, pos] = high
                self.lows[idx, pos] = low

    def update_futures_data(self, symbol: str, data_type: str, value: float, extra_data: dict = None):
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return

        with self.lock:
            if data_type == 'funding_rate':
                self.funding_rate[idx] = value
                if extra_data and 'next_funding_time' in extra_data:
                    self.next_funding_time[idx] = np.uint64(extra_data['next_funding_time'])

            elif data_type == 'open_interest':
                self.open_interest[idx] = value

            elif data_type == 'long_short_accounts':
                self.long_short_ratio_accounts[idx] = value

            elif data_type == 'long_short_positions':
                self.long_short_ratio_positions[idx] = value

            elif data_type == 'long_short_global':
                self.long_short_ratio_global[idx] = value

            elif data_type == 'taker_ratio':
                self.taker_buy_sell_ratio[idx] = value

            elif data_type == 'mark_price':
                self.mark_price[idx] = value

            elif data_type == 'index_price':
                self.index_price[idx] = value

            elif data_type == 'premium_index':
                self.premium_index[idx] = value

            elif data_type == 'price_change_24h':
                self.price_change_24h[idx] = value

            elif data_type == 'volume_24h':
                self.volume_24h[idx] = value

            elif data_type == 'turnover_24h':
                self.turnover_24h[idx] = value

            elif data_type == 'liquidity_5pct':
                # value is total bid liquidity in USDT
                self.liquidity_bid_5pct[idx] = value

                if extra_data:
                    ask_val = extra_data.get('ask')
                    if ask_val is not None:
                        self.liquidity_ask_5pct[idx] = float(ask_val)

                    bid_bins = extra_data.get('bid_bins')
                    if bid_bins is not None:
                        arr = np.asarray(bid_bins, dtype=np.float32)
                        if arr.shape[0] == self.liquidity_bid_bins.shape[1]:
                            self.liquidity_bid_bins[idx] = arr

                    ask_bins = extra_data.get('ask_bins')
                    if ask_bins is not None:
                        arr = np.asarray(ask_bins, dtype=np.float32)
                        if arr.shape[0] == self.liquidity_ask_bins.shape[1]:
                            self.liquidity_ask_bins[idx] = arr



    def get_analysis(self):
        """Returns a DICT of all analysis including futures data for the bot to read instantly."""
        results = {}
        with self.lock:
            # Shallow copies of result arrays (fast)
            rsi_copy = self.rsi
            mfi_copy = self.mfi
            adx_copy = self.adx
            vol_z_copy = self.vol_z
            closes_copy = self.closes
            volumes_copy = self.volumes

            # Futures data copies
            funding_rate_copy = self.funding_rate
            open_interest_copy = self.open_interest
            long_short_accounts_copy = self.long_short_ratio_accounts
            long_short_positions_copy = self.long_short_ratio_positions
            long_short_global_copy = self.long_short_ratio_global
            taker_ratio_copy = self.taker_buy_sell_ratio
            mark_price_copy = self.mark_price
            index_price_copy = self.index_price
            premium_index_copy = self.premium_index
            price_change_24h_copy = self.price_change_24h
            volume_24h_copy = self.volume_24h
            turnover_24h_copy = self.turnover_24h
            bid_liq_copy = self.liquidity_bid_5pct
            ask_liq_copy = self.liquidity_ask_5pct

            # Iterate efficiently
            for sym, i in self.symbol_map.items():
                price = float(closes_copy[i, -1])
                prev = float(closes_copy[i, -2]) if closes_copy.shape[1] >= 2 else 0.0
                change = ((price - prev) / prev) * 100 if prev > 0 else 0.0

                results[sym] = {
                    # Spot data
                    'price': price,
                    'volume': float(volumes_copy[i, -1]),
                    'change_tf': change,
                    'rsi': float(rsi_copy[i]),
                    'mfi': float(mfi_copy[i]),
                    'adx': float(adx_copy[i]),
                    'atr': float(self.atr[i]),
                    'vol_z': float(vol_z_copy[i]),

                    # Futures data
                    'funding_rate': float(funding_rate_copy[i]),
                    'open_interest': float(open_interest_copy[i]),
                    'long_short_accounts': float(long_short_accounts_copy[i]),
                    'long_short_positions': float(long_short_positions_copy[i]),
                    'long_short_global': float(long_short_global_copy[i]),
                    'taker_buy_sell_ratio': float(taker_ratio_copy[i]),
                    'mark_price': float(mark_price_copy[i]),
                    'index_price': float(index_price_copy[i]),
                    'premium_index': float(premium_index_copy[i]),
                    'price_change_24h': float(price_change_24h_copy[i]),
                    'volume_24h': float(volume_24h_copy[i]),
                    'turnover_24h': float(turnover_24h_copy[i]),
                    'liquidity_bid_5pct': float(bid_liq_copy[i]),
                    'liquidity_ask_5pct': float(ask_liq_copy[i]),
                }
        return results
    
    def check_and_fill_gaps(self):
        """
        Active Health Check: Scans for symbols that have stopped updating.
        If a symbol is silent for > 2 intervals, it force-fills the gap.
        """
        # 1. Determine timeframe in milliseconds
        TF_MS = {
            '1m': 60000, '5m': 300000, '15m': 900000, 
            '1h': 3600000, '4h': 14400000, '1d': 86400000
        }
        interval = TF_MS.get(self.timeframe)
        if not interval: return

        now = int(time.time() * 1000)
        pos = self.max_candles - 1
        
        with self.lock:
            # Vectorized check: Find all indices where (now - last_update) > 2.5 * interval
            # We use 2.5x to be generous and avoid race conditions with live updates
            time_diffs = now - self.last_update_ts
            
            # Find stale indices where timestamp > 0 (avoid newly added symbols)
            stale_indices = np.where((time_diffs > (interval * 2.5)) & (self.last_update_ts > 0))[0]
            
            if len(stale_indices) == 0:
                return
            
            # Iterate through stale symbols and fill gaps
            for idx in stale_indices:
                last_ts = int(self.timestamps[idx, pos])
                
                # Double check inside the loop
                if last_ts == 0: continue
                
                # Calculate missed candles
                missed = int((now - last_ts) // interval)
                
                # Cap fills to 10 to prevent massive loops on dead coins
                fill_count = min(missed, 10)
                
                if fill_count > 0:
                    prev_close = self.closes[idx, pos]
                    
                    for k in range(fill_count):
                        # Shift arrays
                        self.closes[idx, :-1] = self.closes[idx, 1:]
                        self.volumes[idx, :-1] = self.volumes[idx, 1:]
                        self.highs[idx, :-1] = self.highs[idx, 1:]
                        self.lows[idx, :-1] = self.lows[idx, 1:]
                        self.timestamps[idx, :-1] = self.timestamps[idx, 1:]
                        
                        # Insert Ghost Candle
                        ghost_ts = last_ts + (interval * (k + 1))
                        self.closes[idx, pos] = prev_close
                        self.volumes[idx, pos] = 0.0
                        self.highs[idx, pos] = prev_close
                        self.lows[idx, pos] = prev_close
                        self.timestamps[idx, pos] = ghost_ts
                        
                        # IMPORTANT: Update last_update_ts so we don't re-fill next loop
                        self.last_update_ts[idx] = ghost_ts



# ==============================================================================
# 📘 FUTURES DEPTH CACHE (With Memory Cleanup)
# ==============================================================================

class FuturesDataCollector:
    """
    Collects ALL available futures data from Binance:
    - Funding Rate (REST + WebSocket)
    - Open Interest (REST + WebSocket)
    - Long/Short Ratios (REST)
    - Taker Buy/Sell Volume (REST)
    - Mark Price, Index Price (WebSocket)
    - Premium Index (WebSocket)
    - 24h Ticker Stats (WebSocket)
    """

    def __init__(self, symbols: List[str], matrices: Dict[str, MarketMatrix]):
        self.symbols = [s.upper() for s in symbols]
        self.matrices = matrices  # Reference to all timeframe matrices
        self.running = True
        self.fapi_base = "https://fapi.binance.com/fapi/v1"
        self.fapi_ws = "wss://fstream.binance.com/ws"


    async def fetch_funding_rates(self):
        """
        Fetches current funding rates for all symbols.
        Updated every 10 minutes (funding happens every 8 hours).
        """
        try:
            url = f"{self.fapi_base}/premiumIndex"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as resp:
                    if resp.status != 200:
                        return

                    data = await resp.json()

                    for item in data:
                        symbol = item.get('symbol')
                        if symbol not in self.symbols:
                            continue

                        funding_rate = float(item.get('lastFundingRate', 0)) * 100  # Convert to %      
                        next_funding = int(item.get('nextFundingTime', 0))
                        mark_price = float(item.get('markPrice', 0))
                        index_price = float(item.get('indexPrice', 0))

                        # Calculate premium index
                        if index_price > 0:
                            premium = ((mark_price - index_price) / index_price) * 100
                        else:
                            premium = 0.0

                        # Update all timeframe matrices
                        for matrix in self.matrices.values():
                            matrix.update_futures_data(symbol, 'funding_rate', funding_rate,
                                                     {'next_funding_time': next_funding})
                            matrix.update_futures_data(symbol, 'mark_price', mark_price)
                            matrix.update_futures_data(symbol, 'index_price', index_price)
                            matrix.update_futures_data(symbol, 'premium_index', premium)

                    logger.debug(f"Updated funding rates for {len(data)} symbols")

        except Exception as e:
            logger.error(f"Funding rate fetch error: {e}")


    async def fetch_open_interest(self):
        """
        Fetches open interest for all symbols.
        Updated every 30 seconds.
        """
        try:
            url = f"{self.fapi_base}/openInterest"

            async with aiohttp.ClientSession() as session:
                tasks = []
                for symbol in self.symbols:
                    tasks.append(self._fetch_single_oi(session, symbol))

                await asyncio.gather(*tasks)

        except Exception as e:
            logger.error(f"Open interest fetch error: {e}")


    async def _fetch_single_oi(self, session, symbol):
        """Helper to fetch OI for a single symbol."""
        try:
            url = f"{self.fapi_base}/openInterest"
            async with session.get(url, params={"symbol": symbol}, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    oi_value = float(data.get('openInterest', 0))

                    # Update all matrices
                    for matrix in self.matrices.values():
                        matrix.update_futures_data(symbol, 'open_interest', oi_value)
        except:
            pass


    async def fetch_long_short_ratio(self):
        """
        Fetches long/short ratios from 3 sources:
        - Top Trader Accounts
        - Top Trader Positions
        - Global (All Accounts)

        Updated every 5 minutes.
        """
        try:
            period = "5m"  # 5-minute window
            limit = 1  # Only need latest

            async with aiohttp.ClientSession() as session:
                for symbol in self.symbols:
                    try:
                        # 1. Top Trader Long/Short Ratio (Accounts)
                        url1 = f"{self.fapi_base}/topLongShortAccountRatio"
                        async with session.get(url1, params={"symbol": symbol, "period": period, "limit": limit}, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    ratio = float(data[0].get('longShortRatio', 1.0))
                                    for matrix in self.matrices.values():
                                        matrix.update_futures_data(symbol, 'long_short_accounts', ratio)

                        # 2. Top Trader Long/Short Ratio (Positions)
                        url2 = f"{self.fapi_base}/topLongShortPositionRatio"
                        async with session.get(url2, params={"symbol": symbol, "period": period, "limit": limit}, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    ratio = float(data[0].get('longShortRatio', 1.0))
                                    for matrix in self.matrices.values():
                                        matrix.update_futures_data(symbol, 'long_short_positions', ratio)

                        # 3. Global Long/Short Ratio (All Accounts)
                        url3 = f"{self.fapi_base}/globalLongShortAccountRatio"
                        async with session.get(url3, params={"symbol": symbol, "period": period, "limit": limit}, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    ratio = float(data[0].get('longShortRatio', 1.0))
                                    for matrix in self.matrices.values():
                                        matrix.update_futures_data(symbol, 'long_short_global', ratio)  

                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.1)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Long/short ratio fetch error: {e}")


    async def fetch_taker_buysell_volume(self):
        """
        Fetches taker buy/sell volume ratio.
        Higher ratio = more aggressive buying.
        Updated every 5 minutes.
        """
        try:
            period = "5m"
            limit = 1

            async with aiohttp.ClientSession() as session:
                for symbol in self.symbols:
                    try:
                        url = f"{self.fapi_base}/takerlongshortRatio"
                        async with session.get(url, params={"symbol": symbol, "period": period, "limit": limit}, timeout=5) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                if data:
                                    buy_vol = float(data[0].get('buySellRatio', 1.0))
                                    for matrix in self.matrices.values():
                                        matrix.update_futures_data(symbol, 'taker_ratio', buy_vol)      

                        await asyncio.sleep(0.1)
                    except:
                        continue

        except Exception as e:
            logger.error(f"Taker volume fetch error: {e}")



    async def websocket_24h_ticker(self):
        url = f"{self.fapi_ws}/!ticker@arr"
        backoff = 1.0

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info("📊 Futures 24h Ticker Stream Connected (Aggregate)")
                        backoff = 1.0

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    if isinstance(data, list):
                                        for ticker in data:
                                            symbol = ticker.get('s')
                                            if symbol not in self.symbols:
                                                continue

                                            price_change_pct = float(ticker.get('P', 0.0))
                                            volume = float(ticker.get('v', 0.0))
                                            quote_volume = float(ticker.get('q', 0.0))

                                            for matrix in self.matrices.values():
                                                matrix.update_futures_data(symbol, 'price_change_24h', price_change_pct)
                                                matrix.update_futures_data(symbol, 'volume_24h', volume)
                                                matrix.update_futures_data(symbol, 'turnover_24h', quote_volume)
                                except Exception:
                                    pass

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except Exception as e:
                logger.error(f"24h Ticker WS error: {e}")
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5




    async def websocket_mark_price(self):
        url = f"{self.fapi_ws}/!markPrice@arr@1s"
        backoff = 1.0

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info("📊 Futures Mark Price Stream Connected (Aggregate)")
                        backoff = 1.0

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    data = json.loads(msg.data)
                                    if isinstance(data, list):
                                        for item in data:
                                            symbol = item.get('s')
                                            if symbol not in self.symbols:
                                                continue

                                            mark_price = float(item.get('p', 0.0))
                                            index_price = float(item.get('i', 0.0))
                                            funding_rate = float(item.get('r', 0.0)) * 100.0
                                            next_funding = int(item.get('T', 0))

                                            premium = 0.0
                                            if index_price > 0.0:
                                                premium = ((mark_price - index_price) / index_price) * 100.0

                                            for matrix in self.matrices.values():
                                                matrix.update_futures_data(symbol, 'mark_price', mark_price)
                                                matrix.update_futures_data(symbol, 'index_price', index_price)
                                                matrix.update_futures_data(symbol, 'premium_index', premium)
                                                matrix.update_futures_data(
                                                    symbol, 'funding_rate', funding_rate,
                                                    {'next_funding_time': next_funding}
                                                )
                                except Exception:
                                    pass

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except Exception as e:
                logger.error(f"Mark Price WS error: {e}")
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5

    async def periodic_rest_updates(self):
        """
        Periodic REST API calls for data that doesn't have WebSocket streams.
        """
        logger.info("📊 Futures REST updater started")

        while self.running:
            try:
                # Fetch all REST-only data
                await self.fetch_open_interest()
                await asyncio.sleep(30)  # 30s interval

                await self.fetch_long_short_ratio()
                await asyncio.sleep(300)  # 5 min interval

                await self.fetch_taker_buysell_volume()
                await asyncio.sleep(300)  # 5 min interval

            except Exception as e:
                logger.error(f"REST update error: {e}")
                await asyncio.sleep(60)


    async def run(self):
        """
        Main entry point - starts all data collection tasks.
        """
        logger.info("📊 Futures Data Collector Starting...")

        # Initial fetch (BLOCKING!)
        await self.fetch_funding_rates()
        await self.fetch_open_interest()

        # Start all tasks (ONLY REACHED AFTER ABOVE RETURNS)
        tasks = [
            asyncio.create_task(self.websocket_24h_ticker()),
            asyncio.create_task(self.websocket_mark_price()),
            asyncio.create_task(self.periodic_rest_updates()),
        ]

        await asyncio.gather(*tasks)
class DepthLiquidityCollector:
    def __init__(self, symbols: List[str], matrices: Dict[str, MarketMatrix]):
        self.symbols = [s.upper() for s in symbols]
        self.matrices = matrices
        self.running = True
        self.fapi_base = "https://fapi.binance.com/fapi/v1"
        self.rate_limit_weight = 10  # Weight for limit=1000
        self.max_weight_per_min = 2400
        self.request_delay = 60.0 / max(len(self.symbols), 1)


    async def fetch_and_calculate_depth(self, session, symbol):
        try:
            url = f"{self.fapi_base}/depth"
            params = {"symbol": symbol, "limit": 1000}

            async with session.get(url, params=params, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    bids = data.get('bids', [])
                    asks = data.get('asks', [])
                    if not bids or not asks:
                        return

                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2.0

                    lower_bound = mid_price * 0.95
                    upper_bound = mid_price * 1.05

                    bid_bins = np.zeros(20, dtype=np.float32)
                    ask_bins = np.zeros(20, dtype=np.float32)

                    # Bids: from mid downwards
                    for p_str, q_str in bids:
                        price = float(p_str)
                        if price < lower_bound:
                            break  # Out of range (-5%)
                        qty = float(q_str)
                        dist_pct = ((mid_price - price) / mid_price) * 100.0
                        bin_idx = int(dist_pct / 0.25)
                        if 0 <= bin_idx < 20:
                            bid_bins[bin_idx] += price * qty

                    # Asks: from mid upwards
                    for p_str, q_str in asks:
                        price = float(p_str)
                        if price > upper_bound:
                            break  # Out of range (+5%)
                        qty = float(q_str)
                        dist_pct = ((price - mid_price) / mid_price) * 100.0
                        bin_idx = int(dist_pct / 0.25)
                        if 0 <= bin_idx < 20:
                            ask_bins[bin_idx] += price * qty

                    total_bid_liquidity = float(np.sum(bid_bins))
                    total_ask_liquidity = float(np.sum(ask_bins))

                    extra_data = {
                        'ask': total_ask_liquidity,
                        'bid_bins': bid_bins,
                        'ask_bins': ask_bins,
                    }

                    for matrix in self.matrices.values():
                        matrix.update_futures_data(
                            symbol,
                            'liquidity_5pct',
                            total_bid_liquidity,
                            extra_data
                        )

                elif resp.status == 429:
                    logger.warning(f"DepthCollector: Rate limit hit (429) for {symbol}, slowing down...")
                    await asyncio.sleep(5)

        except Exception as e:
            logger.debug(f"Depth calc error for {symbol}: {e}")



    async def run(self):
        if self.symbols:
            self.request_delay = 60.0 / max(len(self.symbols), 1)
            
        logger.info(f"💧 Depth Liquidity Collector Started ({len(self.symbols)} symbols, ~{self.request_delay:.2f}s delay)")

        connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            while self.running:
                current_symbols = list(self.symbols)
                
                for symbol in current_symbols:
                    if not self.running: break
                    await self.fetch_and_calculate_depth(session, symbol)
                    
                    if current_symbols:
                        count = len(current_symbols)
                        # Ensure we don't go too fast (max 2300 weight/min)
                        safe_delay = 60.0 / (2300 / 10)  # ~0.26s minimum
                        target_delay = 60.0 / max(count, 1)
                        self.request_delay = max(target_delay, safe_delay)
                    
                    await asyncio.sleep(self.request_delay)


class BinanceMarketDataCollector:
    def __init__(self):
        self.api_url = "https://api.binance.com/api/v3"
        self.spot_ws_base = "wss://fstream.binance.com"
        self.top_coins = []
        self.matrices = {}
        self.active_timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.output_dir = "./data_storage"
        self.depth_liquidity_collector = None

        os.makedirs(self.output_dir, exist_ok=True)
        self.liq_monitor = LiquidationMonitor()
        self.liq_history_loader = None
        self.futures_collector = None  

    def get_depth_liquidity_range(
        self,
        symbol: str,
        target_usdt: float = 250000,
        mid_price: Optional[float] = None
    ) -> Optional[Dict[str, float]]:
        """
        Returns calculated liquidity ranges for image generator.
        Calculates how far price needs to move (+/- %) to hit `target_usdt` wall.
        """
        # Always check 1m matrix for freshest price & depth data
        matrix = self.matrices.get('1m')
        if not matrix: return None
        
        idx = matrix.symbol_map.get(symbol)
        if idx is None: return None
        
        with matrix.lock:
            # 1. Get raw bins (20 bins, 0.25% each)
            bid_bins = matrix.liquidity_bid_bins[idx]
            ask_bins = matrix.liquidity_ask_bins[idx]
            current_price = float(matrix.closes[idx, -1])
        
        if current_price <= 0: return None
        
        # 2. Calculate Downside (-X% to hit target_usdt)
        cum_bid = 0.0
        down_pct = 0.0
        BIN_WIDTH = 0.25
        
        for i in range(20):
            bin_val = float(bid_bins[i])
            if (cum_bid + bin_val) >= target_usdt:
                # Interpolate within this bin
                needed = target_usdt - cum_bid
                fraction = needed / bin_val if bin_val > 0 else 0
                down_pct = (i * BIN_WIDTH) + (fraction * BIN_WIDTH)
                cum_bid += bin_val
                break
            cum_bid += bin_val
            down_pct = (i + 1) * BIN_WIDTH # Full bin used
            
        # If we exhausted all bins and still didn't hit target
        if cum_bid < target_usdt:
            down_pct = 5.0 # Max range cap
            
        down_price = current_price * (1 - (down_pct / 100.0))

        # 3. Calculate Upside (+X% to hit target_usdt)
        cum_ask = 0.0
        up_pct = 0.0
        
        for i in range(20):
            bin_val = float(ask_bins[i])
            if (cum_ask + bin_val) >= target_usdt:
                needed = target_usdt - cum_ask
                fraction = needed / bin_val if bin_val > 0 else 0
                up_pct = (i * BIN_WIDTH) + (fraction * BIN_WIDTH)
                cum_ask += bin_val
                break
            cum_ask += bin_val
            up_pct = (i + 1) * BIN_WIDTH
            
        if cum_ask < target_usdt:
            up_pct = 5.0
            
        up_price = current_price * (1 + (up_pct / 100.0))

        return {
            "symbol": symbol,
            "mid": current_price,
            "down_pct": down_pct,
            "up_pct": up_pct,
            "down_price": down_price,
            "up_price": up_price,
            "bid_notional": float(np.sum(bid_bins)),
            "ask_notional": float(np.sum(ask_bins)),
            "book_ts_ms": int(time.time() * 1000)
        }

    
    def get_liq_stats(self, symbol: str) -> dict:
        """Accessor for Telegram Bot to get liquidation stats."""
        return self.liq_monitor.get_liquidation_data(symbol)
    
    async def fetch_top_coins(self):
        url = f"{self.api_url}/ticker/24hr"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as r:
                    data = await r.json()

                    usdt_pairs = []
                    for ticker in data:
                        symbol = ticker.get('symbol', '')

                        if not symbol.endswith('USDT'):
                            continue
                        base_asset = symbol[:-4]  # Remove 'USDT'
                        if base_asset in STABLECOINS:
                            continue
                        usdt_pairs.append(ticker)

                    usdt_pairs.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)

                    new_top_coins = [x['symbol'] for x in usdt_pairs[:170]]        

                    if self.top_coins:
                        old_set = set(self.top_coins)
                        new_set = set(new_top_coins)
                        added = new_set - old_set
                        removed = old_set - new_set

                        if added:
                            logger.info(f"➕ Added symbols: {', '.join(sorted(added)[:5])}..." if len(added) > 5 else f"➕ Added: {', '.join(sorted(added))}")        
                        if removed:
                            logger.info(f"➖ Removed symbols: {', '.join(sorted(removed)[:5])}..." if len(removed) > 5 else f"➖ Removed: {', '.join(sorted(removed))}")

                    self.top_coins = new_top_coins
                    logger.info(f"✔ Loaded {len(self.top_coins)} Top Coins (Stablecoins excluded)")

        except Exception as e:
            logger.exception(f"Failed to fetch top coins: {e}")
            # Fallback to safe defaults if first run
            if not self.top_coins:
                self.top_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

        return self.top_coins


    async def load_state(self):
        """Restores state from disk."""
        loaded = True
        try:
            for tf in self.active_timeframes:
                # FIX: Handle Windows case-insensitivity
                safe_tf = "1mo" if tf == '1M' else tf
                path = os.path.join(self.output_dir, f"matrix_{safe_tf}.npz")
                
                if os.path.exists(path):
                    data = np.load(path, allow_pickle=True)
                    saved_syms = list(data['symbols'])
                    
                    # Create matrix if missing
                    if tf not in self.matrices:
                        self.matrices[tf] = MarketMatrix(saved_syms, tf)
                    
                    try:
                        self.matrices[tf].closes = data['closes']
                        self.matrices[tf].volumes = data['volumes']
                        self.matrices[tf].timestamps = data['timestamps']
                        if not self.top_coins: self.top_coins = saved_syms
                    except Exception: pass
                    loaded = True
            if loaded: logger.info("✔ State restored from disk.")
        except Exception:
            logger.error("Could not load state from disk")
        return loaded


    async def _rebuild_matrices_for_new_symbols(self):
        """
        Called when symbol list changes.
        Preserves data for existing symbols, adds new ones.
        """
        new_symbols = self.top_coins

        for tf in self.active_timeframes:
            # If matrix for this TF doesn't exist yet, just create it fresh
            if tf not in self.matrices:
                self.matrices[tf] = MarketMatrix(new_symbols, tf)
                continue

            old_matrix = self.matrices[tf]
            old_symbols = old_matrix.symbols

            # No change needed
            if set(old_symbols) == set(new_symbols):
                continue

            # Create new matrix with updated symbols, same candle depth
            new_matrix = MarketMatrix(new_symbols, tf, old_matrix.max_candles)

            # Copy data for overlapping symbols
            for new_idx, symbol in enumerate(new_symbols):
                old_idx = old_matrix.symbol_map.get(symbol)
                if old_idx is None:
                    continue

                with old_matrix.lock:
                    new_matrix.closes[new_idx] = old_matrix.closes[old_idx]
                    new_matrix.volumes[new_idx] = old_matrix.volumes[old_idx]
                    new_matrix.highs[new_idx] = old_matrix.highs[old_idx]
                    new_matrix.lows[new_idx] = old_matrix.lows[old_idx]
                    new_matrix.timestamps[new_idx] = old_matrix.timestamps[old_idx]
                    new_matrix.rsi[new_idx] = old_matrix.rsi[old_idx]
                    new_matrix.mfi[new_idx] = old_matrix.mfi[old_idx]
                    new_matrix.adx[new_idx] = old_matrix.adx[old_idx]
                    new_matrix.atr[new_idx] = old_matrix.atr[old_idx]
                    new_matrix.vol_z[new_idx] = old_matrix.vol_z[old_idx]

                    # Futures / liquidity fields (safe to copy as-is)
                    new_matrix.funding_rate[new_idx] = old_matrix.funding_rate[old_idx]
                    new_matrix.next_funding_time[new_idx] = old_matrix.next_funding_time[old_idx]
                    new_matrix.open_interest[new_idx] = old_matrix.open_interest[old_idx]
                    new_matrix.long_short_ratio_accounts[new_idx] = old_matrix.long_short_ratio_accounts[old_idx]
                    new_matrix.long_short_ratio_positions[new_idx] = old_matrix.long_short_ratio_positions[old_idx]
                    new_matrix.long_short_ratio_global[new_idx] = old_matrix.long_short_ratio_global[old_idx]
                    new_matrix.taker_buy_sell_ratio[new_idx] = old_matrix.taker_buy_sell_ratio[old_idx]
                    new_matrix.mark_price[new_idx] = old_matrix.mark_price[old_idx]
                    new_matrix.index_price[new_idx] = old_matrix.index_price[old_idx]
                    new_matrix.premium_index[new_idx] = old_matrix.premium_index[old_idx]
                    new_matrix.price_change_24h[new_idx] = old_matrix.price_change_24h[old_idx]
                    new_matrix.volume_24h[new_idx] = old_matrix.volume_24h[old_idx]
                    new_matrix.turnover_24h[new_idx] = old_matrix.turnover_24h[old_idx]
                    new_matrix.liquidity_bid_5pct[new_idx] = old_matrix.liquidity_bid_5pct[old_idx]
                    new_matrix.liquidity_ask_5pct[new_idx] = old_matrix.liquidity_ask_5pct[old_idx]
                    new_matrix.liquidity_bid_bins[new_idx] = old_matrix.liquidity_bid_bins[old_idx]
                    new_matrix.liquidity_ask_bins[new_idx] = old_matrix.liquidity_ask_bins[old_idx]

            # Atomically swap matrix reference
            self.matrices[tf] = new_matrix
            del old_matrix
            gc.collect()
            logger.info(f"🔄 Rebuilt matrix for {tf} (symbols: {len(new_symbols)}) | Memory Cleaned")


    async def symbol_refresh_task(self):
        logger.info("🔄 Symbol Auto-Refresh Task Started (runs every 1 hour)")     

        while self.running:
            await asyncio.sleep(3600)  # 1 hour
            try:
                old_symbols = set(self.top_coins) if self.top_coins else set()
                await self.fetch_top_coins()
                new_symbols = set(self.top_coins)

                if old_symbols != new_symbols:
                    logger.info("📊 Symbol list changed - rebuilding matrices...") 
                    await self._rebuild_matrices_for_new_symbols()

                    # Update depth manager watchlist
                    if self.depth_liquidity_collector:
                        self.depth_liquidity_collector.symbols = list(self.top_coins)

                    logger.info("✅ Symbol refresh complete")
                else:
                    logger.debug("Symbol list unchanged")
            except Exception as e:
                logger.error(f"Symbol refresh failed: {e}")
                await asyncio.sleep(300)  # Retry in 5 min on error


    async def fetch_historical_snapshot(self):
        """Fills the matrix with initial history."""
        logger.info("Fetching historical snapshot (Warmup)...")
        concurrency = 50
        sem = asyncio.Semaphore(concurrency)
        timeout = aiohttp.ClientTimeout(total=30, sock_connect=5)
        connector = aiohttp.TCPConnector(limit=60, ttl_dns_cache=300)

        async def fetch_klines(session, symbol, tf, limit):
            url = f"{self.api_url}/klines"
            params = {"symbol": symbol, "interval": tf, "limit": limit}
            for attempt in range(3):
                try:
                    async with sem:
                        async with session.get(url, params=params) as resp:
                            if resp.status == 429:
                                await asyncio.sleep(float(resp.headers.get("Retry-After", 2)))
                                continue
                            if resp.status == 200:
                                data = await resp.json()
                                return symbol, tf, data
                except Exception:
                    await asyncio.sleep(1)
            return symbol, tf, None

        # Ensure matrices
        for tf in self.active_timeframes:
            if tf not in self.matrices:
                self.matrices[tf] = MarketMatrix(self.top_coins, tf)

        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            for tf in self.active_timeframes:
                matrix = self.matrices[tf]
                limit = min(matrix.max_candles, 1000)
                tasks = [fetch_klines(session, s, tf, limit) for s in self.top_coins]
                results = await asyncio.gather(*tasks)

                for res in results:
                    sym, _tf, klines = res
                    if not klines: continue
                    idx = matrix.symbol_map.get(sym)
                    if idx is None: continue

                    try:
                        # Parsing
                        c_arr = np.array([float(k[4]) for k in klines], dtype=np.float32)
                        v_arr = np.array([float(k[5]) for k in klines], dtype=np.float32)
                        h_arr = np.array([float(k[2]) for k in klines], dtype=np.float32)
                        l_arr = np.array([float(k[3]) for k in klines], dtype=np.float32)
                        ts_arr = np.array([int(k[0]) for k in klines], dtype=np.uint64)
                        
                        n = min(len(c_arr), matrix.max_candles)
                        with matrix.lock:
                            matrix.closes[idx, -n:] = c_arr[-n:]
                            matrix.volumes[idx, -n:] = v_arr[-n:]
                            matrix.highs[idx, -n:] = h_arr[-n:]
                            matrix.lows[idx, -n:] = l_arr[-n:]
                            matrix.timestamps[idx, -n:] = ts_arr[-n:]
                    except Exception: pass
        logger.info("✔ Warmup Complete.")

    async def _single_futures_kline_handler(self, streams: List[str], ws_id: int):
        url = f"{self.spot_ws_base}/stream?streams=" + "/".join(streams)
        backoff = 1.0

        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info(f"WS-{ws_id}: Connected ({len(streams)} streams)")
                        backoff = 1.0

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                    data = payload.get("data", payload)
                                    if data.get('e') == 'kline':
                                        self._process_kline(data)
                                except Exception as e:
                                    logger.debug(f"WS-{ws_id} parse error: {e}")
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except Exception as e:
                logger.error(f"WS-{ws_id} connection error: {e}")
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5


    def _process_kline(self, data: dict):
        try:
            k = data.get('k')
            if not k:
                return

            tf = k.get('i')
            symbol = data.get('s')
            if not tf or not symbol:
                return

            matrix = self.matrices.get(tf)
            if matrix is None:
                return

            close = float(k.get('c', 0.0))
            volume = float(k.get('v', 0.0))
            high = float(k.get('h', 0.0))
            low = float(k.get('l', 0.0))
            ts_ms = int(k.get('t', 0))
            is_closed = bool(k.get('x', False))

            matrix.update_candle(symbol, close, volume, high, low, ts_ms, is_closed)

        except Exception as e:
            logger.debug(f"_process_kline error: {e}")


    def _update_indicators_job(self, matrix: MarketMatrix):
        """
        Runs in ThreadPool.
        Reads snapshot of OHLCV under lock, then computes indicators in-place.
        """
        try:
            # Take snapshot under lock to avoid partial writes
            with matrix.lock:
                closes = np.ascontiguousarray(matrix.closes)
                volumes = np.ascontiguousarray(matrix.volumes)
                highs = np.ascontiguousarray(matrix.highs)
                lows = np.ascontiguousarray(matrix.lows)

                # Outputs are the pre-allocated arrays on the matrix itself.
                out_rsi = matrix.rsi
                out_mfi = matrix.mfi
                out_adx = matrix.adx
                out_vol_z = matrix.vol_z
                out_atr = matrix.atr

            # Heavy math outside the lock to minimize blocking writers
            calc_indicators_inplace(
                closes, volumes, highs, lows,
                out_rsi, out_mfi, out_adx, out_vol_z, out_atr
            )
            matrix.last_calc_time = time.time()
        except Exception as e:
            logger.error(f"Indicator Job Failed for timeframe {matrix.timeframe}: {e}", exc_info=True)

    async def calculation_loop(self):
        logger.info("Starting Math Loop")

        loop = asyncio.get_running_loop()

        while self.running:
            start = time.time()

            # Submit one job per timeframe matrix
            futures = []
            for m in self.matrices.values():
                # Skip empty matrices (no symbols yet)
                if m.n_symbols == 0:
                    continue
                futures.append(loop.run_in_executor(self.executor, self._update_indicators_job, m))

            if futures:
                # Wait for all indicator jobs to finish
                try:
                    await asyncio.gather(*futures)
                except Exception as e:
                    logger.error(f"Error in calculation_loop gather: {e}", exc_info=True)

            elapsed = time.time() - start
            # Target ~1s cycle, never negative sleep
            await asyncio.sleep(max(0.1, 1.0 - elapsed))


    def _save_to_disk_thread(self):
        try:
            for tf, matrix in self.matrices.items():
                # FIX: Handle Windows case-insensitivity (1M vs 1m)
                safe_tf = "1mo" if tf == '1M' else tf
                fname = os.path.join(self.output_dir, f"matrix_{safe_tf}.npz")
                
                with matrix.lock:
                    data = {
                        "closes": matrix.closes.copy(),
                        "volumes": matrix.volumes.copy(),
                        "timestamps": matrix.timestamps.copy(),
                        "symbols": matrix.symbols
                    }
                np.savez_compressed(fname, **data)
            logger.info("💾 State Saved")
        except Exception as e: logger.error(f"Save failed: {e}")

    async def periodic_save_task(self):
        while self.running:
            await asyncio.sleep(60)
            await asyncio.get_running_loop().run_in_executor(self.executor, self._save_to_disk_thread)

    async def health_log_task(self):        
        import psutil  # Ensure psutil is installed
        process = psutil.Process(os.getpid())
        
        while self.running:
            await asyncio.sleep(60)
            try:
                # 1. System Resources (CPU & Memory)
                # cpu_percent(interval=None) returns float representing CPU utilization as a percentage
                # First call might return 0.0, subsequent calls return avg utilization since last call.
                cpu_usage = process.cpu_percent(interval=None)
                mem_mb = process.memory_info().rss / 1024 / 1024
                
                # 2. System Lag (Math Loop)
                now = time.time()
                lags = []
                for tf, m in self.matrices.items():
                    lag = now - m.last_calc_time
                    lags.append(f"{tf}:{lag:.1f}s")
                
                lag_str = " | ".join(lags)
                
                # 3. Liquidity & Futures
                liq_count = len(self.liq_monitor.history) if self.liq_monitor else 0
                total_symbols = sum(m.n_symbols for m in self.matrices.values())
                
                logger.info(
                    f"💚 HEALTH | CPU: {cpu_usage:.1f}% | Mem: {mem_mb:.0f}MB | "
                    f"Sym: {total_symbols} | "
                    f"LiqEvents: {liq_count} | "
                    f"Calc Lag: [{lag_str}]"
                )
            except Exception as e:
                logger.error(f"Health check error: {e}")


            
    async def gap_check_task(self):
        """Runs every 60s to fix silent symbols."""
        while self.running:
            await asyncio.sleep(60)
            try:
                for matrix in self.matrices.values():
                    # Run the heavy check in a thread to avoid blocking loop
                    await asyncio.get_running_loop().run_in_executor(
                        self.executor, matrix.check_and_fill_gaps
                    )
            except Exception as e:
                logger.error(f"Gap check failed: {e}")

    async def run(self):
        logger.info("🚀 Data Collector Starting...")
        force_refresh = env_flag("FORCE_REFRESH")

        try:
            # 1) Load or warmup
            state_loaded = False
            if not force_refresh:
                state_loaded = await self.load_state()

            if not state_loaded:
                await self.fetch_top_coins()
                await self.fetch_historical_snapshot()
            elif not self.top_coins:
                # If state loaded but top_coins empty, derive from one matrix
                if self.matrices:
                    any_tf = next(iter(self.matrices))
                    self.top_coins = self.matrices[any_tf].symbols
                else:
                    await self.fetch_top_coins()

            # Safety: ensure matrices exist for all active TFs
            for tf in self.active_timeframes:
                if tf not in self.matrices:
                    self.matrices[tf] = MarketMatrix(self.top_coins, tf)

            # 2) Initialize side collectors AFTER symbols & matrices are ready
            self.liq_history_loader = LiquidationHistoryLoader(
                liq_monitor=self.liq_monitor,
                symbols=self.top_coins
            )

            self.futures_collector = FuturesDataCollector(
                symbols=self.top_coins,
                matrices=self.matrices
            )

            self.depth_liquidity_collector = DepthLiquidityCollector(
                symbols=self.top_coins,
                matrices=self.matrices
            )

            # 3) Build kline streams
            all_streams = []
            for s in self.top_coins:
                sym_l = s.lower()
                for tf in self.active_timeframes:
                    all_streams.append(f"{sym_l}@kline_{tf}")

            chunks = [all_streams[i:i + 200] for i in range(0, len(all_streams), 200)]

            tasks = []

            # 4) Start kline handlers
            for i, chunk in enumerate(chunks):
                tasks.append(asyncio.create_task(
                    self._single_futures_kline_handler(chunk, i),
                    name=f"kline_ws_{i}"
                ))

            # 5) Start core internal loops
            tasks.append(asyncio.create_task(self.calculation_loop(), name="calc_loop"))
            tasks.append(asyncio.create_task(self.periodic_save_task(), name="save_loop"))
            tasks.append(asyncio.create_task(self.health_log_task(), name="health_loop"))
            tasks.append(asyncio.create_task(self.symbol_refresh_task(), name="symbol_refresh"))

            # 6) Start external data collectors
            tasks.append(asyncio.create_task(self.liq_history_loader.start(), name="liq_history"))
            tasks.append(asyncio.create_task(self.futures_collector.run(), name="futures_collector"))
            tasks.append(asyncio.create_task(self.depth_liquidity_collector.run(), name="depth_collector"))
            tasks.append(asyncio.create_task(self.gap_check_task(), name="gap_check"))
            # 7) Run all
            await asyncio.gather(*tasks)

        except Exception as e:
            logger.critical(f"Main Loop Crash: {e}", exc_info=True)
        finally:
            self.stop()
            self.executor.shutdown(wait=False)
            logger.info("Data Collector stopped.")

    def stop(self):
        self.running = False
        if self.depth_liquidity_collector:
            self.depth_liquidity_collector.running = False
        if self.futures_collector:
            self.futures_collector.running = False
        if self.liq_monitor:
            self.liq_monitor.running = False
        if self.liq_history_loader:
            self.liq_history_loader.running = False


if __name__ == "__main__":
    bot = BinanceMarketDataCollector()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
