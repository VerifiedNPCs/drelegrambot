# Data_Collector.py
import asyncio
import json
import os
import logging
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

from config import config, setup_logging

logger = setup_logging("DataCollector")

# ==============================================================================
# ⚡ NUMBA JIT ENGINE (Zero-Allocation Optimization)
# ==============================================================================
def env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "y", "on")

@njit(parallel=True, fastmath=True, cache=True)
def calc_indicators_inplace(
    closes, volumes, highs, lows, 
    out_rsi, out_mfi, out_adx, out_vol_z, 
    period_rsi=14, period_mfi=14, period_adx=14
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
# ==============================================================================
# 💀 LIQUIDATION MONITOR (Full History & Stats)
# ==============================================================================

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

    def _process_event(self, data):
        """Parses a single websocket message."""
        payload = data.get('o', {})
        if not payload: return

        # Extract Fields
        symbol = payload.get('s')
        side = payload.get('S') # SELL = Long Liquidated, BUY = Short Liquidated
        qty = float(payload.get('q', 0))
        price = float(payload.get('p', 0))
        ts = float(data.get('o', {}).get('T', time.time() * 1000)) / 1000.0
        
        usd_val = qty * price

        with self._lock:
            # 1. Add to raw history
            self.history.append((ts, symbol, side, usd_val, price))
            
            # 2. Update real-time stats immediately (Add only)
            stats = self.symbol_stats[symbol]
            stats["last_price"] = price
            
            if side == 'SELL':
                stats["24h_long"] += usd_val
                stats["1h_long"] += usd_val
            else:
                stats["24h_short"] += usd_val
                stats["1h_short"] += usd_val

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

        # Arrays (Float32 for 50% memory saving)
        self.closes = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.volumes = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.highs = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.lows = np.zeros((self.n_symbols, max_candles), dtype=np.float32)
        self.timestamps = np.zeros((self.n_symbols, max_candles), dtype=np.uint64)
        self.last_update_ts = np.zeros(self.n_symbols, dtype=np.uint64)

        # Indicator Buffers (Pre-allocated)
        self.rsi = np.full(self.n_symbols, 50.0, dtype=np.float32)
        self.mfi = np.full(self.n_symbols, 50.0, dtype=np.float32)
        self.adx = np.zeros(self.n_symbols, dtype=np.float32)
        self.vol_z = np.zeros(self.n_symbols, dtype=np.float32)

        self.lock = threading.Lock()

    def update_candle(self, symbol, close, volume, high, low, ts_ms: int, is_closed: bool):
        idx = self.symbol_map.get(symbol)
        if idx is None: return
        pos = self.max_candles - 1

        with self.lock:
            # Atomic write
            self.closes[idx, pos] = close
            self.volumes[idx, pos] = volume
            self.highs[idx, pos] = high
            self.lows[idx, pos] = low
            if ts_ms:
                self.timestamps[idx, pos] = np.uint64(ts_ms)
                self.last_update_ts[idx] = np.uint64(ts_ms)

            if is_closed:
                # Optimized Shift
                self.closes[idx, :-1] = self.closes[idx, 1:]
                self.volumes[idx, :-1] = self.volumes[idx, 1:]
                self.highs[idx, :-1] = self.highs[idx, 1:]
                self.lows[idx, :-1] = self.lows[idx, 1:]
                self.timestamps[idx, :-1] = self.timestamps[idx, 1:]

                # Reset new candle
                self.closes[idx, pos] = close
                self.volumes[idx, pos] = 0.0
                self.highs[idx, pos] = high
                self.lows[idx, pos] = low

    def get_analysis(self):
        """Returns a DICT of all analysis for the bot to read instantly."""
        results = {}
        with self.lock:
            # Shallow copies of result arrays (fast)
            rsi_copy = self.rsi
            mfi_copy = self.mfi
            adx_copy = self.adx
            vol_z_copy = self.vol_z
            closes_copy = self.closes
            volumes_copy = self.volumes
            
            # Iterate efficiently
            for sym, i in self.symbol_map.items():
                price = float(closes_copy[i, -1])
                prev = float(closes_copy[i, -2]) if closes_copy.shape[1] >= 2 else 0.0
                change = ((price - prev) / prev) * 100 if prev > 0 else 0.0

                results[sym] = {
                    'price': price,
                    'volume': float(volumes_copy[i, -1]),
                    'change_tf': change,
                    'rsi': float(rsi_copy[i]),
                    'mfi': float(mfi_copy[i]),
                    'adx': float(adx_copy[i]),
                    'vol_z': float(vol_z_copy[i]),
                }
        return results

# ==============================================================================
# 📘 FUTURES DEPTH CACHE (With Memory Cleanup)
# ==============================================================================

class _DepthBook:
    __slots__ = ("bids", "asks", "max_levels", "last_event_ms", "last_access")
    def __init__(self, max_levels=20):
        self.bids = {}; self.asks = {}
        self.max_levels = max_levels
        self.last_event_ms = 0
        self.last_access = time.time()

    def touch(self): self.last_access = time.time()

    def seed_from_snapshot(self, bids, asks):
        self.bids.clear(); self.asks.clear()
        for p, q in bids: self.bids[float(p)] = float(q)
        for p, q in asks: self.asks[float(p)] = float(q)
        self._trim(); self.touch()

    def apply_updates(self, b_upd, a_upd, event_ms):
        for p, q in b_upd:
            fp, fq = float(p), float(q)
            if fq <= 0: self.bids.pop(fp, None)
            else: self.bids[fp] = fq
        for p, q in a_upd:
            fp, fq = float(p), float(q)
            if fq <= 0: self.asks.pop(fp, None)
            else: self.asks[fp] = fq
        if event_ms: self.last_event_ms = int(event_ms)
        self._trim()

    def _trim(self):
        keep = 60
        if len(self.bids) > keep:
            for p in sorted(self.bids.keys())[:-keep]: self.bids.pop(p, None)
        if len(self.asks) > keep:
            for p in sorted(self.asks.keys())[keep:]: self.asks.pop(p, None)

    def top_levels(self):
        bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[: self.max_levels]
        asks = sorted(self.asks.items(), key=lambda x: x[0])[: self.max_levels]
        return bids, asks

class FuturesDepthManager:
    def __init__(self, levels=20, speed="100ms"):
        self.levels = levels
        self.speed = speed
        self._desired = set()
        self._desired_lock = asyncio.Lock()
        self._books = {}
        self._books_lock = threading.Lock()
        self.running = True
        self._reconnect_event = asyncio.Event()

    async def set_watchlist(self, symbols: List[str]):
        syms = set(s.upper() for s in symbols if s)
        async with self._desired_lock:
            if syms != self._desired:
                self._desired = syms
                self._reconnect_event.set()
                self._cleanup_unused(syms)

    def _cleanup_unused(self, active_symbols: Set[str]):
        with self._books_lock:
            for k in list(self._books.keys()):
                if k not in active_symbols: del self._books[k]

    def _ensure_book(self, symbol: str) -> _DepthBook:
        with self._books_lock:
            if symbol not in self._books:
                self._books[symbol] = _DepthBook(self.levels)
            return self._books[symbol]

    async def _rest_snapshot(self, session, symbol):
        try:
            url = "https://fapi.binance.com/fapi/v1/depth"
            async with session.get(url, params={"symbol": symbol, "limit": 50}, timeout=5) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    book = self._ensure_book(symbol)
                    book.seed_from_snapshot(data.get("bids", []), data.get("asks", []))
        except Exception: pass 

    def compute_liquidity_range(self, symbol, target_usdt, mid_price=None):
        with self._books_lock:
            book = self._books.get(symbol.upper())
            if not book: return None
            book.touch()
            bids, asks = book.top_levels()
            ts = book.last_event_ms

        if not bids or not asks: return None
        best_bid = bids[0][0]; best_ask = asks[0][0]
        mid = float(mid_price) if (mid_price and mid_price > 0) else (best_bid + best_ask) / 2.0
        
        cum = 0.0; down_price = bids[-1][0]
        for p, q in bids:
            cum += p * q
            down_price = p
            if cum >= target_usdt: break
        
        cum = 0.0; up_price = asks[-1][0]
        for p, q in asks:
            cum += p * q
            up_price = p
            if cum >= target_usdt: break
            
        down_pct = max(0.0, (mid - down_price) / mid * 100.0)
        up_pct = max(0.0, (up_price - mid) / mid * 100.0)

        return {
            "mid": mid, "down_price": down_price, "up_price": up_price,
            "down_pct": down_pct, "up_pct": up_pct, "book_ts_ms": ts
        }

    async def run(self):
        while self.running:
            try:
                self._reconnect_event.clear()
                async with self._desired_lock:
                    symbols = sorted(list(self._desired))
                
                if not symbols:
                    await asyncio.sleep(1)
                    continue

                streams = [f"{s.lower()}@depth{self.levels}@{self.speed}" for s in symbols]
                url = "wss://fstream.binance.com/stream?streams=" + "/".join(streams)

                async with aiohttp.ClientSession() as session:
                    await asyncio.gather(*[self._rest_snapshot(session, s) for s in symbols])
                    async with session.ws_connect(url, heartbeat=30) as ws:
                        logger.info(f"DepthManager: Connected {len(symbols)} syms")
                        while self.running:
                            if self._reconnect_event.is_set(): break
                            try:
                                msg = await ws.receive(timeout=10)
                                if msg.type == aiohttp.WSMsgType.TEXT:
                                    data = json.loads(msg.data)
                                    payload = data.get("data", {})
                                    if payload.get("e") == "depthUpdate":
                                        s = payload.get("s")
                                        if s: self._ensure_book(s).apply_updates(
                                            payload.get("b", []), payload.get("a", []), payload.get("E", 0))
                                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR): break
                            except asyncio.TimeoutError: continue
            except Exception as e:
                logger.error(f"DepthManager Error: {e}")
                await asyncio.sleep(5)

# ==============================================================================
# 📡 DATA COLLECTOR (Full Logic)
# ==============================================================================

class BinanceMarketDataCollector:
    def __init__(self):
        self.api_url = "https://api.binance.com/api/v3"
        self.spot_ws_base = "wss://stream.binance.com:9443"
        self.top_coins = []
        self.matrices = {}
        self.active_timeframes = ['1m', '5m', '15m', '1h', '2h', '4h', '1d', '3d', '1w', '1M']
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.output_dir = "./data_storage"
        os.makedirs(self.output_dir, exist_ok=True)
        self.futures_depth = FuturesDepthManager()
        self.liq_monitor = LiquidationMonitor()
    async def set_depth_watchlist(self, symbols: List[str]):
        await self.futures_depth.set_watchlist(symbols)

    def get_depth_liquidity_range(
        self,
        symbol: str,
        target_usdt: float,
        mid_price: Optional[float] = None  # <--- THIS WAS MISSING
    ) -> Optional[Dict[str, float]]:
        """
        Synchronous read accessor (safe) used by telegram_bot to enrich img_data.
        """
        # Pass the arguments through to the futures_depth manager
        return self.futures_depth.compute_liquidity_range(
            symbol, 
            target_usdt, 
            mid_price=mid_price
        )
    def get_liq_stats(self, symbol: str) -> dict:
        """Accessor for Telegram Bot to get liquidation stats."""
        return self.liq_monitor.get_liquidation_data(symbol)
    
    async def get_top_100_coins(self):
        url = f"{self.api_url}/ticker/24hr"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as r:
                    data = await r.json()
                    usdt = [x for x in data if x.get('symbol', '').endswith('USDT')]
                    usdt.sort(key=lambda x: float(x.get('quoteVolume', 0)), reverse=True)
                    self.top_coins = [x['symbol'] for x in usdt[:100]]
                    logger.info(f"Loaded {len(self.top_coins)} Top Coins.")
        except Exception:
            logger.exception("Failed to fetch top coins")
            self.top_coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        return self.top_coins

    async def load_state(self):
        """Restores state from disk."""
        loaded = False
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


    async def fetch_historical_snapshot(self):
        """Fills the matrix with initial history."""
        logger.info("Fetching historical snapshot (Warmup)...")
        concurrency = 15
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

    async def _single_spot_ws_handler(self, streams: List[str], ws_id: int):
        url = f"{self.spot_ws_base}/stream?streams=" + "/".join(streams)
        backoff = 1.0
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info(f"WS-{ws_id}: Connected")
                        backoff = 1.0
                        async for msg in ws:
                            if not self.running: break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = json.loads(msg.data)
                                    data = payload.get("data", payload)
                                    if data.get('e') == 'kline':
                                        self._process_kline(data)
                                except Exception: pass
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR): break
            except Exception:
                await asyncio.sleep(min(backoff, 60))
                backoff *= 1.5

    def _process_kline(self, data):
        try:
            k = data['k']; tf = k['i']
            if tf not in self.matrices: return
            self.matrices[tf].update_candle(
                data['s'], float(k['c']), float(k['v']), float(k['h']), float(k['l']),
                int(k['t']), bool(k['x'])
            )
        except Exception: pass

    def _update_indicators_job(self, matrix: MarketMatrix):
        """Runs in ThreadPool."""
        try:
            with matrix.lock:
                c = np.ascontiguousarray(matrix.closes)
                v = np.ascontiguousarray(matrix.volumes)
                h = np.ascontiguousarray(matrix.highs)
                l = np.ascontiguousarray(matrix.lows)
            
            calc_indicators_inplace(
                c, v, h, l, 
                matrix.rsi, matrix.mfi, matrix.adx, matrix.vol_z
            )
        except Exception as e:
            logger.error(f"Indicator Job Failed: {e}")

    async def calculation_loop(self):
        logger.info("Starting Math Loop")
        while self.running:
            start = time.time()
            futures = [
                self.executor.submit(self._update_indicators_job, m) 
                for m in self.matrices.values()
            ]
            if futures:
                await asyncio.gather(*[asyncio.wrap_future(f) for f in futures])
            
            elapsed = time.time() - start
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
        while self.running:
            await asyncio.sleep(300)
            try:
                tf = ['1m', '5m', '15m', '1h', '2h', '4h', '1d', '3d', '1w', '1M']

                active_tfs = []
                total_symbols = 0

                for t in tf:
                    if t in self.matrices:
                        active_tfs.append(t)
                        total_symbols += self.matrices[t].n_symbols

                count = len(active_tfs)
                tf_list = ", ".join(active_tfs)

                logger.info(
                    f"Health: {count} matrices active, total symbols = {total_symbols}. "
                    f"Active TFs: {tf_list}"
                )

            except Exception:
                pass


    async def run(self):
        logger.info("🚀 Data Collector Starting...")
        force_refresh = env_flag("FORCE_REFRESH")
        
        if not force_refresh and await self.load_state():
            pass
        else:
            await self.get_top_100_coins()
            await self.fetch_historical_snapshot()
        
        tasks = []
        all_streams = []
        for s in self.top_coins:
            for tf in self.active_timeframes:
                all_streams.append(f"{s.lower()}@kline_{tf}")
                
        chunks = [all_streams[i:i + 200] for i in range(0, len(all_streams), 200)]
        for i, chunk in enumerate(chunks):
            tasks.append(asyncio.create_task(self._single_spot_ws_handler(chunk, i)))

        tasks.append(asyncio.create_task(self.calculation_loop()))
        tasks.append(asyncio.create_task(self.periodic_save_task()))
        tasks.append(asyncio.create_task(self.futures_depth.run()))
        tasks.append(asyncio.create_task(self.health_log_task()))
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.critical(f"Main Loop Crash: {e}")
        finally:
            self.executor.shutdown(wait=False)

if __name__ == "__main__":
    bot = BinanceMarketDataCollector()
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        pass
