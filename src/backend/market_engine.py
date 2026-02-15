# Market_Engine.py
import time
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict, Counter

# ------------------------------------------------------------------------------
# Config Management (Fail-safe)
# ------------------------------------------------------------------------------
try:
    from config import config, setup_logging
except ImportError:
    raise ImportError("CRITICAL: config.py not found. Please ensure config.py is in the root directory.")

# Init Logger
logger = setup_logging("MarketEngine")


# ==============================================================================
# 🚨 DATA STRUCTURES
# ==============================================================================

@dataclass
class MarketEvent:
    symbol: str
    type: str       # 'PUMP', 'DUMP', 'SPIKE'
    timeframe: str
    price: float
    change_pct: float
    volume_ratio: float
    rsi: float
    score: int      # 0-100
    timestamp: float
    urgency: int    # 1=Med, 2=High


# ==============================================================================
# ⚙️ MARKET ENGINE (Vectorized, Futures-Aware)
# ==============================================================================

class MarketEngine:
    def __init__(self, collector):
        self.collector = collector
        self.alerts_history: List[MarketEvent] = []
        self.max_history = 100
        self.lock = asyncio.Lock()
        
        # Debounce State: {symbol: {tf: {'last_ts': 0.0, 'last_score': 0}}}
        self.signal_states = defaultdict(lambda: defaultdict(dict))
        
        # Hyper-Sensitivity Tracking: {symbol: count}
        # Tracks how many times a symbol has alerted recently to boost its score
        self.hot_symbols = defaultdict(int)

        # Scanning settings (match collector timeframes)
        self.scan_tfs = ['1m', '5m', '1h', '1d']
        self.cooldown_map = {
            '1m': 60,
            '5m': 300,
            '1h': 3600,
            '1d': 43200,
        }

    async def start_background_task(self):
        """Main loop that runs analysis periodically."""
        logger.info("🚀 Market Engine Analysis Loop Started")
        while True:
            try:
                await self.scan_markets_vectorized()
                await asyncio.sleep(getattr(config, 'ALERT_REFRESH_RATE', 2.0))
            except Exception as e:
                logger.error(f"Analysis Loop Crash: {e}")
                await asyncio.sleep(5)

    async def scan_markets_vectorized(self):
        """
        ⚡ TRUE VECTORIZATION ⚡
        Analyzes ALL symbols simultaneously using NumPy array operations.
        """
        new_events = []
        now_ts = time.time()
        
        pump_thr = getattr(config, 'PUMP_THRESHOLD_PCT', 2.5)
        dump_thr = getattr(config, 'DUMP_THRESHOLD_PCT', -2.5)
        spike_thr = getattr(config, 'VOLUME_SPIKE_RATIO', 3.0)
        atr_min_moves = getattr(config, 'MIN_ATR_MOVES', 2.0)

        # Clean up hot symbols counter every cycle (decay)
        if len(self.hot_symbols) > 200:
            self.hot_symbols.clear()

        for tf in self.scan_tfs:
            if tf not in self.collector.matrices:
                continue
            
            matrix = self.collector.matrices[tf]
            
            # --- 1. Thread-Safe Snapshot (Fast Copy) ---
            with matrix.lock:
                if matrix.closes.shape[1] < 2:
                    continue
                
                closes = matrix.closes[:, -1].copy()
                opens = matrix.closes[:, -2].copy()
                rsi = matrix.rsi.copy()
                vol_z = matrix.vol_z.copy()
                atr = getattr(matrix, "atr", np.zeros_like(closes))
                funding = getattr(matrix, "funding_rate", np.zeros_like(closes))
                oi = getattr(matrix, "open_interest", np.zeros_like(closes))
                bid_liq = getattr(matrix, "liquidity_bid_5pct", np.zeros_like(closes))
                ask_liq = getattr(matrix, "liquidity_ask_5pct", np.zeros_like(closes))
                timestamps = matrix.timestamps[:, -1].copy()
                symbols = matrix.symbols
            
            # --- 2. Vectorized Math ---
            valid_mask = (opens > 0)
            if not np.any(valid_mask):
                continue
                
            pct_changes = np.zeros_like(closes)
            np.divide((closes - opens), opens, out=pct_changes, where=valid_mask)
            pct_changes *= 100.0

            # ATR-normalized move (how many ATRs the candle moved)
            price_range = np.abs(closes - opens)
            atr_norm = np.zeros_like(closes)
            np.divide(price_range, atr, out=atr_norm, where=(atr > 0))

            # --- 3. Boolean Logic Masks (ATR-aware) ---
            mask_pump = (pct_changes >= pump_thr) & (atr_norm >= atr_min_moves)
            mask_dump = (pct_changes <= dump_thr) & (atr_norm >= atr_min_moves)

            # Noise filter: Spike = strong volume + decent % move or strong ATR move
            mask_spike = (
                (vol_z >= spike_thr) &
                ((np.abs(pct_changes) >= 0.5) | (atr_norm >= atr_min_moves))
            )

            all_triggers = mask_pump | mask_dump | mask_spike
            hit_indices = np.where(all_triggers)[0]

            if len(hit_indices) == 0:
                continue

            # --- 4. Event Generation ---
            for idx in hit_indices:
                sym = symbols[idx]
                pc = float(pct_changes[idx])
                vz = float(vol_z[idx])
                r = float(rsi[idx])
                ts = float(timestamps[idx])
                a = float(atr[idx])
                f = float(funding[idx])
                oi_val = float(oi[idx])
                bid_l = float(bid_liq[idx])
                ask_l = float(ask_liq[idx])

                # Determine Type & Base Score
                if mask_pump[idx]:
                    etype = 'PUMP'
                    score = 50.0 + (vz * 5.0)
                    if r > 70:
                        score += 10.0
                elif mask_dump[idx]:
                    etype = 'DUMP'
                    score = 50.0 + (vz * 5.0)
                    if r < 30:
                        score += 10.0
                else:
                    etype = 'SPIKE'
                    score = 40.0 + (vz * 5.0)

                # --- Volatility confidence: ATR moves ---
                if a > 0:
                    atr_moves = abs((closes[idx] - opens[idx]) / a)
                    score += min(atr_moves * 2.0, 15.0)

                # --- Liquidity bonus: big moves with big ±5% liquidity ---
                liq_total = bid_l + ask_l
                if liq_total > 0:
                    liq_score = min(liq_total / 1_000_000.0 * 5.0, 15.0)
                    score += liq_score

                # --- Funding bias: extreme funding in direction ---
                if etype == 'PUMP' and f > 0:
                    score += min(f * 50.0, 10.0)  # funding in %
                elif etype == 'DUMP' and f < 0:
                    score += min(abs(f) * 50.0, 10.0)

                # --- Open interest: larger OI => more meaningful ---
                if oi_val > 0:
                    oi_bonus = min(oi_val / 10_000_000.0 * 5.0, 10.0)
                    score += oi_bonus

                # 🔥 SENSITIVITY BOOST 🔥
                boost = self.hot_symbols[sym] * 5.0
                score += boost

                # Cap score
                score = float(min(100.0, score))

                # --- 5. Debounce Check ---
                if not self._should_alert(sym, tf, score, now_ts):
                    continue

                # Mark as hot
                self.hot_symbols[sym] += 1

                new_events.append(MarketEvent(
                    symbol=sym,
                    type=etype,
                    timeframe=tf,
                    price=float(closes[idx]),
                    change_pct=pc,
                    volume_ratio=vz,
                    rsi=r,
                    score=int(score),
                    timestamp=ts,
                    urgency=2 if score > 75 else 1
                ))

        # --- 6. Update History & LOGGING ---
        if new_events:
            new_events.sort(key=lambda x: x.score, reverse=True)
            
            async with self.lock:
                self.alerts_history.extend(new_events)
                if len(self.alerts_history) > self.max_history:
                    self.alerts_history = self.alerts_history[-self.max_history:]
            
            # 1. Count alerts per timeframe
            tf_counts = Counter(ev.timeframe for ev in new_events)
            tf_summary = ", ".join([f"{k}:{v}" for k, v in tf_counts.items()])
            
            # 2. Compact vs Detailed Log
            count_str = f"⚡ {len(new_events)} alerts [{tf_summary}]"
            
            if len(new_events) <= 5:
                # One-liner mode
                details = []
                for ev in new_events:
                    icon = "🟢" if ev.type == "PUMP" else ("🔴" if ev.type == "DUMP" else "🟡")
                    details.append(f"{icon} {ev.symbol} [{ev.timeframe}] {ev.change_pct:+.1f}%")
                
                logger.info(f"{count_str}: {', '.join(details)}")
            else:
                # Multiline mode for spam
                logger.info(count_str)
                for ev in new_events[:5]:
                    icon = "🟢" if ev.type == "PUMP" else ("🔴" if ev.type == "DUMP" else "🟡")
                    logger.info(
                        f"   {icon} {ev.symbol} [{ev.timeframe}] {ev.type} "
                        f"({ev.change_pct:+.2f}%) | Score: {ev.score}"
                    )
                logger.info(f"   ... and {len(new_events) - 5} others.")

    def _should_alert(self, symbol, tf, score, now_ts) -> bool:
        state = self.signal_states[symbol][tf]
        last_ts = state.get('last_ts', 0.0)
        last_score = state.get('last_score', 0.0)
        cooldown = self.cooldown_map.get(tf, 60)

        time_diff = now_ts - last_ts
        is_ready = False
        
        if time_diff > cooldown:
            is_ready = True
        elif score > (last_score + 15.0):
            is_ready = True 
        
        if is_ready:
            state['last_ts'] = now_ts
            state['last_score'] = score
            return True
        return False

    async def get_leaderboard(self, timeframe: str, sort_by: str = 'change_desc', limit: int = 10) -> List[Dict]:
        if timeframe not in self.collector.matrices:
            return []
            
        matrix = self.collector.matrices[timeframe]
        
        with matrix.lock:
            closes = matrix.closes[:, -1]
            opens = matrix.closes[:, -2]
            vols = matrix.volumes[:, -1]
            rsi = matrix.rsi
            vol_z = matrix.vol_z
            atr = getattr(matrix, "atr", np.zeros_like(closes))
            funding = getattr(matrix, "funding_rate", np.zeros_like(closes))
            oi = getattr(matrix, "open_interest", np.zeros_like(closes))
            symbols = matrix.symbols
        
        # Metric selection
        if sort_by.startswith('change'):
            valid = opens > 0
            metric = np.zeros_like(closes)
            np.divide((closes - opens), opens, out=metric, where=valid)
            metric *= 100.0
        elif sort_by == 'volume':
            metric = vol_z
        elif sort_by == 'usdt_volume':
            metric = closes * vols
        elif sort_by == 'hot':
            metric = np.abs(vol_z) * (np.abs(rsi - 50.0) / 50.0)
        elif sort_by == 'atr_move':
            price_range = np.abs(closes - opens)
            metric = np.zeros_like(closes)
            np.divide(price_range, atr, out=metric, where=(atr > 0))
        else:
            metric = np.zeros_like(closes)

        n = len(metric)
        if n == 0:
            return []
        limit = min(limit, n)
        
        if 'asc' in sort_by:
            top_indices = np.argpartition(metric, limit)[:limit]
            top_indices = top_indices[np.argsort(metric[top_indices])]
        else:
            top_indices = np.argpartition(metric, -limit)[-limit:]
            top_indices = top_indices[np.argsort(metric[top_indices])[::-1]]

        results = []
        for idx in top_indices:
            prev = opens[idx]
            chg = ((closes[idx] - prev) / prev * 100.0) if prev > 0 else 0.0
            results.append({
                'symbol': symbols[idx],
                'change': float(chg),
                'price': float(closes[idx]),
                'volume': float(vols[idx]),
                'usdt_volume': float(closes[idx] * vols[idx]),
                'rsi': float(rsi[idx]),
                'vol_zscore': float(vol_z[idx]),
                'atr': float(atr[idx]),
                'funding_rate': float(funding[idx]),
                'open_interest': float(oi[idx]),
                'hot_score': float(abs(vol_z[idx]) * abs(rsi[idx] - 50.0) / 50.0),
            })
            
        return results


if __name__ == "__main__":
    pass
