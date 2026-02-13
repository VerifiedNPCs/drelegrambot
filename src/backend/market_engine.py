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
# ⚙️ MARKET ENGINE (Vectorized)
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

        # Scanning settings
        self.scan_tfs = ['1m', '5m', '15m', '1h', '2h', '4h', '1d', '3d', '1w', '1M']
        self.cooldown_map = {
            '1m': 60,     
            '5m': 300,    
            '15m': 900,   
            '1h': 3600,  
            '2h': 7200,  
            '4h': 14400,  
            '1d': 43200,  
            '3d': 86400,  
            '1w': 86400,
            '1M': 86400   
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

        # Clean up hot symbols counter every cycle (decay)
        # This prevents infinite sensitivity
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
                timestamps = matrix.timestamps[:, -1].copy()
                symbols = matrix.symbols 
            
            # --- 2. Vectorized Math ---
            valid_mask = (opens > 0)
            if not np.any(valid_mask):
                continue
                
            pct_changes = np.zeros_like(closes)
            np.divide((closes - opens), opens, out=pct_changes, where=valid_mask)
            pct_changes *= 100

            # --- 3. Boolean Logic Masks ---
            mask_pump = (pct_changes >= pump_thr)
            mask_dump = (pct_changes <= dump_thr)
            
            # Noise filter: Spike requires vol_z > 3 AND at least 0.5% move
            mask_spike = (vol_z >= spike_thr) & (np.abs(pct_changes) >= 0.5)

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
                
                # Determine Type & Base Score
                if mask_pump[idx]:
                    etype = 'PUMP'
                    score = 50 + (vz * 5)
                    if r > 70: score += 10
                elif mask_dump[idx]:
                    etype = 'DUMP'
                    score = 50 + (vz * 5)
                    if r < 30: score += 10
                else:
                    etype = 'SPIKE'
                    score = 40 + (vz * 5)

                # 🔥 SENSITIVITY BOOST 🔥
                # If symbol triggered recently, boost score to ensure it bypasses "escalation" check
                boost = self.hot_symbols[sym] * 5 
                score += boost
                # Cap score at 100
                score = min(100, score)

                # --- 5. Debounce Check ---
                if not self._should_alert(sym, tf, score, now_ts):
                    continue
                
                # Mark as hot
                self.hot_symbols[sym] += 1
                
                new_events.append(MarketEvent(
                    symbol=sym, type=etype, timeframe=tf,
                    price=float(closes[idx]), change_pct=pc,
                    volume_ratio=vz, rsi=r, score=int(score),
                    timestamp=ts, urgency=2 if score > 75 else 1
                ))

        # --- 6. Update History & LOGGING ---
        if new_events:
            new_events.sort(key=lambda x: x.score, reverse=True)
            
            async with self.lock:
                self.alerts_history.extend(new_events)
                if len(self.alerts_history) > self.max_history:
                    self.alerts_history = self.alerts_history[-self.max_history:]
            
            # --- CUSTOM LOGGING LOGIC ---
            
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
                    # Short format: 🟢 BTC [1m] +2.5%
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
        last_ts = state.get('last_ts', 0)
        last_score = state.get('last_score', 0)
        cooldown = self.cooldown_map.get(tf, 60)

        time_diff = now_ts - last_ts
        is_ready = False
        
        if time_diff > cooldown:
            is_ready = True
        elif score > (last_score + 15):
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
            symbols = matrix.symbols
        
        if sort_by.startswith('change'):
            valid = opens > 0
            metric = np.zeros_like(closes)
            np.divide((closes - opens), opens, out=metric, where=valid)
            metric *= 100
        elif sort_by == 'volume':
            metric = vol_z
        elif sort_by == 'usdt_volume':
            metric = closes * vols
        elif sort_by == 'hot':
            metric = np.abs(vol_z) * (np.abs(rsi - 50) / 50)
        else:
            metric = np.zeros_like(closes)

        n = len(metric)
        if n == 0: return []
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
            chg = ((closes[idx] - prev) / prev * 100) if prev > 0 else 0.0
            results.append({
                'symbol': symbols[idx],
                'change': float(chg),
                'price': float(closes[idx]),
                'volume': float(vols[idx]),
                'usdt_volume': float(closes[idx] * vols[idx]),
                'rsi': float(rsi[idx]),
                'vol_zscore': float(vol_z[idx]),
                'hot_score': float(abs(vol_z[idx]) * abs(rsi[idx] - 50) / 50)
            })
            
        return results

if __name__ == "__main__":
    pass
