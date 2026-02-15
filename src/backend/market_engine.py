# Market_Engine.py (enhanced)
import time
import asyncio
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
from collections import defaultdict, Counter

# -------------------------------------------------------------------------------
# Config Management (Fail-safe)
# -------------------------------------------------------------------------------
try:
    from config import config, setup_logging
except ImportError:
    class Config:
        ALERT_REFRESH_RATE = 2.0
        PUMP_THRESHOLD_PCT = 2.5
        DUMP_THRESHOLD_PCT = -2.5
        VOLUME_SPIKE_RATIO = 3.0
        MIN_ATR_MOVES = 2.0
    config = Config()

    def setup_logging(name):
        return logging.getLogger(name)

logger = setup_logging("MarketEngine")

# =======================================================================
# 🚨 DATA STRUCTURES
# =======================================================================
@dataclass
class MarketEvent:
    symbol: str
    type: str       # 'PUMP', 'DUMP', 'SPIKE', 'LIQ', 'LIQ_DEPTH'
    timeframe: str
    price: float
    change_pct: float
    volume_ratio: float
    rsi: float
    score: int      # 0-100
    timestamp: float
    urgency: int    # 1=Med, 2=High

    # Extra context for UI/Logs
    funding_rate: float = 0.0
    oi_value: float = 0.0
    liq_5pct: float = 0.0
    depth_down_pct: Optional[float] = None
    depth_up_pct: Optional[float] = None
    depth_bid_notional: Optional[float] = None
    depth_ask_notional: Optional[float] = None
    liq_24h_long: Optional[float] = None
    liq_24h_short: Optional[float] = None

# =======================================================================
# ⚙️ MARKET ENGINE (Vectorized, Futures-Aware, Liquidity & Liquidation)
# =======================================================================
class MarketEngine:
    def __init__(self, collector):
        """
        collector: instance of BinanceMarketDataCollector (or similar)
        which exposes:
          - collector.matrices (dict of MarketMatrix per timeframe)
          - collector.liq_monitor.get_liquidation_data(symbol)
          - collector.get_depth_liquidity_range(symbol, target_usdt)
        """
        self.collector = collector
        self.alerts_history: List[MarketEvent] = []
        self.max_history = 200
        self.lock = asyncio.Lock()

        # Debounce State: {symbol: {tf: {'last_ts': 0.0, 'last_score': 0}}}
        self.signal_states = defaultdict(lambda: defaultdict(dict))

        # Hyper-Sensitivity Tracking: {symbol: count}
        self.hot_symbols = defaultdict(int)

        # Scanning settings (match collector timeframes)
        self.scan_tfs = ['1m', '5m', '15m', '1h', '4h', '1d']
        self.cooldown_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400,
        }

        # Liquidity thresholds (USDT) used to boost scores
        self.liq_boost_thresholds = [250_000, 1_000_000, 5_000_000]  # example tiers

    async def start_background_task(self):
        """Main loop that runs analysis periodically."""
        logger.info("🚀 Market Engine Analysis Loop Started")
        while True:
            try:
                await self.scan_markets_vectorized()
                await asyncio.sleep(getattr(config, 'ALERT_REFRESH_RATE', 2.0))
            except Exception as e:
                logger.error(f"Analysis Loop Crash: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def scan_markets_vectorized(self):
        """
        Vectorized analysis across all timeframes and symbols.
        Enhancements:
          - Enrich events with futures metrics (funding, OI)
          - Add liquidity depth signals (how far to hit a target wall)
          - Add liquidation-based signals (large 1h/24h liq imbalance)
        """
        new_events: List[MarketEvent] = []
        now_ts = time.time()

        pump_thr = getattr(config, 'PUMP_THRESHOLD_PCT', 2.5)
        dump_thr = getattr(config, 'DUMP_THRESHOLD_PCT', -2.5)
        spike_thr = getattr(config, 'VOLUME_SPIKE_RATIO', 3.0)
        atr_min_moves = getattr(config, 'MIN_ATR_MOVES', 2.0)

        # Decay hot symbols map occasionally
        if len(self.hot_symbols) > 1000:
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
                atr = matrix.atr.copy()

                funding = matrix.funding_rate.copy()
                oi = matrix.open_interest.copy()
                bid_liq = matrix.liquidity_bid_5pct.copy()
                ask_liq = matrix.liquidity_ask_5pct.copy()

                timestamps = matrix.timestamps[:, -1].copy()
                symbols = matrix.symbols

            # --- 2. Vectorized Math ---
            valid_mask = (opens > 0)
            if not np.any(valid_mask):
                continue

            pct_changes = np.zeros_like(closes)
            np.divide((closes - opens), opens, out=pct_changes, where=valid_mask)
            pct_changes *= 100.0

            price_range = np.abs(closes - opens)
            atr_norm = np.zeros_like(closes)
            np.divide(price_range, atr, out=atr_norm, where=(atr > 0))

            # --- 3. Boolean Logic Masks (ATR-aware) ---
            mask_pump = (pct_changes >= pump_thr) & (atr_norm >= atr_min_moves)
            mask_dump = (pct_changes <= dump_thr) & (atr_norm >= atr_min_moves)
            mask_spike = (
                (vol_z >= spike_thr) &
                ((np.abs(pct_changes) >= 0.5) | (atr_norm >= atr_min_moves))
            )

            all_triggers = mask_pump | mask_dump | mask_spike
            hit_indices = np.where(all_triggers)[0]

            if len(hit_indices) == 0:
                continue

            # --- 4. Event Generation & Enrichment ---
            for idx in hit_indices:
                sym = symbols[idx]
                pc = float(pct_changes[idx])
                vz = float(vol_z[idx])
                r = float(rsi[idx])
                ts = float(timestamps[idx]) or now_ts
                a = float(atr[idx])
                f = float(funding[idx])
                oi_val = float(oi[idx])
                liq_val = float(bid_liq[idx] + ask_liq[idx])

                # Determine Type & Base Score
                if mask_pump[idx]:
                    etype = 'PUMP'
                    score = 50.0 + (vz * 5.0)
                    if r > 70: score += 10.0
                elif mask_dump[idx]:
                    etype = 'DUMP'
                    score = 50.0 + (vz * 5.0)
                    if r < 30: score += 10.0
                else:
                    etype = 'SPIKE'
                    score = 40.0 + (vz * 5.0)

                # 1. ATR confidence
                if a > 0:
                    atr_moves = abs((closes[idx] - opens[idx]) / a)
                    score += min(atr_moves * 2.0, 15.0)

                # 2. Liquidity bonus (USDT)
                if liq_val > 0:
                    liq_score = min(liq_val / 1_000_000.0 * 1.0, 15.0)
                    score += liq_score

                # 3. Funding bias
                if etype == 'PUMP' and f > 0.01:
                    score += min(f * 50.0, 10.0)
                elif etype == 'DUMP' and f < -0.01:
                    score += min(abs(f) * 50.0, 10.0)

                # 4. Open Interest
                if oi_val > 0:
                    oi_bonus = min(oi_val / 10_000_000.0 * 1.0, 10.0)
                    score += oi_bonus

                # 5. Hot Symbol Boost
                boost = self.hot_symbols.get(sym, 0) * 5.0
                score += boost

                # 6. Depth-based enrichment (synchronous helper)
                depth_info = None
                try:
                    # collector.get_depth_liquidity_range is expected to be synchronous
                    depth_info = self.collector.get_depth_liquidity_range(sym, target_usdt=250_000)
                except Exception:
                    depth_info = None

                if depth_info:
                    # If a small % move hits the target wall, increase urgency/score
                    down_pct = depth_info.get('down_pct', 5.0)
                    up_pct = depth_info.get('up_pct', 5.0)
                    bid_notional = depth_info.get('bid_notional', 0.0)
                    ask_notional = depth_info.get('ask_notional', 0.0)

                    # If a large wall exists within 1%, boost score
                    if down_pct <= 1.0 or up_pct <= 1.0:
                        score += 10.0
                    # Add liquidity magnitude bonus
                    wall_liq = max(bid_notional, ask_notional)
                    score += min(wall_liq / 1_000_000.0 * 0.5, 10.0)

                # 7. Liquidation imbalance signals (use liq monitor)
                liq_stats = {}
                try:
                    liq_stats = self.collector.get_liq_stats(sym)
                except Exception:
                    liq_stats = {}

                # If 1h or 24h side is heavily skewed, increase score
                try:
                    l1h_long = liq_stats.get('1h_long', 0.0)
                    l1h_short = liq_stats.get('1h_short', 0.0)
                    l24_long = liq_stats.get('24h_long', 0.0)
                    l24_short = liq_stats.get('24h_short', 0.0)

                    # If recent 1h long liquidations are much larger than shorts, treat as bearish pressure
                    if l1h_long > (l1h_short * 3.0) and l1h_long > 50_000:
                        # heavy long liquidations -> price may drop further
                        if etype == 'DUMP':
                            score += 12.0
                        else:
                            score += 6.0

                    if l1h_short > (l1h_long * 3.0) and l1h_short > 50_000:
                        if etype == 'PUMP':
                            score += 12.0
                        else:
                            score += 6.0
                except Exception:
                    pass

                # Cap score
                score = float(min(100.0, score))

                # Debounce
                if not self._should_alert(sym, tf, score, now_ts):
                    continue

                # Mark as hot
                self.hot_symbols[sym] += 1

                # Build event with enrichment
                ev = MarketEvent(
                    symbol=sym,
                    type=etype,
                    timeframe=tf,
                    price=float(closes[idx]),
                    change_pct=pc,
                    volume_ratio=vz,
                    rsi=r,
                    score=int(score),
                    timestamp=ts,
                    urgency=2 if score > 75 else 1,
                    funding_rate=f,
                    oi_value=oi_val,
                    liq_5pct=liq_val
                )

                if depth_info:
                    ev.depth_down_pct = depth_info.get('down_pct')
                    ev.depth_up_pct = depth_info.get('up_pct')
                    ev.depth_bid_notional = depth_info.get('bid_notional')
                    ev.depth_ask_notional = depth_info.get('ask_notional')

                if liq_stats:
                    ev.liq_24h_long = liq_stats.get('24h_long')
                    ev.liq_24h_short = liq_stats.get('24h_short')

                new_events.append(ev)

        # --- 5. Update History & LOGGING ---
        if new_events:
            new_events.sort(key=lambda x: x.score, reverse=True)

            async with self.lock:
                self.alerts_history.extend(new_events)
                if len(self.alerts_history) > self.max_history:
                    self.alerts_history = self.alerts_history[-self.max_history:]

            tf_counts = Counter(ev.timeframe for ev in new_events)
            tf_summary = ", ".join([f"{k}:{v}" for k, v in tf_counts.items()])
            count_str = f"⚡ {len(new_events)} alerts [{tf_summary}]"

            if len(new_events) <= 6:
                details = []
                for ev in new_events:
                    icon = "🟢" if ev.type == "PUMP" else ("🔴" if ev.type == "DUMP" else "🟡")
                    depth_hint = ""
                    if ev.depth_down_pct is not None:
                        depth_hint = f" | depth -{ev.depth_down_pct:.2f}%/+{ev.depth_up_pct:.2f}%"
                    details.append(f"{icon} {ev.symbol} {ev.change_pct:+.1f}% ({ev.score}){depth_hint}")
                logger.info(f"{count_str}: {', '.join(details)}")
            else:
                logger.info(count_str)
                for ev in new_events[:3]:
                    icon = "🟢" if ev.type == "PUMP" else ("🔴" if ev.type == "DUMP" else "🟡")
                    logger.info(f"   {icon} {ev.symbol} [{ev.timeframe}] {ev.type} {ev.change_pct:+.1f}% | Score: {ev.score}")

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
            atr = matrix.atr
            funding = matrix.funding_rate
            oi = matrix.open_interest
            symbols = matrix.symbols

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
        if n == 0: return []
        limit = min(limit, n)

        if 'asc' in sort_by:
            unsorted_top = np.argpartition(metric, limit)[:limit]
            top_indices = unsorted_top[np.argsort(metric[unsorted_top])]
        else:
            unsorted_top = np.argpartition(metric, -limit)[-limit:]
            top_indices = unsorted_top[np.argsort(metric[unsorted_top])[::-1]]

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

# If run as script for quick smoke test (non-blocking)
if __name__ == "__main__":
    # Minimal smoke test: requires a collector instance to be meaningful.
    logger.info("MarketEngine module loaded. Instantiate MarketEngine(collector) to run.")
