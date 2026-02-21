from __future__ import annotations

# Standard Library

import asyncio
import collections
from collections import deque
import datetime
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
import json
import logging
import os
import random
import threading
import time
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Set, TypedDict

# Third‑Party Libraries

import aiohttp
from aiohttp import ClientSession, ClientResponse
from multidict import CIMultiDictProxy
import numpy as np
import orjson
import psutil as _ps

# End of imports

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# --- Replace existing calc_indicators_inplace with this version ---
try:
    from numba import njit, prange

    @njit(parallel=True)
    def calc_indicators_inplace(
        closes, volumes, highs, lows, counts,
        out_rsi, out_mfi, out_adx, out_vol_z, out_atr,
        out_ema_7, out_ema_14, out_ema_21, out_ema_50, out_ema_200,
        out_macd, out_macd_signal, out_macd_hist,
        out_stoch_k, out_stoch_d,
        out_vwap,
        out_supertrend, out_supertrend_dir,
        full: bool = False
    ):
        """
        Numba-compiled indicator calculation.
        - counts: 1D int array with per-symbol number of CLOSED candles (<= n_cols)
        - If full is False: compute only the last column (existing fast path).
        - If full is True: compute every valid column for each symbol (warmup).
        """
        n_symbols, n_cols = closes.shape

        # Replace NaN with 0 for safety inside the compiled function
        for i in prange(n_symbols):
            for j in range(n_cols):
                if np.isnan(closes[i, j]):
                    closes[i, j] = 0.0
                if np.isnan(volumes[i, j]):
                    volumes[i, j] = 0.0
                if np.isnan(highs[i, j]):
                    highs[i, j] = 0.0
                if np.isnan(lows[i, j]):
                    lows[i, j] = 0.0

        # Helper: compute EMA series in-place into out_arr (period > 0)
        def _ema_series(row, period, out_arr, valid_len):
            if valid_len <= 0:
                return
            alpha = 2.0 / (period + 1.0)
            val = row[0]
            out_arr[0] = val
            for jj in range(1, valid_len):
                val = alpha * row[jj] + (1.0 - alpha) * val
                out_arr[jj] = val
            # fill remaining positions with last computed value for safety
            for jj in range(valid_len, n_cols):
                out_arr[jj] = val

        # Full-history mode: compute per-column series up to counts[i]
        if full:
            for i in prange(n_symbols):
                c = closes[i]
                v = volumes[i]
                h = highs[i]
                l = lows[i]
                valid = counts[i]
                if valid <= 0:
                    continue
                # Bound valid to n_cols
                if valid > n_cols:
                    valid = n_cols

                # EMAs
                _ema_series(c, 7, out_ema_7[i], valid)
                _ema_series(c, 14, out_ema_14[i], valid)
                _ema_series(c, 21, out_ema_21[i], valid)
                _ema_series(c, 50, out_ema_50[i], valid)
                _ema_series(c, 200, out_ema_200[i], valid)

                # MACD (ema12 - ema26) using out_macd and out_macd_signal as temps
                _ema_series(c, 12, out_macd[i], valid)
                _ema_series(c, 26, out_macd_signal[i], valid)
                for jj in range(valid):
                    out_macd_hist[i, jj] = out_macd[i, jj] - out_macd_signal[i, jj]
                    out_macd[i, jj] = out_macd[i, jj] - out_macd_signal[i, jj]
                for jj in range(valid, n_cols):
                    out_macd[i, jj] = 0.0
                    out_macd_signal[i, jj] = 0.0
                    out_macd_hist[i, jj] = 0.0

                # VWAP rolling 50 (naive)
                win_vwap = 50
                for jj in range(valid):
                    s_vwap = jj - win_vwap + 1
                    if s_vwap < 0:
                        s_vwap = 0
                    pv_sum = 0.0
                    vol_sum = 0.0
                    for k in range(s_vwap, jj + 1):
                        tp = (h[k] + l[k] + c[k]) / 3.0
                        pv_sum += tp * v[k]
                        vol_sum += v[k]
                    if vol_sum > 0.0:
                        out_vwap[i, jj] = pv_sum / vol_sum
                    else:
                        out_vwap[i, jj] = c[jj]
                for jj in range(valid, n_cols):
                    out_vwap[i, jj] = 0.0

                # ATR rolling 14
                period_atr = 14
                for jj in range(1, valid):
                    s_atr = jj - period_atr + 1
                    if s_atr < 1:
                        s_atr = 1
                    tr_sum = 0.0
                    cnt = 0
                    for k in range(s_atr, jj + 1):
                        high = h[k]
                        low = l[k]
                        prev_close = c[k - 1]
                        hl = high - low
                        hc = high - prev_close
                        if hc < 0.0:
                            hc = -hc
                        lc = low - prev_close
                        if lc < 0.0:
                            lc = -lc
                        tr = hl
                        if hc > tr:
                            tr = hc
                        if lc > tr:
                            tr = lc
                        tr_sum += tr
                        cnt += 1
                    if cnt > 0:
                        out_atr[i, jj] = tr_sum / cnt
                    else:
                        out_atr[i, jj] = 0.0
                for jj in range(valid, n_cols):
                    out_atr[i, jj] = 0.0

                # RSI rolling 14 (simple)
                period_rsi = 14
                for jj in range(1, valid):
                    s_r = jj - period_rsi + 1
                    if s_r < 1:
                        s_r = 1
                    avg_gain = 0.0
                    avg_loss = 0.0
                    cnt = 0
                    for k in range(s_r, jj + 1):
                        ch = c[k] - c[k - 1]
                        if ch > 0.0:
                            avg_gain += ch
                        else:
                            avg_loss -= ch
                        cnt += 1
                    if cnt > 0:
                        if avg_loss == 0.0:
                            out_rsi[i, jj] = 100.0
                        else:
                            rs = avg_gain / avg_loss
                            out_rsi[i, jj] = 100.0 - (100.0 / (1.0 + rs))
                    else:
                        out_rsi[i, jj] = 50.0
                for jj in range(valid, n_cols):
                    out_rsi[i, jj] = 0.0

                # Stochastic (14)
                period_stoch = 14
                for jj in range(valid):
                    s_st = jj - period_stoch + 1
                    if s_st < 0:
                        s_st = 0
                    hh = h[s_st]
                    ll = l[s_st]
                    for k in range(s_st + 1, jj + 1):
                        if h[k] > hh:
                            hh = h[k]
                        if l[k] < ll:
                            ll = l[k]
                    if hh != ll:
                        out_stoch_k[i, jj] = 100.0 * (c[jj] - ll) / (hh - ll)
                        out_stoch_d[i, jj] = out_stoch_k[i, jj]
                    else:
                        out_stoch_k[i, jj] = 0.0
                        out_stoch_d[i, jj] = 0.0
                for jj in range(valid, n_cols):
                    out_stoch_k[i, jj] = 0.0
                    out_stoch_d[i, jj] = 0.0

                # Volume Z-score (20)
                count = 20
                for jj in range(valid):
                    s_v = jj - count + 1
                    if s_v < 0:
                        s_v = 0
                    v_sum = 0.0
                    v_sq_sum = 0.0
                    cnt = 0
                    for k in range(s_v, jj):
                        val = v[k]
                        v_sum += val
                        v_sq_sum += val * val
                        cnt += 1
                    if cnt > 5:
                        v_mean = v_sum / cnt
                        v_var = (v_sq_sum / cnt) - (v_mean * v_mean)
                        if v_var > 0.0:
                            v_std = np.sqrt(v_var)
                        else:
                            v_std = 1.0
                        out_vol_z[i, jj] = (v[jj] - v_mean) / v_std
                    else:
                        out_vol_z[i, jj] = 0.0
                for jj in range(valid, n_cols):
                    out_vol_z[i, jj] = 0.0

                # Supertrend simple per-column using ATR value
                for jj in range(valid):
                    if out_atr[i, jj] > 0.0:
                        multiplier = 3.0
                        upper = c[jj] + multiplier * out_atr[i, jj]
                        lower = c[jj] - multiplier * out_atr[i, jj]
                        mid = (upper + lower) * 0.5
                        if c[jj] >= mid:
                            out_supertrend[i, jj] = lower
                            out_supertrend_dir[i, jj] = 1.0
                        else:
                            out_supertrend[i, jj] = upper
                            out_supertrend_dir[i, jj] = -1.0
                    else:
                        out_supertrend[i, jj] = c[jj]
                        out_supertrend_dir[i, jj] = 0.0
                for jj in range(valid, n_cols):
                    out_supertrend[i, jj] = 0.0
                    out_supertrend_dir[i, jj] = 0.0

            return

        # Default (existing) last-column-only path
        last = n_cols - 1
        for i in prange(n_symbols):
            c = closes[i]
            v = volumes[i]
            h = highs[i]
            l = lows[i]

            if c[last] == 0.0:
                continue

            # RSI (14) simplified: compute last value only
            period_rsi = 14
            if n_cols > period_rsi:
                avg_gain = 0.0
                avg_loss = 0.0
                start = max(1, n_cols - period_rsi - 1)
                for j in range(start + 1, n_cols):
                    ch = c[j] - c[j - 1]
                    if ch > 0:
                        avg_gain += ch
                    else:
                        avg_loss -= ch
                if avg_loss == 0.0:
                    out_rsi[i, last] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    out_rsi[i, last] = 100.0 - (100.0 / (1.0 + rs))

            # MFI (14) simplified
            period_mfi = 14
            if n_cols > period_mfi:
                pos_flow = 0.0
                neg_flow = 0.0
                s_mfi = max(1, n_cols - period_mfi)
                for j in range(s_mfi, n_cols):
                    tp_curr = (h[j] + l[j] + c[j]) / 3.0
                    tp_prev = (h[j - 1] + l[j - 1] + c[j - 1]) / 3.0
                    raw_flow = tp_curr * v[j]
                    if tp_curr > tp_prev:
                        pos_flow += raw_flow
                    elif tp_curr < tp_prev:
                        neg_flow += raw_flow
                if neg_flow == 0.0:
                    out_mfi[i, last] = 100.0
                else:
                    mfi_ratio = pos_flow / neg_flow
                    out_mfi[i, last] = 100.0 - (100.0 / (1.0 + mfi_ratio))

            # ADX (14) simplified
            period_adx = 14
            if n_cols > period_adx:
                tr_sum = 0.0
                dm_pos_sum = 0.0
                dm_neg_sum = 0.0
                s_adx = max(1, n_cols - period_adx)
                for j in range(s_adx, n_cols):
                    up = h[j] - h[j - 1]
                    down = l[j - 1] - l[j]
                    pdm = up if (up > down and up > 0) else 0.0
                    ndm = down if (down > up and down > 0) else 0.0
                    tr = h[j] - l[j]
                    hc = abs(h[j] - c[j - 1])
                    lc = abs(l[j] - c[j - 1])
                    if hc > tr:
                        tr = hc
                    if lc > tr:
                        tr = lc
                    tr_sum += tr
                    dm_pos_sum += pdm
                    dm_neg_sum += ndm
                if tr_sum > 0.0:
                    pdi = 100.0 * dm_pos_sum / tr_sum
                    ndi = 100.0 * dm_neg_sum / tr_sum
                    sum_di = pdi + ndi
                    if sum_di > 0.0:
                        out_adx[i, last] = 100.0 * abs(pdi - ndi) / sum_di
                    else:
                        out_adx[i, last] = 0.0
                else:
                    out_adx[i, last] = 0.0

            # Volume Z-score (20)
            count = 20
            s_v = max(0, n_cols - count)
            v_sum = 0.0
            v_sq_sum = 0.0
            cnt = 0
            for j in range(s_v, n_cols - 1):
                val = v[j]
                v_sum += val
                v_sq_sum += val * val
                cnt += 1
            if cnt > 5:
                v_mean = v_sum / cnt
                v_var = (v_sq_sum / cnt) - (v_mean * v_mean)
                v_std = np.sqrt(v_var) if v_var > 0 else 1.0
                out_vol_z[i, last] = (v[last] - v_mean) / v_std
            else:
                out_vol_z[i, last] = 0.0

            # ATR (14) simplified
            period_atr = 14
            if n_cols > period_atr:
                tr_sum = 0.0
                s_atr = max(1, n_cols - period_atr)
                for j in range(s_atr, n_cols):
                    high = h[j]
                    low = l[j]
                    prev_close = c[j - 1]
                    hl = high - low
                    hc = abs(high - prev_close)
                    lc = abs(low - prev_close)
                    tr = hl
                    if hc > tr:
                        tr = hc
                    if lc > tr:
                        tr = lc
                    tr_sum += tr
                out_atr[i, last] = tr_sum / period_atr if period_atr > 0 else 0.0

            # EMA last values (simple last-value EMA)
            def ema_last(row, period):
                alpha = 2.0 / (period + 1.0)
                val = row[0]
                for j in range(1, n_cols):
                    val = alpha * row[j] + (1.0 - alpha) * val
                return val

            out_ema_7[i, last] = ema_last(c, 7)
            out_ema_14[i, last] = ema_last(c, 14)
            out_ema_21[i, last] = ema_last(c, 21)
            out_ema_50[i, last] = ema_last(c, 50)
            out_ema_200[i, last] = ema_last(c, 200)

            # MACD simplified (12,26,9)
            ema_12 = ema_last(c, 12)
            ema_26 = ema_last(c, 26)
            macd_val = ema_12 - ema_26
            signal_val = macd_val  # crude placeholder
            hist_val = macd_val - signal_val
            out_macd[i, last] = macd_val
            out_macd_signal[i, last] = signal_val
            out_macd_hist[i, last] = hist_val

            # Stochastic (14) simplified
            period_stoch = 14
            if n_cols > period_stoch:
                s_st = n_cols - period_stoch
                hh = h[s_st:n_cols].max()
                ll = l[s_st:n_cols].min()
                if hh != ll:
                    k_val = 100.0 * (c[last] - ll) / (hh - ll)
                else:
                    k_val = 0.0
                out_stoch_k[i, last] = k_val
                out_stoch_d[i, last] = k_val

            # VWAP (rolling 50)
            win_vwap = 50
            s_vwap = max(0, n_cols - win_vwap)
            pv_sum = 0.0
            vol_sum = 0.0
            for j in range(s_vwap, n_cols):
                tp = (h[j] + l[j] + c[j]) / 3.0
                pv_sum += tp * v[j]
                vol_sum += v[j]
            out_vwap[i, last] = pv_sum / vol_sum if vol_sum > 0 else c[last]

            # Supertrend (basic)
            if out_atr[i, last] > 0:
                multiplier = 3.0
                upper = c[last] + multiplier * out_atr[i, last]
                lower = c[last] - multiplier * out_atr[i, last]
                mid = (upper + lower) * 0.5
                if c[last] >= mid:
                    out_supertrend[i, last] = lower
                    out_supertrend_dir[i, last] = 1.0
                else:
                    out_supertrend[i, last] = upper
                    out_supertrend_dir[i, last] = -1.0
            else:
                out_supertrend[i, last] = c[last]
                out_supertrend_dir[i, last] = 0.0

except Exception:
    # Fallback: keep existing pure-Python fallback but accept counts and full parameter
    def calc_indicators_inplace(
        closes, volumes, highs, lows, counts,
        out_rsi, out_mfi, out_adx, out_vol_z, out_atr,
        out_ema_7, out_ema_14, out_ema_21, out_ema_50, out_ema_200,
        out_macd, out_macd_signal, out_macd_hist,
        out_stoch_k, out_stoch_d,
        out_vwap,
        out_supertrend, out_supertrend_dir,
        full: bool = False
    ):
        n_symbols, n_cols = closes.shape
        for i in range(n_symbols):
            try:
                c = closes[i]
                v = volumes[i]
                h = highs[i]
                l = lows[i]
                valid = int(counts[i]) if counts is not None else n_cols
                if valid <= 0:
                    continue
                if not full:
                    last = n_cols - 1
                    if np.isnan(c[last]) or c[last] == 0.0:
                        continue
                    out_rsi[i, last] = 50.0
                    out_mfi[i, last] = 50.0
                    out_adx[i, last] = 0.0
                    out_vol_z[i, last] = 0.0
                    out_atr[i, last] = 0.0
                    out_ema_7[i, last] = c[last]
                    out_ema_14[i, last] = c[last]
                    out_ema_21[i, last] = c[last]
                    out_ema_50[i, last] = c[last]
                    out_ema_200[i, last] = c[last]
                    out_macd[i, last] = 0.0
                    out_macd_signal[i, last] = 0.0
                    out_macd_hist[i, last] = 0.0
                    out_stoch_k[i, last] = 0.0
                    out_stoch_d[i, last] = 0.0
                    out_vwap[i, last] = c[last]
                    out_supertrend[i, last] = c[last]
                    out_supertrend_dir[i, last] = 0.0
                else:
                    # simple full-history fallback: compute per-column naive values up to valid
                    for j in range(valid):
                        out_rsi[i, j] = 50.0
                        out_mfi[i, j] = 50.0
                        out_adx[i, j] = 0.0
                        out_vol_z[i, j] = 0.0
                        out_atr[i, j] = 0.0
                        out_ema_7[i, j] = c[j]
                        out_ema_14[i, j] = c[j]
                        out_ema_21[i, j] = c[j]
                        out_ema_50[i, j] = c[j]
                        out_ema_200[i, j] = c[j]
                        out_macd[i, j] = 0.0
                        out_macd_signal[i, j] = 0.0
                        out_macd_hist[i, j] = 0.0
                        out_stoch_k[i, j] = 0.0
                        out_stoch_d[i, j] = 0.0
                        out_vwap[i, j] = c[j]
                        out_supertrend[i, j] = c[j]
                        out_supertrend_dir[i, j] = 0.0
                    for j in range(valid, n_cols):
                        out_rsi[i, j] = 0.0
                        out_mfi[i, j] = 0.0
                        out_adx[i, j] = 0.0
                        out_vol_z[i, j] = 0.0
                        out_atr[i, j] = 0.0
                        out_ema_7[i, j] = 0.0
                        out_ema_14[i, j] = 0.0
                        out_ema_21[i, j] = 0.0
                        out_ema_50[i, j] = 0.0
                        out_ema_200[i, j] = 0.0
                        out_macd[i, j] = 0.0
                        out_macd_signal[i, j] = 0.0
                        out_macd_hist[i, j] = 0.0
                        out_stoch_k[i, j] = 0.0
                        out_stoch_d[i, j] = 0.0
                        out_vwap[i, j] = 0.0
                        out_supertrend[i, j] = 0.0
                        out_supertrend_dir[i, j] = 0.0
            except Exception:
                logger.debug("Indicator fallback error for symbol index %d", i, exc_info=True)
                continue

# Stablecoin set used across the collector
STABLECOINS = {
    'USDT', 'USDC', 'BUSD', 'TUSD', 'DAI', 'FDUSD',
    'USDD', 'USDP', 'FRAX', 'USDB', 'USDE'
}
class Ticker(TypedDict, total=False):
    symbol: str
    quoteVolume: str | float
def _tf_to_ms(tf: str) -> int:
    tf = tf.strip().lower()
    if tf.endswith('m'):
        return int(tf[:-1]) * 60_000
    if tf.endswith('h'):
        return int(tf[:-1]) * 3_600_000
    if tf.endswith('d'):
        return int(tf[:-1]) * 86_400_000
    if tf.endswith('w'):
        return int(tf[:-1]) * 7 * 86_400_000
    raise ValueError(f"Unsupported timeframe: {tf}")

class MarketMatrix:
    """
    Ring-buffer based, versioned market data container.

    Notes and safety:
    - Per-symbol locks are implemented with threading.Lock for low-overhead
      synchronous protection. This is safe as long as callers do not hold
      these locks across `await` points. If you need to await while holding a
      lock, convert to asyncio.Lock and update call sites accordingly.
    - snapshot_consistent returns copies (not views) and converts NaN/inf to 0
      to make downstream numeric code robust.
    """

    def __init__(self, symbols: List[str], timeframe: str, max_candles: int = 1000):
        self.symbols = list(symbols)
        self.symbol_map: Dict[str, int] = {s: i for i, s in enumerate(self.symbols)}
        self.n_symbols = len(self.symbols)
        self.last_calc_time = 0.0
        self.min_calc_interval = 1.0  # seconds, tune per timeframe
        self.timeframe = timeframe
        self.max_candles = int(max_candles)

        # --- Core OHLCV ring buffers ---
        shape = (self.n_symbols, self.max_candles)
        self.closes = np.full(shape, np.nan, dtype=np.float64)
        self.highs = np.full(shape, np.nan, dtype=np.float64)
        self.lows = np.full(shape, np.nan, dtype=np.float64)
        self.volumes = np.full(shape, np.nan, dtype=np.float64)
        self.timestamps = np.zeros(shape, dtype=np.int64)

        # --- Ring write index per symbol ---
        self.write_idx = np.zeros(self.n_symbols, dtype=np.int32)

        # Number of CLOSED candles loaded (per symbol)
        self.count = np.zeros(self.n_symbols, dtype=np.int32)

        # --- Indicator outputs ---
        self.rsi = np.full(shape, np.nan, dtype=np.float32)
        self.mfi = np.full(shape, np.nan, dtype=np.float32)
        self.adx = np.full(shape, np.nan, dtype=np.float32)
        self.atr = np.full(shape, np.nan, dtype=np.float32)
        self.vol_z = np.full(shape, np.nan, dtype=np.float32)

        self.ema_7 = np.full(shape, np.nan, dtype=np.float32)
        self.ema_14 = np.full(shape, np.nan, dtype=np.float32)
        self.ema_21 = np.full(shape, np.nan, dtype=np.float32)
        self.ema_50 = np.full(shape, np.nan, dtype=np.float32)
        self.ema_200 = np.full(shape, np.nan, dtype=np.float32)

        self.macd = np.full(shape, np.nan, dtype=np.float32)
        self.macd_signal = np.full(shape, np.nan, dtype=np.float32)
        self.macd_hist = np.full(shape, np.nan, dtype=np.float32)

        self.stoch_k = np.full(shape, np.nan, dtype=np.float32)
        self.stoch_d = np.full(shape, np.nan, dtype=np.float32)

        self.vwap = np.full(shape, np.nan, dtype=np.float32)

        self.supertrend = np.full(shape, np.nan, dtype=np.float32)
        self.supertrend_dir = np.full(shape, np.nan, dtype=np.float32)

        # --- Versioning ---
        self.data_version = 0
        self.indicator_version = 0
        self.last_calc_time = 0.0

        # --- Locks (async-friendly) ---
        # Use asyncio.Lock for per-row protection and versioning to avoid blocking the event loop.
        # All methods that acquire these locks are async (update_candle, fill_missing_candles,
        # snapshot_consistent, commit_indicators). This prevents accidental blocking of the loop.
        # If you must call these from synchronous code, use the provided sync wrappers.
        self._symbol_locks = [asyncio.Lock() for _ in range(self.n_symbols)]
        self._version_lock = asyncio.Lock()

        # Serialize indicator commits to reduce lost commits under contention.
        self._indicator_lock = asyncio.Lock()
    # Internal Ring Mechanics
    def _advance_pointer(self, idx: int):
        self.write_idx[idx] = (self.write_idx[idx] + 1) % self.max_candles
    def _current_slot(self, idx: int) -> int:
        return int(self.write_idx[idx])
    async def update_candle(
        self,
        symbol: str,
        close: float,
        high: float,
        low: float,
        volume: float,
        ts_ms: int,
        is_closed: bool
    ):
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return

        # Acquire per-symbol lock asynchronously
        async with self._symbol_locks[idx]:
            slot = self._current_slot(idx)
            # Always write the forming slot values (either updating forming candle or writing closed)
            self.closes[idx, slot] = close
            self.highs[idx, slot] = high
            self.lows[idx, slot] = low
            self.volumes[idx, slot] = volume
            self.timestamps[idx, slot] = ts_ms

            if is_closed:
                # Mark closed: increment count and advance pointer AFTER writing closed candle
                self.count[idx] = min(self.count[idx] + 1, self.max_candles)
                self._advance_pointer(idx)

                # Initialize next forming candle slot with last close
                next_slot = self._current_slot(idx)
                self.closes[idx, next_slot] = close
                self.highs[idx, next_slot] = close
                self.lows[idx, next_slot] = close
                self.volumes[idx, next_slot] = 0.0

                # compute next candle start timestamp using timeframe step
                try:
                    step = _tf_to_ms(self.timeframe)
                    self.timestamps[idx, next_slot] = int(ts_ms + step)
                except Exception:
                    # fallback: use same ts_ms if timeframe parsing fails
                    self.timestamps[idx, next_slot] = ts_ms

        # Bump data_version once under version lock to signal snapshotters
        async with self._version_lock:
            self.data_version += 1

    async def snapshot_consistent(self, max_retries: int = 2):
        """
        Row-wise consistent snapshot:
        - read data_version
        - copy each symbol row under its symbol lock
        - re-read data_version; if changed, retry
        Returns: (version, closes, highs, lows, volumes, counts) where arrays are copies
        and NaN/inf are converted to 0.0 for numeric safety. `counts` is an int array
        with per-symbol number of CLOSED candles (caller can use it to detect insufficient history).
        """
        for _ in range(max_retries + 1):
            async with self._version_lock:
                v1 = self.data_version

            # Preallocate output arrays with same shapes and dtypes
            closes = np.empty_like(self.closes)
            highs = np.empty_like(self.highs)
            lows = np.empty_like(self.lows)
            volumes = np.empty_like(self.volumes)
            counts = np.empty_like(self.count)

            # Copy each symbol row under its symbol lock to ensure per-row consistency
            for i in range(self.n_symbols):
                async with self._symbol_locks[i]:
                    closes[i, :] = self.closes[i, :].copy()
                    highs[i, :] = self.highs[i, :].copy()
                    lows[i, :] = self.lows[i, :].copy()
                    volumes[i, :] = self.volumes[i, :].copy()
                    counts[i] = int(self.count[i])

            async with self._version_lock:
                v2 = self.data_version

            if v1 == v2:
                # Convert NaNs/Infs to zeros on the copies to avoid in-place mutation of shared arrays
                return v1, self._nan_to_zero(closes), self._nan_to_zero(highs), self._nan_to_zero(lows), self._nan_to_zero(volumes), counts

        # Return last attempt even if versions differ (caller must handle stale snapshot)
        return v2, self._nan_to_zero(closes), self._nan_to_zero(highs), self._nan_to_zero(lows), self._nan_to_zero(volumes), counts
    async def commit_indicators(
        self,
        snapshot_version: int,
        rsi: np.ndarray,
        mfi: np.ndarray,
        adx: np.ndarray,
        atr: np.ndarray,
        vol_z: np.ndarray,
        ema_7: np.ndarray,
        ema_14: np.ndarray,
        ema_21: np.ndarray,
        ema_50: np.ndarray,
        ema_200: np.ndarray,
        macd: np.ndarray,
        macd_signal: np.ndarray,
        macd_hist: np.ndarray,
        stoch_k: np.ndarray,
        stoch_d: np.ndarray,
        vwap: np.ndarray,
        supertrend: np.ndarray,
        supertrend_dir: np.ndarray,
    ) -> bool:
        # Serialize commits to avoid interleaved partial writes from concurrent committers.
        async with self._indicator_lock:
            async with self._version_lock:
                current_version = self.data_version
                # allow commit if snapshot_version is not too stale
                if snapshot_version < current_version - 1:
                    logger.debug("Indicator commit rejected: snapshot_version=%s current_data_version=%s", snapshot_version, current_version)
                    return False

            try:
                # Bulk assign arrays (these are numpy arrays; assignment is fast)
                self.rsi[:] = rsi
                self.mfi[:] = mfi
                self.adx[:] = adx
                self.atr[:] = atr
                self.vol_z[:] = vol_z

                self.ema_7[:] = ema_7
                self.ema_14[:] = ema_14
                self.ema_21[:] = ema_21
                self.ema_50[:] = ema_50
                self.ema_200[:] = ema_200

                self.macd[:] = macd
                self.macd_signal[:] = macd_signal
                self.macd_hist[:] = macd_hist

                self.stoch_k[:] = stoch_k
                self.stoch_d[:] = stoch_d

                self.vwap[:] = vwap

                self.supertrend[:] = supertrend
                self.supertrend_dir[:] = supertrend_dir

                async with self._version_lock:
                    self.indicator_version = snapshot_version
                    self.last_calc_time = time.time()
                return True
            except Exception:
                logger.exception("Failed during commit_indicators bulk assign")
                return False
    async def commit_indicators_slice(
        self,
        snapshot_version: int,
        start_idx: int,
        end_idx: int,
        rsi: np.ndarray,
        mfi: np.ndarray,
        adx: np.ndarray,
        atr: np.ndarray,
        vol_z: np.ndarray,
        ema_7: np.ndarray,
        ema_14: np.ndarray,
        ema_21: np.ndarray,
        ema_50: np.ndarray,
        ema_200: np.ndarray,
        macd: np.ndarray,
        macd_signal: np.ndarray,
        macd_hist: np.ndarray,
        stoch_k: np.ndarray,
        stoch_d: np.ndarray,
        vwap: np.ndarray,
        supertrend: np.ndarray,
        supertrend_dir: np.ndarray,
    ) -> bool:
        """
        Commit a slice of symbol rows [start_idx:end_idx) into the matrix indicator arrays.
        Arrays passed must have shape (end_idx - start_idx, n_cols).
        """
        async with self._indicator_lock:
            async with self._version_lock:
                current_version = self.data_version
                if snapshot_version < current_version - 1:
                    logger.debug("Indicator slice commit rejected: snapshot_version=%s current_data_version=%s", snapshot_version, current_version)
                    return False

            try:
                # Write slices under indicator lock
                self.rsi[start_idx:end_idx, :] = rsi
                self.mfi[start_idx:end_idx, :] = mfi
                self.adx[start_idx:end_idx, :] = adx
                self.atr[start_idx:end_idx, :] = atr
                self.vol_z[start_idx:end_idx, :] = vol_z

                self.ema_7[start_idx:end_idx, :] = ema_7
                self.ema_14[start_idx:end_idx, :] = ema_14
                self.ema_21[start_idx:end_idx, :] = ema_21
                self.ema_50[start_idx:end_idx, :] = ema_50
                self.ema_200[start_idx:end_idx, :] = ema_200

                self.macd[start_idx:end_idx, :] = macd
                self.macd_signal[start_idx:end_idx, :] = macd_signal
                self.macd_hist[start_idx:end_idx, :] = macd_hist

                self.stoch_k[start_idx:end_idx, :] = stoch_k
                self.stoch_d[start_idx:end_idx, :] = stoch_d

                self.vwap[start_idx:end_idx, :] = vwap

                self.supertrend[start_idx:end_idx, :] = supertrend
                self.supertrend_dir[start_idx:end_idx, :] = supertrend_dir

                async with self._version_lock:
                    # For slices we set indicator_version to snapshot_version to indicate progress.
                    self.indicator_version = snapshot_version
                    self.last_calc_time = time.time()
                return True
            except Exception:
                logger.exception("Failed during commit_indicators_slice bulk assign")
                return False

# In __init__ keep: self._symbol_locks = [asyncio.Lock() for _ in range(self.n_symbols)]
# Remove any threading.Lock usage for per-symbol locks.

    # Replace fill_missing_candles with async version
    async def fill_missing_candles(self, symbol: str, missing_count: int, last_ts_ms: Optional[int] = None):
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return

        step = _tf_to_ms(self.timeframe)
        lock = self._symbol_locks[idx]
        async with lock:
            last_closed_slot = (self._current_slot(idx) - 1) % self.max_candles
            last_close = float(self.closes[idx, last_closed_slot])
            if np.isnan(last_close):
                return

            if last_ts_ms is None:
                last_ts_ms = int(self.timestamps[idx, last_closed_slot])

            if last_ts_ms <= 0:
                return

            ts = last_ts_ms
            for _ in range(int(missing_count)):
                ts += step
                self._advance_pointer(idx)
                slot = self._current_slot(idx)

                self.closes[idx, slot] = last_close
                self.highs[idx, slot] = last_close
                self.lows[idx, slot] = last_close
                self.volumes[idx, slot] = 0.0
                self.timestamps[idx, slot] = ts

                self.count[idx] = min(self.count[idx] + 1, self.max_candles)

        async with self._version_lock:
            self.data_version += 1

    async def bulk_load_candles(self, symbol: str, rows: List[Tuple[int, float, float, float, float]]):
        """
        Bulk-load a list of closed candles for `symbol`.
        `rows` is a list of tuples: (ts_ms, high, low, close, volume).
        This writes directly into the ring buffer under the per-row lock and
        increments data_version once at the end to avoid repeated version bumps.
        """
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return

        async with self._symbol_locks[idx]:
            logger.debug("bulk_load_candles: writing %d rows for %s", len(rows), symbol)
            for ts_ms, high, low, close, volume in rows:
                slot = self._current_slot(idx)
                self.closes[idx, slot] = close
                self.highs[idx, slot] = high
                self.lows[idx, slot] = low
                self.volumes[idx, slot] = volume
                self.timestamps[idx, slot] = ts_ms
                # mark closed and advance
                self.count[idx] = min(self.count[idx] + 1, self.max_candles)
                self._advance_pointer(idx)

        async with self._version_lock:
            self.data_version += 1

    async def get_latest(self, symbol: str, closed_only: bool = True) -> Optional[dict]:
        """
        Async-only safe read of the latest candle for `symbol`.

        This method requires callers to `await` it. It acquires the per-symbol
        asyncio.Lock to ensure a consistent read of the ring buffer row.
        """
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return None

        lock = self._symbol_locks[idx]

        async with lock:
            if self.count[idx] <= 0 and closed_only:
                return None

            cur = self._current_slot(idx)
            slot = (cur - 1) % self.max_candles if closed_only else cur

            close = self.closes[idx, slot]
            if np.isnan(close):
                return None

            # Helper to safely extract and coerce numeric values
            def _safe_val(arr: np.ndarray, i: int, s: int, default: float = 0.0) -> float:
                try:
                    v = arr[i, s]
                    if np.isnan(v):
                        return default
                    return float(v)
                except Exception:
                    return default

            return {
                "close": float(close),
                "high": _safe_val(self.highs, idx, slot, default=0.0),
                "low": _safe_val(self.lows, idx, slot, default=0.0),
                "volume": _safe_val(self.volumes, idx, slot, default=0.0),
                "timestamp": int(self.timestamps[idx, slot]),
                "rsi": (None if np.isnan(self.rsi[idx, slot]) else float(self.rsi[idx, slot])),
                "atr": (None if np.isnan(self.atr[idx, slot]) else float(self.atr[idx, slot])),
            }

    def get_latest_sync(self, symbol: str, closed_only: bool = True, timeout: float = 5.0) -> Optional[dict]:
        """
        Thread-safe synchronous wrapper for get_latest.

        - If called from a thread where the event loop is running, this uses
          asyncio.run_coroutine_threadsafe to schedule the coroutine on that loop
          and waits for the result (with timeout).
        - If no running loop is found, it uses asyncio.run to execute the coroutine.
        """
        coro = self.get_latest(symbol, closed_only=closed_only)

        # If there's no running loop in this thread, run the coroutine directly.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is None:
            try:
                return asyncio.run(coro)
            except Exception:
                logger.exception("get_latest_sync: asyncio.run failed")
                return None
        else:
            try:
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                return fut.result(timeout=timeout)
            except Exception:
                logger.exception("get_latest_sync: run_coroutine_threadsafe failed or timed out")
                return None


    def _nan_to_zero(self, arr: np.ndarray) -> np.ndarray:
        """
        Return a copy of arr with NaN/inf replaced by 0.0.
        Copy is explicit to avoid reading partially-written memory.
        """
        out = arr.copy()
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

class SymbolState:
    def __init__(self, symbol: str, max_liqs: int = 50_000):
        self.symbol: str = symbol
        self.lock: asyncio.Lock = asyncio.Lock()
        self.depth_metrics: Dict[str, float] = {}
        self.rest_metrics: Dict[str, float] = {}
        self.liquidations: deque[dict] = deque(maxlen=max_liqs)

    async def snapshot(self) -> dict:
        async with self.lock:
            return {
                "depth": dict(self.depth_metrics),
                "rest": dict(self.rest_metrics),
                "liqs": list(self.liquidations),
            }

    async def append_liq(self, item: dict) -> None:
        """Append a liquidation entry under the state lock."""
        async with self.lock:
            self.liquidations.append(item)

# End of patched part 1
# unified_collector.py part 2
@dataclass
class EndpointState:
    last_call: float = 0.0
    min_interval: float = 0.25
    ban_until: float = 0.0
    ban_duration: float = 0.0
    ban_logged_at: float = 0.0
    ban_last_warn: float = 0.0


RestartFactory = Coroutine[Any, Any, Any]


class BinanceUnifiedCollector:
    """
    Unified class that merges:
    - BinanceMarketDataCollector (matrices, warmup, kline WS, batch updates, indicator loop, persistence)
    - BinanceFuturesSuite (depth polling, futures REST metrics, liquidation WS + rolling aggregates, symbol management)
    """

    def __init__(
        self,
        max_candles: int = 1000,
        output_dir: str = "./data_storage",
        top_limit: int = 165,
        depth_interval: int = 1000,
    ):
    
        # REST endpoints
        self.spot_rest = "https://api.binance.com/api/v3"
        self.futures_rest = "https://fapi.binance.com/fapi/v1"
        self.logger = logger = logging.getLogger(__name__)
        # WS endpoints
        self.futures_ws_base = "wss://fstream.binance.com"

        # Token bucket: conservative capacity and refill rate
        self._weight_bucket = self._TokenBucket(capacity=900.0, refill_per_sec=900.0 / 60.0)

        # reduce connector concurrency to avoid TCP bursts
        self._connector_limit = 10

        # session ownership metadata
        self._session_owner = "unified_collector"
        self._session: Optional[aiohttp.ClientSession] = None

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.active_timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
        self.max_candles = int(max_candles)
        self.top_limit = int(top_limit)

        # health monitoring
        self._health_net_history = deque(maxlen=360)
        self._health_prev_net = None

        # Symbol universe
        self.top_coins: List[str] = []
        self.top_coins_lock = asyncio.Lock()

        # Futures symbol cache
        self._futures_symbols_cache: Optional[set[str]] = None
        self._futures_symbols_cache_ts: float = 0.0

        # Matrices
        self.matrices: Dict[str, Any] = {}

        # Futures per-symbol states
        self._symbol_states: Dict[str, Any] = {}
        self._symbols_snapshot: Tuple[str, ...] = tuple()
        self._symbols_lock = asyncio.Lock()

        # Pause ingestion when symbol list changes
        self._ingestion_paused = asyncio.Event()
        self._ingestion_paused.set()

        # Futures suite parameters
        self._depth_interval = int(depth_interval)
        self._depth_endpoint = f"{self.futures_rest}/depth"
        self._premium_endpoint = f"{self.futures_rest}/premiumIndex"
        self._oi_endpoint = f"{self.futures_rest}/openInterest"
        self._ratio_endpoint = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"

        # Liquidations WS (all-market)
        self._liq_allmarket_ws = f"{self.futures_ws_base}/ws/!forceOrder@arr"

        # Async control
        self.running = True
        # use a set for tasks to avoid duplicates and O(1) removals
        self.tasks: Set[asyncio.Task] = set()

        # Batch update queue
        self.update_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._connector = aiohttp.TCPConnector(limit=self._connector_limit, ttl_dns_cache=300)
        # Counters
        self.error_counters = {
            "fetch_top_coins_failures": 0,
            "ws_parse_errors": 0,
            "ws_reconnections": {},
            "liq_ws_reconnects": 0,
            "liq_ws_parse_errors": 0,
            "depth_429": 0,
        }

        # Liq rolling aggregation
        # Use asyncio.Lock for async code paths
        self._liq_lock = asyncio.Lock()
        self.history = deque()  # (ts_s, symbol, side, usd, price)
        self.symbol_stats = collections.defaultdict(lambda: {
            "1h_long": 0.0, "1h_short": 0.0,
            "24h_long": 0.0, "24h_short": 0.0,
            "last_price": 0.0,
        })

        # Unified REST client state
        self.spot_base = self.spot_rest.rstrip("/")
        self.futures_base = self.futures_rest.rstrip("/")
        self._lock = asyncio.Lock()
        self._state: Dict[str, EndpointState] = {}
        self._default_min_interval = 0.25

        self._rest_lock = asyncio.Lock()
        self._rest_last_call = 0.0
        self._rest_min_interval = 0.25

        self._health_monitor_name = "health-monitor"

        # restart registry for background tasks (name -> factory)
        self._restart_map: Dict[str, RestartFactory] = {}
    # ---------------------------
    # Token bucket implementation
    # ---------------------------
    class _TokenBucket:
        """
        Simple async token bucket with wait_for(amount, timeout) API.

        - capacity: max tokens
        - refill_per_sec: tokens added per second
        """

        def __init__(self, capacity: float, refill_per_sec: float) -> None:
            self.capacity = float(capacity)
            self.refill_per_sec = float(refill_per_sec)
            self._tokens = float(capacity)
            # use monotonic clock to avoid system time jumps
            self._last = time.monotonic()
            self.tasks: Set[asyncio.Task] = set()
            self.running: bool = False
            self._session: Optional[Any] = None
            self._health_monitor_name: str = "health-monitor"

            # Logger for the instance
            self.logger = logging.getLogger(f"{__name__}.TokenBucket")

        async def wait_for(self, amount: float, timeout: float | None = None) -> None:
            """
            Wait until `amount` tokens are available, then consume them.
            Raises asyncio.TimeoutError if timeout is provided and exceeded.
            """
            if not hasattr(self, "_lock") or self._lock is None:
                self._lock = asyncio.Lock()

            amount = float(amount)
            start = time.monotonic()
            while True:
                now = time.monotonic()
                async with self._lock:
                    elapsed = now - self._last
                    if elapsed > 0:
                        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_per_sec)
                        self._last = now
                    if self._tokens >= amount:
                        self._tokens -= amount
                        # sanity check
                        if self._tokens < -1e-6:
                            self.logger.warning("TokenBucket negative tokens: %s", self._tokens)
                            self._tokens = max(0.0, self._tokens)
                        return

                # compute remaining time if timeout provided and sleep only up to remaining
                if timeout is not None:
                    elapsed_total = time.monotonic() - start
                    remaining = timeout - elapsed_total
                    if remaining <= 0:
                        raise asyncio.TimeoutError("Token bucket wait timed out")
                    await asyncio.sleep(min(0.05, remaining))
                else:
                    await asyncio.sleep(0.05)

        def _create_task(self, coro: Coroutine[Any, Any, Any], name: Optional[str] = None) -> asyncio.Task:
            """
            Create an asyncio.Task, register it in self.tasks, and set a name if supported.
            Use this helper everywhere you spawn background tasks.
            """
            t = asyncio.create_task(coro)
            try:
                if name and hasattr(t, "set_name"):
                    t.set_name(name)
            except Exception:
                pass
            self.tasks.add(t)
            def _on_done(task):
                try:
                    self.tasks.discard(task)
                except Exception:
                    pass
            t.add_done_callback(_on_done)
            return t

        async def start(self, start_default_loops: bool = False, restart_map: Optional[Dict[str, RestartFactory]] = None) -> None:
            """
            Start background infrastructure (health monitor). Call this after instantiation.
            """
            self.running = True
            restart_map = restart_map or {}

            root = logging.getLogger()
            if not root.handlers:
                logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

            self._create_task(self._task_health_monitor(restart_map=restart_map), name=self._health_monitor_name)
            self.logger.info("TokenBucket: health monitor started")

            if start_default_loops:
                if hasattr(self, "fetch_top_coins"):
                    self._create_task(self.fetch_top_coins(), name="fetch_top_coins")

        async def _task_health_monitor(self, interval: float = 10.0, restart_map: Optional[Dict[str, RestartFactory]] = None) -> None:
            restart_map = restart_map or {}
            while self.running:
                await asyncio.sleep(interval)
                for t in list(self.tasks):
                    try:
                        name = t.get_name() if hasattr(t, "get_name") else repr(t)
                    except Exception:
                        name = repr(t)

                    if t.done():
                        exc = None
                        try:
                            exc = t.exception()
                        except asyncio.CancelledError:
                            exc = None
                        if exc:
                            self.logger.error("Background task %s finished with exception: %s", name, exc)
                        else:
                            self.logger.info("Background task %s finished (no exception)", name)

                        try:
                            self.tasks.remove(t)
                        except ValueError:
                            pass

                        factory = restart_map.get(name)
                        if factory:
                            try:
                                new_t = self._create_task(factory(), name=name)
                                self.logger.info("Restarted task %s -> %s", name, new_t)
                            except Exception as e:
                                self.logger.error("Failed to restart task %s: %s", name, e)

                if self.tasks:
                    alive = sum(1 for t in self.tasks if not t.done())
                    self.logger.debug("TokenBucket task health: %d alive, %d total", alive, len(self.tasks))
        async def close(self) -> None:
            self.running = False

            # Cancel and await background tasks
            tasks_snapshot = list(self.tasks)
            for t in tasks_snapshot:
                try:
                    t.cancel()
                except Exception:
                    pass

            if tasks_snapshot:
                results = await asyncio.gather(*tasks_snapshot, return_exceptions=True)
                for t, r in zip(tasks_snapshot, results):
                    if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                        try:
                            name = t.get_name() if hasattr(t, "get_name") else repr(t)
                        except Exception:
                            name = repr(t)
                        self.logger.error("Task %s raised during close: %s", name, r)

            # Clear task set
            self.tasks.clear()

            # Close any internal session
            session = getattr(self, "_session", None)
            if session is not None:
                try:
                    closed_attr = getattr(session, "closed", None)
                    if closed_attr is False:
                        await session.close()
                except Exception as e:
                    self.logger.warning("Error closing tokenbucket session: %s", e)
            self._session = None

    # ---------------------------
    # Session management
    # ---------------------------
    async def _ensure_session(self) -> aiohttp.ClientSession:
        # Reuse existing session if valid
        if self._session and not getattr(self._session, "closed", True):
            return self._session

        async with self._lock:
            if self._session and not getattr(self._session, "closed", True):
                return self._session
            timeout = aiohttp.ClientTimeout(total=20)
            self._session = aiohttp.ClientSession(timeout=timeout, connector=self._connector)
            setattr(self._session, "_owner", self._session_owner)
            logger.debug("Created new aiohttp session owner=%s connector_limit=%s", self._session_owner, self._connector_limit)
            return self._session


    # ---------------------------
    # Endpoint state helpers
    # ---------------------------
    def _get_state(self, key: str) -> EndpointState:
        st = self._state.get(key)
        if st is None:
            st = EndpointState(last_call=0.0, min_interval=self._default_min_interval)
            self._state[key] = st
        return st

    def _mark_call(self, key: str, min_interval: Optional[float]):
        st = self._get_state(key)
        st.last_call = time.time()
        if min_interval is not None:
            st.min_interval = max(st.min_interval, float(min_interval))

    def _format_duration(self, seconds: float) -> str:
        s = int(max(0, round(seconds)))
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def _mark_ban(self, key: str, seconds: float):
        st = self._get_state(key)
        now = time.time()
        st.ban_until = now + float(seconds)
        st.ban_duration = float(seconds)
        st.ban_logged_at = now
        unban_ts = datetime.datetime.fromtimestamp(st.ban_until).isoformat(sep=" ", timespec="seconds")
        logger.error(f"🚫 Endpoint {key} banned for {seconds:.0f}s ({self._format_duration(seconds)}), unban at {unban_ts}")

    def _clear_ban(self, key: str):
        st = self._get_state(key)
        if st.ban_until > 0:
            logger.info(f"✅ Endpoint {key} unbanned (was banned for {self._format_duration(st.ban_duration)})")
        st.ban_until = 0.0
        st.ban_duration = 0.0
        st.ban_logged_at = 0.0
        st.ban_last_warn = 0.0

    async def _throttle(self, key: str, min_interval: Optional[float]):
        now = time.time()
        st = self._get_state(key)

        ban_until = st.ban_until
        if ban_until > now:
            sleep_for = ban_until - now
            last_logged = st.ban_last_warn
            if now - last_logged > 30:
                st.ban_last_warn = now
                logger.warning(f"⏳ Endpoint {key} still banned for {sleep_for:.1f}s ({self._format_duration(sleep_for)})")
            await asyncio.sleep(sleep_for)
            return

        interval = float(min_interval) if min_interval is not None else st.min_interval
        delta = now - st.last_call
        if delta < interval:
            await asyncio.sleep(interval - delta)

    # ---------------------------
    # REST helper with robust throttling and token bucket integration
    # ---------------------------
    async def get_json(
        self,
        market: str,
        path: str,
        *,
        params: Optional[dict] = None,
        endpoint_key: Optional[str] = None,
        min_interval: Optional[float] = None,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_factor: float = 2.0,
        estimated_weight: float = 1.0,
    ) -> dict:
        base = self.spot_base if market == "spot" else self.futures_base
        url = f"{base}{path}"
        key = endpoint_key or f"{market}:{path}"

        attempt = 0
        while True:
            attempt += 1
            await self._throttle(key, min_interval)

            # ensure we don't exceed token bucket (estimated request weight)
            try:
                # set a reasonable timeout for token acquisition to avoid indefinite blocking
                await self._weight_bucket.wait_for(float(estimated_weight), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("Token bucket timeout for %s %s (weight=%s) attempt=%d", market, path, estimated_weight, attempt)
                if attempt >= max_retries:
                    raise RuntimeError(f"Token bucket timeout for {url} after {attempt} attempts")
                await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))
                continue
            except Exception as e:
                logger.exception("Token bucket error: %s", e)
                await asyncio.sleep(0.5)

            session = await self._ensure_session()
            try:
                async with session.get(url, params=params) as resp:
                    status = resp.status

                    # log weight headers for monitoring (normalize header names)
                    headers_lower = {k.lower(): v for k, v in resp.headers.items()}
                    used = headers_lower.get("x-mbx-used-weight")
                    # some servers expose X-MBX-USED-WEIGHT-1M or similar; find any key that starts with x-mbx-used-weight
                    used_1m = None
                    for k, v in headers_lower.items():
                        if k.startswith("x-mbx-used-weight"):
                            used_1m = v
                            break
                    logger.debug("Used weight header: %s; Used weight 1M: %s", used, used_1m)
                    # 418: IP banned
                    if status == 418:
                        retry_after = None
                        try:
                            hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                            if hdr:
                                try:
                                    retry_after = int(hdr)
                                except Exception:
                                    try:
                                        dt = parsedate_to_datetime(hdr)
                                        retry_after = max(0, int((dt.timestamp() - time.time())))
                                    except Exception:
                                        retry_after = None
                        except Exception:
                            retry_after = None

                        ban_seconds = float(retry_after) if retry_after is not None else 600.0
                        st = self._get_state(key)
                        now = time.time()
                        if st.ban_until <= now:
                            self._mark_ban(key, ban_seconds)
                        else:
                            if ban_seconds > st.ban_duration:
                                self._mark_ban(key, ban_seconds)

                        logger.error("🚫 418 from %s (attempt %d); banning for %0.0fs", url, attempt, ban_seconds)
                        if attempt >= max_retries:
                            raise RuntimeError(f"418 from {url} after {attempt} attempts")
                        await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))
                        continue

                    # 429: rate limit
                    if status == 429:
                        logger.warning("⚠️ 429 from %s (attempt %d)", url, attempt)
                        st = self._get_state(key)
                        # increase min_interval conservatively
                        st.min_interval *= 1.5
                        retry_after = None
                        try:
                            hdr = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
                            if hdr:
                                try:
                                    retry_after = int(hdr)
                                except Exception:
                                    try:
                                        dt = parsedate_to_datetime(hdr)
                                        retry_after = max(0, int((dt.timestamp() - time.time())))
                                    except Exception:
                                        retry_after = None
                        except Exception:
                            retry_after = None
                        if retry_after:
                            await asyncio.sleep(retry_after)
                        else:
                            if attempt >= max_retries:
                                raise RuntimeError(f"429 from {url} after {attempt} attempts")
                            await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))
                        continue

                    # other non-2xx
                    if status < 200 or status >= 300:
                        text = await resp.text()
                        logger.error("REST error %s from %s: %s", status, url, text[:200])
                        if attempt >= max_retries or (400 <= status < 500):
                            resp.raise_for_status()
                        await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))
                        continue

                    # success
                    self._mark_call(key, min_interval)
                    self._clear_ban(key)
                    try:
                        return await resp.json()
                    except Exception as e:
                        logger.exception("Failed to parse JSON from %s: %s", url, e)
                        if attempt >= max_retries:
                            raise
                        await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))
                        continue

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("REST call failed to %s (attempt %d): %s", url, attempt, e)
                if attempt >= max_retries:
                    raise
                await asyncio.sleep(backoff_base * (backoff_factor ** (attempt - 1)))

    # ---------------------------
    # Public accessors
    # ---------------------------
    def get_matrix(self, timeframe: str) -> Optional[Any]:
        return self.matrices.get(timeframe)

    async def get_symbols_snapshot(self) -> List[str]:
        async with self.top_coins_lock:
            return list(self.top_coins)

    async def get_liq_stats(self, symbol: str) -> dict:
        """
        Returns futures REST metrics for a symbol.
        """
        try:
            state = self._symbol_states.get(symbol)
            if not state:
                return {"rest": {}}

            snap = await state.snapshot()
            rest = snap.get("rest", {})

            return {"rest": rest}

        except Exception:
            return {"rest": {}}

    async def get_7day_liqs(self, symbol: str) -> List[dict]:
        if not await self._is_tracked(symbol):
            return []
        cutoff = int(time.time() * 1000) - 7 * 86400000
        state = self._symbol_states[symbol]
        async with state.lock:
            return [x for x in state.liquidations if x.get("ts", 0) >= cutoff]

    async def get_depth_liquidity_range(
        self,
        symbol: str,
        target_usdt: float | None = None,
        mid_price: float | None = None,
    ) -> dict:
        """
        Normalized depth API.

        target_usdt / mid_price are accepted for compatibility with image_generator,
        but this implementation currently returns aggregate depth only.

        Always returns a dict with bid_usd, ask_usd, imbalance, ts, book_ts_ms.
        Never returns None.
        """

        # --- Check if symbol is tracked ---
        try:
            tracked = (
                await self._is_tracked(symbol)
                if asyncio.iscoroutinefunction(self._is_tracked)
                else self._is_tracked(symbol)
            )
        except Exception:
            tracked = symbol in self._symbol_states

        if not tracked:
            return {
                "symbol": symbol,
                "bid_usd": 0.0,
                "ask_usd": 0.0,
                "imbalance": 0.0,
                "ts": 0,
                "down_price": None,
                "up_price": None,
                "down_pct": None,
                "up_pct": None,
                "mid": None,
                "book_ts_ms": 0,
                "bins": None,
            }

        # --- Get state ---
        state = self._symbol_states.get(symbol)
        if not state:
            return {
                "symbol": symbol,
                "bid_usd": 0.0,
                "ask_usd": 0.0,
                "imbalance": 0.0,
                "ts": 0,
                "down_price": None,
                "up_price": None,
                "down_pct": None,
                "up_pct": None,
                "mid": None,
                "book_ts_ms": 0,
                "bins": None,
            }

        # --- Read depth safely ---
        async with state.lock:
            depth = dict(state.depth_metrics)

        bid = float(depth.get("bid_usd", 0.0) or 0.0)
        ask = float(depth.get("ask_usd", 0.0) or 0.0)
        ts = int(depth.get("ts", 0) or 0)

        # --- Compute imbalance if missing ---
        total = bid + ask
        if total > 0:
            imbalance = (bid - ask) / total
        else:
            imbalance = 0.0

        return {
            "symbol": symbol,
            "bid_usd": bid,
            "ask_usd": ask,
            "imbalance": float(depth.get("imbalance", imbalance)),
            "ts": ts,
            "down_price": None,
            "up_price": None,
            "down_pct": None,
            "up_pct": None,
            "mid": None,
            "book_ts_ms": ts,
            "bins": None,
        }
    # ---------------------------
    # Helpers / placeholders for missing pieces in this snippet
    # ---------------------------
    # Example of registering a restartable background task
    def register_restartable(self, name: str, factory: RestartFactory) -> None:
        self._restart_map[name] = factory
    def _create_task(self, coro: Coroutine[Any, Any, Any], name: Optional[str] = None) -> asyncio.Task:
        """
        Create an asyncio.Task, register it in self.tasks, and attach a done callback
        that removes the task and logs any non-cancel exceptions.
        """
        # Create task with name if supported (Python 3.8+ supports name kwarg)
        try:
            t = asyncio.create_task(coro, name=name) if name is not None else asyncio.create_task(coro)
        except TypeError:
            # Fallback for Python versions that don't accept name= in create_task
            t = asyncio.create_task(coro)
            if name:
                try:
                    if hasattr(t, "set_name"):
                        t.set_name(name)
                except Exception:
                    pass

        # Ensure tasks container exists
        if not hasattr(self, "tasks") or self.tasks is None:
            try:
                self.tasks = set()
            except Exception:
                # best-effort: if we can't create the set, still return the task
                pass

        # Register task
        try:
            self.tasks.add(t)
        except Exception:
            # If registration fails, still keep going but log
            try:
                self.logger.debug("Failed to add task to self.tasks", exc_info=True)
            except Exception:
                pass

        def _on_done(task: asyncio.Task):
            # Remove from registry
            try:
                self.tasks.discard(task)
            except Exception:
                pass

            # Log exceptions (but ignore CancelledError)
            try:
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    try:
                        task_name = name or (task.get_name() if hasattr(task, "get_name") else repr(task))
                    except Exception:
                        task_name = repr(task)
                    try:
                        self.logger.error("Background task %s raised: %s", task_name, exc, exc_info=True)
                    except Exception:
                        # fallback to print if logger is not available
                        print(f"Background task {task_name} raised: {exc}")
            except asyncio.CancelledError:
                # expected during shutdown; ignore
                pass
            except Exception:
                # If task.exception() itself raised, log defensively
                try:
                    self.logger.exception("Error retrieving task exception")
                except Exception:
                    pass

        t.add_done_callback(_on_done)
        return t


    # ---------------------------
    # Symbol management
    # ---------------------------
    async def _get_symbols_snapshot_tuple(self) -> Tuple[str, ...]:
        async with self._symbols_lock:
            return self._symbols_snapshot
    async def _is_tracked(self, symbol: str) -> bool:
        # Keep this async for compatibility with callers
        async with self._symbols_lock:
            return symbol in self._symbol_states
    async def update_symbols(self, symbols: List[str]):
        """
        Update tracked symbols, enforcing futures-only.
        This method pauses ingestion while the symbol set is updated to avoid races.
        """
        symbols = await self._filter_futures_only(symbols)

        new_symbols = set(symbols)
        # Pause ingestion while we mutate symbol state
        self._ingestion_paused.clear()

        async with self._symbols_lock:
            old_symbols = set(self._symbol_states.keys())
            to_add = new_symbols - old_symbols
            to_remove = old_symbols - new_symbols

            for sym in to_remove:
                # best-effort cleanup
                try:
                    self._symbol_states.pop(sym, None)
                except Exception:
                    logger.exception("Failed to remove symbol state for %s", sym)

            for sym in to_add:
                try:
                    self._symbol_states[sym] = SymbolState(sym)
                except Exception:
                    logger.exception("Failed to create SymbolState for %s", sym)

            self._symbols_snapshot = tuple(sorted(self._symbol_states.keys()))

        # Resume ingestion
        self._ingestion_paused.set()
    # ---------------------------
    # Futures symbol filter
    # ---------------------------
    async def _get_futures_trading_symbols(self, ttl_sec: int = 3600) -> set[str]:
        """
        Fetch and cache futures exchangeInfo symbols (PERPETUAL USDT).
        Honors local ban state to avoid calling exchangeInfo while banned.
        """
        now = time.time()

        # Return cached if still valid
        if self._futures_symbols_cache and (now - self._futures_symbols_cache_ts) < ttl_sec:
            return self._futures_symbols_cache

        # avoid calling exchangeInfo while endpoint is banned
        st = self._get_state("futures:exchangeInfo")
        # st is EndpointState dataclass in patched core
        if getattr(st, "ban_until", 0) > now:
            logger.warning("Using cached futures symbols while exchangeInfo is banned")
            return self._futures_symbols_cache or set()

        try:
            data = await self.get_json(
                market="futures",
                path="/exchangeInfo",
                endpoint_key="futures:exchangeInfo",
                min_interval=1.0,
                max_retries=3,
                estimated_weight=10.0,
            )
        except Exception as e:
            logger.warning("Failed to fetch futures symbols: %s", e)
            return self._futures_symbols_cache or set()

        syms = set()
        for s in data.get("symbols", []):
            try:
                if s.get("status") != "TRADING":
                    continue
                if (s.get("quoteAsset") or "").upper() != "USDT":
                    continue
                if (s.get("contractType") or "").upper() != "PERPETUAL":
                    continue
                sym = (s.get("symbol") or "").upper()
                if sym:
                    syms.add(sym)
            except Exception:
                continue

        self._futures_symbols_cache = syms
        self._futures_symbols_cache_ts = now
        return syms


    # ---------------------------
    # Top coins discovery
    # ---------------------------
    async def _throttle_rest(self):
        """
        Simple per-process throttle for ad-hoc REST calls that don't use get_json.
        Prefer get_json for adaptive throttling; this helper is conservative.
        """
        async with self._rest_lock:
            now = time.time()
            delta = now - self._rest_last_call
            if delta < self._rest_min_interval:
                await asyncio.sleep(self._rest_min_interval - delta)
            self._rest_last_call = time.time()

    async def fetch_top_coins(self, limit: Optional[int] = None) -> List[str]:
        """
        Fetch top spot USDT pairs by 24h quoteVolume, then filter to futures-perp symbols.
        """

        if limit is None:
            limit = self.top_limit

        try:
            # JSON response is expected to be list[dict]
            raw_data: Any = await self.get_json(
                market="spot",
                path="/ticker/24hr",
                endpoint_key="spot:/ticker/24hr",
                min_interval=0.5,
                max_retries=3,
                estimated_weight=5.0,
            )

            # Normalize to list[dict]
            if isinstance(raw_data, dict):
                data: List[Dict[str, Any]] = raw_data.get("data", raw_data)  # type: ignore
            else:
                data = raw_data  # type: ignore

            # Ensure type safety
            if not isinstance(data, list):
                data = []

            # 2) build sorted list of spot USDT pairs by quoteVolume
            usdt_pairs: List[tuple[str, float]] = []

            for item in data:
                if not isinstance(item, dict):
                    continue

                ticker: Ticker = item  # type: ignore

                symbol = (ticker.get("symbol") or "").upper()
                if not symbol.endswith("USDT"):
                    continue

                base = symbol[:-4]
                if base in STABLECOINS:
                    continue

                qv_raw = ticker.get("quoteVolume", 0.0)
                try:
                    qv = float(qv_raw)
                except (TypeError, ValueError):
                    qv = 0.0

                usdt_pairs.append((symbol, qv))

            usdt_pairs.sort(key=lambda x: x[1], reverse=True)

            # 3) futures-perp symbols
            futures_syms = await self._get_futures_trading_symbols()

            # 4) pick top futures symbols
            selected: List[str] = []
            for sym, _ in usdt_pairs:
                if sym in futures_syms:
                    selected.append(sym)
                    if len(selected) >= limit:
                        break

            # 5) fill missing
            if len(selected) < limit:
                missing = limit - len(selected)
                extra = sorted(list(futures_syms - set(selected)))
                selected.extend(extra[:missing])

            # 6) commit to shared state
            async with self.top_coins_lock:
                old_set = set(self.top_coins)
                new_set = set(selected)

                added = new_set - old_set
                removed = old_set - new_set

                if added:
                    logger.info("➕ Added: %s", ", ".join(sorted(list(added))[:5]))
                if removed:
                    logger.info("➖ Removed: %s", ", ".join(sorted(list(removed))[:5]))

                self.top_coins = selected
                logger.info("✔ Loaded %d top USDT coins (requested %d)", len(self.top_coins), limit)

            self.error_counters["fetch_top_coins_failures"] = 0

        except Exception as e:
            self.error_counters["fetch_top_coins_failures"] = (
                self.error_counters.get("fetch_top_coins_failures", 0) + 1
            )
            logger.error(
                "Failed to fetch top coins (attempt %d): %s",
                self.error_counters["fetch_top_coins_failures"],
                e,
                exc_info=True,
            )
            async with self.top_coins_lock:
                if not self.top_coins:
                    self.top_coins = []

        async with self.top_coins_lock:
            return list(self.top_coins)

    async def _filter_futures_only(self, symbols: list[str]) -> list[str]:
        """
        Ensures only valid USDT PERPETUAL futures symbols remain.
        """
        futures_syms = await self._get_futures_trading_symbols()
        return [s for s in symbols if s in futures_syms]

    async def _ensure_matrices(self):
        async with self.top_coins_lock:
            symbols = await self._filter_futures_only(self.top_coins)

        for tf in self.active_timeframes:
            if tf not in self.matrices:
                self.matrices[tf] = MarketMatrix(
                    symbols=symbols,
                    timeframe=tf,
                    max_candles=self.max_candles,
                )
                logger.info("🧱 Created MarketMatrix for %s with %d symbols", tf, len(symbols))

    async def _rebuild_matrices_for_new_symbols(self):
        async with self.top_coins_lock:
            new_symbols = list(self.top_coins)

        for tf, old_matrix in list(self.matrices.items()):
            try:
                # If symbol sets are identical, skip
                if set(old_matrix.symbols) == set(new_symbols):
                    continue

                logger.info("🔄 Rebuilding MarketMatrix for %s (symbols changed)", tf)

                # Create new matrix with same capacity
                new_matrix = MarketMatrix(
                    symbols=new_symbols,
                    timeframe=tf,
                    max_candles=old_matrix.max_candles,
                )

                # For each symbol present in both old and new, copy row data under the old per-row lock.
                for sym in new_symbols:
                    old_idx = old_matrix.symbol_map.get(sym)
                    if old_idx is None:
                        # No historical data for this symbol in the old matrix; leave new row as-is (NaNs)
                        continue

                    new_idx = new_matrix.symbol_map[sym]

                    # Acquire the old matrix per-symbol lock to get a consistent snapshot of that row.
                    async with old_matrix._symbol_locks[old_idx]:
                        try:
                            # Copy core OHLCV and timestamps
                            new_matrix.closes[new_idx] = old_matrix.closes[old_idx].copy()
                            new_matrix.highs[new_idx] = old_matrix.highs[old_idx].copy()
                            new_matrix.lows[new_idx] = old_matrix.lows[old_idx].copy()
                            new_matrix.volumes[new_idx] = old_matrix.volumes[old_idx].copy()
                            new_matrix.timestamps[new_idx] = old_matrix.timestamps[old_idx].copy()

                            # Copy ring metadata
                            new_matrix.write_idx[new_idx] = int(old_matrix.write_idx[old_idx])
                            new_matrix.count[new_idx] = int(old_matrix.count[old_idx])

                            # Copy indicator arrays for this row (preserve dtype/shape)
                            try:
                                new_matrix.rsi[new_idx] = old_matrix.rsi[old_idx].copy()
                                new_matrix.mfi[new_idx] = old_matrix.mfi[old_idx].copy()
                                new_matrix.adx[new_idx] = old_matrix.adx[old_idx].copy()
                                new_matrix.atr[new_idx] = old_matrix.atr[old_idx].copy()
                                new_matrix.vol_z[new_idx] = old_matrix.vol_z[old_idx].copy()

                                new_matrix.ema_7[new_idx] = old_matrix.ema_7[old_idx].copy()
                                new_matrix.ema_14[new_idx] = old_matrix.ema_14[old_idx].copy()
                                new_matrix.ema_21[new_idx] = old_matrix.ema_21[old_idx].copy()
                                new_matrix.ema_50[new_idx] = old_matrix.ema_50[old_idx].copy()
                                new_matrix.ema_200[new_idx] = old_matrix.ema_200[old_idx].copy()

                                new_matrix.macd[new_idx] = old_matrix.macd[old_idx].copy()
                                new_matrix.macd_signal[new_idx] = old_matrix.macd_signal[old_idx].copy()
                                new_matrix.macd_hist[new_idx] = old_matrix.macd_hist[old_idx].copy()

                                new_matrix.stoch_k[new_idx] = old_matrix.stoch_k[old_idx].copy()
                                new_matrix.stoch_d[new_idx] = old_matrix.stoch_d[old_idx].copy()

                                new_matrix.vwap[new_idx] = old_matrix.vwap[old_idx].copy()
                                new_matrix.supertrend[new_idx] = old_matrix.supertrend[old_idx].copy()
                                new_matrix.supertrend_dir[new_idx] = old_matrix.supertrend_dir[old_idx].copy()
                            except Exception:
                                # Non-fatal: log and continue copying other symbols
                                logger.debug("Failed to copy some indicator arrays for %s (symbol=%s)", tf, sym, exc_info=True)

                        except Exception:
                            logger.exception("Failed to copy row for symbol %s from old matrix %s", sym, tf)
                            # continue with next symbol

                # Preserve matrix-level versions and last_calc_time
                try:
                    new_matrix.data_version = int(old_matrix.data_version)
                    new_matrix.indicator_version = int(old_matrix.indicator_version)
                    new_matrix.last_calc_time = float(old_matrix.last_calc_time)
                except Exception:
                    logger.debug("Failed to copy matrix-level metadata for %s", tf, exc_info=True)

                # Replace matrix atomically in the collector
                self.matrices[tf] = new_matrix
                logger.info("✅ Rebuilt MarketMatrix for %s with %d symbols", tf, len(new_symbols))

            except Exception:
                logger.exception("Failed to rebuild matrix for %s", tf)



    async def load_state(self) -> bool:
        loaded_any = False
        try:
            for tf in self.active_timeframes:
                path = os.path.join(self.output_dir, f"matrix_{tf}.npz")
                if not os.path.exists(path):
                    continue

                data = np.load(path, allow_pickle=True)
                symbols = list(data["symbols"])

                m = MarketMatrix(symbols, tf, max_candles=int(data["closes"].shape[1]))
                m.closes[:] = data["closes"]
                m.highs[:] = data["highs"]
                m.lows[:] = data["lows"]
                m.volumes[:] = data["volumes"]
                m.timestamps[:] = data["timestamps"]
                m.write_idx[:] = data["write_idx"]
                if "count" in data:
                    m.count[:] = data["count"]

                # Restore indicators if present
                if "rsi" in data:
                    try:
                        m.rsi[:] = data["rsi"]
                        m.mfi[:] = data["mfi"]
                        m.adx[:] = data["adx"]
                        m.atr[:] = data["atr"]
                        m.vol_z[:] = data["vol_z"]
                        m.ema_7[:] = data["ema_7"]
                        m.ema_14[:] = data["ema_14"]
                        m.ema_21[:] = data["ema_21"]
                        m.ema_50[:] = data["ema_50"]
                        m.ema_200[:] = data["ema_200"]
                        m.macd[:] = data["macd"]
                        m.macd_signal[:] = data["macd_signal"]
                        m.macd_hist[:] = data["macd_hist"]
                        m.stoch_k[:] = data["stoch_k"]
                        m.stoch_d[:] = data["stoch_d"]
                        m.vwap[:] = data["vwap"]
                        m.supertrend[:] = data["supertrend"]
                        m.supertrend_dir[:] = data["supertrend_dir"]
                        if "indicator_version" in data:
                            m.indicator_version = int(data["indicator_version"])
                        if "data_version" in data:
                            m.data_version = int(data["data_version"])
                        if "last_calc_time" in data:
                            m.last_calc_time = float(data["last_calc_time"])
                    except Exception:
                        logger.exception("Failed to restore indicators for %s", tf)

                self.matrices[tf] = m
                loaded_any = True

            if loaded_any:
                logger.info("✔ Restored MarketMatrix state from disk")
        except Exception:
            logger.exception("Failed to load state")

        return loaded_any

    async def save_state(self):
        try:
            for tf, m in self.matrices.items():
                path = os.path.join(self.output_dir, f"matrix_{tf}.npz")
                np.savez_compressed(
                    path,
                    symbols=np.array(m.symbols, dtype=object),
                    closes=m.closes,
                    highs=m.highs,
                    lows=m.lows,
                    volumes=m.volumes,
                    timestamps=m.timestamps,
                    write_idx=m.write_idx,
                    count=m.count,
                    # persist indicators and versions
                    rsi=m.rsi,
                    mfi=m.mfi,
                    adx=m.adx,
                    atr=m.atr,
                    vol_z=m.vol_z,

                    ema_7=m.ema_7,
                    ema_14=m.ema_14,
                    ema_21=m.ema_21,
                    ema_50=m.ema_50,
                    ema_200=m.ema_200,

                    macd=m.macd,
                    macd_signal=m.macd_signal,
                    macd_hist=m.macd_hist,

                    stoch_k=m.stoch_k,
                    stoch_d=m.stoch_d,

                    vwap=m.vwap,
                    supertrend=m.supertrend,
                    supertrend_dir=m.supertrend_dir,

                    indicator_version=m.indicator_version,
                    data_version=m.data_version,
                    last_calc_time=m.last_calc_time,
                )
            logger.info("💾 Saved MarketMatrix state to disk")
        except Exception:
            logger.exception("Failed to save state")
    # ---------------------------
    # Historical warmup (FUTURES klines)
    # ---------------------------
    async def fetch_historical_snapshot(self, limit_per_symbol: int = 1000):
        logger.info("📥 Fetching historical snapshot (warmup)...")
        await self._ensure_matrices()

        # conservative semaphore for klines warmup
        sem = asyncio.Semaphore(20)

        async with self.top_coins_lock:
            symbols = await self._filter_futures_only(self.top_coins)

        # helper to fetch one kline; pass estimated_weight based on limit
        async def fetch_klines(symbol: str, tf: str, limit: int):
            params = {"symbol": symbol, "interval": tf, "limit": limit}
            endpoint_key = f"futures:klines:{tf}"
            async with sem:
                try:
                    est_weight = 1.0
                    if limit >= 500:
                        est_weight = 5.0
                    elif limit >= 100:
                        est_weight = 2.0
                    data = await self.get_json(
                        market="futures",
                        path="/klines",
                        params=params,
                        endpoint_key=endpoint_key,
                        min_interval=0.5,
                        max_retries=3,
                        estimated_weight=est_weight,
                    )
                    return symbol, tf, data
                except Exception as e:
                    logger.warning("Failed to fetch klines for %s %s: %s", symbol, tf, e)
                    return symbol, tf, None

        # batch symbols to avoid huge bursts
        batch_size = 100
        for tf in self.active_timeframes:
            matrix = self.matrices.get(tf)
            if matrix is None:
                continue
            limit = min(matrix.max_candles, int(limit_per_symbol))

            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                tasks = [fetch_klines(s, tf, limit) for s in batch]
                results = await asyncio.gather(*tasks)
                for sym, _tf, klines in results:
                    if not klines:
                        logger.warning("Warmup: no klines for %s %s (symbol=%s)", _tf, tf, sym)
                        continue

                    # Normalize payload shape
                    if isinstance(klines, dict):
                        # Some APIs return {"code":..., "msg":...} or wrapped responses
                        logger.warning("Warmup: unexpected klines dict for %s %s: %s", sym, _tf, klines)
                        continue

                    if not isinstance(klines, list):
                        logger.warning("Warmup: unexpected klines payload type for %s %s: %s", sym, _tf, type(klines))
                        continue

                    # Build rows list (ts_ms, high, low, close, volume)
                    rows: List[Tuple[int, float, float, float, float]] = []
                    for k in klines:
                        try:
                            # Typical kline shape: [open_time, open, high, low, close, volume, close_time, ...]
                            ts_ms = int(k[6]) if len(k) > 6 else int(k[0])
                            high = float(k[2])
                            low = float(k[3])
                            close = float(k[4])
                            volume = float(k[5])
                            rows.append((ts_ms, high, low, close, volume))
                        except Exception:
                            logger.debug("Warmup: malformed kline for %s %s: %s", sym, _tf, k)

                    if not rows:
                        logger.warning("Warmup: no valid rows to bulk-load for %s %s", sym, _tf)
                        continue

                    # Ensure chronological order (oldest -> newest)
                    if rows[0][0] > rows[-1][0]:
                        rows.reverse()
                        logger.debug("Warmup: reversed rows for %s to ensure chronological order", sym)

                    logger.info("Warmup: bulk-loading %d rows for %s %s", len(rows), sym, _tf)

                    # Attempt bulk load under per-row lock; fall back to per-candle writes on failure
                    try:
                        await matrix.bulk_load_candles(sym, rows)
                        logger.debug("Warmup: bulk_load_candles succeeded for %s %s (rows=%d)", sym, _tf, len(rows))
                    except Exception as e:
                        logger.exception("Warmup: bulk_load_candles failed for %s %s: %s; falling back to per-candle writes", sym, _tf, e)
                        # Fallback: write candles one by one (older -> newer)
                        try:
                            for ts_ms, high, low, close, volume in rows:
                                try:
                                    await matrix.update_candle(sym, close, high, low, volume, ts_ms, True)
                                except Exception:
                                    logger.debug("Warmup: per-candle write failed for %s %s ts=%d", sym, _tf, ts_ms)
                                    continue
                        except Exception:
                            logger.exception("Warmup: per-candle fallback failed for %s %s", sym, _tf)
                            # small backoff to avoid tight failure loops
                            await asyncio.sleep(0.1)


                    except Exception:
                        # fallback to per-candle writes if bulk fails
                        for k in klines:
                            try:
                                ts_ms = int(k[6])
                                high = float(k[2])
                                low = float(k[3])
                                close = float(k[4])
                                volume = float(k[5])
                                await matrix.update_candle(sym, close, high, low, volume, ts_ms, True)
                            except Exception:
                                continue
                # small pause between batches to smooth weight
                await asyncio.sleep(0.25)
            # small pause between timeframes
            await asyncio.sleep(0.5)

        logger.info("✔ Historical warmup complete")
    # ---------------------------
    # WebSocket ingestion helpers
    # ---------------------------
    
    async def _run_warmup_indicators(self, batch_symbols: int = 256):
        """
        Run full-history indicator warmup using the Numba full-mode.
        - Processes each timeframe matrix in symbol batches to bound memory.
        - Uses counts returned by snapshot_consistent to compute only valid columns.
        """
        logger.info("🔁 Running full-history indicator warmup (Numba full-mode)")

        for tf, matrix in list(self.matrices.items()):
            try:
                snapshot_version, closes, highs, lows, volumes, counts = await matrix.snapshot_consistent()
                # Convert to contiguous arrays
                c_all = np.ascontiguousarray(closes, dtype=np.float64)
                h_all = np.ascontiguousarray(highs, dtype=np.float64)
                l_all = np.ascontiguousarray(lows, dtype=np.float64)
                v_all = np.ascontiguousarray(volumes, dtype=np.float64)
                counts_arr = np.ascontiguousarray(counts, dtype=np.int32)

                n_symbols, n_cols = c_all.shape
                if n_symbols == 0 or n_cols == 0:
                    continue

                bs = int(batch_symbols)
                for start in range(0, n_symbols, bs):
                    end = start + bs
                    if end > n_symbols:
                        end = n_symbols

                    # slice views for this batch (copy to avoid sharing with other threads)
                    c_batch = c_all[start:end].copy()
                    h_batch = h_all[start:end].copy()
                    l_batch = l_all[start:end].copy()
                    v_batch = v_all[start:end].copy()
                    counts_batch = counts_arr[start:end].copy()

                    shape = c_batch.shape
                    # Preallocate out buffers for the batch (float32)
                    out_bufs = {
                        "rsi": np.full(shape, np.nan, np.float32),
                        "mfi": np.full(shape, np.nan, np.float32),
                        "adx": np.full(shape, np.nan, np.float32),
                        "vol_z": np.full(shape, np.nan, np.float32),
                        "atr": np.full(shape, np.nan, np.float32),
                        "ema_7": np.full(shape, np.nan, np.float32),
                        "ema_14": np.full(shape, np.nan, np.float32),
                        "ema_21": np.full(shape, np.nan, np.float32),
                        "ema_50": np.full(shape, np.nan, np.float32),
                        "ema_200": np.full(shape, np.nan, np.float32),
                        "macd": np.full(shape, np.nan, np.float32),
                        "macd_signal": np.full(shape, np.nan, np.float32),
                        "macd_hist": np.full(shape, np.nan, np.float32),
                        "stoch_k": np.full(shape, np.nan, np.float32),
                        "stoch_d": np.full(shape, np.nan, np.float32),
                        "vwap": np.full(shape, np.nan, np.float32),
                        "supertrend": np.full(shape, np.nan, np.float32),
                        "supertrend_dir": np.full(shape, np.nan, np.float32),
                    }

                    # Run the heavy Numba full calc off the event loop
                    try:
                        await asyncio.to_thread(
                            calc_indicators_inplace,
                            c_batch, v_batch, h_batch, l_batch, counts_batch,
                            out_bufs["rsi"], out_bufs["mfi"], out_bufs["adx"], out_bufs["vol_z"], out_bufs["atr"],
                            out_bufs["ema_7"], out_bufs["ema_14"], out_bufs["ema_21"], out_bufs["ema_50"], out_bufs["ema_200"],
                            out_bufs["macd"], out_bufs["macd_signal"], out_bufs["macd_hist"],
                            out_bufs["stoch_k"], out_bufs["stoch_d"],
                            out_bufs["vwap"],
                            out_bufs["supertrend"], out_bufs["supertrend_dir"],
                            True  # full=True
                        )
                    except Exception:
                        logger.exception("Warmup Numba calc failed for %s batch %d:%d", tf, start, end)
                        # fallback to Python fallback (calc_indicators_inplace will run fallback)
                        try:
                            calc_indicators_inplace(
                                c_batch, v_batch, h_batch, l_batch, counts_batch,
                                out_bufs["rsi"], out_bufs["mfi"], out_bufs["adx"], out_bufs["vol_z"], out_bufs["atr"],
                                out_bufs["ema_7"], out_bufs["ema_14"], out_bufs["ema_21"], out_bufs["ema_50"], out_bufs["ema_200"],
                                out_bufs["macd"], out_bufs["macd_signal"], out_bufs["macd_hist"],
                                out_bufs["stoch_k"], out_bufs["stoch_d"],
                                out_bufs["vwap"],
                                out_bufs["supertrend"], out_bufs["supertrend_dir"],
                                True
                            )
                        except Exception:
                            logger.exception("Warmup fallback calc also failed for %s batch %d:%d", tf, start, end)
                            continue

                    # Commit the batch slice into the matrix
                    committed = await matrix.commit_indicators_slice(
                        snapshot_version,
                        start, end,
                        out_bufs["rsi"], out_bufs["mfi"], out_bufs["adx"], out_bufs["atr"], out_bufs["vol_z"],
                        out_bufs["ema_7"], out_bufs["ema_14"], out_bufs["ema_21"], out_bufs["ema_50"], out_bufs["ema_200"],
                        out_bufs["macd"], out_bufs["macd_signal"], out_bufs["macd_hist"],
                        out_bufs["stoch_k"], out_bufs["stoch_d"],
                        out_bufs["vwap"],
                        out_bufs["supertrend"], out_bufs["supertrend_dir"]
                    )
                    if committed:
                        logger.debug("Warmup batch committed for %s symbols %d:%d", tf, start, end)
                    else:
                        logger.warning("Warmup batch commit rejected for %s symbols %d:%d", tf, start, end)

                    # small pause to yield and avoid long blocking of event loop
                    await asyncio.sleep(0.01)

                logger.info("Warmup indicators completed for timeframe %s", tf)

            except Exception:
                logger.exception("Warmup failed for timeframe %s", tf)

    def _safe_parse_ws(self, raw: str) -> Optional[dict]:
        """
        Try orjson then json, return None on failure and increment parse counter.
        """
        try:
            return orjson.loads(raw)
        except Exception:
            try:
                return json.loads(raw)
            except Exception:
                # Sample raw payload in debug to avoid log spam
                logger.debug("WS parse failed (sampled): %s", raw[:200])
                self.error_counters["ws_parse_errors"] = self.error_counters.get("ws_parse_errors", 0) + 1
                return None
    def _enqueue_update(self, item: dict):
        """
        Enqueue update with bounded policy: drop and count if full.
        """
        try:
            self.update_queue.put_nowait(item)
        except asyncio.QueueFull:
            self.error_counters["queue_drops"] = self.error_counters.get("queue_drops", 0) + 1
            logger.debug("Dropped kline update for %s (queue full)", item.get("symbol"))
    def _process_kline_to_queue(self, data: dict):
        try:
            k = data.get("k")
            if not k:
                return

            tf = k.get("i")
            symbol = (data.get("s") or "").upper()
            if not tf or not symbol:
                return

            if tf not in self.matrices:
                return

            close = float(k.get("c", 0.0))
            high = float(k.get("h", 0.0))
            low = float(k.get("l", 0.0))
            volume = float(k.get("v", 0.0))
            ts_ms = int(k.get("T") or k.get("t") or 0)
            is_closed = bool(k.get("x", False))

            self._enqueue_update({
                "tf": tf,
                "symbol": symbol,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "ts_ms": ts_ms,
                "is_closed": is_closed,
            })

        except Exception:
            logger.exception("Error processing kline to queue")
    async def batch_update_processor(self):
        logger.info("🚀 Batch update processor started")
        while self.running:
            try:
                first = await asyncio.wait_for(self.update_queue.get(), timeout=1.0)
                batch = [first]

                deadline = time.time() + 0.05
                while len(batch) < 200 and time.time() < deadline:
                    try:
                        batch.append(self.update_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                by_tf: Dict[str, List[dict]] = {}
                for upd in batch:
                    by_tf.setdefault(upd["tf"], []).append(upd)

                for tf, updates in by_tf.items():
                    matrix = self.matrices.get(tf)
                    if matrix is None:
                        continue

                    for u in updates:
                        try:
                            await matrix.update_candle(
                                symbol=u["symbol"],
                                close=u["close"],
                                high=u["high"],
                                low=u["low"],
                                volume=u["volume"],
                                ts_ms=u["ts_ms"],
                                is_closed=u["is_closed"],
                            )
                        except Exception:
                            logger.exception("Failed to apply update to matrix %s", tf)

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Batch processor error: %s", e, exc_info=True)
                await asyncio.sleep(0.1)
    async def _ws_kline_handler(self, streams: List[str], ws_id: int):
        url = f"{self.futures_ws_base}/stream?streams=" + "/".join(streams)
        backoff = 1.0

        self.error_counters["ws_reconnections"].setdefault(ws_id, 0)

        while self.running:
            try:
                connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
                timeout = aiohttp.ClientTimeout(total=300)

                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.ws_connect(url, heartbeat=20) as ws:
                        logger.info("WS-%d: Connected (%d streams)", ws_id, len(streams))
                        backoff = 1.0
                        self.error_counters["ws_reconnections"][ws_id] += 1

                        last_msg_time = time.time()

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                last_msg_time = time.time()
                                try:
                                    payload = self._safe_parse_ws(msg.data)
                                    if not payload:
                                        continue
                                    data = payload.get("data", payload)
                                    if isinstance(data, dict) and data.get("e") == "kline":
                                        self._process_kline_to_queue(data)
                                except Exception:
                                    self.error_counters["ws_parse_errors"] = self.error_counters.get("ws_parse_errors", 0) + 1
                                    logger.exception("WS parse/process error")

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

                            if time.time() - last_msg_time > 60:
                                logger.warning("WS-%d: no messages for 60s, reconnecting", ws_id)
                                break

            except Exception as e:
                logger.error("WS-%d error: %s", ws_id, e, exc_info=True)
                backoff = min(backoff * 1.5, 300)
                await asyncio.sleep(backoff + random.uniform(0, 0.3 * backoff))
    async def _launch_ws_handlers(self):
        async with self.top_coins_lock:
            symbols = await self._filter_futures_only(self.top_coins)

        streams = []
        for sym in symbols:
            for tf in self.active_timeframes:
                streams.append(f"{sym.lower()}@kline_{tf}")

        chunk_size = 200
        chunks = [streams[i:i + chunk_size] for i in range(0, len(streams), chunk_size)]

        for ws_id, chunk in enumerate(chunks):
            # Use centralized task creation helper to register tasks
            self._create_task(self._ws_kline_handler(chunk, ws_id), name=f"ws_kline_{ws_id}")
    # ---------------------------
    # Indicator calculation loop
    # ---------------------------
    async def get_depth_snapshot(self) -> dict[str, dict]:
        out: Dict[str, dict] = {}
        # Get current symbols snapshot and filter to futures
        symbols_tuple = await self._get_symbols_snapshot_tuple()
        symbols = await self._filter_futures_only(list(symbols_tuple))

        for sym in symbols:
            state = self._symbol_states.get(sym)
            if not state:
                continue
            async with state.lock:
                if state.depth_metrics:
                    out[sym] = dict(state.depth_metrics)
        return out
    async def calculation_loop(self, interval_sec: float = 1.0):
        logger.info("📐 Indicator calculation loop started")

        # If no matrices yet, wait until they exist
        while not self.matrices and self.running:
            await asyncio.sleep(0.1)

        # Preallocate ALL output buffers for each timeframe (exact order for calc_indicators_inplace)
        indicator_bufs = {}
        for tf, m in list(self.matrices.items()):
            if tf not in indicator_bufs:
                try:
                    shape = m.closes.shape
                    indicator_bufs[tf] = {
                        "rsi": np.full(shape, np.nan, np.float32),
                        "mfi": np.full(shape, np.nan, np.float32),
                        "adx": np.full(shape, np.nan, np.float32),
                        "vol_z": np.full(shape, np.nan, np.float32),
                        "atr": np.full(shape, np.nan, np.float32),
                        "ema_7": np.full(shape, np.nan, np.float32),
                        "ema_14": np.full(shape, np.nan, np.float32),
                        "ema_21": np.full(shape, np.nan, np.float32),
                        "ema_50": np.full(shape, np.nan, np.float32),
                        "ema_200": np.full(shape, np.nan, np.float32),
                        "macd": np.full(shape, np.nan, np.float32),
                        "macd_signal": np.full(shape, np.nan, np.float32),
                        "macd_hist": np.full(shape, np.nan, np.float32),
                        "stoch_k": np.full(shape, np.nan, np.float32),
                        "stoch_d": np.full(shape, np.nan, np.float32),
                        "vwap": np.full(shape, np.nan, np.float32),
                        "supertrend": np.full(shape, np.nan, np.float32),
                        "supertrend_dir": np.full(shape, np.nan, np.float32),
                    }
                except Exception:
                    logger.exception("Failed to allocate indicator buffers for %s", tf)

        # Per-timeframe cadence control
        last_calc_ts = {tf: 0.0 for tf in self.matrices}
        min_interval = {
            "1m": 1.0,
            "5m": 2.0,
            "15m": 5.0,
            "1h": 10.0,
            "4h": 20.0,
            "1d": 30.0,
        }

        while self.running:
            loop_start = time.time()

            for tf, matrix in list(self.matrices.items()):
                try:
                    now = time.time()
                    if now - last_calc_ts.get(tf, 0.0) < min_interval.get(tf, 2.0):
                        continue

                    last_calc_ts[tf] = now

                    # RETRY LOOP - fixes the commit race condition
                    # inside calculation_loop, replace the retry loop with:
                    max_attempts = 6
                    base_backoff = 0.01
                    for attempt in range(max_attempts):
                        # Unpack counts returned by snapshot_consistent
                        snapshot_version, closes, highs, lows, volumes, counts = await matrix.snapshot_consistent()

                        # convert to contiguous arrays
                        try:
                            c = np.ascontiguousarray(closes, dtype=np.float64).copy()
                            h = np.ascontiguousarray(highs, dtype=np.float64).copy()
                            l = np.ascontiguousarray(lows, dtype=np.float64).copy()
                            v = np.ascontiguousarray(volumes, dtype=np.float64).copy()
                        except Exception:
                            logger.exception("Snapshot conversion failed for %s", tf)
                            break

                        out_bufs = indicator_bufs.get(tf)
                        if out_bufs is None:
                            logger.warning("No indicator buffers for %s; skipping", tf)
                            break

                        try:
                            calc_indicators_inplace(
                                c, v, h, l,
                                out_bufs["rsi"], out_bufs["mfi"], out_bufs["adx"], out_bufs["vol_z"], out_bufs["atr"],
                                out_bufs["ema_7"], out_bufs["ema_14"], out_bufs["ema_21"], out_bufs["ema_50"], out_bufs["ema_200"],
                                out_bufs["macd"], out_bufs["macd_signal"], out_bufs["macd_hist"],
                                out_bufs["stoch_k"], out_bufs["stoch_d"],
                                out_bufs["vwap"],
                                out_bufs["supertrend"], out_bufs["supertrend_dir"],
                                False
                            )
                        except Exception:
                            logger.exception("Indicator calc failed for %s (Numba or fallback)", tf)
                            break

                        committed = await matrix.commit_indicators(
                            snapshot_version,
                            out_bufs["rsi"], out_bufs["mfi"], out_bufs["adx"], out_bufs["atr"], out_bufs["vol_z"],
                            out_bufs["ema_7"], out_bufs["ema_14"], out_bufs["ema_21"], out_bufs["ema_50"], out_bufs["ema_200"],
                            out_bufs["macd"], out_bufs["macd_signal"], out_bufs["macd_hist"],
                            out_bufs["stoch_k"], out_bufs["stoch_d"],
                            out_bufs["vwap"],
                            out_bufs["supertrend"], out_bufs["supertrend_dir"]
                        )
                        if committed:
                            logger.debug("✅ Indicators committed for %s (v%d)", tf, snapshot_version)
                            break  # Success!

                        # Commit failed: fetch current data_version for diagnostics and backoff
                        with matrix._version_lock:
                            current_ver = matrix.data_version
                        logger.warning(
                            "⚠️ Indicators commit failed for %s (attempt %d/%d): snapshot_version=%d current_data_version=%d",
                            tf, attempt + 1, max_attempts, snapshot_version, current_ver
                        )

                        # Exponential backoff with jitter
                        await asyncio.sleep(base_backoff * (2 ** attempt) + random.uniform(0, base_backoff))
                    else:
                        logger.warning("❌ Failed to commit indicators for %s after %d retries", tf, max_attempts)


                except Exception:
                    logger.exception("💥 Indicator calc failed for %s", tf)

            # Control loop rate
            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0.0, float(interval_sec) - elapsed))
# unified_collector_part4.py (patched)
    # ---------------------------
    # Futures: Depth Collector
    # ---------------------------
    async def _depth_loop(self):
        """
        Periodically fetch depth for all tracked symbols.
        Uses the shared session from _ensure_session to avoid creating many connectors.
        """
        while self.running:
            await self._ingestion_paused.wait()
            symbols = await self._get_symbols_snapshot_tuple()
            if not symbols:
                await asyncio.sleep(self._depth_interval)
                continue

            # Limit concurrency to avoid bursts
            sem = asyncio.Semaphore(20)
            async def _task_for_symbol(sym: str):
                async with sem:
                    try:
                        await self._fetch_and_calculate_depth(sym)
                    except Exception:
                        logger.exception("Depth fetch failed for %s", sym)

            tasks = [self._create_task(_task_for_symbol(s), name=f"depth:{s}") for s in symbols]
            # Wait for tasks to finish but don't let one failing task cancel others
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(self._depth_interval)
    async def _fetch_and_calculate_depth(self, symbol: str):
        """
        Fetch order book depth and compute simple USD aggregates.
        Uses get_json to benefit from throttling and token-bucket.
        """
        state = self._symbol_states.get(symbol)
        if state is None:
            return

        try:
            # Use centralized REST helper to get consistent throttling and ban handling
            data = await self.get_json(
                market="futures",
                path="/depth",
                params={"symbol": symbol, "limit": 100},
                endpoint_key="futures:depth",
                min_interval=0.2,
                max_retries=2,
                estimated_weight=2.0,
            )
            if not data:
                return

            bids = data.get("bids", [])
            asks = data.get("asks", [])

            # Defensive parsing
            bid_usd = 0.0
            ask_usd = 0.0
            try:
                for p, q in bids:
                    bid_usd += float(p) * float(q)
                for p, q in asks:
                    ask_usd += float(p) * float(q)
            except Exception:
                logger.exception("Malformed depth entries for %s", symbol)

            denom = (bid_usd + ask_usd)
            imbalance = (bid_usd - ask_usd) / denom if denom > 0 else 0.0

            async with state.lock:
                state.depth_metrics = {
                    "bid_usd": float(bid_usd),
                    "ask_usd": float(ask_usd),
                    "imbalance": float(imbalance),
                    "ts": int(time.time() * 1000),
                }

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error fetching depth for %s", symbol)
    # ---------------------------
    # Futures: REST Collectors
    # ---------------------------
    async def _rest_scheduler(self):
        """
        Periodically fetch per-symbol REST metrics using centralized get_json.
        Runs in a loop and schedules per-symbol fetches concurrently with bounded concurrency.
        """
        while self.running:
            symbols = await self._get_symbols_snapshot_tuple()
            if not symbols:
                await asyncio.sleep(30)
                continue

            sem = asyncio.Semaphore(20)

            async def _fetch_all_for_symbol(s: str):
                async with sem:
                    try:
                        await asyncio.gather(
                            self._rest_fetch_premium_index(s),
                            self._rest_fetch_open_interest(s),
                            self._rest_fetch_ratios(s),
                            return_exceptions=True,
                        )
                    except Exception:
                        logger.exception("REST scheduler per-symbol error for %s", s)

            tasks = [self._create_task(_fetch_all_for_symbol(s), name=f"rest:{s}") for s in symbols]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            await asyncio.sleep(30)
    async def _rest_fetch_premium_index(self, symbol: str):
        state = self._symbol_states.get(symbol)
        if state is None:
            return

        try:
            data = await self.get_json(
                market="futures",
                path="/premiumIndex",
                params={"symbol": symbol},
                endpoint_key="futures:premiumIndex",
                min_interval=0.5,
                max_retries=2,
                estimated_weight=1.0,
            )
            if not data:
                return

            async with state.lock:
                state.rest_metrics["fundingRate"] = float(data.get("lastFundingRate", 0.0))
                state.rest_metrics["markPrice"] = float(data.get("markPrice", 0.0))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to fetch premium index for %s", symbol)
    async def _rest_fetch_open_interest(self, symbol: str):
        state = self._symbol_states.get(symbol)
        if state is None:
            return

        try:
            data = await self.get_json(
                market="futures",
                path="/openInterest",
                params={"symbol": symbol},
                endpoint_key="futures:openInterest",
                min_interval=0.5,
                max_retries=2,
                estimated_weight=1.0,
            )
            if not data:
                return

            async with state.lock:
                state.rest_metrics["openInterest"] = float(data.get("openInterest", 0.0))

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to fetch open interest for %s", symbol)
    async def _rest_fetch_ratios(self, symbol: str):
        state = self._symbol_states.get(symbol)
        if state is None:
            return

        try:
            # Use the explicit full endpoint URL stored in self._ratio_endpoint
            # and pass params directly to session.get to avoid double-prepending bases.
            session = await self._ensure_session()
            url = f"{self._ratio_endpoint}"
            params = {"symbol": symbol, "period": "5m", "limit": 1}

            async with session.get(url, params=params) as resp:
                if resp.status == 404:
                    # Endpoint not found: log once per symbol and skip
                    logger.warning("Ratio endpoint 404 for %s: %s", symbol, url)
                    return
                if resp.status == 429:
                    logger.warning("Ratio endpoint rate limited for %s", symbol)
                    return
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("Unexpected status %s from %s: %s", resp.status, url, text[:200])
                    return

                data = await resp.json()

            # Normalize response shape: API returns a list for this endpoint
            val = 0.0
            if isinstance(data, list) and data:
                try:
                    val = float(data[0].get("longShortRatio", 0.0))
                except Exception:
                    logger.debug("Malformed ratio payload for %s: %s", symbol, data)
                    val = 0.0
            elif isinstance(data, dict):
                val = float(data.get("longShortRatio", 0.0))

            async with state.lock:
                state.rest_metrics["longShortRatio"] = val

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to fetch ratios for %s", symbol)
    # ---------------------------
    # Futures: Liquidation WebSocket + rolling aggregation
    # ---------------------------
    async def _liq_ws_loop(self):
        """
        Connect to the all-market liquidation stream and process events.
        Uses a dedicated session per connection (closed on exit) and robust parsing.
        """
        backoff = 1.0
        while self.running:
            try:
                connector = aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
                timeout = aiohttp.ClientTimeout(total=300)
                async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                    async with session.ws_connect(self._liq_allmarket_ws, heartbeat=20) as ws:
                        logger.info("💀 Liquidation stream connected (!forceOrder@arr)")
                        backoff = 1.0
                        self.error_counters["liq_ws_reconnects"] += 1

                        async for msg in ws:
                            if not self.running:
                                break

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                try:
                                    payload = self._safe_parse_ws(msg.data)
                                    if not payload:
                                        continue
                                    # payload may be an array or dict; normalize
                                    # The payload structure for forceOrder@arr is typically an array of events
                                    if isinstance(payload, list):
                                        for item in payload:
                                            await self._process_liq_event(item)
                                    else:
                                        await self._process_liq_event(payload)
                                except Exception:
                                    self.error_counters["liq_ws_parse_errors"] = self.error_counters.get("liq_ws_parse_errors", 0) + 1
                                    logger.exception("Liq WS parse/process error")

                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                break

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Liq stream disconnected: %s", e, exc_info=True)
                backoff = min(backoff * 1.5, 300)
                await asyncio.sleep(backoff + random.uniform(0, 0.3 * backoff))
    async def _process_liq_event(self, data: dict):
        """
        Process a single liquidation event payload.
        This method is async because it uses the async liq lock and may await.
        """
        payload = data.get("o") or data.get("data", {}).get("o") or data.get("o", {}) or data
        if not payload:
            return

        try:
            symbol = (payload.get("s") or "").upper()
            side = payload.get("S")  # SELL = long liq, BUY = short liq
            qty = float(payload.get("q", 0.0))
            price = float(payload.get("p", 0.0))
            ts_ms = int(payload.get("T") or data.get("E") or int(time.time() * 1000))

            if not symbol or side not in ("BUY", "SELL") or qty <= 0 or price <= 0:
                return
            if symbol not in self._symbol_states:
                return

            usd_val = qty * price
            ts_s = ts_ms / 1000.0

            # instead of: self._symbol_states[symbol].liquidations.append({...})
            await self._symbol_states[symbol].append_liq({
                "ts": ts_ms,
                "side": side,
                "usd": usd_val,
                "price": price
            })

            # Rolling aggregate (use async lock)
            async with self._liq_lock:
                self.history.append((ts_s, symbol, side, usd_val, price))
                stats = self.symbol_stats[symbol]
                stats["last_price"] = price
                if side == "SELL":
                    stats["24h_long"] += usd_val
                    stats["1h_long"] += usd_val
                else:
                    stats["24h_short"] += usd_val
                    stats["1h_short"] += usd_val

        except Exception:
            logger.exception("Error processing liq event")
    async def _liq_cleanup_loop(self):
        while self.running:
            await asyncio.sleep(60)
            try:
                await self._liq_cleanup_once()
            except Exception:
                logger.exception("Liq cleanup loop error")
    async def _liq_cleanup_once(self):
        """
        Recompute rolling aggregates from history and trim old entries.
        Uses async lock for consistency with async producers.
        """
        now = time.time()
        cutoff_24h = now - 86400
        cutoff_1h = now - 3600

        async with self._liq_lock:
            # Trim history older than 24h
            while self.history and self.history[0][0] < cutoff_24h:
                self.history.popleft()

            new_stats = collections.defaultdict(lambda: {
                "1h_long": 0.0, "1h_short": 0.0,
                "24h_long": 0.0, "24h_short": 0.0,
                "last_price": 0.0,
            })

            # Preserve last_price from previous stats where possible
            for sym, old_s in self.symbol_stats.items():
                new_stats[sym]["last_price"] = old_s.get("last_price", 0.0)

            for ts, sym, side, usd, price in self.history:
                s = new_stats[sym]
                if side == "SELL":
                    s["24h_long"] += usd
                    if ts > cutoff_1h:
                        s["1h_long"] += usd
                else:
                    s["24h_short"] += usd
                    if ts > cutoff_1h:
                        s["1h_short"] += usd
                s["last_price"] = price

            self.symbol_stats = new_stats
    # ---------------------------
    # Symbol auto-refresh (every hour)
    # ---------------------------
    async def symbol_refresh_task(self):
        logger.info("🔄 Symbol Auto-Refresh Task Started (every 1h)")
        # initial cooldown logic moved into loop
        while self.running:
            try:
                # initial sleep for 1 hour between refreshes
                await asyncio.sleep(3600)

                async with self.top_coins_lock:
                    old_symbols = set(self.top_coins)

                await self.fetch_top_coins(limit=self.top_limit)

                async with self.top_coins_lock:
                    new_symbols = set(self.top_coins)

                if old_symbols != new_symbols:
                    logger.info("📊 Symbol list changed — rebuilding matrices + futures states")
                    await self._rebuild_matrices_for_new_symbols()
                    await self.update_symbols(list(new_symbols))
                    logger.info("✅ Symbol refresh complete")

            except Exception:
                logger.exception("Symbol refresh failed; cooling down for 5 minutes")
                await asyncio.sleep(300)
    # ---------------------------
    # Health / Network logging
    # ---------------------------
    async def health_log_task(self, sample_interval: float = 60.0):
        """
        Periodically sample CPU/memory and system network counters and append to history.
        Also logs a compact health line. Runs as a background task.
        """
        try:
            process = _ps.Process(os.getpid())
            cur = _ps.net_io_counters()
            self._health_prev_net = {
                "ts": time.time(),
                "bytes_sent": getattr(cur, "bytes_sent", 0),
                "bytes_recv": getattr(cur, "bytes_recv", 0),
            }
            self._health_net_history.append(self._health_prev_net.copy())
        except Exception:
            process = None
            self._health_prev_net = None

        def _fmt_rate(bps: float) -> str:
            if bps >= 1024 * 1024:
                return f"{bps / (1024*1024):.2f}MB/s"
            if bps >= 1024:
                return f"{bps / 1024:.1f}KB/s"
            return f"{bps:.0f}B/s"

        while self.running:
            await asyncio.sleep(sample_interval)
            try:
                # CPU / memory / conns
                if process:
                    cpu_usage = process.cpu_percent(interval=None)
                    mem_mb = process.memory_info().rss / 1024 / 1024
                    try:
                        if hasattr(process, "net_connections"):
                            conn_count = len(process.net_connections(kind="inet"))
                        elif hasattr(process, "connections"):
                            conn_count = len(process.connections(kind="inet"))
                        else:
                            conn_count = 0
                    except Exception:
                        conn_count = 0
                else:
                    cpu_usage, mem_mb, conn_count = 0.0, 0.0, 0

                # Network throughput (system-wide) - sample now
                net_sent_rate = net_recv_rate = 0.0
                try:
                    cur = _ps.net_io_counters()
                    now = time.time()
                    prev = getattr(self, "_health_prev_net", None)
                    if prev is not None:
                        dt = max(1e-6, now - prev["ts"])
                        net_sent_rate = (getattr(cur, "bytes_sent", 0) - prev["bytes_sent"]) / dt
                        net_recv_rate = (getattr(cur, "bytes_recv", 0) - prev["bytes_recv"]) / dt
                    # update snapshot and append to history
                    self._health_prev_net = {
                        "ts": now,
                        "bytes_sent": getattr(cur, "bytes_sent", 0),
                        "bytes_recv": getattr(cur, "bytes_recv", 0),
                    }
                    self._health_net_history.append(self._health_prev_net.copy())
                except Exception:
                    net_sent_rate = net_recv_rate = 0.0

                # Matrix calc lags
                now_ts = time.time()
                lags = []
                for tf, m in self.matrices.items():
                    lag = now_ts - getattr(m, "last_calc_time", 0)
                    lags.append(f"{tf}:{lag:.1f}s")
                lag_str = " | ".join(lags)

                total_symbols = sum(getattr(m, "n_symbols", 0) for m in self.matrices.values())
                suite_info = ""
                # best-effort: futures_suite may not exist
                if getattr(self, "futures_suite", None):
                    suite_info = (
                        f" | LiqWSre:{self.futures_suite.counters.get('liq_ws_reconnects', 0)}"
                        f" | LiqParseErr:{self.futures_suite.counters.get('liq_ws_parse_errors', 0)}"
                        f" | Depth429:{self.futures_suite.counters.get('depth_429', 0)}"
                    )

                logger.info(
                    "💚 HEALTH | CPU: %.1f%% | Mem: %.0fMB | Net: ↑ %s ↓ %s | Conns: %d | "
                    "Sym: %d | Queue: %d | Calc Lag: [%s]%s",
                    cpu_usage,
                    mem_mb,
                    _fmt_rate(net_sent_rate),
                    _fmt_rate(net_recv_rate),
                    conn_count,
                    total_symbols,
                    self.update_queue.qsize(),
                    lag_str,
                    suite_info,
                )
            except Exception:
                logger.exception("Health check error")
    # ---------------------------
    # Internal network logger
    # ---------------------------
    async def _network_logger(self, interval_sec: int = 1, log_interval: int = 10):
        """
        Background task: every `log_interval` seconds, compute total network usage
        over the last `interval_sec` seconds and log it.
        """
        def _fmt_bytes(n: int) -> str:
            if n >= 1024 * 1024:
                return f"{n / (1024*1024):.2f}MB"
            if n >= 1024:
                return f"{n / 1024:.1f}KB"
            return f"{n}B"

        try:
            while self.running:
                await asyncio.sleep(log_interval)
                try:
                    # ensure we have a fresh sample in history before computing
                    try:
                        cur = _ps.net_io_counters()
                        now = time.time()
                        self._health_net_history.append({
                            "ts": now,
                            "bytes_sent": getattr(cur, "bytes_sent", 0),
                            "bytes_recv": getattr(cur, "bytes_recv", 0),
                        })
                    except Exception:
                        pass

                    usage = self.get_network_usage(interval_sec)
                    sent = usage.get("bytes_sent", 0)
                    recv = usage.get("bytes_recv", 0)
                    sent_rate = (usage.get("sent_rate", 0.0)/1024**2)
                    recv_rate = (usage.get("recv_rate", 0.0)/1024**2)

                    logger.info(
                        "📡 NET | last %ds: ↑ %s  (%.3f MB/s) ↓ %s (%.3f MB/s)",
                        int(interval_sec),
                        _fmt_bytes(sent),
                        sent_rate,
                        _fmt_bytes(recv),
                        recv_rate,
                    )
                except Exception:
                    logger.exception("Network logger error")
                    await asyncio.sleep(5)
        except asyncio.CancelledError:
            return
    def get_network_usage(self, interval_sec: float) -> dict:
        """
        Return total network usage over the last `interval_sec` seconds.
        If history is insufficient, returns totals from oldest available sample to latest.
        """
        now = time.time()
        history = getattr(self, "_health_net_history", None)
        if not history or len(history) == 0:
            return {"interval": interval_sec, "bytes_sent": 0, "bytes_recv": 0, "sent_rate": 0.0, "recv_rate": 0.0}

        latest = history[-1]
        target_ts = now - float(interval_sec)

        # find oldest sample with ts <= target_ts; if none, use oldest available
        oldest = None
        for sample in history:
            if sample["ts"] <= target_ts:
                oldest = sample
                break
        if oldest is None:
            oldest = history[0]

        bytes_sent = max(0, int(latest["bytes_sent"] - oldest["bytes_sent"]))
        bytes_recv = max(0, int(latest["bytes_recv"] - oldest["bytes_recv"]))

        # Use requested interval for deterministic rate calculation
        dt = max(1e-6, float(interval_sec))
        sent_rate = bytes_sent / dt
        recv_rate = bytes_recv / dt

        return {
            "interval": interval_sec,
            "bytes_sent": bytes_sent,
            "bytes_recv": bytes_recv,
            "sent_rate": sent_rate,
            "recv_rate": recv_rate,
        }
    # ---------------------------
    # Start / Run / Shutdown
    # ---------------------------
    async def start(self, warmup: bool = False, load_state: bool = True, start_network_logger: bool = True):
        """
        Start core services and background tasks. This method registers tasks in self.tasks.
        """
        if load_state:
            try:
                await self.load_state()
            except Exception:
                logger.exception("load_state failed during start")

        await self.fetch_top_coins(limit=self.top_limit)
        await self._ensure_matrices()

        symbols = await self.get_symbols_snapshot()
        await self.update_symbols(symbols)
        # call once early (e.g., in start before _run_warmup_indicators)
        try:
            dummy_c = np.zeros((1, 10), dtype=np.float64)
            dummy_v = np.zeros((1, 10), dtype=np.float64)
            dummy_h = np.zeros((1, 10), dtype=np.float64)
            dummy_l = np.zeros((1, 10), dtype=np.float64)
            dummy_counts = np.array([10], dtype=np.int32)
            dummy_out = np.zeros((1, 10), dtype=np.float32)
            # run in thread to compile
            await asyncio.to_thread(
                calc_indicators_inplace,
                dummy_c, dummy_v, dummy_h, dummy_l, dummy_counts,
                dummy_out, dummy_out, dummy_out, dummy_out, dummy_out,
                dummy_out, dummy_out, dummy_out, dummy_out, dummy_out,
                dummy_out, dummy_out, dummy_out,
                dummy_out, dummy_out,
                dummy_out,
                dummy_out, dummy_out,
                False
            )
        except Exception:
            logger.debug("Numba warmup compile failed or skipped", exc_info=True)

        if warmup:
            # run warmup synchronously here (or set warmup=False to run in background)
            await self.fetch_historical_snapshot(limit_per_symbol=self.max_candles)
            await self._run_warmup_indicators(batch_symbols=256)

        # Core tasks (use _create_task to register)
        self._create_task(self.batch_update_processor(), name="batch_updates")
        self._create_task(self.calculation_loop(), name="calc_indicators")
        self._create_task(self.symbol_refresh_task(), name="symbol_refresh")

        # health + network logging
        self._create_task(self.health_log_task(sample_interval=60.0), name="health_loop")
        if start_network_logger:
            self._create_task(self._network_logger(interval_sec=1, log_interval=10), name="net_logger")

        # Futures tasks
        self._create_task(self._depth_loop(), name="futures_depth")
        self._create_task(self._rest_scheduler(), name="futures_rest")
        self._create_task(self._liq_ws_loop(), name="liq_ws")
        self._create_task(self._liq_cleanup_loop(), name="liq_cleanup")

        # Kline WS tasks (launches and appends to self.tasks internally)
        await self._launch_ws_handlers()

        logger.info("🚀 BinanceUnifiedCollector started")
    async def run(self):
        """
        Convenience runner: start with defaults and wait for tasks to finish.
        """
        await self.start(warmup=True, load_state=True, start_network_logger=True)
        try:
            # Wait until all tasks complete (they typically run until shutdown)
            await asyncio.gather(*list(self.tasks))
        except asyncio.CancelledError:
            pass
        finally:
            await self.shutdown()
    async def shutdown(self):
        logger.info("🛑 Shutting down BinanceUnifiedCollector...")
        self.running = False
        self._ingestion_paused.set()

        try:
            await self.save_state()
        except Exception:
            logger.exception("save_state failed during shutdown")

        # cancel tasks
        for t in list(self.tasks):
            try:
                t.cancel()
            except Exception:
                pass

        if self.tasks:
            try:
                await asyncio.gather(*list(self.tasks), return_exceptions=True)
            except Exception:
                logger.exception("Error awaiting tasks during shutdown")

        # close shared session if present
        try:
            if getattr(self, "_session", None) and not getattr(self._session, "closed", True):
                await self._session.close()
        except Exception:
            logger.exception("Error closing shared session during shutdown")
        self._session = None

        # close token bucket
        try:
            await self._weight_bucket.close()
        except Exception:
            logger.exception("Error closing token bucket during shutdown")

        logger.info("✔ Shutdown complete")
    async def close(self):
        """
        Backwards-compatible alias for shutdown/cleanup.
        """
        await self.shutdown()
    async def close(self) -> None:
        """
        Graceful shutdown for the collector: stop tasks, close session, and close token bucket.
        """
        self.running = False

        tasks_snapshot = list(self.tasks)
        for t in tasks_snapshot:
            try:
                t.cancel()
            except Exception:
                pass

            if tasks_snapshot:
                results = await asyncio.gather(*tasks_snapshot, return_exceptions=True)
                for t, r in zip(tasks_snapshot, results):
                    if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                        try:
                            name = t.get_name() if hasattr(t, "get_name") else repr(t)
                        except Exception:
                            name = repr(t)
                        self.logger.error("Task %s raised during close: %s", name, r)
            self.tasks.clear()

        session = getattr(self, "_session", None)
        if session is not None:
            try:
                if not getattr(session, "closed", True):
                    await session.close()
            except Exception:
                logger.exception("Error closing shared session during shutdown")
            finally:
                self._session = None
        try:
            await self._weight_bucket.close()
        except Exception:
            pass

# ---------------------------
# Minimal runner (if executed as script)
# ---------------------------
if __name__ == "__main__":
    import logging
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    async def main():
        bot = BinanceUnifiedCollector()
        try:
            await bot.run()
        except asyncio.CancelledError:
            raise
        finally:
            bot.running = False
            for t in list(bot.tasks):
                try:
                    t.cancel()
                except Exception:
                    pass
            if bot.tasks:
                await asyncio.gather(*bot.tasks, return_exceptions=True)
            await bot.close()

        m = bot.get_matrix("1m")
        if m:
            print(m.symbols[:5], m.count.sum(), np.nanmax(m.closes))
            print("count:", m.count.sum())
            print("any RSI:", np.nanmax(m.rsi))
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
