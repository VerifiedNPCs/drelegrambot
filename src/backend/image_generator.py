#image_generator.py
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import io
import numpy as np
import re
import logging
import gc
import asyncio
from typing import List, Dict, Any
from config import setup_logging

logger = setup_logging("ImageGen")
# ------------------------------------------------------------------------------
# Data Enrichment Logic (Moved from Bot)
# ------------------------------------------------------------------------------

def _normalize_symbol(sym: str) -> str:
    """Ensures symbol is uppercase and ends with USDT."""
    s = (sym or "").upper().strip()
    if not s:
        return s
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s

async def _enrich_with_liquidations(img_data: List[Dict], collector: Any) -> None:
    """
    Populates 'liq_1h_long', 'liq_1h_short' from the collector's liquidation monitor.
    """
    if not img_data or not collector:
        return
    
    for row in img_data:
        sym = row.get("symbol_raw") or row.get("symbol") or ""
        sym = _normalize_symbol(sym)
        if not sym: continue

        try:
            # Requires get_liq_stats() to be implemented in Data_Collector
            if hasattr(collector, 'get_liq_stats'):
                stats = collector.get_liq_stats(sym)
                if stats:
                    row.update({
                        'liq_1h_long': stats.get('1h_long', 0.0),
                        'liq_1h_short': stats.get('1h_short', 0.0),
                        'liq_24h_long': stats.get('24h_long', 0.0),
                        'liq_24h_short': stats.get('24h_short', 0.0)
                    })
        except Exception:
            pass

async def _enrich_with_depth(img_data: List[Dict], collector: Any, target_usdt: float) -> None:
    """
    Adds depth/liquidity data to image rows by checking the collector's order book.
    """
    if not img_data or not collector:
        return

    # Extract symbols needing depth check
    syms = []
    for row in img_data:
        ds = row.get("depth_symbol") or row.get("symbol_raw") or row.get("symbol")
        if not isinstance(ds, str):
            continue
        base = ds.split()[0].strip().split("(")[0].strip()
        base = _normalize_symbol(base)
        if base:
            syms.append(base)

    # De-duplicate
    uniq = list(set(syms))
    if not uniq:
        return

    # Cap watchlist size to prevent overload
    MAX_WATCHLIST = 20
    if len(uniq) > MAX_WATCHLIST:
        uniq = uniq[:MAX_WATCHLIST]

    try:
        # Tell collector to watch these symbols on Futures WebSocket
        if hasattr(collector, 'set_depth_watchlist'):
            await collector.set_depth_watchlist(uniq)
            # Brief pause to let WS connect/snapshot
            await asyncio.sleep(0.5)
    except Exception:
        logger.exception("Failed to set depth watchlist")
        return

    # Fetch data
    for row in img_data:
        try:
            ds = row.get("depth_symbol") or row.get("symbol_raw") or row.get("symbol")
            if not isinstance(ds, str):
                continue

            base = ds.split()[0].strip().split("(")[0].strip()
            base = _normalize_symbol(base)
            if not base:
                continue

            mid_hint = row.get("price")
            if not isinstance(mid_hint, (int, float)) or mid_hint <= 0:
                mid_hint = None

            # Get liquidity range from Collector's in-memory book
            rng = None
            if hasattr(collector, 'get_depth_liquidity_range'):
                rng = collector.get_depth_liquidity_range(base, target_usdt, mid_price=mid_hint)

                # Optional retry if data missing initially
                if rng is None:
                    await asyncio.sleep(0.1)
                    rng = collector.get_depth_liquidity_range(base, target_usdt, mid_price=mid_hint)

            row["depth_target_usdt"] = float(target_usdt)

            if rng is None:
                # Fill Nones so generator handles it gracefully
                row.update({
                    "depth_down_price": None, "depth_up_price": None,
                    "depth_down_pct": None, "depth_up_pct": None
                })
                continue

            row.update({
                "depth_down_price": rng.get("down_price"),
                "depth_up_price": rng.get("up_price"),
                "depth_down_pct": rng.get("down_pct"),
                "depth_up_pct": rng.get("up_pct"),
                "depth_bid_notional": rng.get("bid_notional"),
                "depth_ask_notional": rng.get("ask_notional"),
                "depth_best_bid": rng.get("best_bid"),
                "depth_best_ask": rng.get("best_ask"),
                "depth_mid": rng.get("mid"),
                "depth_book_ts_ms": rng.get("book_ts_ms")
            })

        except Exception:
            logger.exception("Depth enrichment failed for row: %s", row.get("symbol"))

async def enrich_data(img_data: List[Dict], collector: Any, target_usdt: float) -> None:
    """
    Public entry point to enrich data before generation.
    Call this from the bot before calling generate_market_image.
    """
    await _enrich_with_depth(img_data, collector, target_usdt)
    await _enrich_with_liquidations(img_data, collector)


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def clean_text(text):
    """Removes non-ASCII characters to prevent Matplotlib warnings."""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def fmt_large_num(num):
    """Formats large numbers (Volume / Notional) like 1.2M, 500K."""
    try:
        num = float(num)
    except Exception:
        return "0"
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f}B"
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.0f}K"
    return f"{num:.0f}"


def fmt_price_extended(val):
    """Smart price formatting."""
    try:
        val = float(val)
    except Exception:
        return "0.00"

    if val == 0:
        return "0.00"
    if val < 0.0001:
        return f"{val:.8f}"
    if val < 0.01:
        return f"{val:.6f}"
    if val < 1:
        return f"{val:.4f}"
    if val < 100:
        return f"{val:.2f}"
    return f"{val:,.0f}"


def _safe_float(x, default=None):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v):
            return default
        return v
    except Exception:
        return default


def _depth_available(item: dict) -> bool:
    """Checks if row has valid depth fields for liquidity bar."""
    dp = item.get("depth_down_price", None)
    up = item.get("depth_up_price", None)
    return _safe_float(dp, None) is not None and _safe_float(up, None) is not None


# ------------------------------------------------------------------------------
# Core Generator
# ------------------------------------------------------------------------------
def generate_market_image(data_list):
    """
    Generates a market snapshot image using Matplotlib (OO Interface).
    Supports RSI, MFI, ADX, Sparklines, Liquidity/Range bars, and Liquidation Ratios.
    """
    if not data_list:
        return None

    # --- CONSTANTS & CONFIG ---
    ROWS = len(data_list)
    ROW_HEIGHT = 0.6
    HEADER_HEIGHT = 0.8
    FIG_H = (ROWS * ROW_HEIGHT) + HEADER_HEIGHT
    FIG_W = 20  # Width

    # Liquidity bar scaling: 5% distance fills the half-bar
    LIQ_MAX_PCT_DISPLAY = 5.0

    # Colors
    BG_COLOR = '#0e1117'
    TEXT_COLOR = '#ffffff'
    SUB_TEXT_COLOR = "#A1A1A1"
    GREEN = '#00ff7f'
    RED = '#ff4d4d'
    YELLOW = '#ffd700'
    BLUE = '#00bfff'
    DIVIDER_COLOR = '#222222'
    MID_DOT = '#ffffff'
    BAR_GREY = '#444444'

    # Layout Positions (0.0 to 1.0)
    X_SYM = 0.02
    X_TF = 0.08
    X_PRICE = 0.16
    X_CHG = 0.22
    X_VOL = 0.28
    
    # Indicators
    X_RSI = 0.35
    X_MFI = 0.40
    X_ADX = 0.45

    # New Liquidation Column
    X_LIQ = 0.52 
    W_LIQ = 0.10

    # Charts (Small)
    X_SPARK = 0.65
    W_SPARK = 0.12
    X_RANGE = 0.84
    W_RANGE = 0.12

    # --- FIGURE INITIALIZATION ---
    # Use Figure directly (no pyplot) to avoid memory leaks
    fig = Figure(figsize=(FIG_W, FIG_H), dpi=100)
    canvas = FigureCanvasAgg(fig)
    fig.patch.set_facecolor(BG_COLOR)

    # Main Axes for Text
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.axis('off')
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)

    # --- HEADER ---
    header_y = 1.0 - (0.5 / FIG_H)

    def add_text(x, y, text, color=TEXT_COLOR, size=11, weight='normal', align='left'):
        ax_bg.text(
            x, y, str(text),
            color=color, fontsize=size, fontweight=weight,
            ha=align, va='center'
        )

    # Draw Headers
    add_text(X_SYM, header_y, "SYMBOL", SUB_TEXT_COLOR, 11, 'bold', 'left')
    add_text(X_TF, header_y, "TF", SUB_TEXT_COLOR, 11, 'bold', 'left')
    add_text(X_PRICE, header_y, "PRICE", SUB_TEXT_COLOR, 11, 'bold', 'right')
    add_text(X_CHG, header_y, "24H %", SUB_TEXT_COLOR, 11, 'bold', 'right')
    add_text(X_VOL, header_y, "VOL ($)", SUB_TEXT_COLOR, 11, 'bold', 'right')

    add_text(X_RSI, header_y, "RSI", SUB_TEXT_COLOR, 11, 'bold', 'center')
    add_text(X_MFI, header_y, "MFI", SUB_TEXT_COLOR, 11, 'bold', 'center')
    add_text(X_ADX, header_y, "ADX", SUB_TEXT_COLOR, 11, 'bold', 'center')

    # Liquidation Header
    liq_center = X_LIQ + (W_LIQ / 2)
    add_text(liq_center, header_y, "LIQ (1H)", SUB_TEXT_COLOR, 11, 'bold', 'center')

    trend_center = X_SPARK + (W_SPARK / 2)
    add_text(trend_center, header_y, "TREND", SUB_TEXT_COLOR, 11, 'bold', 'center')

    range_center = X_RANGE + (W_RANGE / 2)
    add_text(range_center, header_y, "RANGE", SUB_TEXT_COLOR, 11, 'bold', 'center')

    # Header Divider Line
    line_y = header_y - (0.3 / FIG_H)
    ax_bg.plot([0.02, 0.98], [line_y, line_y], color=DIVIDER_COLOR, lw=1)

    # --- ROWS ---
    row_start_y = line_y - (0.4 / FIG_H)
    y_step = ROW_HEIGHT / FIG_H

    for i, item in enumerate(data_list):
        y = row_start_y - (i * y_step)

        # Extract Data
        raw_symbol = item.get('symbol', 'N/A')
        symbol = clean_text(raw_symbol)
        tf = clean_text(item.get('tf', ''))
        price = _safe_float(item.get('price', 0), 0.0) or 0.0
        change = _safe_float(item.get('change', 0), 0.0) or 0.0
        usdt_vol = _safe_float(item.get('usdt_volume', 0), 0.0) or 0.0

        rsi = _safe_float(item.get('rsi', 50), 50.0) or 50.0
        mfi = _safe_float(item.get('mfi', 50), 50.0) or 50.0
        adx = _safe_float(item.get('adx', 0), 0.0) or 0.0
        history = item.get('history', [])

        # Liquidation Data (New)
        liq_long = _safe_float(item.get('liq_1h_long', 0), 0.0)
        liq_short = _safe_float(item.get('liq_1h_short', 0), 0.0)

        is_up = change >= 0
        val_color = GREEN if is_up else RED

        # 1. Text Columns
        add_text(X_SYM, y, symbol, 'white', 12, 'bold', 'left')
        add_text(X_TF, y, tf, SUB_TEXT_COLOR, 11, 'normal', 'left')
        add_text(X_PRICE, y, f"${fmt_price_extended(price)}", 'white', 12, 'normal', 'right')
        add_text(X_CHG, y, f"{change:+.2f}%", val_color, 12, 'normal', 'right')
        add_text(X_VOL, y, fmt_large_num(usdt_vol), 'white', 11, 'normal', 'right')

        # 2. Indicators
        rsi_c = RED if rsi > 70 else (GREEN if rsi < 30 else TEXT_COLOR)
        add_text(X_RSI, y, f"{rsi:.0f}", rsi_c, 12, 'bold', 'center')

        mfi_c = RED if mfi > 80 else (GREEN if mfi < 20 else TEXT_COLOR)
        add_text(X_MFI, y, f"{mfi:.0f}", mfi_c, 12, 'bold', 'center')

        adx_c = SUB_TEXT_COLOR
        if adx > 25:
            adx_c = YELLOW
        if adx > 50:
            adx_c = BLUE
        add_text(X_ADX, y, f"{adx:.0f}", adx_c, 12, 'bold', 'center')

        sp_h = 0.5 * y_step
        sp_y = y - (sp_h / 2)

        # 3. Liquidation Bar Chart
        total_liq = liq_long + liq_short
        if total_liq > 0:
            ax_liq = fig.add_axes([X_LIQ, sp_y + 0.005, W_LIQ, sp_h * 0.4])
            ax_liq.axis('off')
            
            # Normalize to 0-1
            long_pct = liq_long / total_liq
            short_pct = liq_short / total_liq
            
            # Draw bars (Longs = RED because they sold, Shorts = GREEN because they bought)
            ax_liq.barh([0], [long_pct], color=RED, height=0.8, left=0)
            ax_liq.barh([0], [short_pct], color=GREEN, height=0.8, left=long_pct)
            ax_liq.set_xlim(0, 1)

            # Text label below bar
            label_txt = f"L:{fmt_large_num(liq_long)} S:{fmt_large_num(liq_short)}"
            add_text(X_LIQ + (W_LIQ/2), y - 0.015, label_txt, SUB_TEXT_COLOR, 8, 'normal', 'center')
        else:
            add_text(X_LIQ + (W_LIQ/2), y, "-", SUB_TEXT_COLOR, 11, 'normal', 'center')


        # Clean History
        hist_clean = []
        if isinstance(history, np.ndarray):
            history = history.tolist()

        if history:
            for val in history[-20:]:
                fval = _safe_float(val, None)
                if fval is not None and fval > 0:
                    hist_clean.append(fval)

        # 4. Sparkline (Trend)
        if len(hist_clean) > 1:
            ax_spark = fig.add_axes([X_SPARK, sp_y, W_SPARK, sp_h])
            ax_spark.patch.set_alpha(0)  # Transparent background
            ax_spark.axis('off')
            
            low_p = min(hist_clean)
            line_color = GREEN if hist_clean[-1] >= hist_clean[0] else RED
            
            ax_spark.plot(hist_clean, color=line_color, lw=1.5)
            ax_spark.fill_between(range(len(hist_clean)), hist_clean, low_p, color=line_color, alpha=0.1)

        # 5. Range Bar / Liquidity Range Bar
        ax_range = fig.add_axes([X_RANGE, sp_y, W_RANGE, sp_h])
        ax_range.patch.set_alpha(0)
        ax_range.axis('off')
        ax_range.set_xlim(0, 1)
        ax_range.set_ylim(0, 1)

        # Draw Base Line (Grey bar background)
        ax_range.plot([0, 1], [0.5, 0.5], color=BAR_GREY, lw=2, zorder=1)
        center = 0.5

        if _depth_available(item):
            # --- LIQUIDITY RANGE BAR ---
            down_price = _safe_float(item.get("depth_down_price"), None)
            up_price = _safe_float(item.get("depth_up_price"), None)
            down_pct = _safe_float(item.get("depth_down_pct"), None)
            up_pct = _safe_float(item.get("depth_up_pct"), None)

            # Recalculate percent if missing but prices exist
            if (down_pct is None or up_pct is None) and (down_price is not None and up_price is not None and price > 0):
                down_pct = max(0.0, (price - down_price) / price * 100.0)
                up_pct = max(0.0, (up_price - price) / price * 100.0)

            down_pct = float(down_pct) if down_pct is not None else 0.0
            up_pct = float(up_pct) if up_pct is not None else 0.0

            # Scale and Clamp
            down_len = min(max(down_pct, 0.0) / LIQ_MAX_PCT_DISPLAY, 1.0) * 0.5
            up_len = min(max(up_pct, 0.0) / LIQ_MAX_PCT_DISPLAY, 1.0) * 0.5

            # Draw Depth Bars (Red/Green extending from center)
            if down_len > 0:
                ax_range.plot([center - down_len, center], [0.5, 0.5], color=RED, lw=3, zorder=2)
            if up_len > 0:
                ax_range.plot([center, center + up_len], [0.5, 0.5], color=GREEN, lw=3, zorder=2)

            # Center Dot (Current Price)
            ax_range.scatter(
                [center], [0.5],
                color=MID_DOT, s=40, zorder=3,
                edgecolors='black', linewidth=0.6,
                clip_on=False
            )

            # Labels (Depth Prices)
            if down_price is not None:
                add_text(X_RANGE - 0.015, y, fmt_price_extended(down_price), SUB_TEXT_COLOR, 9, 'normal', 'right')
            if up_price is not None:
                add_text(X_RANGE + W_RANGE + 0.015, y, fmt_price_extended(up_price), SUB_TEXT_COLOR, 9, 'normal', 'left')

        else:
            # --- CANDLE RANGE BAR (Fallback) ---
            if len(hist_clean) > 1:
                curr_price = hist_clean[-1]
                low_v = min(hist_clean)
                high_v = max(hist_clean)
                rng = high_v - low_v
                
                # Position of current price within the range (0.0 - 1.0)
                pct = (curr_price - low_v) / rng if rng > 0 else 0.5
                pct = max(0.0, min(1.0, pct))
                dot_color = GREEN if pct > 0.5 else RED

                # Draw Dot
                ax_range.scatter(
                    [pct], [0.5],
                    color=dot_color, s=50, zorder=2,
                    edgecolors='white', linewidth=1,
                    clip_on=False
                )

                # Labels (Low/High of candle history)
                add_text(X_RANGE - 0.015, y, fmt_price_extended(low_v), SUB_TEXT_COLOR, 9, 'normal', 'right')
                add_text(X_RANGE + W_RANGE + 0.015, y, fmt_price_extended(high_v), SUB_TEXT_COLOR, 9, 'normal', 'left')

    # --- OUTPUT ---
    buf = io.BytesIO()
    try:
        canvas.print_png(buf)
        buf.seek(0)
        return buf
    except Exception:
        logger.exception("Failed to render PNG image")
        return None
    finally:
        # Crucial Memory Cleanup
        try:
            plt.close(fig)
            fig.clf()
            del canvas
            del fig
            gc.collect()
        except Exception:
            pass
