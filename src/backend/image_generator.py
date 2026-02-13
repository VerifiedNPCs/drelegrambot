import matplotlib
# Set backend to Agg for non-interactive, headless generation (MUST be first)
matplotlib.use('Agg')
import matplotlib.style as mplstyle
# Use 'fast' style to disable unnecessary features
mplstyle.use('fast')

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import io
import numpy as np
import re
import logging
import gc
import asyncio
from typing import List, Dict, Any

# Keep your existing imports
try:
    from config import setup_logging
    logger = setup_logging("ImageGen")
except ImportError:
    # Fallback if config is missing during testing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ImageGen")

# ------------------------------------------------------------------------------
# Data Enrichment Logic (Preserved)
# ------------------------------------------------------------------------------

def _normalize_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    if not s:
        return s
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s

async def _enrich_with_liquidations(img_data: List[Dict], collector: Any) -> None:
    if not img_data or not collector:
        return
    
    for row in img_data:
        sym = row.get("symbol_raw") or row.get("symbol") or ""
        sym = _normalize_symbol(sym)
        if not sym: continue

        try:
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
    if not img_data or not collector:
        return

    syms = []
    for row in img_data:
        ds = row.get("depth_symbol") or row.get("symbol_raw") or row.get("symbol")
        if not isinstance(ds, str):
            continue
        base = ds.split()[0].strip().split("(")[0].strip()
        base = _normalize_symbol(base)
        if base:
            syms.append(base)

    uniq = list(set(syms))
    if not uniq:
        return

    MAX_WATCHLIST = 20
    if len(uniq) > MAX_WATCHLIST:
        uniq = uniq[:MAX_WATCHLIST]

    try:
        if hasattr(collector, 'set_depth_watchlist'):
            await collector.set_depth_watchlist(uniq)
            await asyncio.sleep(0.5)
    except Exception:
        logger.exception("Failed to set depth watchlist")
        return

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

            rng = None
            if hasattr(collector, 'get_depth_liquidity_range'):
                rng = collector.get_depth_liquidity_range(base, target_usdt, mid_price=mid_hint)
                if rng is None:
                    await asyncio.sleep(0.1)
                    rng = collector.get_depth_liquidity_range(base, target_usdt, mid_price=mid_hint)

            row["depth_target_usdt"] = float(target_usdt)

            if rng is None:
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
    await _enrich_with_depth(img_data, collector, target_usdt)
    await _enrich_with_liquidations(img_data, collector)

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()

def fmt_large_num(num):
    try:
        num = float(num)
    except Exception:
        return "0"
    if num >= 1_000_000_000: return f"{num/1_000_000_000:.1f}B"
    if num >= 1_000_000: return f"{num/1_000_000:.1f}M"
    if num >= 1_000: return f"{num/1_000:.0f}K"
    return f"{num:.0f}"

def fmt_price_extended(val):
    try:
        val = float(val)
    except Exception:
        return "0.00"
    if val == 0: return "0.00"
    if val < 0.0001: return f"{val:.8f}"
    if val < 0.01: return f"{val:.6f}"
    if val < 1: return f"{val:.4f}"
    if val < 100: return f"{val:.2f}"
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
    dp = item.get("depth_down_price", None)
    up = item.get("depth_up_price", None)
    return _safe_float(dp, None) is not None and _safe_float(up, None) is not None

# ------------------------------------------------------------------------------
# Optimized Generator Class
# ------------------------------------------------------------------------------
class MarketImageGenerator:
    """
    Optimized generator that reuses Figure instances and uses direct 
    coordinate drawing to avoid expensive Axes creation.
    """
    def __init__(self):
        # Constants
        self.ROW_HEIGHT = 0.6
        self.HEADER_HEIGHT = 0.8
        self.FIG_W = 20
        self.BG_COLOR = '#0e1117'
        self.TEXT_COLOR = '#ffffff'
        self.SUB_TEXT_COLOR = "#A1A1A1"
        self.DIVIDER_COLOR = '#222222'
        self.RED = '#ff4d4d'
        self.GREEN = '#00ff7f'
        self.YELLOW = '#ffd700'
        self.BLUE = '#00bfff'
        self.BAR_GREY = '#444444'
        self.MID_DOT = '#ffffff'

        # Layout X Positions
        self.X_SYM = 0.02
        self.X_TF = 0.08
        self.X_PRICE = 0.16
        self.X_CHG = 0.22
        self.X_VOL = 0.28
        self.X_RSI = 0.35
        self.X_MFI = 0.40
        self.X_ADX = 0.45
        self.X_LIQ = 0.52
        self.W_LIQ = 0.10
        self.X_SPARK = 0.65
        self.W_SPARK = 0.12
        self.X_RANGE = 0.84
        self.W_RANGE = 0.12

        # Initialize Figure ONCE
        # We start with a default size, will resize if needed
        self.fig = Figure(figsize=(self.FIG_W, 10), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.fig.patch.set_facecolor(self.BG_COLOR)
        
        # Single main axis for everything
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')

    def render(self, data_list: List[Dict]) -> io.BytesIO:
        if not data_list:
            return None

        rows_count = len(data_list)
        needed_height = (rows_count * self.ROW_HEIGHT) + self.HEADER_HEIGHT
        
        # Resize figure if the height requirement changed significantly
        # (Avoid resizing for micro-adjustments to save speed)
        curr_w, curr_h = self.fig.get_size_inches()
        if abs(curr_h - needed_height) > 0.1:
            self.fig.set_size_inches(self.FIG_W, needed_height)
        
        # Clear previous content
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        # Pre-calculation for Y coordinates
        # Map 0..1 to the figure coordinates
        # In Matplotlib 0,0 is bottom-left.
        # We draw from top down.
        
        header_y = 1.0 - (0.5 / needed_height)
        line_y = header_y - (0.3 / needed_height)
        
        # Draw Header Divider
        self.ax.plot([0.02, 0.98], [line_y, line_y], color=self.DIVIDER_COLOR, lw=1)
        
        # Draw Headers
        self._draw_headers(header_y)

        # Draw Rows
        row_start_y = line_y - (0.4 / needed_height)
        y_step = self.ROW_HEIGHT / needed_height
        
        # Optimization: Collect patch objects and add them in batches if possible, 
        # or just add iteratively. Iterative add_patch is fast enough for <100 objects.
        
        for i, item in enumerate(data_list):
            y = row_start_y - (i * y_step)
            self._draw_row(i, item, y, y_step, needed_height)

        # Output to buffer
        buf = io.BytesIO()
        try:
            self.canvas.print_png(buf)
            buf.seek(0)
            return buf
        except Exception:
            logger.exception("Failed to render PNG image")
            return None
        # No need to close/clf/gc.collect() - we reuse self.fig

    def _draw_headers(self, y):
        """Draws the table headers."""
        bold = 'bold'
        size = 11
        sub = self.SUB_TEXT_COLOR
        
        t = self.ax.text
        t(self.X_SYM, y, "SYMBOL", color=sub, fontsize=size, fontweight=bold, ha='left', va='center')
        t(self.X_TF, y, "TF", color=sub, fontsize=size, fontweight=bold, ha='left', va='center')
        t(self.X_PRICE, y, "PRICE", color=sub, fontsize=size, fontweight=bold, ha='right', va='center')
        t(self.X_CHG, y, "24H %", color=sub, fontsize=size, fontweight=bold, ha='right', va='center')
        t(self.X_VOL, y, "VOL ($)", color=sub, fontsize=size, fontweight=bold, ha='right', va='center')
        t(self.X_RSI, y, "RSI", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_MFI, y, "MFI", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_ADX, y, "ADX", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        
        t(self.X_LIQ + (self.W_LIQ/2), y, "LIQ (1H)", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_SPARK + (self.W_SPARK/2), y, "TREND", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_RANGE + (self.W_RANGE/2), y, "RANGE", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')

    def _draw_row(self, i, item, y, y_step, total_h):
        """Draws a single data row."""
        # --- Data Extraction ---
        symbol = clean_text(item.get('symbol', 'N/A'))
        tf = clean_text(item.get('tf', ''))
        price = _safe_float(item.get('price', 0), 0.0) or 0.0
        change = _safe_float(item.get('change', 0), 0.0) or 0.0
        usdt_vol = _safe_float(item.get('usdt_volume', 0), 0.0) or 0.0
        rsi = _safe_float(item.get('rsi', 50), 50.0) or 50.0
        mfi = _safe_float(item.get('mfi', 50), 50.0) or 50.0
        adx = _safe_float(item.get('adx', 0), 0.0) or 0.0
        
        is_up = change >= 0
        val_color = self.GREEN if is_up else self.RED
        
        # --- Text Columns ---
        t = self.ax.text
        t(self.X_SYM, y, symbol, color='white', fontsize=12, fontweight='bold', ha='left', va='center')
        t(self.X_TF, y, tf, color=self.SUB_TEXT_COLOR, fontsize=11, ha='left', va='center')
        t(self.X_PRICE, y, f"${fmt_price_extended(price)}", color='white', fontsize=12, ha='right', va='center')
        t(self.X_CHG, y, f"{change:+.2f}%", color=val_color, fontsize=12, ha='right', va='center')
        t(self.X_VOL, y, fmt_large_num(usdt_vol), color='white', fontsize=11, ha='right', va='center')

        # Indicators
        rsi_c = self.RED if rsi > 70 else (self.GREEN if rsi < 30 else self.TEXT_COLOR)
        t(self.X_RSI, y, f"{rsi:.0f}", color=rsi_c, fontsize=12, fontweight='bold', ha='center', va='center')
        
        mfi_c = self.RED if mfi > 80 else (self.GREEN if mfi < 20 else self.TEXT_COLOR)
        t(self.X_MFI, y, f"{mfi:.0f}", color=mfi_c, fontsize=12, fontweight='bold', ha='center', va='center')
        
        adx_c = self.SUB_TEXT_COLOR
        if adx > 25: adx_c = self.YELLOW
        if adx > 50: adx_c = self.BLUE
        t(self.X_ADX, y, f"{adx:.0f}", color=adx_c, fontsize=12, fontweight='bold', ha='center', va='center')

        # --- Visuals Calculation ---
        sp_h = 0.5 * y_step
        sp_y_bottom = y - (sp_h / 2) # Bottom of the visual element
        
        # 1. Liquidation Bars (Using Rectangles directly on main axis)
        liq_long = _safe_float(item.get('liq_1h_long', 0), 0.0)
        liq_short = _safe_float(item.get('liq_1h_short', 0), 0.0)
        total_liq = liq_long + liq_short
        
        if total_liq > 0:
            long_pct = liq_long / total_liq
            short_pct = liq_short / total_liq
            
            # Bar geometry
            bar_x = self.X_LIQ
            bar_y = sp_y_bottom + (0.005 / total_h) # slight offset
            bar_w = self.W_LIQ
            bar_h = sp_h * 0.4
            
            # Add Patches
            if long_pct > 0:
                self.ax.add_patch(Rectangle((bar_x, bar_y), bar_w * long_pct, bar_h, 
                                            facecolor=self.RED, edgecolor=None))
            if short_pct > 0:
                self.ax.add_patch(Rectangle((bar_x + (bar_w * long_pct), bar_y), bar_w * short_pct, bar_h, 
                                            facecolor=self.GREEN, edgecolor=None))
            
            label_txt = f"L:{fmt_large_num(liq_long)} S:{fmt_large_num(liq_short)}"
            t(self.X_LIQ + (self.W_LIQ/2), y - (0.015 / total_h), label_txt, 
              color=self.SUB_TEXT_COLOR, fontsize=8, ha='center', va='top')
        else:
            t(self.X_LIQ + (self.W_LIQ/2), y, "-", color=self.SUB_TEXT_COLOR, fontsize=11, ha='center', va='center')

        # 2. Sparklines (Direct Line Drawing + Shadow)
        history = item.get('history', [])
        if isinstance(history, np.ndarray): history = history.tolist()
        hist_clean = [float(x) for x in history[-20:] if _safe_float(x) is not None and x > 0]
        
        if len(hist_clean) > 1:
            h_arr = np.array(hist_clean)
            h_min, h_max = h_arr.min(), h_arr.max()
            h_range = h_max - h_min if h_max > h_min else 1.0
            
            # Normalize to 0-1
            norm_h = (h_arr - h_min) / h_range
            
            # Scale to Sparkline Box dimensions
            # X coordinates
            x_vals = np.linspace(self.X_SPARK, self.X_SPARK + self.W_SPARK, len(hist_clean))
            # Y coordinates
            y_vals = sp_y_bottom + (norm_h * sp_h)
            
            line_color = self.GREEN if hist_clean[-1] >= hist_clean[0] else self.RED
            
            # -- SHADOW (Polygon) --
            # Create a closed polygon: [Start, Points..., End, Start]
            # We want it to go down to sp_y_bottom (the floor of the sparkline area)
            
            # Add first point (bottom-left)
            poly_points = [(x_vals[0], sp_y_bottom)]
            # Add line points
            poly_points.extend(zip(x_vals, y_vals))
            # Add last point (bottom-right)
            poly_points.append((x_vals[-1], sp_y_bottom))
            
            # Add Polygon Patch (Fast Shadow)
            poly = Polygon(poly_points, closed=True, facecolor=line_color, edgecolor=None, alpha=0.15)
            self.ax.add_patch(poly)
            
            # -- LINE --
            self.ax.add_line(Line2D(x_vals, y_vals, color=line_color, lw=1.5))


        # 3. Range Bar (Direct Drawing)
        range_center_x = self.X_RANGE + (self.W_RANGE / 2)
        range_y_center = y # vertically centered on row
        range_w_half = self.W_RANGE / 2
        
        # Base Line
        self.ax.add_line(Line2D([self.X_RANGE, self.X_RANGE + self.W_RANGE], 
                                [range_y_center, range_y_center], 
                                color=self.BAR_GREY, lw=2, zorder=1))

        if _depth_available(item):
            # Depth Logic
            depth_down = _safe_float(item.get("depth_down_pct"), 0.0) or 0.0
            depth_up = _safe_float(item.get("depth_up_pct"), 0.0) or 0.0
            
            # Scale 5% -> Full Width
            LIQ_MAX_PCT = 5.0
            down_len_norm = min(max(depth_down, 0.0) / LIQ_MAX_PCT, 1.0)
            up_len_norm = min(max(depth_up, 0.0) / LIQ_MAX_PCT, 1.0)
            
            # Draw Red/Green Bars
            if down_len_norm > 0:
                x_start = range_center_x - (down_len_norm * range_w_half)
                self.ax.add_line(Line2D([x_start, range_center_x], [range_y_center, range_y_center], 
                                        color=self.RED, lw=3, zorder=2))
            if up_len_norm > 0:
                x_end = range_center_x + (up_len_norm * range_w_half)
                self.ax.add_line(Line2D([range_center_x, x_end], [range_y_center, range_y_center], 
                                        color=self.GREEN, lw=3, zorder=2))
            
            # Center Dot
            self.ax.scatter([range_center_x], [range_y_center], color=self.MID_DOT, s=40, zorder=3, edgecolors='black', linewidth=0.6)
            
            # Labels
            dp = item.get("depth_down_price")
            up = item.get("depth_up_price")
            if dp: t(self.X_RANGE - 0.005, y, fmt_price_extended(dp), color=self.SUB_TEXT_COLOR, fontsize=9, ha='right', va='center')
            if up: t(self.X_RANGE + self.W_RANGE + 0.005, y, fmt_price_extended(up), color=self.SUB_TEXT_COLOR, fontsize=9, ha='left', va='center')
            
        elif len(hist_clean) > 1:
            # Candle Range Fallback
            curr = hist_clean[-1]
            low_v, high_v = min(hist_clean), max(hist_clean)
            rng = high_v - low_v
            
            if rng > 0:
                pct = (curr - low_v) / rng
                pct = max(0.0, min(1.0, pct))
                dot_x = self.X_RANGE + (pct * self.W_RANGE)
                dot_c = self.GREEN if pct > 0.5 else self.RED
                
                self.ax.scatter([dot_x], [range_y_center], color=dot_c, s=50, zorder=2, edgecolors='white', linewidth=1)
                
                t(self.X_RANGE - 0.005, y, fmt_price_extended(low_v), color=self.SUB_TEXT_COLOR, fontsize=9, ha='right', va='center')
                t(self.X_RANGE + self.W_RANGE + 0.005, y, fmt_price_extended(high_v), color=self.SUB_TEXT_COLOR, fontsize=9, ha='left', va='center')

# ------------------------------------------------------------------------------
# Global Instance (Singleton) for Performance
# ------------------------------------------------------------------------------
_generator = MarketImageGenerator()

def generate_market_image(data_list):
    """
    Public API wrapper. Uses the persistent _generator instance to ensure speed.
    """
    return _generator.render(data_list)
