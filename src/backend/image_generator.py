# image_generator.py (final, corrected)
import matplotlib
matplotlib.use('Agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Rectangle, Polygon
from matplotlib.lines import Line2D
import io
import numpy as np
import re
import logging
import asyncio
from typing import List, Dict, Any

try:
    from config import setup_logging
    logger = setup_logging("ImageGen")
except Exception:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("ImageGen")


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def _normalize_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    if not s:
        return s
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s


def clean_text(text):
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'[^\x00-\x7F]+', '', text).strip()


def fmt_large_num(num):
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


# -------------------------------------------------------------------------
# Enrichment helpers (used by services before rendering)
# -------------------------------------------------------------------------
async def _enrich_with_liquidations(img_data: List[Dict], collector: Any) -> None:
    if not img_data or not collector:
        return
    for row in img_data:
        sym = row.get("symbol_raw") or row.get("symbol") or ""
        sym = _normalize_symbol(sym)
        if not sym:
            continue
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
            logger.debug("Failed to enrich liquidation for %s", sym, exc_info=False)


async def _enrich_with_depth(img_data: List[Dict], collector: Any, target_usdt: float) -> None:
    if not img_data or not collector:
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
                    await asyncio.sleep(0.02)
                    rng = collector.get_depth_liquidity_range(base, target_usdt, mid_price=mid_hint)

            row["depth_target_usdt"] = float(target_usdt)
            if rng is None:
                row.update({
                    "depth_down_price": None, "depth_up_price": None,
                    "depth_down_pct": None, "depth_up_pct": None,
                    "depth_bid_notional": None, "depth_ask_notional": None,
                    "depth_mid": None, "depth_book_ts_ms": None,
                    "depth_bins": None
                })
                continue

            # Expect rng to include bid_bins and ask_bins arrays (20 bins each)
            row.update({
                "depth_down_price": rng.get("down_price"),
                "depth_up_price": rng.get("up_price"),
                "depth_down_pct": rng.get("down_pct"),
                "depth_up_pct": rng.get("up_pct"),
                "depth_bid_notional": rng.get("bid_notional"),
                "depth_ask_notional": rng.get("ask_notional"),
                "depth_best_bid": rng.get("mid"),
                "depth_best_ask": rng.get("mid"),
                "depth_mid": rng.get("mid"),
                "depth_book_ts_ms": rng.get("book_ts_ms"),
                "depth_bins": rng.get("bins") or {
                    "bid_bins": rng.get("bid_bins"),
                    "ask_bins": rng.get("ask_bins")
                }
            })
        except Exception:
            logger.exception("Depth enrichment failed for row: %s", row.get("symbol"))


async def enrich_data(img_data: List[Dict], collector: Any, target_usdt: float) -> None:
    await _enrich_with_depth(img_data, collector, target_usdt)
    await _enrich_with_liquidations(img_data, collector)


# -------------------------------------------------------------------------
# Image generator (renders depth mountain + liquidity + indicators)
# -------------------------------------------------------------------------
class MarketImageGenerator:
    def __init__(self):
        # Layout constants
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

        # X positions
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

        # Figure
        self.fig = Figure(figsize=(self.FIG_W, 10), dpi=100)
        self.canvas = FigureCanvasAgg(self.fig)
        self.fig.patch.set_facecolor(self.BG_COLOR)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.axis('off')

    def render(self, data_list: List[Dict]) -> io.BytesIO:
        if not data_list:
            return None

        rows_count = len(data_list)
        needed_height = (rows_count * self.ROW_HEIGHT) + self.HEADER_HEIGHT

        curr_w, curr_h = self.fig.get_size_inches()
        if abs(curr_h - needed_height) > 0.1:
            self.fig.set_size_inches(self.FIG_W, needed_height)

        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        header_y = 1.0 - (0.5 / needed_height)
        line_y = header_y - (0.3 / needed_height)
        self.ax.plot([0.02, 0.98], [line_y, line_y], color=self.DIVIDER_COLOR, lw=1)
        self._draw_headers(header_y)

        row_start_y = line_y - (0.4 / needed_height)
        y_step = self.ROW_HEIGHT / needed_height

        for i, item in enumerate(data_list):
            y = row_start_y - (i * y_step)
            self._draw_row(i, item, y, y_step, needed_height)

        buf = io.BytesIO()
        try:
            self.canvas.print_png(buf)
            buf.seek(0)
            return buf
        except Exception:
            logger.exception("Failed to render PNG image")
            return None

    def _draw_headers(self, y):
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
        t(self.X_LIQ + (self.W_LIQ / 2), y, "LIQ (1H)", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_SPARK + (self.W_SPARK / 2), y, "TREND", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')
        t(self.X_RANGE + (self.W_RANGE / 2), y, "RANGE", color=sub, fontsize=size, fontweight=bold, ha='center', va='center')

    def _draw_row(self, i, item, y, y_step, total_h):
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

        t = self.ax.text
        t(self.X_SYM, y, symbol, color='white', fontsize=12, fontweight='bold', ha='left', va='center')
        t(self.X_TF, y, tf, color=self.SUB_TEXT_COLOR, fontsize=11, ha='left', va='center')
        t(self.X_PRICE, y, f"${fmt_price_extended(price)}", color='white', fontsize=12, ha='right', va='center')
        t(self.X_CHG, y, f"{change:+.2f}%", color=val_color, fontsize=12, ha='right', va='center')
        t(self.X_VOL, y, fmt_large_num(usdt_vol), color='white', fontsize=11, ha='right', va='center')

        rsi_c = self.RED if rsi > 70 else (self.GREEN if rsi < 30 else self.TEXT_COLOR)
        t(self.X_RSI, y, f"{rsi:.0f}", color=rsi_c, fontsize=12, fontweight='bold', ha='center', va='center')

        mfi_c = self.RED if mfi > 80 else (self.GREEN if mfi < 20 else self.TEXT_COLOR)
        t(self.X_MFI, y, f"{mfi:.0f}", color=mfi_c, fontsize=12, fontweight='bold', ha='center', va='center')

        adx_c = self.SUB_TEXT_COLOR
        if adx > 25:
            adx_c = self.YELLOW
        if adx > 50:
            adx_c = self.BLUE
        t(self.X_ADX, y, f"{adx:.0f}", color=adx_c, fontsize=12, fontweight='bold', ha='center', va='center')

        # -------------------------
        # Liquidity bar (1H long vs short)
        # -------------------------
        sp_h = 0.5 * y_step
        sp_y_bottom = y - (sp_h / 2)

        liq_long = _safe_float(item.get('liq_1h_long', 0), 0.0)
        liq_short = _safe_float(item.get('liq_1h_short', 0), 0.0)
        total_liq = liq_long + liq_short

        bar_x = self.X_LIQ
        bar_y = sp_y_bottom + (0.005 / total_h)
        bar_w = self.W_LIQ
        bar_h = sp_h * 0.4

        if total_liq > 0:
            long_pct = liq_long / total_liq
            short_pct = liq_short / total_liq

            left_w = bar_w * long_pct
            right_w = bar_w * short_pct

            if left_w > 0:
                self.ax.add_patch(Rectangle((bar_x, bar_y), left_w, bar_h, facecolor=self.RED, edgecolor=None))
            if right_w > 0:
                self.ax.add_patch(Rectangle((bar_x + left_w, bar_y), right_w, bar_h, facecolor=self.GREEN, edgecolor=None))

            label_txt = f"L:{fmt_large_num(liq_long)} S:{fmt_large_num(liq_short)}"
            t(self.X_LIQ + (self.W_LIQ / 2), y - (0.015 / total_h), label_txt, color=self.SUB_TEXT_COLOR, fontsize=8, ha='center', va='top')
        else:
            t(self.X_LIQ + (self.W_LIQ / 2), y, "-", color=self.SUB_TEXT_COLOR, fontsize=11, ha='center', va='center')

        # -------------------------
        # Sparklines (trend)
        # -------------------------
        history = item.get('history', [])
        if isinstance(history, np.ndarray):
            history = history.tolist()
        hist_clean = [float(x) for x in history[-20:] if _safe_float(x) is not None and x > 0]

        x_vals = []
        y_vals = []
        if len(hist_clean) > 1:
            h_arr = np.array(hist_clean)
            h_min, h_max = h_arr.min(), h_arr.max()
            h_range = h_max - h_min if h_max > h_min else 1.0
            norm_h = (h_arr - h_min) / h_range

            x_vals = np.linspace(self.X_SPARK, self.X_SPARK + self.W_SPARK, len(hist_clean))
            y_vals = sp_y_bottom + (norm_h * sp_h)
            line_color = self.GREEN if hist_clean[-1] >= hist_clean[0] else self.RED

            poly_points = [(x_vals[0], sp_y_bottom)]
            poly_points.extend(zip(x_vals, y_vals))
            poly_points.append((x_vals[-1], sp_y_bottom))

            poly = Polygon(poly_points, closed=True, facecolor=line_color, edgecolor=None, alpha=0.15)
            self.ax.add_patch(poly)
            self.ax.add_line(Line2D(x_vals, y_vals, color=line_color, lw=1.5))

            # --- Plot liquidation events on sparkline if available ---
            liq_events = item.get('liq_events') or []
            if liq_events:
                try:
                    hist_ts = item.get('history_ts')  # optional list of ms timestamps
                    if hist_ts and len(hist_ts) >= len(hist_clean):
                        ts_arr = np.array(hist_ts[-len(hist_clean):], dtype=np.int64)
                        for ev in liq_events:
                            ev_ts = int(ev.get('ts_ms', 0))
                            idx = int(np.argmin(np.abs(ts_arr - ev_ts)))
                            if 0 <= idx < len(x_vals):
                                px = x_vals[idx]
                                py = y_vals[idx]
                                color = self.RED if ev.get('side', '').upper() == 'SELL' else self.GREEN
                                marker = 'v' if color == self.RED else '^'
                                self.ax.scatter([px], [py], color=color, s=24, marker=marker, zorder=5, edgecolors='black', linewidth=0.6)
                    else:
                        # Proportional mapping if no timestamps
                        # Use event order relative to history window
                        for ev in liq_events:
                            ev_ts = ev.get('ts_ms', None)
                            if ev_ts is None:
                                continue
                            # approximate mapping: place by recency relative to now if history_ts not available
                            # fallback: evenly distribute by order
                            idx = min(len(x_vals) - 1, max(0, int((len(x_vals) - 1) * 0.5)))
                            px = x_vals[idx]
                            py = y_vals[idx] if len(y_vals) else sp_y_bottom
                            color = self.RED if ev.get('side', '').upper() == 'SELL' else self.GREEN
                            marker = 'v' if color == self.RED else '^'
                            self.ax.scatter([px], [py], color=color, s=24, marker=marker, zorder=5, edgecolors='black', linewidth=0.6)
                except Exception:
                    logger.debug("Failed to plot liquidation markers", exc_info=False)

        # -------------------------
        # Depth mountain histogram (centered in RANGE column)
        # -------------------------
        range_center_x = self.X_RANGE + (self.W_RANGE / 2)
        range_y_center = y
        depth_bins = item.get('depth_bins')
        # Accept also depth_info -> bins
        if not depth_bins and isinstance(item.get('depth_info'), dict):
            depth_bins = item['depth_info'].get('bins') or {
                'bid_bins': item['depth_info'].get('bid_bins'),
                'ask_bins': item['depth_info'].get('ask_bins')
            }

        if depth_bins and (depth_bins.get('bid_bins') or depth_bins.get('ask_bins')):
            try:
                bid_arr = np.asarray(depth_bins.get('bid_bins') or [], dtype=np.float32)
                ask_arr = np.asarray(depth_bins.get('ask_bins') or [], dtype=np.float32)
            except Exception:
                bid_arr = np.zeros(20, dtype=np.float32)
                ask_arr = np.zeros(20, dtype=np.float32)

            if bid_arr.size == 0:
                bid_arr = np.zeros(20, dtype=np.float32)
            if ask_arr.size == 0:
                ask_arr = np.zeros(20, dtype=np.float32)

            left = bid_arr[::-1]
            right = ask_arr

            max_notional = max(left.max() if left.size else 0.0, right.max() if right.size else 0.0, 1.0)
            half_w = self.W_RANGE / 2.0
            cx = range_center_x

            n_left = len(left)
            n_right = len(right)
            left_xs = [cx - (half_w * (i + 1) / max(n_left, 1)) for i in range(n_left)]
            right_xs = [cx + (half_w * (i + 1) / max(n_right, 1)) for i in range(n_right)]

            max_height = sp_h * 0.9
            left_heights = (left / max_notional) * max_height
            right_heights = (right / max_notional) * max_height

            left_poly = [(cx, range_y_center - (max_height / 2))]
            for xi, hi in zip(left_xs, left_heights):
                left_poly.append((xi, (range_y_center - (max_height / 2)) + hi))
            left_poly.append((cx, range_y_center - (max_height / 2)))
            if len(left_poly) >= 3:
                self.ax.add_patch(Polygon(left_poly, closed=True, facecolor=self.RED, alpha=0.18, edgecolor=None))

            right_poly = [(cx, range_y_center - (max_height / 2))]
            for xi, hi in zip(right_xs, right_heights):
                right_poly.append((xi, (range_y_center - (max_height / 2)) + hi))
            right_poly.append((cx, range_y_center - (max_height / 2)))
            if len(right_poly) >= 3:
                self.ax.add_patch(Polygon(right_poly, closed=True, facecolor=self.GREEN, alpha=0.18, edgecolor=None))

            self.ax.scatter([cx], [range_y_center], color=self.MID_DOT, s=18, zorder=4, edgecolors='black', linewidth=0.5)
            mid = _safe_float(item.get('depth_mid') or item.get('depth_best_bid') or item.get('price'), None)
            if mid:
                t(cx, range_y_center - (max_height / 2) - (0.01 / total_h), fmt_price_extended(mid), color=self.SUB_TEXT_COLOR, fontsize=8, ha='center', va='top')

            # annotate top-of-mountain notional values
            if left.size:
                top_left_val = left.max()
                if top_left_val > 0:
                    t(cx - half_w, range_y_center + (max_height / 2) + (0.01 / total_h), fmt_large_num(top_left_val), color=self.SUB_TEXT_COLOR, fontsize=7, ha='left', va='bottom')
            if right.size:
                top_right_val = right.max()
                if top_right_val > 0:
                    t(cx + half_w, range_y_center + (max_height / 2) + (0.01 / total_h), fmt_large_num(top_right_val), color=self.SUB_TEXT_COLOR, fontsize=7, ha='right', va='bottom')
        else:
            # fallback: simple range dot or annotate totals
            self.ax.add_line(Line2D([self.X_RANGE, self.X_RANGE + self.W_RANGE], [range_y_center, range_y_center], color=self.BAR_GREY, lw=2, zorder=1))
            bid_not = _safe_float(item.get('depth_bid_notional'), 0)
            ask_not = _safe_float(item.get('depth_ask_notional'), 0)
            if bid_not or ask_not:
                t(self.X_RANGE - 0.005, y, fmt_large_num(bid_not), color=self.SUB_TEXT_COLOR, fontsize=8, ha='right', va='center')
                t(self.X_RANGE + self.W_RANGE + 0.005, y, fmt_large_num(ask_not), color=self.SUB_TEXT_COLOR, fontsize=8, ha='left', va='center')


# Singleton generator
_generator = MarketImageGenerator()


def generate_market_image(data_list: List[Dict]) -> io.BytesIO:
    return _generator.render(data_list)
