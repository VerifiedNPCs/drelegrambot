import asyncio
import html
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import ContextTypes, CallbackQueryHandler
from telegram.error import RetryAfter, TimedOut, BadRequest, NetworkError

from config import config, setup_logging
from database import SUBS, USER_SETTINGS
import services
import image_generator
logger = setup_logging("BotHandlers")
# --- Utilities ---
def escape_html(text: str) -> str:
    return html.escape(str(text))

def _normalize_symbol(sym: str) -> str:
    s = (sym or "").upper().strip()
    if not s:
        return s
    if not s.endswith("USDT"):
        s = s + "USDT"
    return s

async def safe_answer(query) -> None:
    """Safely answer a callback query, ignoring network errors/timeouts."""
    try:
        await query.answer()
    except (TimedOut, NetworkError):
        # Network lag; user probably won't notice.
        pass
    except BadRequest:
        # Query likely too old or already answered.
        pass
    except Exception as e:
        logger.warning(f"⚠ Callback answer failed: {e}")

async def safe_send_message(context, chat_id: int, text: str, **kwargs) -> bool:
    """Robust message sender that handles Retries and Timeouts"""
    pm = kwargs.pop('parse_mode', None)
    try:
        for attempt in range(config.MAX_RETRIES):
            try:
                await context.bot.send_message(chat_id=chat_id, text=text, parse_mode=pm, **kwargs)
                return True
            except (RetryAfter, TimedOut) as e:
                sleep_time = getattr(e, 'retry_after', 2.0)
                await asyncio.sleep(sleep_time)
            except Exception:
                if attempt == config.MAX_RETRIES - 1:
                    logger.warning(f"⚠ Failed to send message to {chat_id} after retries")
                await asyncio.sleep(config.RETRY_DELAY)
    except Exception:
        logger.error("Critical error in safe_send_message", exc_info=False)
    return False

def _build_tf_keyboard(prefix: str, user_tf: str = None) -> InlineKeyboardMarkup:
    buttons = []
    row = []
    for tf in config.TIMEFRAME_CHOICES:
        label = tf
        if user_tf and tf == user_tf:
            label = f"⭐ {tf}"
        if tf == 'ALL' and prefix != 'alert':
            continue
        row.append(InlineKeyboardButton(label, callback_data=f"{prefix}:{tf}"))
        if len(row) == 4:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    return InlineKeyboardMarkup(buttons)

async def _send_leaderboard_image(context, chat_id, tf, img_data, caption):
    try:
        if not img_data:
            await safe_send_message(context, chat_id, "No data available.")
            return

        await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
        loop = asyncio.get_running_loop()
        
        # Enrich Data (Async)
        await image_generator.enrich_data(img_data, services.COLLECTOR, config.DEPTH_TARGET_USDT)

        # Generate Image (Thread Pool)
        photo_buf = await loop.run_in_executor(
            services.IMG_EXECUTOR,
            image_generator.generate_market_image,
            img_data
        )

        if photo_buf:
            try:
                photo_buf.seek(0)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_buf,
                    caption=caption,
                    parse_mode=ParseMode.MARKDOWN
                )
            finally:
                photo_buf.close()
        else:
            await safe_send_message(context, chat_id, "⚠ Error generating image.")
    except Exception:
        logger.error("Image generation failed", exc_info=True)
        await safe_send_message(context, chat_id, "⚠ Failed to generate image.")

async def _send_leaderboard(update, context, tf: str, sort_key: str, title: str):
    raw_data = await services.get_leaderboard(tf, sort_key, limit=10)

    if not raw_data:
        await safe_send_message(context, update.effective_chat.id, "No data available yet.")
        return

    img_data = []
    for row in raw_data:
        sym = row['symbol']
        symbol_data = await services.get_symbol_data(tf, sym, limit=20)

        img_data.append({
            'symbol': sym,
            'symbol_raw': sym,
            'depth_symbol': sym,
            'tf': tf,
            'price': row['close'],
            'change': row['change'],
            'history': symbol_data.get('close', []),
            'rsi': row.get('rsi', 50),
            'mfi': row.get('mfi', 50),
            'adx': row.get('adx', 0),
            'usdt_volume': row.get('usdt_volume', 0)
        })

    caption = f"{title} ({tf})"
    await _send_leaderboard_image(context, update.effective_chat.id, tf, img_data, caption)

# --- Command Handlers ---

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    keyboard = [
        [InlineKeyboardButton("🖥 Status", callback_data="status"), InlineKeyboardButton("🚨 All Alerts", callback_data="alerts_all")],
        [InlineKeyboardButton("🚨 Recent (TF)", callback_data="alerts_tf_cmd")],
        [InlineKeyboardButton("📈 Top Gainers", callback_data="top_tf"), InlineKeyboardButton("📉 Top Losers", callback_data="bottom_tf")],
        [InlineKeyboardButton("🔊 High Volume", callback_data="vol_tf"), InlineKeyboardButton("ℹ️ Symbol Info", callback_data="symbol_help")],
        [InlineKeyboardButton("✅ Subscribe", callback_data="sub"), InlineKeyboardButton("❌ Unsubscribe", callback_data="unsub")],
        [InlineKeyboardButton("⚙️ Set Default TF", callback_data="settf_help")]
    ]
    msg = "🚀 **Crypto Market Bot (Visual)**\n\n• 📈 Top Gainers / Losers\n• 🚨 Pump / Dump Alerts\n• 📊 Real-time Data\n\n**Select a command:**"
    await safe_send_message(context, update.effective_chat.id, msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not services.COLLECTOR:
        await safe_send_message(context, update.effective_chat.id, "System Initializing...")
        return
    n_symbols = services.COLLECTOR.top_coins
    n_timeframes = len(services.COLLECTOR.matrices)
    hist_len = len(getattr(services.ENGINE, 'alerts_history', [])) if services.ENGINE else 0
    liq_stats = "Active" if hasattr(services.COLLECTOR, 'liq_monitor') and services.COLLECTOR.liq_monitor.running else "Inactive"

    msg = (
        f"🖥 **System Status**\n"
        f"------------------\n"
        f"• Symbols Monitored: {len(n_symbols)}\n"
        f"• Active Timeframes: {n_timeframes}\n"
        f"• Engine: Vectorized Matrix\n"
        f"• Alerts in memory: {hist_len}\n"
        f"• Liquidation Monitor: {liq_stats}\n"
        f"• Depth target: {int(config.DEPTH_TARGET_USDT):,} USDT\n"
    )
    await safe_send_message(context, update.effective_chat.id, msg, parse_mode=ParseMode.MARKDOWN)

async def top_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tf = getattr(context, 'args', [])[0] if getattr(context, 'args', []) else '1h'
    await _send_leaderboard(update, context, tf, 'change_desc', "📈 Top Gainers")

async def bottom_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tf = getattr(context, 'args', [])[0] if getattr(context, 'args', []) else '1h'
    await _send_leaderboard(update, context, tf, 'change_asc', "📉 Top Losers")

async def vol_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    tf = getattr(context, 'args', [])[0] if getattr(context, 'args', []) else '1h'
    await _send_leaderboard(update, context, tf, 'volume', "🔊 High Volume")

async def alerts_all_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not services.ENGINE:
        return

    async with services.ENGINE.lock:
        alerts = services.ENGINE.alerts_history[-15:]
        alerts.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)

    if not alerts:
        await safe_send_message(context, update.effective_chat.id, "No recent alerts generated.")
        return

    img_data = []
    for a in alerts:
        symbol_data = await services.get_symbol_data(a.timeframe, a.symbol, limit=20)
        
        mfi_val = 50
        adx_val = 0
        usdt_vol = 0
        try:
            if services.COLLECTOR and a.timeframe in services.COLLECTOR.matrices:
                analysis = services.COLLECTOR.matrices[a.timeframe].get_analysis()
                if a.symbol in analysis:
                    d = analysis[a.symbol]
                    mfi_val = d.get('mfi', 50)
                    adx_val = d.get('adx', 0)
                    usdt_vol = d.get('price', 0) * d.get('volume', 0)
        except Exception:
            pass

        img_data.append({
            'symbol': a.symbol,
            'symbol_raw': a.symbol,
            'depth_symbol': a.symbol,
            'tf': a.timeframe,
            'price': getattr(a, 'price', 0),
            'change': getattr(a, 'change_pct', 0),
            'history': symbol_data.get('close', []),
            'rsi': getattr(a, 'rsi', 50),
            'mfi': mfi_val,
            'adx': adx_val,
            'usdt_volume': usdt_vol
        })

    await _send_leaderboard_image(context, update.effective_chat.id, 'ALL', img_data, "🚨 **Recent Alerts (Sorted)**")

async def alerts_by_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not services.ENGINE: return
    tf = getattr(context, 'args', [])[0] if getattr(context, 'args', []) else '1h'
    
    async with services.ENGINE.lock:
        tf_alerts = [a for a in services.ENGINE.alerts_history if a.timeframe == tf]
        tf_alerts = tf_alerts[-15:]
        tf_alerts.sort(key=lambda x: getattr(x, 'change_pct', 0), reverse=True)

    if not tf_alerts:
        await safe_send_message(context, update.effective_chat.id, f"No alerts for **{tf}** timeframe yet.", parse_mode=ParseMode.MARKDOWN)
        return

    analysis = {}
    try:
        if services.COLLECTOR and tf in services.COLLECTOR.matrices:
            analysis = services.COLLECTOR.matrices[tf].get_analysis()
    except Exception:
        pass

    img_data = []
    for a in tf_alerts:
        symbol_data = await services.get_symbol_data(a.timeframe, a.symbol, limit=20)
        d = analysis.get(a.symbol, {})

        img_data.append({
            'symbol': a.symbol,
            'symbol_raw': a.symbol,
            'depth_symbol': a.symbol,
            'tf': a.timeframe,
            'price': getattr(a, 'price', 0),
            'change': getattr(a, 'change_pct', 0),
            'history': symbol_data.get('close', []),
            'rsi': getattr(a, 'rsi', 50),
            'mfi': d.get('mfi', 50),
            'adx': d.get('adx', 0),
            'usdt_volume': d.get('price', 0) * d.get('volume', 0)
        })

    await _send_leaderboard_image(context, update.effective_chat.id, tf, img_data, f"🚨 **Recent Alerts ({tf})**")

async def symbol_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await safe_send_message(context, update.effective_chat.id, "ℹ️ **Usage:** `/symbol <COIN>`", parse_mode=ParseMode.MARKDOWN)
        return

    requested_symbols = context.args[:5]
    img_data = []

    for raw_sym in requested_symbols:
        sym = _normalize_symbol(raw_sym)
        tfs_to_scan = [tf for tf in config.TIMEFRAME_CHOICES if tf != 'ALL']

        for tf in tfs_to_scan:
            symbol_data = await services.get_symbol_data(tf, sym, limit=20)
            if not symbol_data.get('close'):
                continue

            closes = symbol_data.get('close', [])
            latest_close = symbol_data.get('latest_close', 0.0)
            change_val = 0.0
            
            if len(closes) >= 2:
                prev_close = closes[-2]
                if prev_close > 0:
                    change_val = ((latest_close - prev_close) / prev_close) * 100.0

            rsi_val = 50.0
            mfi_val = 50.0
            adx_val = 0.0
            usdt_vol = 0.0

            try:
                if services.COLLECTOR and tf in services.COLLECTOR.matrices:
                    matrix_analysis = services.COLLECTOR.matrices[tf].get_analysis()
                    if sym in matrix_analysis:
                        d = matrix_analysis[sym]
                        rsi_val = d.get('rsi', 50.0)
                        mfi_val = d.get('mfi', 50.0)
                        adx_val = d.get('adx', 0.0)
                        usdt_vol = d.get('price', 0.0) * d.get('volume', 0.0)
            except Exception:
                pass

            img_data.append({
                'symbol': f"{sym} ({tf})",
                'symbol_raw': sym,
                'depth_symbol': sym,
                'tf': tf,
                'price': latest_close,
                'change': change_val,
                'history': closes,
                'rsi': rsi_val,
                'mfi': mfi_val,
                'adx': adx_val,
                'usdt_volume': usdt_vol
            })

    if not img_data:
        await safe_send_message(
            context,
            update.effective_chat.id,
            "❌ Symbols not found or no data.\n(Note: Only Top 100 volume coins are monitored)"
        )
        return

    await _send_leaderboard_image(context, update.effective_chat.id, 'MULTI', img_data, f"📊 **Symbol Analysis**")

# --- Interactive Menus ---

async def set_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("1m", callback_data="settf:1m"), InlineKeyboardButton("5m", callback_data="settf:5m"),
         InlineKeyboardButton("15m", callback_data="settf:15m"), InlineKeyboardButton("1h", callback_data="settf:1h")],
        [InlineKeyboardButton("2h", callback_data="settf:2h"), InlineKeyboardButton("4h", callback_data="settf:4h"),
         InlineKeyboardButton("1d", callback_data="settf:1d"), InlineKeyboardButton("3d", callback_data="settf:3d")],
        [InlineKeyboardButton("1w", callback_data="settf:1w"), InlineKeyboardButton("1M", callback_data="settf:1M")]
    ]
    msg = "⚙️ **Set your default timeframe:**\n\n⭐ Your current default will be marked.\nALL in other menus will use this setting."
    await safe_send_message(context, update.effective_chat.id, msg, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)

async def alerts_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    user_tf = await USER_SETTINGS.get_default_tf(user_id) if user_id else None
    await safe_send_message(context, update.effective_chat.id, "🚨 Choose timeframe for **Recent Alerts**:", reply_markup=_build_tf_keyboard("alert", user_tf), parse_mode=ParseMode.MARKDOWN)

async def top_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    user_tf = await USER_SETTINGS.get_default_tf(user_id) if user_id else None
    await safe_send_message(context, update.effective_chat.id, "📈 Choose timeframe for **Top Gainers**:", reply_markup=_build_tf_keyboard("top", user_tf), parse_mode=ParseMode.MARKDOWN)

async def bottom_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    user_tf = await USER_SETTINGS.get_default_tf(user_id) if user_id else None
    await safe_send_message(context, update.effective_chat.id, "📉 Choose timeframe for **Top Losers**:", reply_markup=_build_tf_keyboard("bottom", user_tf), parse_mode=ParseMode.MARKDOWN)

async def vol_tf_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id if update.effective_user else None
    user_tf = await USER_SETTINGS.get_default_tf(user_id) if user_id else None
    await safe_send_message(context, update.effective_chat.id, "🔊 Choose timeframe for **High Volume**:", reply_markup=_build_tf_keyboard("vol", user_tf), parse_mode=ParseMode.MARKDOWN)

# --- Callback Handlers ---

async def timeframe_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await safe_answer(query)  # USE SAFE ANSWER

    data = query.data
    try:
        cmd, tf = data.split(":", 1)
    except ValueError:
        return

    context.args = [tf]

    if cmd == "top":
        await top_cmd(update, context)
    elif cmd == "bottom":
        await bottom_cmd(update, context)
    elif cmd == "vol":
        await vol_cmd(update, context)
    elif cmd == "alert":
        await alerts_by_tf_cmd(update, context)

async def settf_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await safe_answer(query)  # USE SAFE ANSWER

    if not update.effective_user:
        return

    data = query.data
    try:
        _, tf = data.split(":", 1)
    except ValueError:
        return

    try:
        await USER_SETTINGS.set_default_tf(update.effective_user.id, tf)
        current_tf = await USER_SETTINGS.get_default_tf(update.effective_user.id)
        await safe_send_message(
            context, update.effective_chat.id,
            f"✅ **Default timeframe set to {current_tf}!**",
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception:
        await safe_send_message(context, update.effective_chat.id, "⚠ Failed to set default timeframe.")

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await safe_answer(query)  # USE SAFE ANSWER

    data = query.data

    if data == "status":
        await status_cmd(update, context)
    elif data == "alerts_all":
        await alerts_all_cmd(update, context)
    elif data == "alerts_tf_cmd":
        await alerts_tf_cmd(update, context)
    elif data == "top_tf":
        await top_tf_cmd(update, context)
    elif data == "bottom_tf":
        await bottom_tf_cmd(update, context)
    elif data == "vol_tf":
        await vol_tf_cmd(update, context)
    elif data == "settf_help":
        await set_tf_cmd(update, context)
    elif data == "symbol_help":
        await safe_send_message(context, update.effective_chat.id, "ℹ️ **Usage:** `/symbol <COIN>`", parse_mode=ParseMode.MARKDOWN)
    elif data == "sub":
        try:
            await SUBS.add(update.effective_chat.id)
            await safe_send_message(context, update.effective_chat.id, "✅ Subscribed to alerts.")
        except Exception:
            await safe_send_message(context, update.effective_chat.id, "⚠ Failed to subscribe.")
    elif data == "unsub":
        try:
            await SUBS.remove(update.effective_chat.id)
            await safe_send_message(context, update.effective_chat.id, "❌ Unsubscribed.")
        except Exception:
            await safe_send_message(context, update.effective_chat.id, "⚠ Failed to unsubscribe.")
