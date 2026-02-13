import time
import asyncio
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from telegram.error import NetworkError, TimedOut, Forbidden

from config import config, setup_logging
from database import SUBS, USER_SETTINGS
import services
import bot_handlers
logger = setup_logging("Main")
async def post_init(app: Application) -> None:
    logger.info("⚡ Bot starting up... initializing services.")
    try:
        # Initialize Databases
        await SUBS.ensure_table()
        await USER_SETTINGS.ensure_table()
        
        # Initialize Services (Collector, Engine, Health)
        await services.init_services(app)
        
        logger.info("✔ Services Started")
        if config.ADMIN_CHAT_ID:
            try:
                await app.bot.send_message(
                    chat_id=config.ADMIN_CHAT_ID,
                    text="🤖 **Bot Online & Optimized**\nSYSTEM: All systems nominal.",
                    parse_mode=ParseMode.MARKDOWN
                )
            except Exception:
                logger.warning("Could not send startup msg to admin (check ID/Network)")
    except Exception:
        logger.critical("❌ CRITICAL startup failure", exc_info=True)

async def error_handler(update: object, context) -> None:
    """Log the error and be silent, don't crash."""
    # Filter out common network noise
    e = context.error
    if isinstance(e, (TimedOut, NetworkError)):
        logger.warning(f"⚠ Network Warning: {e}")
        return
    if isinstance(e, Forbidden):
        logger.warning(f"⚠ Forbidden Warning: {e}")
        return
    
    # Log other real errors
    logger.error("Exception while handling an update:", exc_info=context.error)

def main():
    print("Starting Optimized Bot...")
    
    if not config.BOT_TOKEN:
        print("ERROR: BOT_TOKEN not found in config.py or environment variables")
        return

    # Build Application
    app = Application.builder().token(config.BOT_TOKEN).post_init(post_init).build()

    # --- Error Handler ---
    app.add_error_handler(error_handler)

    # --- Command Handlers ---
    app.add_handler(CommandHandler("start", bot_handlers.start_cmd))
    app.add_handler(CommandHandler("status", bot_handlers.status_cmd))
    
    # Leaderboards
    app.add_handler(CommandHandler(["top", "gainers"], bot_handlers.top_cmd))
    app.add_handler(CommandHandler(["bottom", "losers"], bot_handlers.bottom_cmd))
    app.add_handler(CommandHandler("vol", bot_handlers.vol_cmd))
    
    # Analysis & Alerts
    app.add_handler(CommandHandler("symbol", bot_handlers.symbol_cmd))
    app.add_handler(CommandHandler("alerts", bot_handlers.alerts_by_tf_cmd))
    
    # Settings & TF Shortcuts
    app.add_handler(CommandHandler("settf", bot_handlers.set_tf_cmd))
    app.add_handler(CommandHandler("alertstf", bot_handlers.alerts_tf_cmd))
    app.add_handler(CommandHandler("toptf", bot_handlers.top_tf_cmd))
    app.add_handler(CommandHandler("bottomtf", bot_handlers.bottom_tf_cmd))
    app.add_handler(CommandHandler("voltf", bot_handlers.vol_tf_cmd))

    # --- Callback Handlers ---
    app.add_handler(CallbackQueryHandler(bot_handlers.timeframe_button_handler, pattern="^(top|bottom|vol|alert):"))
    app.add_handler(CallbackQueryHandler(bot_handlers.settf_button_handler, pattern="^settf:"))
    app.add_handler(CallbackQueryHandler(bot_handlers.button_handler))

    print("✓ Bot configuration complete. Starting polling...")

    # --- ROBUST POLLING LOOP ---
    while True:
        try:
            app.run_polling(
                drop_pending_updates=True, 
                allowed_updates=Update.ALL_TYPES, 
                poll_interval=2.0, 
                timeout=60,
                close_loop=False
            )
            break
        except (NetworkError, TimedOut) as e:
            logger.warning(f"⚠ Connection dip: {e}. Reconnecting in 5s...")
            time.sleep(5)
        except Exception as e:
            # Check for specific HTTPX/Telegram disconnection errors
            if "RemoteProtocolError" in str(e) or "Server disconnected" in str(e):
                logger.warning("⚠ Telegram Server Reset (Normal). Reconnecting...")
                time.sleep(3)
            else:
                logger.critical("❌ Fatal crash in polling loop", exc_info=True)
                time.sleep(10)

if __name__ == "__main__":
    main()
