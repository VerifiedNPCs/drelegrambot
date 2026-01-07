import asyncio
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

from config import Config
from database import DatabaseManager
from handlers import (
    start_command,
    help_command,
    account_command,
    plans_command,
    status_command,
    stats_command,
    backup_command,
    admin_command,
    callback_handler,
    handle_email,
    error_handler,
    set_db_manager,
)


# ==========================================
# Logging Configuration
# ==========================================

def setup_logging():
    """Configure logging with file rotation"""
    import os
    from pathlib import Path
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler (always enabled)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # File handler ONLY for local development (not on Railway/production)
    if os.getenv('RAILWAY_ENVIRONMENT') is None:  # Not on Railway
        try:
            # Create log directory if it doesn't exist
            log_dir = Path(Config.LOG_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler with rotation
            file_handler = RotatingFileHandler(
                Config.LOG_FILE,
                maxBytes=Config.LOG_MAX_BYTES,
                backupCount=Config.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging: {e}")
    
    # Reduce noise from libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)



# Get logger for this module
logger = logging.getLogger(__name__)


# ==========================================
# Database Manager Instance
# ==========================================

db_manager = DatabaseManager()


# ==========================================
# Background Jobs
# ==========================================

async def expire_subscriptions_job(context):
    """Background job to expire old subscriptions"""
    try:
        count = await db_manager.expire_subscriptions()
        if count > 0:
            logger.info(f"Background job: Expired {count} subscriptions")
    except Exception:
        logger.exception("Error in expire_subscriptions_job")


async def auto_backup_job(context):
    """Background job for automatic backups"""
    if not Config.AUTO_BACKUP_ENABLED:
        return

    try:
        backup_path = await db_manager.backup_to_json()
        if backup_path:
            logger.info(f"Auto backup created: {backup_path}")
    except Exception:
        logger.exception("Error in auto_backup_job")


# ==========================================
# Application Lifecycle
# ==========================================

async def post_init(application: Application):
    """Initialize resources on startup"""
    logger.info("="*60)
    logger.info("Starting Subscription Management Bot...")
    logger.info("="*60)

    # Validate configuration
    try:
        Config.validate()
        logger.info("✓ Configuration validated")
    except ValueError as e:
        logger.error(f"✗ Configuration error: {e}")
        raise

    # Initialize database
    logger.info("Initializing database...")
    success = await db_manager.initialize()
    if not success:
        logger.error("✗ Failed to initialize database")
        raise RuntimeError("Failed to initialize database")
    logger.info("✓ Database initialized successfully")
    
    commands = [
        BotCommand("start", "🏠 Start the bot and show main menu"),
        BotCommand("account", "👤 View your account information"),
        BotCommand("plans", "💳 Browse subscription plans"),
        BotCommand("status", "📊 Check subscription status"),
        BotCommand("help", "❓ Get help and support"),
    ]
    
    # Add admin commands if configured
    if Config.ADMIN_USER_IDS:
        commands.extend([
            BotCommand("admin", "🔐 Admin panel (admin only)"),
            BotCommand("stats", "📊 Bot statistics (admin only)"),
            BotCommand("backup", "💾 Create backup (admin only)"),
        ])
    
    await application.bot.set_my_commands(commands)
    logger.info("✓ Bot commands menu configured")

    # Inject database manager into handlers
    set_db_manager(db_manager)
    logger.info("✓ Database manager injected into handlers")

    # Schedule background jobs
    job_queue = application.job_queue

    # Expire subscriptions every hour
    job_queue.run_repeating(
        expire_subscriptions_job,
        interval=3600,  # 1 hour
        first=10  # First run after 10 seconds
    )
    logger.info("✓ Subscription expiration job scheduled (every hour)")

    # Auto backup if enabled
    if Config.AUTO_BACKUP_ENABLED:
        job_queue.run_repeating(
            auto_backup_job,
            interval=Config.AUTO_BACKUP_INTERVAL_HOURS * 3600,
            first=60  # First run after 1 minute
        )
        logger.info(f"✓ Auto backup job scheduled (every {Config.AUTO_BACKUP_INTERVAL_HOURS}h)")

    logger.info("="*60)
    logger.info("✓ Bot initialized successfully and ready!")
    logger.info("="*60)


async def post_shutdown(application: Application):
    """Cleanup resources on shutdown"""
    logger.info("="*60)
    logger.info("Shutting down bot...")
    logger.info("="*60)

    # Close database connection
    await db_manager.close()
    logger.info("✓ Database connections closed")

    logger.info("="*60)
    logger.info("✓ Bot shutdown complete")
    logger.info("="*60)


# ==========================================
# Application Builder
# ==========================================

def build_application() -> Application:
    """Build and configure the Telegram application"""
    logger.info("Building application...")

    # Create application
    app = Application.builder().token(Config.BOT_TOKEN).build()

    # Set lifecycle hooks
    app.post_init = post_init
    app.post_shutdown = post_shutdown

    # ==========================================
    # Register Handlers
    # ==========================================

    # Command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("account", account_command))
    app.add_handler(CommandHandler("plans", plans_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("backup", backup_command))
    app.add_handler(CommandHandler("admin", admin_command))

    # Callback query handler
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Message handler for email input
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("✓ All handlers registered")
    return app


# ==========================================
# Main Entry Point
# ==========================================

def main():
    """Main function - entry point of the application"""
    # Setup logging first
    setup_logging()

    try:
        # Build application
        application = build_application()

        # Run bot
        logger.info("Starting polling... Press Ctrl+C to stop")
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )

    except KeyboardInterrupt:
        logger.info("Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.exception(f"Bot crashed with error: {e}")
        raise


if __name__ == "__main__":
    main()
