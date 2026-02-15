# config.py
import os
import logging
from dataclasses import dataclass

# --- 1. Centralized Configuration Class ---
class Config:
    # Telegram Credentials
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8016811126:AAFCu9xj3sTFRNWAHYTSxkfIJjAJzQkTmKE")
    ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))

    # Database Paths
    SUBS_FILE = os.getenv("SUBS_FILE", "subscriptions.db")
    USER_SETTINGS_FILE = os.getenv("USER_SETTINGS_FILE", "user_settings.db")

    # Analysis Settings
    ALERT_REFRESH_RATE = 2.0
    PUMP_THRESHOLD_PCT = 2.5
    DUMP_THRESHOLD_PCT = -2.5
    VOLUME_SPIKE_RATIO = 3.0
    DEPTH_TARGET_USDT = float(os.getenv("DEPTH_TARGET_USDT", "250000"))

    # Limits
    MAX_RETRIES = 3
    RETRY_DELAY = 1.0
    TIMEFRAME_CHOICES = ['1m', '5m', '15m', '1h', '2h', '4h', '1d', '3d', '1w', '1M', 'ALL']

    # Logging Settings
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"

# Create a single instance to be imported elsewhere
config = Config()

# --- 2. Centralized Logging Setup ---
def setup_logging(module_name: str = "Root"):
    """
    Call this in every file to get a configured logger.
    It ensures basicConfig is only called once properly.
    """
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=config.LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Suppress noisy libraries
    noisy_libs = ['apscheduler', 'httpx', 'httpcore', 'telegram', 'websockets', 'aiosqlite', 'matplotlib']
    for lib in noisy_libs:
        logging.getLogger(lib).setLevel(logging.WARNING)

    return logging.getLogger(module_name)
