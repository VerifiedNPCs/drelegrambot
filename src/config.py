import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class Config:
    """Centralized application configuration"""

    # ==========================================
    # Bot Configuration
    # ==========================================
    BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")

    # ==========================================
    # Database Configuration
    # ==========================================
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    DB_MIN_POOL_SIZE: int = int(os.getenv("DB_MIN_POOL_SIZE", "2"))
    DB_MAX_POOL_SIZE: int = int(os.getenv("DB_MAX_POOL_SIZE", "10"))
    DB_COMMAND_TIMEOUT: int = int(os.getenv("DB_COMMAND_TIMEOUT", "60"))

    # ==========================================
    # Server Configuration
    # ==========================================
    PORT: int = int(os.getenv("PORT", "8080"))
    HOST: str = os.getenv("HOST", "localhost")

    # ==========================================
    # Backup Configuration
    # ==========================================
    BACKUP_DIR: str = os.getenv("BACKUP_DIR", "./backups")
    AUTO_BACKUP_ENABLED: bool = os.getenv("AUTO_BACKUP_ENABLED", "false").lower() == "true"
    AUTO_BACKUP_INTERVAL_HOURS: int = int(os.getenv("AUTO_BACKUP_INTERVAL_HOURS", "24"))

    # ==========================================
    # Logging Configuration
    # ==========================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")
    LOG_FILE: str = os.path.join(LOG_DIR, "bot.log")
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", "10485760"))  # 10MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # ==========================================
    # Admin Configuration
    # ==========================================
    ADMIN_USER_IDS: list = [
        int(uid.strip()) 
        for uid in os.getenv("ADMIN_USER_IDS", "").split(",") 
        if uid.strip().isdigit()
    ]

    # ==========================================
    # Subscription Plans
    # ==========================================
    SUBSCRIPTION_PLANS: Dict[str, Dict[str, Any]] = {
        "standard": {
            "name": "Standard Plan",
            "base_price": 9.99,
            "cost_per_coin": 0.50,
            "max_coins": 50,
            "duration_days": 30,
            "features": ["Real-Time Market Watch", "Pump/Dump Detection", "Market Rankings"],
            "emoji": "📦"
        },
        "pro": {
            "name": "Pro Plan",
            "base_price": 29.99,
            "cost_per_coin": 1.50,
            "max_coins": 100,
            "features": ["Technical Validation (RSI/MACD)", "Price Snapshots", "Liquidation Data"],
            "emoji": "⭐"
        },
        "business+": {
            "name": "Business+ Plan",
            "base_price": 79.99,
            "cost_per_coin": 2.00,
            "max_coins": 200,
            "duration_days": 30,
            "features": ["Order Signals", "Order Tracking", "High-Frequency Scanning"],
            "emoji": "💼"
        },
        "enterprise+": {
            "name": "Enterprise+ Plan",
            "base_price": 499.00, 
            "cost_per_coin": 0.00,
            "max_coins": 9999,
            "duration_days": 30,
            "features": ["White-Labeling", "Dedicated Server", "API Access"],
            "emoji": "🏢"
        }
    }
    
    # ==========================================
    # Payment URL
    # ==========================================
    TOKEN_SECRET_SALT = os.getenv("TOKEN_SECRET_SALT", "change-this-in-production")
    TOKEN_EXPIRY_HOURS = int(os.getenv("TOKEN_EXPIRY_HOURS", "24"))
    PAYMENT_TOKEN_EXPIRY_MINUTES = int(os.getenv("PAYMENT_TOKEN_EXPIRY_MINUTES", "15"))

    # ==========================================
    # Dashboard URL
    # ==========================================
    DASHBOARD_URL = os.getenv("DASHBOARD_URL", "http://localhost:5173/dashboard")
    PAYMENT_URL = os.getenv("PAYMENT_URL", "http://localhost:5173/payment")
    FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
    
    # ==========================================
    # API Configuration
    # ==========================================
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "3141")))
    
    # ==========================================
    # Admin Wallets (Set your actual wallet addresses here)
    # ==========================================
    ADMIN_WALLETS = {
        "btc": "", 
        "eth": "", 
        "usdt": "0x1e6beaa0a4671620ccfe025d4087054b99f3ae77",       # BSC 
        "ltc": "ltc1qee5k9v4hnyjcznhgmn2vspjukkwkejedgvynnr" # litecoin, REPLACE
    }
    
    # ==========================================
    # Crypto Payment Configuration
    # ==========================================
    # DEV MODE (Set to False in production)
    DEV_MODE = os.getenv("DEV_MODE", "True").lower() == "true"
    TEST_TX_HASH = "sepigoli314fiif9482933#@@8498fufhnvnBUFUEI#*8e98fhIUHFOh8ife83o2jf38f9j29f8ijsjffsajslkcvi32903#"
    
    # API Keys (Get free keys from blockchain explorers)
    ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
    
    
    # ==========================================
    # PAYMENT VERIFICATION SETTINGS
    # ==========================================
    PAYMENT_TOLERANCE = {
        "btc": {
            "percentage": 0.02,  # Allow 2% deviation (price fluctuation + rounding)
            "min_shortfall_usd": 5.0  # Accept if within $5 USD equivalent
        },
        "eth": {
            "percentage": 0.02,
            "min_shortfall_usd": 5.0
        },
        "usdt": {
            "percentage": 0.01,  # 1% for stablecoins (less volatility)
            "min_shortfall_usd": 2.0  # USDT fees are usually $1-2
        },
        "ltc": {
            "percentage": 0.02,
            "min_shortfall_usd": 3.0
        }
    }
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration values"""
        errors = []

        if not cls.BOT_TOKEN or cls.BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
            errors.append("BOT_TOKEN must be set in environment variables")

        if not cls.DATABASE_URL:
            errors.append("DATABASE_URL must be set in environment variables")

        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {err}" for err in errors))

        # Create necessary directories
        os.makedirs(cls.BACKUP_DIR, exist_ok=True)
        os.makedirs(cls.LOG_DIR, exist_ok=True)

        return True

    @classmethod
    def get_plan(cls, plan_key: str) -> Optional[Dict[str, Any]]:
        """Get subscription plan details by key"""
        return cls.SUBSCRIPTION_PLANS.get(plan_key)

    @classmethod
    def get_all_plan_keys(cls) -> list:
        """Get all available plan keys"""
        return list(cls.SUBSCRIPTION_PLANS.keys())

    @classmethod
    def is_admin(cls, user_id: int) -> bool:
        """Check if user is admin"""
        return user_id in cls.ADMIN_USER_IDS

    @classmethod
    def calculate_price(cls, plan_key: str, coin_count: int) -> float:
        """Calculate total price based on plan and coin count"""
        plan = cls.SUBSCRIPTION_PLANS.get(plan_key)
        if not plan:
            return 0.0
        
        # Enterprise logic is custom, return base for now
        if plan_key == "enterprise":
            return plan["base_price"]
            
        total = plan["base_price"] + (plan["cost_per_coin"] * coin_count)
        return round(total, 2)