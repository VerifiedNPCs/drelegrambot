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
            "price": "$9.99/month",
            "duration_days": 30,
            "features": [
                "✓ Basic features access",
                "✓ 5 API calls per day",
                "✓ Email support",
                "✓ 1 user account"
            ],
            "emoji": "📦"
        },
        "pro": {
            "name": "Pro Plan",
            "price": "$29.99/month",
            "duration_days": 30,
            "features": [
                "✓ All Standard features",
                "✓ 100 API calls per day",
                "✓ Priority support",
                "✓ 5 user accounts",
                "✓ Advanced analytics"
            ],
            "emoji": "⭐"
        },
        "business+": {
            "name": "Business+ Plan",
            "price": "$79.99/month",
            "duration_days": 30,
            "features": [
                "✓ All Pro features",
                "✓ Unlimited API calls",
                "✓ 24/7 dedicated support",
                "✓ 20 user accounts",
                "✓ Custom integrations",
                "✓ White-label options"
            ],
            "emoji": "💼"
        },
        "enterprise+": {
            "name": "Enterprise+ Plan",
            "price": "Custom pricing",
            "duration_days": 30,
            "features": [
                "✓ All Business+ features",
                "✓ Unlimited user accounts",
                "✓ Dedicated account manager",
                "✓ SLA guarantee",
                "✓ On-premise deployment",
                "✓ Custom development"
            ],
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
    API_HOST = os.getenv("API_HOST", "localhost")
    API_PORT = int(os.getenv("API_PORT", "3141"))


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
