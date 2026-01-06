from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import asyncpg

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

# -------------------------
# Configuration
# -------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/subscription_bot")
PORT = int(os.getenv("PORT", "8080"))

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("subscription_bot")

# -------------------------
# Database connection pool
# -------------------------
db_pool: Optional[asyncpg.Pool] = None

# Subscription plans with pricing and features
SUBSCRIPTION_PLANS = {
    "standard": {
        "name": "Standard Plan",
        "price": "$9.99/month",
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

# -------------------------
# Database Functions
# -------------------------

async def init_db_pool():
    """Initialize database connection pool"""
    global db_pool
    try:
        db_pool = await asyncpg.create_pool(
            DATABASE_URL,
            min_size=2,
            max_size=10,
            command_timeout=60
        )
        logger.info("Database pool created successfully")

        # Test connection and create tables if needed
        async with db_pool.acquire() as conn:
            # Read and execute schema
            try:
                with open('schema.sql', 'r') as f:
                    schema = f.read()
                    await conn.execute(schema)
                    logger.info("Database schema initialized")
            except FileNotFoundError:
                logger.warning("schema.sql not found, assuming tables exist")
            except Exception as e:
                logger.error(f"Schema initialization error: {e}")

    except Exception as e:
        logger.exception(f"Failed to create database pool: {e}")
        raise

async def close_db_pool():
    """Close database connection pool"""
    global db_pool
    if db_pool:
        await db_pool.close()
        logger.info("Database pool closed")

async def get_user(user_id: int) -> Optional[dict]:
    """Get user from database"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM users WHERE user_id = $1",
                user_id
            )
            return dict(row) if row else None
    except Exception as e:
        logger.exception(f"Error fetching user {user_id}: {e}")
        return None

async def create_user(user_id: int, username: str, email: str) -> Optional[dict]:
    """Create a new user account"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO users (user_id, username, email)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO UPDATE
                SET username = EXCLUDED.username, email = EXCLUDED.email
                RETURNING *
                """,
                user_id, username, email
            )
            logger.info(f"Created/updated user account: {username} ({user_id})")
            return dict(row) if row else None
    except asyncpg.UniqueViolationError:
        logger.warning(f"Email {email} already exists")
        return None
    except Exception as e:
        logger.exception(f"Error creating user: {e}")
        return None

async def get_active_subscription(user_id: int) -> Optional[dict]:
    """Get user's active subscription"""
    try:
        async with db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM subscriptions
                WHERE user_id = $1 AND status = 'active' AND end_date > NOW()
                ORDER BY created_at DESC
                LIMIT 1
                """,
                user_id
            )
            return dict(row) if row else None
    except Exception as e:
        logger.exception(f"Error fetching subscription for user {user_id}: {e}")
        return None

async def create_subscription(user_id: int, plan: str) -> Optional[dict]:
    """Create a new subscription for user"""
    try:
        start_date = datetime.now(timezone.utc)
        end_date = start_date + timedelta(days=30)  # 30-day subscription

        async with db_pool.acquire() as conn:
            # Deactivate any existing active subscriptions
            await conn.execute(
                """
                UPDATE subscriptions
                SET status = 'cancelled'
                WHERE user_id = $1 AND status = 'active'
                """,
                user_id
            )

            # Create new subscription
            row = await conn.fetchrow(
                """
                INSERT INTO subscriptions (user_id, plan, status, start_date, end_date)
                VALUES ($1, $2, 'active', $3, $4)
                RETURNING *
                """,
                user_id, plan, start_date, end_date
            )
            logger.info(f"Created subscription for user {user_id}: {plan}")
            return dict(row) if row else None
    except Exception as e:
        logger.exception(f"Error creating subscription: {e}")
        return None

async def get_days_left(user_id: int) -> Optional[int]:
    """Calculate days left in subscription"""
    try:
        subscription = await get_active_subscription(user_id)
        if not subscription:
            return None

        end_date = subscription['end_date']
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)

        if end_date > now:
            delta = end_date - now
            return delta.days
        return 0
    except Exception as e:
        logger.exception(f"Error calculating days left: {e}")
        return None

async def format_user_status(user_id: int) -> str:
    """Format user status message"""
    try:
        user = await get_user(user_id)
        if not user:
            return "❌ No account found"

        status = f"👤 **Account Information**\n\n"
        status += f"Username: @{user['username']}\n"
        status += f"Email: {user['email']}\n\n"

        subscription = await get_active_subscription(user_id)

        if subscription:
            plan_info = SUBSCRIPTION_PLANS.get(subscription['plan'])
            status += f"📊 **Subscription Status**\n\n"
            status += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
            status += f"Price: {plan_info['price']}\n"

            days_left = await get_days_left(user_id)
            if days_left is not None:
                if days_left > 0:
                    status += f"Days Left: {days_left} days\n"
                else:
                    status += f"Status: ⚠️ Expired\n"
        else:
            status += "📊 **Subscription Status**\n\n"
            status += "Plan: No active subscription\n"

        return status
    except Exception as e:
        logger.exception(f"Error formatting user status: {e}")
        return "❌ Error loading account information"

# -------------------------
# Glass Button UI Keyboards
# -------------------------

def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Main menu with glass button style"""
    keyboard = [
        [
            InlineKeyboardButton("📝 Create Account", callback_data="create_account"),
            InlineKeyboardButton("👤 My Account", callback_data="my_account"),
        ],
        [
            InlineKeyboardButton("💳 Choose Plan", callback_data="choose_plan"),
            InlineKeyboardButton("📊 Plan Details", callback_data="plan_details"),
        ],
        [
            InlineKeyboardButton("⏰ Subscription Status", callback_data="subscription_status"),
            InlineKeyboardButton("ℹ️ Help", callback_data="help"),
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_plans_keyboard() -> InlineKeyboardMarkup:
    """Subscription plans keyboard"""
    keyboard = [
        [InlineKeyboardButton(f"{SUBSCRIPTION_PLANS['standard']['emoji']} Standard - {SUBSCRIPTION_PLANS['standard']['price']}", 
                             callback_data="plan_standard")],
        [InlineKeyboardButton(f"{SUBSCRIPTION_PLANS['pro']['emoji']} Pro - {SUBSCRIPTION_PLANS['pro']['price']}", 
                             callback_data="plan_pro")],
        [InlineKeyboardButton(f"{SUBSCRIPTION_PLANS['business+']['emoji']} Business+ - {SUBSCRIPTION_PLANS['business+']['price']}", 
                             callback_data="plan_business+")],
        [InlineKeyboardButton(f"{SUBSCRIPTION_PLANS['enterprise+']['emoji']} Enterprise+ - {SUBSCRIPTION_PLANS['enterprise+']['price']}", 
                             callback_data="plan_enterprise+")],
        [InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_plan_details_keyboard() -> InlineKeyboardMarkup:
    """Plan details keyboard"""
    keyboard = [
        [InlineKeyboardButton("📦 Standard Details", callback_data="detail_standard")],
        [InlineKeyboardButton("⭐ Pro Details", callback_data="detail_pro")],
        [InlineKeyboardButton("💼 Business+ Details", callback_data="detail_business+")],
        [InlineKeyboardButton("🏢 Enterprise+ Details", callback_data="detail_enterprise+")],
        [InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_back_keyboard() -> InlineKeyboardMarkup:
    """Simple back button"""
    keyboard = [[InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]]
    return InlineKeyboardMarkup(keyboard)

# -------------------------
# Command Handlers
# -------------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Start command - shows main menu"""
    user = update.effective_user

    welcome_text = (
        f"👋 Welcome {user.first_name}!\n\n"
        "🎯 **Subscription Management Bot**\n\n"
        "Manage your subscription plans with ease. "
        "Choose from our flexible plans designed for your needs.\n\n"
        "Select an option below:"
    )

    try:
        if update.message:
            await update.message.reply_text(
                welcome_text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=welcome_text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
    except Exception:
        logger.exception("start_command failed")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = (
        "📚 **Help & Support**\n\n"
        "**Available Commands:**\n"
        "/start - Main menu\n"
        "/account - View your account\n"
        "/plans - View subscription plans\n"
        "/status - Check subscription status\n"
        "/help - Show this help message\n\n"
        "**Need assistance?**\n"
        "Contact: support@example.com"
    )

    try:
        await update.message.reply_text(
            help_text,
            reply_markup=get_back_keyboard(),
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("help_command failed")

async def account_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show account information"""
    user_id = update.effective_user.id
    text = await format_user_status(user_id)

    if "No account found" in text:
        text = (
            "❌ You don't have an account yet.\n\n"
            "Click 'Create Account' to get started!"
        )

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("account_command failed")

async def plans_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available plans"""
    text = "💳 **Choose Your Subscription Plan**\n\nSelect a plan that fits your needs:"

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_plans_keyboard(),
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("plans_command failed")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show subscription status"""
    user_id = update.effective_user.id
    subscription = await get_active_subscription(user_id)

    if not subscription:
        text = "⚠️ You don't have an active subscription.\n\nChoose a plan to get started!"
    else:
        days_left = await get_days_left(user_id)
        plan_info = SUBSCRIPTION_PLANS.get(subscription['plan'])

        text = f"📊 **Subscription Status**\n\n"
        text += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
        text += f"Price: {plan_info['price']}\n\n"

        if days_left and days_left > 0:
            text += f"⏰ Days Remaining: **{days_left} days**\n\n"
            if days_left <= 7:
                text += "⚠️ Your subscription is expiring soon!\n"
                text += "Consider renewing to avoid service interruption."
        else:
            text += "❌ Status: **Expired**\n\n"
            text += "Please renew your subscription to continue using our services."

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("status_command failed")

# -------------------------
# Callback Query Handlers
# -------------------------

async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all callback queries"""
    query = update.callback_query
    if not query:
        return

    data = query.data
    user = update.effective_user
    user_id = user.id

    try:
        await query.answer()
    except Exception:
        pass

    # Main menu
    if data == "main_menu":
        text = (
            f"👋 Welcome back {user.first_name}!\n\n"
            "Select an option below:"
        )
        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )

    # Create account flow
    elif data == "create_account":
        db_user = await get_user(user_id)
        if db_user:
            text = "✅ You already have an account!\n\n" + await format_user_status(user_id)
            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            text = (
                "📝 **Create Your Account**\n\n"
                "Please send your email address to complete registration.\n\n"
                "Format: youremail@example.com"
            )
            context.user_data["awaiting_email"] = True
            await query.edit_message_text(
                text,
                reply_markup=get_back_keyboard(),
                parse_mode="Markdown"
            )

    # My account
    elif data == "my_account":
        db_user = await get_user(user_id)
        if not db_user:
            text = (
                "❌ You don't have an account yet.\n\n"
                "Click 'Create Account' to get started!"
            )
        else:
            text = await format_user_status(user_id)

        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )

    # Choose plan
    elif data == "choose_plan":
        db_user = await get_user(user_id)
        if not db_user:
            text = "❌ Please create an account first!"
            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            text = "💳 **Choose Your Subscription Plan**\n\nSelect a plan that fits your needs:"
            await query.edit_message_text(
                text,
                reply_markup=get_plans_keyboard(),
                parse_mode="Markdown"
            )

    # Plan selection
    elif data.startswith("plan_"):
        plan_key = data.replace("plan_", "")
        db_user = await get_user(user_id)

        if not db_user:
            text = "❌ Please create an account first!"
            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            subscription = await create_subscription(user_id, plan_key)

            if subscription:
                plan_info = SUBSCRIPTION_PLANS[plan_key]

                text = (
                    f"✅ **Subscription Activated!**\n\n"
                    f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
                    f"Price: {plan_info['price']}\n"
                    f"Duration: 30 days\n\n"
                    f"Thank you for subscribing! 🎉"
                )
            else:
                text = "❌ Failed to activate subscription. Please try again."

            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )

    # Plan details menu
    elif data == "plan_details":
        text = "📋 **Subscription Plans Overview**\n\nSelect a plan to see detailed information:"
        await query.edit_message_text(
            text,
            reply_markup=get_plan_details_keyboard(),
            parse_mode="Markdown"
        )

    # Individual plan details
    elif data.startswith("detail_"):
        plan_key = data.replace("detail_", "")
        plan_info = SUBSCRIPTION_PLANS[plan_key]

        text = f"{plan_info['emoji']} **{plan_info['name']}**\n\n"
        text += f"💰 Price: {plan_info['price']}\n\n"
        text += "**Features:**\n"
        for feature in plan_info['features']:
            text += f"{feature}\n"

        keyboard = [
            [InlineKeyboardButton(f"Subscribe to {plan_info['name']}", callback_data=f"plan_{plan_key}")],
            [InlineKeyboardButton("« Back to Plans", callback_data="plan_details")]
        ]

        await query.edit_message_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="Markdown"
        )

    # Subscription status
    elif data == "subscription_status":
        subscription = await get_active_subscription(user_id)

        if not subscription:
            text = "⚠️ You don't have an active subscription.\n\nChoose a plan to get started!"
        else:
            days_left = await get_days_left(user_id)
            plan_info = SUBSCRIPTION_PLANS.get(subscription['plan'])

            text = f"📊 **Subscription Status**\n\n"
            text += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
            text += f"Price: {plan_info['price']}\n\n"

            if days_left and days_left > 0:
                text += f"⏰ Days Remaining: **{days_left} days**\n\n"
                if days_left <= 7:
                    text += "⚠️ Your subscription is expiring soon!"
            else:
                text += "❌ Status: **Expired**\n\n"
                text += "Please renew your subscription."

        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )

    # Help
    elif data == "help":
        help_text = (
            "📚 **Help & Support**\n\n"
            "**How to use this bot:**\n\n"
            "1️⃣ Create an account\n"
            "2️⃣ Choose a subscription plan\n"
            "3️⃣ Monitor your subscription status\n\n"
            "**Need assistance?**\n"
            "Contact: support@example.com"
        )
        await query.edit_message_text(
            help_text,
            reply_markup=get_back_keyboard(),
            parse_mode="Markdown"
        )

# -------------------------
# Message Handlers
# -------------------------

async def handle_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle email input for account creation"""
    if not context.user_data.get("awaiting_email"):
        return

    email = update.message.text.strip()
    user = update.effective_user

    # Basic email validation
    if "@" not in email or "." not in email:
        await update.message.reply_text(
            "❌ Invalid email format. Please try again.\n\n"
            "Format: youremail@example.com",
            parse_mode="Markdown"
        )
        return

    # Create account
    username = user.username or f"user_{user.id}"
    db_user = await create_user(user.id, username, email)

    context.user_data["awaiting_email"] = False

    if db_user:
        text = (
            "✅ **Account Created Successfully!**\n\n"
            f"Username: @{username}\n"
            f"Email: {email}\n\n"
            "You can now choose a subscription plan!"
        )
    else:
        text = (
            "❌ **Account Creation Failed**\n\n"
            "This email might already be in use. Please try a different email."
        )

    await update.message.reply_text(
        text,
        reply_markup=get_main_menu_keyboard(),
        parse_mode="Markdown"
    )

# -------------------------
# Error Handler
# -------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.exception("Unhandled error: %s", getattr(context, "error", None))

    try:
        if isinstance(update, Update) and update.effective_chat:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="❌ An error occurred. Please try again later."
            )
    except Exception:
        logger.exception("Failed to send error notification")

# -------------------------
# Application Builder
# -------------------------

async def post_init(application: Application):
    """Initialize database and other resources"""
    logger.info("Running post_init: initializing database")
    await init_db_pool()

async def post_shutdown(application: Application):
    """Cleanup resources on shutdown"""
    logger.info("Running post_shutdown: closing database")
    await close_db_pool()

def build_application() -> Application:
    """Build and configure the application"""
    if BOT_TOKEN == "YOUR_BOT_TOKEN_HERE":
        raise SystemExit("ERROR: Please set BOT_TOKEN environment variable")

    app = Application.builder().token(BOT_TOKEN).build()

    # Set lifecycle hooks
    app.post_init = post_init
    app.post_shutdown = post_shutdown

    # Command handlers
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("account", account_command))
    app.add_handler(CommandHandler("plans", plans_command))
    app.add_handler(CommandHandler("status", status_command))

    # Callback query handler
    app.add_handler(CallbackQueryHandler(callback_handler))

    # Message handler for email input
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_email))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("Application configured successfully")
    return app

# -------------------------
# Main
# -------------------------

def main():
    """Main function"""
    logger.info("Starting Subscription Management Bot...")

    application = build_application()

    logger.info("Bot is running... Press Ctrl+C to stop")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
