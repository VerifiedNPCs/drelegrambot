"""Bot command and callback handlers"""
import logging
from telegram import Update
from telegram.ext import ContextTypes

from config import Config
from keyboards import *

logger = logging.getLogger(__name__)

# Database manager will be injected
db_manager = None


def set_db_manager(manager):
    """Set the database manager instance"""
    global db_manager
    db_manager = manager


# ==========================================
# Helper Functions
# ==========================================

async def format_user_status(user_id: int) -> str:
    """Format user status message"""
    try:
        user = await db_manager.get_user(user_id)
        if not user:
            return "❌ No account found"

        status = "👤 **Account Information**\n\n"
        status += f"Username: @{user['username']}\n"
        status += f"Email: {user['email']}\n\n"

        subscription = await db_manager.get_active_subscription(user_id)

        if subscription:
            plan_info = Config.get_plan(subscription['plan'])
            status += "📊 **Subscription Status**\n\n"
            status += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
            status += f"Price: {plan_info['price']}\n"

            days_left = await db_manager.get_days_left(user_id)
            if days_left is not None:
                if days_left > 0:
                    status += f"Days Left: {days_left} days\n"
                else:
                    status += "Status: ⚠️ Expired\n"
        else:
            status += "📊 **Subscription Status**\n\n"
            status += "Plan: No active subscription\n"

        return status

    except Exception as e:
        logger.exception(f"Error formatting user status: {e}")
        return "❌ Error loading account information"


# ==========================================
# Command Handlers
# ==========================================

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
    )

    # Add admin commands if user is admin
    if Config.is_admin(update.effective_user.id):
        help_text += (
            "**Admin Commands:**\n"
            "/stats - View bot statistics\n"
            "/backup - Create database backup\n"
            "/admin - Admin panel\n\n"
        )

    help_text += "**Need assistance?**\nContact: support@example.com"

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
    subscription = await db_manager.get_active_subscription(user_id)

    if not subscription:
        text = "⚠️ You don't have an active subscription.\n\nChoose a plan to get started!"
    else:
        days_left = await db_manager.get_days_left(user_id)
        plan_info = Config.get_plan(subscription['plan'])

        text = "📊 **Subscription Status**\n\n"
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


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show bot statistics (admin only)"""
    if not Config.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ This command is for admins only.")
        return

    try:
        stats = await db_manager.get_statistics()

        text = "📊 **Bot Statistics**\n\n"
        text += f"👥 Total Users: {stats.get('total_users', 0)}\n"
        text += f"✅ Active Subscriptions: {stats.get('active_subscriptions', 0)}\n"
        text += f"❌ Expired Subscriptions: {stats.get('expired_subscriptions', 0)}\n"
        text += f"🚫 Cancelled Subscriptions: {stats.get('cancelled_subscriptions', 0)}\n\n"

        if stats.get('subscriptions_by_plan'):
            text += "**Subscriptions by Plan:**\n"
            for plan, count in stats['subscriptions_by_plan'].items():
                plan_info = Config.get_plan(plan)
                if plan_info:
                    text += f"{plan_info['emoji']} {plan_info['name']}: {count}\n"

        await update.message.reply_text(text, parse_mode="Markdown")

    except Exception:
        logger.exception("stats_command failed")


async def backup_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Create database backup (admin only)"""
    if not Config.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ This command is for admins only.")
        return

    try:
        await update.message.reply_text("🔄 Creating backup...")

        backup_path = await db_manager.backup_to_json()

        if backup_path:
            await update.message.reply_text(
                f"✅ Backup created successfully!\n\n"
                f"File: `{backup_path}`",
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text("❌ Backup failed. Check logs.")

    except Exception:
        logger.exception("backup_command failed")


async def admin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Admin panel (admin only)"""
    if not Config.is_admin(update.effective_user.id):
        await update.message.reply_text("❌ This command is for admins only.")
        return

    text = (
        "🔐 **Admin Panel**\n\n"
        "Welcome to the admin control panel. "
        "Use the buttons below to manage the bot."
    )

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="Markdown"
        )
    except Exception:
        logger.exception("admin_command failed")


# ==========================================
# Callback Query Handler
# ==========================================

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
        text = f"👋 Welcome back {user.first_name}!\n\nSelect an option below:"
        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="Markdown"
        )

    # Create account
    elif data == "create_account":
        db_user = await db_manager.get_user(user_id)
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
        db_user = await db_manager.get_user(user_id)
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
        db_user = await db_manager.get_user(user_id)
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
        db_user = await db_manager.get_user(user_id)

        if not db_user:
            text = "❌ Please create an account first!"
            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="Markdown"
            )
        else:
            plan_info = Config.get_plan(plan_key)
            duration = plan_info['duration_days']

            subscription = await db_manager.create_subscription(user_id, plan_key, duration)

            if subscription:
                text = (
                    f"✅ **Subscription Activated!**\n\n"
                    f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
                    f"Price: {plan_info['price']}\n"
                    f"Duration: {duration} days\n\n"
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
        plan_info = Config.get_plan(plan_key)

        text = f"{plan_info['emoji']} **{plan_info['name']}**\n\n"
        text += f"💰 Price: {plan_info['price']}\n"
        text += f"⏰ Duration: {plan_info['duration_days']} days\n\n"
        text += "**Features:**\n"
        for feature in plan_info['features']:
            text += f"{feature}\n"

        await query.edit_message_text(
            text,
            reply_markup=get_plan_action_keyboard(plan_key),
            parse_mode="Markdown"
        )

    # Subscription status
    elif data == "subscription_status":
        subscription = await db_manager.get_active_subscription(user_id)

        if not subscription:
            text = "⚠️ You don't have an active subscription.\n\nChoose a plan to get started!"
        else:
            days_left = await db_manager.get_days_left(user_id)
            plan_info = Config.get_plan(subscription['plan'])

            text = "📊 **Subscription Status**\n\n"
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

    # Admin callbacks
    elif data == "admin_stats":
        if not Config.is_admin(user_id):
            await query.answer("❌ Admin only", show_alert=True)
            return

        stats = await db_manager.get_statistics()
        text = "📊 **Bot Statistics**\n\n"
        text += f"👥 Total Users: {stats.get('total_users', 0)}\n"
        text += f"✅ Active: {stats.get('active_subscriptions', 0)}\n"
        text += f"❌ Expired: {stats.get('expired_subscriptions', 0)}\n"

        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="Markdown"
        )

    elif data == "admin_backup":
        if not Config.is_admin(user_id):
            await query.answer("❌ Admin only", show_alert=True)
            return

        await query.answer("Creating backup...", show_alert=False)
        backup_path = await db_manager.backup_to_json()

        if backup_path:
            text = f"✅ Backup created!\n\n`{backup_path}`"
        else:
            text = "❌ Backup failed"

        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="Markdown"
        )


# ==========================================
# Message Handlers
# ==========================================

async def handle_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle email input for account creation"""
    if not context.user_data.get("awaiting_email"):
        return

    email = update.message.text.strip()
    user = update.effective_user

    # Basic email validation
    if "@" not in email or "." not in email.split("@")[-1]:
        await update.message.reply_text(
            "❌ Invalid email format. Please try again.\n\n"
            "Format: youremail@example.com",
            parse_mode="Markdown"
        )
        return

    # Create account
    username = user.username or f"user_{user.id}"
    db_user = await db_manager.create_user(user.id, username, email)

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


# ==========================================
# Error Handler
# ==========================================

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
