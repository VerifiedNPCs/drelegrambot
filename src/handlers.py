"""Bot command and callback handlers"""
import logging
import time
from payment import CryptoPaymentGateway
from token_manager import TokenManager
from payment_manager import PaymentManager
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from telegram.ext import ContextTypes

from config import Config
from keyboards import *

logger = logging.getLogger(__name__)

# Database manager will be injected
db_manager = None
payment_manager = None


def set_db_manager(manager):
    """Set the database manager instance and init payment manager"""
    global db_manager, payment_manager
    db_manager = manager
    payment_manager = PaymentManager(manager)


# ==========================================
# Helper Functions
# ==========================================

async def format_user_status(user_id: int) -> str:
    """Format user status message"""
    try:
        user = await db_manager.get_user(user_id)
        if not user:
            return "❌ No account found"

        status = "👤 <b>Account Information</b>\n\n"
        status += f"Username: @{user['username']}\n"
        status += f"Email: {user['email']}\n\n"

        subscription = await db_manager.get_active_subscription(user_id)

        if subscription:
            plan_info = Config.get_plan(subscription['plan'])
            status += "📊 <b>Subscription Status</b>\n\n"
            status += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
            status += f"Price: {plan_info['price']}\n"

            days_left = await db_manager.get_days_left(user_id)
            if days_left is not None:
                if days_left > 0:
                    status += f"Days Left: {days_left} days\n"
                else:
                    status += "Status: ⚠️ Expired\n"
        else:
            status += "📊 <b>Subscription Status</b>\n\n"
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
        "🎯 <b>Subscription Management Bot</b>\n\n"
        "Manage your subscription plans with ease. "
        "Choose from our flexible plans designed for your needs.\n\n"
        "Select an option below:"
    )

    try:
        if update.message:
            await update.message.reply_text(
                welcome_text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="HTML"
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=welcome_text,
                reply_markup=get_main_menu_keyboard(),
                parse_mode="HTML"
            )
    except Exception:
        logger.exception("start_command failed")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Help command"""
    help_text = (
        "📚 <b>Help & Support</b>\n\n"
        "<b>Available Commands:</b>\n"
        "/start - Main menu\n"
        "/account - View your account\n"
        "/plans - View subscription plans\n"
        "/status - Check subscription status\n"
        "/help - Show this help message\n\n"
    )

    # Add admin commands if user is admin
    if Config.is_admin(update.effective_user.id):
        help_text += (
            "<b>Admin Commands:</b>\n"
            "/stats - View bot statistics\n"
            "/backup - Create database backup\n"
            "/admin - Admin panel\n\n"
        )

    help_text += "<b>Need assistance?</b>\nContact: support@example.com"

    try:
        await update.message.reply_text(
            help_text,
            reply_markup=get_back_keyboard(),
            parse_mode="HTML"
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
            parse_mode="HTML"
        )
    except Exception:
        logger.exception("account_command failed")


async def plans_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show available plans"""
    text = "💳 <b>Choose Your Subscription Plan</b>\n\nSelect a plan that fits your needs:"

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_plans_keyboard(),
            parse_mode="HTML"
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

        text = "📊 <b>Subscription Status</b>\n\n"
        text += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
        text += f"Price: {plan_info['price']}\n\n"

        if days_left and days_left > 0:
            text += f"⏰ Days Remaining: <b>{days_left} days</b>\n\n"
            if days_left <= 7:
                text += "⚠️ Your subscription is expiring soon!\n"
                text += "Consider renewing to avoid service interruption."
        else:
            text += "❌ Status: <b>Expired</b>\n\n"
            text += "Please renew your subscription to continue using our services."

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_main_menu_keyboard(),
            parse_mode="HTML"
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

        text = "📊 <b>Bot Statistics</b>\n\n"
        text += f"👥 Total Users: {stats.get('total_users', 0)}\n"
        text += f"✅ Active Subscriptions: {stats.get('active_subscriptions', 0)}\n"
        text += f"❌ Expired Subscriptions: {stats.get('expired_subscriptions', 0)}\n"
        text += f"🚫 Cancelled Subscriptions: {stats.get('cancelled_subscriptions', 0)}\n\n"

        if stats.get('subscriptions_by_plan'):
            text += "<b>Subscriptions by Plan:</b>\n"
            for plan, count in stats['subscriptions_by_plan'].items():
                plan_info = Config.get_plan(plan)
                if plan_info:
                    text += f"{plan_info['emoji']} {plan_info['name']}: {count}\n"

        await update.message.reply_text(text, parse_mode="HTML")

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
                f"File: <code>{backup_path}</code>",
                parse_mode="HTML"
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
        "🔐 <b>Admin Panel</b>\n\n"
        "Welcome to the admin control panel. "
        "Use the buttons below to manage the bot."
    )

    try:
        await update.message.reply_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="HTML"
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

    # --- Main Navigation ---
    if data == "main_menu":
        text = f"👋 Welcome back {user.first_name}!\n\nSelect an option below:"
        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(user_id),
            parse_mode="HTML"
        )

    # --- Account Management ---
    elif data == "create_account":
        db_user = await db_manager.get_user(user_id)
        if db_user:
            text = "✅ You already have an account!\n\n" + await format_user_status(user_id)
            await query.edit_message_text(
                text,
                reply_markup=get_main_menu_keyboard(user_id),
                parse_mode="HTML"
            )
        else:
            text = (
                "📝 <b>Create Your Account</b>\n\n"
                "Please send your email address to complete registration.\n\n"
                "Format: youremail@example.com"
            )
            context.user_data["awaiting_email"] = True
            await query.edit_message_text(
                text,
                reply_markup=get_back_keyboard(),
                parse_mode="HTML"
            )

    elif data == "my_account":
        db_user = await db_manager.get_user(user_id)
        if not db_user:
            text = "❌ You don't have an account yet.\n\nClick 'Create Account' to get started!"
        else:
            text = await format_user_status(user_id)

        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(user_id),
            parse_mode="HTML"
        )

    # --- Plan Selection & Payment Flow ---
    elif data == "choose_plan":
        db_user = await db_manager.get_user(user_id)
        if not db_user:
            await query.edit_message_text(
                "❌ Please create an account first!",
                reply_markup=get_main_menu_keyboard(),
                parse_mode="HTML"
            )
        else:
            text = "💳 <b>Choose Your Subscription Plan</b>\n\nSelect a plan that fits your needs:"
            await query.edit_message_text(
                text,
                reply_markup=get_plans_keyboard(),
                parse_mode="HTML"
            )

    elif data == "plan_details":
        text = "📋 <b>Subscription Plans Overview</b>\n\nSelect a plan to see detailed information:"
        await query.edit_message_text(
            text,
            reply_markup=get_plan_details_keyboard(),
            parse_mode="HTML"
        )
        
    elif data.startswith("detail_"):
        plan_key = data.replace("detail_", "")
        plan_info = Config.get_plan(plan_key)

        text = f"{plan_info['emoji']} <b>{plan_info['name']}</b>\n\n"
        text += f"💰 Price: {plan_info['price']}\n"
        text += f"⏰ Duration: {plan_info['duration_days']} days\n\n"
        text += "<b>Features:</b>\n"
        for feature in plan_info['features']:
            text += f"• {feature}\n"

        await query.edit_message_text(
            text,
            reply_markup=get_plan_action_keyboard(plan_key),
            parse_mode="HTML"
        )

    # --- Core Payment Logic (Consolidated) ---
    elif data.startswith("plan_"):
        plan_key = data.replace("plan_", "")
        
        # 1. Check Account
        db_user = await db_manager.get_user(user_id)
        if not db_user:
            await query.edit_message_text(
                "❌ Please create an account first!",
                reply_markup=get_main_menu_keyboard(),
                parse_mode="HTML"
            )
            return

        plan_info = Config.get_plan(plan_key)
        
        # 2. Calculate Cost & Credit
        calculation = await payment_manager.calculate_proration(user_id, plan_key)
        
        cost = calculation['cost_to_pay']
        credit = calculation['credit_applied']
        message = calculation['message']
        
        # 3. Create Payment Request in DB
        payment_record = await db_manager.create_payment_request(
            user_id=user_id,
            plan=plan_key,
            amount=cost,
            currency="USD",
            gateway_name="manual_browser" # Updated gateway name
        )
        
        if not payment_record:
            await query.edit_message_text("❌ Error processing request. Try again.")
            return

        payment_id = payment_record['payment_id']
        
        # 4. Generate Standard Browser Link
        # This will open in Chrome/Safari instead of Telegram WebApp
        payment_url = f"{Config.PAYMENT_URL}?payment_id={payment_id}"
        
        # Message construction
        msg_text = f"💳 <b>{message}</b>\n\n"
        msg_text += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
        
        if credit > 0:
            msg_text += f"Unused Credit: -${credit:.2f}\n"
        
        msg_text += f"<b>Total Due: ${cost:.2f}</b>\n\n"
        
        if cost <= 0:
            msg_text += "✨ Your credit covers this change!\nClick below to confirm."
            btn_text = "✅ Confirm Switch"
        else:
            msg_text += "Click below to proceed to payment:"
            btn_text = "💎 Pay via Browser"

        # UPDATED: Use 'url' instead of 'web_app'
        keyboard = [
            [InlineKeyboardButton(btn_text, url=payment_url)], 
            [InlineKeyboardButton("« Cancel", callback_data="choose_plan")]
        ]
        
        await query.edit_message_text(
            msg_text,
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode="HTML"
        )

    # --- Dashboard Access ---
    elif data == "open_dashboard":
        db_user = await db_manager.get_user(user_id)
        if not db_user:
            text = "❌ <b>Account Required</b>\n\nPlease create an account first."
            await query.edit_message_text(text, reply_markup=get_main_menu_keyboard(), parse_mode="HTML")
        else:
            # 1. Generate Raw Token
            dashboard_token = TokenManager.generate_secure_token(user_id, db_user['username'], "dashboard")
            
            # 2. Hash it for storage
            dashboard_token_hash = TokenManager.hash_token(dashboard_token)
            
            # 3. Store HASH in DB
            await db_manager.create_user_token(
                user_id, dashboard_token_hash, "dashboard", expires_hours=24
            )
            
            # 4. Construct URL with RAW Token
            dashboard_url = f"{Config.DASHBOARD_URL}?user_id={user_id}&token={dashboard_token}"
            
            text = (
                "🌐 <b>Dashboard Access</b>\n\n"
                "Click below to open your personal dashboard.\n"
                "Link is valid for 24 hours."
            )
            
            # 5. Pass URL to keyboard
            await query.edit_message_text(
                text,
                reply_markup=get_dashboard_keyboard(dashboard_url), # PASS URL HERE
                parse_mode="HTML"
            )

    # --- Subscription Status ---
    elif data == "subscription_status":
        subscription = await db_manager.get_active_subscription(user_id)

        if not subscription:
            text = "⚠️ You don't have an active subscription.\n\nChoose a plan to get started!"
        else:
            days_left = await db_manager.get_days_left(user_id)
            plan_info = Config.get_plan(subscription['plan'])

            text = "📊 <b>Subscription Status</b>\n\n"
            text += f"Plan: {plan_info['emoji']} {plan_info['name']}\n"
            text += f"Price: {plan_info['price']}\n\n"

            if days_left is not None and days_left > 0:
                text += f"⏰ Days Remaining: <b>{days_left} days</b>\n"
                if days_left <= 7:
                    text += "\n⚠️ Your subscription is expiring soon!"
            else:
                text += "❌ Status: <b>Expired</b>\nPlease renew your subscription."

        await query.edit_message_text(
            text,
            reply_markup=get_main_menu_keyboard(user_id),
            parse_mode="HTML"
        )

    # --- Help ---
    elif data == "help":
        help_text = (
            "📚 <b>Help & Support</b>\n\n"
            "<b>How to use this bot:</b>\n"
            "1️⃣ Create an account\n"
            "2️⃣ Choose a subscription plan\n"
            "3️⃣ Pay securely via Dashboard\n\n"
            "<b>Need assistance?</b>\n"
            "Contact: @drele_gram"
        )
        await query.edit_message_text(
            help_text,
            reply_markup=get_back_keyboard(),
            parse_mode="HTML"
        )

    # --- Admin Section ---
    elif data == "admin_panel":
        if not Config.is_admin(user_id):
            await query.answer("❌ Admin only", show_alert=True)
            return
        
        text = "🔐 <b>Admin Panel</b>\n\nSelect an option:"
        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="HTML"
        )

    elif data == "admin_stats":
        if not Config.is_admin(user_id): return
        stats = await db_manager.get_statistics()
        text = "📊 <b>Bot Statistics</b>\n\n"
        text += f"👥 Total Users: {stats.get('total_users', 0)}\n"
        text += f"✅ Active Subs: {stats.get('active_subscriptions', 0)}\n"
        text += f"❌ Expired Subs: {stats.get('expired_subscriptions', 0)}\n"
        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="HTML"
        )

    elif data == "admin_backup":
        if not Config.is_admin(user_id): return
        await query.answer("Creating backup...", show_alert=False)
        backup_path = await db_manager.backup_to_json()
        if backup_path:
            text = f"✅ Backup created!\n\n<code>{backup_path}</code>"
        else:
            text = "❌ Backup failed"
        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="HTML"
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
            parse_mode="HTML"
        )
        return

    # Create account
    username = user.username or f"user_{user.id}"
    db_user = await db_manager.create_user(user.id, username, email)

    context.user_data["awaiting_email"] = False

    if db_user:
        text = (
            "✅ <b>Account Created Successfully!</b>\n\n"
            f"Username: @{username}\n"
            f"Email: {email}\n\n"
            "You can now choose a subscription plan!"
        )
    else:
        text = (
            "❌ <b>Account Creation Failed</b>\n\n"
            "This email might already be in use. Please try a different email."
        )

    await update.message.reply_text(
        text,
        reply_markup=get_main_menu_keyboard(),
        parse_mode="HTML"
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
