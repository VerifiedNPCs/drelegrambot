"""Bot command and callback handlers"""
import logging
import time
from payment import CryptoPaymentGateway
from token_manager import TokenManager
from payment_manager import PaymentManager
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from telegram.ext import ContextTypes
from datetime import datetime, timezone

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
    user_id = user.id
    username = user.username or f"user_{user_id}"

    # Check if user exists in database
    db_user = await db_manager.get_user(user_id)

    if context.args:
        deep_link = context.args[0]
        
        # Ensure user exists before handling deep links
        if not db_user:
            await db_manager.create_user(user_id, username, f"{user_id}@telegram.user")
        
        # ✅ Payment Success Deep Link
        if deep_link == "payment_success":
            subscription = await db_manager.get_active_subscription(user_id)
            
            if subscription:
                days_left = await db_manager.get_days_left(user_id)
                plan_info = Config.get_plan(subscription['plan'])
                
                success_text = (
                    "✅ <b>Payment Successful!</b>\n\n"
                    f"🎉 Your <b>{plan_info['name']}</b> subscription is now active!\n\n"
                    f"📊 <b>Subscription Details:</b>\n"
                    f"• Plan: {plan_info['emoji']} {plan_info['name']}\n"
                    f"• Price: {plan_info['price']}\n"
                    f"• Days Remaining: {days_left} days\n"
                    f"• Expires: {subscription['end_date'].strftime('%Y-%m-%d')}\n\n"
                    "Your premium features are ready to use! 🚀\n\n"
                    "What would you like to do?"
                )
                
                keyboard = [
                    [InlineKeyboardButton("🌐 Open Dashboard", callback_data="open_dashboard")],
                    [InlineKeyboardButton("📜 Payment History", callback_data="payment_history")],
                    [InlineKeyboardButton("« Main Menu", callback_data="main_menu")],
                ]
                
                await update.message.reply_text(
                    success_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode='HTML'
                )
                return
            else:
                # Payment successful but no subscription found (edge case)
                await update.message.reply_text(
                    "✅ Payment received!\n\n"
                    "Your subscription is being activated. Please wait a moment...",
                    reply_markup=get_main_menu_keyboard(),
                    parse_mode='HTML'
                )
                return
        
        # ❌ Payment Failed Deep Link
        elif deep_link == "payment_failed":
            failed_text = (
                "⚠️ <b>Payment Issue Detected</b>\n\n"
                "We noticed there was an issue with your payment.\n\n"
                "<b>Common Issues:</b>\n"
                "• Insufficient amount sent\n"
                "• Wrong wallet address\n"
                "• Network delays (can take 10-30 minutes)\n"
                "• Incorrect transaction hash\n\n"
                "<b>What to do next:</b>\n"
                "1️⃣ Check your wallet transaction history\n"
                "2️⃣ Wait 15-30 minutes for blockchain confirmation\n"
                "3️⃣ If you sent the correct amount, contact support\n\n"
                "💬 <b>Need help?</b> Click the button below:"
            )
            
            keyboard = [
                [InlineKeyboardButton("💬 Contact Support", url="https://t.me/drele_gram")],
                [InlineKeyboardButton("🔄 Try Payment Again", callback_data="view_plans")],
                [InlineKeyboardButton("« Main Menu", callback_data="main_menu")],
            ]
            
            await update.message.reply_text(
                failed_text,
                reply_markup=InlineKeyboardMarkup(keyboard),
                parse_mode='HTML'
            )
            return
        
        # ⏱️ Payment Cancelled Deep Link
        elif deep_link == "payment_cancelled":
            cancelled_text = (
                "❌ <b>Payment Cancelled</b>\n\n"
                "No worries! You can start a new payment anytime.\n\n"
                "Ready to subscribe?"
            )
            
            await update.message.reply_text(
                cancelled_text,
                reply_markup=get_plans_keyboard(),
                parse_mode='HTML'
            )
            return
        
        # 📊 Subscription Status Deep Link
        elif deep_link == "subscription":
            subscription = await db_manager.get_active_subscription(user_id)
            
            if subscription:
                days_left = await db_manager.get_days_left(user_id)
                plan_info = Config.get_plan(subscription['plan'])
                
                status_text = (
                    "📊 <b>Subscription Status</b>\n\n"
                    f"✅ Active: {plan_info['emoji']} {plan_info['name']}\n"
                    f"💰 Price: {plan_info['price']}\n"
                    f"⏰ Days Left: {days_left}\n"
                    f"📆 Expires: {subscription['end_date'].strftime('%Y-%m-%d')}"
                )
                
                keyboard = [
                    [InlineKeyboardButton("🔄 Renew/Upgrade", callback_data="view_plans")],
                    [InlineKeyboardButton("🌐 Open Dashboard", callback_data="open_dashboard")],
                    [InlineKeyboardButton("« Main Menu", callback_data="main_menu")],
                ]
                
                await update.message.reply_text(
                    status_text,
                    reply_markup=InlineKeyboardMarkup(keyboard),
                    parse_mode='HTML'
                )
            else:
                await update.message.reply_text(
                    "❌ <b>No Active Subscription</b>\n\n"
                    "You don't have an active subscription yet.\n"
                    "Choose a plan to get started!",
                    reply_markup=get_plans_keyboard(),
                    parse_mode='HTML'
                )
            return
    
    # Default behavior (no deep link or new user)
    if not db_user:
        # New user - create account
        await db_manager.create_user(user_id, username, f"{user_id}@telegram.user")
        
        welcome_text = (
            f"👋 <b>Welcome {user.first_name}!</b>\n\n"
            "🎯 Your account has been created!\n\n"
            "Manage your subscription plans with ease. "
            "Choose from our flexible plans designed for your needs.\n\n"
            "Select an option below:"
        )
    else:
        # Returning user
        welcome_text = (
            f"👋 <b>Welcome back, {user.first_name}!</b>\n\n"
            "What would you like to do today?"
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
        
        # --- Cancellation Logic ---
    elif data == "cancel_plan_warning":
        # 1. Check if they actually have an active plan first
        subscription = await db_manager.get_active_subscription(user_id)
        if not subscription:
            await query.edit_message_text(
                "⚠️ <b>Error</b>\n\nYou do not have an active subscription to cancel.",
                reply_markup=get_main_menu_keyboard(user_id),
                parse_mode="HTML"
            )
            return

        # 2. Show the Warning Message
        warning_msg = (
            "⚠️ <b>ARE YOU SURE?</b>\n\n"
            "You are about to cancel your subscription. \n\n"
            "❗️ <b>This action cannot be undone.</b>\n"
            "💸 <b>NO REFUNDS:</b> You will NOT receive money back for remaining days.\n"
            "🛑 <b>Access Revoked:</b> Your access will be terminated immediately.\n\n"
            "If you have billing issues, please contact @drele_gram instead."
        )
        
        await query.edit_message_text(
            text=warning_msg,
            reply_markup=get_cancel_confirmation_keyboard(), # From your new keyboard.py
            parse_mode="HTML"
        )
    # --- Cancellation Logic ---
    elif data == "confirm_cancel_plan":
        # 3. Execute Cancellation
        success = await db_manager.cancel_subscription(user_id)
        
        if success:
            msg = (
                "✅ <b>Subscription Cancelled</b>\n\n"
                "Your plan has been terminated effective immediately.\n"
                "You will no longer be charged.\n\n"
                "We are sorry to see you go!"
            )
        else:
            msg = (
                "⚠️ <b>Cancellation Failed</b>\n\n"
                "Your subscription may have already expired or been cancelled."
            )
            
        await query.edit_message_text(
            text=msg,
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
        
    elif data == "admin_users":
        if not Config.is_admin(user_id): 
            await query.answer("❌ Admin only", show_alert=True)
            return
        
        # Fetch users with pagination
        users = await db_manager.get_all_users(limit=20)
        total_users = await db_manager.count_users()
        
        if not users:
            text = "📋 <b>User List</b>\n\n❌ No users found."
        else:
            text = f"👥 <b>User List</b> (Total: {total_users})\n\n"
            
            for idx, user in enumerate(users, 1):
                # Get active subscription for this user
                sub = await db_manager.get_active_subscription(user['user_id'])
                plan_emoji = "❌"
                plan_name = "No Plan"
                
                if sub:
                    plan_info = Config.get_plan(sub['plan'])
                    if plan_info:
                        plan_emoji = plan_info['emoji']
                        plan_name = plan_info['name']
                
                text += f"{idx}. @{user['username']} ({user['user_id']})\n"
                text += f"   {plan_emoji} {plan_name} | {user['email']}\n\n"
                
                # Limit to prevent message overflow
                if idx >= 15:
                    text += f"... and {total_users - 15} more users"
                    break
        
        await query.edit_message_text(
            text,
            reply_markup=get_admin_keyboard(),
            parse_mode="HTML"
        )

    elif data == "admin_subs":
        if not Config.is_admin(user_id): 
            await query.answer("❌ Admin only", show_alert=True)
            return
        
        # Fetch all active subscriptions
        async with db_manager.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT s.*, u.username, u.email 
                FROM subscriptions s
                JOIN users u ON s.user_id = u.user_id
                WHERE s.status = 'active'
                ORDER BY s.end_date ASC
                LIMIT 20
            """)
        
        if not rows:
            text = "📋 <b>Active Subscriptions</b>\n\n❌ No active subscriptions found."
        else:
            text = f"✅ <b>Active Subscriptions</b> ({len(rows)})\n\n"
            
            for idx, sub in enumerate(rows, 1):
                plan_info = Config.get_plan(sub['plan'])
                days_left = (sub['end_date'] - datetime.now(timezone.utc)).days
                
                text += f"{idx}. @{sub['username']}\n"
                text += f"   Plan: {plan_info['emoji']} {plan_info['name']}\n"
                text += f"   Expires: {sub['end_date'].strftime('%Y-%m-%d')} ({days_left}d left)\n\n"
                
                if idx >= 10:
                    text += f"... and {len(rows) - 10} more subscriptions"
                    break
        
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
