"""Telegram keyboard layouts"""
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, WebAppInfo
from config import Config
from token_manager import TokenManager

# Update get_dashboard_keyboard function:
def get_dashboard_keyboard(dashboard_url: str) -> InlineKeyboardMarkup:
    """
    Dashboard button.
    NOTE: Takes the FULL URL now, doesn't generate its own token.
    """
    keyboard = [
        [
            InlineKeyboardButton(
                "🌐 Open Dashboard", 
                url=dashboard_url
            )
        ],
        [InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)

def get_payment_plans_keyboard() -> InlineKeyboardMarkup:
    """Payment selection for subscription plans"""
    keyboard = []
    for plan_key, plan_info in Config.SUBSCRIPTION_PLANS.items():
        keyboard.append([
            InlineKeyboardButton(
                f"💳 Pay for {plan_info['name']} - {plan_info['price']}",
                callback_data=f"pay_{plan_key}"
            )
        ])
    keyboard.append([InlineKeyboardButton("« Back", callback_data="my_account")])
    return InlineKeyboardMarkup(keyboard)


def get_main_menu_keyboard(user_id: int = None) -> InlineKeyboardMarkup:
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
            InlineKeyboardButton("🌐 Dashboard & Pay", callback_data="open_dashboard"),
            InlineKeyboardButton("⏰ Subscription Status", callback_data="subscription_status"),
        ],
        [
            InlineKeyboardButton("ℹ️ Help", callback_data="help"),
        ]
    ]
    
    # Add admin button only for admins
    if user_id and Config.is_admin(user_id):
        keyboard.append([InlineKeyboardButton("🔐 Admin Panel", callback_data="admin_panel")])
    
    return InlineKeyboardMarkup(keyboard)


def get_plans_keyboard() -> InlineKeyboardMarkup:
    """Subscription plans selection keyboard"""
    keyboard = []
    for plan_key, plan_info in Config.SUBSCRIPTION_PLANS.items():
        keyboard.append([
            InlineKeyboardButton(
                f"{plan_info['emoji']} {plan_info['name']} - {plan_info['price']}",
                callback_data=f"plan_{plan_key}"
            )
        ])
    keyboard.append([InlineKeyboardButton("« Back to Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(keyboard)


def get_plan_details_keyboard() -> InlineKeyboardMarkup:
    """Plan details menu keyboard"""
    keyboard = []
    for plan_key, plan_info in Config.SUBSCRIPTION_PLANS.items():
        keyboard.append([
            InlineKeyboardButton(
                f"{plan_info['emoji']} {plan_info['name']} Details",
                callback_data=f"detail_{plan_key}"
            )
        ])
    keyboard.append([InlineKeyboardButton("« Back to Menu", callback_data="main_menu")])
    return InlineKeyboardMarkup(keyboard)


def get_back_keyboard() -> InlineKeyboardMarkup:
    """Simple back to menu button"""
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("« Back to Menu", callback_data="main_menu")
    ]])


def get_plan_action_keyboard(plan_key: str) -> InlineKeyboardMarkup:
    """Keyboard for specific plan actions"""
    plan_info = Config.get_plan(plan_key)
    keyboard = [
        [InlineKeyboardButton(f"Subscribe to {plan_info['name']}", callback_data=f"plan_{plan_key}")],
        [InlineKeyboardButton("« Back to Plans", callback_data="plan_details")],
        [InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)


def get_admin_keyboard() -> InlineKeyboardMarkup:
    """Admin control panel keyboard"""
    keyboard = [
        [
            InlineKeyboardButton("📊 Statistics", callback_data="admin_stats"),
            InlineKeyboardButton("💾 Backup", callback_data="admin_backup"),
        ],
        [
            InlineKeyboardButton("👥 Users List", callback_data="admin_users"),
            InlineKeyboardButton("📋 Subscriptions", callback_data="admin_subs"),
        ],
        [InlineKeyboardButton("« Back to Menu", callback_data="main_menu")]
    ]
    return InlineKeyboardMarkup(keyboard)
