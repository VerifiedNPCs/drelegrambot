"""
Payment logic manager for handling subscriptions, upgrades, and downgrades
"""
import logging
from datetime import datetime, timedelta, timezone
from config import Config
from database import DatabaseManager

logger = logging.getLogger(__name__)

class PaymentManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def calculate_proration(self, user_id: int, new_plan_key: str, coin_count: int = None):
        """
        Calculate upgrade/downgrade cost and credit.
        Returns dict with cost, credit, new_expiry details.
        
        Args:
            user_id: User ID
            new_plan_key: Target plan key (standard/pro/business)
            coin_count: Number of coins to track (if None, uses current subscription's count)
        """
        try:
            # 1. Get current subscription
            current_sub = await self.db.get_active_subscription(user_id)
            new_plan_info = Config.get_plan(new_plan_key)
            
            # Determine coin count
            if coin_count is None:
                # If switching plans, keep current coin count
                if current_sub and 'coin_count' in current_sub:
                    coin_count = current_sub['coin_count']
                else:
                    # Default to minimum (1 coin) if not specified
                    coin_count = 1
            
            # Calculate new price using dynamic pricing
            new_price = Config.calculate_price(new_plan_key, coin_count)

            # CASE A: New User or Expired Subscription
            if not current_sub:
                return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': datetime.now(timezone.utc) + timedelta(days=30),
                    'is_upgrade': True,
                    'message': "New Subscription",
                    'coin_count': coin_count
                }

            # CASE B: Active Subscription Switch
            current_plan_key = current_sub['plan']
            current_coin_count = current_sub.get('coin_count', 1)
            
            # Calculate current plan price with current coin count
            current_price = Config.calculate_price(current_plan_key, current_coin_count)
            
            # If same plan AND same coin count, just extend
            if current_plan_key == new_plan_key and current_coin_count == coin_count:
                return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': current_sub['end_date'] + timedelta(days=30),
                    'is_upgrade': False,
                    'message': "Renewal",
                    'coin_count': coin_count
                }

            # Calculate days remaining/unused value
            now = datetime.now(timezone.utc)
            # Ensure timezone-aware datetime
            expiry = current_sub['end_date']
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            
            if expiry <= now:
                return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': now + timedelta(days=30),
                    'is_upgrade': True,
                    'message': "Renewal (Expired)",
                    'coin_count': coin_count
                }

            days_remaining = (expiry - now).days
            if days_remaining < 0: 
                days_remaining = 0
            
            # Calculate Unused Value (Daily Rate * Days Remaining)
            # Using 30-day standard month
            daily_rate = current_price / 30.0 
            unused_value = daily_rate * days_remaining

            logger.info(f"User {user_id}: Switching {current_plan_key} ({current_coin_count} coins, ${current_price}) -> {new_plan_key} ({coin_count} coins, ${new_price})")
            logger.info(f"Days left: {days_remaining}, Unused Value: ${unused_value:.2f}")

            # Calculate Difference
            cost_difference = new_price - unused_value
            
            if cost_difference > 0:
                # UPGRADE/INCREASE: Pay difference
                # Expiry starts fresh 30 days from now
                return {
                    'cost_to_pay': round(cost_difference, 2),
                    'credit_applied': round(unused_value, 2),
                    'new_expiry': now + timedelta(days=30),
                    'is_upgrade': True,
                    'message': f"Upgrade from {new_plan_info['name']}",
                    'coin_count': coin_count
                }
            else:
                # DOWNGRADE/DECREASE: Credit covers cost
                # Pay $0, remaining credit extends duration
                credit_surplus = abs(cost_difference) 
                
                # How many extra days does surplus buy in new plan?
                new_daily_rate = new_price / 30.0
                if new_daily_rate > 0:
                    extra_days = int(credit_surplus / new_daily_rate)
                else:
                    extra_days = 0 
                
                total_days = 30 + extra_days
                
                return {
                    'cost_to_pay': 0.0,
                    'credit_applied': round(unused_value, 2),
                    'new_expiry': now + timedelta(days=total_days),
                    'is_upgrade': False,
                    'message': f"Downgrade from {new_plan_info['name']}",
                    'coin_count': coin_count
                }
                
        except Exception as e:
            logger.exception(f"Error calculating proration: {e}")
            # Fallback to full price on error
            fallback_coins = coin_count if coin_count else 1
            fallback_price = Config.calculate_price(new_plan_key, fallback_coins)
            return {
                'cost_to_pay': fallback_price,
                'credit_applied': 0.0,
                'new_expiry': datetime.now(timezone.utc) + timedelta(days=30),
                'is_upgrade': True,
                'message': "Error Fallback",
                'coin_count': fallback_coins
            }
