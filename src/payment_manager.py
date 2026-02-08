"""
Payment logic manager for handling subscriptions, upgrades, and downgrades
"""
import logging
from datetime import datetime, timedelta
from config import Config
from database import DatabaseManager

logger = logging.getLogger(__name__)

class PaymentManager:
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def calculate_proration(self, user_id: int, new_plan_key: str):
        """
        Calculate upgrade/downgrade cost and credit.
        Returns dict with cost, credit, new_expiry details.
        """
        try:
            # 1. Get current subscription
            current_sub = await self.db.get_active_subscription(user_id)
            new_plan_info = Config.get_plan(new_plan_key)
            
            # Extract numeric price (e.g., "$29.99/month" -> 29.99)
            new_price = float(new_plan_info['price'].replace('$', '').split('/')[0])

            # CASE A: New User or Expired Subscription
            if not current_sub:
                return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': datetime.utcnow() + timedelta(days=30),
                    'is_upgrade': True,
                    'message': "New Subscription"
                }

            # CASE B: Active Subscription Switch
            current_plan_key = current_sub['plan']
            
            # If same plan, just extend
            if current_plan_key == new_plan_key:
                 return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': current_sub['end_date'] + timedelta(days=30),
                    'is_upgrade': False,
                    'message': "Renewal"
                }

            current_plan_info = Config.get_plan(current_plan_key)
            current_price = float(current_plan_info['price'].replace('$', '').split('/')[0])
            
            # Calculate days remaining/unused value
            now = datetime.utcnow()
            # Ensure naive datetimes for calculation
            expiry = current_sub['end_date'].replace(tzinfo=None) if current_sub['end_date'].tzinfo else current_sub['end_date']
            
            if expiry <= now:
                 return {
                    'cost_to_pay': new_price,
                    'credit_applied': 0.0,
                    'new_expiry': now + timedelta(days=30),
                    'is_upgrade': True,
                    'message': "Renewal (Expired)"
                }

            days_remaining = (expiry - now).days
            if days_remaining < 0: days_remaining = 0
            
            # Calculate Unused Value (Daily Rate * Days Remaining)
            # Using 30-day standard month
            daily_rate = current_price / 30.0 
            unused_value = daily_rate * days_remaining

            logger.info(f"User {user_id}: Switching {current_plan_key} (${current_price}) -> {new_plan_key} (${new_price})")
            logger.info(f"Days left: {days_remaining}, Unused Value: ${unused_value:.2f}")

            # Calculate Difference
            cost_difference = new_price - unused_value
            
            if cost_difference > 0:
                # UPGRADE: Pay difference
                # Expiry starts fresh 30 days from now (simplest approach for upgrades)
                return {
                    'cost_to_pay': round(cost_difference, 2),
                    'credit_applied': round(unused_value, 2),
                    'new_expiry': now + timedelta(days=30),
                    'is_upgrade': True,
                    'message': f"Upgrade from {current_plan_info['name']}"
                }
            else:
                # DOWNGRADE: Credit covers cost
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
                    'message': f"Downgrade from {current_plan_info['name']}"
                }
                
        except Exception as e:
            logger.exception(f"Error calculating proration: {e}")
            # Fallback to full price on error
            return {
                'cost_to_pay': new_price,
                'credit_applied': 0.0,
                'new_expiry': datetime.utcnow() + timedelta(days=30),
                'is_upgrade': True,
                'message': "Error Fallback"
            }
