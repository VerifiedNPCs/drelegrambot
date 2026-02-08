"""Crypto payment gateway integration"""
import logging
import httpx
import hashlib
import hmac
from typing import Optional, Dict
from config import Config

logger = logging.getLogger(__name__)


class CryptoPaymentGateway:
    """NOWPayments API integration for crypto payments"""
    
    def __init__(self):
        self.api_key = Config.NOWPAYMENTS_API_KEY  # Add to config.py
        self.api_url = "https://api.nowpayments.io/v1"
        self.headers = {"x-api-key": self.api_key}
    
    async def create_payment(
        self, 
        price_amount: float, 
        price_currency: str, 
        pay_currency: str,
        order_id: str,
        order_description: str,
        ipn_callback_url: str
    ) -> Optional[Dict]:
        """Create a payment request"""
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "price_amount": price_amount,
                    "price_currency": price_currency,
                    "pay_currency": pay_currency,
                    "order_id": order_id,
                    "order_description": order_description,
                    "ipn_callback_url": ipn_callback_url,
                }
                
                response = await client.post(
                    f"{self.api_url}/payment",
                    json=payload,
                    headers=self.headers,
                    timeout=30.0
                )
                
                if response.status_code == 201:
                    return response.json()
                else:
                    logger.error(f"Payment creation failed: {response.text}")
                    return None
                    
        except Exception as e:
            logger.exception(f"Payment gateway error: {e}")
            return None
    
    async def get_available_currencies(self) -> list:
        """Get list of available cryptocurrencies"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.api_url}/currencies",
                    headers=self.headers
                )
                if response.status_code == 200:
                    data = response.json()
                    return data.get("currencies", [])
                return []
        except Exception as e:
            logger.exception(f"Failed to fetch currencies: {e}")
            return []


# Alternative: Simple payment link generator (no API needed)
class SimpleCryptoPayment:
    """Generate crypto payment links without external API"""
    
    @staticmethod
    def generate_payment_link(
        user_id: int,
        plan_key: str,
        wallet_address: str
    ) -> str:
        """Generate a simple crypto payment instruction"""
        plan_info = Config.get_plan(plan_key)
        amount = plan_info['price'].replace('$', '').replace('/month', '')
        
        # This is a placeholder - replace with your actual wallet
        return f"https://your-payment-page.com?user={user_id}&plan={plan_key}&amount={amount}"
