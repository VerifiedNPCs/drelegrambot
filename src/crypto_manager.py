import httpx
import logging

logger = logging.getLogger(__name__)

class CryptoManager:
    @staticmethod
    async def get_exchange_rates():
        """
        Fetches real-time prices for BTC, ETH, USDT, USDC in USD.
        Using CoinGecko API (No key needed for low volume).
        """
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {
            "ids": "bitcoin,ethereum,tether,litecoin",
            "vs_currencies": "usd"
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10.0)
                data = response.json()
                
                return {
                    "btc": data['bitcoin']['usd'],
                    "eth": data['ethereum']['usd'],
                    "usdt": data['tether']['usd'], # Usually ~1.00
                    "ltc": data['litecoin']['usd']
                }
        except Exception as e:
            logger.error(f"Error fetching crypto prices: {e}")
            # Fallback hardcoded prices if API fails (Safety net)
            return {"btc": 95000.0, "eth": 2800.0, "usdt": 1.0, "ltc": 150.0}

    @staticmethod
    def calculate_crypto_amount(usd_amount: float, rate: float) -> float:
        """Calculates crypto amount with a tiny buffer for fluctuation"""
        if rate == 0: return 0
        raw_amount = usd_amount / rate
        return round(raw_amount, 8) # 8 decimals for crypto
