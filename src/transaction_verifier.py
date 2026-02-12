# transaction_verifier.py
import httpx
import logging
from typing import Optional, Dict
from config import Config

logger = logging.getLogger(__name__)

class TransactionVerifier:
    """
    Verifies cryptocurrency transactions using blockchain explorer APIs.
    Implements intelligent fee tolerance to handle real-world payment scenarios.
    """
    
    @staticmethod
    async def verify_transaction(
        tx_hash: str,
        crypto_currency: str,
        expected_address: str,
        expected_amount: float,  # Crypto amount (e.g., 0.001 BTC)
        expected_usd_value: float  # USD value of the purchase
    ) -> Dict:
        """
        Verify transaction with intelligent fee tolerance.
        Returns: {"valid": bool, "message": str, "details": dict}
        """
        
        # TEST MODE BYPASS
        if Config.DEV_MODE and tx_hash == Config.TEST_TX_HASH:
            logger.warning("⚠️ TEST MODE: Bypassing verification")
            return {
                "valid": True, 
                "message": "Test mode - transaction accepted", 
                "details": {"test_mode": True}
            }
        
        try:
            crypto_lower = crypto_currency.lower()
            
            if crypto_lower in ['eth', 'usdc']:
                return await TransactionVerifier._verify_ethereum(
                    tx_hash, expected_address, expected_amount, expected_usd_value, crypto_lower
                )
            
            elif crypto_lower == 'btc':
                return await TransactionVerifier._verify_bitcoin(
                    tx_hash, expected_address, expected_amount, expected_usd_value
                )
            
            elif crypto_lower == 'usdt':
                return await TransactionVerifier._verify_tron(
                    tx_hash, expected_address, expected_amount, expected_usd_value
                )
                
            elif crypto_lower == 'ltc':
                return await TransactionVerifier._verify_litecoin(
                    tx_hash, expected_address, expected_amount, expected_usd_value
                )
            
            else:
                return {
                    "valid": False, 
                    "message": f"Unsupported cryptocurrency: {crypto_currency}",
                    "details": {}
                }
                
        except Exception as e:
            logger.exception(f"Transaction verification error: {e}")
            return {
                "valid": False, 
                "message": "Verification service error. Please try again.", 
                "details": {"error": str(e)}
            }
    
    
    @staticmethod
    def _is_amount_acceptable(
        received_amount: float,
        expected_amount: float,
        crypto_currency: str,
        expected_usd_value: float
    ) -> Dict:
        """
        Check if received amount is acceptable considering fees and price fluctuation.
        Returns: {"acceptable": bool, "reason": str, "needs_review": bool}
        """
        
        # Get tolerance settings for this crypto
        tolerance = Config.PAYMENT_TOLERANCE.get(crypto_currency.lower(), {
            "percentage": 0.02,
            "min_shortfall_usd": 5.0
        })
        
        # Calculate minimum acceptable amount
        min_acceptable = expected_amount * (1 - tolerance["percentage"])
        
        # Case 1: Full amount or more received
        if received_amount >= expected_amount:
            return {
                "acceptable": True,
                "reason": f"Full amount received: {received_amount} {crypto_currency.upper()}",
                "needs_review": False
            }
        
        # Case 2: Within percentage tolerance
        elif received_amount >= min_acceptable:
            shortfall = expected_amount - received_amount
            shortfall_usd = (shortfall / expected_amount) * expected_usd_value
            
            # If USD shortfall is small, auto-approve
            if shortfall_usd <= tolerance["min_shortfall_usd"]:
                return {
                    "acceptable": True,
                    "reason": f"Within tolerance: received {received_amount}, expected {expected_amount} (${shortfall_usd:.2f} short)",
                    "needs_review": False
                }
            else:
                # Close but needs manual review
                return {
                    "acceptable": False,
                    "reason": f"Insufficient: received {received_amount}, expected {expected_amount} (${shortfall_usd:.2f} short)",
                    "needs_review": Config.MANUAL_REVIEW_ENABLED
                }
        
        # Case 3: Significantly underpaid
        else:
            return {
                "acceptable": False,
                "reason": f"Significantly underpaid: received {received_amount}, expected {expected_amount}",
                "needs_review": False
            }
    
    
    @staticmethod
    async def _verify_ethereum(
        tx_hash: str, 
        expected_address: str, 
        expected_amount: float, 
        expected_usd_value: float, 
        crypto_currency: str
    ) -> Dict:
        """Verify Ethereum/ERC-20 transaction using Etherscan API"""
        
        api_key = Config.ETHERSCAN_API_KEY or "YourApiKeyToken"
        url = "https://api.etherscan.io/api"
        
        params = {
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash,
            "apikey": api_key
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params, timeout=15.0)
            data = response.json()
            
            if not data.get("result"):
                return {"valid": False, "message": "Transaction not found on Ethereum network"}
            
            tx = data["result"]
            
            # Null result means transaction doesn't exist
            if tx is None:
                return {"valid": False, "message": "Transaction hash not found"}
            
            # Check recipient address
            to_address = tx.get("to", "").lower()
            if to_address != expected_address.lower():
                return {
                    "valid": False, 
                    "message": f"Wrong recipient address: sent to {to_address}",
                    "details": {"expected": expected_address, "received": to_address}
                }
            
            # Convert Wei to ETH
            value_wei = int(tx.get("value", "0"), 16)
            value_eth = value_wei / 1e18
            
            # CHECK AMOUNT WITH TOLERANCE
            amount_check = TransactionVerifier._is_amount_acceptable(
                value_eth, expected_amount, crypto_currency, expected_usd_value
            )
            
            if amount_check["acceptable"]:
                return {
                    "valid": True,
                    "message": f"✅ {amount_check['reason']}",
                    "details": {
                        "received": value_eth,
                        "expected": expected_amount,
                        "from": tx.get("from"),
                        "block": tx.get("blockNumber")
                    }
                }
            
            elif amount_check["needs_review"]:
                return {
                    "valid": False,
                    "message": f"⏳ {amount_check['reason']}",
                    "details": {
                        "manual_review": True,
                        "received": value_eth,
                        "expected": expected_amount,
                        "from": tx.get("from")
                    }
                }
            
            else:
                return {
                    "valid": False,
                    "message": amount_check["reason"],
                    "details": {"received": value_eth, "expected": expected_amount}
                }
    
    
    @staticmethod
    async def _verify_bitcoin(
        tx_hash: str, 
        expected_address: str, 
        expected_amount: float, 
        expected_usd_value: float
    ) -> Dict:
        """Verify Bitcoin transaction using Blockchain.com API"""
        
        url = f"https://blockchain.info/rawtx/{tx_hash}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            
            if response.status_code != 200:
                return {"valid": False, "message": "Bitcoin transaction not found"}
            
            data = response.json()
            
            # Check outputs for our address
            for output in data.get("out", []):
                if output.get("addr") == expected_address:
                    # Convert Satoshi to BTC
                    value_btc = output.get("value", 0) / 1e8
                    
                    # CHECK AMOUNT WITH TOLERANCE
                    amount_check = TransactionVerifier._is_amount_acceptable(
                        value_btc, expected_amount, "btc", expected_usd_value
                    )
                    
                    if amount_check["acceptable"]:
                        return {
                            "valid": True,
                            "message": f"✅ {amount_check['reason']}",
                            "details": {
                                "received": value_btc,
                                "expected": expected_amount,
                                "confirmations": data.get("block_height", 0)
                            }
                        }
                    
                    elif amount_check["needs_review"]:
                        return {
                            "valid": False,
                            "message": f"⏳ {amount_check['reason']}",
                            "details": {
                                "manual_review": True,
                                "received": value_btc,
                                "expected": expected_amount
                            }
                        }
                    
                    else:
                        return {
                            "valid": False,
                            "message": amount_check["reason"],
                            "details": {"received": value_btc, "expected": expected_amount}
                        }
            
            return {
                "valid": False, 
                "message": f"Payment not sent to your address: {expected_address}",
                "details": {}
            }
    
    
    @staticmethod
    async def _verify_tron(
        tx_hash: str, 
        expected_address: str, 
        expected_amount: float, 
        expected_usd_value: float
    ) -> Dict:
        """Verify Tron/USDT-TRC20 transaction using TronGrid API"""
        
        url = f"https://api.trongrid.io/v1/transactions/{tx_hash}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            
            if response.status_code != 200:
                return {"valid": False, "message": "Tron transaction not found"}
            
            data = response.json()
            
            # Check transaction success
            if not data.get("ret") or data["ret"][0].get("contractRet") != "SUCCESS":
                return {"valid": False, "message": "Transaction failed or not confirmed"}
            
            # For USDT TRC-20, we need to parse contract data
            # This is simplified - production needs full TRC-20 parsing
            contract_data = data.get("raw_data", {}).get("contract", [{}])[0]
            contract_type = contract_data.get("type")
            
            # Basic validation (amount checking for USDT requires decoding contract parameters)
            if contract_type in ["TransferContract", "TriggerSmartContract"]:
                logger.info(f"Tron transaction found: {tx_hash}")
                
                # SIMPLIFIED: Auto-approve Tron transactions
                # In production, decode the smart contract call to verify USDT amount
                return {
                    "valid": True,
                    "message": "⚠️ Tron transaction found (amount verification requires manual review)",
                    "details": {
                        "manual_review": True,  # Always review USDT manually
                        "tx_id": data.get("txID"),
                        "note": "USDT amount must be verified manually"
                    }
                }
            
            return {"valid": False, "message": "Invalid Tron transaction type"}
    
    
    @staticmethod
    async def _verify_litecoin(
        tx_hash: str, 
        expected_address: str, 
        expected_amount: float, 
        expected_usd_value: float
    ) -> Dict:
        """Verify Litecoin transaction using BlockCypher API"""
        
        url = f"https://api.blockcypher.com/v1/ltc/main/txs/{tx_hash}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=15.0)
            
            if response.status_code != 200:
                return {"valid": False, "message": "Litecoin transaction not found"}
            
            data = response.json()
            
            # Check outputs for our address
            for output in data.get("outputs", []):
                if expected_address in output.get("addresses", []):
                    # Convert Litoshi to LTC
                    value_ltc = output.get("value", 0) / 1e8
                    
                    # CHECK AMOUNT WITH TOLERANCE
                    amount_check = TransactionVerifier._is_amount_acceptable(
                        value_ltc, expected_amount, "ltc", expected_usd_value
                    )
                    
                    if amount_check["acceptable"]:
                        return {
                            "valid": True,
                            "message": f"✅ {amount_check['reason']}",
                            "details": {
                                "received": value_ltc,
                                "expected": expected_amount,
                                "confirmations": data.get("confirmations", 0)
                            }
                        }
                    
                    elif amount_check["needs_review"]:
                        return {
                            "valid": False,
                            "message": f"⏳ {amount_check['reason']}",
                            "details": {
                                "manual_review": True,
                                "received": value_ltc,
                                "expected": expected_amount
                            }
                        }
                    
                    else:
                        return {
                            "valid": False,
                            "message": amount_check["reason"],
                            "details": {"received": value_ltc, "expected": expected_amount}
                        }
            
            return {
                "valid": False, 
                "message": f"Payment not sent to your address: {expected_address}",
                "details": {}
            }
