"""FastAPI backend for dashboard and payment"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import hmac
import hashlib
from urllib.parse import parse_qs
import json

from config import Config
from database import DatabaseManager
from token_manager import TokenManager
from contextlib import asynccontextmanager
from crypto_manager import CryptoManager
from transaction_verifier import TransactionVerifier

logger = logging.getLogger(__name__)

# Initialize FastAPI
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    global db_manager
    db_manager = DatabaseManager()
    await db_manager.initialize()
    logger.info("=" * 50)
    logger.info("API Server Started")
    logger.info(f"Database: Connected")
    logger.info(f"Listening on: http://{Config.API_HOST}:{Config.API_PORT}")
    logger.info("=" * 50)
    
    yield
    
    # Shutdown
    if db_manager:
        await db_manager.close()
    logger.info("API server shutdown")


# Update app initialization
app = FastAPI(
    title="Drelegram Dashboard API", 
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        Config.FRONTEND_ORIGIN
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database manager instance
db_manager = None

# ==========================================
# TELEGRAM WEBAPP AUTHENTICATION
# ==========================================

def verify_telegram_webapp_data(init_data: str, bot_token: str) -> dict:
    """
    Verify Telegram WebApp initData
    https://core.telegram.org/bots/webapps#validating-data-received-via-the-web-app
    """
    try:
        parsed = parse_qs(init_data)
        hash_value = parsed.get('hash', [None])[0]
        
        if not hash_value:
            return None
        
        # Remove hash from data
        data_check_string_parts = []
        for key in sorted(parsed.keys()):
            if key != 'hash':
                values = parsed[key]
                for value in values:
                    data_check_string_parts.append(f"{key}={value}")
        
        data_check_string = '\n'.join(data_check_string_parts)
        
        # Create secret key
        secret_key = hmac.new(
            "WebAppData".encode(),
            bot_token.encode(),
            hashlib.sha256
        ).digest()
        
        # Calculate hash
        calculated_hash = hmac.new(
            secret_key,
            data_check_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if calculated_hash != hash_value:
            return None
        
        # Parse user data
        user_json = parsed.get('user', [None])[0]
        if user_json:
            user_data = json.loads(user_json)
            return user_data
        
        return None
        
    except Exception as e:
        logger.exception(f"Telegram data verification failed: {e}")
        return None


async def verify_token(authorization: str = Header(None)) -> dict:
    """Dependency to verify authentication token"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        token = authorization.replace("Bearer ", "")
        token_hash = TokenManager.hash_token(token)
        
        token_data = await db_manager.verify_user_token(token_hash)
        
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        return token_data
        
    except Exception as e:
        logger.exception(f"Token verification failed: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")


# ==========================================
# RESPONSE MODELS
# ==========================================

class UserResponse(BaseModel):
    user_id: int
    username: str
    email: str
    created_at: str


class SubscriptionResponse(BaseModel):
    id: int
    plan: str
    status: str
    start_date: str
    end_date: str
    days_remaining: Optional[int]


class TransactionResponse(BaseModel):
    transaction_id: str
    amount: float
    currency: str
    crypto_currency: Optional[str]
    status: str
    created_at: str
    tx_hash: Optional[str]


class NotificationResponse(BaseModel):
    id: int
    notification_type: str
    title: str
    message: str
    is_read: bool
    priority: str
    created_at: str


class WalletResponse(BaseModel):
    wallet_address: str
    wallet_type: str
    balance: float
    is_verified: bool


# ==========================================
# AUTHENTICATION ENDPOINTS
# ==========================================

@app.post("/api/auth/login")
async def login_user(credentials: dict):
    """
    Regular login for dashboard access
    Body: {"user_id": 123456789, "token": "secure_token_from_bot"}
    """
    user_id = credentials.get("user_id")
    provided_token = credentials.get("token")
    
    if not user_id or not provided_token:
        raise HTTPException(status_code=400, detail="Missing credentials")
    
    # Verify the token from bot
    token_hash = TokenManager.hash_token(provided_token)
    token_data = await db_manager.verify_user_token(token_hash)
    
    if not token_data:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    # Check if user exists
    db_user = await db_manager.get_user(user_id)
    
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Generate new session token
    session_token = TokenManager.generate_secure_token(user_id, db_user['username'], "dashboard")
    session_hash = TokenManager.hash_token(session_token)
    
    await db_manager.create_user_token(user_id, session_hash, "dashboard", expires_hours=24)
    
    return {
        "token": session_token,
        "user_id": user_id,
        "username": db_user['username'],
        "email": db_user['email']
    }


@app.post("/api/auth/telegram-payment")
async def authenticate_telegram_payment(init_data: dict):
    init_data_string = init_data.get("initData")
    if not init_data_string:
        raise HTTPException(400, "Missing initData")
    
    user_data = verify_telegram_webapp_data(init_data_string, Config.BOT_TOKEN)
    if not user_data:
        raise HTTPException(401, "Invalid Telegram data")
    
    user_id = user_data.get("id")
    username = user_data.get("username", f"user_{user_id}")
    
    db_user = await db_manager.get_user(user_id)
    if not db_user:
        raise HTTPException(404, "User not found")
    
    token = TokenManager.generate_secure_token(user_id, username, "payment")
    token_hash = TokenManager.hash_token(token)
    await db_manager.create_user_token(user_id, token_hash, "payment", expires_hours=1)
    
    return {
        "token": token,
        "user_id": user_id,
        "username": username,
        "email": db_user['email']
    }

# ==========================================
# USER ENDPOINTS
# ==========================================

@app.get("/api/user/me")
async def get_current_user(token_data: dict = Depends(verify_token)):
    """Get current user information"""
    user_id = token_data['user_id']
    user = await db_manager.get_user(user_id)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user['user_id'],
        "username": user['username'],
        "email": user['email'],
        "created_at": user['created_at'].isoformat()
    }


@app.get("/api/user/subscription")
async def get_user_subscription(token_data: dict = Depends(verify_token)):
    """Get active subscription"""
    user_id = token_data['user_id']
    subscription = await db_manager.get_active_subscription(user_id)
    
    if not subscription:
        return {"subscription": None}
    
    days_left = await db_manager.get_days_left(user_id)
    
    return {
        "subscription": {
            "id": subscription['id'],
            "plan": subscription['plan'],
            "status": subscription['status'],
            "start_date": subscription['start_date'].isoformat(),
            "end_date": subscription['end_date'].isoformat(),
            "days_remaining": days_left
        }
    }


@app.get("/api/user/transactions")
async def get_user_transactions(
    limit: int = 50,
    token_data: dict = Depends(verify_token)
):
    """Get transaction history"""
    user_id = token_data['user_id']
    transactions = await db_manager.get_user_transactions(user_id, limit)
    
    return {
        "transactions": [
            {
                "transaction_id": tx['transaction_id'],
                "amount": float(tx['amount']),
                "currency": tx['currency'],
                "crypto_currency": tx.get('crypto_currency'),
                "crypto_amount": float(tx['crypto_amount']) if tx.get('crypto_amount') else None,
                "status": tx['status'],
                "created_at": tx['payment_date'].isoformat(),
                "tx_hash": tx.get('tx_hash'),
                "plan": tx.get('plan')
            }
            for tx in transactions
        ]
    }


@app.get("/api/user/notifications")
async def get_user_notifications(
    unread_only: bool = False,
    token_data: dict = Depends(verify_token)
):
    """Get user notifications"""
    user_id = token_data['user_id']
    notifications = await db_manager.get_user_notifications(user_id, unread_only)
    
    return {
        "notifications": [
            {
                "id": notif['id'],
                "notification_type": notif['notification_type'],
                "title": notif['title'],
                "message": notif['message'],
                "is_read": notif['is_read'],
                "priority": notif['priority'],
                "created_at": notif['created_at'].isoformat()
            }
            for notif in notifications
        ]
    }


@app.post("/api/user/notifications/{notification_id}/read")
async def mark_notification_read(
    notification_id: int,
    token_data: dict = Depends(verify_token)
):
    """Mark notification as read"""
    success = await db_manager.mark_notification_read(notification_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"success": True}


@app.get("/api/user/wallet")
async def get_user_wallet(token_data: dict = Depends(verify_token)):
    """Get user wallet"""
    user_id = token_data['user_id']
    wallet = await db_manager.get_user_wallet(user_id)
    
    if not wallet:
        return {"wallet": None}
    
    return {
        "wallet": {
            "wallet_address": wallet['wallet_address'],
            "wallet_type": wallet['wallet_type'],
            "balance": float(wallet['balance']),
            "is_verified": wallet['is_verified']
        }
    }


# ==========================================
# PAYMENT ENDPOINTS
# ==========================================

@app.post("/api/payment/create")
async def create_payment(
    payment_data: dict,
    token_data: dict = Depends(verify_token)
):
    """Create payment request"""
    user_id = token_data['user_id']
    plan = payment_data.get("plan")
    crypto_currency = payment_data.get("crypto_currency", "USDT")
    
    if not plan:
        raise HTTPException(status_code=400, detail="Missing plan")
    
    plan_info = Config.get_plan(plan)
    if not plan_info:
        raise HTTPException(status_code=400, detail="Invalid plan")
    
    # Extract price
    price = float(plan_info['price'].replace('$', '').split('/')[0])
    
    # Create payment request
    payment_request = await db_manager.create_payment_request(
        user_id=user_id,
        plan=plan,
        amount=price,
        currency="USD",
        crypto_currency=crypto_currency,
        gateway_name="manual",
        expires_minutes=30
    )
    
    if not payment_request:
        raise HTTPException(status_code=500, detail="Failed to create payment")
    
    # Generate payment URL
    payment_url = f"{Config.PAYMENT_URL}?payment_id={payment_request['payment_id']}"
    
    # Update with payment URL
    await db_manager.update_payment_request(
        payment_request['payment_id'],
        status="pending",
        payment_url=payment_url
    )
    
    return {
        "payment_id": payment_request['payment_id'],
        "amount": price,
        "currency": "USD",
        "crypto_currency": crypto_currency,
        "payment_url": payment_url,
        "expires_at": payment_request['expires_at'].isoformat()
    }

@app.get("/api/payment/{payment_id}")
async def get_payment_details(payment_id: str):
    # Fetch details for the UI
    payment = await db_manager.get_payment_request(payment_id)
    if not payment:
        raise HTTPException(404, "Payment not found")
    
    return {
        "id": payment['payment_id'],
        "plan": payment['plan'],
        "amount": float(payment['amount']),
        "coin_count": payment.get('coin_count', 0),
        "userId": str(payment['user_id']),
        "status": payment['status']
    }

@app.post("/api/payment/confirm")
async def confirm_payment(data: dict):
    payment_id = data.get("payment_id")
    tx_hash = data.get("tx_hash")
    
    if not payment_id or not tx_hash:
        raise HTTPException(status_code=400, detail="Missing data")
    
    # 1. Get payment details
    payment_req = await db_manager.get_payment_request(payment_id)
    if not payment_req:
        raise HTTPException(status_code=404, detail="Payment not found")
    
    # 2. Get expected amounts
    usd_amount = float(payment_req['amount'])
    crypto_currency = payment_req['crypto_currency']
    
    # Recalculate crypto amount with current rate (for verification)
    rates = await CryptoManager.get_exchange_rates()
    expected_crypto = CryptoManager.calculate_crypto_amount(usd_amount, rates[crypto_currency])
    
    wallet_address = Config.ADMIN_WALLETS.get(crypto_currency, "")
    
    # 3. VERIFY WITH TOLERANCE
    verification = await TransactionVerifier.verify_transaction(
        tx_hash=tx_hash,
        crypto_currency=crypto_currency,
        expected_address=wallet_address,
        expected_amount=expected_crypto,
        expected_usd_value=usd_amount  # Pass USD value for fee calculation
    )
    
    if not verification["valid"]:
        # Check if it needs manual review
        if verification.get("details", {}).get("manual_review"):
            # Create a pending review record
            await db_manager.create_notification(
                user_id=payment_req['user_id'],
                notification_type="payment_review",
                title="Payment Under Review",
                message=f"Your payment is being manually reviewed. TX: {tx_hash}",
                priority="high"
            )
            return {
                "success": False,
                "message": "Payment requires manual verification. You will be notified within 24 hours.",
                "manual_review": True
            }
        
        raise HTTPException(status_code=400, detail=verification["message"])
    
    # 4. APPROVED - Activate subscription
    logger.info(f"✅ Payment verified: {verification}")
    
    await db_manager.create_transaction(
        user_id=payment_req['user_id'],
        payment_request_id=payment_req['id'],
        amount=usd_amount,
        currency="USD",
        crypto_currency=crypto_currency,
        crypto_amount=expected_crypto,
        tx_hash=tx_hash,
        gateway_name="crypto_verified"
    )
    
    plan_info = Config.get_plan(payment_req['plan'])
    await db_manager.create_subscription(
        user_id=payment_req['user_id'],
        plan=payment_req['plan'],
        duration_days=plan_info['duration_days']
    )
    
    user = await db_manager.get_user(payment_req['user_id'])
    
    dashboard_token = TokenManager.generate_secure_token(
        user_id=payment_req['user_id'], 
        username=user['username'], 
        token_type="dashboard"
    )
    dashboard_token_hash = TokenManager.hash_token(dashboard_token)
    
    await db_manager.create_user_token(
        user_id=payment_req['user_id'],
        token_hash=dashboard_token_hash,
        token_type="dashboard",
        expires_hours=24
    )
    
    logger.info(f"🎫 Dashboard token issued for user {payment_req['user_id']}")
    
    return {
        "success": True,
        "message": "Payment verified and subscription activated",
        "dashboard_token": dashboard_token
    }

@app.get("/api/payment/{payment_id}/crypto-options")
async def get_crypto_options(payment_id: str):
    """
    Returns available crypto payment options with real-time conversion
    and destination wallets.
    """
    # 1. Get the payment request to know the USD amount
    payment = await db_manager.get_payment_request(payment_id)
    if not payment:
        raise HTTPException(status_code=404, detail="Payment not found")
        
    usd_amount = float(payment['amount'])
    
    # 2. Get Real-time Rates
    rates = await CryptoManager.get_exchange_rates()
    
    # 3. Build Response Options
    options = []
    
    # Define assets metadata
    assets_meta = [
        {"id": "btc", "name": "Bitcoin", "network": "Bitcoin Network", "icon": "currency_bitcoin"},
        {"id": "eth", "name": "Ethereum", "network": "ERC-20", "icon": "diamond"},
        {"id": "usdt", "name": "Tether (USDT)", "network": "BSC-20", "icon": "attach_money"},
        {"id": "ltc", "name": "Litecoin", "network": "Litecoin Network", "icon": "monetization_on"},
    ]
    
    for asset in assets_meta:
        ticker = asset['id']
        rate = rates.get(ticker, 0)
        
        if rate > 0:
            crypto_amount = CryptoManager.calculate_crypto_amount(usd_amount, rate)
            
            options.append({
                **asset,
                "rate": rate,
                "crypto_amount": crypto_amount,
                "wallet_address": Config.ADMIN_WALLETS.get(ticker, "")
            })
            
    return {
        "payment_id": payment_id,
        "usd_amount": usd_amount,
        "options": options
    }
# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "drelegram-dashboard-api"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",  # Changed from app to "api:app" string
        host=Config.API_HOST, 
        port=Config.API_PORT, 
        log_level="info",
        reload=True
    )