"""Secure token management for dashboard and payment access"""
import hashlib
import secrets
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TokenManager:
    """Manage secure access tokens"""
    
    SALT = "your-secret-salt-change-this-in-production"  # Move to .env
    
    @staticmethod
    def generate_secure_token(user_id: int, username: str, token_type: str = "dashboard") -> str:
        """
        Generate a secure token for user access
        Format: {user_id}:{timestamp}:{random}:{signature}
        """
        timestamp = int(time.time())
        random_part = secrets.token_hex(8)
        
        # Create signature
        data = f"{user_id}:{username}:{timestamp}:{random_part}:{TokenManager.SALT}"
        signature = hashlib.sha256(data.encode()).hexdigest()[:16]
        
        # Combine into token
        token = f"{user_id}:{timestamp}:{random_part}:{signature}"
        return token
    
    @staticmethod
    def verify_token(token: str, username: str, max_age_hours: int = 24) -> Optional[int]:
        """
        Verify token and return user_id if valid
        Returns None if invalid or expired
        """
        try:
            parts = token.split(':')
            if len(parts) != 4:
                return None
            
            user_id, timestamp, random_part, signature = parts
            user_id = int(user_id)
            timestamp = int(timestamp)
            
            # Check expiration
            if time.time() - timestamp > (max_age_hours * 3600):
                logger.warning(f"Token expired for user {user_id}")
                return None
            
            # Verify signature
            data = f"{user_id}:{username}:{timestamp}:{random_part}:{TokenManager.SALT}"
            expected_sig = hashlib.sha256(data.encode()).hexdigest()[:16]
            
            if signature != expected_sig:
                logger.warning(f"Invalid signature for user {user_id}")
                return None
            
            return user_id
            
        except Exception as e:
            logger.exception(f"Token verification error: {e}")
            return None
    
    @staticmethod
    def hash_token(token: str) -> str:
        """Hash token for database storage"""
        return hashlib.sha256(token.encode()).hexdigest()
    
    @staticmethod
    def generate_payment_token(
        user_id: int, 
        plan: str, 
        amount: float
    ) -> str:
        """
        Generate unique payment token
        Format: PAY_{user_id}_{timestamp}_{hash}
        """
        timestamp = int(time.time())
        data = f"{user_id}:{plan}:{amount}:{timestamp}:{secrets.token_hex(16)}"
        hash_part = hashlib.sha256(data.encode()).hexdigest()[:12]
        
        payment_id = f"PAY_{user_id}_{timestamp}_{hash_part}"
        return payment_id.upper()
