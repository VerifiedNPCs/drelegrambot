import asyncpg
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
import json
import os
from pathlib import Path
from config import Config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages all database operations with connection pooling"""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    # ==========================================
    # Connection Pool Management
    # ==========================================

    async def initialize(self) -> bool:
        """
        Initialize database connection pool and schema
        Returns True if successful, False otherwise
        """
        try:
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                Config.DATABASE_URL,
                min_size=Config.DB_MIN_POOL_SIZE,
                max_size=Config.DB_MAX_POOL_SIZE,
                command_timeout=Config.DB_COMMAND_TIMEOUT
            )
            logger.info("Database connection pool created successfully")

            # Check if database is initialized
            if not await self._is_database_initialized():
                logger.info("Database not initialized. Initializing schema...")
                await self._initialize_schema()
            else:
                logger.info("Database already initialized")

            self._initialized = True
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize database: {e}")
            return False

    async def close(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("Database connection pool closed")
            self._initialized = False

    async def _is_database_initialized(self) -> bool:
        """Check if database tables exist"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'users'
                    );
                """)
                return result
        except Exception as e:
            logger.error(f"Error checking database initialization: {e}")
            return False

    async def _initialize_schema(self):
        """Initialize database schema from schema.sql or create basic schema"""
        try:
            # Try to load from schema.sql file
            schema_path = Path(__file__).parent.parent / "schema.sql"

            if schema_path.exists():
                with open(schema_path, 'r', encoding='utf-8') as f:
                    schema_sql = f.read()
                async with self.pool.acquire() as conn:
                    await conn.execute(schema_sql)
                logger.info("Database schema initialized from schema.sql")
            else:
                logger.warning("schema.sql not found, creating basic schema")
                await self._create_basic_schema()

        except Exception as e:
            logger.exception(f"Failed to initialize schema: {e}")
            raise

    async def _create_basic_schema(self):
        """Create basic schema if schema.sql doesn't exist"""
        schema = """
            -- Users table
            CREATE TABLE IF NOT EXISTS users (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            -- Subscriptions table
            CREATE TABLE IF NOT EXISTS subscriptions (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                plan VARCHAR(50) NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                start_date TIMESTAMP WITH TIME ZONE NOT NULL,
                end_date TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            -- User Tokens (for Dashboard Auth)
            CREATE TABLE IF NOT EXISTS user_tokens (
                id SERIAL PRIMARY KEY,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                token_hash VARCHAR(255) NOT NULL,
                token_type VARCHAR(50) DEFAULT 'dashboard',
                is_active BOOLEAN DEFAULT TRUE,
                last_used_at TIMESTAMP WITH TIME ZONE,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            -- Payment Requests
            CREATE TABLE IF NOT EXISTS payment_requests (
                id SERIAL PRIMARY KEY,
                payment_id VARCHAR(100) NOT NULL UNIQUE,
                user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                plan VARCHAR(50) NOT NULL,
                amount DECIMAL(10, 2) NOT NULL,
                currency VARCHAR(10) DEFAULT 'USD',
                crypto_currency VARCHAR(10),
                status VARCHAR(20) DEFAULT 'pending', 
                gateway_name VARCHAR(50),
                gateway_order_id VARCHAR(100),
                payment_url TEXT,
                metadata JSONB,
                completed_at TIMESTAMP WITH TIME ZONE,
                expires_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            -- Transactions (History)
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id VARCHAR(100) PRIMARY KEY,
                payment_request_id INT REFERENCES payment_requests(id),
                user_id BIGINT NOT NULL REFERENCES users(user_id),
                amount DECIMAL(10, 2) NOT NULL,
                currency VARCHAR(10) DEFAULT 'USD',
                crypto_amount DECIMAL(18, 8),
                crypto_currency VARCHAR(10),
                gateway_transaction_id VARCHAR(100),
                gateway_name VARCHAR(50),
                tx_hash VARCHAR(255),
                status VARCHAR(20) DEFAULT 'confirmed',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            -- User Wallets
            CREATE TABLE IF NOT EXISTS user_wallets (
                user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                wallet_address VARCHAR(255),
                wallet_type VARCHAR(50) DEFAULT 'usdt',
                is_active BOOLEAN DEFAULT TRUE,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Functions needed for IDs
            CREATE OR REPLACE FUNCTION generate_payment_id() RETURNS text AS $$
            BEGIN
                RETURN 'PAY-' || floor(extract(epoch from clock_timestamp()) * 1000)::text;
            END;
            $$ LANGUAGE plpgsql;

            CREATE OR REPLACE FUNCTION generate_transaction_id() RETURNS text AS $$
            BEGIN
                RETURN 'TX-' || floor(extract(epoch from clock_timestamp()) * 1000)::text;
            END;
            $$ LANGUAGE plpgsql;
        """

        async with self.pool.acquire() as conn:
            await conn.execute(schema)
        logger.info("Basic schema created successfully (Users, Subs, Payments, Tokens)")


    # ==========================================
    # User CRUD Operations
    # ==========================================

    async def create_user(self, user_id: int, username: str, email: str) -> Optional[Dict[str, Any]]:
        """
        Create a new user or update existing one
        Returns user dict if successful, None otherwise
        """
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    INSERT INTO users (user_id, username, email)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (user_id) DO UPDATE
                    SET username = EXCLUDED.username,
                        email = EXCLUDED.email,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING *
                """, user_id, username, email)

                logger.info(f"User created/updated: {username} ({user_id})")
                return dict(row) if row else None

        except asyncpg.UniqueViolationError:
            logger.warning(f"Email {email} already exists for different user")
            return None
        except Exception as e:
            logger.exception(f"Error creating user: {e}")
            return None

    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM users WHERE user_id = $1", user_id)
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error fetching user {user_id}: {e}")
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM users WHERE email = $1", email)
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error fetching user by email: {e}")
            return None

    async def update_user(self, user_id: int, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Update user fields dynamically
        Example: update_user(123, username='new_name', email='new@email.com')
        """
        if not kwargs:
            return await self.get_user(user_id)

        # Build dynamic UPDATE query
        fields = []
        values = []
        param_count = 1

        for key, value in kwargs.items():
            if key in ['username', 'email']:
                fields.append(f"{key} = ${param_count}")
                values.append(value)
                param_count += 1

        if not fields:
            return await self.get_user(user_id)

        values.append(user_id)
        query = f"""
            UPDATE users 
            SET {', '.join(fields)}, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ${param_count}
            RETURNING *
        """

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow(query, *values)
                logger.info(f"User {user_id} updated")
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error updating user: {e}")
            return None

    async def delete_user(self, user_id: int) -> bool:
        """
        Delete user (cascades to subscriptions)
        Returns True if successful
        """
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("DELETE FROM users WHERE user_id = $1", user_id)
                deleted = result.split()[-1] == '1'
                if deleted:
                    logger.info(f"User {user_id} deleted")
                return deleted
        except Exception as e:
            logger.exception(f"Error deleting user: {e}")
            return False

    async def get_all_users(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """Get all users with pagination"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT * FROM users ORDER BY created_at DESC LIMIT $1 OFFSET $2",
                    limit, offset
                )
                return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Error fetching users: {e}")
            return []

    async def count_users(self) -> int:
        """Get total user count"""
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval("SELECT COUNT(*) FROM users")
                return count or 0
        except Exception as e:
            logger.exception(f"Error counting users: {e}")
            return 0

    # ==========================================
    # Subscription CRUD Operations
    # ==========================================

    async def create_subscription(
        self,
        user_id: int,
        plan: str,
        duration_days: int = 30
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new subscription (cancels existing active ones)
        Returns subscription dict if successful
        """
        try:
            start_date = datetime.now(timezone.utc)
            end_date = start_date + timedelta(days=duration_days)

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Cancel existing active subscriptions
                    await conn.execute("""
                        UPDATE subscriptions 
                        SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = $1 AND status = 'active'
                    """, user_id)

                    # Create new subscription
                    row = await conn.fetchrow("""
                        INSERT INTO subscriptions (user_id, plan, status, start_date, end_date)
                        VALUES ($1, $2, 'active', $3, $4)
                        RETURNING *
                    """, user_id, plan, start_date, end_date)

                    logger.info(f"Subscription created for user {user_id}: {plan}")
                    return dict(row) if row else None

        except Exception as e:
            logger.exception(f"Error creating subscription: {e}")
            return None

    async def get_subscription(self, subscription_id: int) -> Optional[Dict[str, Any]]:
        """Get subscription by ID"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("SELECT * FROM subscriptions WHERE id = $1", subscription_id)
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error fetching subscription: {e}")
            return None
        
    async def cancel_subscription(self, user_id: int) -> bool:
        """
        Cancels the user's active subscription and logs the action.
        Returns True if a subscription was cancelled, False if no active sub found.
        """
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # 1. Update the subscription status
                # We only cancel 'active' subscriptions
                result = await conn.execute(
                    """
                    UPDATE subscriptions 
                    SET status = 'cancelled' 
                    WHERE user_id = $1 AND status = 'active'
                    """,
                    user_id
                )
                
                # Check if a row was actually updated
                if result == "UPDATE 0":
                    return False

                # 2. Add to Audit Log
                await conn.execute(
                    """
                    INSERT INTO audit_logs (user_id, action_type, action_details)
                    VALUES ($1, 'subscription_cancelled', 'User cancelled plan via Telegram Bot')
                    """,
                    user_id
                )
                
                return True

    async def get_active_subscription(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user's active subscription"""
        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT * FROM subscriptions
                    WHERE user_id = $1 
                    AND status = 'active'
                    AND end_date > NOW()
                    ORDER BY created_at DESC
                    LIMIT 1
                """, user_id)
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error fetching active subscription: {e}")
            return None

    async def get_user_subscriptions(
        self,
        user_id: int,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all subscriptions for a user"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT * FROM subscriptions
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2
                """, user_id, limit)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Error fetching user subscriptions: {e}")
            return []

    async def update_subscription_status(
        self,
        subscription_id: int,
        status: str
    ) -> Optional[Dict[str, Any]]:
        """Update subscription status"""
        if status not in ['active', 'expired', 'cancelled']:
            logger.error(f"Invalid status: {status}")
            return None

        try:
            async with self.pool.acquire() as conn:
                row = await conn.fetchrow("""
                    UPDATE subscriptions
                    SET status = $1, updated_at = CURRENT_TIMESTAMP
                    WHERE id = $2
                    RETURNING *
                """, status, subscription_id)

                logger.info(f"Subscription {subscription_id} status updated to {status}")
                return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Error updating subscription status: {e}")
            return None

    async def cancel_subscription(self, subscription_id: int) -> bool:
        """Cancel a subscription"""
        result = await self.update_subscription_status(subscription_id, 'cancelled')
        return result is not None

    async def delete_subscription(self, subscription_id: int) -> bool:
        """Delete a subscription"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("DELETE FROM subscriptions WHERE id = $1", subscription_id)
                deleted = result.split()[-1] == '1'
                if deleted:
                    logger.info(f"Subscription {subscription_id} deleted")
                return deleted
        except Exception as e:
            logger.exception(f"Error deleting subscription: {e}")
            return False

    async def get_expiring_subscriptions(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get subscriptions expiring within specified days"""
        try:
            async with self.pool.acquire() as conn:
                rows = await conn.fetch("""
                    SELECT s.*, u.username, u.email
                    FROM subscriptions s
                    JOIN users u ON s.user_id = u.user_id
                    WHERE s.status = 'active'
                    AND s.end_date <= NOW() + INTERVAL '1 day' * $1
                    AND s.end_date > NOW()
                    ORDER BY s.end_date ASC
                """, days)
                return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Error fetching expiring subscriptions: {e}")
            return []

    async def expire_subscriptions(self) -> int:
        """Mark expired subscriptions as expired. Returns count of expired"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute("""
                    UPDATE subscriptions
                    SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                    WHERE status = 'active' AND end_date <= NOW()
                """)
                count = int(result.split()[-1]) if result else 0
                if count > 0:
                    logger.info(f"Marked {count} subscriptions as expired")
                return count
        except Exception as e:
            logger.exception(f"Error expiring subscriptions: {e}")
            return 0

    # ==========================================
    # Helper Methods
    # ==========================================

    async def get_days_left(self, user_id: int) -> Optional[int]:
        """Calculate days left in active subscription"""
        subscription = await self.get_active_subscription(user_id)
        if not subscription:
            return None

        end_date = subscription['end_date']
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        now = datetime.now(timezone.utc)
        if end_date > now:
            delta = end_date - now
            return delta.days
        return 0

    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self.pool.acquire() as conn:
                stats = {}

                # Total users
                stats['total_users'] = await conn.fetchval("SELECT COUNT(*) FROM users") or 0

                # Active subscriptions
                stats['active_subscriptions'] = await conn.fetchval(
                    "SELECT COUNT(*) FROM subscriptions WHERE status = 'active' AND end_date > NOW()"
                ) or 0

                # Expired subscriptions
                stats['expired_subscriptions'] = await conn.fetchval(
                    "SELECT COUNT(*) FROM subscriptions WHERE status = 'expired'"
                ) or 0

                # Cancelled subscriptions
                stats['cancelled_subscriptions'] = await conn.fetchval(
                    "SELECT COUNT(*) FROM subscriptions WHERE status = 'cancelled'"
                ) or 0

                # Subscriptions by plan
                plan_stats = await conn.fetch("""
                    SELECT plan, COUNT(*) as count
                    FROM subscriptions
                    WHERE status = 'active'
                    GROUP BY plan
                """)
                stats['subscriptions_by_plan'] = {row['plan']: row['count'] for row in plan_stats}

                return stats

        except Exception as e:
            logger.exception(f"Error fetching statistics: {e}")
            return {}

    # ==========================================
    # Backup and Restore
    # ==========================================

    async def backup_to_json(self, filename: Optional[str] = None) -> Optional[str]:
        """
        Backup database to JSON file
        Returns backup file path if successful
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backup_{timestamp}.json"

        backup_path = os.path.join(Config.BACKUP_DIR, filename)

        try:
            async with self.pool.acquire() as conn:
                # Fetch all users
                users = await conn.fetch("SELECT * FROM users ORDER BY user_id")

                # Fetch all subscriptions
                subscriptions = await conn.fetch("SELECT * FROM subscriptions ORDER BY id")

                # Convert to dict
                backup_data = {
                    "backup_date": datetime.now(timezone.utc).isoformat(),
                    "users": [dict(row) for row in users],
                    "subscriptions": [dict(row) for row in subscriptions]
                }

                # Serialize dates
                for user in backup_data['users']:
                    for key, value in user.items():
                        if isinstance(value, datetime):
                            user[key] = value.isoformat()

                for sub in backup_data['subscriptions']:
                    for key, value in sub.items():
                        if isinstance(value, datetime):
                            sub[key] = value.isoformat()

                # Write to file
                with open(backup_path, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, indent=2, ensure_ascii=False)

                logger.info(f"Database backed up to {backup_path}")
                return backup_path

        except Exception as e:
            logger.exception(f"Error backing up database: {e}")
            return None

    async def restore_from_json(self, backup_path: str, clear_existing: bool = False) -> bool:
        """
        Restore database from JSON backup
        If clear_existing=True, clears existing data first
        """
        try:
            if not os.path.exists(backup_path):
                logger.error(f"Backup file not found: {backup_path}")
                return False

            # Load backup data
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)

            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    if clear_existing:
                        logger.info("Clearing existing data...")
                        await conn.execute("TRUNCATE TABLE subscriptions, users CASCADE")

                    # Restore users
                    logger.info(f"Restoring {len(backup_data['users'])} users...")
                    for user in backup_data['users']:
                        await conn.execute("""
                            INSERT INTO users (user_id, username, email, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5)
                            ON CONFLICT (user_id) DO UPDATE
                            SET username = EXCLUDED.username,
                                email = EXCLUDED.email,
                                updated_at = EXCLUDED.updated_at
                        """,
                            user['user_id'],
                            user['username'],
                            user['email'],
                            datetime.fromisoformat(user['created_at']),
                            datetime.fromisoformat(user['updated_at'])
                        )

                    # Restore subscriptions
                    logger.info(f"Restoring {len(backup_data['subscriptions'])} subscriptions...")
                    for sub in backup_data['subscriptions']:
                        await conn.execute("""
                            INSERT INTO subscriptions 
                                (id, user_id, plan, status, start_date, end_date, created_at, updated_at)
                            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                            ON CONFLICT (id) DO UPDATE
                            SET user_id = EXCLUDED.user_id,
                                plan = EXCLUDED.plan,
                                status = EXCLUDED.status,
                                start_date = EXCLUDED.start_date,
                                end_date = EXCLUDED.end_date,
                                updated_at = EXCLUDED.updated_at
                        """,
                            sub['id'],
                            sub['user_id'],
                            sub['plan'],
                            sub['status'],
                            datetime.fromisoformat(sub['start_date']),
                            datetime.fromisoformat(sub['end_date']),
                            datetime.fromisoformat(sub['created_at']),
                            datetime.fromisoformat(sub['updated_at'])
                        )

            logger.info(f"Database restored from {backup_path}")
            return True

        except Exception as e:
            logger.exception(f"Error restoring database: {e}")
            return False

    async def list_backups(self) -> List[Dict[str, Any]]:
        """List all backup files"""
        try:
            backup_dir = Path(Config.BACKUP_DIR)
            if not backup_dir.exists():
                return []

            backups = []
            for file in backup_dir.glob("backup_*.json"):
                stats = file.stat()
                backups.append({
                    'filename': file.name,
                    'path': str(file),
                    'size_mb': round(stats.st_size / (1024 * 1024), 2),
                    'created': datetime.fromtimestamp(stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                })

            backups.sort(key=lambda x: x['created'], reverse=True)
            return backups

        except Exception as e:
            logger.exception(f"Error listing backups: {e}")
            return []
        
    # ==========================================
    # TOKEN MANAGEMENT
    # ==========================================
    
    async def create_user_token(
        self, 
        user_id: int, 
        token_hash: str, 
        token_type: str = "dashboard",
        expires_hours: int = 24
    ) -> Optional[Dict]:
        """Create a new access token"""
        try:
            expires_at = datetime.now() + timedelta(hours=expires_hours)
            
            query = """
                INSERT INTO user_tokens (user_id, token_hash, token_type, expires_at)
                VALUES ($1, $2, $3, $4)
                RETURNING id, token_hash, expires_at, created_at
            """
            
            row = await self.pool.fetchrow(query, user_id, token_hash, token_type, expires_at)
            return dict(row) if row else None
            
        except Exception as e:
            logger.exception(f"Failed to create token: {e}")
            return None
    
    async def verify_user_token(self, token_hash: str) -> Optional[Dict]:
        """Verify token and return user data if valid"""
        try:
            query = """
                SELECT ut.*, u.username, u.email
                FROM user_tokens ut
                JOIN users u ON ut.user_id = u.user_id
                WHERE ut.token_hash = $1 
                  AND ut.is_active = TRUE 
                  AND ut.expires_at > CURRENT_TIMESTAMP
            """
            
            row = await self.pool.fetchrow(query, token_hash)
            
            if row:
                # Update last_used_at
                await self.pool.execute(
                    "UPDATE user_tokens SET last_used_at = CURRENT_TIMESTAMP WHERE token_hash = $1",
                    token_hash
                )
                return dict(row)
            
            return None
            
        except Exception as e:
            logger.exception(f"Token verification failed: {e}")
            return None
    
    # ==========================================
    # PAYMENT MANAGEMENT
    # ==========================================
    
    async def create_payment_request(
        self,
        user_id: int,
        plan: str,
        amount: float,
        currency: str = "USD",
        crypto_currency: Optional[str] = None,
        gateway_name: Optional[str] = None,
        expires_minutes: int = 30
    ) -> Optional[Dict]:
        """Create a new payment request"""
        try:
            payment_id = await self.pool.fetchval("SELECT generate_payment_id()")
            expires_at = datetime.now() + timedelta(minutes=expires_minutes)
            
            query = """
                INSERT INTO payment_requests 
                (payment_id, user_id, plan, amount, currency, crypto_currency, gateway_name, expires_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                RETURNING id, payment_id, user_id, plan, amount, status, created_at, expires_at
            """
            
            row = await self.pool.fetchrow(
                query, payment_id, user_id, plan, amount, currency, 
                crypto_currency, gateway_name, expires_at
            )
            
            return dict(row) if row else None
            
        except Exception as e:
            logger.exception(f"Failed to create payment request: {e}")
            return None
    
    async def update_payment_request(
        self,
        payment_id: str,
        status: str,
        gateway_order_id: Optional[str] = None,
        payment_url: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """Update payment request status"""
        try:
            query = """
                UPDATE payment_requests
                SET status = $2,
                    gateway_order_id = COALESCE($3, gateway_order_id),
                    payment_url = COALESCE($4, payment_url),
                    metadata = COALESCE($5::jsonb, metadata),
                    completed_at = CASE WHEN $2 = 'completed' THEN CURRENT_TIMESTAMP ELSE completed_at END
                WHERE payment_id = $1
            """
            
            await self.pool.execute(query, payment_id, status, gateway_order_id, payment_url, metadata)
            return True
            
        except Exception as e:
            logger.exception(f"Failed to update payment request: {e}")
            return False
    
    async def get_payment_request(self, payment_id: str) -> Optional[Dict]:
        """Get payment request by ID"""
        try:
            query = "SELECT * FROM payment_requests WHERE payment_id = $1"
            row = await self.pool.fetchrow(query, payment_id)
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Failed to get payment request: {e}")
            return None
    
    async def create_transaction(
        self,
        user_id: int,
        payment_request_id: int,
        amount: float,
        currency: str = "USD",
        crypto_amount: Optional[float] = None,
        crypto_currency: Optional[str] = None,
        gateway_transaction_id: Optional[str] = None,
        gateway_name: Optional[str] = None,
        tx_hash: Optional[str] = None
    ) -> Optional[Dict]:
        """Create a transaction record"""
        try:
            transaction_id = await self.pool.fetchval("SELECT generate_transaction_id()")
            
            query = """
                INSERT INTO transactions 
                (transaction_id, payment_request_id, user_id, amount, currency, 
                 crypto_amount, crypto_currency, gateway_transaction_id, gateway_name, tx_hash, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'confirmed')
                RETURNING *
            """
            
            row = await self.pool.fetchrow(
                query, transaction_id, payment_request_id, user_id, amount, currency,
                crypto_amount, crypto_currency, gateway_transaction_id, gateway_name, tx_hash
            )
            
            return dict(row) if row else None
            
        except Exception as e:
            logger.exception(f"Failed to create transaction: {e}")
            return None
    
    async def get_user_transactions(self, user_id: int, limit: int = 50) -> list:
        """Get user transaction history"""
        try:
            query = """
                SELECT * FROM user_payment_history 
                WHERE user_id = $1 
                ORDER BY payment_date DESC 
                LIMIT $2
            """
            rows = await self.pool.fetch(query, user_id, limit)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Failed to get transactions: {e}")
            return []
    
    # ==========================================
    # WALLET MANAGEMENT
    # ==========================================
    
    async def create_user_wallet(
        self,
        user_id: int,
        wallet_address: str,
        wallet_type: str = "usdt"
    ) -> Optional[Dict]:
        """Create user wallet"""
        try:
            query = """
                INSERT INTO user_wallets (user_id, wallet_address, wallet_type)
                VALUES ($1, $2, $3)
                ON CONFLICT (user_id) DO UPDATE 
                SET wallet_address = $2, wallet_type = $3
                RETURNING *
            """
            row = await self.pool.fetchrow(query, user_id, wallet_address, wallet_type)
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Failed to create wallet: {e}")
            return None
    
    async def get_user_wallet(self, user_id: int) -> Optional[Dict]:
        """Get user wallet"""
        try:
            query = "SELECT * FROM user_wallets WHERE user_id = $1 AND is_active = TRUE"
            row = await self.pool.fetchrow(query, user_id)
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Failed to get wallet: {e}")
            return None
    
    # ==========================================
    # NOTIFICATIONS
    # ==========================================
    
    async def create_notification(
        self,
        user_id: int,
        notification_type: str,
        title: str,
        message: str,
        priority: str = "normal"
    ) -> Optional[Dict]:
        """Create a notification"""
        try:
            query = """
                INSERT INTO notifications (user_id, notification_type, title, message, priority)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING *
            """
            row = await self.pool.fetchrow(query, user_id, notification_type, title, message, priority)
            return dict(row) if row else None
        except Exception as e:
            logger.exception(f"Failed to create notification: {e}")
            return None
    
    async def get_user_notifications(self, user_id: int, unread_only: bool = False) -> list:
        """Get user notifications"""
        try:
            if unread_only:
                query = """
                    SELECT * FROM notifications 
                    WHERE user_id = $1 AND is_read = FALSE 
                    ORDER BY created_at DESC 
                    LIMIT 50
                """
            else:
                query = """
                    SELECT * FROM notifications 
                    WHERE user_id = $1 
                    ORDER BY created_at DESC 
                    LIMIT 100
                """
            
            rows = await self.pool.fetch(query, user_id)
            return [dict(row) for row in rows]
        except Exception as e:
            logger.exception(f"Failed to get notifications: {e}")
            return []
    
    async def mark_notification_read(self, notification_id: int) -> bool:
        """Mark notification as read"""
        try:
            await self.pool.execute(
                "UPDATE notifications SET is_read = TRUE, read_at = CURRENT_TIMESTAMP WHERE id = $1",
                notification_id
            )
            return True
        except Exception as e:
            logger.exception(f"Failed to mark notification read: {e}")
            return False
