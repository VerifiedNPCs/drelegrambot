import asyncio
import aiosqlite
from dataclasses import dataclass
from typing import Optional, Set
from config import config

@dataclass
class SubscriptionStore:
    db_path: str
    lock: asyncio.Lock = asyncio.Lock()

    async def ensure_table(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY
                )
            """)
            await db.commit()

    async def load(self) -> Set[int]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT chat_id FROM chats") as cursor:
                rows = await cursor.fetchall()
                return set(row[0] for row in rows)

    async def add(self, chat_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT OR IGNORE INTO chats (chat_id) VALUES (?)", (chat_id,))
            await db.commit()

    async def remove(self, chat_id: int):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM chats WHERE chat_id = ?", (chat_id,))
            await db.commit()

@dataclass
class UserSettingsStore:
    db_path: str
    lock: asyncio.Lock = asyncio.Lock()

    async def ensure_table(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    user_id INTEGER PRIMARY KEY,
                    default_tf TEXT
                )
            """)
            await db.commit()

    async def set_default_tf(self, user_id: int, tf: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR REPLACE INTO settings (user_id, default_tf) VALUES (?, ?)",
                (user_id, tf)
            )
            await db.commit()

    async def get_default_tf(self, user_id: int) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT default_tf FROM settings WHERE user_id = ?",
                (user_id,)
            ) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else '1h'

# Global Instances to be imported by other files
SUBS = SubscriptionStore(config.SUBS_FILE)
USER_SETTINGS = UserSettingsStore(config.USER_SETTINGS_FILE)
