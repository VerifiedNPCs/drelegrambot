# health_checker.py
import asyncio
import psutil
import os
import logging
import glob
import time
from datetime import datetime

logger = logging.getLogger("HealthCheck")

class HealthMonitor:
    def __init__(self, data_dir="./data_storage"):
        self.data_dir = data_dir
        self.running = True
        self.history = []

    async def get_system_stats(self):
        """Async-friendly wrapper for blocking psutil calls."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._collect_stats)

    def _collect_stats(self):
        """Blocking resource collection (runs in thread)."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        
        return {
            "cpu_percent": process.cpu_percent(interval=0.1),
            "memory_mb": mem_info.rss / (1024 * 1024),
            "threads": process.num_threads(),
            "open_files": len(process.open_files())
        }

    def check_data_freshness(self):
        """Checks if .npz matrix files are being updated."""
        status = {}
        now = time.time()
        # Check specific Timeframes to ensure the loop isn't stuck
        files = glob.glob(os.path.join(self.data_dir, "matrix_*.npz"))
        
        stale_count = 0
        for f in files:
            try:
                mtime = os.path.getmtime(f)
                age = now - mtime
                # Alert if older than 2 minutes (120s)
                if age > 120:
                    stale_count += 1
                    logger.warning(f"⚠️ STALE DATA: {os.path.basename(f)} is {age:.1f}s old")
            except Exception:
                pass
        
        return stale_count

    async def start_monitoring(self, interval=60):
        """Main async loop to run inside Telegram Bot."""
        logger.info("🏥 Health Monitor Task Started")
        
        while self.running:
            try:
                # 1. System Stats
                stats = await self.get_system_stats()
                
                # 2. Data Freshness
                stale_files = self.check_data_freshness()
                
                # 3. Log Report
                log_msg = (
                    f"Health Report | "
                    f"CPU: {stats['cpu_percent']:.1f}% | "
                    f"RAM: {stats['memory_mb']:.0f}MB | "
                    f"Threads: {stats['threads']} | "
                    f"Stale Files: {stale_files}"
                )
                
                # Critical Warnings
                if stats['memory_mb'] > 1500: # 1.5 GB limit
                    logger.critical(f"❌ HIGH MEMORY USAGE: {stats['memory_mb']:.0f}MB")
                    # Optional: Trigger a garbage collection explicitly
                    import gc
                    gc.collect()

                if stats['memory_mb'] > 300: 
                    logger.info(log_msg)

            except Exception as e:
                logger.error(f"Health check failed: {e}")

            await asyncio.sleep(interval)
