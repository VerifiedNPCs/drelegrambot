"""Start both API server and Telegram bot"""
import asyncio
import multiprocessing
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_api():
    """Run FastAPI server"""
    import uvicorn
    import os
    from config import Config
    
    # Force use of PORT env var if available (Railway standard)
    port = int(os.getenv("PORT", Config.API_PORT))
    host = os.getenv("API_HOST", Config.API_HOST)

    logger.info(f"🚀 Starting API Server on {host}:{port}...")
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )



def run_bot():
    """Run Telegram bot"""
    import asyncio
    from main import main as bot_main
    
    logger.info("🤖 Starting Telegram Bot...")
    asyncio.run(bot_main())


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🔥 Starting Drelegram Services")
    logger.info("=" * 60)
    
    # Create processes
    api_process = multiprocessing.Process(target=run_api, name="API-Server")
    bot_process = multiprocessing.Process(target=run_bot, name="Telegram-Bot")
    
    try:
        # Start both processes
        api_process.start()
        logger.info("✅ API Server started (PID: %s)", api_process.pid)
        
        # Wait a moment for API to initialize
        import time
        time.sleep(2)
        
        bot_process.start()
        logger.info("✅ Telegram Bot started (PID: %s)", bot_process.pid)
        
        logger.info("=" * 60)
        logger.info("🎉 All services running! Press Ctrl+C to stop.")
        logger.info("=" * 60)
        
        # Wait for both processes
        api_process.join()
        bot_process.join()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Shutting down services...")
        
        if bot_process.is_alive():
            bot_process.terminate()
            bot_process.join(timeout=5)
            logger.info("✅ Telegram Bot stopped")
        
        if api_process.is_alive():
            api_process.terminate()
            api_process.join(timeout=5)
            logger.info("✅ API Server stopped")
        
        logger.info("👋 All services stopped")
        sys.exit(0)
    
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        
        # Cleanup
        if bot_process.is_alive():
            bot_process.terminate()
        if api_process.is_alive():
            api_process.terminate()
        
        sys.exit(1)