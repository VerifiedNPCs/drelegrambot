import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional
from config import config, setup_logging
from health_checker import HealthMonitor
logger = setup_logging("Services")
# Import collector and engine 
try:
    from Data_Collector import BinanceMarketDataCollector
    from market_engine import MarketEngine
    import image_generator 
except ImportError as e:
    logger.critical(f"Critical Import Error: {e}")
    raise

# Global Service Instances
COLLECTOR: Optional[BinanceMarketDataCollector] = None
ENGINE: Optional[MarketEngine] = None
HEALTH: Optional[HealthMonitor] = None
IMG_EXECUTOR = ThreadPoolExecutor(max_workers=4)

async def init_services(app):
    """Initialize global services and background tasks"""
    global COLLECTOR, ENGINE, HEALTH
    
    COLLECTOR = BinanceMarketDataCollector()
    ENGINE = MarketEngine(COLLECTOR)
    HEALTH = HealthMonitor(data_dir=COLLECTOR.output_dir)
    
    # --- START BACKGROUND LOOPS ---
    # We use asyncio.create_task() instead of app.create_task() to avoid 
    # PTBUserWarnings during post_init.
    asyncio.create_task(COLLECTOR.run())
    asyncio.create_task(ENGINE.start_background_task())
    asyncio.create_task(HEALTH.start_monitoring(interval=60))
    
    # Store references in app context if needed later (optional but good practice)
    app.bot_data['collector'] = COLLECTOR
    app.bot_data['engine'] = ENGINE

async def get_symbol_data(tf: str, symbol: str, limit: int = 20) -> Dict:
    """FAST ACCESS to Matrix data. Returns lists suitable for image_generator."""
    if not COLLECTOR or tf not in COLLECTOR.matrices:
        return {}

    matrix = COLLECTOR.matrices[tf]
    idx = matrix.symbol_map.get(symbol)
    if idx is None:
        return {}

    try:
        with matrix.lock:
            # We must copy inside lock to be thread safe
            closes_slice = matrix.closes[idx, -limit:].copy()
            volumes_slice = matrix.volumes[idx, -limit:].copy()
            latest_close = float(matrix.closes[idx, -1])
            latest_volume = float(matrix.volumes[idx, -1])
    except Exception:
        logger.exception("Failed to read symbol data for %s %s", symbol, tf)
        return {}

    return {
        'close': closes_slice.tolist(),
        'volume': volumes_slice.tolist(),
        'latest_close': latest_close,
        'latest_volume': latest_volume
    }

async def get_leaderboard(tf: str, sort_key: str, limit: int = 10) -> List[Dict]:
    """Vectorized leaderboard using matrix.get_analysis()"""
    if not COLLECTOR or tf not in COLLECTOR.matrices:
        return []

    try:
        results = COLLECTOR.matrices[tf].get_analysis()
    except Exception:
        logger.exception("Failed to get analysis for timeframe %s", tf)
        return []

    leaderboard = []
    for sym, data in results.items():
        usdt_vol = data['price'] * data['volume']
        leaderboard.append({
            'symbol': sym,
            'close': data['price'],
            'price': data['price'],
            'rsi': data['rsi'],
            'vol_z': data['vol_z'],
            'change': data['change_tf'],
            'mfi': data['mfi'],
            'adx': data['adx'],
            'usdt_volume': usdt_vol
        })

    # Sort
    if sort_key == 'change_desc':
        leaderboard.sort(key=lambda x: x['change'], reverse=True)
    elif sort_key == 'change_asc':
        leaderboard.sort(key=lambda x: x['change'], reverse=False)
    elif sort_key == 'volume':
        leaderboard.sort(key=lambda x: x['usdt_volume'], reverse=True)
    elif sort_key == 'vol_z':
        leaderboard.sort(key=lambda x: x['vol_z'], reverse=True)

    return leaderboard[:limit]
