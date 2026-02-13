import asyncio
import os
import time
import io
from concurrent.futures import ProcessPoolExecutor 
from typing import Dict, List, Optional
from config import config, setup_logging
from health_checker import HealthMonitor

logger = setup_logging("Services")

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

# --- OPTIMIZATION 1: DYNAMIC THREAD POOL ---
# Scale workers based on CPU core count, but cap at 16 to avoid thrashing
MAX_WORKERS = max(2, os.cpu_count())
IMG_EXECUTOR = ProcessPoolExecutor(max_workers=MAX_WORKERS)

# --- CACHED GENERATION FUNCTION ---
async def get_generated_image(cache_key: str, img_data: List[Dict], target_usdt: float):
    """
    Smart wrapper that checks cache first, enriches data, then generates image in ProcessPool.
    """
    now = time.time()
    
    # 1. Check Cache (Fast Path)
    if cache_key in _IMG_CACHE:
        ts, buf_bytes = _IMG_CACHE[cache_key]
        if now - ts < 30: # 30 Second TTL
            # Return new buffer from cached bytes so we don't close the original
            return io.BytesIO(buf_bytes)

    # 2. Enrich Data (Must happen in Main Thread / Async Loop)
    # We do this OUTSIDE the executor because Collector is an async object and cannot be pickled
    if COLLECTOR:
        try:
            await image_generator.enrich_data(img_data, COLLECTOR, target_usdt)
        except Exception as e:
            logger.warning(f"Enrichment partial failure: {e}")

    # 3. Generate Image (CPU Bound - Run in Process Pool)
    loop = asyncio.get_running_loop()
    
    try:
        # CRITICAL: We pass the function reference and the data argument separately.
        # This allows the ProcessPoolExecutor to pickle them correctly.
        photo_buf = await loop.run_in_executor(
            IMG_EXECUTOR, 
            image_generator.generate_market_image, 
            img_data
        )
    except Exception as e:
        logger.error(f"Image Executor Failed: {e}", exc_info=True)
        return None

    # 4. Save to Cache
    if photo_buf:
        try:
            photo_buf.seek(0)
            bytes_data = photo_buf.read()
            photo_buf.seek(0) # Reset cursor for the caller to use
            
            async with CACHE_LOCK:
                _IMG_CACHE[cache_key] = (now, bytes_data)
        except Exception as e:
            logger.warning(f"Cache save failed: {e}")

    return photo_buf


# --- OPTIMIZATION 2: CACHE STORAGE ---
_IMG_CACHE = {}
CACHE_LOCK = asyncio.Lock()

async def init_services(app):
    global COLLECTOR, ENGINE, HEALTH
    COLLECTOR = BinanceMarketDataCollector()
    ENGINE = MarketEngine(COLLECTOR)
    HEALTH = HealthMonitor(data_dir=COLLECTOR.output_dir)
    
    asyncio.create_task(COLLECTOR.run())
    asyncio.create_task(ENGINE.start_background_task())
    asyncio.create_task(HEALTH.start_monitoring(interval=60))
    
    # Auto-clean cache every 5 minutes
    asyncio.create_task(_cache_cleanup_loop())

    app.bot_data['collector'] = COLLECTOR
    app.bot_data['engine'] = ENGINE

async def _cache_cleanup_loop():
    """Periodically removes old items from cache to save RAM."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        keys_to_del = [k for k, v in _IMG_CACHE.items() if now - v[0] > 60]
        for k in keys_to_del:
            del _IMG_CACHE[k]

# --- CACHED GENERATION FUNCTION ---

# ... (Keep get_symbol_data and get_leaderboard EXACTLY as they were) ...
async def get_symbol_data(tf: str, symbol: str, limit: int = 20) -> Dict:
    if not COLLECTOR or tf not in COLLECTOR.matrices: return {}
    matrix = COLLECTOR.matrices[tf]
    idx = matrix.symbol_map.get(symbol)
    if idx is None: return {}
    try:
        with matrix.lock:
            closes_slice = matrix.closes[idx, -limit:].copy()
            volumes_slice = matrix.volumes[idx, -limit:].copy()
            latest_close = float(matrix.closes[idx, -1])
            latest_volume = float(matrix.volumes[idx, -1])
    except Exception:
        return {}
    return {
        'close': closes_slice.tolist(),
        'volume': volumes_slice.tolist(),
        'latest_close': latest_close,
        'latest_volume': latest_volume
    }

async def get_leaderboard(tf: str, sort_key: str, limit: int = 10) -> List[Dict]:
    if not COLLECTOR or tf not in COLLECTOR.matrices: return []
    try:
        results = COLLECTOR.matrices[tf].get_analysis()
    except Exception:
        return []
    leaderboard = []
    for sym, data in results.items():
        leaderboard.append({
            'symbol': sym,
            'close': data['price'],
            'price': data['price'],
            'rsi': data['rsi'],
            'vol_z': data['vol_z'],
            'change': data['change_tf'],
            'mfi': data['mfi'],
            'adx': data['adx'],
            'usdt_volume': data['price'] * data['volume']
        })
    if sort_key == 'change_desc': leaderboard.sort(key=lambda x: x['change'], reverse=True)
    elif sort_key == 'change_asc': leaderboard.sort(key=lambda x: x['change'], reverse=False)
    elif sort_key == 'volume': leaderboard.sort(key=lambda x: x['usdt_volume'], reverse=True)
    elif sort_key == 'vol_z': leaderboard.sort(key=lambda x: x['vol_z'], reverse=True)
    return leaderboard[:limit]
