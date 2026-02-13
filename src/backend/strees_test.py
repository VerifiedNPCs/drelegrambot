import asyncio
import time
import sys
import random
import logging
import os
import csv
import shutil
from unittest.mock import MagicMock, AsyncMock

# --- 1. MOCK DEPENDENCIES ---
sys.modules['config'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['Data_Collector'] = MagicMock()
sys.modules['market_engine'] = MagicMock()

# Config
config_mock = sys.modules['config']
config_mock.config.MAX_RETRIES = 1
config_mock.config.RETRY_DELAY = 0.0
config_mock.config.TIMEFRAME_CHOICES = ['1m', '5m', '15m', '1h', '4h', '1d']
config_mock.config.DEPTH_TARGET_USDT = 100000
config_mock.config.ADMIN_CHAT_ID = 12345
config_mock.setup_logging = lambda x: logging.getLogger(x)

# Database
db_mock = sys.modules['database']
db_mock.SUBS = AsyncMock()
db_mock.USER_SETTINGS = AsyncMock()
db_mock.USER_SETTINGS.get_default_tf.return_value = '1h'

# --- 2. DISABLE MULTIPROCESSING ---
class SyncExecutor:
    def __init__(self, *args, **kwargs): pass
    def submit(self, fn, *args, **kwargs):
        class Future:
            def result(self): return fn(*args, **kwargs)
            def add_done_callback(self, fn): fn(self)
            def exception(self): return None
        return Future()
    def shutdown(self, wait=True): pass

import concurrent.futures
concurrent.futures.ProcessPoolExecutor = SyncExecutor

# --- 3. RELOAD MODULES ---
for k in ['services', 'bot_handlers', 'image_generator']:
    if k in sys.modules: del sys.modules[k]

import services
import bot_handlers
import image_generator 

# --- 4. OUTPUTS ---
OUTPUT_DIR = "stress_outputs"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")

def setup_directories():
    if os.path.exists(IMG_DIR):
        try: shutil.rmtree(IMG_DIR)
        except: pass
    os.makedirs(IMG_DIR, exist_ok=True)

# --- 5. DATA GENERATION ---
def generate_random_analysis_data(idx_offset=0):
    data = {}
    for i in range(20):
        symbol = f"COIN{i}USDT"
        base = float(random.uniform(10, 1000)) + idx_offset
        data[symbol] = {
            'symbol': symbol,
            'price': base,
            'volume': float(random.uniform(100000, 10e6)),
            'usdt_volume': float(random.uniform(100000, 10e6)),
            'rsi': float(random.uniform(15, 85)),
            'mfi': float(random.uniform(15, 85)),
            'adx': float(random.uniform(10, 70)),
            'change_tf': random.uniform(-10, 10),
            'vol_z': float(random.uniform(-3, 6)),
            'liq_1h_long': float(random.uniform(0, 5e5)),
            'liq_1h_short': float(random.uniform(0, 5e5)),
            'depth_down_pct': random.uniform(0.1, 2.0),
            'depth_up_pct': random.uniform(0.1, 2.0),
            'depth_down_price': base * 0.98,
            'depth_up_price': base * 1.02,
        }
    return data

def setup_global_mocks():
    mock_collector = MagicMock()
    mock_collector.top_coins = [f"COIN{i}USDT" for i in range(20)]
    mock_collector.matrices = {}
    
    initial_data = generate_random_analysis_data()

    for tf in ['1m', '5m', '1h', '4h']:
        mat = MagicMock()
        mat.symbol_map = {f"COIN{i}USDT": i for i in range(20)}
        mat.lock = MagicMock()
        mat.lock.__enter__ = MagicMock()
        mat.lock.__exit__ = MagicMock()
        mat.get_analysis.return_value = initial_data
        mock_collector.matrices[tf] = mat

    mock_collector.get_data = MagicMock(return_value=initial_data)

    services.COLLECTOR = mock_collector
    services.ENGINE = MagicMock()
    services.ENGINE.alerts_history = []
    services.ENGINE.lock = AsyncMock()

    # Mock Symbol Data (Sparklines)
    async def fake_get_symbol_data(tf, symbol, limit=20):
        start = random.uniform(100, 200)
        return {
            'close': [start + i for i in range(20)],
            'volume': [1000] * 20,
            'latest_close': start + 20,
            'latest_volume': 1000.0
        }
    services.get_symbol_data = fake_get_symbol_data
    
    # --- CRITICAL FIX: MOCK IMAGE GENERATION SERVICE ---
    # This connects the bot handler to the actual image generator logic
    async def fake_get_generated_image(key, data, depth):
        # Call the REAL image generator here to test its performance
        return image_generator.generate_market_image(data)
        
    services.get_generated_image = AsyncMock(side_effect=fake_get_generated_image)
    
    # Also mock get_leaderboard since top_cmd uses it
    async def fake_get_leaderboard(tf, sort_key, limit=10):
        # Return fake leaderboard data matching the structure expected
        return [
            {
                'symbol': f"COIN{i}USDT",
                'close': 100.0 + i,
                'change': 5.0 - (i*0.5),
                'rsi': 50, 'mfi': 50, 'adx': 20, 'usdt_volume': 1000000
            }
            for i in range(limit)
        ]
    services.get_leaderboard = AsyncMock(side_effect=fake_get_leaderboard)


# --- 6. TEST RUNNER ---
async def run_single_request(req_id, handler_func, args=None, data=None):
    new_data = generate_random_analysis_data(idx_offset=req_id)
    for tf in services.COLLECTOR.matrices:
        services.COLLECTOR.matrices[tf].get_analysis.return_value = new_data

    # Save Hook
    async def save_photo_hook(*args, **kwargs):
        photo = kwargs.get('photo')
        if not photo and len(args) > 1: photo = args[1]
        
        if hasattr(photo, 'getbuffer'):
            fname = f"req_{req_id:04d}_{handler_func.__name__}.png"
            fpath = os.path.join(IMG_DIR, fname)
            try:
                with open(fpath, 'wb') as f:
                    f.write(photo.getbuffer())
            except Exception as e:
                print(f"❌ Write Error: {e}")
        return True

    # Debug Hook
    async def send_message_hook(*args, **kwargs):
        text = kwargs.get('text')
        if not text and len(args) > 1: text = args[1]
        # print(f"\n⚠️ Bot Text Response: {text}") 
        return True

    bot = AsyncMock()
    bot.send_photo = AsyncMock(side_effect=save_photo_hook)
    bot.send_message = AsyncMock(side_effect=send_message_hook)
    
    context = MagicMock()
    context.bot = bot
    context.args = args if args else []

    update = MagicMock()
    update.effective_user.id = 1000 + req_id
    update.effective_chat.id = 2000 + req_id
    update.message = MagicMock()
    
    if args:
        update.message.text = f"/command {' '.join(args)}"
    else:
        update.message.text = "/command"

    if data:
        update.callback_query = MagicMock()
        update.callback_query.data = data
        update.callback_query.answer = AsyncMock()

    start = time.perf_counter()
    try:
        await handler_func(update, context)
        success = True
    except Exception as e:
        print(f"\n❌ Req {req_id} Error: {e}")
        success = False
    
    return {
        "id": req_id, 
        "type": handler_func.__name__, 
        "success": success, 
        "duration": time.perf_counter() - start,
        "image_saved": bot.send_photo.called
    }

async def main(num_requests=10):
    print(f"🚀 Starting Stress Test: {num_requests} requests")
    setup_global_mocks()
    
    scenarios = [
        (bot_handlers.top_cmd, [], None),       
        (bot_handlers.vol_cmd, [], None),       
        (bot_handlers.symbol_cmd, ["COIN1USDT"], None) 
    ]
    
    results = []
    start_global = time.perf_counter()
    
    for i in range(num_requests):
        func, args, data = random.choice(scenarios)
        sys.stdout.write(f"\rProcessing: {i+1}/{num_requests}")
        sys.stdout.flush()
        
        res = await run_single_request(i, func, args, data)
        results.append(res)
        
    total_time = time.perf_counter() - start_global
    
    # Save CSV
    csv_path = os.path.join(OUTPUT_DIR, "results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["id", "type", "success", "duration", "image_saved"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\n✅ Finished in {total_time:.2f}s")
    print(f"📸 Images in folder: {len(os.listdir(IMG_DIR))}")
    print(f"📂 Location: {os.path.abspath(IMG_DIR)}")

if __name__ == "__main__":
    setup_directories()
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(main(num_requests=500))
    except KeyboardInterrupt:
        pass
