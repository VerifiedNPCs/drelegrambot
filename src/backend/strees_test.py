import asyncio
import time
import sys
import random
import logging
import os
import csv
import shutil
import io
from unittest.mock import MagicMock, AsyncMock, patch

# --- 1. CONFIG & MOCKS (PRE-IMPORT) ---
sys.modules['config'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['Data_Collector'] = MagicMock()
sys.modules['market_engine'] = MagicMock()

# Setup Config
config_mock = sys.modules['config']
config_mock.config.MAX_RETRIES = 1
config_mock.config.RETRY_DELAY = 0.0
config_mock.config.TIMEFRAME_CHOICES = ['1m', '5m', '15m', '1h', '4h', '1d']
config_mock.config.DEPTH_TARGET_USDT = 100000
config_mock.config.ADMIN_CHAT_ID = 12345
config_mock.setup_logging = lambda x: logging.getLogger(x)

# Setup Database
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

# --- 4. SETUP OUTPUTS ---
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
    mock_collector.top_coins = [f"COIN{i}USDT" for i in range(100)]
    mock_collector.matrices = {}
    for tf in ['1m', '5m', '1h', '4h']:
        mat = MagicMock()
        mat.symbol_map = {f"COIN{i}USDT": i for i in range(100)}
        mat.lock = MagicMock()
        mat.lock.__enter__ = MagicMock()
        mat.lock.__exit__ = MagicMock()
        mat.get_analysis.return_value = generate_random_analysis_data()
        mock_collector.matrices[tf] = mat

    services.COLLECTOR = mock_collector
    services.ENGINE = MagicMock()
    services.ENGINE.alerts_history = []
    services.ENGINE.lock = AsyncMock()

    async def fake_get_symbol_data(tf, symbol, limit=20):
        start = random.uniform(100, 200)
        return {
            'close': [start + i for i in range(20)],
            'volume': [1000] * 20,
            'latest_close': start + 20,
            'latest_volume': 1000.0
        }
    services.get_symbol_data = fake_get_symbol_data

# --- 6. TEST RUNNER ---
async def run_single_request(req_id, handler_func, args=None, data=None):
    # Refresh Global Data
    new_data = generate_random_analysis_data(idx_offset=req_id)
    for tf in services.COLLECTOR.matrices:
        services.COLLECTOR.matrices[tf].get_analysis.return_value = new_data

    # --- THE CRITICAL PART: SAVE HOOK ---
    async def save_photo_hook(*args, **kwargs):
        # print(f"DEBUG: send_photo called for Req {req_id}") 
        
        # 1. Try Keyword Arg 'photo'
        photo = kwargs.get('photo')
        
        # 2. Try Positional Arg (usually 2nd arg after chat_id)
        if not photo and len(args) > 1:
            photo = args[1]

        if not photo:
            print(f"❌ Req {req_id}: send_photo called but NO PHOTO found in args/kwargs!")
            return True

        # 3. Check type
        if hasattr(photo, 'getbuffer'):
            fname = f"req_{req_id:04d}_{handler_func.__name__}.png"
            fpath = os.path.join(IMG_DIR, fname)
            try:
                with open(fpath, 'wb') as f:
                    f.write(photo.getbuffer())
                # print(f"✅ Req {req_id}: Saved {fname}")
            except Exception as e:
                print(f"❌ Req {req_id}: File Write Error: {e}")
        else:
            print(f"❌ Req {req_id}: Photo object is not bytes/buffer! Type: {type(photo)}")
        
        return True

    # Setup Bot
    bot = AsyncMock()
    bot.send_photo = AsyncMock(side_effect=save_photo_hook)
    bot.send_message = AsyncMock() # Consume text messages
    
    context = MagicMock()
    context.bot = bot
    context.args = args if args else []

    update = MagicMock()
    update.effective_user.id = 1000 + req_id
    update.effective_chat.id = 2000 + req_id
    update.message = MagicMock()
    if data:
        update.callback_query = MagicMock()
        update.callback_query.data = data
        update.callback_query.answer = AsyncMock()

    # Run
    start = time.perf_counter()
    try:
        await handler_func(update, context)
        success = True
    except Exception as e:
        print(f"\n❌ Req {req_id} Exception: {e}")
        import traceback; traceback.print_exc()
        success = False
    
    if success and not bot.send_photo.called:
        # If success but no photo, check if maybe send_message was called instead?
        # print(f"⚠️ Req {req_id}: Finished but send_photo was NOT called.")
        pass

    return {"id": req_id, "type": handler_func.__name__, "success": success, "duration": time.perf_counter() - start}

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
        writer = csv.DictWriter(f, fieldnames=["id", "type", "success", "duration"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\n✅ Finished in {total_time:.2f}s")
    print(f"📸 Images Generated: {len(os.listdir(IMG_DIR))}")
    print(f"📂 Location: {os.path.abspath(IMG_DIR)}")

if __name__ == "__main__":
    setup_directories()
    logging.disable(logging.CRITICAL)
    try:
        asyncio.run(main(num_requests=500))
    except KeyboardInterrupt:
        pass
