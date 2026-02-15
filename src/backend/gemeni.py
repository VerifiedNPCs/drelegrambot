market_matrix_updates = """
================================================================================
MARKET MATRIX UPDATES (For Histogram Storage)
================================================================================

You need to update `MarketMatrix` to store the 20-bin histogram data.

STEP 1: Update __init__
-----------------------
Initialize 2D arrays for the bins. Shape = (Number of Symbols, 20 Bins).

    def __init__(self, symbols, timeframe):
        # ... existing code ...
        
        # Liquidity Totals (Existing)
        self.liquidity_bid_5pct = np.zeros(self.n_symbols, dtype=np.float32)
        self.liquidity_ask_5pct = np.zeros(self.n_symbols, dtype=np.float32)
        
        # Liquidity Histograms (NEW)
        # 20 bins per symbol (0.25% steps up to 5%)
        self.liquidity_bid_bins = np.zeros((self.n_symbols, 20), dtype=np.float32)
        self.liquidity_ask_bins = np.zeros((self.n_symbols, 20), dtype=np.float32)


STEP 2: Update update_futures_data
----------------------------------
Modify the 'liquidity_5pct' block to save the bins if they exist.

    def update_futures_data(self, symbol, data_type, value, extra_data=None):
        idx = self.symbol_map.get(symbol)
        if idx is None:
            return

        with self.lock:
            # ... existing types ...
            
            elif data_type == 'liquidity_5pct':
                # value is total bid liquidity
                self.liquidity_bid_5pct[idx] = value
                
                if extra_data:
                    # Save Total Ask
                    if 'ask' in extra_data:
                        self.liquidity_ask_5pct[idx] = float(extra_data['ask'])
                    
                    # Save Bins (NEW)
                    if 'bid_bins' in extra_data:
                        self.liquidity_bid_bins[idx] = extra_data['bid_bins']
                    
                    if 'ask_bins' in extra_data:
                        self.liquidity_ask_bins[idx] = extra_data['ask_bins']


STEP 3: Update get_analysis (or get_depth_liquidity_range)
----------------------------------------------------------
When reading data for the bot, you need to extract these bins.

    # In get_depth_liquidity_range (in Collector or Engine):
    
    with matrix.lock:
        bid_liq = float(matrix.liquidity_bid_5pct[idx])
        ask_liq = float(matrix.liquidity_ask_5pct[idx])
        
        # Read Bins (Copy to avoid race conditions)
        bid_bins_copy = matrix.liquidity_bid_bins[idx].copy()
        ask_bins_copy = matrix.liquidity_ask_bins[idx].copy()

    return {
        "symbol": symbol,
        "bid_liquidity_5pct": bid_liq,
        "ask_liquidity_5pct": ask_liq,
        "bid_bins": bid_bins_copy.tolist(),  # Convert to list for JSON/Bot
        "ask_bins": ask_bins_copy.tolist(),
        # ... other fields
    }
"""

print(market_matrix_updates)