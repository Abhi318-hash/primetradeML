"""
generate_data.py — One-time script to create a reproducible synthetic OHLCV dataset.
Run: python generate_data.py
Produces: data.csv (10,000 rows)
"""

import numpy as np
import pandas as pd

SEED = 42
ROWS = 10_000
BASE_PRICE = 100.0
OUTPUT_FILE = "data.csv"

np.random.seed(SEED)

# Simulate a random walk for close prices
returns = np.random.normal(loc=0.0001, scale=0.01, size=ROWS)
close = BASE_PRICE * np.cumprod(1 + returns)

# Derive OHLCV from close
spread = np.abs(np.random.normal(loc=0, scale=0.5, size=ROWS))
high = close + spread
low = close - spread
open_ = close * (1 + np.random.normal(0, 0.003, size=ROWS))
volume = np.random.randint(1_000, 100_000, size=ROWS)

df = pd.DataFrame({
    "open":   np.round(open_, 4),
    "high":   np.round(high, 4),
    "low":    np.round(low, 4),
    "close":  np.round(close, 4),
    "volume": volume,
})

df.to_csv(OUTPUT_FILE, index=False)
print(f"[OK] Generated {OUTPUT_FILE} with {len(df)} rows.")
print(df.head())
