"""
feature_engineering.py
=======================
Day 2: Transform raw market data into ML-ready features.

CONCEPT: Raw price data alone can't predict cascades. We need to
*engineer* signals that capture leverage buildup, positioning extremes,
and volatility compression — the fingerprint of a cascade before it happens.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# CONCEPT: Path() is a clean way to handle file paths across Mac/Windows.
# Instead of hardcoding "data/raw/...", we build paths programmatically.
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw")
FEATURES_DIR = Path("data/features")
FEATURES_DIR.mkdir(parents=True, exist_ok=True)  # create folder if it doesn't exist


# ===========================================================================
# STEP 1: DATA LOADERS
# Each function loads one type of raw file and standardizes it.
# CONCEPT: We separate loading from processing — clean code principle.
# ===========================================================================

def load_ohlcv(symbol: str) -> pd.DataFrame:
    """
    Load hourly OHLCV (price + volume) data for a symbol.
    
    CONCEPT: OHLCV = Open, High, Low, Close, Volume.
    This is the standard format for price data in every financial system.
    
    Args:
        symbol: "BTC", "ETH", or "SOL"
    Returns:
        DataFrame indexed by timestamp with price/volume columns
    """
    # CryptoCompare files are named like BTCUSDT_ohlcv_cc.parquet
    path = RAW_DIR / f"{symbol}USDT_ohlcv_cc.parquet"
    df = pd.read_parquet(path)
    
    # CONCEPT: set_index() makes timestamp the row label instead of 0,1,2...
    # This is critical for time-series math — pandas needs to know which
    # column is time so it can do rolling windows, resampling, merges by time.
    df = df.set_index("timestamp")
    
    # CONCEPT: sort_index() ensures rows are in chronological order.
    # APIs don't always return data in order — always sort time-series data.
    df = df.sort_index()
    
    # Keep only the columns we need
    return df[["open", "high", "low", "close", "volume_base", "volume_quote"]]


def load_funding(symbol: str) -> pd.DataFrame:
    """
    Load hourly funding rate data for a symbol.
    
    CONCEPT: Funding rate is the most important cascade predictor.
    It's the periodic payment between longs and shorts in futures markets.
    - Positive rate = longs pay shorts = market is overleveraged long
    - Strongly positive for many hours = danger zone
    
    Args:
        symbol: "BTC", "ETH", or "SOL"
    Returns:
        DataFrame indexed by timestamp with fundingRate column
    """
    path = RAW_DIR / f"{symbol}_funding_hl.parquet"
    df = pd.read_parquet(path)
    
    # CONCEPT: The Hyperliquid timestamps have millisecond noise
    # (e.g. 22:00:00.037). We round to the nearest hour so it aligns
    # cleanly with OHLCV data when we merge later.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.round("h")
    df = df.set_index("timestamp").sort_index()
    
    return df[["fundingRate", "premium"]]


# ===========================================================================
# STEP 2: MERGE ALL DATA SOURCES
# CONCEPT: Feature engineering requires joining multiple data sources
# on a common time axis. This is exactly what SQL JOINs do — but in pandas.
# ===========================================================================

def merge_data(symbol: str) -> pd.DataFrame:
    """
    Merge OHLCV + funding rate into one unified DataFrame.
    
    CONCEPT: pd.merge() with how="inner" keeps only timestamps
    that exist in BOTH datasets — no gaps, no nulls from mismatches.
    We use "left" join to keep all price rows and attach funding where available.
    """
    ohlcv = load_ohlcv(symbol)
    funding = load_funding(symbol)
    
    # CONCEPT: join() aligns on the index (timestamp).
    # how="left" = keep all rows from ohlcv, attach funding where timestamp matches.
    # If a funding row is missing for a timestamp, it fills with NaN.
    df = ohlcv.join(funding, how="left")
    
    # CONCEPT: forward-fill (ffill) fills NaN with the last known value.
    # Funding rates don't change every hour — it's fine to carry forward.
    df["fundingRate"] = df["fundingRate"].ffill()
    df["premium"] = df["premium"].ffill()
    
    print(f"[{symbol}] Merged data shape: {df.shape}")
    print(f"[{symbol}] Date range: {df.index.min()} → {df.index.max()}")
    print(f"[{symbol}] Null counts:\n{df.isnull().sum()}\n")
    
    return df


# ===========================================================================
# STEP 3: FEATURE ENGINEERING
# This is the core of Day 2. Each feature captures a different dimension
# of cascade risk. Think of features as "questions we ask the data."
# ===========================================================================

def engineer_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Transform raw OHLCV + funding data into 15+ predictive features.
    
    CONCEPT: Raw price and funding rate values are not enough for ML.
    We need to capture *change*, *acceleration*, *extremity*, and *divergence*.
    Each feature below answers a specific risk question.
    """
    feat = df.copy()  # never modify the original — always work on a copy
    
    # -----------------------------------------------------------------------
    # FEATURE GROUP 1: PRICE FEATURES
    # CONCEPT: We calculate returns (% change) instead of raw prices.
    # Why? Because a $1000 move means different things at $5K vs $90K BTC.
    # Returns are comparable across time and assets.
    # -----------------------------------------------------------------------
    
    # pct_change() calculates: (current - previous) / previous
    # This is the hourly return — how much did price move this hour?
    feat["returns_1h"] = feat["close"].pct_change()
    
    # CONCEPT: Rolling window = look at the last N rows as a sliding window.
    # .rolling(4).sum() = sum of the last 4 hours of returns = 4h total return
    feat["returns_4h"] = feat["close"].pct_change(4)
    
    # CONCEPT: Volatility = standard deviation of returns.
    # High volatility = price is moving unpredictably = risk is elevated.
    # We use a 24-hour rolling window to capture "how wild has it been today?"
    feat["volatility_24h"] = feat["returns_1h"].rolling(24).std()
    
    # CONCEPT: Volatility compression — ratio of short-term to long-term vol.
    # When 4h vol is MUCH lower than 24h vol, the market is coiling like a spring.
    # This "calm before the storm" pattern often precedes explosive moves.
    feat["volatility_4h"] = feat["returns_1h"].rolling(4).std()
    feat["volatility_compression"] = feat["volatility_4h"] / (feat["volatility_24h"] + 1e-10)
    # CONCEPT: We add 1e-10 (a tiny number) to avoid division by zero errors.
    
    # -----------------------------------------------------------------------
    # FEATURE GROUP 2: VOLUME FEATURES
    # CONCEPT: Volume tells us conviction. A big price move on low volume
    # is suspicious — it might reverse. High volume confirms the move.
    # -----------------------------------------------------------------------
    
    # Rolling mean of volume over 24h = "normal" volume baseline
    feat["volume_ma_24h"] = feat["volume_base"].rolling(24).mean()
    
    # Volume ratio: how does current volume compare to the 24h average?
    # ratio > 2.0 = volume spike = something unusual is happening
    feat["volume_ratio"] = feat["volume_base"] / (feat["volume_ma_24h"] + 1e-10)
    
    # -----------------------------------------------------------------------
    # FEATURE GROUP 3: FUNDING RATE FEATURES
    # These are the MOST important features for cascade prediction.
    # CONCEPT: Funding rate extremes signal leverage buildup.
    # -----------------------------------------------------------------------
    
    # Raw funding rate (already have it, just rename for clarity)
    feat["funding_rate"] = feat["fundingRate"]
    
    # CONCEPT: Z-score = how many standard deviations from the mean?
    # funding_zscore of +3 means funding rate is 3 standard deviations above
    # its 30-day average — extremely unusual, danger zone.
    rolling_mean = feat["fundingRate"].rolling(720).mean()   # 720h = 30 days
    rolling_std = feat["fundingRate"].rolling(720).std()
    feat["funding_zscore"] = (feat["fundingRate"] - rolling_mean) / (rolling_std + 1e-10)
    
    # CONCEPT: Acceleration = rate of change of funding rate itself.
    # If funding rate is rising fast, leverage is building fast.
    # diff() = current value minus previous value
    feat["funding_acceleration"] = feat["fundingRate"].diff()
    
    # CONCEPT: Consecutive positive funding = how many hours in a row
    # has the funding rate been positive (longs paying shorts)?
    # This captures sustained overleveraging, not just a single spike.
    is_positive = (feat["fundingRate"] > 0).astype(int)
    # cumsum trick: count consecutive 1s, reset to 0 when a 0 appears
    feat["consecutive_positive_funding"] = is_positive.groupby(
        (is_positive == 0).cumsum()
    ).cumsum()
    
    # Rolling max of funding rate over 24 hours — was it extreme recently?
    feat["funding_max_24h"] = feat["fundingRate"].rolling(24).max()
    
    # -----------------------------------------------------------------------
    # FEATURE GROUP 4: MOMENTUM FEATURES
    # CONCEPT: Momentum = is the price accelerating or decelerating?
    # Cascades often start with a momentum reversal — price was climbing,
    # then suddenly flips and drops, triggering liquidations.
    # -----------------------------------------------------------------------
    
    # Rate of change over 4 hours — is the move speeding up?
    feat["roc_4h"] = feat["close"].pct_change(4)
    
    # RSI (Relative Strength Index) — classic overbought/oversold indicator
    # CONCEPT: RSI > 70 = overbought (due for a drop), RSI < 30 = oversold
    delta = feat["close"].diff()
    gain = delta.clip(lower=0)   # clip() removes negative values (only keep gains)
    loss = -delta.clip(upper=0)  # flip sign: only keep losses as positive numbers
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))
    # CONCEPT: RSI formula: 100 - (100 / (1 + RS)) where RS = avg_gain/avg_loss
    
    # -----------------------------------------------------------------------
    # FEATURE GROUP 5: PRICE STRUCTURE FEATURES
    # CONCEPT: These capture where price is relative to recent history.
    # -----------------------------------------------------------------------
    
    # Distance from 24h high — how far has price dropped from the recent peak?
    feat["high_24h"] = feat["high"].rolling(24).max()
    feat["low_24h"] = feat["low"].rolling(24).min()
    
    # Price position within 24h range: 0 = at the low, 1 = at the high
    # CONCEPT: If price is near the top of a range AND funding is high,
    # that's a classic setup for a cascade — everyone's long at the top.
    feat["price_position"] = (feat["close"] - feat["low_24h"]) / (
        feat["high_24h"] - feat["low_24h"] + 1e-10
    )
    
    # Distance from 7-day moving average (as a %)
    feat["ma_168h"] = feat["close"].rolling(168).mean()  # 168h = 7 days
    feat["price_vs_ma"] = (feat["close"] - feat["ma_168h"]) / (feat["ma_168h"] + 1e-10)
    
    # -----------------------------------------------------------------------
    # CLEAN UP
    # CONCEPT: dropna() removes rows where features couldn't be calculated
    # because there weren't enough prior rows for the rolling window.
    # For example, a 720-hour rolling window needs 720 rows before it works.
    # We lose the first 720 rows — that's fine, we still have 8000+ left.
    # -----------------------------------------------------------------------
    feat = feat.dropna()
    
    # Add symbol column for tracking when we combine BTC+ETH+SOL later
    feat["symbol"] = symbol
    
    print(f"[{symbol}] Features shape after dropna: {feat.shape}")
    print(f"[{symbol}] Features: {[c for c in feat.columns if c not in ['open','high','low','close','volume_base','volume_quote','fundingRate','premium','symbol']]}\n")
    
    return feat


# ===========================================================================
# STEP 4: RUN PIPELINE FOR ALL 3 SYMBOLS
# CONCEPT: We build features independently per symbol, then stack them.
# pd.concat() with axis=0 stacks DataFrames vertically (adds more rows).
# ===========================================================================

def run_feature_pipeline():
    """
    Run the full feature engineering pipeline for BTC, ETH, and SOL.
    Save the result to data/features/ as a Parquet file.
    """
    symbols = ["BTC", "ETH", "SOL"]
    all_features = []
    
    for symbol in symbols:
        print(f"=== Processing {symbol} ===")
        df = merge_data(symbol)              # Step 1: load + merge raw data
        features = engineer_features(df, symbol)  # Step 2: compute all features
        all_features.append(features)        # Step 3: collect results
    
    # CONCEPT: pd.concat() stacks all three DataFrames into one big DataFrame.
    # Now we have BTC + ETH + SOL features all in one place.
    combined = pd.concat(all_features, axis=0).sort_index()
    
    # Save to Parquet — this becomes the input to our ML models on Day 3
    out_path = FEATURES_DIR / "features_combined.parquet"
    combined.to_parquet(out_path)
    
    print(f"\n✅ Feature engineering complete!")
    print(f"   Total rows: {len(combined):,}")
    print(f"   Symbols: {combined['symbol'].unique()}")
    print(f"   Date range: {combined.index.min()} → {combined.index.max()}")
    print(f"   Saved to: {out_path}")
    
    return combined


# ===========================================================================
# CONCEPT: if __name__ == "__main__" means:
# "Only run this block if I run this file directly (not if I import it)."
# This is standard Python practice for runnable scripts.
# ===========================================================================
if __name__ == "__main__":
    df_features = run_feature_pipeline()
    print("\nSample output:")
    print(df_features.head())