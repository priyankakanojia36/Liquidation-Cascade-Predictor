"""
CascadeWatch — Data Collection Module (Final Version)
5 data sources, all US-accessible, zero gaps.

Sources:
1. CryptoCompare — Historical hourly OHLCV (BTC, ETH, SOL) — 12 months
2. Coinbase — Spot price candles (US-based, always works)
3. CoinGecko — Daily market data + derivatives snapshots
4. dYdX — Historical funding rates + candles with open interest (decentralized, no geo-block)
5. Hyperliquid — Historical funding rates + real-time OI (decentralized, no geo-block)
"""

import requests
import pandas as pd
import time
import os
import json
from datetime import datetime, timedelta, timezone


LOOKBACK_DAYS = 365
REQUEST_DELAY = 0.3


def _get(url, params=None):
    """Makes GET request with error handling."""
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ GET failed: {e}")
        return None


def _post(url, payload=None):
    """Makes POST request with error handling (used by Hyperliquid)."""
    try:
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        print(f"  ❌ POST failed: {e}")
        return None


# =============================================================
# SOURCE 1: CRYPTOCOMPARE — Hourly OHLCV (free, no key, 12 months)
# =============================================================
def fetch_ohlcv_cryptocompare(symbol="BTC", currency="USD"):
    """
    Why: Foundation for all price-based features (volatility, ROC, volume ratios).
    Returns up to 2000 hourly candles per request. We paginate to get 12 months.
    """
    print(f"📊 [CryptoCompare] Fetching hourly OHLCV for {symbol}...")

    all_data = []
    end_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp())

    while end_ts > start_ts:
        data = _get("https://min-api.cryptocompare.com/data/v2/histohour", {
            "fsym": symbol, "tsym": currency, "limit": 2000, "toTs": end_ts
        })
        if not data or data.get("Response") != "Success":
            break

        rows = data["Data"]["Data"]
        if not rows:
            break

        all_data.extend(rows)
        oldest = rows[0]["time"]
        if oldest <= start_ts:
            break
        end_ts = oldest - 1
        time.sleep(REQUEST_DELAY)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"volumefrom": "volume_base", "volumeto": "volume_quote"})
    df["symbol"] = f"{symbol}USDT"
    df = df[["timestamp", "symbol", "open", "high", "low", "close", "volume_base", "volume_quote"]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    cutoff = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS))
    df = df[df["timestamp"] >= cutoff].sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} candles ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


# =============================================================
# SOURCE 2: COINBASE — Spot candles (US-based, always works)
# =============================================================
def fetch_candles_coinbase(product_id="BTC-USD"):
    """
    Why: Clean spot price from a US-regulated exchange.
    Cross-reference with derivatives prices to detect basis/premium.
    Max 300 candles per request.
    """
    print(f"🏦 [Coinbase] Fetching hourly candles for {product_id}...")

    all_data = []
    end_time = datetime.now(timezone.utc)
    start_limit = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    while end_time > start_limit:
        start_time = max(end_time - timedelta(hours=300), start_limit)

        data = _get(f"https://api.exchange.coinbase.com/products/{product_id}/candles", {
            "granularity": 3600,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
        })
        if not data or len(data) == 0:
            break

        all_data.extend(data)
        end_time = start_time - timedelta(seconds=1)
        time.sleep(REQUEST_DELAY)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data, columns=["timestamp", "low", "high", "open", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["symbol"] = product_id
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} candles ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


# =============================================================
# SOURCE 3: COINGECKO — Daily market data + derivatives snapshots
# =============================================================
def fetch_daily_market_coingecko(coin_id="bitcoin", days=365):
    """
    Why: Daily price, volume, and market cap for macro-level features.
    Also used to compute market cap ratios (OI/market cap).
    """
    print(f"💰 [CoinGecko] Fetching {days}-day market data for {coin_id}...")

    data = _get(f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart", {
        "vs_currency": "usd", "days": days, "interval": "daily"
    })
    if not data:
        return pd.DataFrame()

    prices = pd.DataFrame(data["prices"], columns=["timestamp", "price"])
    volumes = pd.DataFrame(data["total_volumes"], columns=["timestamp", "total_volume"])
    mcaps = pd.DataFrame(data["market_caps"], columns=["timestamp", "market_cap"])

    df = prices.merge(volumes, on="timestamp").merge(mcaps, on="timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["coin_id"] = coin_id
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} daily records")
    time.sleep(REQUEST_DELAY)
    return df


def fetch_derivatives_snapshot_coingecko():
    """
    Why: Current funding rates, OI, and spreads across all major exchanges.
    Snapshot data (not historical), but captures the market state right now.
    """
    print(f"⚖️ [CoinGecko] Fetching derivatives tickers snapshot...")

    all_tickers = []
    for page in range(1, 6):
        data = _get("https://api.coingecko.com/api/v3/derivatives", {"page": page})
        if not data or len(data) == 0:
            break
        all_tickers.extend(data)
        time.sleep(REQUEST_DELAY)

    if not all_tickers:
        return pd.DataFrame()

    df = pd.DataFrame(all_tickers)
    targets = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BTC", "ETH", "SOL", "XBTUSD", "ETHUSD"]
    df = df[df["symbol"].isin(targets)].copy()

    for col in ["funding_rate", "open_interest", "volume_24h", "spread", "basis"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["snapshot_time"] = datetime.now(timezone.utc)
    print(f"  ✅ {len(df)} perpetual tickers with funding rates")
    return df


# =============================================================
# SOURCE 4: dYdX — Historical funding rates + candles with OI
# =============================================================
def fetch_candles_dydx(ticker="BTC-USD", resolution="1HOUR"):
    """
    Why: dYdX candles include startingOpenInterest in every row.
    This gives us HISTORICAL open interest at hourly granularity,
    which is the #2 most important feature for cascade prediction.
    dYdX is decentralized so it never geo-blocks.
    Max 100 candles per request.
    """
    print(f"📈 [dYdX] Fetching hourly candles + OI for {ticker}...")

    all_data = []
    end_time = datetime.now(timezone.utc)
    start_limit = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    while end_time > start_limit:
        start_time = max(end_time - timedelta(hours=100), start_limit)

        data = _get(f"https://indexer.dydx.trade/v4/candles/perpetualMarkets/{ticker}", {
            "resolution": resolution,
            "fromISO": start_time.isoformat(),
            "toISO": end_time.isoformat(),
            "limit": 100,
        })

        if not data or not data.get("candles"):
            break

        all_data.extend(data["candles"])
        end_time = start_time - timedelta(seconds=1)
        time.sleep(REQUEST_DELAY)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["startedAt"], utc=True)

    for col in ["open", "high", "low", "close", "baseTokenVolume", "usdVolume", "startingOpenInterest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["trades"] = pd.to_numeric(df["trades"], errors="coerce")
    df["symbol"] = ticker

    df = df.rename(columns={
        "baseTokenVolume": "volume_base",
        "usdVolume": "volume_usd",
        "startingOpenInterest": "open_interest",
    })

    keep = ["timestamp", "symbol", "open", "high", "low", "close",
            "volume_base", "volume_usd", "trades", "open_interest"]
    df = df[[c for c in keep if c in df.columns]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} candles with OI ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


def fetch_funding_dydx(ticker="BTC-USD"):
    """
    Why: Historical funding rates are the #1 cascade predictor.
    dYdX provides exact funding rate at every funding period.
    Decentralized, no geo-blocking, free.
    Max 100 records per request.
    """
    print(f"💰 [dYdX] Fetching historical funding rates for {ticker}...")

    all_data = []
    end_time = datetime.now(timezone.utc)
    start_limit = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)

    while end_time > start_limit:
        data = _get(f"https://indexer.dydx.trade/v4/historicalFunding/{ticker}", {
            "limit": 100,
            "effectiveBeforeOrAt": end_time.isoformat(),
        })

        if not data or not data.get("historicalFunding"):
            break

        rows = data["historicalFunding"]
        all_data.extend(rows)

        oldest = pd.to_datetime(rows[-1]["effectiveAt"])
        if oldest.tz_localize(None) <= start_limit.replace(tzinfo=None):
            break

        end_time = oldest - timedelta(seconds=1)
        time.sleep(REQUEST_DELAY)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["effectiveAt"], utc=True)
    df["fundingRate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["symbol"] = ticker

    df = df[["timestamp", "symbol", "fundingRate", "price"]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} funding rate records")
    return df


# =============================================================
# SOURCE 5: HYPERLIQUID — Funding history + real-time OI
# =============================================================
def fetch_funding_hyperliquid(coin="BTC"):
    """
    Why: Second source of historical funding rates (cross-validation).
    Hyperliquid is decentralized, fully public, no key, no geo-block.
    Funding every 1 hour (more granular than most exchanges).
    """
    print(f"🔗 [Hyperliquid] Fetching funding history for {coin}...")

    start_ms = int((datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).timestamp() * 1000)

    all_data = []
    current_start = start_ms

    while True:
        data = _post("https://api.hyperliquid.xyz/info", {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": current_start,
        })

        if not data or len(data) == 0:
            break

        all_data.extend(data)

        # Move forward: last timestamp + 1
        last_ts = data[-1]["time"]
        if last_ts == current_start:
            break
        current_start = last_ts + 1
        time.sleep(REQUEST_DELAY)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["premium"] = pd.to_numeric(df["premium"], errors="coerce")
    df["symbol"] = coin
    df = df[["timestamp", "symbol", "fundingRate", "premium"]]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp").reset_index(drop=True)

    print(f"  ✅ {len(df)} funding records ({df['timestamp'].min().date()} to {df['timestamp'].max().date()})")
    return df


def fetch_market_snapshot_hyperliquid():
    """
    Why: Real-time open interest, funding, mark price for all assets.
    Captures the current state of the largest decentralized perp exchange.
    """
    print(f"🔗 [Hyperliquid] Fetching real-time market snapshot...")

    data = _post("https://api.hyperliquid.xyz/info", {"type": "metaAndAssetCtxs"})
    if not data:
        return pd.DataFrame()

    meta = data[0]["universe"]
    contexts = data[1]

    records = []
    for i, ctx in enumerate(contexts):
        records.append({
            "symbol": meta[i]["name"],
            "funding_rate": float(ctx.get("funding", 0)),
            "open_interest": float(ctx.get("openInterest", 0)),
            "mark_price": float(ctx.get("markPx", 0)),
            "oracle_price": float(ctx.get("oraclePx", 0)),
            "day_volume_usd": float(ctx.get("dayNtlVlm", 0)),
            "premium": float(ctx.get("premium", 0)),
            "prev_day_price": float(ctx.get("prevDayPx", 0)),
        })

    df = pd.DataFrame(records)
    df["snapshot_time"] = datetime.now(timezone.utc)

    # Filter to our target symbols
    targets = df[df["symbol"].isin(["BTC", "ETH", "SOL"])].copy()
    print(f"  ✅ {len(targets)} assets (BTC, ETH, SOL snapshot captured)")
    return targets


# =============================================================
# MAIN: Run full 5-source data collection
# =============================================================
def collect_all_data(save_dir="data/raw"):
    """Runs the complete multi-source data collection pipeline."""

    os.makedirs(save_dir, exist_ok=True)
    summary = {}

    # ---- SOURCE 1: CryptoCompare OHLCV ----
    print(f"\n{'='*60}")
    print("📡 SOURCE 1: CryptoCompare (hourly OHLCV, 12 months)")
    print(f"{'='*60}")
    for sym in ["BTC", "ETH", "SOL"]:
        df = fetch_ohlcv_cryptocompare(sym)
        if not df.empty:
            df.to_parquet(f"{save_dir}/{sym}USDT_ohlcv_cc.parquet", index=False)
            summary[f"{sym}_ohlcv"] = len(df)

    # ---- SOURCE 2: Coinbase spot candles ----
    print(f"\n{'='*60}")
    print("📡 SOURCE 2: Coinbase (hourly spot candles, 12 months)")
    print(f"{'='*60}")
    for prod in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        df = fetch_candles_coinbase(prod)
        if not df.empty:
            safe = prod.replace("-", "")
            df.to_parquet(f"{save_dir}/{safe}_candles_cb.parquet", index=False)
            summary[f"{prod}_coinbase"] = len(df)

    # ---- SOURCE 3: CoinGecko daily + derivatives ----
    print(f"\n{'='*60}")
    print("📡 SOURCE 3: CoinGecko (daily market data + derivatives)")
    print(f"{'='*60}")
    for coin_id in ["bitcoin", "ethereum", "solana"]:
        df = fetch_daily_market_coingecko(coin_id)
        if not df.empty:
            df.to_parquet(f"{save_dir}/{coin_id}_daily_cg.parquet", index=False)
            summary[f"{coin_id}_daily"] = len(df)

    df = fetch_derivatives_snapshot_coingecko()
    if not df.empty:
        df.to_parquet(f"{save_dir}/derivatives_snapshot_cg.parquet", index=False)
        summary["derivatives_snapshot"] = len(df)

    # ---- SOURCE 4: dYdX candles (with OI!) + funding rates ----
    print(f"\n{'='*60}")
    print("📡 SOURCE 4: dYdX (candles with OI + funding rates)")
    print(f"{'='*60}")
    for ticker in ["BTC-USD", "ETH-USD", "SOL-USD"]:
        df = fetch_candles_dydx(ticker)
        if not df.empty:
            safe = ticker.replace("-", "")
            df.to_parquet(f"{save_dir}/{safe}_candles_dydx.parquet", index=False)
            summary[f"{ticker}_dydx_candles"] = len(df)

        df = fetch_funding_dydx(ticker)
        if not df.empty:
            safe = ticker.replace("-", "")
            df.to_parquet(f"{save_dir}/{safe}_funding_dydx.parquet", index=False)
            summary[f"{ticker}_dydx_funding"] = len(df)

    # ---- SOURCE 5: Hyperliquid funding history + snapshot ----
    print(f"\n{'='*60}")
    print("📡 SOURCE 5: Hyperliquid (funding history + real-time snapshot)")
    print(f"{'='*60}")
    for coin in ["BTC", "ETH", "SOL"]:
        df = fetch_funding_hyperliquid(coin)
        if not df.empty:
            df.to_parquet(f"{save_dir}/{coin}_funding_hl.parquet", index=False)
            summary[f"{coin}_hl_funding"] = len(df)

    df = fetch_market_snapshot_hyperliquid()
    if not df.empty:
        df.to_parquet(f"{save_dir}/market_snapshot_hl.parquet", index=False)
        summary["hl_snapshot"] = len(df)

    # ---- SUMMARY ----
    print(f"\n{'='*60}")
    print("📋 DATA COLLECTION SUMMARY")
    print(f"{'='*60}")

    total = 0
    for name, count in sorted(summary.items()):
        print(f"  {name}: {count:,} records")
        total += count

    print(f"\n📦 Total records collected: {total:,}")
    print(f"💾 Saved to: {save_dir}/")

    print(f"\n📂 Files:")
    for f in sorted(os.listdir(save_dir)):
        if f.endswith(".parquet"):
            size = os.path.getsize(f"{save_dir}/{f}") / 1024
            print(f"  {f} ({size:.1f} KB)")

    return summary


# =============================================================
# RUN IT
# =============================================================
if __name__ == "__main__":
    print("🌊 CascadeWatch — Data Collection Pipeline (5-Source)")
    print(f"📅 Collecting {LOOKBACK_DAYS} days of historical data")
    print(f"🔗 Sources: CryptoCompare + Coinbase + CoinGecko + dYdX + Hyperliquid")
    print(f"⏰ Started at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print()

    start = time.time()
    summary = collect_all_data()
    elapsed = time.time() - start

    print(f"\n⏱️ Completed in {elapsed/60:.1f} minutes")