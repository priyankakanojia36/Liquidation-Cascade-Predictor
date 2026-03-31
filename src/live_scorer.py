"""
live_scorer.py
==============
Mode 2: Live data fetcher and risk scorer.

WHAT THIS DOES:
    1. Fetches the latest hour of market data from APIs
    2. Computes features from the fresh data
    3. Scores all 5 models
    4. Computes ensemble risk score
    5. Writes results to DynamoDB
    6. Sends SNS alert if risk > 75
    7. Returns the scores for the dashboard to display

WHEN IT RUNS:
    Only when the user clicks "Refresh Live Data" in the dashboard.
    Never runs automatically. No background processes.
    Zero AWS cost unless explicitly triggered.

CONCEPT: This is the bridge between the static Parquet world
and the live AWS world. Before Mode 2, the dashboard only read
from a frozen file. Now it can fetch today's real market data
and update the risk score in real time.
"""

import boto3
import requests
import pandas as pd
import numpy as np
import joblib
import warnings
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
MODELS_DIR   = ROOT / "data" / "models"
FEATURES_DIR = ROOT / "data" / "features"

AWS_REGION     = "us-east-1"
DYNAMODB_TABLE = "CascadeWatchRiskScores"
SNS_TOPIC_ARN  = "arn:aws:sns:us-east-1:377228489449:CascadeWatchAlerts"
ALERT_THRESHOLD = 75

# Weights — same as ensemble.py
WEIGHTS = {
    "classifier":  0.35,
    "anomaly":     0.25,
    "fear_index":  0.20,
    "severity":    0.10,
    "survival":    0.10,
}


# ===========================================================================
# STEP 1: FETCH LIVE MARKET DATA
# ===========================================================================

def fetch_live_data(symbol: str) -> dict:
    """
    Fetch the latest hour of OHLCV + funding rate for one symbol.

    CONCEPT: We call the same APIs we used in data_collection.py
    but only ask for the most recent 2 hours (limit=2).
    We take the second-to-last row because the current hour is
    incomplete — the previous hour is fully formed.

    Returns a dict with all the raw values we need to compute features.
    """
    print(f"  Fetching live data for {symbol}...")

    # Map our symbol names to API names
    cc_symbol  = symbol          # CryptoCompare: BTC, ETH, SOL
    hl_symbol  = symbol          # Hyperliquid: BTC, ETH, SOL

    result = {"symbol": symbol, "timestamp": datetime.now(timezone.utc)}

    # --- Price data from CryptoCompare ---
    try:
        url = "https://min-api.cryptocompare.com/data/v2/histohour"
        params = {"fsym": cc_symbol, "tsym": "USD", "limit": 5}
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()

        if data["Response"] == "Success":
            candles = data["Data"]["Data"]
            # Use second-to-last (last complete hour)
            latest  = candles[-2]
            prev    = candles[-3]

            result["open"]         = float(latest["open"])
            result["high"]         = float(latest["high"])
            result["low"]          = float(latest["low"])
            result["close"]        = float(latest["close"])
            result["volume_base"]  = float(latest["volumefrom"])
            result["volume_quote"] = float(latest["volumeto"])
            result["prev_close"]   = float(prev["close"])
            print(f"    Price: ${result['close']:,.0f}")
        else:
            raise ValueError(f"CryptoCompare error: {data.get('Message')}")

    except Exception as e:
        print(f"    Price fetch failed: {e}")
        return None

    # --- Funding rate from Hyperliquid ---
    try:
        url  = "https://api.hyperliquid.xyz/info"
        body = {"type": "metaAndAssetCtxs"}
        resp = requests.post(url, json=body, timeout=10)
        data = resp.json()

        # Find our symbol in the response
        universe = data[0]["universe"]
        ctxs     = data[1]

        funding_rate = 0.0
        for i, asset in enumerate(universe):
            if asset["name"] == hl_symbol:
                funding_rate = float(ctxs[i].get("funding", 0))
                break

        result["funding_rate"] = funding_rate
        print(f"    Funding: {funding_rate:.6f}")

    except Exception as e:
        print(f"    Funding fetch failed: {e}, using 0.0")
        result["funding_rate"] = 0.0

    return result


# ===========================================================================
# STEP 2: COMPUTE FEATURES FROM LIVE DATA
# ===========================================================================

def compute_live_features(live_data: dict, symbol: str) -> pd.DataFrame:
    """
    Compute the features needed for model scoring from live data.

    CONCEPT: We need the same 15+ features we engineered in Day 2.
    For a single live data point we use the pre-computed historical
    features as a baseline, then update with the fresh values.

    This is called "feature serving" — computing features at
    inference time rather than training time.
    """
    # Load historical features as baseline context
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    sym_df = df[df["symbol"] == symbol].sort_index()

    # Get the last known row as baseline
    baseline = sym_df.iloc[-1].copy()

    # Update with fresh values
    close     = live_data["close"]
    prev_close = live_data["prev_close"]
    returns_1h = (close - prev_close) / prev_close if prev_close > 0 else 0

    # Compute simple features from live data
    baseline["close"]        = close
    baseline["returns_1h"]   = returns_1h
    baseline["funding_rate"] = live_data["funding_rate"]
    baseline["volume_base"]  = live_data["volume_base"]

    # Volume ratio: compare to 24h average from historical data
    vol_ma_24h = sym_df["volume_base"].tail(24).mean()
    baseline["volume_ratio"] = live_data["volume_base"] / (vol_ma_24h + 1e-10)

    # Funding zscore: compare to 30-day rolling stats
    fund_mean = sym_df["funding_rate"].tail(720).mean()
    fund_std  = sym_df["funding_rate"].tail(720).std()
    baseline["funding_zscore"] = (
        (live_data["funding_rate"] - fund_mean) / (fund_std + 1e-10)
    )

    # Convert to DataFrame for model input
    return pd.DataFrame([baseline])


# ===========================================================================
# STEP 3: SCORE ALL MODELS
# ===========================================================================

def score_models(features_df: pd.DataFrame) -> dict:
    """
    Run all 5 models on the live feature row.

    CONCEPT: This is inference — applying trained models to new data.
    We load each model from disk, prepare the features in the same
    format used during training, and get a prediction.
    """
    print("  Scoring models...")

    CLASSIFIER_FEATURES = [
        "returns_1h", "returns_4h",
        "volatility_24h", "volatility_4h", "volatility_compression",
        "volume_ratio", "funding_rate", "funding_zscore",
        "funding_acceleration", "consecutive_positive_funding",
        "funding_max_24h", "roc_4h", "rsi_14",
        "price_position", "price_vs_ma",
        "btc_eth_corr_24h", "btc_sol_corr_24h", "simultaneous_drops_4h",
    ]

    ANOMALY_FEATURES = [
        "funding_zscore", "funding_acceleration",
        "consecutive_positive_funding", "funding_max_24h",
        "volume_ratio", "volatility_compression",
        "volatility_24h", "rsi_14", "price_position",
    ]

    SEVERITY_FEATURES = [
        "returns_1h", "returns_4h",
        "volatility_24h", "volatility_4h", "volatility_compression",
        "volume_ratio", "funding_rate", "funding_zscore",
        "funding_acceleration", "consecutive_positive_funding",
        "funding_max_24h", "roc_4h", "rsi_14",
        "price_position", "price_vs_ma",
    ]

    scores = {}

    # Fill cross-asset features with 0 for single-symbol live scoring
    for col in ["btc_eth_corr_24h", "btc_sol_corr_24h", "simultaneous_drops_4h"]:
        if col not in features_df.columns:
            features_df[col] = 0.0

    features_df = features_df.fillna(0)

    # Model 1: Classifier
    try:
        clf    = joblib.load(MODELS_DIR / "classifier.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        X = features_df[CLASSIFIER_FEATURES].values
        X_scaled = scaler.transform(X)
        scores["classifier"] = float(clf.predict_proba(X_scaled)[0, 1])
        print(f"    Classifier: {scores['classifier']:.4f}")
    except Exception as e:
        print(f"    Classifier failed: {e}")
        scores["classifier"] = 0.0

    # Model 2: Anomaly detector
    try:
        anom        = joblib.load(MODELS_DIR / "anomaly_detector.joblib")
        anom_scaler = joblib.load(MODELS_DIR / "anomaly_scaler.joblib")
        X = features_df[ANOMALY_FEATURES].values
        X_scaled = anom_scaler.transform(X)
        raw  = anom.decision_function(X_scaled)[0]
        # Normalize against historical range
        df_hist = pd.read_parquet(FEATURES_DIR / "ensemble_scores.parquet")
        hist_min = df_hist["score_anomaly"].min()
        hist_max = df_hist["score_anomaly"].max()
        norm = (-raw - hist_min) / (hist_max - hist_min + 1e-10)
        scores["anomaly"] = float(np.clip(norm, 0, 1))
        print(f"    Anomaly: {scores['anomaly']:.4f}")
    except Exception as e:
        print(f"    Anomaly failed: {e}")
        scores["anomaly"] = 0.0

    # Model 3: Severity
    try:
        sev        = joblib.load(MODELS_DIR / "severity_model.joblib")
        sev_scaler = joblib.load(MODELS_DIR / "severity_scaler.joblib")
        X = features_df[SEVERITY_FEATURES].values
        X_scaled = sev_scaler.transform(X)
        pred = float(np.clip(sev.predict(X_scaled)[0], 0, 0.15))
        scores["severity"] = pred / 0.15
        print(f"    Severity: {scores['severity']:.4f}")
    except Exception as e:
        print(f"    Severity failed: {e}")
        scores["severity"] = 0.0

    # Model 4: Fear index (rule-based, no model file needed)
    try:
        funding_zscore = float(features_df["funding_zscore"].iloc[0])
        rsi            = float(features_df["rsi_14"].iloc[0])
        volatility     = float(features_df["volatility_24h"].iloc[0])
        price_vs_ma    = float(features_df["price_vs_ma"].iloc[0])
        volume_ratio   = float(features_df["volume_ratio"].iloc[0])

        # Load historical ranges for normalization
        df_hist = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")

        def norm_val(val, col, reverse=False):
            mn = df_hist[col].min()
            mx = df_hist[col].max()
            n  = (val - mn) / (mx - mn + 1e-10) * 100
            return (100 - n) if reverse else n

        fear = (
            0.30 * norm_val(abs(funding_zscore), "funding_zscore") +
            0.20 * norm_val(abs(rsi - 50), "rsi_14") +
            0.25 * norm_val(volatility, "volatility_24h") +
            0.15 * norm_val(price_vs_ma, "price_vs_ma", reverse=True) +
            0.10 * norm_val(volume_ratio, "volume_ratio")
        )
        scores["fear_index"] = float(np.clip(fear, 0, 100)) / 100
        print(f"    Fear Index: {scores['fear_index']:.4f}")
    except Exception as e:
        print(f"    Fear Index failed: {e}")
        scores["fear_index"] = 0.2

    # Model 5: Survival (use fear index as proxy — survival NaN issue documented)
    scores["survival"] = scores["fear_index"] * 0.5

    return scores


# ===========================================================================
# STEP 4: COMPUTE ENSEMBLE RISK SCORE
# ===========================================================================

def compute_risk_score(scores: dict) -> tuple:
    """Combine all model scores into one 0-100 risk score."""
    risk_score = (
        WEIGHTS["classifier"] * scores["classifier"] +
        WEIGHTS["anomaly"]    * scores["anomaly"]    +
        WEIGHTS["fear_index"] * scores["fear_index"] +
        WEIGHTS["severity"]   * scores["severity"]   +
        WEIGHTS["survival"]   * scores["survival"]
    ) * 100

    risk_score = float(np.clip(risk_score, 0, 100))

    if risk_score >= 75:   level = "CRITICAL"
    elif risk_score >= 50: level = "HIGH"
    elif risk_score >= 25: level = "ELEVATED"
    else:                  level = "LOW"

    return round(risk_score, 2), level


# ===========================================================================
# STEP 5: WRITE TO DYNAMODB
# ===========================================================================

def write_to_dynamodb(symbol: str, live_data: dict,
                       scores: dict, risk_score: float,
                       risk_level: str):
    """
    Write the live risk score to DynamoDB.

    CONCEPT: Decimal type is required by DynamoDB for numbers.
    Python floats cannot be stored directly — they must be
    converted to Decimal first.
    """
    print(f"  Writing {symbol} to DynamoDB...")

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table    = dynamodb.Table(DYNAMODB_TABLE)

    item = {
        "symbol":           symbol,
        "risk_score":       Decimal(str(round(risk_score, 2))),
        "risk_level":       risk_level,
        "close":            Decimal(str(round(live_data.get("close", 0), 2))),
        "funding_rate":     Decimal(str(round(live_data.get("funding_rate", 0), 8))),
        "volume_ratio":     Decimal(str(round(scores.get("anomaly", 0), 4))),
        "score_classifier": Decimal(str(round(scores["classifier"], 4))),
        "score_anomaly":    Decimal(str(round(scores["anomaly"], 4))),
        "score_fear":       Decimal(str(round(scores["fear_index"], 4))),
        "score_severity":   Decimal(str(round(scores["severity"], 4))),
        "last_updated":     datetime.now(timezone.utc).isoformat(),
    }

    table.put_item(Item=item)
    print(f"    Written: {symbol} risk={risk_score} level={risk_level}")


# ===========================================================================
# STEP 6: SEND ALERT IF NEEDED
# ===========================================================================

def check_alert(symbol: str, risk_score: float, risk_level: str):
    """Send SNS alert if risk exceeds threshold."""
    if risk_score >= ALERT_THRESHOLD:
        print(f"  ALERT: {symbol} risk={risk_score} — sending SNS...")
        sns = boto3.client("sns", region_name=AWS_REGION)
        sns.publish(
            TopicArn=SNS_TOPIC_ARN,
            Subject=f"CascadeWatch Alert: {symbol} Risk {risk_score:.0f}/100",
            Message=(
                f"CASCADEWATCH CRITICAL ALERT\n\n"
                f"Symbol:     {symbol}/USD\n"
                f"Risk Score: {risk_score:.1f}/100\n"
                f"Risk Level: {risk_level}\n"
                f"Time:       {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"Cascade conditions detected across multiple risk signals."
            )
        )
        print(f"    Alert sent.")


# ===========================================================================
# MAIN FUNCTION — called by the dashboard Refresh button
# ===========================================================================

def run_live_scoring() -> dict:
    """
    Full live scoring pipeline for all 3 symbols.
    Returns a dict of results for the dashboard to display.

    This is the single function the dashboard calls when
    the Refresh button is clicked.
    """
    print(f"\nLive scoring started: "
          f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 50)

    results = {}

    for symbol in ["BTC", "ETH", "SOL"]:
        print(f"\nProcessing {symbol}...")

        # Step 1: Fetch live data
        live_data = fetch_live_data(symbol)
        if live_data is None:
            print(f"  Skipping {symbol} — data fetch failed")
            continue

        # Step 2: Compute features
        features_df = compute_live_features(live_data, symbol)

        # Step 3: Score models
        scores = score_models(features_df)

        # Step 4: Ensemble risk score
        risk_score, risk_level = compute_risk_score(scores)
        print(f"  Risk Score: {risk_score}/100 ({risk_level})")

        # Step 5: Write to DynamoDB
        write_to_dynamodb(symbol, live_data, scores, risk_score, risk_level)

        # Step 6: Send alert if needed
        check_alert(symbol, risk_score, risk_level)

        results[symbol] = {
            "risk_score":   risk_score,
            "risk_level":   risk_level,
            "close":        live_data.get("close", 0),
            "funding_rate": live_data.get("funding_rate", 0),
            "scores":       scores,
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }

    print("\n" + "=" * 50)
    print(f"Live scoring complete. {len(results)} symbols updated.")
    return results


if __name__ == "__main__":
    results = run_live_scoring()
    print("\nResults:")
    for sym, data in results.items():
        print(f"  {sym}: {data['risk_score']}/100 "
              f"({data['risk_level']}) @ ${data['close']:,.0f}")