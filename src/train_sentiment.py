"""
train_sentiment.py
==================
Day 4 — Model 5: Market Fear Index

WHAT THIS MODEL DOES:
    Produces a composite fear score (0-100) for each market hour.
    0   = maximum calm/greed (no cascade risk from sentiment)
    100 = maximum fear (multiple danger signals active simultaneously)

WHY THIS APPROACH (REROUTE DECISION):
    Original plan: FinBERT NLP on crypto news headlines
    Actual build:  Rule-based quantitative fear index

    Reason for reroute:
    1. No real-time news API in our data pipeline
    2. FinBERT requires GPU for reasonable speed
    3. Market microstructure signals are MORE reliable than social
       sentiment — they reflect actual money moving, not just words
    4. Every component is explainable — critical for interviews

    This is similar to how professional quant funds build fear
    indicators: from order flow, funding rates, and volatility
    rather than from Twitter sentiment.

THE 5 COMPONENTS:
    1. fear_funding    (weight 30%) — funding rate extremity
    2. fear_rsi        (weight 20%) — overbought/oversold signal
    3. fear_volatility (weight 25%) — volatility level
    4. fear_price      (weight 15%) — price deviation from trend
    5. fear_volume     (weight 10%) — unusual volume activity

OUTPUT FILE: data/features/fear_index.parquet
    This file is loaded by ensemble.py on Day 5.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
from pathlib import Path
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
DOCS_DIR     = Path("docs")

# Component weights — must sum to 1.0
# CONCEPT: Weights reflect our EDA findings.
# Funding and volatility were the strongest signals in Day 2 EDA,
# so they get the highest weights.
WEIGHTS = {
    "fear_funding":    0.30,
    "fear_rsi":        0.20,
    "fear_volatility": 0.25,
    "fear_price":      0.15,
    "fear_volume":     0.10,
}


# ===========================================================================
# STEP 1: LOAD DATA
# ===========================================================================

def load_data():
    """Load the labeled feature dataset."""
    print("Loading data...")
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    df = df.sort_index()
    print(f"Loaded {len(df):,} rows across {df['symbol'].nunique()} symbols")
    print(f"Date range: {df.index.min().date()} → {df.index.max().date()}\n")
    return df


# ===========================================================================
# STEP 2: BUILD EACH FEAR COMPONENT
# ===========================================================================

def normalize_to_100(series, reverse=False):
    """
    Normalize any series to a 0-100 scale.

    CONCEPT: Min-max normalization.
    Formula: (value - min) / (max - min) × 100

    If reverse=True, we flip it: 100 - normalized_value
    We use this when LOWER values mean MORE fear.
    Example: low RSI means oversold = more fear, so we reverse RSI.

    Args:
        series:  pandas Series to normalize
        reverse: if True, flip so low values = high fear score
    Returns:
        Series with values between 0 and 100
    """
    mn  = series.min()
    mx  = series.max()
    # Add 1e-10 to denominator to prevent division by zero
    normalized = (series - mn) / (mx - mn + 1e-10) * 100
    return (100 - normalized) if reverse else normalized


def build_fear_components(df):
    """
    Build all 5 fear components and combine into composite score.

    Each component answers one specific fear question:
    - fear_funding:    "Is leverage dangerously extreme right now?"
    - fear_rsi:        "Is the market overbought or oversold?"
    - fear_volatility: "How wildly is price moving?"
    - fear_price:      "How far has price fallen from its trend?"
    - fear_volume:     "Is there unusual panic buying or selling?"
    """
    result = df.copy()

    # -----------------------------------------------------------------------
    # COMPONENT 1: FUNDING RATE FEAR (weight: 30%)
    #
    # CONCEPT: We use the ABSOLUTE VALUE of funding zscore because
    # both extreme positive (overleveraged longs) and extreme negative
    # (overleveraged shorts) funding rates signal danger.
    # A zscore of +3 or -3 are equally dangerous — different directions,
    # same underlying problem: too much leverage in the system.
    # -----------------------------------------------------------------------
    result["fear_funding"] = normalize_to_100(
        result["funding_zscore"].abs()
    )
    print("✅ Component 1: Funding fear built")

    # -----------------------------------------------------------------------
    # COMPONENT 2: RSI FEAR (weight: 20%)
    #
    # CONCEPT: RSI ranges from 0 to 100.
    # RSI > 70 = overbought (price rose too fast, correction likely)
    # RSI < 30 = oversold (price fell too fast, but also cascade risk)
    # We measure DISTANCE FROM NEUTRAL (50):
    # |RSI - 50| → 0 means perfectly neutral, 50 means extreme
    # Then normalize to 0-100.
    # -----------------------------------------------------------------------
    rsi_deviation = (result["rsi_14"] - 50).abs()
    result["fear_rsi"] = normalize_to_100(rsi_deviation)
    print("✅ Component 2: RSI fear built")

    # -----------------------------------------------------------------------
    # COMPONENT 3: VOLATILITY FEAR (weight: 25%)
    #
    # CONCEPT: High 24h volatility = price is moving unpredictably.
    # Unpredictability is dangerous for leveraged positions.
    # High volatility can trigger stop losses and margin calls
    # even without a directional cascade.
    # -----------------------------------------------------------------------
    result["fear_volatility"] = normalize_to_100(result["volatility_24h"])
    print("✅ Component 3: Volatility fear built")

    # -----------------------------------------------------------------------
    # COMPONENT 4: PRICE vs MOVING AVERAGE FEAR (weight: 15%)
    #
    # CONCEPT: price_vs_ma measures how far price is above/below
    # its 7-day moving average. Positive = above MA (overbought zone).
    # Negative = below MA (oversold/crash territory).
    #
    # reverse=True because NEGATIVE price_vs_ma (price below MA)
    # = more fear. We flip so low values become high fear scores.
    # -----------------------------------------------------------------------
    result["fear_price"] = normalize_to_100(
        result["price_vs_ma"], reverse=True
    )
    print("✅ Component 4: Price fear built")

    # -----------------------------------------------------------------------
    # COMPONENT 5: VOLUME FEAR (weight: 10%)
    #
    # CONCEPT: volume_ratio compares current volume to the 24h average.
    # ratio > 2.0 = panic buying or selling = elevated fear.
    # Normal trading has ratio near 1.0.
    # -----------------------------------------------------------------------
    result["fear_volume"] = normalize_to_100(result["volume_ratio"])
    print("✅ Component 5: Volume fear built")

    # -----------------------------------------------------------------------
    # COMPOSITE FEAR INDEX
    #
    # CONCEPT: Weighted average of all 5 components.
    # Each component contributes proportionally to its weight.
    # The weights reflect our EDA findings about which features
    # matter most for cascade prediction.
    # -----------------------------------------------------------------------
    result["fear_index"] = sum(
        result[component] * weight
        for component, weight in WEIGHTS.items()
    )

    # Verify weights sum to 1.0 — sanity check
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

    print(f"\n✅ Composite fear index built")
    print(f"   Range: {result['fear_index'].min():.1f} → "
          f"{result['fear_index'].max():.1f}")
    print(f"   Mean:  {result['fear_index'].mean():.1f}")

    return result


# ===========================================================================
# STEP 3: EVALUATE THE FEAR INDEX
# ===========================================================================

def evaluate_fear_index(df_with_fear):
    """
    Evaluate: does the fear index actually separate normal from
    pre-cascade conditions? And does it give advance warning?

    Metrics:
    - AUC-ROC: does higher fear correlate with cascade labels?
    - Mean fear in normal vs pre-cascade conditions
    - Fear level distribution by category
    """
    print("\n" + "="*50)
    print("Evaluating Fear Index")
    print("="*50)

    valid = df_with_fear[["fear_index", "pre_cascade",
                           "symbol"]].dropna()

    # AUC-ROC — same metric as classifiers
    # CONCEPT: AUC measures whether higher fear scores tend to
    # appear before cascades. Even a small AUC above 0.5 is
    # meaningful given the extreme class imbalance.
    auc = roc_auc_score(valid["pre_cascade"], valid["fear_index"])

    # Average fear by condition
    normal_fear  = valid[valid["pre_cascade"] == 0]["fear_index"].mean()
    cascade_fear = valid[valid["pre_cascade"] == 1]["fear_index"].mean()
    elevation    = cascade_fear - normal_fear

    # Fear level buckets
    # CONCEPT: We categorize the continuous 0-100 score into
    # intuitive levels for the dashboard display
    bins   = [0, 25, 50, 75, 100]
    labels = ["Low", "Elevated", "High", "Critical"]
    valid["fear_level"] = pd.cut(
        valid["fear_index"], bins=bins, labels=labels
    )

    print(f"\n📊 Fear Index Performance:")
    print(f"   AUC-ROC:                    {auc:.4f}")
    print(f"   Avg fear (normal):          {normal_fear:.1f}/100")
    print(f"   Avg fear (pre-cascade):     {cascade_fear:.1f}/100")
    print(f"   Pre-cascade elevation:      +{elevation:.1f} points")

    print(f"\n📊 Fear level distribution:")
    level_counts = valid["fear_level"].value_counts().sort_index()
    for level, count in level_counts.items():
        pct = count / len(valid) * 100
        bar = "█" * int(pct / 2)
        print(f"   {level:<10} {count:>6,} rows  {pct:>5.1f}%  {bar}")

    print(f"\n📊 Fear by symbol:")
    for sym in ["BTC", "ETH", "SOL"]:
        sym_data = valid[valid["symbol"] == sym]
        print(f"   {sym}: mean={sym_data['fear_index'].mean():.1f}  "
              f"max={sym_data['fear_index'].max():.1f}")

    return auc, normal_fear, cascade_fear, elevation


# ===========================================================================
# STEP 4: VISUALIZE
# ===========================================================================

def plot_fear_index(df_with_fear, auc, normal_fear, cascade_fear):
    """
    Three plots:
    1. Fear index over time for BTC with cascade events marked
    2. Fear component breakdown (how much does each component contribute?)
    3. Fear distribution: normal vs pre-cascade
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model 5: Market Fear Index",
                 fontsize=13, fontweight="bold")

    # --- Plot 1: BTC fear index timeline ---
    ax1 = axes[0]
    btc = df_with_fear[df_with_fear["symbol"] == "BTC"].copy()
    btc_cascades = btc[btc["cascade_event"] == 1]

    ax1.plot(btc.index, btc["fear_index"],
             color="#d29922", linewidth=0.8, label="Fear Index")
    ax1.fill_between(btc.index, btc["fear_index"], 0,
                     alpha=0.15, color="#d29922")

    # Fear level zones
    ax1.axhspan(75, 100, alpha=0.08, color="#da3633", label="Critical (75+)")
    ax1.axhspan(50, 75,  alpha=0.05, color="#d29922", label="High (50-75)")

    for t in btc_cascades.index:
        ax1.axvline(t, color="#da3633", linewidth=1.5,
                    linestyle="--", alpha=0.8)

    ax1.set_title(f"BTC Fear Index Over Time\nAUC: {auc:.4f}")
    ax1.set_ylabel("Fear Score (0-100)")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Plot 2: Component contributions ---
    ax2 = axes[1]
    components = list(WEIGHTS.keys())
    weights    = list(WEIGHTS.values())
    comp_colors = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f78166"]

    bars = ax2.bar(
        [c.replace("fear_", "") for c in components],
        weights, color=comp_colors
    )
    for bar, w in zip(bars, weights):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.005,
                 f"{w*100:.0f}%", ha="center",
                 fontsize=10, fontweight="bold")

    ax2.set_title("Fear Index Component Weights")
    ax2.set_ylabel("Weight")
    ax2.set_ylim(0, 0.4)
    ax2.grid(True, alpha=0.2, axis="y")

    # --- Plot 3: Normal vs Pre-cascade distribution ---
    ax3 = axes[2]
    normal_vals  = df_with_fear[
        df_with_fear["pre_cascade"] == 0]["fear_index"].dropna()
    cascade_vals = df_with_fear[
        df_with_fear["pre_cascade"] == 1]["fear_index"].dropna()

    ax3.hist(normal_vals, bins=50, alpha=0.6, color="#58a6ff",
             label=f"Normal (n={len(normal_vals):,})", density=True)
    ax3.hist(cascade_vals, bins=20, alpha=0.8, color="#da3633",
             label=f"Pre-cascade (n={len(cascade_vals)})", density=True)

    ax3.axvline(normal_fear,  color="#58a6ff", linewidth=2,
                linestyle="--", label=f"Normal mean: {normal_fear:.0f}")
    ax3.axvline(cascade_fear, color="#da3633", linewidth=2,
                linestyle="--", label=f"Cascade mean: {cascade_fear:.0f}")

    ax3.set_title("Fear Score Distribution\nNormal vs Pre-Cascade")
    ax3.set_xlabel("Fear Index (0-100)")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    out = DOCS_DIR / "plot_fear_index.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved {out}")


# ===========================================================================
# STEP 5: SAVE FEAR INDEX DATA FOR ENSEMBLE
# ===========================================================================

def save_fear_data(df_with_fear):
    """
    Save the fear index values to a Parquet file.
    This is loaded by ensemble.py on Day 5 to combine with
    the classifier and anomaly detector outputs.
    """
    fear_cols = [
        "symbol", "fear_index", "fear_funding",
        "fear_rsi", "fear_volatility", "fear_price", "fear_volume"
    ]
    fear_output = df_with_fear[fear_cols].copy()
    out = FEATURES_DIR / "fear_index.parquet"
    fear_output.to_parquet(out)
    print(f"✅ Saved {out}")
    print(f"   Rows: {len(fear_output):,}")
    print(f"   Columns: {fear_cols}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Day 4 — Model 5: Market Fear Index")
    print("=" * 50)

    df = load_data()

    print("Building fear components...")
    df_with_fear = build_fear_components(df)

    auc, normal_fear, cascade_fear, elevation = evaluate_fear_index(
        df_with_fear)

    plot_fear_index(df_with_fear, auc, normal_fear, cascade_fear)

    save_fear_data(df_with_fear)

    print(f"\n✅ Day 4 Model 5 complete!")
    print(f"\n🎯 HEADLINE METRIC:")
    print(f"   Pre-cascade fear: {cascade_fear:.1f}/100")
    print(f"   Normal fear:      {normal_fear:.1f}/100")
    print(f"   Elevation:        +{elevation:.1f} points before cascades")
    print(f"   AUC-ROC:          {auc:.4f}")
    print(f"\n📌 Interview talking point:")
    print(f"   'Built a quantitative fear index combining funding rate")
    print(f"    extremity, RSI deviation, volatility, price trend, and")
    print(f"    volume signals — analogous to CNN Fear & Greed Index")
    print(f"    but calibrated specifically for derivatives markets.'")


    