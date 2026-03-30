"""
ensemble.py
===========
Day 5 Part 1: CascadeWatch Ensemble Risk Score Engine

WHAT THIS DOES:
    Combines outputs from all 5 models into one number:
    the CascadeWatch Risk Score (0 to 100).

    0-25:  LOW      — normal market conditions
    25-50: ELEVATED — leverage building, monitor closely
    50-75: HIGH     — multiple risk signals active
    75-100: CRITICAL — cascade conditions detected

WHY AN ENSEMBLE?
    No single model is reliable enough alone:
    - Classifier: AUC 0.5563 — decent but not great
    - Anomaly:    AUC 0.5922 — better but misses 91% of cascades
    - Fear index: AUC 0.6266 — best standalone but small elevation
    Together they cover each other's blind spots.
    This is exactly how production ML systems work in industry.

WEIGHTS (starting point, tunable):
    Classifier:  0.35  (strongest labeled signal)
    Anomaly:     0.25  (unsupervised, catches novel patterns)
    Fear index:  0.20  (best AUC, most stable signal)
    Severity:    0.10  (magnitude contribution)
    Survival:    0.10  (timing contribution)

REROUTES TAKEN:
    - Survival model outputs hazard rate, not a 0-1 probability.
      We normalize it to 0-1 using min-max before combining.
    - Severity output is a % drop magnitude.
      We normalize it too so all inputs are on the same scale.
    - Cross-asset features needed during classifier inference
      must be recomputed from raw features at scoring time.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
DOCS_DIR     = Path("docs")

# Ensemble weights — must sum to 1.0
WEIGHTS = {
    "classifier":  0.35,
    "anomaly":     0.25,
    "fear_index":  0.20,
    "severity":    0.10,
    "survival":    0.10,
}

# Risk level thresholds
RISK_LEVELS = {
    "LOW":      (0,  25),
    "ELEVATED": (25, 50),
    "HIGH":     (50, 75),
    "CRITICAL": (75, 100),
}

# Features the classifier was trained on (including cross-asset)
CLASSIFIER_FEATURES = [
    "returns_1h", "returns_4h",
    "volatility_24h", "volatility_4h", "volatility_compression",
    "volume_ratio",
    "funding_rate", "funding_zscore", "funding_acceleration",
    "consecutive_positive_funding", "funding_max_24h",
    "roc_4h", "rsi_14",
    "price_position", "price_vs_ma",
    "btc_eth_corr_24h", "btc_sol_corr_24h", "simultaneous_drops_4h",
]

# Features the anomaly detector was trained on
ANOMALY_FEATURES = [
    "funding_zscore", "funding_acceleration",
    "consecutive_positive_funding", "funding_max_24h",
    "volume_ratio", "volatility_compression",
    "volatility_24h", "rsi_14", "price_position",
]

# Features the severity model was trained on
SEVERITY_FEATURES = [
    "returns_1h", "returns_4h",
    "volatility_24h", "volatility_4h", "volatility_compression",
    "volume_ratio", "funding_rate", "funding_zscore",
    "funding_acceleration", "consecutive_positive_funding",
    "funding_max_24h", "roc_4h", "rsi_14",
    "price_position", "price_vs_ma",
]

# Features the survival model was trained on
SURVIVAL_FEATURES = [
    "funding_zscore", "consecutive_positive_funding",
    "volume_ratio", "volatility_compression",
    "rsi_14", "price_position",
]


# ===========================================================================
# STEP 1: LOAD ALL MODELS AND DATA
# ===========================================================================

def load_everything():
    """
    Load all 5 trained models, their scalers, and the feature dataset.

    CONCEPT: joblib.load() deserializes a saved Python object back into
    memory. Each model was serialized with joblib.dump() in Days 3 and 4.
    We load them all here so we can run inference on the full dataset.
    """
    print("Loading models...")

    # Load all trained models
    classifier    = joblib.load(MODELS_DIR / "classifier.joblib")
    clf_scaler    = joblib.load(MODELS_DIR / "scaler.joblib")
    anomaly       = joblib.load(MODELS_DIR / "anomaly_detector.joblib")
    anom_scaler   = joblib.load(MODELS_DIR / "anomaly_scaler.joblib")
    severity      = joblib.load(MODELS_DIR / "severity_model.joblib")
    sev_scaler    = joblib.load(MODELS_DIR / "severity_scaler.joblib")
    survival      = joblib.load(MODELS_DIR / "survival_model.joblib")

    print("  ✅ Classifier loaded")
    print("  ✅ Anomaly detector loaded")
    print("  ✅ Severity model loaded")
    print("  ✅ Survival model loaded")

    # Load feature data
    print("\nLoading feature data...")
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    df = df.sort_index()

    # Load pre-computed fear index
    fear_df = pd.read_parquet(FEATURES_DIR / "fear_index.parquet")
    fear_df = fear_df.sort_index()

    # CONCEPT: Merge on BOTH timestamp AND symbol to avoid row tripling.
    # fear_index.parquet already has timestamp as a named column.
    df_reset   = df.reset_index()   # timestamp becomes a regular column
    fear_reset = fear_df[["symbol", "fear_index"]].reset_index()
    # fear_reset now has columns: timestamp, symbol, fear_index

    df_merged = df_reset.merge(
        fear_reset, on=["timestamp", "symbol"], how="left"
    )
    df = df_merged.set_index("timestamp")
    df["fear_index"] = df["fear_index"].ffill().fillna(20.0)

    print(f"  ✅ Feature data loaded: {len(df):,} rows")

    return (classifier, clf_scaler, anomaly, anom_scaler,
            severity, sev_scaler, survival, df)


# ===========================================================================
# STEP 2: ADD CROSS-ASSET FEATURES
# (needed for classifier inference — same as train_classifier.py)
# ===========================================================================

def add_cross_asset_features(df):
    """
    Recompute cross-asset features for classifier inference.

    CONCEPT: The classifier was trained with 3 cross-asset features.
    At inference time we must compute the same features — otherwise
    the model receives inputs it never trained on and gives wrong outputs.
    This is called "training-serving consistency" — a critical principle
    in production ML.
    """
    returns_pivot = df.pivot_table(
        index=df.index, columns="symbol", values="returns_1h"
    )

    btc_eth_corr = returns_pivot["BTC"].rolling(24).corr(returns_pivot["ETH"])
    btc_sol_corr = returns_pivot["BTC"].rolling(24).corr(returns_pivot["SOL"])

    all_negative = (
        (returns_pivot["BTC"] < 0) &
        (returns_pivot["ETH"] < 0) &
        (returns_pivot["SOL"] < 0)
    ).astype(int)
    simultaneous_drops = all_negative.rolling(4).sum()

    df = df.join(btc_eth_corr.rename("btc_eth_corr_24h"), how="left")
    df = df.join(btc_sol_corr.rename("btc_sol_corr_24h"), how="left")
    df = df.join(simultaneous_drops.rename("simultaneous_drops_4h"),
                 how="left")

    df[["btc_eth_corr_24h", "btc_sol_corr_24h",
        "simultaneous_drops_4h"]] = df[[
        "btc_eth_corr_24h", "btc_sol_corr_24h",
        "simultaneous_drops_4h"]].ffill().fillna(0)

    return df


# ===========================================================================
# STEP 3: RUN INFERENCE WITH ALL 5 MODELS
# ===========================================================================

def run_inference(df, classifier, clf_scaler, anomaly, anom_scaler,
                  severity, sev_scaler, survival):
    """
    Score every row in the dataset with all 5 models.
    Returns a DataFrame with one score column per model.

    CONCEPT: "Inference" means running a trained model on new data
    to get predictions. We're not training here — we're just applying
    the patterns each model learned to every row in our dataset.
    """
    print("\nRunning inference with all 5 models...")
    result = df.copy()

    # -----------------------------------------------------------------------
    # MODEL 1: CLASSIFIER — probability of cascade in next 4 hours
    # predict_proba() returns [[prob_class_0, prob_class_1], ...]
    # We take [:, 1] = probability of class 1 (pre-cascade)
    # -----------------------------------------------------------------------
    clf_X = result[CLASSIFIER_FEATURES].ffill().fillna(0)
    clf_X_scaled = clf_scaler.transform(clf_X)
    result["score_classifier"] = classifier.predict_proba(clf_X_scaled)[:, 1]
    print("  ✅ Classifier scored")

    # -----------------------------------------------------------------------
    # MODEL 2: ANOMALY DETECTOR — how unusual is this market condition?
    # decision_function() returns raw scores, more negative = more anomalous
    # We negate and normalize to 0-1
    # -----------------------------------------------------------------------
    anom_X = result[ANOMALY_FEATURES].ffill().fillna(0)
    anom_X_scaled = anom_scaler.transform(anom_X)
    raw_anomaly = anomaly.decision_function(anom_X_scaled)
    # Negate: more negative original score = higher anomaly = higher risk
    anomaly_scores = -raw_anomaly
    # Normalize to 0-1
    result["score_anomaly"] = (
        (anomaly_scores - anomaly_scores.min()) /
        (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
    )
    print("  ✅ Anomaly detector scored")

    # -----------------------------------------------------------------------
    # MODEL 3: SEVERITY — predicted magnitude of price drop
    # Output is a % (e.g., 0.065 = 6.5% drop)
    # We normalize to 0-1 using a max of 15% drop as ceiling
    # CONCEPT: We clip at 0.15 (15%) because that's a realistic max
    # for a 4-hour crypto move. Anything above that is an outlier.
    # -----------------------------------------------------------------------
    sev_X = result[SEVERITY_FEATURES].ffill().fillna(0)
    sev_X_scaled = sev_scaler.transform(sev_X)
    sev_pred = np.clip(severity.predict(sev_X_scaled), 0, 0.15)
    result["score_severity"] = sev_pred / 0.15  # normalize to 0-1
    print("  ✅ Severity model scored")

    # -----------------------------------------------------------------------
    # MODEL 4: SURVIVAL — hazard rate (instantaneous cascade risk)
    # predict_partial_hazard() returns relative hazard for each row.
    # Higher = higher instantaneous risk.
    # We normalize to 0-1.
    # -----------------------------------------------------------------------
    surv_X = result[SURVIVAL_FEATURES].ffill().fillna(0)
    hazard  = survival.predict_partial_hazard(surv_X)
    result["score_survival"] = (
        (hazard - hazard.min()) /
        (hazard.max() - hazard.min() + 1e-10)
    )
    print("  ✅ Survival model scored")

    # -----------------------------------------------------------------------
    # MODEL 5: FEAR INDEX — already computed, normalize to 0-1
    # fear_index ranges 0-100, divide by 100 to get 0-1
    # -----------------------------------------------------------------------
    result["score_fear"] = result["fear_index"] / 100.0
    print("  ✅ Fear index scored")

    return result


# ===========================================================================
# STEP 4: COMPUTE ENSEMBLE RISK SCORE
# ===========================================================================

def compute_ensemble_score(result):
    """
    Combine all 5 model scores into one CascadeWatch Risk Score.

    CONCEPT: Weighted average.
    Each model contributes proportionally to its assigned weight.
    The final score is multiplied by 100 to give a 0-100 range.

    This is the number that appears on the dashboard gauge.
    """
    result["risk_score"] = (
        WEIGHTS["classifier"] * result["score_classifier"] +
        WEIGHTS["anomaly"]    * result["score_anomaly"]    +
        WEIGHTS["fear_index"] * result["score_fear"]       +
        WEIGHTS["severity"]   * result["score_severity"]   +
        WEIGHTS["survival"]   * result["score_survival"]
    ) * 100

    # Clip to valid range
    result["risk_score"] = result["risk_score"].clip(0, 100)

    # Assign risk level label
    def get_risk_level(score):
        if score >= 75:   return "CRITICAL"
        elif score >= 50: return "HIGH"
        elif score >= 25: return "ELEVATED"
        else:             return "LOW"

    result["risk_level"] = result["risk_score"].apply(get_risk_level)

    print(f"\n📊 Risk Score Summary:")
    print(f"   Range:  {result['risk_score'].min():.1f} → "
          f"{result['risk_score'].max():.1f}")
    print(f"   Mean:   {result['risk_score'].mean():.1f}")
    print(f"\n📊 Risk level distribution:")
    level_counts = result["risk_level"].value_counts()
    for level in ["LOW", "ELEVATED", "HIGH", "CRITICAL"]:
        count = level_counts.get(level, 0)
        pct   = count / len(result) * 100
        bar   = "█" * int(pct / 2)
        print(f"   {level:<10} {count:>6,}  {pct:>5.1f}%  {bar}")

    return result


# ===========================================================================
# STEP 5: BACKTEST — did the risk score warn before known cascades?
# ===========================================================================

def backtest(result):
    """
    For each known cascade event, check:
    1. Did the risk score cross 50 before the cascade?
    2. How many hours before?
    3. What was the peak score in the 8h window?

    CONCEPT: Backtesting validates the ensemble on historical data.
    This gives us our headline metric:
    "CascadeWatch detected X of Y cascades with Z hours average lead time"
    """
    print("\n" + "="*50)
    print("BACKTEST RESULTS")
    print("="*50)

    cascade_times = result[result["cascade_event"] == 1].index
    threshold     = 50  # risk score above this = alert

    detected    = 0
    lead_times  = []
    peak_scores = []

    for t in cascade_times:
        sym = result.loc[t, "symbol"]
        if isinstance(sym, pd.Series):
            sym = sym.iloc[0]

        sym_result = result[result["symbol"] == sym]

        # Look at 8 hours before each cascade
        window_start = t - pd.Timedelta("8h")
        pre_window   = sym_result.loc[window_start:t]

        # Peak risk score in the window
        peak = pre_window["risk_score"].max()
        peak_scores.append(peak)

        # Did score cross threshold before cascade?
        above = pre_window[pre_window["risk_score"] >= threshold]
        if len(above) > 0:
            first_alert = above.index[0]
            lead_h = (t - first_alert).total_seconds() / 3600
            lead_times.append(lead_h)
            detected += 1

    total    = len(cascade_times)
    det_rate = detected / total * 100 if total > 0 else 0
    avg_lead = np.mean(lead_times) if lead_times else 0
    avg_peak = np.mean(peak_scores) if peak_scores else 0

    print(f"\n   Threshold:               {threshold}/100")
    print(f"   Cascades detected:       {detected}/{total} ({det_rate:.1f}%)")
    print(f"   Avg lead time:           {avg_lead:.1f} hours before cascade")
    print(f"   Avg peak score:          {avg_peak:.1f}/100")

    # AUC of ensemble vs cascade labels
    valid = result[["risk_score", "pre_cascade"]].dropna()
    auc   = roc_auc_score(valid["pre_cascade"], valid["risk_score"])
    print(f"   Ensemble AUC-ROC:        {auc:.4f}")

    print(f"\n🎯 HEADLINE METRIC:")
    print(f"   'CascadeWatch detected {detected}/{total} historical cascade "
          f"events ({det_rate:.1f}%)")
    print(f"    with an average lead time of {avg_lead:.1f} hours.'")

    return detected, total, avg_lead, auc


# ===========================================================================
# STEP 6: VISUALIZE ENSEMBLE RESULTS
# ===========================================================================

def plot_ensemble(result, detected, total, avg_lead, auc):
    """
    Two plots:
    1. BTC risk score timeline with cascade events
    2. Risk score distribution: normal vs pre-cascade
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    fig.suptitle(
        f"CascadeWatch Ensemble Risk Score\n"
        f"Detected {detected}/{total} cascades | "
        f"Avg lead time: {avg_lead:.1f}h | AUC: {auc:.4f}",
        fontsize=13, fontweight="bold"
    )

    # --- Plot 1: BTC risk score over time ---
    ax1 = axes[0]
    btc = result[result["symbol"] == "BTC"].copy()
    btc_cascades = btc[btc["cascade_event"] == 1]

    # Color the risk score line by level
    ax1.plot(btc.index, btc["risk_score"],
             color="#58a6ff", linewidth=0.8, label="Risk Score", zorder=2)

    # Risk level zones
    ax1.axhspan(75, 100, alpha=0.08, color="#da3633", label="Critical (75+)")
    ax1.axhspan(50, 75,  alpha=0.05, color="#d29922", label="High (50-75)")
    ax1.axhspan(25, 50,  alpha=0.03, color="#3fb950", label="Elevated (25-50)")

    # Cascade events
    for t in btc_cascades.index:
        ax1.axvline(t, color="#da3633", linewidth=2,
                    linestyle="--", alpha=0.9, zorder=3)

    ax1.axvline(btc_cascades.index[0], color="#da3633",
                linewidth=2, linestyle="--",
                label=f"Cascade event ({len(btc_cascades)} total)")

    ax1.set_ylabel("CascadeWatch Risk Score (0-100)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # --- Plot 2: Distribution by risk level ---
    ax2 = axes[1]
    normal  = result[result["pre_cascade"] == 0]["risk_score"].dropna()
    cascade = result[result["pre_cascade"] == 1]["risk_score"].dropna()

    ax2.hist(normal,  bins=60, alpha=0.5, color="#58a6ff",
             label=f"Normal (n={len(normal):,})", density=True)
    ax2.hist(cascade, bins=20, alpha=0.8, color="#da3633",
             label=f"Pre-cascade (n={len(cascade)})", density=True)

    ax2.axvline(normal.mean(),  color="#58a6ff", linewidth=2,
                linestyle="--",
                label=f"Normal mean: {normal.mean():.1f}")
    ax2.axvline(cascade.mean(), color="#da3633", linewidth=2,
                linestyle="--",
                label=f"Cascade mean: {cascade.mean():.1f}")
    ax2.axvline(50, color="white", linewidth=1.5,
                linestyle=":", label="Alert threshold (50)")

    ax2.set_xlabel("Risk Score (0-100)")
    ax2.set_ylabel("Density")
    ax2.set_title("Risk Score Distribution: Normal vs Pre-Cascade")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = DOCS_DIR / "plot_ensemble_risk_score.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved {out}")


# ===========================================================================
# STEP 7: SAVE ENSEMBLE SCORES
# ===========================================================================

def save_scores(result):
    """
    Save the full dataset with risk scores to Parquet.
    This file is loaded by the Streamlit dashboard on Day 5 Part 2.
    """
    score_cols = [
        "symbol", "open", "high", "low", "close",
        "score_classifier", "score_anomaly", "score_severity",
        "score_survival", "score_fear",
        "risk_score", "risk_level",
        "pre_cascade", "cascade_event", "fear_index",
        "funding_rate", "funding_zscore", "rsi_14",
        "volume_ratio", "volatility_24h"
    ]

    # Keep only columns that exist
    available = [c for c in score_cols if c in result.columns]
    out_df    = result[available].copy()

    out_path  = FEATURES_DIR / "ensemble_scores.parquet"
    out_df.to_parquet(out_path)
    print(f"✅ Ensemble scores saved to {out_path}")
    print(f"   Rows: {len(out_df):,}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Day 5 Part 1: CascadeWatch Ensemble Risk Score")
    print("=" * 50)

    # Load everything
    (classifier, clf_scaler, anomaly, anom_scaler,
     severity, sev_scaler, survival, df) = load_everything()

    # Add cross-asset features for classifier
    print("\nAdding cross-asset features...")
    df = add_cross_asset_features(df)

    # Run all 5 models
    result = run_inference(df, classifier, clf_scaler, anomaly, anom_scaler,
                           severity, sev_scaler, survival)

    # Diagnostic: check for NaN in each score column
    score_cols = ["score_classifier", "score_anomaly",
                  "score_severity", "score_survival", "score_fear"]
    print("\nNaN check per score column:")
    for col in score_cols:
        n_nan = result[col].isna().sum()
        print(f"  {col:<25} NaN count: {n_nan}")

    # Fill any remaining NaN with 0 before combining
    result[score_cols] = result[score_cols].fillna(0)

    # Compute ensemble score
    result = compute_ensemble_score(result)

    # Backtest
    detected, total, avg_lead, auc = backtest(result)

    # Plot
    plot_ensemble(result, detected, total, avg_lead, auc)

    # Save scores for dashboard
    save_scores(result)

    print(f"\n✅ Day 5 Part 1 complete!")
    print(f"   Ensemble scores saved for Streamlit dashboard.")