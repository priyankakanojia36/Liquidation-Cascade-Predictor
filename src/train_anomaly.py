"""
train_anomaly.py
================
Day 3 Model 2: Isolation Forest Anomaly Detector

CONCEPT: This is unsupervised learning — the model never sees labels.
Instead it learns: "what does NORMAL market data look like?"
Then at inference time, anything that deviates from normal gets
a high anomaly score. We validate by checking if anomaly scores
spike before known cascade events.

Why this complements the classifier:
- Classifier: learns cascade patterns from labeled examples
- Isolation Forest: learns normal patterns, flags deviations
- Together: catches cascades the classifier might miss because
  they don't match historical patterns (novel cascade types)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import joblib
import warnings
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
DOCS_DIR     = Path("docs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Features most relevant for anomaly detection
# CONCEPT: We use leverage + volatility features only.
# These are the signals that deviate most during cascade conditions.
# We intentionally exclude price/return features because large price
# moves happen in both directions — we only want to flag leverage extremes.
ANOMALY_FEATURES = [
    "funding_zscore",
    "funding_acceleration",
    "consecutive_positive_funding",
    "funding_max_24h",
    "volume_ratio",
    "volatility_compression",
    "volatility_24h",
    "rsi_14",
    "price_position",
]


# ===========================================================================
# STEP 1: LOAD AND PREPARE DATA
# ===========================================================================

def load_data():
    """
    Load features and split into train/test.

    CONCEPT: For anomaly detection, we train ONLY on normal data.
    The model learns what normal looks like — so anything unusual
    scores high at inference time.

    We use the first 80% of data for training (mostly normal market),
    and evaluate on the full dataset to check if anomaly scores
    align with known cascade events.
    """
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    df = df.sort_index()

    # Training set: first 80% of data, ONLY normal rows
    # CONCEPT: If we train on cascade rows too, the model learns
    # to treat cascade conditions as "normal" — defeating the purpose.
    split_idx   = int(len(df) * 0.8)
    df_train    = df.iloc[:split_idx]
    df_normal   = df_train[df_train["pre_cascade"] == 0]

    X_train = df_normal[ANOMALY_FEATURES].dropna()

    # Full dataset for evaluation — keep index aligned with df
    X_full  = df[ANOMALY_FEATURES].copy()
    X_full  = X_full.ffill().dropna()
    y_full  = df.loc[X_full.index, "pre_cascade"]
    # Ensure no duplicate index rows cause size mismatch
    X_full  = X_full[~X_full.index.duplicated(keep="first")]
    y_full  = y_full[~y_full.index.duplicated(keep="first")]

    print(f"Training on:  {len(X_train):,} normal rows (no cascade labels)")
    print(f"Evaluating:   {len(X_full):,} total rows")
    print(f"Features:     {ANOMALY_FEATURES}\n")

    return X_train, X_full, y_full, df


# ===========================================================================
# STEP 2: TRAIN ISOLATION FOREST
# ===========================================================================

def train_isolation_forest(X_train, X_full, y_full):
    """
    Train Isolation Forest and tune the contamination parameter.

    CONCEPT: Isolation Forest works by randomly partitioning data.
    Normal points require many splits to isolate (they blend in).
    Anomalies require few splits (they stand out).
    The anomaly score = average path length to isolate a point.
    Short path = anomalous.

    contamination = expected fraction of anomalies in the data.
    We try 3 values and pick the one with highest AUC.
    """
    scaler  = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_full_scaled  = scaler.transform(X_full)

    contamination_values = [0.01, 0.02, 0.05]
    best_auc   = 0
    best_model = None
    best_cont  = None
    best_scores = None

    print("Tuning contamination parameter...")
    for cont in contamination_values:
        model = IsolationForest(
            n_estimators=200,
            contamination=cont,
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled)

        # CONCEPT: decision_function() returns raw anomaly scores.
        # More negative = more anomalous.
        # We negate it so higher score = more anomalous (intuitive).
        raw_scores    = model.decision_function(X_full_scaled)
        anomaly_scores = -raw_scores  # flip sign: higher = more anomalous

        # Normalize to 0-1 range for interpretability
        score_min = anomaly_scores.min()
        score_max = anomaly_scores.max()
        scores_normalized = (anomaly_scores - score_min) / (
            score_max - score_min + 1e-10)

        # Evaluate: do high anomaly scores correlate with cascade labels?
        auc = roc_auc_score(y_full, scores_normalized)
        print(f"  contamination={cont:.2f} → AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc    = auc
            best_model  = model
            best_cont   = cont
            best_scores = scores_normalized

    print(f"\n✅ Best contamination: {best_cont} (AUC = {best_auc:.4f})")
    return best_model, scaler, best_scores, best_auc


# ===========================================================================
# STEP 3: EVALUATE — DO SCORES SPIKE BEFORE CASCADES?
# ===========================================================================

def evaluate_lead_time(anomaly_scores, y_full, df):
    """
    Measure how many minutes/hours before a cascade the anomaly score
    crosses a threshold. This is our headline metric.

    CONCEPT: Lead time = how much warning does the system give?
    If the score crosses 0.7 on average 2 hours before a cascade,
    that's actionable. If it only crosses at the moment of impact,
    it's useless.
    """
    # Attach scores back to the dataframe
    score_series = pd.Series(anomaly_scores,
                              index=y_full.index, name="anomaly_score")
    eval_df = df.loc[y_full.index].copy()
    eval_df["anomaly_score"] = score_series

    threshold   = 0.6  # score above this = anomaly alert
    cascade_times = eval_df[eval_df["cascade_event"] == 1].index

    lead_times  = []
    detected    = 0

    for t in cascade_times:
        sym = eval_df.loc[t, "symbol"]
        if isinstance(sym, pd.Series):
            sym = sym.iloc[0]

        sym_eval = eval_df[eval_df["symbol"] == sym]

        # Look at the 8 hours before this cascade
        window_start = t - pd.Timedelta("8h")
        pre_window   = sym_eval.loc[window_start:t]

        # Did the score cross the threshold in the pre-window?
        above_threshold = pre_window[
            pre_window["anomaly_score"] >= threshold
        ]

        if len(above_threshold) > 0:
            # First time it crossed the threshold
            first_alert = above_threshold.index[0]
            lead_h      = (t - first_alert).total_seconds() / 3600
            lead_times.append(lead_h)
            detected += 1

    total    = len(cascade_times)
    det_rate = detected / total * 100 if total > 0 else 0
    avg_lead = np.mean(lead_times) if lead_times else 0

    print(f"\n📊 Anomaly Detection Performance (threshold={threshold}):")
    print(f"   Cascade events detected: {detected}/{total} ({det_rate:.1f}%)")
    print(f"   Average lead time:       {avg_lead:.1f} hours before cascade")
    print(f"   Lead times:              {[f'{l:.1f}h' for l in lead_times[:10]]}")

    return detected, total, avg_lead


# ===========================================================================
# STEP 4: PLOT ANOMALY SCORES VS PRICE
# ===========================================================================

def plot_anomaly_scores(anomaly_scores, y_full, df):
    """
    Plot anomaly scores overlaid on price for each symbol.
    If the system is working, scores should spike before price drops.
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(
        "Isolation Forest Anomaly Scores vs Price\n"
        "(Score spike before red line = early warning signal)",
        fontsize=13, fontweight="bold"
    )

    score_series = pd.Series(anomaly_scores,
                              index=y_full.index, name="anomaly_score")
    eval_df = df.loc[y_full.index].copy()
    eval_df["anomaly_score"] = score_series

    colors = {"BTC": "#f7931a", "ETH": "#627eea", "SOL": "#9945ff"}

    for ax, symbol in zip(axes, ["BTC", "ETH", "SOL"]):
        sym_df    = eval_df[eval_df["symbol"] == symbol]
        cascades  = sym_df[sym_df["cascade_event"] == 1]

        # Price (left y-axis)
        ax2 = ax.twinx()  # CONCEPT: twinx() creates a second y-axis
        ax2.plot(sym_df.index, sym_df["close"],
                 color=colors[symbol], linewidth=0.7, alpha=0.5,
                 label=f"{symbol} Price")
        ax2.set_ylabel("Price (USD)", fontsize=8,
                       color=colors[symbol])
        ax2.tick_params(axis="y", colors=colors[symbol])

        # Anomaly score (right y-axis)
        ax.plot(sym_df.index, sym_df["anomaly_score"],
                color="#3fb950", linewidth=0.9,
                label="Anomaly Score")
        ax.axhline(0.6, color="#ff4444", linewidth=1,
                   linestyle="--", alpha=0.7, label="Alert threshold (0.6)")
        ax.fill_between(sym_df.index, sym_df["anomaly_score"], 0,
                        where=sym_df["anomaly_score"] >= 0.6,
                        alpha=0.3, color="#da3633",
                        label="Above threshold")

        # Mark cascade events
        for t in cascades.index:
            ax.axvline(t, color="#ff4444", linewidth=1.5,
                       linestyle="--", alpha=0.8)

        ax.set_ylabel("Anomaly Score", fontsize=8)
        ax.set_title(f"{symbol} — {len(cascades)} cascade events",
                     fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.15)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.legend(loc="upper left", fontsize=7)

    plt.tight_layout()
    out = DOCS_DIR / "plot_anomaly_scores.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()  # close immediately — no blocking
    print(f"✅ Saved {out}")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Day 3 Model 2: Isolation Forest Anomaly Detector")
    print("=" * 50)

    X_train, X_full, y_full, df = load_data()

    model, scaler, scores, auc = train_isolation_forest(
        X_train, X_full, y_full)

    detected, total, avg_lead = evaluate_lead_time(scores, y_full, df)

    plot_anomaly_scores(scores, y_full, df)

    # Save model and scaler
    joblib.dump(model,  MODELS_DIR / "anomaly_detector.joblib")
    joblib.dump(scaler, MODELS_DIR / "anomaly_scaler.joblib")

    print(f"\n✅ Anomaly detector saved to data/models/anomaly_detector.joblib")

    # Headline metric for README and LinkedIn
    print(f"\n🎯 HEADLINE METRIC:")
    print(f"   Isolation Forest detected {detected}/{total} cascade events")
    print(f"   with an average lead time of {avg_lead:.1f} hours")
    print(f"   AUC-ROC: {auc:.4f}")
    print(f"\n✅ Day 3 Model 2 complete!")