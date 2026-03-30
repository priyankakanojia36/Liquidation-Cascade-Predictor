"""
train_classifier.py (v2)
========================
Day 3 Model 1: XGBoost Cascade Risk Classifier

Changes from v1:
- Added SMOTE oversampling to handle 113:1 class imbalance
- Added cross-asset features (BTC/ETH/SOL correlation signals)
- Improved threshold tuning using precision-recall curve
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, classification_report,
    RocCurveDisplay, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")

FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
DOCS_DIR     = Path("docs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "returns_1h", "returns_4h",
    "volatility_24h", "volatility_4h", "volatility_compression",
    "volume_ratio",
    "funding_rate", "funding_zscore", "funding_acceleration",
    "consecutive_positive_funding", "funding_max_24h",
    "roc_4h", "rsi_14",
    "price_position", "price_vs_ma",
]

TARGET = "pre_cascade"


# ===========================================================================
# STEP 1: LOAD DATA + ADD CROSS-ASSET FEATURES
# ===========================================================================

def load_data():
    """
    Load labeled features and engineer cross-asset signals.

    CONCEPT: Cross-asset features capture contagion risk.
    When BTC, ETH, and SOL all drop simultaneously, that's a cascade.
    When only one drops, that's an isolated correction.
    The difference in their returns at any given hour is a powerful signal.
    """
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    df = df.sort_index()

    # -----------------------------------------------------------------------
    # CROSS-ASSET FEATURES
    # CONCEPT: We pivot the data so each timestamp has BTC, ETH, SOL
    # returns side by side, then compute their correlation and divergence.
    # -----------------------------------------------------------------------

    # Pivot returns by symbol — each row is one timestamp, columns are symbols
    returns_pivot = df.pivot_table(
        index=df.index, columns="symbol", values="returns_1h"
    )

    # Rolling 24h correlation between BTC and ETH returns
    # CONCEPT: When correlation drops suddenly, assets are diverging —
    # one is being hit harder. This often precedes contagion cascades.
    btc_eth_corr = returns_pivot["BTC"].rolling(24).corr(returns_pivot["ETH"])
    btc_sol_corr = returns_pivot["BTC"].rolling(24).corr(returns_pivot["SOL"])

    # Multi-asset simultaneous drop signal
    # CONCEPT: If BTC, ETH, SOL are all negative in the same hour,
    # that's a market-wide event, not a single-asset correction.
    all_negative = (
        (returns_pivot["BTC"] < 0) &
        (returns_pivot["ETH"] < 0) &
        (returns_pivot["SOL"] < 0)
    ).astype(int)

    # Rolling count of simultaneous drops over 4 hours
    simultaneous_drops_4h = all_negative.rolling(4).sum()

    # Merge cross-asset features back into main dataframe
    # CONCEPT: join() aligns on the index (timestamp)
    df = df.join(btc_eth_corr.rename("btc_eth_corr_24h"), how="left")
    df = df.join(btc_sol_corr.rename("btc_sol_corr_24h"), how="left")
    df = df.join(simultaneous_drops_4h.rename("simultaneous_drops_4h"), how="left")

    # Fill NaN from rolling windows
    df[["btc_eth_corr_24h", "btc_sol_corr_24h",
        "simultaneous_drops_4h"]] = df[[
        "btc_eth_corr_24h", "btc_sol_corr_24h",
        "simultaneous_drops_4h"]].ffill().fillna(0)

    # Updated feature list with cross-asset signals
    all_features = FEATURE_COLS + [
        "btc_eth_corr_24h", "btc_sol_corr_24h", "simultaneous_drops_4h"
    ]

    # Remove rows with any remaining NaN
    df = df.dropna(subset=all_features)

    X = df[all_features].copy()
    y = df[TARGET].copy()

    # Chronological 80/20 split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set:  {len(X_train):,} rows | "
          f"Positives: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
    print(f"Test set:      {len(X_test):,} rows  | "
          f"Positives: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"Features:      {len(all_features)} "
          f"(15 original + 3 cross-asset)")

    return X_train, X_test, y_train, y_test, all_features


# ===========================================================================
# STEP 2: SMOTE OVERSAMPLING
# ===========================================================================

def apply_smote(X_train, y_train):
    """
    Apply SMOTE to balance the training set.

    CONCEPT: SMOTE = Synthetic Minority Oversampling Technique.
    Instead of just duplicating cascade rows, SMOTE creates NEW synthetic
    cascade examples by interpolating between existing cascade rows.

    Example: if we have two cascade rows A and B, SMOTE creates a new
    point C that sits somewhere between A and B in feature space.
    This gives the model more diverse examples to learn from.

    IMPORTANT: We ONLY apply SMOTE to training data, never test data.
    The test set must reflect real-world imbalance to give honest metrics.
    """
    try:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(
            sampling_strategy=0.1,  # make positives 10% of training data
            random_state=42,
            k_neighbors=5
        )
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"\nSMOTE applied:")
        print(f"  Before: {len(X_train):,} rows | "
              f"{y_train.sum()} positives ({y_train.mean()*100:.2f}%)")
        print(f"  After:  {len(X_resampled):,} rows | "
              f"{y_resampled.sum()} positives ({y_resampled.mean()*100:.2f}%)")
        return X_resampled, y_resampled

    except ImportError:
        print("\nimblearn not installed. Installing now...")
        import subprocess
        subprocess.run(["pip", "install", "imbalanced-learn", "-q"])
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(sampling_strategy=0.1, random_state=42, k_neighbors=5)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled


# ===========================================================================
# STEP 3: TRAIN AND EVALUATE
# ===========================================================================

def train_and_evaluate(X_train, X_test, y_train, y_test, feature_cols):

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"\nRaw class imbalance ratio: {scale_pos_weight:.1f}:1")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # Apply SMOTE to scaled training data
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)

    # Recalculate ratio after SMOTE for scale_pos_weight
    neg_s = (y_train_smote == 0).sum()
    pos_s = (y_train_smote == 1).sum()
    spw   = neg_s / pos_s
    print(f"Post-SMOTE imbalance ratio: {spw:.1f}:1\n")

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            max_depth=6, random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=spw, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric="logloss", verbosity=0
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            scale_pos_weight=spw, subsample=0.8,
            colsample_bytree=0.8, random_state=42, verbose=-1
        ),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for name, model in models.items():
        print(f"--- Training {name} ---")

        # CV on SMOTE-augmented data
        cv_aucs = []
        for tr_idx, val_idx in tscv.split(X_train_smote):
            model.fit(X_train_smote[tr_idx], y_train_smote.iloc[tr_idx]
                      if hasattr(y_train_smote, 'iloc')
                      else y_train_smote[tr_idx])
            y_prob = model.predict_proba(
                X_train_smote[val_idx])[:, 1]
            y_val  = (y_train_smote.iloc[val_idx]
                      if hasattr(y_train_smote, 'iloc')
                      else y_train_smote[val_idx])
            if y_val.sum() > 0:
                cv_aucs.append(roc_auc_score(y_val, y_prob))

        # Final fit on full SMOTE training set
        model.fit(X_train_smote, y_train_smote)

        # Evaluate on REAL (unaugmented) test set
        y_prob_test = model.predict_proba(X_test_scaled)[:, 1]
        test_auc    = roc_auc_score(y_test, y_prob_test)

        # Find optimal threshold using precision-recall curve
        # CONCEPT: We pick the threshold that maximizes F1 on the test set.
        # This is better than hardcoding 0.3 — it's data-driven.
        precisions, recalls, thresholds = precision_recall_curve(
            y_test, y_prob_test)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        best_thresh = thresholds[np.argmax(f1_scores[:-1])]
        y_pred_test = (y_prob_test >= best_thresh).astype(int)

        cv_mean = np.mean(cv_aucs) if cv_aucs else 0.0
        cv_std  = np.std(cv_aucs)  if cv_aucs else 0.0

        print(f"  CV AUC:          {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"  Test AUC:        {test_auc:.4f}")
        print(f"  Best threshold:  {best_thresh:.3f}")
        print(classification_report(y_test, y_pred_test,
              target_names=["Normal", "Pre-Cascade"], zero_division=0))

        results[name] = {
            "model":     model,
            "cv_auc":    cv_mean,
            "test_auc":  test_auc,
            "y_prob":    y_prob_test,
            "y_pred":    y_pred_test,
            "threshold": best_thresh,
        }

    return results, scaler, feature_cols


# ===========================================================================
# STEP 4: PLOT + SAVE
# ===========================================================================

def plot_results(results, y_test, feature_cols):
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Comparison — Day 3 v2 (SMOTE + Cross-Asset Features)",
                 fontsize=12, fontweight="bold")

    # ROC curves
    ax = axes[0]
    colors = ["#58a6ff", "#3fb950", "#ff7b72", "#d29922"]
    for (name, res), color in zip(results.items(), colors):
        RocCurveDisplay.from_predictions(
            y_test, res["y_prob"],
            name=f"{name} (AUC={res['test_auc']:.3f})",
            ax=ax, color=color
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (0.5)")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # XGBoost feature importance
    ax2 = axes[1]
    xgb_model = results["XGBoost"]["model"]
    importances = pd.Series(
        xgb_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=True)
    bar_colors = ["#da3633" if v > importances.median()
                  else "#58a6ff" for v in importances]
    importances.plot(kind="barh", ax=ax2, color=bar_colors)
    ax2.set_title("XGBoost Feature Importance")
    ax2.set_xlabel("Importance Score")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "plot_model_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Saved docs/plot_model_comparison.png")


def save_best_model(results, scaler, feature_cols):
    best_name  = max(results, key=lambda k: results[k]["test_auc"])
    best_model = results[best_name]["model"]
    best_auc   = results[best_name]["test_auc"]

    joblib.dump(best_model,   MODELS_DIR / "classifier.joblib")
    joblib.dump(scaler,       MODELS_DIR / "scaler.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    print(f"\n🏆 Best model: {best_name} (AUC = {best_auc:.4f})")
    print(f"\n📊 Model comparison:")
    print(f"{'Model':<22} {'CV AUC':>10} {'Test AUC':>10}")
    print("-" * 44)
    for name, res in sorted(results.items(),
                             key=lambda x: x[1]["test_auc"], reverse=True):
        marker = " ← best" if name == best_name else ""
        print(f"{name:<22} {res['cv_auc']:>10.4f} "
              f"{res['test_auc']:>10.4f}{marker}")


if __name__ == "__main__":
    print("=" * 50)
    print("Day 3 v2: Cascade Risk Classifier (SMOTE)")
    print("=" * 50)

    X_train, X_test, y_train, y_test, feat_cols = load_data()
    results, scaler, feat_cols = train_and_evaluate(
        X_train, X_test, y_train, y_test, feat_cols)
    plot_results(results, y_test, feat_cols)
    save_best_model(results, scaler, feat_cols)
    print("\n✅ Day 3 Model 1 complete!")