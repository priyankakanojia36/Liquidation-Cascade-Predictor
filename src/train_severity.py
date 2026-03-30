"""
train_survival.py
=================
Day 4 — Model 4: Survival Analysis (Time-to-Cascade)

WHAT THIS MODEL DOES:
    Predicts HOW LONG until the next cascade occurs.
    Output: a hazard rate (instantaneous risk score)
    High hazard = cascade imminent
    Low hazard  = cascade unlikely soon

WHY SURVIVAL ANALYSIS INSTEAD OF REGULAR REGRESSION:
    Regular regression would require a known "time to cascade"
    for every row. But most rows never cascade — we don't know
    when (or if) they will. These are called CENSORED observations.
    Survival analysis is specifically designed to handle censored data.
    Regular regression would throw those rows away — wasting 99% of data.

ALGORITHM: Cox Proportional Hazards (lifelines library)
    - Industry standard for time-to-event prediction
    - Used in medical trials, churn prediction, fraud detection
    - Outputs hazard ratios per feature — highly interpretable

REROUTES TAKEN:
    - Originally planned to use hours_to_cascade directly as target
    - Switched to proper survival framework after recognizing that
      non-cascade rows are censored, not missing data
    - Added penalizer=0.1 to Cox model to prevent overfitting
      on small dataset (56 events)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
FEATURES_DIR = Path("data/features")
MODELS_DIR   = Path("data/models")
DOCS_DIR     = Path("docs")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Survival features — focused on leverage and momentum signals
# CONCEPT: We use fewer features than other models intentionally.
# Cox models are sensitive to multicollinearity (when features are
# highly correlated with each other). Using 6 well-chosen features
# gives more stable hazard ratio estimates than 15 noisy ones.
SURVIVAL_FEATURES = [
    "funding_zscore",               # leverage extremity
    "consecutive_positive_funding", # duration of leverage buildup
    "volume_ratio",                 # unusual activity
    "volatility_compression",       # coiling before explosion
    "rsi_14",                       # momentum weakening
    "price_position",               # price at top of range
]


# ===========================================================================
# STEP 1: LOAD AND PREPARE SURVIVAL DATA
# ===========================================================================

def load_and_prepare():
    """
    Prepare data in the format lifelines expects:
    - duration_col: how long until the event (or end of observation)
    - event_col:    did the event actually occur? (1=yes, 0=censored)

    CONCEPT: Every row needs a duration and an event flag.

    For PRE-CASCADE rows: duration = hours_to_cascade (1, 2, 3, or 4)
                          event    = 1 (cascade DID happen)

    For NORMAL rows:      duration = large number (we assign 720h = 30 days)
                          event    = 0 (censored — no cascade in our window)

    This tells the Cox model: "this row survived at least 720 hours
    without a cascade" — which is truthful and statistically valid.
    """
    print("Loading data...")
    df = pd.read_parquet(FEATURES_DIR / "features_labeled.parquet")
    df = df.sort_index()

    # ---------------------------------------------------------------------------
    # BUILD DURATION COLUMN
    # ---------------------------------------------------------------------------

    # For pre-cascade rows: hours_to_cascade is already 1, 2, 3, or 4
    # For cascade rows: duration = 0 (event happening right now)
    # For normal rows: we don't know → assign 720 (30 days in hours)
    #   CONCEPT: 720 is a censoring time — we're saying "we observed
    #   this row for 720 hours and no cascade happened in our window."
    #   The Cox model uses this information correctly — it knows these
    #   rows survived at least 720 hours, even if we don't know what
    #   happened after our observation window ended.

    CENSORING_TIME = 720  # 30 days in hours

    df["duration"] = df["hours_to_cascade"].fillna(CENSORING_TIME)
    df.loc[df["cascade_event"] == 1, "duration"] = 0.5
    # CONCEPT: We use 0.5 instead of 0 because lifelines requires
    # duration > 0. A cascade row gets duration=0.5h (30 minutes)
    # to represent "event happened almost immediately."

    # Clip to ensure no zero or negative durations slip through
    df["duration"] = df["duration"].clip(lower=0.1)

    # ---------------------------------------------------------------------------
    # BUILD EVENT COLUMN
    # ---------------------------------------------------------------------------

    # event=1 means the cascade DID occur (pre-cascade + cascade rows)
    # event=0 means censored (normal rows — no cascade observed)
    df["event"] = ((df["pre_cascade"] == 1) |
                   (df["cascade_event"] == 1)).astype(int)

    # ---------------------------------------------------------------------------
    # CHRONOLOGICAL 80/20 SPLIT
    # ---------------------------------------------------------------------------
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx]
    df_test   = df.iloc[split_idx:]

    # Keep only the columns we need
    cols_needed = SURVIVAL_FEATURES + ["duration", "event"]

    train_surv = df_train[cols_needed].dropna()
    test_surv  = df_test[cols_needed].dropna()

    print(f"Training rows:         {len(train_surv):,}")
    print(f"Training events:       {train_surv['event'].sum()} "
          f"({train_surv['event'].mean()*100:.2f}%)")
    print(f"Test rows:             {len(test_surv):,}")
    print(f"Test events:           {test_surv['event'].sum()} "
          f"({test_surv['event'].mean()*100:.2f}%)")

    return train_surv, test_surv, df


# ===========================================================================
# STEP 2: TRAIN COX PROPORTIONAL HAZARDS MODEL
# ===========================================================================

def train_cox_model(train_surv, test_surv):
    """
    Fit a Cox Proportional Hazards model and evaluate with C-index.

    CONCEPT: The Cox model estimates a "baseline hazard" that
    increases over time, multiplied by a feature-dependent factor.

    hazard(t) = baseline_hazard(t) × exp(β₁x₁ + β₂x₂ + ... + βₙxₙ)

    Where:
    - β (beta) = coefficient learned from data
    - x = feature value at this row
    - exp(β) = hazard ratio — the key output

    If exp(β) for funding_zscore = 1.3, it means:
    "A one-unit increase in funding_zscore increases cascade
    hazard by 30%." This is directly interpretable.

    penalizer=0.1 adds L2 regularization to prevent overfitting.
    Important here because we have only 168 events in training.
    """
    print("\nFitting Cox Proportional Hazards model...")

    cph = CoxPHFitter(penalizer=0.1)

    # .fit() takes the full dataframe with duration and event columns
    # It learns coefficients for each feature in SURVIVAL_FEATURES
    cph.fit(
        train_surv,
        duration_col="duration",
        event_col="event",
        show_progress=False
    )

    # ---------------------------------------------------------------------------
    # C-INDEX (CONCORDANCE INDEX)
    # CONCEPT: C-index measures ranking accuracy.
    # "Did the model correctly rank row A as higher risk than row B
    # when row A actually cascaded sooner?"
    # 0.5 = random ranking, 1.0 = perfect ranking
    # > 0.6 is considered useful for financial prediction
    # ---------------------------------------------------------------------------
    c_index_train = cph.score(train_surv,
                               scoring_method="concordance_index")
    c_index_test  = cph.score(test_surv,
                               scoring_method="concordance_index")

    print(f"\n📊 Survival Model Results:")
    print(f"  C-index (train): {c_index_train:.4f}")
    print(f"  C-index (test):  {c_index_test:.4f}")
    print(f"  (0.5 = random, 1.0 = perfect, >0.6 = useful)")

    # ---------------------------------------------------------------------------
    # HAZARD RATIOS — the most interpretable output
    # exp(coef) > 1.0 means feature INCREASES cascade risk
    # exp(coef) < 1.0 means feature DECREASES cascade risk
    # ---------------------------------------------------------------------------
    print(f"\n🔍 Hazard Ratios (how each feature affects cascade timing):")
    print(f"  {'Feature':<35} {'Hazard Ratio':>12} {'Interpretation'}")
    print(f"  {'-'*70}")

    summary = cph.summary[["coef", "exp(coef)", "p"]].copy()
    summary = summary.sort_values("exp(coef)", ascending=False)

    for feat, row in summary.iterrows():
        hr     = row["exp(coef)"]
        p_val  = row["p"]
        effect = "↑ increases risk" if hr > 1 else "↓ decreases risk"
        sig    = "**" if p_val < 0.05 else "  "
        print(f"  {sig}{feat:<33} {hr:>12.4f}   {effect}")

    print(f"\n  ** = statistically significant (p < 0.05)")

    return cph, c_index_train, c_index_test, summary


# ===========================================================================
# STEP 3: GENERATE SURVIVAL CURVES
# ===========================================================================

def plot_survival_curves(cph, train_surv, test_surv, summary,
                          c_index_test):
    """
    Plot 1: Survival curves for high-risk vs low-risk market conditions
    Plot 2: Hazard ratios with confidence intervals

    CONCEPT: A survival curve shows P(no cascade after T hours).
    High-risk conditions → curve drops faster (cascade happens sooner)
    Low-risk conditions  → curve stays high (cascade unlikely soon)
    """
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model 4: Survival Analysis — Time to Cascade",
                 fontsize=13, fontweight="bold")

    # --- Plot 1: Survival curves ---
    ax1 = axes[0]

    # Create two hypothetical market profiles to compare
    # HIGH RISK: high funding zscore, many consecutive positive hours,
    #            high volume, compressed volatility, low RSI, high position
    high_risk = pd.DataFrame({
        "funding_zscore":               [2.5],
        "consecutive_positive_funding": [500],
        "volume_ratio":                 [2.0],
        "volatility_compression":       [0.3],
        "rsi_14":                       [35.0],
        "price_position":               [0.8],
    })

    # LOW RISK: normal funding, low consecutive hours,
    #           normal volume, no compression, neutral RSI
    low_risk = pd.DataFrame({
        "funding_zscore":               [-0.5],
        "consecutive_positive_funding": [50],
        "volume_ratio":                 [0.9],
        "volatility_compression":       [1.0],
        "rsi_14":                       [52.0],
        "price_position":               [0.5],
    })

    # CONCEPT: predict_survival_function() returns P(survive past time T)
    # for each profile. We plot this curve over time.
    surv_high = cph.predict_survival_function(high_risk)
    surv_low  = cph.predict_survival_function(low_risk)

    ax1.plot(surv_high.index, surv_high.iloc[:, 0],
             color="#da3633", linewidth=2.5, label="High-risk conditions")
    ax1.plot(surv_low.index,  surv_low.iloc[:, 0],
             color="#3fb950", linewidth=2.5, label="Low-risk conditions")

    ax1.axvline(4,  color="#8b949e", linewidth=1, linestyle=":",
                label="4h window")
    ax1.axvline(24, color="#8b949e", linewidth=1, linestyle="--",
                label="24h window")
    ax1.set_xlabel("Hours")
    ax1.set_ylabel("P(no cascade)")
    ax1.set_title(f"Survival Curves\nC-index (test): {c_index_test:.4f}")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Hazard ratios ---
    ax2 = axes[1]
    hr_vals = summary["exp(coef)"].sort_values()
    colors  = ["#da3633" if v > 1 else "#58a6ff" for v in hr_vals]
    hr_vals.plot(kind="barh", ax=ax2, color=colors)
    ax2.axvline(1.0, color="white", linewidth=1.5,
                linestyle="--", label="No effect (HR=1)")
    ax2.set_title("Hazard Ratios by Feature\n"
                  "Red = increases risk, Blue = decreases risk")
    ax2.set_xlabel("Hazard Ratio exp(coef)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out = DOCS_DIR / "plot_survival_model.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n✅ Saved {out}")


# ===========================================================================
# STEP 4: SAVE MODEL
# ===========================================================================

def save_model(cph):
    joblib.dump(cph, MODELS_DIR / "survival_model.joblib")
    print(f"✅ Saved data/models/survival_model.joblib")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("Day 4 — Model 4: Survival Analysis")
    print("=" * 50)

    train_surv, test_surv, df = load_and_prepare()

    cph, c_train, c_test, summary = train_cox_model(train_surv, test_surv)

    plot_survival_curves(cph, train_surv, test_surv, summary, c_test)

    save_model(cph)

    print(f"\n✅ Day 4 Model 4 complete!")
    print(f"\n📌 Interview talking point:")
    print(f"   'Cox Proportional Hazards model outputs hazard rates")
    print(f"    per feature — directly interpretable as percentage")
    print(f"    increase in cascade risk per unit change in the feature.'")