"""
labeling.py
===========
Day 2 Part 2: Label cascade events in our feature dataset.

CONCEPT: Supervised ML needs labeled data — rows tagged as
"dangerous" (pre_cascade=1) or "normal" (pre_cascade=0).

FIX v2: Loosened threshold to 5% price drop over 4 hours.
Volume spike removed as a hard gate — it becomes a feature instead.
This captures realistic cascade events in hourly crypto data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

FEATURES_DIR = Path("data/features")

# CONCEPT: Threshold tuning is a real ML engineering decision.
# Too strict = almost no positive labels, model has nothing to learn from.
# Too loose = everything is labeled positive, model learns nothing useful.
# 5% drop in 4 hours is historically significant for leveraged crypto markets.
CASCADE_PRICE_DROP = 0.05   # 5% drop within 4 hours


def label_cascades(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Identify cascade events and label the 4 hours preceding each one.

    CONCEPT: We look FORWARD in time (future returns) to find cascades,
    then label the rows BEFORE them. This simulates the real prediction
    task: "given current conditions, will a cascade happen soon?"

    Args:
        df: Feature DataFrame for one symbol (output of feature_engineering.py)
        symbol: "BTC", "ETH", or "SOL"
    Returns:
        DataFrame with new columns: cascade_event, pre_cascade, hours_to_cascade
    """
    sym_df = df[df["symbol"] == symbol].copy()
    sym_df = sym_df.sort_index()

    # -----------------------------------------------------------------------
    # DETECT CASCADE EVENTS
    #
    # CONCEPT: pct_change(4) on the close column calculates the % change
    # from 4 rows ago to now. By shifting it -4 rows, each current row
    # gets the return that will happen over the NEXT 4 hours.
    # This is "lookahead" — only valid during labeling, never during inference.
    # -----------------------------------------------------------------------

    # Forward 4h return: what does price do in the next 4 hours from this row?
    sym_df["future_return_4h"] = sym_df["close"].pct_change(4).shift(-4)

    # A cascade = price drops 5%+ in the next 4 hours
    # pct_change returns negative numbers for drops, so we check < -0.05
    cascade_mask = sym_df["future_return_4h"] < -CASCADE_PRICE_DROP
    sym_df["cascade_event"] = cascade_mask.astype(int)

    # -----------------------------------------------------------------------
    # MERGE NEARBY CASCADE EVENTS
    #
    # CONCEPT: A real cascade lasts 2-6 hours. Without merging, we'd label
    # every hour of the drop as a separate "cascade event" — that's noise.
    # We group events that are within 6 hours of each other into one event,
    # keeping only the first hour (the true onset).
    #
    # This is called "event deduplication" — standard in anomaly detection.
    # -----------------------------------------------------------------------

    # Get all cascade timestamps
    cascade_times = sym_df[sym_df["cascade_event"] == 1].index.tolist()

    # Deduplicate: if two cascade times are within 6 hours, keep only the first
    deduped = []
    last_kept = None
    for t in cascade_times:
        if last_kept is None or (t - last_kept) > pd.Timedelta("6h"):
            deduped.append(t)
            last_kept = t

    # Reset cascade_event — only mark the deduplicated onset hours
    sym_df["cascade_event"] = 0
    for t in deduped:
        sym_df.loc[t, "cascade_event"] = 1

    # -----------------------------------------------------------------------
    # LABEL PRE-CASCADE WINDOWS
    #
    # CONCEPT: The model needs to fire BEFORE the cascade, not during it.
    # We label the 4 hours before each cascade onset as pre_cascade = 1.
    # These are the rows the model will learn to recognize as dangerous.
    # -----------------------------------------------------------------------

    sym_df["pre_cascade"] = 0

    for t in deduped:
        window_start = t - pd.Timedelta("4h")
        window_end   = t - pd.Timedelta("1h")
        sym_df.loc[window_start:window_end, "pre_cascade"] = 1

    # -----------------------------------------------------------------------
    # HOURS TO CASCADE
    #
    # CONCEPT: regression target for Model 4 (survival analysis).
    # Tells us not just "danger" but "how many hours until impact?"
    # -----------------------------------------------------------------------

    sym_df["hours_to_cascade"] = np.nan

    for t in deduped:
        for h in range(1, 5):
            target_time = t - pd.Timedelta(f"{h}h")
            if target_time in sym_df.index:
                sym_df.loc[target_time, "hours_to_cascade"] = h

    # -----------------------------------------------------------------------
    # SUMMARY — always print these, they go in your model card
    # -----------------------------------------------------------------------
    n_cascades = int(sym_df["cascade_event"].sum())
    n_pre      = int(sym_df["pre_cascade"].sum())
    n_total    = len(sym_df)
    imbalance  = n_pre / n_total * 100

    print(f"[{symbol}] Total rows:        {n_total:,}")
    print(f"[{symbol}] Cascade events:    {n_cascades}")
    print(f"[{symbol}] Pre-cascade rows:  {n_pre}")
    print(f"[{symbol}] Class imbalance:   {imbalance:.2f}% positive")
    print(f"[{symbol}] Ratio normal:cascade = {n_total - n_pre} : {n_pre}\n")

    return sym_df


def run_labeling_pipeline():
    """
    Load features, label cascades for BTC/ETH/SOL, save labeled dataset.
    """
    print("Loading feature data...")
    df = pd.read_parquet(FEATURES_DIR / "features_combined.parquet")

    labeled_parts = []
    for symbol in ["BTC", "ETH", "SOL"]:
        print(f"=== Labeling {symbol} ===")
        labeled = label_cascades(df, symbol)
        labeled_parts.append(labeled)

    combined = pd.concat(labeled_parts, axis=0).sort_index()

    # Drop rows where future return couldn't be computed (last 4 rows per symbol)
    combined = combined.dropna(subset=["cascade_event"])
    combined["cascade_event"] = combined["cascade_event"].astype(int)

    # Save labeled dataset — direct input to Day 3 model training
    out_path = FEATURES_DIR / "features_labeled.parquet"
    combined.to_parquet(out_path)

    total_cascades = combined["cascade_event"].sum()
    total_pre      = combined["pre_cascade"].sum()

    print(f"=== COMBINED SUMMARY ===")
    print(f"Total rows:           {len(combined):,}")
    print(f"Total cascade events: {int(total_cascades)}")
    print(f"Total pre-cascade:    {int(total_pre)}")
    print(f"Overall imbalance:    {total_pre/len(combined)*100:.2f}% positive")
    print(f"Saved to:             {out_path}")

    return combined


if __name__ == "__main__":
    df_labeled = run_labeling_pipeline()

    # Sanity check: show pre-cascade rows with key risk indicators
    pre = df_labeled[df_labeled["pre_cascade"] == 1]
    print(f"\nSample pre-cascade rows:")
    print(pre[["symbol", "close", "funding_rate", "funding_zscore",
               "volume_ratio", "pre_cascade", "hours_to_cascade"]].head(12))

    # Show the actual cascade dates — important for your README and model card
    print(f"\nCascade event dates:")
    cascades = df_labeled[df_labeled["cascade_event"] == 1]
    print(cascades[["symbol", "close", "future_return_4h"]].to_string())