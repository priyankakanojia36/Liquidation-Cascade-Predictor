"""
dashboard/app.py
================
CascadeWatch Streamlit Dashboard — Mode 2 (Live Refresh)

HOW TO RUN:
    cd ~/projects/cascade-predictor
    streamlit run dashboard/app.py

PAGES:
    1. Risk Overview   — current risk gauge + alert status
    2. Market Data     — price + funding rate charts
    3. Model Insights  — individual model score breakdown
    4. Backtest        — historical detection performance

MODE 2 UPGRADE:
    Added "Refresh Live Data" button in sidebar.
    When clicked: fetches live market data, scores all 5 models,
    writes results to DynamoDB, updates the gauge in real time.
    Zero AWS cost unless button is clicked.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import boto3
import sys

# Add src/ to path so we can import live_scorer
sys.path.append(str(Path(__file__).parent.parent / "src"))

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CascadeWatch",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "data" / "models"
DOCS_DIR     = ROOT / "docs"

AWS_REGION     = "us-east-1"
DYNAMODB_TABLE = "CascadeWatchRiskScores"


# ===========================================================================
# DATA LOADING
# ===========================================================================

@st.cache_data
def load_data():
    """Load historical ensemble scores from Parquet."""
    df = pd.read_parquet(FEATURES_DIR / "ensemble_scores.parquet")
    df = df.sort_index()
    return df


@st.cache_data(ttl=0)
def load_live_data_from_dynamodb():
    """
    Load the latest risk scores from DynamoDB.

    CONCEPT: ttl=0 means never cache — always fetch fresh from DynamoDB.
    This ensures every time the user refreshes, they see the latest
    scores that were written by live_scorer.py.
    """
    try:
        dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
        table    = dynamodb.Table(DYNAMODB_TABLE)
        response = table.scan()
        items    = response.get("Items", [])

        if not items:
            return None

        live = {}
        for item in items:
            sym = item["symbol"]
            live[sym] = {
                "risk_score":       float(item.get("risk_score", 0)),
                "risk_level":       item.get("risk_level", "LOW"),
                "close":            float(item.get("close", 0)),
                "funding_rate":     float(item.get("funding_rate", 0)),
                "last_updated":     item.get("last_updated", ""),
                "score_classifier": float(item.get("score_classifier", 0)),
                "score_anomaly":    float(item.get("score_anomaly", 0)),
                "score_fear":       float(item.get("score_fear", 0)),
                "score_severity":   float(item.get("score_severity", 0)),
            }
        return live

    except Exception as e:
        return None


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def get_risk_color(score):
    if score >= 75:   return "#da3633"
    elif score >= 50: return "#d29922"
    elif score >= 25: return "#3fb950"
    else:             return "#58a6ff"


def get_risk_label(score):
    if score >= 75:   return "🔴 CRITICAL"
    elif score >= 50: return "🟠 HIGH"
    elif score >= 25: return "🟡 ELEVATED"
    else:             return "🟢 LOW"


def make_gauge(score, title="CascadeWatch Risk Score"):
    color = get_risk_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={"text": title, "font": {"size": 18, "color": "#e6edf3"}},
        number={"font": {"size": 48, "color": color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#8b949e",
                "tickfont": {"color": "#8b949e"}
            },
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "#161b22",
            "borderwidth": 2,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0,  25], "color": "#0d419d"},
                {"range": [25, 50], "color": "#1a7f37"},
                {"range": [50, 75], "color": "#9e6a03"},
                {"range": [75, 100],"color": "#6e1c1c"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.75,
                "value": score
            }
        }
    ))
    fig.update_layout(
        height=280,
        paper_bgcolor="#0d1117",
        font={"color": "#e6edf3"},
        margin=dict(t=60, b=20, l=30, r=30)
    )
    return fig


# ===========================================================================
# SIDEBAR
# ===========================================================================

def render_sidebar(df):
    """
    Sidebar with symbol selector, live refresh button, and project info.
    Returns: (symbol, live_data)
    """
    st.sidebar.image(
        "https://img.shields.io/badge/CascadeWatch-ML%20Risk%20System-blue",
        use_container_width=True
    )
    st.sidebar.title("🌊 CascadeWatch")
    st.sidebar.caption("ML-Powered Crypto Liquidation Cascade Predictor")
    st.sidebar.divider()

    symbol = st.sidebar.selectbox(
        "Select Symbol",
        options=["BTC", "ETH", "SOL"],
        index=0
    )

    st.sidebar.divider()

    # ------------------------------------------------------------------
    # LIVE REFRESH BUTTON
    # CONCEPT: When clicked, runs live_scorer.py which:
    # 1. Fetches fresh API data
    # 2. Scores all 5 models
    # 3. Writes to DynamoDB
    # 4. Clears Streamlit cache so the gauge updates
    # AWS cost: ~$0.000001 per click (3 DynamoDB writes)
    # ------------------------------------------------------------------
    st.sidebar.markdown("**Live Data**")

    if st.sidebar.button("🔄 Refresh Live Data", type="primary"):
        with st.spinner("Fetching live market data..."):
            try:
                from live_scorer import run_live_scoring
                results = run_live_scoring()
                st.cache_data.clear()
                st.sidebar.success("✅ Live data updated!")
                for sym, data in results.items():
                    st.sidebar.caption(
                        f"{sym}: {data['risk_score']}/100 "
                        f"({data['risk_level']}) @ "
                        f"${data['close']:,.0f}"
                    )
            except Exception as e:
                st.sidebar.error(f"Refresh failed: {e}")

    # Load current DynamoDB data to show status
    live = load_live_data_from_dynamodb()
    if live:
        st.sidebar.caption("🟢 Live data available")
        for sym, data in live.items():
            updated = data["last_updated"][:16].replace("T", " ")
            st.sidebar.caption(
                f"{sym}: **{data['risk_score']}/100** "
                f"({data['risk_level']}) — {updated} UTC"
            )
    else:
        st.sidebar.caption("⚪ Showing historical data")

    st.sidebar.divider()
    st.sidebar.markdown("**Model Performance**")
    st.sidebar.metric("Ensemble AUC", "0.7229", "+0.09 vs best single")
    st.sidebar.metric("Cascades Detected", "4/56", "3.8h avg lead time")
    st.sidebar.markdown("**Data Period**")
    st.sidebar.markdown(
        "<small>Apr 2025 to Mar 2026 · 24,123 rows</small>",
        unsafe_allow_html=True
    )

    st.sidebar.divider()
    st.sidebar.markdown("**Built by**")
    st.sidebar.markdown(
        "[Priyanka Kanojia](https://linkedin.com/in/priyanka-datascience)"
    )
    st.sidebar.markdown(
        "[GitHub](https://github.com/priyankakanojia36/"
        "Liquidation-Cascade-Predictor)"
    )

    return symbol, live


# ===========================================================================
# PAGE 1: RISK OVERVIEW
# ===========================================================================

def page_risk_overview(df, symbol, live):
    """
    Main dashboard page. Shows live score if available,
    otherwise falls back to historical data.
    """
    st.title("🌊 CascadeWatch Risk Overview")

    # Use live DynamoDB data if available, otherwise use historical
    if live and symbol in live:
        live_sym      = live[symbol]
        current_score = live_sym["risk_score"]
        current_price = live_sym["close"]
        funding_rate  = live_sym["funding_rate"]
        last_updated  = live_sym["last_updated"][:16].replace("T", " ")
        st.caption(
            f"Live data for {symbol}/USD — last updated {last_updated} UTC"
        )
        data_source = "live"
    else:
        sym_df        = df[df["symbol"] == symbol].copy()
        latest        = sym_df.iloc[-1]
        current_score = latest["risk_score"]
        current_price = latest["close"]
        funding_rate  = latest["funding_rate"]
        st.caption(
            f"Historical data for {symbol}/USD — "
            f"click Refresh for live data"
        )
        data_source = "historical"

    risk_label = get_risk_label(current_score)

    col_gauge, col1, col2, col3 = st.columns([2, 1, 1, 1])

    with col_gauge:
        fig_gauge = make_gauge(round(current_score, 1))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col1:
        st.metric(label="Risk Level", value=risk_label)
        st.metric(
            label="Current Price",
            value=f"${current_price:,.0f}",
            delta="Live" if data_source == "live" else "Historical"
        )

    with col2:
        st.metric(
            label="Funding Rate",
            value=f"{funding_rate*100:.4f}%",
            delta="vs neutral 0%"
        )
        if data_source == "historical":
            sym_df = df[df["symbol"] == symbol].copy()
            latest = sym_df.iloc[-1]
            st.metric(
                label="RSI (14h)",
                value=f"{latest['rsi_14']:.1f}",
                delta="overbought >70" if latest["rsi_14"] > 70
                      else "oversold <30" if latest["rsi_14"] < 30
                      else "neutral"
            )

    with col3:
        if live and symbol in live:
            live_sym = live[symbol]
            st.metric(
                label="Classifier Score",
                value=f"{live_sym['score_classifier']:.3f}"
            )
            st.metric(
                label="Fear Score",
                value=f"{live_sym['score_fear']:.3f}"
            )
        elif data_source == "historical":
            sym_df = df[df["symbol"] == symbol].copy()
            latest = sym_df.iloc[-1]
            st.metric(
                label="Volume Ratio",
                value=f"{latest['volume_ratio']:.2f}x",
                delta="vs 24h average"
            )
            st.metric(
                label="Fear Index",
                value=f"{latest['fear_index']:.1f}/100"
            )

    st.divider()

    # Risk score timeline from historical data
    st.subheader(f"{symbol} Risk Score Timeline (Historical)")
    sym_df       = df[df["symbol"] == symbol].copy()
    cascade_rows = sym_df[sym_df["cascade_event"] == 1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sym_df.index, y=sym_df["risk_score"],
        mode="lines", name="Risk Score",
        line=dict(color="#58a6ff", width=1),
        fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"
    ))

    fig.add_hrect(y0=75, y1=100, fillcolor="#da3633",
                  opacity=0.08, line_width=0, annotation_text="CRITICAL")
    fig.add_hrect(y0=50, y1=75, fillcolor="#d29922",
                  opacity=0.06, line_width=0, annotation_text="HIGH")
    fig.add_hrect(y0=25, y1=50, fillcolor="#3fb950",
                  opacity=0.04, line_width=0, annotation_text="ELEVATED")

    for t in cascade_rows.index:
        fig.add_vline(x=t, line_color="#da3633",
                      line_dash="dash", line_width=1.5, opacity=0.8)

    # Live score shown in sidebar — no chart marker needed

    fig.update_layout(
        height=350,
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        yaxis=dict(range=[0, 100], title="Risk Score (0-100)",
                   gridcolor="#21262d"),
        xaxis=dict(title="Date", gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22"),
        margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2: MARKET DATA
# ===========================================================================

def page_market_data(df, symbol):
    st.title(f"📈 {symbol} Market Data")
    sym_df       = df[df["symbol"] == symbol].copy()
    cascade_rows = sym_df[sym_df["cascade_event"] == 1]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["Price (USD)", "Funding Rate (%)", "RSI (14h)"],
        vertical_spacing=0.06
    )

    fig.add_trace(go.Scatter(
        x=sym_df.index, y=sym_df["close"],
        name="Price", line=dict(color="#58a6ff", width=1)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sym_df.index, y=sym_df["funding_rate"] * 100,
        name="Funding Rate %", line=dict(color="#3fb950", width=0.8),
        fill="tozeroy", fillcolor="rgba(63,185,80,0.1)"
    ), row=2, col=1)

    fig.add_hline(y=0, line_color="#8b949e", line_width=0.5, row=2, col=1)

    fig.add_trace(go.Scatter(
        x=sym_df.index, y=sym_df["rsi_14"],
        name="RSI", line=dict(color="#bc8cff", width=0.8)
    ), row=3, col=1)

    fig.add_hline(y=70, line_color="#da3633", line_dash="dot",
                  line_width=1, row=3, col=1)
    fig.add_hline(y=30, line_color="#da3633", line_dash="dot",
                  line_width=1, row=3, col=1)

    for t in cascade_rows.index:
        for row in [1, 2, 3]:
            fig.add_vline(x=t, line_color="#da3633",
                          line_dash="dash", line_width=1,
                          opacity=0.7, row=row, col=1)

    fig.update_layout(
        height=700, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"), showlegend=True,
        legend=dict(bgcolor="#161b22"),
    )
    fig.update_xaxes(gridcolor="#21262d")
    fig.update_yaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

    if len(cascade_rows) > 0:
        st.subheader(f"{symbol} Cascade Events Detected")
        display_cols = ["close", "risk_score", "risk_level",
                        "funding_rate", "rsi_14", "volume_ratio"]
        available = [c for c in display_cols if c in cascade_rows.columns]
        st.dataframe(
            cascade_rows[available].style.format({
                "close": "${:,.0f}", "risk_score": "{:.1f}",
                "funding_rate": "{:.6f}", "rsi_14": "{:.1f}",
                "volume_ratio": "{:.2f}x"
            }),
            use_container_width=True
        )


# ===========================================================================
# PAGE 3: MODEL INSIGHTS
# ===========================================================================

def page_model_insights(df, symbol, live):
    st.title("🧠 Model Insights")
    st.caption("How each of the 5 models contributes to the ensemble score")

    sym_df = df[df["symbol"] == symbol].copy()
    latest = sym_df.iloc[-1]

    # Use live scores if available
    if live and symbol in live:
        live_sym = live[symbol]
        score_data = {
            "Classifier (w=0.35)":  live_sym["score_classifier"] * 0.35 * 100,
            "Anomaly (w=0.25)":     live_sym["score_anomaly"]    * 0.25 * 100,
            "Fear Index (w=0.20)":  live_sym["score_fear"]       * 0.20 * 100,
            "Severity (w=0.10)":    live_sym["score_severity"]   * 0.10 * 100,
            "Survival (w=0.10)":    0.0,
        }
        st.caption("Showing live model scores from DynamoDB")
    else:
        score_data = {
            "Classifier (w=0.35)":  latest.get("score_classifier", 0) * 0.35 * 100,
            "Anomaly (w=0.25)":     latest.get("score_anomaly", 0)    * 0.25 * 100,
            "Fear Index (w=0.20)":  latest.get("score_fear", 0)       * 0.20 * 100,
            "Severity (w=0.10)":    latest.get("score_severity", 0)   * 0.10 * 100,
            "Survival (w=0.10)":    latest.get("score_survival", 0)   * 0.10 * 100,
        }

    st.subheader("Current Score Decomposition")
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure(go.Bar(
            x=list(score_data.values()),
            y=list(score_data.keys()),
            orientation="h",
            marker_color=["#58a6ff", "#3fb950",
                          "#d29922", "#bc8cff", "#f78166"],
            text=[f"{v:.1f}" for v in score_data.values()],
            textposition="outside"
        ))
        fig.update_layout(
            title="Score Contribution per Model (out of 100)",
            height=300, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(range=[0, 50], gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"),
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Raw model scores (0-1)**")
        if live and symbol in live:
            live_sym = live[symbol]
            raw_scores = {
                "Model": ["Classifier", "Anomaly", "Fear Index",
                          "Severity", "Survival"],
                "Raw Score": [
                    f"{live_sym['score_classifier']:.4f}",
                    f"{live_sym['score_anomaly']:.4f}",
                    f"{live_sym['score_fear']:.4f}",
                    f"{live_sym['score_severity']:.4f}",
                    "0.0000",
                ],
                "Weight": ["35%", "25%", "20%", "10%", "10%"],
                "Contribution": [f"{v:.1f}" for v in score_data.values()]
            }
        else:
            raw_scores = {
                "Model": ["Classifier", "Anomaly", "Fear Index",
                          "Severity", "Survival"],
                "Raw Score": [
                    f"{latest.get('score_classifier', 0):.4f}",
                    f"{latest.get('score_anomaly', 0):.4f}",
                    f"{latest.get('score_fear', 0):.4f}",
                    f"{latest.get('score_severity', 0):.4f}",
                    f"{latest.get('score_survival', 0):.4f}",
                ],
                "Weight": ["35%", "25%", "20%", "10%", "10%"],
                "Contribution": [f"{v:.1f}" for v in score_data.values()]
            }
        st.dataframe(pd.DataFrame(raw_scores), use_container_width=True)

    st.divider()
    st.subheader("Model Score Timelines (Historical)")

    fig2 = go.Figure()
    for name, (col, color) in {
        "Classifier": ("score_classifier", "#58a6ff"),
        "Anomaly":    ("score_anomaly",    "#3fb950"),
        "Fear Index": ("score_fear",       "#d29922"),
        "Severity":   ("score_severity",   "#bc8cff"),
    }.items():
        if col in sym_df.columns:
            fig2.add_trace(go.Scatter(
                x=sym_df.index, y=sym_df[col],
                name=name, line=dict(color=color, width=0.8), opacity=0.8
            ))

    for t in sym_df[sym_df["cascade_event"] == 1].index:
        fig2.add_vline(x=t, line_color="#da3633",
                       line_dash="dash", line_width=1.5, opacity=0.8)

    fig2.update_layout(
        height=350, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        font=dict(color="#e6edf3"),
        yaxis=dict(range=[0, 1], title="Score (0-1)", gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d"),
        legend=dict(bgcolor="#161b22"), margin=dict(t=20, b=40)
    )
    st.plotly_chart(fig2, use_container_width=True)


# ===========================================================================
# PAGE 4: BACKTEST PERFORMANCE
# ===========================================================================

def page_backtest(df):
    st.title("📊 Backtest Performance")
    st.caption("How well did CascadeWatch detect historical cascade events?")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ensemble AUC-ROC", "0.7229", "+9pp vs best single model")
    col2.metric("Cascades Detected", "4 / 56", "at threshold 50/100")
    col3.metric("Avg Lead Time", "3.8 hours", "before cascade onset")
    col4.metric("Best Single Model", "Fear Index 0.6266", "AUC before ensemble")

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Score: Normal vs Pre-Cascade")
        normal  = df[df["pre_cascade"] == 0]["risk_score"].dropna()
        cascade = df[df["pre_cascade"] == 1]["risk_score"].dropna()

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=normal, name=f"Normal (n={len(normal):,})",
            opacity=0.6, marker_color="#58a6ff",
            histnorm="probability density", nbinsx=60
        ))
        fig.add_trace(go.Histogram(
            x=cascade, name=f"Pre-Cascade (n={len(cascade)})",
            opacity=0.8, marker_color="#da3633",
            histnorm="probability density", nbinsx=25
        ))
        fig.add_vline(x=normal.mean(), line_color="#58a6ff",
                      line_dash="dash",
                      annotation_text=f"Normal mean: {normal.mean():.1f}")
        fig.add_vline(x=cascade.mean(), line_color="#da3633",
                      line_dash="dash",
                      annotation_text=f"Cascade mean: {cascade.mean():.1f}")
        fig.add_vline(x=50, line_color="white", line_dash="dot",
                      annotation_text="Alert threshold")
        fig.update_layout(
            barmode="overlay", height=350,
            paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(title="Risk Score", gridcolor="#21262d"),
            yaxis=dict(title="Density", gridcolor="#21262d"),
            legend=dict(bgcolor="#161b22"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Model Comparison Summary")
        model_results = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "XGBoost",
                      "LightGBM", "Anomaly Detector", "Fear Index",
                      "🌊 Ensemble"],
            "AUC-ROC": [0.5563, 0.5307, 0.5228, 0.5352,
                        0.5922, 0.6266, 0.7229],
        }).sort_values("AUC-ROC", ascending=True)

        fig2 = go.Figure(go.Bar(
            x=model_results["AUC-ROC"], y=model_results["Model"],
            orientation="h",
            marker_color=["#da3633" if m == "🌊 Ensemble" else "#58a6ff"
                          for m in model_results["Model"]],
            text=[f"{v:.4f}" for v in model_results["AUC-ROC"]],
            textposition="outside"
        ))
        fig2.add_vline(x=0.5, line_color="#8b949e", line_dash="dot",
                       annotation_text="Random (0.5)")
        fig2.update_layout(
            height=350, paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
            font=dict(color="#e6edf3"),
            xaxis=dict(range=[0.45, 0.80], title="AUC-ROC",
                       gridcolor="#21262d"),
            yaxis=dict(gridcolor="#21262d"), margin=dict(t=20, r=60)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Key Findings")
    findings = {
        "🔑 Funding signals predict occurrence":
            "consecutive_positive_funding and funding_zscore are the top "
            "features in the classifier. Leverage buildup precedes cascades.",
        "📊 Volume signals predict severity":
            "volume_ratio is the #1 severity feature. Panic selling "
            "magnitude correlates with pre-crash volume spikes.",
        "⏱️ RSI gives 2-hour warning":
            "RSI begins declining approximately 2 hours before cascade "
            "onset on average. The clearest leading indicator in the dataset.",
        "🌐 Cross-asset contagion matters":
            "btc_sol_corr_24h ranked 3rd in XGBoost importance. When BTC "
            "and SOL decouple, cascades are more likely.",
        "🤝 Ensemble beats every single model":
            "AUC 0.7229 ensemble vs 0.6266 best single. Combining 5 "
            "models with different failure modes improves overall detection."
    }
    for title, desc in findings.items():
        with st.expander(title):
            st.write(desc)


# ===========================================================================
# MAIN APP
# ===========================================================================

def main():
    try:
        df = load_data()
    except FileNotFoundError:
        st.error("ensemble_scores.parquet not found. Run src/ensemble.py first.")
        st.stop()

    # Sidebar returns symbol AND live data from DynamoDB
    symbol, live = render_sidebar(df)

    page = st.sidebar.radio(
        "Navigate",
        ["🏠 Risk Overview",
         "📈 Market Data",
         "🧠 Model Insights",
         "📊 Backtest Performance"],
        index=0
    )

    if page == "🏠 Risk Overview":
        page_risk_overview(df, symbol, live)
    elif page == "📈 Market Data":
        page_market_data(df, symbol)
    elif page == "🧠 Model Insights":
        page_model_insights(df, symbol, live)
    elif page == "📊 Backtest Performance":
        page_backtest(df)


if __name__ == "__main__":
    main()