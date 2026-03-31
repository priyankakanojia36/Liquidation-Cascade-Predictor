# 🌊 CascadeWatch: ML-Powered Crypto Liquidation Cascade Predictor

> An end-to-end machine learning system that detects early warning signals of liquidation cascades in cryptocurrency derivatives markets, combining 5 ML models into a real-time ensemble risk score with a live Streamlit dashboard and automated AWS pipeline.

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![AWS](https://img.shields.io/badge/AWS-S3%20%7C%20DynamoDB%20%7C%20SNS-orange)](https://aws.amazon.com)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🎯 The Problem

On October 10, 2025, the crypto market lost over $2 billion in liquidations within 4 hours. Retail traders had no warning. Leveraged positions were force-closed in a cascade, each liquidation triggering the next and amplifying the crash.

**CascadeWatch is an early warning system for exactly this scenario.**

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Ensemble AUC-ROC | **0.7229** |
| Best single model AUC | 0.6266 (Fear Index) |
| Ensemble improvement | +9.6 percentage points vs best single model |
| Cascade events detected | 4 / 56 at threshold 50/100 |
| Average lead time | **3.8 hours** before cascade onset |
| Training data | 24,123 hourly rows (Apr 2025 to Mar 2026) |
| Symbols covered | BTC, ETH, SOL |

---

## 🏗️ System Architecture

```
Data Sources          Feature Engineering      ML Models            Output
─────────────         ───────────────────      ─────────            ──────
CryptoCompare ──┐
Coinbase      ──┤     19 features              Classifier    ──┐
CoinGecko     ──┼───► + cross-asset    ──────► Anomaly Det.  ──┤
dYdX          ──┤     signals                  Severity      ──┼──► Ensemble ──► Dashboard
Hyperliquid   ──┘     (Day 2)                  Survival      ──┤    Risk Score   Streamlit
                                               Fear Index    ──┘    (0-100)
                                                                        │
                                               AWS Pipeline ───────────┘
                                               S3 + DynamoDB + SNS + Airflow
```

---

## 🔬 The 5 Models

### Model 1: Cascade Risk Classifier
- **Algorithm:** Logistic Regression (best), Random Forest, XGBoost, LightGBM compared
- **Problem type:** Binary classification (pre-cascade vs normal)
- **Key technique:** SMOTE oversampling for 113:1 class imbalance
- **AUC-ROC:** 0.5563
- **Top features:** `consecutive_positive_funding`, `funding_zscore`, `btc_sol_corr_24h`

### Model 2: Isolation Forest Anomaly Detector
- **Algorithm:** Isolation Forest (unsupervised)
- **Problem type:** Anomaly detection, no labels used during training
- **AUC-ROC:** 0.5922
- **Lead time:** 6.2 hours average on detected events

### Model 3: Cascade Severity Predictor
- **Algorithm:** Gradient Boosting Regressor
- **Problem type:** Regression, predicts magnitude of price drop
- **MAE:** 1.95% average error on predicted severity
- **Top features:** `volume_ratio`, `returns_1h`, `price_vs_ma`

### Model 4: Survival Analysis (Time-to-Cascade)
- **Algorithm:** Cox Proportional Hazards (lifelines library)
- **Problem type:** Time-to-event prediction
- **C-index:** 0.5748
- **Key insight:** Volume ratio has hazard ratio above 1.0, increasing cascade speed

### Model 5: Market Fear Index
- **Algorithm:** Rule-based weighted composite (no training required)
- **Components:** Funding extremity (30%) + RSI deviation (20%) + Volatility (25%) + Price vs MA (15%) + Volume spike (10%)
- **AUC-ROC:** 0.6266 (best single model)
- **Analogy:** CNN Fear and Greed Index but calibrated for derivatives markets

---

## 🔑 Key Findings from EDA

1. **Funding signals predict occurrence.** `consecutive_positive_funding` and `funding_zscore` are the top classifier features. Leverage buildup precedes cascades.

2. **Volume signals predict severity.** `volume_ratio` is the #1 severity feature. Panic selling magnitude correlates with pre-crash volume spikes.

3. **RSI gives 2-hour warning.** RSI begins declining approximately 2 hours before cascade onset on average. This is the clearest leading indicator in the dataset.

4. **Cross-asset contagion matters.** `btc_sol_corr_24h` ranked 3rd in XGBoost importance. When BTC and SOL decouple, cascades are more likely.

5. **Ensemble beats every single model.** AUC 0.7229 vs 0.6266 best single. Combining 5 models with different failure modes improves overall detection.

6. **Linear correlation is misleading.** All features show near-zero Pearson correlation with the target. This confirms cascade risk is non-linear, validating the tree-based ensemble approach.

---

## 🗂️ Project Structure

```
cascade-predictor/
├── src/
│   ├── data_collection.py       # 5-source API pipeline (121K+ records)
│   ├── feature_engineering.py  # 19 features from raw market data
│   ├── labeling.py              # Cascade event detection and labeling
│   ├── train_classifier.py      # Model 1: Binary classifier (4 algorithms)
│   ├── train_anomaly.py         # Model 2: Isolation Forest
│   ├── train_severity.py        # Model 3: Severity regressor
│   ├── train_survival.py        # Model 4: Cox PH survival analysis
│   ├── train_sentiment.py       # Model 5: Market Fear Index
│   └── ensemble.py              # Ensemble risk score engine
├── dashboard/
│   └── app.py                   # Streamlit 4-page dashboard
├── airflow/
│   └── dags/cascade_pipeline.py # Hourly automated pipeline
├── notebooks/
│   └── 01_eda.ipynb             # Exploratory data analysis
├── data/
│   ├── raw/                     # 19 Parquet files (5 sources x 3 symbols)
│   ├── features/                # Engineered features and labels
│   └── models/                  # 8 trained model artifacts
└── docs/                        # EDA plots and model comparison charts
```

---

## ⚙️ Tech Stack

| Layer | Technologies |
|---|---|
| Languages | Python 3.11, SQL |
| ML and Data | Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, lifelines |
| Visualization | Streamlit, Plotly, Matplotlib, Seaborn |
| Data Engineering | Parquet, Apache Airflow, REST APIs |
| Cloud | AWS S3, DynamoDB, SNS, CloudWatch |
| Dev Tools | Git, Conda, VS Code, Jupyter |

---

## 🚀 How to Run

### 1. Clone and setup
```bash
git clone https://github.com/priyankakanojia36/Liquidation-Cascade-Predictor.git
cd Liquidation-Cascade-Predictor
conda create -n cascade python=3.11 -y
conda activate cascade
pip install -r requirements.txt
```

### 2. Collect data
```bash
python src/data_collection.py
```

### 3. Build features and labels
```bash
python src/feature_engineering.py
python src/labeling.py
```

### 4. Train all models
```bash
python src/train_classifier.py
python src/train_anomaly.py
python src/train_severity.py
python src/train_survival.py
python src/train_sentiment.py
python src/ensemble.py
```

### 5. Launch dashboard
```bash
streamlit run dashboard/app.py
```

Open `http://localhost:8501` in your browser.

---

## ☁️ AWS Infrastructure

| Resource | Service | Purpose |
|---|---|---|
| `cascadewatch-377228489449` | S3 | Raw data, features, and models storage |
| `CascadeWatchRiskScores` | DynamoDB | Real-time risk scores per symbol |
| `CascadeWatchAlerts` | SNS | Email alerts when risk score exceeds 75/100 |
| `cascade_hourly_pipeline` | Airflow | Automated hourly scoring pipeline |

---

## ⚠️ Limitations and Future Work

**Current limitations:**
- 1 year of training data (56 cascade events). More data would improve severity model R2 and classifier recall.
- Cascade labels derived from price proxy (5% drop in 4 hours) rather than actual liquidation feed data.
- Survival model score contribution currently zero due to feature alignment issue in ensemble (documented, fix planned for v2).

**V2 roadmap:**
- Extend to 3 years of data capturing 2022 LUNA/UST and FTX collapse events
- Add real-time Binance liquidation feed (requires VPN or non-US deployment)
- Deploy dashboard to Streamlit Cloud for public access
- Add Lambda function for on-demand risk scoring endpoint
- Implement model retraining trigger when performance degrades

---

## 👩‍💻 Author

**Priyanka Kanojia**
MS Analytics, Northeastern University (2025 to 2027)
7+ years experience in data science, product management, and iOS development

[![LinkedIn](https://img.shields.io/badge/LinkedIn-priyanka--datascience-blue)](https://linkedin.com/in/priyanka-datascience)
[![GitHub](https://img.shields.io/badge/GitHub-priyankakanojia36-black)](https://github.com/priyankakanojia36)
[![Tableau](https://img.shields.io/badge/Tableau-priyanka.kanojia1211-orange)](https://public.tableau.com/app/profile/priyanka.kanojia1211)
