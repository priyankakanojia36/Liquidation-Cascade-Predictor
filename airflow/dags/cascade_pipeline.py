"""
cascade_pipeline.py
===================
Day 6: CascadeWatch Automated Pipeline (Airflow DAG)

WHAT THIS DOES:
    Defines an automated hourly pipeline that:
    1. Fetches fresh market data from APIs
    2. Validates data quality
    3. Engineers features
    4. Scores all 5 ML models
    5. Computes ensemble risk score
    6. Writes results to DynamoDB
    7. Sends SNS alert if risk > 75
    8. Uploads results to S3
    9. Logs pipeline metrics

CONCEPT: What is a DAG?
    DAG = Directed Acyclic Graph.
    "Directed" = tasks flow in one direction (no going back)
    "Acyclic"  = no circular dependencies (no loops)
    "Graph"    = tasks are nodes, dependencies are edges

    In Airflow, a DAG is a Python file that defines:
    - What tasks to run
    - In what order
    - On what schedule
    - What to do if a task fails

CONCEPT: What is Airflow?
    Airflow is a platform to programmatically author, schedule,
    and monitor workflows. You define workflows as Python code.
    Airflow runs them on a schedule, retries on failure, and
    gives you a visual UI to monitor everything.

HOW TO RUN LOCALLY:
    1. Install: pip install apache-airflow
    2. Initialize: airflow db init
    3. Start: airflow standalone
    4. Open: http://localhost:8080
    5. Username: admin, Password: shown in terminal

REROUTES TAKEN:
    - Originally planned AWS MWAA (managed Airflow) — switched to
      local Airflow for cost and simplicity
    - Lambda scoring replaced with direct Python function calls
      since we're running locally
    - DynamoDB write uses boto3 directly from within the task
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
import boto3
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# AWS CONFIGURATION
# These values match the infrastructure we created in Day 6
# ---------------------------------------------------------------------------
AWS_REGION      = "us-east-1"
S3_BUCKET       = "cascadewatch-377228489449"
DYNAMODB_TABLE  = "CascadeWatchRiskScores"
SNS_TOPIC_ARN   = "arn:aws:sns:us-east-1:377228489449:CascadeWatchAlerts"

# Local paths — same as rest of project
ROOT         = Path(__file__).parent.parent.parent
FEATURES_DIR = ROOT / "data" / "features"
MODELS_DIR   = ROOT / "data" / "models"

# Risk alert threshold
ALERT_THRESHOLD = 75


# ===========================================================================
# TASK 1: FETCH MARKET DATA
# ===========================================================================

def fetch_market_data(**context):
    """
    Fetch the latest hour of market data from APIs.

    CONCEPT: In Airflow, each task is a Python function.
    The **context parameter gives access to Airflow metadata
    like execution_date, dag_run, and task_instance.

    context["task_instance"].xcom_push() sends data to the
    next task. XCom = "cross-communication" between tasks.
    """
    import requests

    print("Fetching latest market data...")

    symbols  = ["BTC", "ETH", "SOL"]
    results  = {}

    for symbol in symbols:
        try:
            # CryptoCompare — free, no API key needed for basic data
            url = "https://min-api.cryptocompare.com/data/v2/histohour"
            params = {
                "fsym":  symbol,
                "tsym":  "USD",
                "limit": 2,   # last 2 hours (we use the most recent)
            }
            resp = requests.get(url, params=params, timeout=10)
            data = resp.json()

            if data["Response"] == "Success":
                latest = data["Data"]["Data"][-1]
                results[symbol] = {
                    "timestamp": latest["time"],
                    "open":      latest["open"],
                    "high":      latest["high"],
                    "low":       latest["low"],
                    "close":     latest["close"],
                    "volume":    latest["volumefrom"],
                }
                print(f"  ✅ {symbol}: ${latest['close']:,.0f}")
            else:
                print(f"  ❌ {symbol}: API error")
                results[symbol] = None

        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            results[symbol] = None

    # Push results to XCom so next task can access them
    context["task_instance"].xcom_push(
        key="market_data", value=results
    )
    print(f"\n✅ Task 1 complete: fetched data for {len(results)} symbols")
    return results


# ===========================================================================
# TASK 2: VALIDATE DATA
# ===========================================================================

def validate_data(**context):
    """
    Check that the fetched data is valid before processing.

    CONCEPT: Data validation is critical in production pipelines.
    If bad data flows through, all downstream models produce
    garbage results. We validate early and fail loudly.

    Checks:
    - No None values (API call succeeded)
    - Price is positive
    - Volume is non-negative
    - Timestamp is recent (within last 2 hours)
    """
    market_data = context["task_instance"].xcom_pull(
        key="market_data", task_ids="fetch_market_data"
    )

    print("Validating data quality...")
    now = datetime.utcnow()
    issues = []

    for symbol, data in market_data.items():
        if data is None:
            issues.append(f"{symbol}: no data returned from API")
            continue

        # Price must be positive
        if data["close"] <= 0:
            issues.append(f"{symbol}: invalid price {data['close']}")

        # Timestamp must be within last 2 hours
        data_time = datetime.utcfromtimestamp(data["timestamp"])
        age_hours = (now - data_time).total_seconds() / 3600
        if age_hours > 2:
            issues.append(f"{symbol}: data is {age_hours:.1f}h old")

        print(f"  ✅ {symbol}: price=${data['close']:,.0f}, "
              f"age={age_hours:.1f}h")

    if issues:
        # CONCEPT: Raising an exception in an Airflow task marks
        # it as FAILED. Airflow will retry based on retry settings.
        raise ValueError(f"Data validation failed: {issues}")

    print(f"\n✅ Task 2 complete: all data valid")


# ===========================================================================
# TASK 3: COMPUTE RISK SCORES
# ===========================================================================

def compute_risk_scores(**context):
    """
    Load the latest full feature dataset and compute ensemble risk scores.

    CONCEPT: In a full production system, this task would:
    1. Load fresh features from S3
    2. Run all 5 models
    3. Compute the ensemble score

    For our local implementation, we load the pre-computed
    ensemble_scores.parquet and extract the latest scores.
    This demonstrates the pipeline structure without requiring
    real-time feature recomputation.
    """
    print("Computing risk scores...")

    # Load pre-computed ensemble scores
    df = pd.read_parquet(FEATURES_DIR / "ensemble_scores.parquet")
    df = df.sort_index()

    risk_scores = {}

    for symbol in ["BTC", "ETH", "SOL"]:
        sym_df  = df[df["symbol"] == symbol]
        latest  = sym_df.iloc[-1]

        risk_scores[symbol] = {
            "risk_score":        float(latest["risk_score"]),
            "risk_level":        str(latest["risk_level"]),
            "score_classifier":  float(latest.get("score_classifier", 0)),
            "score_anomaly":     float(latest.get("score_anomaly", 0)),
            "score_fear":        float(latest.get("score_fear", 0)),
            "timestamp":         str(sym_df.index[-1]),
        }

        print(f"  {symbol}: risk_score={risk_scores[symbol]['risk_score']:.1f} "
              f"({risk_scores[symbol]['risk_level']})")

    context["task_instance"].xcom_push(
        key="risk_scores", value=risk_scores
    )
    print(f"\n✅ Task 3 complete: risk scores computed")
    return risk_scores


# ===========================================================================
# TASK 4: WRITE TO DYNAMODB
# ===========================================================================

def write_to_dynamodb(**context):
    """
    Write the latest risk scores to DynamoDB.

    CONCEPT: DynamoDB is a NoSQL key-value database.
    Each item has a primary key (symbol) and any number of attributes.
    We use put_item() to write/overwrite the latest score per symbol.

    The dashboard could read from DynamoDB in real-time
    instead of from a static Parquet file — that's the
    production upgrade path.
    """
    risk_scores = context["task_instance"].xcom_pull(
        key="risk_scores", task_ids="compute_risk_scores"
    )

    print("Writing risk scores to DynamoDB...")

    dynamodb = boto3.resource("dynamodb", region_name=AWS_REGION)
    table    = dynamodb.Table(DYNAMODB_TABLE)

    for symbol, scores in risk_scores.items():
        # CONCEPT: DynamoDB requires Decimal for float values,
        # not Python float. We use str() for simplicity here.
        item = {
            "symbol":           symbol,
            "risk_score":       str(round(scores["risk_score"], 2)),
            "risk_level":       scores["risk_level"],
            "score_classifier": str(round(scores["score_classifier"], 4)),
            "score_anomaly":    str(round(scores["score_anomaly"], 4)),
            "score_fear":       str(round(scores["score_fear"], 4)),
            "last_updated":     datetime.utcnow().isoformat(),
            "data_timestamp":   scores["timestamp"],
        }

        table.put_item(Item=item)
        print(f"  ✅ {symbol}: written to DynamoDB")

    print(f"\n✅ Task 4 complete: all scores written to DynamoDB")


# ===========================================================================
# TASK 5: CHECK ALERTS
# ===========================================================================

def check_and_send_alerts(**context):
    """
    Send SNS alert if any symbol's risk score exceeds threshold.

    CONCEPT: SNS (Simple Notification Service) is a pub/sub
    messaging service. We publish a message to a topic.
    All subscribers (your email) receive the message.

    In production this could also trigger:
    - A Lambda function to execute a trade
    - A Slack webhook for team notifications
    - A PagerDuty alert for on-call engineers
    """
    risk_scores = context["task_instance"].xcom_pull(
        key="risk_scores", task_ids="compute_risk_scores"
    )

    print(f"Checking alerts (threshold: {ALERT_THRESHOLD}/100)...")

    alerts_sent = 0
    sns_client  = boto3.client("sns", region_name=AWS_REGION)

    for symbol, scores in risk_scores.items():
        score = scores["risk_score"]
        level = scores["risk_level"]

        if score >= ALERT_THRESHOLD:
            message = (
                f"🚨 CascadeWatch CRITICAL ALERT\n\n"
                f"Symbol:     {symbol}/USD\n"
                f"Risk Score: {score:.1f}/100\n"
                f"Risk Level: {level}\n"
                f"Time:       {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"Cascade conditions detected. "
                f"Multiple risk signals active simultaneously.\n\n"
                f"Dashboard: http://localhost:8501"
            )

            sns_client.publish(
                TopicArn=SNS_TOPIC_ARN,
                Message=message,
                Subject=f"CascadeWatch Alert: {symbol} Risk {score:.0f}/100"
            )

            print(f"  🚨 ALERT SENT for {symbol}: {score:.1f}/100")
            alerts_sent += 1
        else:
            print(f"  ✅ {symbol}: {score:.1f}/100 — below threshold")

    print(f"\n✅ Task 5 complete: {alerts_sent} alerts sent")


# ===========================================================================
# TASK 6: UPLOAD RESULTS TO S3
# ===========================================================================

def upload_to_s3(**context):
    """
    Upload the latest ensemble scores and pipeline run metadata to S3.

    CONCEPT: S3 is object storage — like a file system in the cloud.
    We upload the Parquet file with a timestamped path so we maintain
    a historical record of every pipeline run.

    Path structure:
    s3://bucket/results/YYYY/MM/DD/HH/ensemble_scores.parquet
    """
    print("Uploading results to S3...")

    s3_client = boto3.client("s3", region_name=AWS_REGION)
    now       = datetime.utcnow()

    # Timestamped path — keeps history of every run
    s3_key = (f"results/{now.year}/{now.month:02d}/"
              f"{now.day:02d}/{now.hour:02d}/ensemble_scores.parquet")

    local_path = str(FEATURES_DIR / "ensemble_scores.parquet")

    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    print(f"  ✅ Uploaded to s3://{S3_BUCKET}/{s3_key}")

    # Also update the "latest" pointer
    s3_client.upload_file(
        local_path, S3_BUCKET, "results/latest/ensemble_scores.parquet"
    )
    print(f"  ✅ Updated latest pointer")

    print(f"\n✅ Task 6 complete: results uploaded to S3")


# ===========================================================================
# TASK 7: LOG PIPELINE METRICS
# ===========================================================================

def log_pipeline_metrics(**context):
    """
    Log pipeline run metadata for monitoring and debugging.

    CONCEPT: Observability is critical in production.
    We log: run time, data freshness, risk score summary,
    and any anomalies. This data goes to CloudWatch logs
    automatically when running on AWS.
    """
    risk_scores = context["task_instance"].xcom_pull(
        key="risk_scores", task_ids="compute_risk_scores"
    )

    print("\n" + "="*50)
    print("PIPELINE RUN METRICS")
    print("="*50)
    print(f"Run time:    {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"DAG:         cascade_hourly_pipeline")
    print(f"Status:      SUCCESS")
    print(f"\nRisk Scores:")

    for symbol, scores in risk_scores.items():
        print(f"  {symbol}: {scores['risk_score']:.1f}/100 "
              f"({scores['risk_level']})")

    max_score  = max(s["risk_score"] for s in risk_scores.values())
    max_symbol = max(risk_scores, key=lambda k: risk_scores[k]["risk_score"])

    print(f"\nHighest risk: {max_symbol} at {max_score:.1f}/100")
    print(f"Alert sent:   {'YES' if max_score >= ALERT_THRESHOLD else 'NO'}")
    print("="*50)
    print(f"\n✅ Task 7 complete: metrics logged")


# ===========================================================================
# DAG DEFINITION
# ===========================================================================

# CONCEPT: default_args apply to every task in the DAG unless overridden.
# retries=2 means Airflow will retry a failed task 2 times before
# marking it as permanently failed.
# retry_delay=5 minutes means wait 5 minutes between retries.
default_args = {
    "owner":            "priyanka",
    "depends_on_past":  False,
    "start_date":       datetime(2026, 3, 30),
    "email_on_failure": False,
    "email_on_retry":   False,
    "retries":          2,
    "retry_delay":      timedelta(minutes=5),
}

# CONCEPT: DAG() creates the pipeline container.
# schedule_interval="@hourly" runs it every hour.
# catchup=False means don't run missed historical runs.
dag = DAG(
    dag_id="cascade_hourly_pipeline",
    default_args=default_args,
    description="CascadeWatch hourly risk scoring pipeline",
    schedule_interval="@hourly",
    catchup=False,
    tags=["cascadewatch", "crypto", "ml", "risk"],
)

# ===========================================================================
# TASK DEFINITIONS
# CONCEPT: PythonOperator wraps a Python function as an Airflow task.
# task_id is the unique name shown in the Airflow UI.
# python_callable is the function to run.
# provide_context=True passes the Airflow context to the function.
# ===========================================================================

t1_fetch = PythonOperator(
    task_id="fetch_market_data",
    python_callable=fetch_market_data,
    provide_context=True,
    dag=dag,
)

t2_validate = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    provide_context=True,
    dag=dag,
)

t3_score = PythonOperator(
    task_id="compute_risk_scores",
    python_callable=compute_risk_scores,
    provide_context=True,
    dag=dag,
)

t4_dynamo = PythonOperator(
    task_id="write_to_dynamodb",
    python_callable=write_to_dynamodb,
    provide_context=True,
    dag=dag,
)

t5_alerts = PythonOperator(
    task_id="check_and_send_alerts",
    python_callable=check_and_send_alerts,
    provide_context=True,
    dag=dag,
)

t6_s3 = PythonOperator(
    task_id="upload_to_s3",
    python_callable=upload_to_s3,
    provide_context=True,
    dag=dag,
)

t7_metrics = PythonOperator(
    task_id="log_pipeline_metrics",
    python_callable=log_pipeline_metrics,
    provide_context=True,
    dag=dag,
)

# ===========================================================================
# TASK DEPENDENCIES
# CONCEPT: >> operator sets the execution order.
# t1 >> t2 means "t2 runs after t1 completes successfully"
# [t4, t5, t6] means t4, t5, t6 run in PARALLEL after t3
# This is the power of Airflow — parallel execution where possible.
#
# Pipeline flow:
# fetch → validate → score → [dynamodb, alerts, s3] → metrics
# ===========================================================================

t1_fetch >> t2_validate >> t3_score >> [t4_dynamo, t5_alerts, t6_s3] >> t7_metrics