"""Airflow DAG: daily fraud model retraining pipeline.

Runs at 02:00 UTC. Tasks mirror the manual steps in training/ and registry/.
The project root is expected at /opt/airflow/project in the Airflow container
(see docker-compose.yaml volumes / PYTHONPATH).
"""
import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

# Add project root to path so project modules are importable inside the container
_PROJECT_ROOT = os.getenv("AIRFLOW_PROJECT_ROOT", "/opt/airflow/project")
for _p in [
    _PROJECT_ROOT,
    os.path.join(_PROJECT_ROOT, "training"),
    os.path.join(_PROJECT_ROOT, "features"),
    os.path.join(_PROJECT_ROOT, "registry"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def _fetch_features(**context):
    """Materialize latest features into the online store and push path to XCom."""
    from materialize import materialize
    materialize()
    features_path = os.path.join(_PROJECT_ROOT, "data", "processed", "features.parquet")
    context["ti"].xcom_push(key="features_path", value=features_path)


def _train_model(**context):
    """Run full training pipeline, push run_id to XCom."""
    from train import train
    run_id = train(run_name="airflow_daily")
    context["ti"].xcom_push(key="mlflow_run_id", value=run_id)


def _evaluate(**context):
    """Pull metrics from MLflow and log pass/fail for the pipeline run."""
    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    auc_pr = metrics.get("auc_pr", 0.0)
    status = "PASS" if auc_pr >= 0.80 else "FAIL"
    print(f"Evaluate [{status}] run_id={run_id[:8]}  AUC-PR={auc_pr:.4f}")
    context["ti"].xcom_push(key="eval_status", value=status)


def _register_model(**context):
    """Gate check and register to Staging."""
    import mlflow
    from mlflow.tracking import MlflowClient
    from register import register

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics
    register(run_id, metrics)


def _notify(**context):
    """Print a full run summary regardless of upstream success/failure."""
    import mlflow
    from mlflow.tracking import MlflowClient

    run_id = context["ti"].xcom_pull(task_ids="train_model", key="mlflow_run_id")
    eval_status = context["ti"].xcom_pull(task_ids="evaluate", key="eval_status") or "UNKNOWN"

    if not run_id:
        print("No run_id available — training task may have failed")
        return

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    client = MlflowClient()
    run = client.get_run(run_id)
    m = run.data.metrics
    tags = run.data.tags

    print("\n" + "=" * 50)
    print("RETRAIN PIPELINE SUMMARY")
    print("=" * 50)
    print(f"  Run ID       : {run_id}")
    print(f"  AUC-PR       : {m.get('auc_pr', 'N/A'):.4f}")
    print(f"  F1           : {m.get('f1', 'N/A'):.4f}")
    print(f"  P@90R        : {m.get('precision_at_90_recall', 'N/A'):.4f}")
    print(f"  Eval status  : {eval_status}")
    print(f"  Promoted     : {tags.get('promoted', 'N/A')}")
    print("=" * 50)


with DAG(
    dag_id="retrain_fraud_model",
    default_args=default_args,
    description="Daily fraud model retraining with MLflow logging and registry gate",
    schedule_interval="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["fraud", "mlops"],
) as dag:

    fetch_features = PythonOperator(
        task_id="fetch_features",
        python_callable=_fetch_features,
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
    )

    evaluate = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=_register_model,
    )

    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    fetch_features >> train_model >> evaluate >> register_model >> notify
