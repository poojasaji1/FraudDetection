import numpy as np
import mlflow
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
)


def evaluate(model, X_val, y_val, run_id: str) -> dict:
    """Compute fraud-relevant metrics and log them to an active MLflow run.

    Must be called inside an active mlflow.start_run() context.
    Never computes accuracy — AUC-PR and precision@recall are the primary signals.
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    auc_pr = float(average_precision_score(y_val, y_pred_proba))
    f1 = float(f1_score(y_val, y_pred_binary, zero_division=0))

    # Precision at the operating point where recall first reaches 0.9
    precisions, recalls, _ = precision_recall_curve(y_val, y_pred_proba)
    mask = recalls >= 0.9
    p_at_90r = float(precisions[np.where(mask)[0].max()]) if mask.any() else 0.0

    metrics = {
        "auc_pr": auc_pr,
        "f1": f1,
        "precision_at_90_recall": p_at_90r,
        "threshold_used": 0.5,
    }

    mlflow.log_metrics(
        {k: v for k, v in metrics.items() if k != "threshold_used"}
    )
    mlflow.log_param("threshold_used", "0.5")

    print(f"\nEvaluation  (run {run_id[:8]}...)")
    print(f"  AUC-PR                 : {auc_pr:.4f}")
    print(f"  F1 @ 0.5               : {f1:.4f}")
    print(f"  Precision @ 90% Recall : {p_at_90r:.4f}")

    return metrics
