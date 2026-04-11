"""Compute evaluation metrics comparing LLM decisions with manual labels."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd

from trial_project.context import results_dir
from trial_project.labeling.storage import load_labels
from trial_project.matching.save_eligibility import eligibility_file

logger = logging.getLogger(__name__)


class MetricsResult(TypedDict):
    """Metrics result dictionary."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    total_pairs: int
    evaluation_timestamp: str


def compare_decisions(model_name: str = "gpt-5-mini") -> MetricsResult:
    """
    Compare LLM eligibility decisions with manual labels.

    Loads manual labels and LLM eligibility decisions, filters by model_name,
    merges on (patient_id, trial_id), excludes "skip" labels, and computes
    binary classification metrics.

    Args:
        model_name: Model name to filter eligibility decisions by.

    Returns:
        MetricsResult dict with accuracy, precision, recall, f1, and pair counts.

    Raises:
        ValueError: If no labels or decisions are found, or if no matching pairs exist.
    """
    # Load manual labels
    labels_df = load_labels()
    if labels_df.empty:
        raise ValueError("No manual labels found. Please label some pairs first.")

    logger.info(f"Loaded {len(labels_df)} total labels")

    # Load eligibility decisions
    if not eligibility_file.exists():
        raise ValueError(
            f"Eligibility decisions file not found: {eligibility_file}"
        )

    decisions_df = pd.read_parquet(eligibility_file)
    logger.info(f"Loaded {len(decisions_df)} total eligibility decisions")

    # Filter decisions by model_name
    decisions_df = decisions_df[decisions_df["model_name"] == model_name]
    if decisions_df.empty:
        raise ValueError(
            f"No eligibility decisions found for model_name='{model_name}'"
        )

    logger.info(f"Filtered to {len(decisions_df)} decisions for model '{model_name}'")

    # Exclude "skip" labels
    labels_df = labels_df[labels_df["label"] != "skip"]
    logger.info(f"After excluding 'skip' labels: {len(labels_df)} labels remain")

    # Merge on (patient_id, trial_id)
    merged = labels_df.merge(
        decisions_df,
        on=["patient_id", "trial_id"],
        how="inner",
    )

    if merged.empty:
        raise ValueError(
            f"No matching pairs between labels and decisions (model_name='{model_name}')"
        )

    logger.info(f"Merged into {len(merged)} matching pairs")

    # Create binary vectors: label -> True/False, eligible -> True/False
    # "eligible" label -> 1, "ineligible" label -> 0
    y_true = (merged["label"] == "eligible").astype(int).values
    y_pred = merged["eligible"].astype(int).values

    # Compute confusion matrix
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    total = tp + fp + tn + fn

    # Compute metrics
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    result: MetricsResult = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "total_pairs": int(total),
        "evaluation_timestamp": datetime.utcnow().isoformat(),
    }

    logger.info(f"Metrics computed: {result}")
    return result


def save_metrics(
    metrics: MetricsResult, output_path: str | Path | None = None
) -> Path:
    """
    Save metrics to JSON file.

    Args:
        metrics: MetricsResult dictionary to save.
        output_path: Output file path. If None, uses auto-generated name in results_dir.

    Returns:
        Path to the saved file.
    """
    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"evaluation_metrics_{timestamp}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Metrics saved to {output_path}")
    return output_path


def print_metrics_report(metrics: MetricsResult) -> None:
    """
    Print formatted metrics report to console.

    Args:
        metrics: MetricsResult dictionary to print.
    """
    lines = [
        "=" * 60,
        "EVALUATION METRICS REPORT",
        "=" * 60,
        "",
        "Classification Metrics:",
        f"  Accuracy:   {metrics['accuracy']:.4f}",
        f"  Precision:  {metrics['precision']:.4f}",
        f"  Recall:     {metrics['recall']:.4f}",
        f"  F1 Score:   {metrics['f1']:.4f}",
        "",
        "Confusion Matrix:",
        f"  True Positives:   {metrics['true_positives']:6d}",
        f"  False Positives:  {metrics['false_positives']:6d}",
        f"  True Negatives:   {metrics['true_negatives']:6d}",
        f"  False Negatives:  {metrics['false_negatives']:6d}",
        "",
        f"Total Pairs Evaluated: {metrics['total_pairs']}",
        f"Evaluation Timestamp:  {metrics['evaluation_timestamp']}",
        "=" * 60,
    ]
    print("\n".join(lines))
