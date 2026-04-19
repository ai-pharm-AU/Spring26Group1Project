"""Compute evaluation metrics comparing LLM decisions with manual labels."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd

from trial_project.context import results_dir
from trial_project.labeling.storage import manual_labels_file
from trial_project.matching.save_eligibility import eligibility_file

logger = logging.getLogger(__name__)

EVALUATION_CLASSES = ("eligible", "ineligible", "indeterminate")


class ClassMetrics(TypedDict):
    """Per-class metrics for the three-class evaluation."""

    support: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


class MetricsResult(TypedDict):
    """Metrics result dictionary."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    exact_matches: int
    mismatched_pairs: int
    label_indeterminate_pairs: int
    decision_indeterminate_pairs: int
    total_pairs: int
    class_metrics: dict[str, ClassMetrics]
    evaluation_timestamp: str


def _normalize_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_label_value(value: object) -> str:
    return _normalize_key(value).lower()


def _normalize_text(value: object) -> str:
    return _normalize_key(value)


def _require_columns(df: pd.DataFrame, required: set[str], context: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def _ensure_unique_pairs(df: pd.DataFrame, context: str) -> None:
    duplicate_mask = df.duplicated(subset=["patient_id", "trial_id"], keep=False)
    if duplicate_mask.any():
        duplicate_pairs = (
            df.loc[duplicate_mask, ["patient_id", "trial_id"]]
            .drop_duplicates()
            .to_dict(orient="records")
        )
        raise ValueError(f"Duplicate {context} found for pairs: {duplicate_pairs}")


def _read_manual_labels() -> pd.DataFrame:
    if not manual_labels_file.exists():
        return pd.DataFrame()

    labels_df = pd.read_parquet(manual_labels_file)
    if labels_df.empty:
        return labels_df

    if "label" not in labels_df.columns and "eligibility" in labels_df.columns:
        labels_df["label"] = labels_df["eligibility"]

    for column in ["patient_id", "trial_id", "label", "notes"]:
        if column not in labels_df.columns:
            labels_df[column] = ""

    labels_df["patient_id"] = labels_df["patient_id"].apply(_normalize_key)
    labels_df["trial_id"] = labels_df["trial_id"].apply(_normalize_key)
    labels_df["notes"] = labels_df["notes"].apply(_normalize_text)
    labels_df["label"] = labels_df["label"].apply(_normalize_label_value)

    labels_df = labels_df[
        (labels_df["patient_id"] != "")
        & (labels_df["trial_id"] != "")
        & (labels_df["label"] != "")
    ].copy()

    invalid_labels = sorted(set(labels_df["label"]) - {"eligible", "ineligible", "skip"})
    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}. Allowed labels: ['eligible', 'ineligible', 'skip']"
        )

    _ensure_unique_pairs(labels_df, "manual labels")
    return labels_df.reset_index(drop=True)


def _read_decisions(model_name: str, criteria_model: str | None = None) -> pd.DataFrame:
    if not eligibility_file.exists():
        raise ValueError(f"Eligibility decisions file not found: {eligibility_file}")

    decisions_df = pd.read_parquet(eligibility_file)
    if decisions_df.empty:
        return decisions_df

    _require_columns(
        decisions_df,
        {"patient_id", "trial_id", "model_name"},
        "Eligibility decisions",
    )

    decisions_df["patient_id"] = decisions_df["patient_id"].apply(_normalize_key)
    decisions_df["trial_id"] = decisions_df["trial_id"].apply(_normalize_key)
    decisions_df["model_name"] = decisions_df["model_name"].apply(_normalize_key)

    # Filter by overall model name
    decisions_df = decisions_df[decisions_df["model_name"] == _normalize_key(model_name)].copy()
    if decisions_df.empty:
        raise ValueError(f"No eligibility decisions found for model_name='{model_name}'")

    # Filter by criteria model if specified
    if criteria_model is not None:
        if "criteria_model" in decisions_df.columns:
            decisions_df["criteria_model"] = decisions_df["criteria_model"].apply(_normalize_key)
            decisions_df = decisions_df[decisions_df["criteria_model"] == _normalize_key(criteria_model)].copy()
            if decisions_df.empty:
                raise ValueError(
                    f"No eligibility decisions found for model_name='{model_name}' and criteria_model='{criteria_model}'"
                )
        else:
            logger.warning(
                "criteria_model filter requested but 'criteria_model' column not found in decisions. "
                "Proceeding with model_name filter only."
            )

    if "overall_decision" in decisions_df.columns:
        decisions_df["overall_decision"] = decisions_df["overall_decision"].apply(_normalize_label_value)
    elif "eligible" in decisions_df.columns:
        decisions_df["overall_decision"] = decisions_df["eligible"].map(
            {True: "eligible", False: "ineligible"}
        ).fillna("indeterminate")
    else:
        raise ValueError(
            "Eligibility decisions are missing both 'overall_decision' and 'eligible' columns"
        )

    for column in ["overall_rationale", "reasoning", "notes"]:
        if column in decisions_df.columns:
            decisions_df[column] = decisions_df[column].apply(_normalize_text)

    invalid_decisions = sorted(set(decisions_df["overall_decision"]) - set(EVALUATION_CLASSES))
    if invalid_decisions:
        raise ValueError(
            f"Invalid eligibility decisions found: {invalid_decisions}. Allowed decisions: {list(EVALUATION_CLASSES)}"
        )

    filter_desc = f"model_name='{model_name}'"
    if criteria_model is not None:
        filter_desc += f" and criteria_model='{criteria_model}'"
    _ensure_unique_pairs(decisions_df, f"eligibility decisions for {filter_desc}")
    return decisions_df.reset_index(drop=True)


def _build_evaluation_frame(labels_df: pd.DataFrame, decisions_df: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labels_df, {"patient_id", "trial_id", "label"}, "Manual labels")
    _require_columns(
        decisions_df,
        {"patient_id", "trial_id", "overall_decision"},
        "Eligibility decisions",
    )

    normalized_labels = labels_df.copy()
    normalized_labels["label"] = normalized_labels["label"].apply(_normalize_label_value)
    normalized_labels["label"] = normalized_labels["label"].replace({"skip": "indeterminate"})

    normalized_labels = normalized_labels[
        normalized_labels["label"].isin(EVALUATION_CLASSES)
    ].copy()

    normalized_decisions = decisions_df.copy()
    normalized_decisions["overall_decision"] = normalized_decisions["overall_decision"].apply(_normalize_label_value)

    merged = normalized_labels.merge(
        normalized_decisions,
        on=["patient_id", "trial_id"],
        how="inner",
        suffixes=("_label", "_decision"),
    )

    if merged.empty:
        raise ValueError("No matching pairs between labels and decisions for evaluation.")

    merged["label"] = merged["label"].apply(_normalize_label_value)
    merged["overall_decision"] = merged["overall_decision"].apply(_normalize_label_value)

    invalid_labels = sorted(set(merged["label"]) - set(EVALUATION_CLASSES))
    if invalid_labels:
        raise ValueError(
            f"Invalid labels found after merge: {invalid_labels}. Allowed labels: {list(EVALUATION_CLASSES)}"
        )

    invalid_decisions = sorted(set(merged["overall_decision"]) - set(EVALUATION_CLASSES))
    if invalid_decisions:
        raise ValueError(
            f"Invalid decisions found after merge: {invalid_decisions}. Allowed decisions: {list(EVALUATION_CLASSES)}"
        )

    return merged.reset_index(drop=True)


def _class_metrics(y_true: pd.Series, y_pred: pd.Series, class_name: str) -> ClassMetrics:
    true_positive = int(((y_true == class_name) & (y_pred == class_name)).sum())
    false_positive = int(((y_true != class_name) & (y_pred == class_name)).sum())
    false_negative = int(((y_true == class_name) & (y_pred != class_name)).sum())
    support = int((y_true == class_name).sum())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "support": support,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _build_mismatch_frame(merged: pd.DataFrame) -> pd.DataFrame:
    mismatch_mask = merged["label"] != merged["overall_decision"]
    mismatches = merged.loc[mismatch_mask].copy()

    if mismatches.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "trial_id",
                "label",
                "overall_decision",
                "confidence",
                "overall_rationale",
                "notes",
            ]
        )

    if "overall_confidence_score" in mismatches.columns:
        confidence = mismatches["overall_confidence_score"]
    elif "confidence" in mismatches.columns:
        confidence = mismatches["confidence"]
    else:
        confidence = pd.Series([None] * len(mismatches), index=mismatches.index)

    if "overall_rationale" in mismatches.columns:
        overall_rationale = mismatches["overall_rationale"]
    elif "reasoning" in mismatches.columns:
        overall_rationale = mismatches["reasoning"]
    else:
        overall_rationale = pd.Series([""] * len(mismatches), index=mismatches.index)

    if "notes" in mismatches.columns:
        notes = mismatches["notes"]
    else:
        notes = pd.Series([""] * len(mismatches), index=mismatches.index)

    return pd.DataFrame(
        {
            "patient_id": mismatches["patient_id"],
            "trial_id": mismatches["trial_id"],
            "label": mismatches["label"],
            "overall_decision": mismatches["overall_decision"],
            "confidence": confidence,
            "overall_rationale": overall_rationale,
            "notes": notes,
        }
    ).reset_index(drop=True)


def evaluate_decisions(
    labels_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
) -> tuple[MetricsResult, pd.DataFrame]:
    """Evaluate two in-memory dataframes and return metrics plus mismatches."""
    merged = _build_evaluation_frame(labels_df, decisions_df)

    y_true = merged["label"].astype(str)
    y_pred = merged["overall_decision"].astype(str)

    exact_matches = int((y_true == y_pred).sum())
    total_pairs = int(len(merged))
    accuracy = exact_matches / total_pairs if total_pairs > 0 else 0.0

    class_metrics = {
        class_name: _class_metrics(y_true, y_pred, class_name)
        for class_name in EVALUATION_CLASSES
    }

    precision = sum(metrics["precision"] for metrics in class_metrics.values()) / len(class_metrics)
    recall = sum(metrics["recall"] for metrics in class_metrics.values()) / len(class_metrics)
    f1 = sum(metrics["f1"] for metrics in class_metrics.values()) / len(class_metrics)

    result: MetricsResult = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "exact_matches": exact_matches,
        "mismatched_pairs": total_pairs - exact_matches,
        "label_indeterminate_pairs": int((merged["label"] == "indeterminate").sum()),
        "decision_indeterminate_pairs": int((merged["overall_decision"] == "indeterminate").sum()),
        "total_pairs": total_pairs,
        "class_metrics": class_metrics,
        "evaluation_timestamp": datetime.utcnow().isoformat(),
    }

    mismatches = _build_mismatch_frame(merged)
    logger.info("Metrics computed: %s", result)
    return result, mismatches


def compare_decisions_with_mismatches(
    model_name: str = "gpt-5-mini", criteria_model: str | None = None
) -> tuple[MetricsResult, pd.DataFrame]:
    """Compare LLM eligibility decisions with manual labels and return mismatches.
    
    Args:
        model_name: Overall model name used for eligibility decisions.
        criteria_model: Criteria model name for filtering. If None, no criteria model filtering applied.
    """
    labels_df = _read_manual_labels()
    if labels_df.empty:
        raise ValueError("No manual labels found. Please label some pairs first.")

    logger.info("Loaded %s total labels", len(labels_df))

    decisions_df = _read_decisions(model_name=model_name, criteria_model=criteria_model)
    filter_desc = f"model '{model_name}'"
    if criteria_model is not None:
        filter_desc += f" with criteria model '{criteria_model}'"
    logger.info("Loaded %s decisions for %s", len(decisions_df), filter_desc)

    return evaluate_decisions(labels_df, decisions_df)


def compare_decisions(model_name: str = "gpt-5-mini", criteria_model: str | None = None) -> MetricsResult:
    """Compare LLM eligibility decisions with manual labels.
    
    Args:
        model_name: Overall model name used for eligibility decisions.
        criteria_model: Criteria model name for filtering. If None, no criteria model filtering applied.
    """
    metrics, _ = compare_decisions_with_mismatches(model_name=model_name, criteria_model=criteria_model)
    return metrics


def save_metrics(metrics: MetricsResult, output_path: str | Path | None = None) -> Path:
    """Save metrics to JSON file."""
    if output_path is None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_path = results_dir / f"evaluation_metrics_{timestamp}.json"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    logger.info("Metrics saved to %s", output_path)
    return output_path


def print_metrics_report(metrics: MetricsResult) -> None:
    """Print formatted metrics report to console."""
    lines = [
        "=" * 72,
        "EVALUATION METRICS REPORT",
        "=" * 72,
        "",
        "Three-Class Metrics (macro average):",
        f"  Accuracy:  {metrics['accuracy']:.4f}",
        f"  Precision: {metrics['precision']:.4f}",
        f"  Recall:    {metrics['recall']:.4f}",
        f"  F1 Score:  {metrics['f1']:.4f}",
        "",
        "Per-Class Metrics:",
    ]

    for class_name in EVALUATION_CLASSES:
        class_metrics = metrics["class_metrics"][class_name]
        lines.extend(
            [
                f"  {class_name}:",
                f"    Support:        {class_metrics['support']:6d}",
                f"    True Positives:  {class_metrics['true_positives']:6d}",
                f"    False Positives: {class_metrics['false_positives']:6d}",
                f"    False Negatives: {class_metrics['false_negatives']:6d}",
                f"    Precision:      {class_metrics['precision']:.4f}",
                f"    Recall:         {class_metrics['recall']:.4f}",
                f"    F1 Score:       {class_metrics['f1']:.4f}",
            ]
        )

    lines.extend(
        [
            "",
            "Counts:",
            f"  Exact Matches:              {metrics['exact_matches']:6d}",
            f"  Mismatched Pairs:           {metrics['mismatched_pairs']:6d}",
            f"  Label Indeterminate Pairs:  {metrics['label_indeterminate_pairs']:6d}",
            f"  Decision Indeterminate Pairs:{metrics['decision_indeterminate_pairs']:6d}",
            f"  Total Pairs Evaluated:      {metrics['total_pairs']:6d}",
            f"  Evaluation Timestamp: {metrics['evaluation_timestamp']}",
            "=" * 72,
        ]
    )

    print("\n".join(lines))


def print_mismatch_report(mismatches: pd.DataFrame) -> None:
    """Print mismatched label/decision rows to console."""
    lines = [
        "",
        "Mismatched Items:",
        "-" * 72,
    ]

    if mismatches.empty:
        lines.append("No mismatched rows found.")
        print("\n".join(lines))
        return

    for index, row in mismatches.reset_index(drop=True).iterrows():
        confidence = row.get("confidence", "")
        overall_rationale = row.get("overall_rationale", "")
        notes = row.get("notes", "")
        lines.append(
            f"{index + 1}. patient_id={row.get('patient_id', '')}, trial_id={row.get('trial_id', '')}, "
            f"human_label={row.get('label', '')}, model_label={row.get('overall_decision', '')}, "
            f"confidence={confidence}, overall_rationale(model)={overall_rationale}, notes(human)={notes}"
        )

    print("\n".join(lines))