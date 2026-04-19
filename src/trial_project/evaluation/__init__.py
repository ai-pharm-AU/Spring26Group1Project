"""Evaluation module for comparing LLM decisions with manual labels."""

from trial_project.evaluation.metrics import (
    MetricsResult,
    compare_decisions,
    compare_decisions_with_mismatches,
    print_metrics_report,
    print_mismatch_report,
    save_metrics,
)

__all__ = [
    "MetricsResult",
    "compare_decisions",
    "compare_decisions_with_mismatches",
    "save_metrics",
    "print_metrics_report",
    "print_mismatch_report",
]
