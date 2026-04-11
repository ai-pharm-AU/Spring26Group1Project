"""Evaluation module for comparing LLM decisions with manual labels."""

from trial_project.evaluation.metrics import (
    MetricsResult,
    compare_decisions,
    print_metrics_report,
    save_metrics,
)

__all__ = [
    "MetricsResult",
    "compare_decisions",
    "save_metrics",
    "print_metrics_report",
]
