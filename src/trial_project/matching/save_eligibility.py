"""
Save patient trial eligibility results
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import pandas as pd
from filelock import FileLock
from trial_project.context import results_dir

@dataclass
class EligibilityDecision:
    patient_id: str
    trial_id: str
    eligible: bool
    exclusion_rule_hit: bool # maybe make string or w/e
    llm_checked: bool
    decision_source: str   # "rule_based" | "llm" | "hybrid"
    reasoning: str | None = None
    confidence: float | None = None
    model_name: str | None = None
    evaluated_at: datetime | None = None

eligibility_file = results_dir / "eligibility_decisions.parquet"

def _normalize_model_name(model_name: str | None) -> str:
    if model_name is None or pd.isna(model_name):
        return ""
    return str(model_name)


def save_eligibility_decision(
    decision: EligibilityDecision,
    conflict_policy: str = "skip",
) -> bool:
    """Persist one row with model-aware conflict handling; return True when row is written."""
    if conflict_policy not in {"skip", "overwrite"}:
        raise ValueError("conflict_policy must be one of: skip, overwrite")

    row = {
        "patient_id": decision.patient_id,
        "trial_id": decision.trial_id,
        "eligible": decision.eligible,
        "exclusion_rule_hit": decision.exclusion_rule_hit,
        "llm_checked": decision.llm_checked,
        "decision_source": decision.decision_source,
        "reasoning": decision.reasoning,
        "confidence": decision.confidence,
        "model_name": decision.model_name,
        "evaluated_at": decision.evaluated_at or datetime.utcnow(),
    }

    expected_columns = list(row.keys())
    
    # Use file locking to prevent concurrent writes from corrupting the parquet file
    lock_file = eligibility_file.parent / (eligibility_file.stem + ".lock")
    with FileLock(str(lock_file)):
        if eligibility_file.exists():
            df = pd.read_parquet(eligibility_file)
        else:
            df = pd.DataFrame(columns=expected_columns)

        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        model_key = _normalize_model_name(decision.model_name)
        existing_model_key = df["model_name"].apply(_normalize_model_name)
        mask = (
            (df["patient_id"] == decision.patient_id)
            & (df["trial_id"] == decision.trial_id)
            & (existing_model_key == model_key)
        )

        if conflict_policy == "skip" and mask.any():
            return False

        df = df.loc[~mask]

        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(eligibility_file, index=False)
        return True

def load_eligibility_decision(patient_id: str, trial_id: str) -> EligibilityDecision | None:
    """Return one decision if present."""

def load_patient_eligibility(patient_id: str) -> pd.DataFrame:
    """All trial decisions for one patient."""

def load_trial_eligibility(trial_id: str) -> pd.DataFrame:
    """All patient decisions for one trial."""