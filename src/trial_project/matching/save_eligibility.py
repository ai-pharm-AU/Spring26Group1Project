"""
Save patient trial eligibility results
"""

from datetime import datetime
import json

import pandas as pd
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from trial_project.context import results_dir
from trial_project.matching.llm import CriterionMatch, MatchedPatientEvidence

class EligibilityDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    trial_id: str
    overall_decision: Literal["eligible", "ineligible", "indeterminate"]
    overall_confidence_score: float | None = Field(default=None, ge=0.0, le=1.0)
    overall_rationale: str | None = None
    hard_stops: list[str] = Field(default_factory=list)
    manual_review_flags: list[str] = Field(default_factory=list)
    matching_notes: list[str] = Field(default_factory=list)

    # Backward-compatible fields retained for existing downstream consumers.
    eligible: bool | None = None
    exclusion_rule_hit: bool = False
    llm_checked: bool = True
    decision_source: Literal["rule_based", "llm", "hybrid"] = "llm"
    reasoning: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    model_name: str | None = None
    criteria_model: str | None = None
    evaluated_at: datetime | None = None

    def to_row(self) -> dict:
        eligible_value = self.eligible
        if eligible_value is None:
            if self.overall_decision == "eligible":
                eligible_value = True
            elif self.overall_decision == "ineligible":
                eligible_value = False

        reasoning_value = self.reasoning if self.reasoning is not None else self.overall_rationale
        confidence_value = (
            self.confidence
            if self.confidence is not None
            else self.overall_confidence_score
        )

        return {
            "patient_id": self.patient_id,
            "trial_id": self.trial_id,
            "overall_decision": self.overall_decision,
            "overall_confidence_score": self.overall_confidence_score,
            "overall_rationale": self.overall_rationale,
            "hard_stops": json.dumps(self.hard_stops, ensure_ascii=True),
            "manual_review_flags": json.dumps(self.manual_review_flags, ensure_ascii=True),
            "matching_notes": json.dumps(self.matching_notes, ensure_ascii=True),
            "eligible": eligible_value,
            "exclusion_rule_hit": self.exclusion_rule_hit,
            "llm_checked": self.llm_checked,
            "decision_source": self.decision_source,
            "reasoning": reasoning_value,
            "confidence": confidence_value,
            "model_name": self.model_name,
            "criteria_model": self.criteria_model,
            "evaluated_at": self.evaluated_at or datetime.utcnow(),
        }


class CriterionEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    trial_id: str
    criterion_id: str
    criterion_type: Literal["inclusion", "exclusion"]
    criterion_text: str
    status: Literal[
        "meets",
        "does_not_meet",
        "insufficient_evidence",
        "excluded",
        "not_excluded",
    ]
    matched_patient_evidence: str | None = None
    possible_proxies: str | None = None
    missing_but_needed: str | None = None
    reasoning: str | None = None
    confidence: float | None = None
    model_name: str | None = None
    evaluated_at: datetime | None = None

    @classmethod
    def from_criterion_match(
        cls,
        *,
        patient_id: str,
        trial_id: str,
        criterion: CriterionMatch,
        model_name: str | None,
        evaluated_at: datetime,
    ) -> "CriterionEvaluation":
        return cls(
            patient_id=patient_id,
            trial_id=trial_id,
            criterion_id=criterion.criterion_id,
            criterion_type=criterion.criterion_type,
            criterion_text=criterion.criterion_text,
            status=criterion.status,
            matched_patient_evidence=json.dumps(
                [evidence.model_dump() for evidence in criterion.matched_patient_evidence],
                ensure_ascii=True,
            ),
            possible_proxies=json.dumps(criterion.possible_proxies, ensure_ascii=True),
            missing_but_needed=json.dumps(criterion.missing_but_needed, ensure_ascii=True),
            reasoning=criterion.reasoning,
            confidence=criterion.confidence,
            model_name=model_name,
            evaluated_at=evaluated_at,
        )

eligibility_file = results_dir / "eligibility_decisions.parquet"
criterion_file = results_dir / "criterion_matches.parquet"

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

    row = decision.to_row()

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


def save_criterion_matches(
    patient_id: str,
    trial_id: str,
    criterion_matches: list[CriterionMatch],
    model_name: str | None,
    conflict_policy: str = "skip",
) -> tuple[int, int]:
    """Persist criterion-level rows; returns (written_count, skipped_count)."""
    if conflict_policy not in {"skip", "overwrite"}:
        raise ValueError("conflict_policy must be one of: skip, overwrite")

    now = datetime.utcnow()
    rows = []
    for criterion in criterion_matches:
        criterion_row = CriterionEvaluation.from_criterion_match(
            patient_id=patient_id,
            trial_id=trial_id,
            criterion=criterion,
            model_name=model_name,
            evaluated_at=now,
        )
        rows.append(criterion_row.model_dump())

    if not rows:
        return 0, 0

    expected_columns = list(rows[0].keys())
    lock_file = criterion_file.parent / (criterion_file.stem + ".lock")

    with FileLock(str(lock_file)):
        if criterion_file.exists():
            df = pd.read_parquet(criterion_file)
        else:
            df = pd.DataFrame(columns=expected_columns)

        for col in expected_columns:
            if col not in df.columns:
                df[col] = None

        written_count = 0
        skipped_count = 0
        model_key = _normalize_model_name(model_name)

        for row in rows:
            existing_model_key = df["model_name"].apply(_normalize_model_name)
            mask = (
                (df["patient_id"] == row["patient_id"])
                & (df["trial_id"] == row["trial_id"])
                & (df["criterion_id"] == row["criterion_id"])
                & (existing_model_key == model_key)
            )

            if conflict_policy == "skip" and mask.any():
                skipped_count += 1
                continue

            df = df.loc[~mask]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            written_count += 1

        df.to_parquet(criterion_file, index=False)

    return written_count, skipped_count


def _parse_json_list(raw_value: str | None) -> list:
    if raw_value is None or pd.isna(raw_value):
        return []

    parsed_value = json.loads(raw_value)
    if isinstance(parsed_value, list):
        return parsed_value

    return []


def _criterion_row_to_match(row: pd.Series) -> CriterionMatch:
    matched_patient_evidence = [
        MatchedPatientEvidence.model_validate(item)
        for item in _parse_json_list(row.get("matched_patient_evidence"))
        if isinstance(item, dict)
    ]

    possible_proxies = [
        str(item)
        for item in _parse_json_list(row.get("possible_proxies"))
        if item is not None
    ]
    missing_but_needed = [
        str(item)
        for item in _parse_json_list(row.get("missing_but_needed"))
        if item is not None
    ]

    confidence_value = row.get("confidence")
    if pd.isna(confidence_value):
        confidence_value = 0.0

    return CriterionMatch(
        criterion_id=str(row.get("criterion_id", "")),
        criterion_type=str(row.get("criterion_type", "inclusion")),
        criterion_text=str(row.get("criterion_text", "")),
        status=str(row.get("status", "insufficient_evidence")),
        matched_patient_evidence=matched_patient_evidence,
        possible_proxies=possible_proxies,
        missing_but_needed=missing_but_needed,
        reasoning=str(row.get("reasoning", "")),
        confidence=float(confidence_value),
    )


def load_saved_criterion_matches(
    patient_id: str,
    trial_id: str,
    model_name: str | None,
) -> list[CriterionMatch]:
    """Load saved criterion matches for one patient/trial/model combination."""
    if not criterion_file.exists():
        return []

    df = pd.read_parquet(criterion_file)
    if df.empty:
        return []

    for column in ["patient_id", "trial_id", "criterion_id", "model_name"]:
        if column not in df.columns:
            return []

    model_key = _normalize_model_name(model_name)
    existing_model_key = df["model_name"].apply(_normalize_model_name)
    matches_df = df[
        (df["patient_id"] == patient_id)
        & (df["trial_id"] == trial_id)
        & (existing_model_key == model_key)
    ]

    if matches_df.empty:
        return []

    matches_df = matches_df.sort_values(by=["criterion_id"])
    matches: list[CriterionMatch] = []
    for _, row in matches_df.iterrows():
        try:
            matches.append(_criterion_row_to_match(row))
        except Exception:
            return []

    return matches

def load_eligibility_decision(patient_id: str, trial_id: str) -> EligibilityDecision | None:
    """Return one decision if present."""

def load_patient_eligibility(patient_id: str) -> pd.DataFrame:
    """All trial decisions for one patient."""

def load_trial_eligibility(trial_id: str) -> pd.DataFrame:
    """All patient decisions for one trial."""