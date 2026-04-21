"""Persistence helpers for patient-trial ranking outputs."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field

from trial_project.context import results_dir

ranking_file = results_dir / "patient_trial_rankings.parquet"


class TrialRankingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patient_id: str
    trial_id: str
    overall_decision: str | None = None
    ranking_model_name: str
    data_generation_model: str | None = None
    criteria_matching_model: str | None = None
    overall_matching_model: str | None = None
    condition_relevance_score: float = Field(ge=0.0, le=100.0)
    potential_benefit_score: float = Field(ge=0.0, le=100.0)
    safety_score: float = Field(ge=0.0, le=100.0)
    evidence_strength_score: float = Field(ge=0.0, le=100.0)
    feasibility_score: float = Field(ge=0.0, le=100.0)
    overall_score: float = Field(ge=0.0, le=100.0)
    evaluated_at: datetime | None = None

    def to_row(self) -> dict:
        return {
            "patient_id": self.patient_id,
            "trial_id": self.trial_id,
            "overall_decision": self.overall_decision,
            "ranking_model_name": self.ranking_model_name,
            "data_generation_model": self.data_generation_model,
            "criteria_matching_model": self.criteria_matching_model,
            "overall_matching_model": self.overall_matching_model,
            "condition_relevance_score": self.condition_relevance_score,
            "potential_benefit_score": self.potential_benefit_score,
            "safety_score": self.safety_score,
            "evidence_strength_score": self.evidence_strength_score,
            "feasibility_score": self.feasibility_score,
            "overall_score": self.overall_score,
            "evaluated_at": self.evaluated_at or datetime.utcnow(),
        }


def _normalize_model_name(model_name: str | None) -> str:
    if model_name is None or pd.isna(model_name):
        return ""
    return str(model_name).strip()


def _load_ranking_df() -> pd.DataFrame:
    if not ranking_file.exists():
        return pd.DataFrame()

    try:
        return pd.read_parquet(ranking_file)
    except (FileNotFoundError, OSError, ValueError):
        return pd.DataFrame()


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = None
    return df


def _filter_rankings_df(
    df: pd.DataFrame,
    *,
    patient_id: str | None = None,
    trial_id: str | None = None,
    data_generation_model: str | None = None,
    criteria_matching_model: str | None = None,
    overall_matching_model: str | None = None,
    ranking_model_name: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    filtered_df = df.copy()

    if patient_id is not None:
        if "patient_id" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[filtered_df["patient_id"].astype(str) == str(patient_id)]

    if trial_id is not None:
        if "trial_id" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[filtered_df["trial_id"].astype(str) == str(trial_id)]

    if data_generation_model is not None:
        if "data_generation_model" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[
            filtered_df["data_generation_model"].apply(_normalize_model_name)
            == _normalize_model_name(data_generation_model)
        ]

    if criteria_matching_model is not None:
        if "criteria_matching_model" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[
            filtered_df["criteria_matching_model"].apply(_normalize_model_name)
            == _normalize_model_name(criteria_matching_model)
        ]

    if overall_matching_model is not None:
        if "overall_matching_model" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[
            filtered_df["overall_matching_model"].apply(_normalize_model_name)
            == _normalize_model_name(overall_matching_model)
        ]

    if ranking_model_name is not None:
        if "ranking_model_name" not in filtered_df.columns:
            return pd.DataFrame(columns=filtered_df.columns)
        filtered_df = filtered_df[
            filtered_df["ranking_model_name"].apply(_normalize_model_name)
            == _normalize_model_name(ranking_model_name)
        ]

    sort_columns = [col for col in ["overall_score", "evaluated_at", "trial_id"] if col in filtered_df.columns]
    if sort_columns:
        ascending = [False] * len(sort_columns)
        ascending[-1] = True
        filtered_df = filtered_df.sort_values(by=sort_columns, ascending=ascending)

    return filtered_df.reset_index(drop=True)


def save_trial_ranking(
    ranking: TrialRankingRecord,
    conflict_policy: str = "skip",
) -> bool:
    """Persist one ranking row; return True when the row is written."""
    if conflict_policy not in {"skip", "overwrite"}:
        raise ValueError("conflict_policy must be one of: skip, overwrite")

    row = ranking.to_row()
    expected_columns = list(row.keys())

    lock_file = ranking_file.parent / (ranking_file.stem + ".lock")
    with FileLock(str(lock_file)):
        if ranking_file.exists():
            df = pd.read_parquet(ranking_file)
        else:
            df = pd.DataFrame(columns=expected_columns)

        df = _ensure_columns(df, expected_columns)

        mask = (
            (df["patient_id"].astype(str) == str(ranking.patient_id))
            & (df["trial_id"].astype(str) == str(ranking.trial_id))
            & (df["data_generation_model"].apply(_normalize_model_name) == _normalize_model_name(ranking.data_generation_model))
            & (df["criteria_matching_model"].apply(_normalize_model_name) == _normalize_model_name(ranking.criteria_matching_model))
            & (df["overall_matching_model"].apply(_normalize_model_name) == _normalize_model_name(ranking.overall_matching_model))
            & (df["ranking_model_name"].apply(_normalize_model_name) == _normalize_model_name(ranking.ranking_model_name))
        )

        if conflict_policy == "skip" and mask.any():
            return False

        df = df.loc[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(ranking_file, index=False)
        return True


def load_patient_trial_rankings(
    patient_id: str,
    data_generation_model: str | None = None,
    criteria_matching_model: str | None = None,
    overall_matching_model: str | None = None,
    ranking_model_name: str | None = None,
) -> list[TrialRankingRecord]:
    """Load cached ranking rows for one patient/model combination."""
    df = _load_ranking_df()
    if df.empty:
        return []

    filtered_df = _filter_rankings_df(
        df,
        patient_id=patient_id,
        data_generation_model=data_generation_model,
        criteria_matching_model=criteria_matching_model,
        overall_matching_model=overall_matching_model,
        ranking_model_name=ranking_model_name,
    )
    if filtered_df.empty:
        return []

    records: list[TrialRankingRecord] = []
    for _, row in filtered_df.iterrows():
        evaluated_at = row.get("evaluated_at")
        if pd.isna(evaluated_at):
            evaluated_at = None

        try:
            records.append(
                TrialRankingRecord(
                    patient_id=str(row.get("patient_id", "")),
                    trial_id=str(row.get("trial_id", "")),
                    overall_decision=row.get("overall_decision"),
                    ranking_model_name=str(row.get("ranking_model_name", "")),
                    data_generation_model=row.get("data_generation_model"),
                    criteria_matching_model=row.get("criteria_matching_model"),
                    overall_matching_model=row.get("overall_matching_model"),
                    condition_relevance_score=float(row.get("condition_relevance_score", 0.0)),
                    potential_benefit_score=float(row.get("potential_benefit_score", 0.0)),
                    safety_score=float(row.get("safety_score", 0.0)),
                    evidence_strength_score=float(row.get("evidence_strength_score", 0.0)),
                    feasibility_score=float(row.get("feasibility_score", 0.0)),
                    overall_score=float(row.get("overall_score", 0.0)),
                    evaluated_at=evaluated_at,
                )
            )
        except Exception:
            continue

    return records


def load_patient_trial_ranking(
    patient_id: str,
    trial_id: str,
    data_generation_model: str | None = None,
    criteria_matching_model: str | None = None,
    overall_matching_model: str | None = None,
    ranking_model_name: str | None = None,
) -> TrialRankingRecord | None:
    """Load one cached ranking row if present."""
    rankings = load_patient_trial_rankings(
        patient_id=patient_id,
        data_generation_model=data_generation_model,
        criteria_matching_model=criteria_matching_model,
        overall_matching_model=overall_matching_model,
        ranking_model_name=ranking_model_name,
    )
    for ranking in rankings:
        if ranking.trial_id == str(trial_id):
            return ranking
    return None