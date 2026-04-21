"""Rank matched trials for one patient."""

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd

from trial_project.data.patients.load_patient import get_patient_llm_json
from trial_project.data.patients.load_patient import load_all_patients
from trial_project.data.trials.eligibility_verification import (
    load_trial_eligibility_verification,
)
from trial_project.matching.save_eligibility import (
    load_patient_eligibility,
    load_saved_criterion_matches,
)

from trial_project.ranking.llm import evaluate_trial_ranking_llm
from trial_project.ranking.storage import (
    TrialRankingRecord,
    load_patient_trial_rankings,
    save_trial_ranking,
)

logger = logging.getLogger(__name__)


def _normalize_text_value(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = str(value).strip()
    return text if text else None


def _parse_json_list(raw_value: Any) -> list[Any]:
    if raw_value is None:
        return []

    if isinstance(raw_value, list):
        return raw_value

    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        try:
            parsed_value = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed_value, list):
            return parsed_value
        return []

    return []


def _build_matching_results(row: Any, criterion_matches: list[Any]) -> dict[str, Any]:
    return {
        "overall_decision": _normalize_text_value(row.get("overall_decision")),
        "overall_confidence_score": row.get("overall_confidence_score"),
        "overall_rationale": _normalize_text_value(row.get("overall_rationale")),
        "hard_stops": [str(item) for item in _parse_json_list(row.get("hard_stops")) if item is not None],
        "manual_review_flags": [
            str(item) for item in _parse_json_list(row.get("manual_review_flags")) if item is not None
        ],
        "matching_notes": [
            str(item) for item in _parse_json_list(row.get("matching_notes")) if item is not None
        ],
        "criterion_matches": [criterion.model_dump() for criterion in criterion_matches],
    }


def _sort_rankings(rankings: list[TrialRankingRecord]) -> list[TrialRankingRecord]:
    return sorted(
        rankings,
        key=lambda ranking: (-float(ranking.overall_score), str(ranking.trial_id)),
    )


def rank_trials(
    patient_id: str,
    data_generation_model: str = "gpt-5-mini",
    criteria_matching_model: str = "gpt-5-mini",
    overall_matching_model: str = "gpt-5-mini",
    conflict_policy: str = "skip",
) -> list[TrialRankingRecord]:
    """Rank all matched trials for one patient and cache the ranking rows."""
    patient_rows = load_patient_eligibility(
        patient_id=patient_id,
        data_generation_model=data_generation_model,
        criteria_matching_model=criteria_matching_model,
        overall_matching_model=overall_matching_model,
    )
    if patient_rows.empty:
        logger.info("No eligibility rows found for patient %s and requested models", patient_id)
        return []

    cached_rankings = {
        ranking.trial_id: ranking
        for ranking in load_patient_trial_rankings(
            patient_id=patient_id,
            data_generation_model=data_generation_model,
            criteria_matching_model=criteria_matching_model,
            overall_matching_model=overall_matching_model,
            ranking_model_name=criteria_matching_model,
        )
    }

    patient_profile_json = get_patient_llm_json(patient_id)
    ranked_trials: list[TrialRankingRecord] = []

    for _, row in patient_rows.iterrows():
        trial_id = str(row.get("trial_id", "")).strip()
        if not trial_id:
            continue

        cached_ranking = cached_rankings.get(trial_id)
        if cached_ranking is not None:
            ranked_trials.append(cached_ranking)
            continue

        trial_profile_json = load_trial_eligibility_verification(
            trial_id=trial_id,
            model_name=data_generation_model,
        )
        if trial_profile_json is None:
            logger.warning(
                "Skipping ranking for patient %s, trial %s because trial profile is missing",
                patient_id,
                trial_id,
            )
            continue

        criterion_matches = load_saved_criterion_matches(
            patient_id=patient_id,
            trial_id=trial_id,
            model_name=criteria_matching_model,
            trial_criteria_model=data_generation_model,
        )
        if not criterion_matches:
            logger.warning(
                "Skipping ranking for patient %s, trial %s because criterion matches are missing",
                patient_id,
                trial_id,
            )
            continue

        matching_results = _build_matching_results(row, criterion_matches)
        ranking_result = evaluate_trial_ranking_llm(
            trial_profile=trial_profile_json,
            patient_profile=patient_profile_json,
            matching_results=matching_results,
            model_name=criteria_matching_model,
        )

        ranking_record = TrialRankingRecord(
            patient_id=patient_id,
            trial_id=trial_id,
            overall_decision=_normalize_text_value(row.get("overall_decision")),
            ranking_model_name=criteria_matching_model,
            data_generation_model=data_generation_model,
            criteria_matching_model=criteria_matching_model,
            overall_matching_model=overall_matching_model,
            condition_relevance_score=ranking_result.condition_relevance_score,
            potential_benefit_score=ranking_result.potential_benefit_score,
            safety_score=ranking_result.safety_score,
            evidence_strength_score=ranking_result.evidence_strength_score,
            feasibility_score=ranking_result.feasibility_score,
            overall_score=ranking_result.overall_score,
        )
        save_trial_ranking(ranking_record, conflict_policy=conflict_policy)
        ranked_trials.append(ranking_record)

    return _sort_rankings(ranked_trials)


def _extract_patient_ids(patients_df: pd.DataFrame) -> list[str]:
    if patients_df.empty:
        return []

    patient_ids: list[str] = []
    for _, row in patients_df.iterrows():
        patient_id = row.get("Id") or row.get("id")
        if patient_id is None or pd.isna(patient_id):
            continue

        patient_id_text = str(patient_id).strip()
        if patient_id_text:
            patient_ids.append(patient_id_text)

    return patient_ids


def rank_all_patients(
    data_generation_model: str = "gpt-5-mini",
    criteria_matching_model: str = "gpt-5-mini",
    overall_matching_model: str = "gpt-5-mini",
    conflict_policy: str = "skip",
) -> dict[str, list[TrialRankingRecord]]:
    """Rank matched trials for every patient that exists in the patient store."""
    patients_df = load_all_patients()
    patient_ids = _extract_patient_ids(patients_df)

    ranked_by_patient: dict[str, list[TrialRankingRecord]] = {}
    for patient_id in patient_ids:
        ranked_by_patient[patient_id] = rank_trials(
            patient_id=patient_id,
            data_generation_model=data_generation_model,
            criteria_matching_model=criteria_matching_model,
            overall_matching_model=overall_matching_model,
            conflict_policy=conflict_policy,
        )

    return ranked_by_patient