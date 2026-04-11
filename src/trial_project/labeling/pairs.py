"""Utilities to build and filter patient-trial labeling queues."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trial_project.context import results_dir

matched_pairs_file = results_dir / "eligible_trials.parquet"


def _normalize_trial_ids(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, pd.Series):
        return [str(v).strip() for v in value.tolist() if str(v).strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        # Handle stringified list payloads in case parquet round-tripping changed type.
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                return _normalize_trial_ids(parsed)
            except json.JSONDecodeError:
                pass
        return [stripped]
    if pd.notna(value):
        return [str(value).strip()]
    return []


def load_matched_pairs(source_path: str | Path | None = None) -> pd.DataFrame:
    source = Path(source_path) if source_path else matched_pairs_file
    if not source.exists():
        raise FileNotFoundError(f"Matched pairs file not found: {source}")

    df = pd.read_parquet(source)
    if "patient_id" not in df.columns or "trial_ids" not in df.columns:
        raise ValueError(f"Expected columns patient_id and trial_ids in: {source}")

    rows: list[dict[str, str]] = []
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"]).strip()
        if not patient_id:
            continue
        for trial_id in _normalize_trial_ids(row["trial_ids"]):
            rows.append({"patient_id": patient_id, "trial_id": trial_id})

    out = pd.DataFrame(rows, columns=["patient_id", "trial_id"])
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["patient_id", "trial_id"])
    return out.sort_values(["patient_id", "trial_id"]).reset_index(drop=True)


def parse_patient_ids(raw_patient_ids: str | None) -> set[str]:
    if not raw_patient_ids:
        return set()
    return {token.strip() for token in raw_patient_ids.split(",") if token.strip()}


def load_pairs_subset_file(pairs_file: str | Path) -> pd.DataFrame:
    path = Path(pairs_file)
    if not path.exists():
        raise FileNotFoundError(f"Pairs subset file not found: {path}")

    if path.suffix.lower() == ".parquet":
        subset_df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        subset_df = pd.read_csv(path)
    else:
        raise ValueError("Pairs subset file must be .csv or .parquet")

    required = {"patient_id", "trial_id"}
    if not required.issubset(set(subset_df.columns)):
        raise ValueError("Pairs subset file must include patient_id and trial_id columns")

    subset_df = subset_df[["patient_id", "trial_id"]].copy()
    subset_df["patient_id"] = subset_df["patient_id"].astype(str).str.strip()
    subset_df["trial_id"] = subset_df["trial_id"].astype(str).str.strip()
    subset_df = subset_df[(subset_df["patient_id"] != "") & (subset_df["trial_id"] != "")]
    return subset_df.drop_duplicates(subset=["patient_id", "trial_id"])


def filter_pairs(
    matched_pairs: pd.DataFrame,
    patient_ids: set[str] | None = None,
    subset_pairs_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    filtered = matched_pairs.copy()

    if patient_ids:
        filtered = filtered[filtered["patient_id"].isin(patient_ids)]

    dropped_unmatched = 0
    if subset_pairs_df is not None:
        candidate_keys = set(zip(filtered["patient_id"], filtered["trial_id"]))
        requested_keys = set(zip(subset_pairs_df["patient_id"], subset_pairs_df["trial_id"]))
        valid_keys = requested_keys & candidate_keys
        dropped_unmatched = len(requested_keys - valid_keys)

        filtered = pd.DataFrame(valid_keys, columns=["patient_id", "trial_id"])

    if filtered.empty:
        return filtered, dropped_unmatched

    filtered = filtered.drop_duplicates(subset=["patient_id", "trial_id"])
    filtered = filtered.sort_values(["patient_id", "trial_id"]).reset_index(drop=True)
    return filtered, dropped_unmatched