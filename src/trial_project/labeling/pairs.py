"""Load patient-trial matched pairs from retrieval outputs or subset files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from trial_project.context import results_dir

eligible_trials_file = results_dir / "eligible_trials.parquet"


def _normalize_string(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_trial_ids(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        return [_normalize_string(item) for item in value if _normalize_string(item)]

    if hasattr(value, "tolist"):
        try:
            return _normalize_trial_ids(value.tolist())
        except Exception:
            pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []

        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return _normalize_trial_ids(parsed)
            except json.JSONDecodeError:
                pass

        if "," in stripped:
            return [item.strip() for item in stripped.split(",") if item.strip()]

        return [stripped]

    return [_normalize_string(value)] if _normalize_string(value) else []


def _load_pairs_source(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Matched pairs source not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError("Matched pairs source must be a .parquet or .csv file")


def _to_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
    if {"patient_id", "trial_id"}.issubset(df.columns):
        pairs_df = df[["patient_id", "trial_id"]].copy()
    elif {"patient_id", "trial_ids"}.issubset(df.columns):
        expanded = df[["patient_id", "trial_ids"]].copy()
        expanded["trial_ids"] = expanded["trial_ids"].apply(_normalize_trial_ids)
        pairs_df = expanded.explode("trial_ids", ignore_index=True)
        pairs_df = pairs_df.rename(columns={"trial_ids": "trial_id"})
        pairs_df = pairs_df[["patient_id", "trial_id"]]
    else:
        raise ValueError(
            "Matched pairs source must contain either (patient_id, trial_id) "
            "or (patient_id, trial_ids) columns"
        )

    pairs_df["patient_id"] = pairs_df["patient_id"].apply(_normalize_string)
    pairs_df["trial_id"] = pairs_df["trial_id"].apply(_normalize_string)

    pairs_df = pairs_df[
        (pairs_df["patient_id"] != "") & (pairs_df["trial_id"] != "")
    ].copy()
    pairs_df = pairs_df.drop_duplicates(subset=["patient_id", "trial_id"], keep="first")
    return pairs_df.reset_index(drop=True)


def load_matched_pairs(
    matched_pairs_source: str | Path | None = None,
) -> pd.DataFrame:
    """Load normalized patient-trial matched pairs.

    Args:
        matched_pairs_source: Optional .parquet/.csv path. Defaults to retrieval output.

    Returns:
        DataFrame with columns: patient_id, trial_id.
    """
    source_path = Path(matched_pairs_source) if matched_pairs_source else eligible_trials_file
    source_df = _load_pairs_source(source_path)
    return _to_pair_rows(source_df)
