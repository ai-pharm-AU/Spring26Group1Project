"""Storage and CSV roundtrip utilities for manual labels."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from trial_project.context import results_dir
from trial_project.data.patients.load_patient import get_patient_llm_json
from trial_project.data.trials.load import load_trial_json_llm
from trial_project.labeling.pairs import load_matched_pairs

manual_labels_file = results_dir / "manual_labels.parquet"

_ALLOWED_LABELS = {"eligible", "ineligible", "skip"}
_EXPORT_COLUMNS = [
    "patient_id",
    "trial_id",
    "patient_json",
    "trial_json",
    "label",
    "notes",
]
_STORAGE_COLUMNS = _EXPORT_COLUMNS + ["imported_at", "source_csv"]


def _empty_labels_df() -> pd.DataFrame:
    return pd.DataFrame(columns=_STORAGE_COLUMNS)


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_label(value: object) -> str:
    return _normalize_text(value).lower()


def _read_labels_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return _empty_labels_df()

    df = pd.read_parquet(path)
    if "label" not in df.columns and "eligibility" in df.columns:
        df["label"] = df["eligibility"]

    for col in _STORAGE_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    return df[_STORAGE_COLUMNS].copy()


def _normalize_labels_frame(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in ["patient_id", "trial_id", "patient_json", "trial_json", "notes", "source_csv"]:
        normalized[col] = normalized[col].apply(_normalize_text)

    normalized["label"] = normalized["label"].apply(_normalize_label)
    normalized["imported_at"] = normalized["imported_at"].apply(_normalize_text)

    normalized = normalized[
        (normalized["patient_id"] != "")
        & (normalized["trial_id"] != "")
        & (normalized["label"] != "")
    ].copy()

    invalid_labels = sorted(set(normalized["label"]) - _ALLOWED_LABELS)
    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}. Allowed labels: {sorted(_ALLOWED_LABELS)}"
        )

    return normalized


def load_labels(path: str | Path | None = None) -> pd.DataFrame:
    """Load manual labels parquet for evaluation and analysis."""
    labels_path = Path(path) if path else manual_labels_file
    labels_df = _read_labels_parquet(labels_path)

    labels_df = _normalize_labels_frame(labels_df)
    labels_df = labels_df.drop_duplicates(
        subset=["patient_id", "trial_id"], keep="last"
    )
    return labels_df.reset_index(drop=True)


def _parse_patient_ids(patient_ids: str | None) -> set[str]:
    if not patient_ids:
        return set()
    return {item.strip() for item in patient_ids.split(",") if item.strip()}


def export_labeling_csv(
    output_csv: str | Path,
    matched_pairs_source: str | Path | None = None,
    patient_ids: str | None = None,
    pairs_file: str | Path | None = None,
    resume: bool = True,
) -> tuple[Path, int]:
    """Export patient-trial pairs with JSON payloads to CSV for manual labeling."""
    pairs_df = load_matched_pairs(matched_pairs_source)

    selected_patient_ids = _parse_patient_ids(patient_ids)
    if selected_patient_ids:
        pairs_df = pairs_df[pairs_df["patient_id"].isin(selected_patient_ids)]

    if pairs_file:
        subset_df = load_matched_pairs(pairs_file)
        pairs_df = pairs_df.merge(subset_df, on=["patient_id", "trial_id"], how="inner")

    if resume:
        existing_labels = load_labels()
        if not existing_labels.empty:
            pairs_df = pairs_df.merge(
                existing_labels[["patient_id", "trial_id"]],
                on=["patient_id", "trial_id"],
                how="left",
                indicator=True,
            )
            pairs_df = pairs_df[pairs_df["_merge"] == "left_only"]
            pairs_df = pairs_df.drop(columns=["_merge"])

    if pairs_df.empty:
        export_df = pd.DataFrame(columns=_EXPORT_COLUMNS)
    else:
        pairs_df = pairs_df.drop_duplicates(subset=["patient_id", "trial_id"], keep="first")
        patient_cache: dict[str, str] = {}
        trial_cache: dict[str, str] = {}

        def patient_json(patient_id: str) -> str:
            if patient_id not in patient_cache:
                patient_cache[patient_id] = get_patient_llm_json(patient_id)
            return patient_cache[patient_id]

        def trial_json(trial_id: str) -> str:
            if trial_id not in trial_cache:
                trial_cache[trial_id] = load_trial_json_llm(trial_id)
            return trial_cache[trial_id]

        export_df = pairs_df.copy()
        export_df["patient_json"] = export_df["patient_id"].map(patient_json)
        export_df["trial_json"] = export_df["trial_id"].map(trial_json)
        export_df["label"] = ""
        export_df["notes"] = ""
        export_df = export_df[_EXPORT_COLUMNS]

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(output_path, index=False)
    return output_path, int(len(export_df))


def import_labels_csv(
    input_csv: str | Path,
    output_path: str | Path | None = None,
    duplicate_policy: Literal["last", "first", "fail"] = "last",
) -> tuple[Path, int]:
    """Import labeled CSV rows and upsert into manual labels parquet."""
    if duplicate_policy not in {"last", "first", "fail"}:
        raise ValueError("duplicate_policy must be one of: last, first, fail")

    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    imported = pd.read_csv(input_path)
    required = {"patient_id", "trial_id", "label"}
    missing = sorted(required - set(imported.columns))
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    for optional in ["patient_json", "trial_json", "notes"]:
        if optional not in imported.columns:
            imported[optional] = ""

    imported = imported[["patient_id", "trial_id", "patient_json", "trial_json", "label", "notes"]]
    imported["imported_at"] = datetime.utcnow().isoformat()
    imported["source_csv"] = str(input_path)

    imported = _normalize_labels_frame(imported)

    if imported.empty:
        destination = Path(output_path) if output_path else manual_labels_file
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            _empty_labels_df().to_parquet(destination, index=False)
        return destination, 0

    duplicate_mask = imported.duplicated(subset=["patient_id", "trial_id"], keep=False)
    if duplicate_policy == "fail" and duplicate_mask.any():
        duplicate_pairs = imported.loc[duplicate_mask, ["patient_id", "trial_id"]]
        duplicate_pairs = duplicate_pairs.drop_duplicates().to_dict(orient="records")
        raise ValueError(f"Duplicate rows found in CSV for pairs: {duplicate_pairs}")

    keep_value = "last" if duplicate_policy == "last" else "first"
    imported = imported.drop_duplicates(subset=["patient_id", "trial_id"], keep=keep_value)

    destination = Path(output_path) if output_path else manual_labels_file
    existing = _read_labels_parquet(destination)
    combined = pd.concat([existing, imported], ignore_index=True)
    combined = _normalize_labels_frame(combined)
    combined = combined.drop_duplicates(subset=["patient_id", "trial_id"], keep="last")

    destination.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(destination, index=False)
    return destination, int(len(imported))
