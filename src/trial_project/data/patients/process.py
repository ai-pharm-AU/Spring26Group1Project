import pandas as pd
from pathlib import Path
from trial_project.context import data_dir

# ---- CONFIG ----
csv_dir = data_dir / "synthea_generated_patients"
out_dir = data_dir / "processed_data"
out_dir.mkdir(parents=True, exist_ok=True)
AGE_REFERENCE_DATE = pd.Timestamp("2026-04-13")

# ---- WHICH COLUMNS TO KEEP ----
# make sure age years is in w/e df cols thing bc not in keep fields
KEEP_FIELDS = {
    "patients": ["Id", "BIRTHDATE", "GENDER", "RACE", "ETHNICITY"],
    # encounters prob should be second
    "encounters": ["Id", "PATIENT", "START", "STOP", "ENCOUNTERCLASS", "DESCRIPTION", "REASONDESCRIPTION"],
    "conditions": ["PATIENT", "ENCOUNTER", "START", "STOP", "DESCRIPTION"],
    "medications": ["PATIENT", "ENCOUNTER", "START", "STOP", "DESCRIPTION"],
    "observations": ["PATIENT", "ENCOUNTER", "DATE", "DESCRIPTION", "VALUE", "UNITS"],
    "procedures": ["PATIENT", "ENCOUNTER", "DATE", "DESCRIPTION"],
    "allergies": ["PATIENT", "ENCOUNTER", "START", "STOP", "DESCRIPTION"],
    "immunizations": ["PATIENT", "ENCOUNTER", "DATE", "DESCRIPTION"],
    "careplans": ["PATIENT", "ENCOUNTER", "START", "STOP", "DESCRIPTION"]
}

RECENT_ONLY_ENCOUNTER_CLASSES = {"wellness", "ambulatory"}
EXCLUDED_ENCOUNTER_CLASSES: set[str] = set()
EXCLUDED_OBSERVATION_DESCRIPTIONS = {
    "Have you spent more than 2 nights in a row in a jail prison detention center or juvenile correctional facility in past 1 year [PRAPARE]",
    "Have you or any family members you live with been unable to get any of the following when it was really needed in past 1 year [PRAPARE]",
    "Has season or migrant farm work been your or your family's main source of income at any point in past 2 years [PRAPARE]",
    "Address",
    "Has lack of transportation kept you from medical appointments meetings work or from getting things needed for daily living",
    "Housing status",
    "Within the last year have you been afraid of your partner or ex-partner",
    "Primary insurance",
    "Do you feel physically and emotionally safe where you currently live [PRAPARE]",
    "Employment status - current",
    "How many people are living or staying at this address [#]",
    "Are you a refugee",
    "What was your best estimate of the total income of all family members from all sources before taxes in last year [PhenX]",
    "Are you worried about losing your housing [PRAPARE]",
    "QALY"

}
EXCLUDED_CONDITION_DESCRIPTIONS = {
    "Housing unsatisfactory (finding)",
    "Unemployed (finding)",
    "Part-time employment (finding)",
    "Full-time employment (finding)",
    "Not in labor force (finding)",
    "Has a criminal record (finding)",
    "Refugee (person)",
    # "Received higher education (finding)",
    "Medication review due (situation)",
    
}


def _normalize_text_values(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def _build_normalized_exclusion_set(values: set[str]) -> set[str]:
    return {" ".join(value.strip().lower().split()) for value in values}


NORMALIZED_EXCLUDED_OBSERVATION_DESCRIPTIONS = _build_normalized_exclusion_set(EXCLUDED_OBSERVATION_DESCRIPTIONS)
NORMALIZED_EXCLUDED_CONDITION_DESCRIPTIONS = _build_normalized_exclusion_set(EXCLUDED_CONDITION_DESCRIPTIONS)


def compute_age_years(birthdates: pd.Series, reference_date: pd.Timestamp) -> pd.Series:
    birthdates = pd.to_datetime(birthdates, errors="coerce")
    age_years = pd.Series(pd.NA, index=birthdates.index, dtype="Int64")

    valid_birthdates = birthdates.notna()
    valid_years = birthdates.loc[valid_birthdates].dt.year
    valid_months = birthdates.loc[valid_birthdates].dt.month
    valid_days = birthdates.loc[valid_birthdates].dt.day

    age_years.loc[valid_birthdates] = reference_date.year - valid_years

    has_had_birthday = (
        (reference_date.month > valid_months)
        | ((reference_date.month == valid_months) & (reference_date.day >= valid_days))
    )
    age_years.loc[valid_birthdates] = age_years.loc[valid_birthdates] - (~has_had_birthday).astype("Int64")

    return age_years

def get_tables_dict():
    tables_dict = {}

    for name in KEEP_FIELDS:
        if name == "patients":
            continue

        file_path = out_dir / f"{name}.parquet"

        if file_path.exists():
            table_df = pd.read_parquet(file_path)
            tables_dict[name] = table_df

    return tables_dict


def _filter_by_encounter_ids(df: pd.DataFrame, encounter_ids: set[str] | None) -> pd.DataFrame:
    if encounter_ids is None or "ENCOUNTER" not in df.columns:
        return df

    encounter_values = df["ENCOUNTER"].astype("string")
    return df[encounter_values.isna() | encounter_values.isin(encounter_ids)]


def _keep_most_recent_encounter_by_class(
    encounters_df: pd.DataFrame,
    recent_only_classes: set[str],
) -> pd.DataFrame:
    if encounters_df.empty or "ENCOUNTERCLASS" not in encounters_df.columns:
        return encounters_df

    normalized_class = encounters_df["ENCOUNTERCLASS"].astype("string").str.strip().str.lower()
    recent_class_mask = normalized_class.isin(recent_only_classes)

    if not recent_class_mask.any():
        return encounters_df

    keep_all_other_classes = encounters_df.loc[~recent_class_mask].copy()
    recent_candidates = encounters_df.loc[recent_class_mask].copy()

    recent_candidates["_CLASS_NORM"] = normalized_class.loc[recent_class_mask]
    recent_candidates["_START_SORT"] = pd.to_datetime(recent_candidates.get("START"), errors="coerce")
    recent_candidates["_STOP_SORT"] = pd.to_datetime(recent_candidates.get("STOP"), errors="coerce")

    sort_columns = ["PATIENT", "_CLASS_NORM", "_START_SORT", "_STOP_SORT"]
    sort_ascending = [True, True, False, False]
    if "Id" in recent_candidates.columns:
        recent_candidates["_ID_SORT"] = recent_candidates["Id"].astype("string")
        sort_columns.append("_ID_SORT")
        sort_ascending.append(False)

    recent_candidates = recent_candidates.sort_values(
        by=sort_columns,
        ascending=sort_ascending,
        na_position="last",
    )
    latest_per_class = recent_candidates.drop_duplicates(
        subset=["PATIENT", "_CLASS_NORM"],
        keep="first",
    )

    latest_per_class = latest_per_class.drop(
        columns=["_CLASS_NORM", "_START_SORT", "_STOP_SORT", "_ID_SORT"],
        errors="ignore",
    )
    filtered_encounters = pd.concat([keep_all_other_classes, latest_per_class], ignore_index=False)

    if "Id" in filtered_encounters.columns:
        filtered_encounters = filtered_encounters.drop_duplicates(subset=["Id"], keep="first")

    return filtered_encounters


def _exclude_encounter_classes(encounters_df: pd.DataFrame, excluded_classes: set[str]) -> pd.DataFrame:
    if encounters_df.empty or not excluded_classes or "ENCOUNTERCLASS" not in encounters_df.columns:
        return encounters_df

    normalized_class = encounters_df["ENCOUNTERCLASS"].astype("string").str.strip().str.lower()
    mask = normalized_class.isin(excluded_classes)
    return encounters_df.loc[~mask]

def load_synthea_tables(n_patients=None):
    # ---- LOAD PATIENTS ----
    patients = pd.read_csv(csv_dir / "patients.csv", dtype=str)
    patients = patients[KEEP_FIELDS["patients"]]
    if n_patients is not None:
        patients = patients.head(n_patients)

    patients["AGE_YEARS"] = compute_age_years(patients["BIRTHDATE"], AGE_REFERENCE_DATE)

    # Get selected patient IDs
    patient_ids = set(patients["Id"])

    # ---- LOAD AND FILTER OTHER TABLES ----
    tables = {}
    encounter_ids = None

    for name in KEEP_FIELDS:
        if name == "patients":
            continue

        df = pd.read_csv(csv_dir / f"{name}.csv", dtype=str)

        # Ensure PATIENT column exists
        if "PATIENT" not in df.columns:
            raise ValueError(f"{name} missing PATIENT column")

        # Filter to selected patients
        df = df[df["PATIENT"].isin(patient_ids)]

        if name == "encounters":
            df = _keep_most_recent_encounter_by_class(df, RECENT_ONLY_ENCOUNTER_CLASSES)
            df = _exclude_encounter_classes(df, EXCLUDED_ENCOUNTER_CLASSES)

        if name == "observations" and "DESCRIPTION" in df.columns:
            print(f"Filtering observations with descriptions in excluded list: {EXCLUDED_OBSERVATION_DESCRIPTIONS}")
            normalized_description = _normalize_text_values(df["DESCRIPTION"])
            mask = normalized_description.isin(NORMALIZED_EXCLUDED_OBSERVATION_DESCRIPTIONS)
            df = df[~mask]

        if name == "conditions" and "DESCRIPTION" in df.columns:
            normalized_description = _normalize_text_values(df["DESCRIPTION"])
            mask = normalized_description.isin(NORMALIZED_EXCLUDED_CONDITION_DESCRIPTIONS)
            df = df[~mask]

        if name != "encounters":
            df = _filter_by_encounter_ids(df, encounter_ids)

        # Keep only desired columns (if they exist)
        cols = [c for c in KEEP_FIELDS[name] if c in df.columns]
        df = df[cols]

        tables[name] = df

        if name == "encounters" and "Id" in df.columns:
            encounter_ids = set(df["Id"].dropna().astype(str))

    return patients, tables

def save_tables(patients, tables):
    # Save patients
    patients.to_parquet(out_dir / "patients.parquet", index=False)

    # Save each table
    for name, df in tables.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)


if __name__ == "__main__":
    patients_df, tables_dict = load_synthea_tables(n_patients=20)
    save_tables(patients_df, tables_dict)
