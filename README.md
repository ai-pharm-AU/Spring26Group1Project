# Trial Matching Project

## Installation

```bash
  poetry install
  wget DATA (TODO)
```

## Usage

TODO Add usage

Activate the virtual environment (on zsh run the below Otherwise just copy and run the output of `poetry env activate`)

```bash
eval $(poetry env activate)
```

from the src directory can run w/e commands using python module syntax. Like I said, it's best if you are in the src directory. Use poetry to run commands (might not matter if virtual env is activated but just to be safe)

Example:

```bash
poetry run python -m trial_project.summarize_synthea sythea gpt-5-mini
```

### Manual Labeling Tool

The labeling workflow has two steps:

1. Export matched patient-trial pairs to CSV for manual labeling.
2. Import the labeled CSV into parquet for evaluation.

Run from the `src` directory.

Export all matched pairs to CSV (includes `patient_json` and `trial_json`):

```bash
poetry run python -m trial_project.labeling export --output-csv ../results/manual_labeling_export.csv
```

Subset export by patient IDs:

```bash
poetry run python -m trial_project.labeling export --patient-ids "patient-1,patient-2"
```

Subset export by explicit pairs file (`.csv` or `.parquet` with `patient_id`, `trial_id`):

```bash
poetry run python -m trial_project.labeling export --pairs-file ../data/processed_data/pairs_subset.csv
```

Disable resume mode to include already labeled pairs in a new export:

```bash
poetry run python -m trial_project.labeling export --no-resume
```

After labeling the CSV, import it to parquet (`results/manual_labels.parquet` by default):

```bash
poetry run python -m trial_project.labeling import --input-csv ../results/manual_labeling_export.csv
```

CSV import requirements:

- required columns: `patient_id`, `trial_id`, `label`
- optional columns: `patient_json`, `trial_json`, `notes`
- label values: `eligible`, `ineligible`, `skip`
- duplicate patient/trial rows: last row wins by default (`--duplicate-policy last`)

## Synthea Usage

```bash
java -jar synthea-with-dependencies.jar -p 20
```

need to have set to csv output; move the output to w/e folder I said in the code

The retrieval flow now reads patient keywords from `data/processed_data/patient_keywords.parquet` and trial data from the existing loaders, so there is no separate `queries.jsonl` step.

## Synthea Config

only alive patients
