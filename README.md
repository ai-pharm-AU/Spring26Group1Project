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

The labeling tool shows full patient and trial JSON for matched patient-trial pairs and asks for a manual label.
Labels are saved to `results/manual_labels.parquet` and the tool is resumable by default.

Run from the `src` directory:

```bash
poetry run python -m trial_project.labeling --all
```

Subset by patient IDs:

```bash
poetry run python -m trial_project.labeling --patient-ids "patient-1,patient-2"
```

Subset by explicit pairs file (`.csv` or `.parquet` with `patient_id`, `trial_id`):

```bash
poetry run python -m trial_project.labeling --pairs-file ../data/processed_data/pairs_subset.csv
```

Disable resume mode if you want to relabel already-labeled pairs:

```bash
poetry run python -m trial_project.labeling --all --no-resume
```

## Synthea Usage

```bash
java -jar synthea-with-dependencies.jar -p 20
```

need to have set to csv output; move the output to w/e folder I said in the code

The retrieval flow now reads patient keywords from `data/processed_data/patient_keywords.parquet` and trial data from the existing loaders, so there is no separate `queries.jsonl` step.
