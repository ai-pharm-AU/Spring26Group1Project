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

## Synthea Usage

```bash
java -jar synthea-with-dependencies.jar -p 20
```

need to have set to csv output; move the output to w/e folder I said in the code

The retrieval flow now reads patient keywords from `data/processed_data/patient_keywords.parquet` and trial data from the existing loaders, so there is no separate `queries.jsonl` step.
