# Project Guidelines

## Code Style

- Keep changes minimal and localized to the relevant pipeline stage.
- Preserve existing module boundaries under `src/trial_project`.
- Prefer explicit return types for new public functions and keep docstrings short.
- Avoid broad refactors unless explicitly requested.

## Architecture

- Main package: `src/trial_project`.
- Data loading and preprocessing live in `src/trial_project/data`.
- Retrieval logic lives in `src/trial_project/retrieval`.
- Eligibility matching logic lives in `src/trial_project/matching`.
- Final ranking logic lives in `src/trial_project/ranking`.
- Path configuration is centralized in `src/trial_project/context.py`; reuse `project_root`, `data_dir`, and `results_dir` instead of hardcoding paths.

## Build and Test

- Environment management uses Poetry.
- Install deps: `poetry install`.
- Optional shell activation (zsh): `eval $(poetry env activate)`.
- Prefer running commands through Poetry to ensure the correct interpreter and deps:
  - `poetry run python -m trial_project.summarize_synthea synthea gpt-5-mini`
- When running module commands, prefer `src` as the working directory.
- This repository currently does not document a canonical test command; if adding tests, use and document a Poetry-based command.

## Conventions

- Follow existing data-path assumptions:
  - Retrieval input defaults are under `data/synthea_processed` and `data/trial_info.json`.
  - Retrieval cache data is under `data/processed_data/retrieval_cache`.
- For patient/trial loading changes, preserve JSON-serializable outputs used by LLM calls.
- If a task is about setup or usage, link to `README.md` instead of duplicating instructions.
