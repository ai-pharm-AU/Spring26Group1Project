"""
load trials
"""

import json
from functools import lru_cache

import pandas as pd

from trial_project.context import data_dir

trials_path = data_dir / "trial_info.json"


@lru_cache(maxsize=1)
def _load_trials_dict() -> dict:
  with trials_path.open("r", encoding="utf-8") as file:
    return json.load(file)

def load_trial_ids() -> list:
  return list(_load_trials_dict().keys())


def load_trial(trial_id: str) -> dict:
  return get_trial_json(trial_id)


def load_all_trials() -> pd.DataFrame:
  trials = _load_trials_dict()
  trials_df = pd.DataFrame.from_dict(trials, orient="index")
  trials_df.index.name = "Id"
  return trials_df.reset_index()


def load_trial_json_llm(trial_id: str) -> str:
  return json.dumps(get_trial_json(trial_id), ensure_ascii=False)


def get_trial_json(trial_id: str) -> dict:
  trials = _load_trials_dict()
  try:
    trial = trials[trial_id]
  except KeyError as exc:
    raise ValueError(f"Trial not found: {trial_id}") from exc

  return {"id": trial_id, **trial}