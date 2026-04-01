# ---- CONFIG ----
from pathlib import Path


project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / "data" # note: don't confuse with the folder with the code related to data
results_dir = project_root / "results"