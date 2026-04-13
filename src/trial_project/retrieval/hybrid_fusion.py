import hashlib
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import tqdm
from nltk import word_tokenize
from rank_bm25 import BM25Okapi
from transformers import AutoModel, AutoTokenizer

from trial_project.context import data_dir, results_dir
from trial_project.data.patients.load_patient import load_all_patients
from trial_project.data.trials.load import load_all_trials
from trial_project.retrieval.keywords.load import load_all_patient_keywords, load_patient_keywords

retrieval_cache_dir = data_dir / "processed_data" / "retrieval_cache"
eligible_trials_file = results_dir / "eligible_trials.parquet"

ID_COLUMNS = ["Id", "id", "trial_id", "nct_id", "nctid", "NCTId", "nct_number"]
TITLE_COLUMNS = ["title", "brief_title", "official_title", "name"]
TEXT_COLUMNS = [
    "text",
    "description",
    "brief_summary",
    "detailed_description",
    "summary",
    "eligibility_criteria",
    "inclusion_criteria",
    "exclusion_criteria",
    "conditions",
]
DISEASE_COLUMNS = ["conditions", "condition", "disease", "diseases", "diseases_list"]


def get_device():
    return "cuda" if torch.cuda.is_available() else ( "mps" if torch.backends.mps.is_available() else "cpu" )


def _load_medcpt_tokenizer(model_name):
    # Fast tokenizers can crash in some environments for pair encoding; prefer slow tokenizer.
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=False)
    except TypeError:
        return AutoTokenizer.from_pretrained(model_name)


def _tokenize(text):
    if not text:
        return []
    try:
        return word_tokenize(str(text).lower())
    except LookupError:
        return re.findall(r"\w+", str(text).lower())


def _normalize_list_like(value):
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, pd.Series):
        return [str(v).strip() for v in value.tolist() if str(v).strip()]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, set):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        parts = re.split(r"[;,|]\s*", value)
        return [part.strip() for part in parts if part.strip()]
    if pd.notna(value):
        return [str(value).strip()]
    return []


def _first_non_empty(row, candidates):
    for col in candidates:
        if col in row.index:
            value = row[col]
            if isinstance(value, str) and value.strip():
                return value.strip()
            if pd.notna(value):
                value_str = str(value).strip()
                if value_str:
                    return value_str
    return ""


def _pick_diseases(row):
    for col in DISEASE_COLUMNS:
        if col in row.index:
            items = _normalize_list_like(row[col])
            if items:
                return items
    return []


def _build_entry(row, row_idx, actual_trial_id=None):
    # Use the provided actual trial ID, fallback to extracting from row, never use synthetic IDs
    trial_id = actual_trial_id or _first_non_empty(row, ID_COLUMNS)
    if not trial_id:
        raise ValueError(f"No trial ID found in row {row_idx}: {row.to_dict()}")

    title = _first_non_empty(row, TITLE_COLUMNS)

    text_parts = []
    for col in TEXT_COLUMNS:
        if col in row.index:
            value = row[col]
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
            elif pd.notna(value):
                value_str = str(value).strip()
                if value_str:
                    text_parts.append(value_str)

    if not text_parts:
        for col in row.index:
            value = row[col]
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())

    text = "\n".join(text_parts)

    return {
        "_id": str(trial_id),
        "title": title,
        "text": text,
        "metadata": {
            "diseases_list": _pick_diseases(row),
        },
    }


def _build_trials_signature(trials_df):
    sig_payload = {
        "shape": list(trials_df.shape),
        "columns": list(trials_df.columns),
    }
    sig_text = json.dumps(sig_payload, sort_keys=True)
    return hashlib.md5(sig_text.encode("utf-8")).hexdigest()[:12]


def load_trial_entries(trials_df=None):
    if trials_df is None:
        trials_df = load_all_trials()

    if trials_df.empty:
        raise ValueError("Trials dataframe is empty; cannot build retrieval index.")

    print(f"[hybrid_fusion] Loaded trials dataframe with shape={trials_df.shape}")

    entries = []
    for row_idx, (_, row) in enumerate(trials_df.iterrows()):
        # Use the "Id" column which contains the actual trial ID from trial_info.json
        actual_trial_id = row.get("Id") if hasattr(row, 'get') else row["Id"]
        entries.append(_build_entry(row, row_idx, actual_trial_id=actual_trial_id))
    
    signature = _build_trials_signature(trials_df)
    cache_key = signature
    print(f"[hybrid_fusion] Built {len(entries)} trial entries with cache_key={cache_key}")
    return entries, cache_key


def _normalize_conditions(value):
    if isinstance(value, dict):
        conditions = value.get("conditions")
        if conditions:
            return _normalize_list_like(conditions)
        summary = value.get("summary")
        if summary:
            return _normalize_list_like(summary)
    return _normalize_list_like(value)


def load_patient_conditions(patient_id, keywords_df=None):
    keyword_value = load_patient_keywords(patient_id, keywords_df)
    return _normalize_conditions(keyword_value)


def load_patient_trial_ground_truth():
    if not eligible_trials_file.exists():
        return None

    try:
        labels_df = pd.read_parquet(eligible_trials_file)
    except Exception:
        return None

    if "patient_id" not in labels_df.columns or "trial_ids" not in labels_df.columns:
        return None

    ground_truth = {}
    for _, row in labels_df.iterrows():
        patient_id = row["patient_id"]
        trial_ids = row["trial_ids"]  # Get actual trial IDs from this patient's row
        if patient_id is None or trial_ids is None or (isinstance(trial_ids, (list, tuple)) and len(trial_ids) == 0):
            continue
        ground_truth[str(patient_id)] = set(trial_ids)

    return ground_truth


def get_bm25_trial_index(entries, cache_key):
    retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = retrieval_cache_dir / f"bm25_trials_{cache_key}.json"

    if corpus_path.exists():
        print(f"[hybrid_fusion] Loading BM25 cache from {corpus_path}")
        with open(corpus_path, "r") as f:
            corpus_data = json.load(f)
        tokenized_corpus = corpus_data["tokenized_corpus"]
        corpus_ids = corpus_data["corpus_ids"]
    else:
        print(f"[hybrid_fusion] Building BM25 cache at {corpus_path}")
        tokenized_corpus = []
        corpus_ids = []

        for entry in entries:
            corpus_ids.append(entry["_id"])

            tokens = _tokenize(entry.get("title", "")) * 3
            for disease in entry.get("metadata", {}).get("diseases_list", []):
                tokens += _tokenize(disease) * 2
            tokens += _tokenize(entry.get("text", ""))
            tokenized_corpus.append(tokens)

        corpus_data = {
            "tokenized_corpus": tokenized_corpus,
            "corpus_ids": corpus_ids,
        }
        with open(corpus_path, "w") as f:
            json.dump(corpus_data, f, indent=2)
        print(f"[hybrid_fusion] Saved BM25 cache with {len(corpus_ids)} trial ids")

    bm25 = BM25Okapi(tokenized_corpus)
    print(f"[hybrid_fusion] BM25 index ready for {len(corpus_ids)} trials")
    return bm25, corpus_ids


def get_medcpt_trial_index(entries, cache_key):
    retrieval_cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = retrieval_cache_dir / f"{cache_key}_embeds.npy"
    ids_path = retrieval_cache_dir / f"{cache_key}_ids.json"
    device = get_device()

    if corpus_path.exists() and ids_path.exists():
        print(f"[hybrid_fusion] Loading MedCPT cache from {corpus_path} and {ids_path}")
        embeds = np.load(corpus_path)
        with open(ids_path, "r") as f:
            corpus_ids = json.load(f)
    else:
        print(f"[hybrid_fusion] Building MedCPT cache on device={device}")
        embeds = []
        corpus_ids = []

        model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        model.eval()
        tokenizer = _load_medcpt_tokenizer("ncbi/MedCPT-Article-Encoder")

        print("Encoding corpus for MedCPT...")
        for entry in tqdm.tqdm(entries):
            print(f"[hybrid_fusion] MedCPT encoding trial_id={entry['_id']} title={entry.get('title', '')!r}")
            corpus_ids.append(entry["_id"])
            title = entry.get("title", "")
            text = entry.get("text", "")

            with torch.no_grad():
                encoded = tokenizer(
                    text=[title or ""],
                    text_pair=[text or ""],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                    max_length=512,
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                embed = model(**encoded).last_hidden_state[:, 0, :]
                embeds.append(embed[0].cpu().numpy())
        embeds = np.array(embeds, dtype=np.float32)
        np.save(corpus_path, embeds)
        with open(ids_path, "w") as f:
            json.dump(corpus_ids, f, indent=2)
        print(f"[hybrid_fusion] Saved MedCPT cache for {len(corpus_ids)} trials")

    embeds = np.array(embeds, dtype=np.float32)
    index = faiss.IndexFlatIP(int(embeds.shape[1]))
    index.add(embeds)
    print(f"[hybrid_fusion] MedCPT index ready with embedding shape={embeds.shape}")
    return index, corpus_ids


def _search_bm25(bm25, corpus_ids, conditions, top_k):
    results = []
    for condition in conditions:
        tokens = _tokenize(condition)
        print(f"[hybrid_fusion] BM25 search condition={condition!r} token_count={len(tokens)} top_k={top_k}")
        top_ids = bm25.get_top_n(tokens, corpus_ids, n=top_k)
        results.append(top_ids)
    return results


def _search_medcpt(index, corpus_ids, conditions, top_k, device):
    print(f"[hybrid_fusion] MedCPT search for {len(conditions)} conditions on device={device} top_k={top_k}")
    model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
    model.eval()
    tokenizer = _load_medcpt_tokenizer("ncbi/MedCPT-Query-Encoder")

    with torch.no_grad():
        encoded = tokenizer(
            conditions,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=256,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
        _, inds = index.search(embeds, k=top_k)

    results = []
    for ind_list in inds:
        results.append([corpus_ids[ind] for ind in ind_list])
    print("[hybrid_fusion] MedCPT search completed")
    return results


def rank_trials_for_conditions(
    conditions,
    k=60,
    bm25_wt=1,
    medcpt_wt=1,
    n_results=200,
    trials_df=None,
):
    if not conditions:
        print("[hybrid_fusion] No conditions provided; returning empty ranking")
        return []

    device = get_device()
    print(f"[hybrid_fusion] Ranking {len(conditions)} conditions bm25_wt={bm25_wt} medcpt_wt={medcpt_wt} n_results={n_results}")
    entries, cache_key = load_trial_entries(trials_df=trials_df)
    bm25, bm25_ids = get_bm25_trial_index(entries, cache_key)

    medcpt = None
    medcpt_ids = []
    if medcpt_wt > 0:
        medcpt, medcpt_ids = get_medcpt_trial_index(entries, cache_key)

    print(f"[hybrid_fusion] Candidate pools ready bm25_ids={len(bm25_ids)} medcpt_ids={len(medcpt_ids)}")

    bm25_top_lists = _search_bm25(bm25, bm25_ids, conditions, n_results) if bm25_wt > 0 else [[] for _ in conditions]
    medcpt_top_lists = (
        _search_medcpt(medcpt, medcpt_ids, conditions, n_results, device)
        if medcpt_wt > 0 and medcpt is not None
        else [[] for _ in conditions]
    )

    scores = {}
    for condition_idx, (bm25_top_ids, medcpt_top_ids) in enumerate(zip(bm25_top_lists, medcpt_top_lists)):
        if bm25_wt > 0:
            for rank, trial_id in enumerate(bm25_top_ids):
                scores[trial_id] = scores.get(trial_id, 0.0) + (
                    bm25_wt * (1.0 / (rank + k)) * (1.0 / (condition_idx + 1))
                )

        if medcpt_wt > 0:
            for rank, trial_id in enumerate(medcpt_top_ids):
                scores[trial_id] = scores.get(trial_id, 0.0) + (
                    medcpt_wt * (1.0 / (rank + k)) * (1.0 / (condition_idx + 1))
                )

    sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
    print(f"[hybrid_fusion] Ranked {len(sorted_scores)} trials from {len(scores)} scored candidates")
    return [trial_id for trial_id, _ in sorted_scores[:n_results]]


def run_hybrid_fusion(
    k=60,
    bm25_wt=1,
    medcpt_wt=1,
    n_results=2000,
    trials_df=None,
    patients_df=None,
    patient_keywords_df=None,
    save_output=True,
    output_path=None,
):
    device = get_device()
    print(f"[hybrid_fusion] Starting hybrid fusion device={device} k={k} bm25_wt={bm25_wt} medcpt_wt={medcpt_wt} n_results={n_results}")
    ground_truth = load_patient_trial_ground_truth()
    if ground_truth is None:
        print("[hybrid_fusion] No eligible_trials.parquet found; recall will be skipped")
    else:
        print(f"[hybrid_fusion] Loaded ground truth for {len(ground_truth)} patients")

    if patients_df is None:
        patients_df = load_all_patients()
    if patient_keywords_df is None:
        patient_keywords_df = load_all_patient_keywords()

    print(f"[hybrid_fusion] Loaded {len(patients_df)} patients and {len(patient_keywords_df)} keyword rows")

    entries, cache_key = load_trial_entries(trials_df=trials_df)
    bm25, bm25_ids = get_bm25_trial_index(entries, cache_key)

    medcpt = None
    medcpt_ids = []
    if medcpt_wt > 0:
        medcpt, medcpt_ids = get_medcpt_trial_index(entries, cache_key)

    if output_path is None:
        output_path = eligible_trials_file
    else:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".parquet":
            output_path = output_path.with_suffix(".parquet")
    print(f"[hybrid_fusion] Output path: {output_path}")

    patient2nctids = {}
    recalls = []
    for _, patient_row in tqdm.tqdm(patients_df.iterrows(), total=len(patients_df)):
        patient_id = patient_row.get("Id")
        if patient_id is None or pd.isna(patient_id):
            continue

        patient_id = str(patient_id)
        conditions = load_patient_conditions(patient_id, patient_keywords_df)
        print(f"[hybrid_fusion] patient_id={patient_id} conditions={conditions}")

        nctid2score = {}
        if conditions:
            if bm25_wt > 0:
                bm25_top_lists = _search_bm25(bm25, bm25_ids, conditions, n_results)
            else:
                bm25_top_lists = [[] for _ in conditions]

            if medcpt_wt > 0 and medcpt is not None:
                medcpt_top_lists = _search_medcpt(medcpt, medcpt_ids, conditions, n_results, device)
            else:
                medcpt_top_lists = [[] for _ in conditions]

            for condition_idx, (bm25_top_ids, medcpt_top_ids) in enumerate(zip(bm25_top_lists, medcpt_top_lists)):
                if bm25_wt > 0:
                    for rank, trial_id in enumerate(bm25_top_ids):
                        nctid2score[trial_id] = nctid2score.get(trial_id, 0.0) + (
                            bm25_wt * (1.0 / (rank + k)) * (1.0 / (condition_idx + 1))
                        )

                if medcpt_wt > 0:
                    for rank, trial_id in enumerate(medcpt_top_ids):
                        nctid2score[trial_id] = nctid2score.get(trial_id, 0.0) + (
                            medcpt_wt * (1.0 / (rank + k)) * (1.0 / (condition_idx + 1))
                        )

        sorted_scores = sorted(nctid2score.items(), key=lambda x: -x[1])
        top_nctids = [nctid for nctid, _ in sorted_scores[:n_results]]
        patient2nctids[patient_id] = top_nctids
        print(f"[hybrid_fusion] patient_id={patient_id} top_results={top_nctids[:5]}")

        if ground_truth is not None:
            truth_trial_ids = ground_truth.get(patient_id)
            if truth_trial_ids:
                actual_sum = len(set(top_nctids) & truth_trial_ids)
                recall = actual_sum / len(truth_trial_ids)
                recalls.append(recall)
                print(f"[hybrid_fusion] patient_id={patient_id} recall={recall:.4f}")

    if save_output:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_df = pd.DataFrame(
            [
                {"patient_id": patient_id, "trial_ids": trial_ids}
                for patient_id, trial_ids in patient2nctids.items()
            ],
            columns=["patient_id", "trial_ids"],
        )
        output_df.to_parquet(output_path, index=False)
        print(f"[hybrid_fusion] Saved {len(patient2nctids)} patient results to {output_path}")

    return {
        "results": patient2nctids,
        "output_path": str(output_path),
        "mean_recall": float(np.mean(recalls)) if recalls else None,
    }


if __name__ == "__main__":
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    bm25_wt = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    medcpt_wt = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    n_results = int(sys.argv[4]) if len(sys.argv) > 4 else 10

    output = run_hybrid_fusion(
        k=k,
        bm25_wt=bm25_wt,
        medcpt_wt=medcpt_wt,
        n_results=n_results,
        save_output=True,
    )

    if output["mean_recall"] is not None:
        print(f"Mean recall: {output['mean_recall']:.4f}")
    print(f"Saved retrieval results to {output['output_path']}")
