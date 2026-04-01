from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
from nltk import word_tokenize
import numpy as np
import os
from pathlib import Path
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

# TODO all †his nonsense

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_DIR = PROJECT_ROOT / "data" / "processed_data" / "retrieval_cache"

def get_device():
	return "cuda" if torch.cuda.is_available() else "cpu"

def get_dataset_dir(corpus):
	return PROJECT_ROOT / "dataset" / corpus

def load_corpus_entries(corpus):
	dataset_corpus_path = get_dataset_dir(corpus) / "corpus.jsonl"

	if dataset_corpus_path.exists():
		entries = []
		with open(str(dataset_corpus_path), "r") as f:
			for line in f.readlines():
				entries.append(json.loads(line))
		return entries

	# Local fallback for this repository: build retrieval corpus from trial_info.json
	trial_info_path = PROJECT_ROOT / "trial_info.json"
	if not trial_info_path.exists():
		raise FileNotFoundError(
			"No corpus source found. Expected either "
			f"{dataset_corpus_path} or {trial_info_path}."
		)

	with open(str(trial_info_path), "r") as f:
		trial_info = json.load(f)

	entries = []
	for nctid, trial in trial_info.items():
		title = trial.get("brief_title", "")
		summary = trial.get("brief_summary", "")
		diseases_list = trial.get("diseases_list", [])

		entries.append(
			{
				"_id": nctid,
				"title": title,
				"text": summary,
				"metadata": {"diseases_list": diseases_list},
			}
		)

	return entries


def load_queries(corpus):
	dataset_queries_path = get_dataset_dir(corpus) / "queries.jsonl"
	if dataset_queries_path.exists():
		return dataset_queries_path

	# Local fallback for this repository
	local_queries_path = PROJECT_ROOT / "data" / "synthea_processed" / "queries.jsonl"
	if local_queries_path.exists():
		return local_queries_path

	raise FileNotFoundError(
		"No queries file found. Expected either "
		f"{dataset_queries_path} or {local_queries_path}."
	)


def load_qrels(corpus):
	dataset_dir = get_dataset_dir(corpus)
	if dataset_dir.exists():
		_, _, qrels = GenericDataLoader(data_folder=str(dataset_dir)).load(split="test")
		return qrels

	# Local setup does not include qrels
	return None


def load_id2queries(corpus):
	id2queries_path = get_dataset_dir(corpus) / "id2queries.json"
	if id2queries_path.exists():
		return json.load(open(id2queries_path))
	return None


def get_query_id(entry):
	return entry.get("_id") or entry.get("id")


def get_query_text(entry):
	return entry.get("text") or entry.get("summary") or ""


def get_query_conditions(q_type, qid, query_text, id2queries):
	# Keep the original path when id2queries exists.
	if id2queries is not None and qid in id2queries:
		if q_type in ["raw", "human_summary"]:
			query_value = id2queries[qid].get(q_type)
			return [query_value] if query_value else []
		if "turbo" in q_type and q_type in id2queries[qid]:
			return id2queries[qid][q_type].get("conditions", [])
		if "Clinician" in q_type:
			return id2queries[qid].get(q_type, [])

	# Local fallback: use direct query text/summary as a single condition.
	return [query_text] if query_text else []

def get_bm25_corpus_index(corpus):
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	corpus_path = CACHE_DIR / f"bm25_corpus_{corpus}.json"

	# if already cached then load, otherwise build
	if corpus_path.exists():
		corpus_data = json.load(open(str(corpus_path)))
		tokenized_corpus = corpus_data["tokenized_corpus"]
		corpus_nctids = corpus_data["corpus_nctids"]

	else:
		tokenized_corpus = []
		corpus_nctids = []
		entries = load_corpus_entries(corpus)

		for entry in entries:
			corpus_nctids.append(entry["_id"])

			# weighting: 3 * title, 2 * disease terms, 1 * body text
			tokens = word_tokenize(entry.get("title", "").lower()) * 3
			for disease in entry.get("metadata", {}).get("diseases_list", []):
				tokens += word_tokenize(str(disease).lower()) * 2
			tokens += word_tokenize(entry.get("text", "").lower())

			tokenized_corpus.append(tokens)

		corpus_data = {
			"tokenized_corpus": tokenized_corpus,
			"corpus_nctids": corpus_nctids,
		}

		with open(str(corpus_path), "w") as f:
			json.dump(corpus_data, f, indent=4)
	
	bm25 = BM25Okapi(tokenized_corpus)

	return bm25, corpus_nctids

			
def get_medcpt_corpus_index(corpus):
	CACHE_DIR.mkdir(parents=True, exist_ok=True)
	corpus_path = CACHE_DIR / f"{corpus}_embeds.npy"
	nctids_path = CACHE_DIR / f"{corpus}_nctids.json"
	device = get_device()

	# if already cached then load, otherwise build
	if corpus_path.exists():
		embeds = np.load(str(corpus_path))
		corpus_nctids = json.load(open(str(nctids_path))) 

	else:
		embeds = []
		corpus_nctids = []
		entries = load_corpus_entries(corpus)

		model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
		tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

		print("Encoding the corpus")
		for entry in tqdm.tqdm(entries):
			corpus_nctids.append(entry["_id"])

			title = entry.get("title", "")
			text = entry.get("text", "")

			with torch.no_grad():
				# tokenize the articles
				encoded = tokenizer(
					[[title, text]], 
					truncation=True, 
					padding=True, 
					return_tensors='pt', 
					max_length=512,
				).to(device)
				
				embed = model(**encoded).last_hidden_state[:, 0, :]

				embeds.append(embed[0].cpu().numpy())

		embeds = np.array(embeds)

		np.save(str(corpus_path), embeds)
		with open(str(nctids_path), "w") as f:
			json.dump(corpus_nctids, f, indent=4)

	index = faiss.IndexFlatIP(768)
	index.add(embeds)
	
	return index, corpus_nctids
	

if __name__ == "__main__":
	# different corpora, "trec_2021", "trec_2022", "sigir"
	corpus = sys.argv[1] if len(sys.argv) > 1 else "synthea"

	# query type
	q_type = sys.argv[2] if len(sys.argv) > 2 else "summary"

	# different k for fusion
	k = int(sys.argv[3]) if len(sys.argv) > 3 else 60

	# bm25 weight 
	bm25_wt = int(sys.argv[4]) if len(sys.argv) > 4 else 1

	# medcpt weight
	medcpt_wt = int(sys.argv[5]) if len(sys.argv) > 5 else 1

	# how many to rank
	N = 2000 
	device = get_device()

	# loading qrels/id2queries when available (dataset-style setup)
	qrels = load_qrels(corpus)
	id2queries = load_id2queries(corpus)

	# loading the indices
	bm25, bm25_nctids = get_bm25_corpus_index(corpus)
	medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)

	# loading the query encoder for MedCPT
	model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
	tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
	
	# then conduct the searches, saving top 1k
	output_path = PROJECT_ROOT / "results" / f"qid2nctids_results_{q_type}_{corpus}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}.json"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	
	qid2nctids = {}
	recalls = []
	queries_path = load_queries(corpus)

	with open(queries_path, "r") as f:
		for line in tqdm.tqdm(f.readlines()):
			entry = json.loads(line)
			query = get_query_text(entry)
			qid = get_query_id(entry)
			if qid is None:
				continue

			if qrels is not None and qid not in qrels:
				continue

			truth_sum = sum(qrels[qid].values()) if qrels is not None else None
			
			# get the keyword list
			conditions = get_query_conditions(q_type, qid, query, id2queries)

			if len(conditions) == 0:
				nctid2score = {}
			else:
				# a list of nctid lists for the bm25 retriever
				bm25_condition_top_nctids = []

				for condition in conditions:
					tokens = word_tokenize(condition.lower())
					top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
					bm25_condition_top_nctids.append(top_nctids)
				
				# doing MedCPT retrieval
				with torch.no_grad():
					encoded = tokenizer(
						conditions, 
						truncation=True, 
						padding=True, 
						return_tensors='pt', 
						max_length=256,
					).to(device)

					# encode the queries (use the [CLS] last hidden states as the representations)
					embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

					# search the Faiss index
					scores, inds = medcpt.search(embeds, k=N)				

				medcpt_condition_top_nctids = []
				for ind_list in inds:
					top_nctids = [medcpt_nctids[ind] for ind in ind_list]
					medcpt_condition_top_nctids.append(top_nctids)

				nctid2score = {}

				for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_condition_top_nctids, medcpt_condition_top_nctids)):

					if bm25_wt > 0:
						for rank, nctid in enumerate(bm25_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))
					
					if medcpt_wt > 0:
						for rank, nctid in enumerate(medcpt_top_nctids):
							if nctid not in nctid2score:
								nctid2score[nctid] = 0
							
							nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

			nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
			top_nctids = [nctid for nctid, _ in nctid2score[:N]]
			qid2nctids[qid] = top_nctids

			if qrels is not None and truth_sum:
				actual_sum = sum([qrels[qid].get(nctid, 0) for nctid in top_nctids])
				recalls.append(actual_sum / truth_sum)
	
	with open(str(output_path), "w") as f:
		json.dump(qid2nctids, f, indent=4)

	if len(recalls) > 0:
		print(f"Mean recall: {np.mean(recalls):.4f}")
	else:
		print(f"Saved retrieval results to {output_path}")
