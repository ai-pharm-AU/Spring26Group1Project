__author__ = "qiao"

"""
generate the search keywords for each patient
"""

import json
import re
from trial_project.api import generate_client
import sys
from trial_project.context import project_root

client = generate_client()


def get_keyword_generation_messages(note):
	system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'
	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages


def parse_model_json_response(text):
	cleaned = text.strip()

	if cleaned.startswith("```"):
		cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
		cleaned = re.sub(r"\s*```$", "", cleaned)
		cleaned = cleaned.strip()

	try:
		return json.loads(cleaned)
	except json.JSONDecodeError:
		start = cleaned.find("{")
		end = cleaned.rfind("}")
		if start != -1 and end != -1 and end > start:
			return json.loads(cleaned[start : end + 1])
		raise


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, sigir, or synthea
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2]


	queries_path = project_root / "data" / "synthea_processed" / "queries.jsonl"
	results_path = project_root / "results"
	output_file = results_path / f"retrieval_keywords_{model}_{corpus}.json"

	# load output file if it exists
	if output_file.exists():
		with open(str(output_file), "r") as f:
			outputs = json.load(f)
	else:
		outputs = {}


	with open(str(queries_path), "r") as f:
		for line in f.readlines():
			entry = json.loads(line)
			entry_id = str(entry["id"])

			if entry_id in outputs:
				print(f"Skipping {entry_id} since it already exists in outputs")
				continue

			note = entry["summary"]
			messages = get_keyword_generation_messages(note)

			response = client.chat.completions.create(
				model=model,
				messages=messages,
				#temperature=0,
			)

			output = response.choices[0].message.content or ""

			try:
				outputs[entry_id] = parse_model_json_response(output)
			except json.JSONDecodeError as e:
				print(f"Failed to parse JSON for {entry_id}: {e}")
				continue


			with open(str(output_file), "w") as f:
				json.dump(outputs, f, indent=4)