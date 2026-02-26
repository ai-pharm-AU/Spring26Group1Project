

""" generate the search keywords for each patient
"""

import json
from trial_project.api import generate_client
import sys
from pathlib import Path
from trial_project.context import project_root

client = generate_client()

# TODO could add age and sex, etc fields from this

def get_keyword_generation_messages(note):
	system = 'You are a helpful assistant and your task is to help summarize a patient\'s medical history. Given the patient\s information as a json object please provide a plaintext 1-2 sentence summary of their medical history and overall health.'

	prompt =  f"Here is the patient information: \n{note}\n\nSummary:"

	print("PROMPT:", prompt)

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]
	
	return messages


if __name__ == "__main__":
	# the corpus: trec_2021, trec_2022, sigir, or synthea
	corpus = sys.argv[1]

	# the model index to use
	model = sys.argv[2]

	outputs = {}

	queries_path = project_root / "data" / "synthea_processed" / "all_patients_filtered.json"

	with open(str(queries_path), "r") as f:
		note = "";
		if corpus == "synthea":
				data = json.load(f)
				for entry in data:
					note = json.dumps(entry, indent=2)
					messages = get_keyword_generation_messages(note)

					response = client.chat.completions.create(
						model=model,
						messages=messages,
						#temperature=0,
					)

					output = response.choices[0].message.content
					#output = output.strip("`").strip("json")
					print(output)
					
					outputs[entry["_id"]] = {"summary": output}
					output_path = project_root / "data" / "synthea_processed"

					with open(str(output_path / f"synthea_processed_{model}_{corpus}.json"), "w") as f:
						json.dump(outputs, f, indent=4)
		
# after being summarized, convert to .jsonl and call queries.jsonl for the keyword generation step