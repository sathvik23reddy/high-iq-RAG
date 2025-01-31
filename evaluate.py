from ragas import SingleTurnSample
from ragas.metrics import BleuScore
import rag
import json
from tqdm import tqdm


input_text_file = "test_rag.txt"
output_json_file = "test_output.json"

def parse_text_to_json(text_file):
    with open(text_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    raw_entries = raw_text.split("\n---\n")
    
    data = []
    for entry in raw_entries:
        fields = {}
        for line in entry.strip().split("\n"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Convert lists stored as strings to actual lists
            if key == "retrieved_contexts":
                value = json.loads(value)  # Convert string to list

            fields[key] = value
        
        data.append(fields)

    return data

def write_results(data):
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def run_tests(data):
    metric = BleuScore()

    for entry in tqdm(data):
        # Prop that can be retrieved
        # entry["user_input"], 
        # entry["retrieved_contexts"], 
        # entry["response"], 
        # entry["reference"]

        user_input, reference = entry["user_input"], entry["reference"]

        response = rag.init(user_input)

        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            reference=reference,
        )

        result = metric.single_turn_score(sample)

        entry["result"] = result

    return data

write_results(run_tests(parse_text_to_json(input_text_file)))


