import json

# Load the provided temporal checklist JSON file
file_path = 'temporal_checklist.json'
with open(file_path, 'r') as file:
    temporal_checklist = json.load(file)

# Transform each record to match the SQuAD format
squad_formatted_data = []
for record in temporal_checklist:
    # Generate a unique ID for each example - here using a simple enumeration
    unique_id = f"temporal_{temporal_checklist.index(record)}"
    
    # Create a SQuAD-formatted record
    squad_record = {
        "id": unique_id,
        "title": "Temporal Reasoning Example",
        "context": record["passage"],
        "question": record["question"],
        "answers": {
            "text": [record["answer"]],
            "answer_start": [record["passage"].find(record["answer"])]  # Find the start index of the answer in the passage
        },
        "metadata": None  # Assuming no metadata is provided
    }
    squad_formatted_data.append(squad_record)

# Display the first few transformed records for verification
with open("squad_temp_checklist.jsonl", "w+") as f:
    json.dump(squad_formatted_data, f, indent=4)

