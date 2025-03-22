import json
from pathlib import Path

file_path = Path("./data/inputs/all_processed_facts.txt")
output_path = Path("./data/outputs/facts_and_productdescriptions.json")  # Output file


if not file_path.exists():
    raise FileNotFoundError(f"File not found: {file_path}")

with file_path.open("r", encoding="utf-8") as f:
    facts_dict = json.loads(f.read())

# Categories to extract
input_category_selection = ['facts', 'product_description']

# Aggregate results from all product pitches, only want the categories for input_category_selection
selected_inputs = {}

for product_key, product_data in facts_dict.items():
    # Ensure the expected keys exist in each product pitch
    extracted_data = {k: v for k, v in product_data.items() if k in input_category_selection}
    
    if extracted_data:  # Only add if there is relevant data
        selected_inputs[product_key] = extracted_data

# # Print sample extracted data
# print(json.dumps(selected_inputs, indent=2))

print(selected_inputs['facts_shark_tank_transcript_0_GarmaGuard.txt'].keys())
print(len(selected_inputs.keys()))

# Save the filtered data as JSON
with output_path.open("w", encoding="utf-8") as f:
    json.dump(selected_inputs, f, indent=2)

print(f"Filtered data saved to {output_path}")