from litellm_client import LiteLLMClient
import pandas as pd
import json


class JudgeLLM:
    """Judge LLM class to evaluate Shark-Pitch interactions and provide decisions."""
    
    def __init__(self, model="o1-mini", num_judges=3, llm_models=None):
        """Initialize the Judge LLM with a specified model."""
        self.client = LiteLLMClient()
        self.model = model
        self.num_judges = num_judges  # Number of independent LLM evaluations
        self.llm_models = llm_models if llm_models else [model] * num_judges  # List of LLM models used


    def judge_pitch(self, processed_data, output_csv="judge_llm_results.csv"):
        """Evaluate the final offer comparison using a multi-LLM prompting strategy based on DebateLLM."""
        system_prompt = (
            "You are a panel of expert venture capitalists analyzing the terms of an investment deal based on structured business data. "
            "You will evaluate key elements including the product details, the Shark LLM's proposed final offer, and the historical actual offer. "
            "The data provided includes: \n"
            "- Scenario Name: The unique business pitch scenario. "
            "- Product Details: Information on the business product details. "
            "- Product Facts: Financial Information on the business and its offering. "
            "- Shark LLM Offer: The proposed investment terms made by the Shark LLM. "
            "- Actual Offer: The real historical investment terms given to this business. "
            "Your objective is to compare the Shark LLM's offer against the actual offer by considering: "
            "1. Business valuation and potential ROI. "
            "2. Fairness in equity and investment structure. "
            "3. Entrepreneur’s leverage in negotiation. "
            "4. Market viability of the deal. "
            "5. Whether the Shark LLM offer is better, worse, or the same as the actual historical deal. "
            "Follow a multi-agent discussion format: \n"
            "1. Analyst 1 provides an argument for or against the Shark LLM offer. "
            "2. Analyst 2 critically assesses and refines the argument. "
            "3. Analyst 3 synthesizes the discussion and finalizes a verdict. "
            "Return the final assessment in JSON format: "
            '{"reasoning": "Panel discussion and key takeaways", "rating of final_offer": score (1-10)}'
            "A score of 5 indicates the Shark LLM offer is equal to the actual offer."
        )
        
        results = []
        individual_scores = []
        for model in self.llm_models:
            messages = [{"role": "user", "content": processed_data}]
            response = self.client.generate_response(model, messages, system_prompt)
            try:
                response_json = json.loads(response)
                if "rating of final_offer" in response_json:
                    individual_scores.append(response_json["rating of final_offer"])
                    results.append({
                        "Scenario": processed_data.get("Scenario", "Unknown"),
                        "LLM Model": model,
                        "Individual Score": response_json["rating of final_offer"],
                        "Final Consensus Score": None  # Placeholder, updated after aggregation
                    })
            except json.JSONDecodeError:
                continue
        
        consensus_score = round(sum(individual_scores) / len(individual_scores), 2) if individual_scores else "Not Available"
        
        # Update final consensus score for each record
        for result in results:
            result["Final Consensus Score"] = consensus_score
        
        # Convert to DataFrame and save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        
        return results_df

    
    # def process_shark_pitch_data(self, csv_file, json_file):
    #     """Process the Shark-Pitch interactions and extract relevant details."""
    #     # Load Shark LLM output from the CSV file
    #     shark_df = pd.read_csv(csv_file)

    #     # Load the actual final offers from the JSON file
    #     with open(json_file, 'r') as f:
    #         actual_facts = json.load(f)

    #     def get_product_details(scenario_name):
    #         """Retrieve product details from processed facts."""
    #         return actual_facts.get(f"facts_{scenario_name}", {}).get("product_description", {})
        
    #     def get_actual_final_offer(scenario_name):
    #         """Retrieve the actual final offer from processed facts."""
    #         return actual_facts.get(f"facts_{scenario_name}", {}).get("final_offer", None)

    #     # Extracting relevant data
    #     processed_data = []
    #     for _, row in shark_df.iterrows():
    #         scenario_name = row.get("scenario_name")  # Matching with JSON keys
    #         product_details = get_product_details(scenario_name)
    #         shark_offer = row.get("response") if row.get("layer") == "shark" else None  # Extract final offer from Shark
    #         actual_offer = get_actual_final_offer(scenario_name)
            
    #         if scenario_name and product_details:
    #             processed_data.append({
    #                 "Scenario": scenario_name,
    #                 "Product Details": product_details,
    #                 "Shark LLM Offer": shark_offer if shark_offer else "Not Found",
    #                 "Actual Offer": actual_offer if actual_offer else "Not Found"
    #             })
        
    #     return pd.DataFrame(processed_data)


