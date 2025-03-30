import os
import json
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

from pitch_team import generate_pitch, save_pitch_to_file
from utils import load_api_key, setup_logging
from guardrails import GLOBAL_ATTEMPT_COUNTS  # Import the global attempt counter
# makes sure data is in a consistent format
def normalize_data(data, default_name="Unknown"):
    """
    makes data consistent regardless of what type it starts as
    
    args:
        data: stuff we want to normalize (dict, list, str, etc)
        default_name: fallback name if we need to create a dict
        
    returns:
        dict: data in a nice clean dictionary
    """
    if isinstance(data, dict):
        # add name if missing
        if "name" not in data:
            data["name"] = default_name
        return data
    elif isinstance(data, list):
        return {
            "name": default_name,
            "items": data
        }
    elif isinstance(data, str):
        return {
            "name": default_name,
            "description": data
        }
    else:
        return {
            "name": default_name,
            "description": str(data) if data is not None else ""
        }

# set up argument parser
parser = argparse.ArgumentParser(description='Generate a pitch for a single SharkTank episode')
parser.add_argument('episode_key', help='The key of the episode to process')
parser.add_argument('--data_path', default='../data/all_processed_facts.json', help='Path to the facts data file')
parser.add_argument('--output_dir', default='output', help='Directory to save the output')
parser.add_argument('--model', default='gpt-4o', help='The OpenAI model to use')
parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='Logging level')

async def main():
    args = parser.parse_args()
    
    log_level = getattr(logging, args.log_level)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/pitch_generation_{timestamp}.log"
    
    setup_logging(log_level=log_level, log_file=log_file)
    
    try:
        # reset global attempt counts for a clean start
        global GLOBAL_ATTEMPT_COUNTS
        GLOBAL_ATTEMPT_COUNTS.clear()
        logging.info("Reset global attempt counter for a fresh start")
        
        api_key = load_api_key()
        
        logging.info(f"Loading data from {args.data_path}")
        with open(args.data_path, 'r', encoding='utf-8') as f:
            facts_data = json.load(f)
        
        episode_key = args.episode_key
        if episode_key not in facts_data:
            # try to find a key that contains the episode_key
            matching_keys = [key for key in facts_data.keys() if episode_key in key]
            if matching_keys:
                episode_key = matching_keys[0]
                logging.info(f"Using matching episode key: {episode_key}")
            else:
                logging.error(f"Episode {episode_key} not found in data")
                print(f"Error: Episode {episode_key} not found in data")
                return
        
        logging.info(f"Generating pitch for {episode_key}")
        
        episode_data = facts_data[episode_key]
        facts = episode_data.get("facts", "")
        product_description = episode_data.get("product_description", {"name": "Unknown Product"})
        
        logging.debug(f"Original facts type: {type(facts).__name__}")
        logging.debug(f"Original product_description type: {type(product_description).__name__}")
        
        product_description = normalize_data(product_description, default_name="Unknown Product")
        product_name = product_description.get("name", "Unknown Product")
        
        if not isinstance(facts, (str, dict)):
            logging.info(f"Converting facts from {type(facts).__name__} to dictionary format")
            facts = normalize_data(facts, default_name="Facts")
        
        logging.info(f"Product: {product_name}")
        
        # store facts and product info in env vars for fact checkers
        os.environ["PITCH_FACTS"] = json.dumps(facts) if isinstance(facts, dict) else facts
        os.environ["PITCH_PRODUCT_DESCRIPTION"] = json.dumps(product_description)
        
        logging.info("Starting pitch generation process...")
        pitch_result = await generate_pitch(facts, product_description, model=args.model)
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(args.output_dir, f"{episode_key}_pitch_{timestamp}.json")
        
        result = {
            "episode_key": episode_key,
            "original_data": {
                "facts": facts,
                "product_description": product_description
            },
            "generated_pitch": pitch_result
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logging.info(f"Pitch generation completed successfully")
        
        # Print the pitch
        if "final_pitch" in pitch_result and "pitch" in pitch_result["final_pitch"]:
            print("\n=== GENERATED PITCH ===\n")
            print(pitch_result["final_pitch"]["pitch"])
            print("\n=== INITIAL OFFER ===\n")
            print(json.dumps(pitch_result["final_pitch"]["initial_offer"], indent=2))
        else:
            print("\n=== PITCH GENERATION INCOMPLETE ===\n")
            print("The pitch generation process did not complete successfully.")
            print("Check the logs for more information.")
    
    except Exception as e:
        logging.error(f"Error generating pitch: {str(e)}")
        logging.error(f"Traceback:", exc_info=True)
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 