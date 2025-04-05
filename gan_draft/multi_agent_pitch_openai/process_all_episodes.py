import os
import json
import asyncio
import logging
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from pitch_team import generate_pitch
from utils import load_api_key, parse_all_episodes_args, setup_logging, get_log_level
from guardrails import GLOBAL_ATTEMPT_COUNTS  # import the global attempt counter

def normalize_data(data, default_name="Unknown"):
    """
    normalize data to make sure format is consistent regardless of input type
    """
    if isinstance(data, dict):
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

def save_output(pitch_result, episode_key, output_dir):
    """
    save the pitch result to a json file with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(output_dir, f"batch_{timestamp}")
    os.makedirs(batch_dir, exist_ok=True)
    
    clean_key = os.path.basename(episode_key).replace(".txt", "")
    output_file = os.path.join(batch_dir, f"{clean_key}_pitch.json")
    
    result = {
        "episode_key": episode_key,
        "original_data": {
            "facts": pitch_result.get("facts", ""),
            "product_description": pitch_result.get("product_description", {})
        },
        "generated_pitch": pitch_result
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return output_file

async def process_episode(episode_key, episode_data, model, output_dir, batch_timestamp):
    """
    process a single episode and generate a pitch
    """
    logger = logging.getLogger(f"episode.{episode_key}")
    logger.info(f"Processing episode: {episode_key}")
    
    global GLOBAL_ATTEMPT_COUNTS
    GLOBAL_ATTEMPT_COUNTS.clear()
    logger.info("Reset global attempt counter for a fresh episode")
    
    try:
        facts = episode_data.get("facts", "")
        product_description = episode_data.get("product_description", {"name": "Unknown Product"})
        
        logger.debug(f"Original facts type: {type(facts).__name__}")
        logger.debug(f"Original product_description type: {type(product_description).__name__}")
        
        product_description = normalize_data(product_description, default_name="Unknown Product")
        product_name = product_description.get("name", "Unknown Product")
        
        if not isinstance(facts, (str, dict)):
            logger.info(f"Converting facts from {type(facts).__name__} to dictionary format for {episode_key}")
            facts = normalize_data(facts, default_name="Facts")
            
        logger.info(f"Product: {product_name}")
        
        # store facts and product info in env vars for fact checkers
        os.environ["PITCH_FACTS"] = json.dumps(facts) if isinstance(facts, dict) else facts
        os.environ["PITCH_PRODUCT_DESCRIPTION"] = json.dumps(product_description)
        
        logger.info("Starting pitch generation process...")
        pitch_result = await generate_pitch(facts, product_description, model=model)
        
        batch_dir = os.path.join(output_dir, f"batch_{batch_timestamp}")
        os.makedirs(batch_dir, exist_ok=True)
        
        output_file = os.path.join(batch_dir, f"{episode_key}_pitch.json")
        
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
        
        logger.info(f"Pitch generated successfully. Output saved to {output_file}")
        
        if "final_pitch" in pitch_result and "pitch" in pitch_result["final_pitch"]:
            print(f"\n=== GENERATED PITCH FOR {product_name} ===")
            print(f"Initial offer: {pitch_result['final_pitch']['initial_offer']}")
            print(f"Output saved to: {output_file}")
        else:
            print(f"\n=== PITCH GENERATION INCOMPLETE FOR {product_name} ===")
        
        return True
    except Exception as e:
        logger.error(f"Error processing episode {episode_key}: {str(e)}", exc_info=True)
        return False

async def main():
    args = parse_all_episodes_args()
    
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    log_dir = os.path.join("logs", f"batch_{batch_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"process_all_episodes_{batch_timestamp}.log")
    
    log_level = get_log_level(args.log_level)
    setup_logging(log_level=log_level, log_file=log_file)
    
    logger = logging.getLogger("batch_processor")
    logger.info(f"Starting batch processing of episodes with timestamp {batch_timestamp}")
    logger.info(f"Logs will be saved to {log_file}")
    logger.info(f"Outputs will be saved to {os.path.join(args.output_dir, f'batch_{batch_timestamp}')}")
    
    api_key = load_api_key()
    os.environ["OPENAI_API_KEY"] = api_key
    
    logger.info(f"Loading data from {args.data_path}")
    try:
        with open(args.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise
    
    episodes = list(data.keys())
    if args.limit:
        episodes = episodes[:args.limit]
    
    logger.info(f"Processing {len(episodes)} episodes")
    
    batch_output_dir = os.path.join(args.output_dir, f"batch_{batch_timestamp}")
    os.makedirs(batch_output_dir, exist_ok=True)
    
    results = []
    for episode_key in tqdm(episodes, desc="Processing episodes"):
        # clean up env vars between episodes
        if "PITCH_FACTS" in os.environ:
            del os.environ["PITCH_FACTS"]
        if "PITCH_PRODUCT_DESCRIPTION" in os.environ:
            del os.environ["PITCH_PRODUCT_DESCRIPTION"]
            
        success = await process_episode(
            episode_key, 
            data[episode_key], 
            args.model, 
            args.output_dir,
            batch_timestamp
        )
        results.append((episode_key, success))
    
    successful = [key for key, success in results if success]
    failed = [key for key, success in results if not success]
    
    logger.info("=== Processing Summary ===")
    logger.info(f"Total episodes: {len(episodes)}")
    logger.info(f"Successful: {len(successful)}")
    logger.info(f"Failed: {len(failed)}")
    
    if failed:
        logger.warning("Failed episodes:")
        for key in failed:
            logger.warning(f"  - {key}")
    
    print("\n=== BATCH PROCESSING SUMMARY ===")
    print(f"Total episodes processed: {len(episodes)}")
    print(f"Successfully processed: {len(successful)}")
    print(f"Failed to process: {len(failed)}")
    print(f"Logs saved to: {log_file}")
    print(f"Outputs saved to: {batch_output_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 