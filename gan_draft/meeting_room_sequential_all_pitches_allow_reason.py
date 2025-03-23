import os
import time
import json
import re
import datetime
import logging
import sys
import csv
import argparse
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, redirect_stderr

from meeting_room_sequential_single_pitch_allow_reason import SharkTankPitchTeam, load_data, TeeLogger, strip_ansi_codes, strip_thinking_tokens

def setup_arg_parser():
    """set up and return the argument parser"""
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches for all instances in the facts data.')
    
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                        help='Model ID to use for all agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of instances to process (default: process all)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug', default=True,
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    return parser

def process_instance(instance_key, instance_data, base_output_dir, args, logger):
    """process a single instance and return stats"""
    instance_dir = os.path.join(base_output_dir, instance_key)
    os.makedirs(instance_dir, exist_ok=True)
    
    instance_log_file = os.path.join(instance_dir, f"{instance_key}_log.txt")
    
    logger.info(f"Processing instance: {instance_key}")
    logger.info(f"Output directory: {instance_dir}")
    logger.info(f"Log file: {instance_log_file}")
    
    facts = str(instance_data.get('facts', "")) if isinstance(instance_data, dict) else str(instance_data)
    product_description = str(instance_data.get('product_description', "")) if isinstance(instance_data, dict) else str(instance_data)
    
    # clean any reasoning tokens from input
    facts = strip_thinking_tokens(facts)
    product_description = strip_thinking_tokens(product_description)
    
    team = SharkTankPitchTeam(
        output_dir=instance_dir,
        log_file_path=instance_log_file,
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    
    start_time = time.time()
    try:
        final_pitch = team.generate_pitch(facts, product_description, instance_key)
        execution_time = time.time() - start_time
        logger.info(f"Successfully generated pitch for {instance_key} in {execution_time:.2f} seconds")
        
        metrics_file = os.path.join(instance_dir, "metrics.json")
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        result_stats = {
            "Scenario": instance_key,
            "Framework": "meeting_room_sequential",
            "Model_name": args.model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": metrics.get("total_input_prompt_length", 0),
            "Total_Output_Response_String_Length": metrics.get("total_output_response_length", 0),
            "Final_Response": final_pitch
        }
        
        stats_file = os.path.join(instance_dir, f"{instance_key}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result_stats, f, indent=2)
        
        logger.info(f"Saved result statistics to {stats_file}")
        return result_stats
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Error processing instance {instance_key}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        result_stats = {
            "Scenario": instance_key,
            "Framework": "meeting_room_sequential",
            "Model_name": args.model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": len(facts) + len(product_description),
            "Total_Output_Response_String_Length": 0,
            "Final_Response": {"error": str(e)}
        }
        
        stats_file = os.path.join(instance_dir, f"{instance_key}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result_stats, f, indent=2)
        
        logger.info(f"Saved error statistics to {stats_file}")
        return result_stats

def save_to_csv(results, output_dir, logger):
    """save results to a csv file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"sequential_pitch_all_allow_reason_results_compile_{timestamp}.csv")
    
    columns = [
        "SN", "Scenario", "Framework", "Model_name", "Temperature", "Max_token",
        "Total_time_taken", "Total_Input_Prompt_String_Length", "Total_Output_Response_String_Length",
        "Final_Response"
    ]
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for i, result in enumerate(results, 1):
            row = {k: v for k, v in result.items() if k in columns}
            if "Final_Response" in result and result["Final_Response"]:
                row["Final_Response"] = json.dumps(result["Final_Response"])
            row["SN"] = i
            writer.writerow(row)
    
    logger.info(f"Saved all results to CSV file: {csv_file}")
    return timestamp

def main():
    """main function to run pitch generation for all instances"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./evaluation_outputs/sequential_pitch_all_{timestamp}"
    os.makedirs(base_output_dir, exist_ok=True)
    
    main_log_file = os.path.join(base_output_dir, f"main_process_log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("pitch_all_instances")
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(main_log_file, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(main_log_file, 'a', 'utf-8', original_stderr)
    
    logger.info(f"Starting SharkTank pitch generator for all instances")
    logger.info(f"Model: {args.model}, Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    logger.info(f"Base output directory: {base_output_dir}")
    
    facts_file_path = "./data/all_processed_facts.json"
    if not os.path.exists(facts_file_path):
        logger.error(f"Facts file not found at {facts_file_path}")
        return
    
    all_facts = load_data(facts_file_path)
    if not all_facts:
        logger.error("Failed to load facts data")
        return
    
    instances = list(all_facts.keys())
    
    if args.limit is not None and args.limit > 0:
        instances = instances[:args.limit]
        logger.info(f"Processing limited set of {len(instances)} instances")
    else:
        logger.info(f"Processing all {len(instances)} instances")
    
    results = []
    for instance_key in instances:
        instance_data = all_facts[instance_key]
        result = process_instance(instance_key, instance_data, base_output_dir, args, logger)
        results.append(result)
    
    csv_timestamp = save_to_csv(results, base_output_dir, logger)
    
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(f"All done! Processed {len(results)} instances.")
    print(f"Results saved to: {base_output_dir}")
    print(f"CSV file: {os.path.join(base_output_dir, f'sequential_pitch_all_allow_reason_results_compile_{csv_timestamp}.csv')}")

if __name__ == "__main__":
    main() 