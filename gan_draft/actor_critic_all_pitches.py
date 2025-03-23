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

from actor_critic_single_pitch import ActorCriticPitchTeam, load_data, TeeLogger, strip_ansi_codes, strip_thinking_tokens

def setup_arg_parser():
    # set up command line args
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches for all instances using Actor-Critic framework.')
    
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                        help='Model ID to use for all agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of instances to process (default: process all)')

    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of iterations for Actor-Critic loop (default: 5)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug', default=True,
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    return parser

def process_instance(instance_key, instance_data, base_output_dir, args, logger):
    # handle a single instance using actor-critic approach
    instance_dir = os.path.join(base_output_dir, instance_key)
    os.makedirs(instance_dir, exist_ok=True)
    
    instance_log_file = os.path.join(instance_dir, f"{instance_key}_log.txt")
    
    logger.info(f"Processing instance: {instance_key}")
    logger.info(f"Output directory: {instance_dir}")
    logger.info(f"Log file: {instance_log_file}")
    
    facts = str(instance_data.get('facts', "")) if isinstance(instance_data, dict) else str(instance_data)
    product_description = str(instance_data.get('product_description', "")) if isinstance(instance_data, dict) else str(instance_data)
    
    # clean up any thinking tokens in input
    facts = strip_thinking_tokens(facts)
    product_description = strip_thinking_tokens(product_description)
    
    team = ActorCriticPitchTeam(
        output_dir=instance_dir,
        log_file_path=instance_log_file,
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug,
        max_iterations=args.max_iterations
    )
    
    start_time = time.time()
    try:
        team.generate_pitch(facts, product_description, instance_key)
        execution_time = time.time() - start_time
        logger.info(f"Completed pitch generation for {instance_key} in {execution_time:.2f} seconds")
        
        final_pitch = {}
        metrics = {}
        iterations_data = []
        
        # load final pitch json
        final_pitch_file = os.path.join(instance_dir, "final_pitch.json")
        if os.path.exists(final_pitch_file):
            try:
                with open(final_pitch_file, 'r', encoding='utf-8') as f:
                    final_pitch = json.load(f)
                logger.info(f"Loaded final pitch from {final_pitch_file}")
            except Exception as e:
                logger.error(f"Could not load final pitch: {str(e)}")
                final_pitch = {"error": f"Could not load final pitch: {str(e)}"}
        else:
            logger.warning(f"Final pitch file not found at {final_pitch_file}")
            final_pitch = {"error": "Final pitch file not found"}
            
        # load metrics json
        metrics_file = os.path.join(instance_dir, "metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    metrics = json.load(f)
                logger.info(f"Loaded metrics from {metrics_file}")
            except Exception as e:
                logger.error(f"Could not load metrics: {str(e)}")
                metrics = {}
        else:
            logger.warning(f"Metrics file not found at {metrics_file}")
            
        # load iterations json
        iterations_file = os.path.join(instance_dir, "iterations.json")
        if os.path.exists(iterations_file):
            try:
                with open(iterations_file, 'r', encoding='utf-8') as f:
                    iterations_data = json.load(f)
                logger.info(f"Loaded iterations data from {iterations_file}")
            except Exception as e:
                logger.error(f"Could not load iterations data: {str(e)}")
        else:
            logger.warning(f"Iterations file not found at {iterations_file}")
        
        num_iterations = metrics.get("iterations_completed", 0)
        best_score = metrics.get("best_score", 0)
        
        if num_iterations == 0 and iterations_data:
            num_iterations = len(iterations_data)
            
        if best_score == 0 and iterations_data and "Feedback" in iterations_data[-1]:
            best_score = iterations_data[-1]["Feedback"].get("Score", 0)
        
        # compile stats about this run
        result_stats = {
            "Scenario": instance_key,
            "Framework": "actor_critic",
            "Model_name": args.model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Max_iterations": args.max_iterations,
            "Actual_iterations": num_iterations,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": metrics.get("total_input_prompt_length", 0),
            "Total_Output_Response_String_Length": metrics.get("total_output_response_length", 0),
            "Final_Score": best_score,
            "Final_Response": final_pitch
        }
        
        # add details for each iteration
        for i in range(1, args.max_iterations + 1):
            iter_data = next((item for item in iterations_data if item.get("Iteration") == i), None)
            
            if iter_data:
                if "Draft" in iter_data:
                    result_stats[f"Iteration_{i}_Draft"] = json.dumps(iter_data["Draft"])
                else:
                    result_stats[f"Iteration_{i}_Draft"] = "(missing data)"
                
                if "Feedback" in iter_data:
                    result_stats[f"Iteration_{i}_Feedback"] = json.dumps(iter_data["Feedback"])
                else:
                    result_stats[f"Iteration_{i}_Feedback"] = "(missing data)"
            else:
                if i > metrics.get("num_iterations", 0):
                    result_stats[f"Iteration_{i}_Draft"] = "(not needed)"
                    result_stats[f"Iteration_{i}_Feedback"] = "(not needed)"
                else:
                    result_stats[f"Iteration_{i}_Draft"] = "(missing data)"
                    result_stats[f"Iteration_{i}_Feedback"] = "(missing data)"
        
        # save stats to json
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
        
        # create error stats
        result_stats = {
            "Scenario": instance_key,
            "Framework": "actor_critic",
            "Model_name": args.model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Max_iterations": args.max_iterations,
            "Actual_iterations": 0,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": len(facts) + len(product_description),
            "Total_Output_Response_String_Length": 0,
            "Final_Score": 0,
            "Final_Response": {"error": str(e)}
        }
        
        for i in range(1, args.max_iterations + 1):
            result_stats[f"Iteration_{i}_Draft"] = "(error)"
            result_stats[f"Iteration_{i}_Feedback"] = "(error)"
        
        stats_file = os.path.join(instance_dir, f"{instance_key}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result_stats, f, indent=2)
        
        logger.info(f"Saved error statistics to {stats_file}")
        return result_stats

def save_to_csv(results, output_dir, logger):
    # save all results to csv with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"actor_critic_all_pitches_results_compile_{timestamp}.csv")
    
    base_columns = [
        "SN", "Scenario", "Framework", "Model_name", "Temperature", "Max_token", 
        "Max_iterations", "Actual_iterations", "Total_time_taken", 
        "Total_Input_Prompt_String_Length", "Total_Output_Response_String_Length",
        "Final_Score", "Final_Response"
    ]
    
    max_iterations = 5
    iteration_columns = []
    for i in range(1, max_iterations + 1):
        iteration_columns.extend([f"Iteration_{i}_Draft", f"Iteration_{i}_Feedback"])
    
    columns = base_columns + iteration_columns
    
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        
        for i, result in enumerate(results, 1):
            row = {k: v for k, v in result.items() if k in columns}
            
            for column in columns:
                if column in row and isinstance(row[column], (dict, list)):
                    try:
                        row[column] = json.dumps(row[column])
                    except (TypeError, ValueError) as e:
                        logger.warning(f"Could not serialize {column} for {result.get('Scenario', 'unknown')}: {str(e)}")
                        row[column] = str(row[column])
            
            row["SN"] = i
            
            writer.writerow(row)
    
    logger.info(f"Saved all results to CSV file: {csv_file}")
    return timestamp

def main():
    # main entry point for running actor-critic pitch generation
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./evaluation_outputs/actor_critic_all_{timestamp}"
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
    logger = logging.getLogger("pitch_all_instances_actor_critic")
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(main_log_file, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(main_log_file, 'a', 'utf-8', original_stderr)
    
    logger.info(f"Starting SharkTank pitch generator for all instances using Actor-Critic approach")
    logger.info(f"Model: {args.model}, Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info(f"Max iterations: {args.max_iterations}")
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
    print(f"CSV file: {os.path.join(base_output_dir, f'actor_critic_all_pitches_results_compile_{csv_timestamp}.csv')}")

if __name__ == "__main__":
    main() 