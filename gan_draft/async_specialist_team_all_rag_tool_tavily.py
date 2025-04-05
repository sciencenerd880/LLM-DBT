import os
import time
import json
import re
import datetime
import logging
import sys
import csv
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, redirect_stderr

from async_specialist_team_single_rag_tool_tavily import AsyncSpecialistTeamRAGTavily, load_data, TeeLogger, strip_ansi_codes, strip_thinking_tokens

def setup_arg_parser():
    """set up and return the argument parser"""
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches for all instances using an asynchronous specialist team with RAG and Tavily search tools.')
    
    parser.add_argument('--specialist-model', type=str, default='llama-3.3-70b-versatile',
                        help='Model ID to use for specialist agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--pitch-model', type=str, default='deepseek-r1-distill-llama-70b',
                        help='Model ID to use for pitch agents (default: deepseek-r1-distill-llama-70b)')
    
    parser.add_argument('--pdf-urls', type=str, nargs='+',
                        help='URLs of PDF documents to load into the knowledge base (default: HBS pitch materials)')
    
    parser.add_argument('--pdf-paths', type=str, nargs='+',
                        help='Local paths to PDF documents (default: data/pdfs/hbs_*.pdf)')
    
    parser.add_argument('--chunking', type=str, default='fixed', choices=['fixed', 'agentic', 'semantic'],
                        help='Chunking strategy for the knowledge base (default: fixed)')
    
    parser.add_argument('--force-reload', action='store_true',
                        help='Force reloading and re-embedding of documents even if they exist in the database')
    
    parser.add_argument('--skip-embedding-check', action='store_true',
                        help='Skip checking if documents are already embedded (useful if check is failing)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of instances to process (default: process all)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug', default=True,
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    parser.add_argument('--concurrent', type=int, default=1,
                        help='Number of instances to process concurrently (default: 1)')
    
    return parser

async def process_instance_async(instance_key, instance_data, base_output_dir, args, logger):
    """process a single instance asynchronously and return stats"""
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
    
    # Set environment variables for RAG configuration so the AsyncSpecialistTeamRAGTavily class can access them
    if args.pdf_urls:
        os.environ["RAG_PDF_URLS"] = json.dumps(args.pdf_urls)
    if args.pdf_paths:
        os.environ["RAG_PDF_PATHS"] = json.dumps(args.pdf_paths)
    if args.chunking:
        os.environ["RAG_CHUNKING_STRATEGY"] = args.chunking
    if args.force_reload:
        os.environ["RAG_FORCE_RELOAD"] = "true"
    if args.skip_embedding_check:
        os.environ["RAG_SKIP_EMBEDDING_CHECK"] = "true"
    
    # Only pass the parameters that AsyncSpecialistTeamRAGTavily's __init__ accepts
    team = AsyncSpecialistTeamRAGTavily(
        output_dir=instance_dir,
        log_file_path=instance_log_file,
        specialist_model_id=args.specialist_model,
        pitch_model_id=args.pitch_model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    
    start_time = time.time()
    try:
        final_pitch = await team.generate_pitch(facts, product_description, instance_key)
        execution_time = time.time() - start_time
        logger.info(f"Successfully generated pitch for {instance_key} in {execution_time:.2f} seconds")
        
        metrics_file = os.path.join(instance_dir, "metrics.json")
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # Get tool call information from metrics
        tool_calls_count = metrics.get("total_tool_calls", 0)
        logger.info(f"Total tool calls for {instance_key}: {tool_calls_count}")
        
        result_stats = {
            "Scenario": instance_key,
            "Framework": "async_specialist_team_rag_tavily",
            "Specialist_Model_name": args.specialist_model,
            "Pitch_Model_name": args.pitch_model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": metrics.get("total_input_prompt_length", 0),
            "Total_Output_Response_String_Length": metrics.get("total_output_response_length", 0),
            "Total_Tool_Calls": tool_calls_count,
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
            "Framework": "async_specialist_team_rag_tavily",
            "Specialist_Model_name": args.specialist_model,
            "Pitch_Model_name": args.pitch_model,
            "Temperature": args.temperature,
            "Max_token": args.max_tokens,
            "Total_time_taken": execution_time,
            "Total_Input_Prompt_String_Length": len(facts) + len(product_description),
            "Total_Output_Response_String_Length": 0,
            "Total_Tool_Calls": 0,
            "Final_Response": {"error": str(e)}
        }
        
        stats_file = os.path.join(instance_dir, f"{instance_key}_stats.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(result_stats, f, indent=2)
        
        logger.info(f"Saved error statistics to {stats_file}")
        return result_stats

def save_to_csv(results, output_dir, logger):
    """save results to a csv file including tool call information"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"async_specialist_team_rag_tavily_all_results_compile_{timestamp}.csv")
    
    columns = [
        "SN", "Scenario", "Framework", "Specialist_Model_name", "Pitch_Model_name", 
        "Temperature", "Max_token", "Total_time_taken", 
        "Total_Input_Prompt_String_Length", "Total_Output_Response_String_Length",
        "Total_Tool_Calls",
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

async def process_batch_async(batch, base_output_dir, args, logger, all_facts):
    """Process a batch of instances concurrently"""
    tasks = []
    for instance_key in batch:
        instance_data = all_facts[instance_key]
        task = process_instance_async(instance_key, instance_data, base_output_dir, args, logger)
        tasks.append(task)
    
    return await asyncio.gather(*tasks, return_exceptions=True)

async def main_async():
    """main async function to run pitch generation for all instances with RAG and Tavily search tools"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./evaluation_outputs/async_specialist_team_rag_tavily_all_{timestamp}"
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
    logger = logging.getLogger("async_pitch_rag_tavily_all_instances")
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(main_log_file, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(main_log_file, 'a', 'utf-8', original_stderr)
    
    logger.info(f"Starting SharkTank pitch generator for all instances using async specialist team with RAG and Tavily search tools")
    logger.info(f"Specialist model: {args.specialist_model}, Pitch model: {args.pitch_model}")
    logger.info(f"Max tokens: {args.max_tokens}, Temperature: {args.temperature}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    logger.info(f"Concurrent instances: {args.concurrent}")
    logger.info(f"Base output directory: {base_output_dir}")
    
    # Log RAG configuration
    logger.info("RAG Configuration:")
    logger.info(f"  PDF URLs: {args.pdf_urls if args.pdf_urls else 'default'}")
    logger.info(f"  PDF Paths: {args.pdf_paths if args.pdf_paths else 'default'}")
    logger.info(f"  Chunking Strategy: {args.chunking}")
    logger.info(f"  Force Reload: {args.force_reload}")
    logger.info(f"  Skip Embedding Check: {args.skip_embedding_check}")
    
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
    
    # Process instances in batches to control concurrency
    results = []
    num_concurrent = max(1, min(args.concurrent, len(instances)))
    batch_size = num_concurrent
    
    for i in range(0, len(instances), batch_size):
        batch = instances[i:i+batch_size]
        logger.info(f"Processing batch of {len(batch)} instances (batch {i//batch_size + 1})")
        
        batch_results = await process_batch_async(batch, base_output_dir, args, logger, all_facts)
        
        # Add successful results to the results list
        for res in batch_results:
            if not isinstance(res, Exception):
                results.append(res)
            else:
                logger.error(f"Batch processing error: {res}")
    
    csv_timestamp = save_to_csv(results, base_output_dir, logger)
    
    sys.stdout.flush()
    sys.stderr.flush()
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    
    print(f"All done! Processed {len(results)} instances with RAG and Tavily search tools.")
    print(f"Results saved to: {base_output_dir}")
    print(f"CSV file: {os.path.join(base_output_dir, f'async_specialist_team_rag_tavily_all_results_compile_{csv_timestamp}.csv')}")

def main():
    """entry point that calls the async main function"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())

if __name__ == "__main__":
    main() 