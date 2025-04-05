import os
import json
import re
import argparse
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from agents import set_default_openai_key
from typing import Dict, Any, Optional
import sys
import openai

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    sets up the logging config with console and optional file output
    """
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    root_logger.propagate = True
    
    logging.info(f"Logging initialized. Log level: {log_level}, Log file: {log_file}")

def load_api_key() -> str:
    """
    loads openai api key from env file or environment vars
    """
    env_path = Path(".env")
    if env_path.exists():
        print(f"Loading .env from: {env_path.absolute()}")
        load_dotenv(dotenv_path=env_path, override=True)
        
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key and env_path.exists():
        with env_path.open("r") as f:
            for line in f:
                if line.strip() and not line.startswith("#"):
                    try:
                        key, value = line.strip().split("=", 1)
                        if key == "OPENAI_API_KEY":
                            api_key = value
                            break
                    except ValueError:
                        continue
    
    if not api_key:
        raise ValueError("OpenAI API key not found. Please set it in the .env file or as an environment variable.")
    
    set_default_openai_key(api_key)
    openai.api_key = api_key
    os.environ["OPENAI_API_KEY"] = api_key
    
    return api_key

def parse_json_from_response(response: Any) -> Dict[str, Any]:
    """
    tries to extract json from agent responses in various formats
    """
    logging.debug(f"Parsing JSON from response: {type(response)}")
    
    if not response:
        logging.warning("Empty response received for JSON parsing")
        return {"error": "Empty response"}
    
    if isinstance(response, dict):
        logging.debug("Response is already a dictionary, no parsing needed")
        return response
    
    if not isinstance(response, str):
        logging.warning(f"Non-string response received for JSON parsing: {type(response)}")
        response = str(response)
    
    try:
        logging.debug("Attempting direct JSON parsing")
        return json.loads(response)
    except json.JSONDecodeError:
        logging.debug("Direct JSON parsing failed, trying alternative methods")
    
    code_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    code_blocks = re.findall(code_block_pattern, response)
    
    if code_blocks:
        logging.debug(f"Found {len(code_blocks)} code blocks in response")
        for block in code_blocks:
            try:
                logging.debug("Attempting to parse JSON from code block")
                return json.loads(block)
            except json.JSONDecodeError:
                continue
    
    json_pattern = r"{[\s\S]*}"
    json_matches = re.search(json_pattern, response)
    
    if json_matches:
        try:
            logging.debug("Attempting to parse JSON using regex match")
            return json.loads(json_matches.group(0))
        except json.JSONDecodeError:
            logging.debug("JSON regex parsing failed")
    
    logging.warning("All JSON parsing methods failed, attempting manual key-value extraction")
    
    kv_pattern = r'"([^"]+)"\s*:\s*(?:"([^"]*)"|\[([^\]]*)\]|(\d+)|(\{[^\}]*\})|(true|false|null)|([^,}\s]+))'
    kv_matches = re.findall(kv_pattern, response)
    
    if kv_matches:
        logging.debug(f"Found {len(kv_matches)} key-value pairs using regex")
        result = {}
        for match in kv_matches:
            key = match[0]
            value = next((v for v in match[1:] if v), "")
            result[key] = value
        return result
    
    logging.error("All parsing methods failed, returning original text in a structured format")
    return {
        "original_text": response,
        "parsing_error": "Could not parse response as JSON"
    }

def format_agent_input(input_text: str, context: Dict[str, Any] = None) -> str:
    """
    formats input text with optional context for agents
    """
    if context:
        context_str = "\n\nContext:\n"
        for key, value in context.items():
            context_str += f"{key}: {value}\n"
        return input_text + context_str
    return input_text

def save_output(output_data, episode_key, output_dir="./output"):
    """
    saves output data to json file in the specified directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    clean_key = os.path.basename(episode_key).replace(".txt", "")
    output_file = os.path.join(output_dir, f"{clean_key}_pitch.json")
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)
    
    return output_file

def parse_single_pitch_args():
    """
    parses command line args for single pitch generation
    """
    parser = argparse.ArgumentParser(description="Generate a pitch for a specific episode")
    parser.add_argument("episode_key", help="The episode key to process")
    parser.add_argument("--data_path", default="../data/all_processed_facts.json", help="Path to the JSON data file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save the output")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    return parser.parse_args()

def parse_all_episodes_args():
    """
    parses command line args for processing all episodes
    """
    parser = argparse.ArgumentParser(description="Process all episodes and generate pitches")
    parser.add_argument("--data_path", default="../data/all_processed_facts.json", help="Path to the JSON data file")
    parser.add_argument("--output_dir", default="./output", help="Directory to save the output")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of episodes to process")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        help="Logging level")
    return parser.parse_args()

def get_log_level(level_str):
    """
    converts string log level to logging constant
    """
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)

def format_agent_response(agent_name, response):
    """
    formats agent response for nice logging output
    """
    try:
        if isinstance(response, str) and response.strip().startswith('{'):
            response_dict = json.loads(response)
            formatted = json.dumps(response_dict, indent=2)
        elif isinstance(response, dict):
            formatted = json.dumps(response, indent=2)
        else:
            formatted = str(response)
        
        return f"\n===== {agent_name} RESPONSE =====\n{formatted}\n===== END {agent_name} RESPONSE ====="
    except Exception as e:
        return f"\n===== {agent_name} RESPONSE (Error formatting: {str(e)}) =====\n{str(response)}\n===== END {agent_name} RESPONSE =====" 