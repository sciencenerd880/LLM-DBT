import os
import time
import json
import re
import datetime
import logging
import sys
import argparse
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, redirect_stderr
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
    if os.path.exists(llm_dbt_env_path):
        load_dotenv(llm_dbt_env_path)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set GROQ_API_KEY in your environment or in the .env file.")

def strip_ansi_codes(text):
    """remove ANSI escape sequences from text to make logs more readable"""
    if not isinstance(text, str):
        return text
        
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def strip_thinking_tokens(text):
    """remove thinking tokens from LLM outputs"""
    if not isinstance(text, str):
        return text
    
    thinking_pattern = re.compile(r'<thinking>.*?</thinking>', re.DOTALL)
    text = thinking_pattern.sub('', text)
    
    reasoning_pattern = re.compile(r'(thinking|reasoning):\s*.*?(?=\n\s*\n|$)', re.DOTALL | re.IGNORECASE)
    text = reasoning_pattern.sub('', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text

class TeeLogger:
    """duplicates output to both a file and the original stream"""
    def __init__(self, filename, mode='w', encoding='utf-8', stream=None):
        self.file = open(filename, mode, encoding=encoding)
        self.stream = stream if stream else sys.stdout
        self.encoding = encoding
        
    def write(self, data):
        clean_data = strip_ansi_codes(data)
        self.file.write(clean_data)
        self.file.flush()
        self.stream.write(data)
        self.stream.flush()
        
    def flush(self):
        self.file.flush()
        self.stream.flush()
        
    def close(self):
        if self.file:
            self.file.close()

class InvestmentOffer(BaseModel):
    Valuation: str = Field(..., description="Estimated company valuation (e.g., $10 million)")
    Equity_Offered: str = Field(..., description="Percentage of equity offered to investors (e.g., 10%)")
    Funding_Amount: str = Field(..., description="The amount of funding requested (e.g., $1 million)")
    Key_Terms: str = Field("None", description="Any additional key terms (optional)")

class SharkTankPitch(BaseModel):
    Pitch: str = Field(..., description="The complete investment pitch text")
    Initial_Offer: InvestmentOffer = Field(..., description="The investment offer details")

class SharkTankPitchTeam:
    """coordinates the multi-agent team for SharkTank pitch generation using a sequential workflow"""
    
    def __init__(self, output_dir, log_file_path, model_id="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.7, debug_mode=True):
        """set up the team with output directory, logging, and model parameters"""
        self.logger = logging.getLogger(f"pitch_team_{os.path.basename(output_dir)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        if not GROQ_API_KEY:
            self.logger.error("GROQ_API_KEY not found. Please set GROQ_API_KEY in your environment variables.")
            raise ValueError("GROQ_API_KEY not found in environment variables. Please set GROQ_API_KEY in your environment or in the .env file.")
        
        self.model_id = model_id
        self.debug_mode = debug_mode
        self.logger.info(f"Using model: {self.model_id}")
        self.logger.info(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'}")
        
        self.log_file_path = log_file_path
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== SHARK TANK PITCH GENERATION LOG (Model: {self.model_id}) ===\n\n")
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger.info(f"Using max_tokens: {self.max_tokens}, temperature: {self.temperature}")
        
        def create_groq_model():
            return Groq(
                id=self.model_id,
                api_key=GROQ_API_KEY,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        # set up the team of specialist agents
        self.financial_strategist = Agent(
            name="Financial Strategist",
            model=create_groq_model(),
            description="You are a Financial Strategist with expertise in startup valuation, investment analysis, and business model evaluation.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to analyze the financial aspects.",
                "Always respond in English only.",
                "Analyze the provided facts and product description to develop a financial strategy for a SharkTank pitch, including:",
                "1. A justified valuation for the company",
                "2. An appropriate investment amount to request",
                "3. A fair equity percentage to offer",
                "4. A breakdown of how the funds will be used",
                "5. A realistic ROI timeline",
                "6. Potential exit strategies",
                "Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.",
                "Do not invent or assume financial data that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your recommendations."
            ],
            debug_mode=self.debug_mode
        )
        
        self.market_research_specialist = Agent(
            name="Market Research Specialist",
            model=create_groq_model(),
            description="You are a Market Research Specialist with deep knowledge of consumer trends, market analysis, and competitive landscapes.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to analyze the market aspects.",
                "Always respond in English only.",
                "Analyze the provided facts and product description to develop market insights for a SharkTank pitch, including:",
                "1. The estimated size of the target market",
                "2. Description of target customer segments",
                "3. Analysis of competitors and their strengths/weaknesses",
                "4. Relevant market trends",
                "5. Potential growth opportunities",
                "6. Challenges in the market",
                "You will also be provided with a financial analysis from the Financial Strategist. Please consider this information in your analysis.",
                "Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.",
                "Do not make up market sizes or competitor information that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            debug_mode=self.debug_mode
        )
        
        self.product_technical_advisor = Agent(
            name="Product Technical Advisor",
            model=create_groq_model(),
            description="You are a Product/Technical Advisor with expertise in product development, technical feasibility, and innovation assessment.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to analyze the product/technical aspects.",
                "Always respond in English only.",
                "Analyze the provided facts and product description to develop product insights for a SharkTank pitch, including:",
                "1. Key product features to highlight",
                "2. Technical advantages over competitors",
                "3. How to effectively demonstrate the product",
                "4. Assessment of production/technical scalability",
                "5. Potential future product developments",
                "You will also be provided with financial and market analyses from previous team members. Please consider this information in your analysis.",
                "Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.",
                "Do not invent capabilities or exaggerate performance in ways that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your assessments."
            ],
            debug_mode=self.debug_mode
        )
        
        self.shark_psychology_expert = Agent(
            name="Shark Psychology Expert",
            model=create_groq_model(),
            description="You are a Shark Psychology Expert who understands the motivations, preferences, and decision patterns of SharkTank investors.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to analyze investor psychology.",
                "Always respond in English only.",
                "Analyze the provided facts and product description to develop investor psychology insights for a SharkTank pitch, including:",
                "1. Points that will appeal to Sharks",
                "2. Potential objections and how to counter them",
                "3. Strategy for negotiating with Sharks",
                "4. Tips for effective presentation",
                "5. Sharks that might be the best fit and why",
                "You will also be provided with financial, market, and technical analyses from previous team members. Please consider this information in your analysis.",
                "Base your analysis primarily on the facts provided, but you may use your knowledge of Shark Tank investors for reasonable inferences.",
                "Focus on general Shark psychology and preferences rather than making specific predictions that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            debug_mode=self.debug_mode
        )
        
        self.pitch_drafter = Agent(
            name="Pitch Drafter",
            model=create_groq_model(),
            description="You are a skilled pitch writer for entrepreneurs appearing on Shark Tank.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to draft the actual pitch.",
                "Always respond in English only.",
                "Your task is to create a compelling pitch based on the specialist analyses provided. You will receive:",
                "1. The original product facts and description",
                "2. Financial analysis",
                "3. Market research analysis",
                "4. Product/technical analysis",
                "5. Shark psychology insights",
                "The pitch should be structured to grab attention, clearly explain the product/service, highlight market potential, showcase competitive advantages, present financial data, make a specific investment ask, and close with a strong call to action."
            ],
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        self.pitch_critic = Agent(
            name="Pitch Critic",
            model=create_groq_model(),
            description="You are a Pitch Critic who identifies strengths, weaknesses, and areas for improvement in SharkTank pitches.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is to critique the draft pitch.",
                "Always respond in English only.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. All specialist analyses",
                "3. The draft pitch in JSON format",
                "Analyze the draft pitch provided and offer constructive criticism to make it more compelling and effective.",
                "Be specific in your feedback and suggest concrete improvements.",
                "Focus on:",
                "1. Strengths of the pitch that should be maintained or emphasized",
                "2. Weaknesses or aspects that could undermine the pitch's effectiveness",
                "3. Specific areas that need improvement (clarity, structure, persuasiveness, etc.)",
                "4. Concrete suggestions for improving the pitch",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your critiques."
            ],
            debug_mode=self.debug_mode
        )
        
        self.pitch_finalizer = Agent(
            name="Pitch Finalizer",
            model=create_groq_model(),
            description="You are a pitch finalization expert for entrepreneurs appearing on Shark Tank.",
            instructions=[
                "You are part of a team creating a SharkTank pitch. Your responsibility is the final version of the pitch.",
                "Always respond in English only.",
                "Your task is to refine and finalize the draft pitch based on all the specialist analyses and critic's feedback.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. All specialist analyses",
                "3. The draft pitch in JSON format",
                "4. The pitch critic's feedback",
                "Create a polished, compelling final pitch that incorporates the strengths identified by the critic while addressing the areas for improvement. The final pitch should be concise, engaging, and strategically structured to maximize appeal to the Sharks."
            ],
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        self.logger.info("SharkTank Pitch Team initialized")
    
    def log_message(self, agent_name, message):
        """log a message from an agent to the consolidated log file"""
        clean_message = strip_ansi_codes(message)
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {agent_name} ===\n\n")
            f.write(clean_message)
    
    def run_agent(self, agent, agent_name, context):
        """run an agent with given context and handle its response"""
        self.logger.info(f"{agent_name} analyzing...")
        
        try:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                response = agent.run(context)
                
            debug_output = f.getvalue()
            
            debug_output = strip_ansi_codes(debug_output)
            
            if debug_output:
                self.log_message(f"{agent_name} Debug Output", debug_output)
            
            if response:
                if hasattr(response, 'content'):
                    if hasattr(response.content, 'model_dump'):
                        content_for_conversation = json.dumps(response.content.model_dump(), indent=2)
                    else:
                        content_for_conversation = strip_thinking_tokens(strip_ansi_codes(str(response.content)))
                else:
                    content_for_conversation = strip_thinking_tokens(strip_ansi_codes(str(response)))
                
                if hasattr(response, 'content'):
                    if hasattr(response.content, 'model_dump'):
                        response_str = json.dumps(response.content.model_dump(), indent=2)
                    else:
                        response_str = strip_thinking_tokens(strip_ansi_codes(str(response.content)))
                else:
                    response_str = strip_thinking_tokens(strip_ansi_codes(str(response)))
                    
                response_length = len(response_str)
                self.logger.info(f"{agent_name} response length: {response_length}")
                
                return {"response": response, "content": content_for_conversation}
            else:
                self.logger.warning(f"{agent_name} returned empty response")
                error_message = f"No response from {agent_name}"
                self.log_message(agent_name, error_message)
                return {"response": error_message, "content": error_message}
        except Exception as e:
            self.logger.error(f"Error getting {agent_name} response: {e}")
            error_message = f"No response from {agent_name} due to error: {str(e)}"
            self.log_message(agent_name, error_message)
            return {"response": error_message, "content": error_message}
    
    def generate_pitch(self, facts, product_description, product_key):
        """generate a pitch for a SharkTank product"""
        facts_str = str(facts) if not isinstance(facts, str) else facts
        product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
        
        self.logger.info(f"Generating pitch for: {product_key}")
        
        total_input_length = len(product_description_str) + len(facts_str)
        total_output_length = 0
        
        self.logger.info(f"Initial input prompt length: {total_input_length}")
        
        base_context = f"""
        FACTS:
        {facts_str}
        
        PRODUCT DESCRIPTION:
        {product_description_str}
        """
        
        conversation_history = []
        
        start_time = time.time()
        
        financial_context = base_context
        financial_context_length = len(financial_context)
        total_input_length += financial_context_length
        
        financial_result = self.run_agent(
            self.financial_strategist, 
            "Financial Strategist", 
            financial_context
        )
        
        financial_response = financial_result["response"]
        financial_content = financial_result["content"]
        financial_content = strip_thinking_tokens(financial_content)
        
        financial_response_length = len(financial_content)
        total_output_length += financial_response_length
        conversation_history.append({"role": "Financial Strategist", "content": financial_content})
        
        market_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        market_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        market_context = strip_thinking_tokens(market_context)
        
        market_context_length = len(market_context)
        total_input_length += market_context_length
        
        market_result = self.run_agent(
            self.market_research_specialist, 
            "Market Research Specialist", 
            market_context
        )
        
        market_response = market_result["response"]
        market_content = market_result["content"]
        market_content = strip_thinking_tokens(market_content)
        
        market_response_length = len(market_content)
        total_output_length += market_response_length
        conversation_history.append({"role": "Market Research Specialist", "content": market_content})
        
        product_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        product_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        product_context += "\n\n=== Market Research Specialist ===\n\n" + market_content
        product_context = strip_thinking_tokens(product_context)
        
        product_context_length = len(product_context)
        total_input_length += product_context_length
        
        product_result = self.run_agent(
            self.product_technical_advisor, 
            "Product Technical Advisor", 
            product_context
        )
        
        product_response = product_result["response"]
        product_content = product_result["content"]
        product_content = strip_thinking_tokens(product_content)
        
        product_response_length = len(product_content)
        total_output_length += product_response_length
        conversation_history.append({"role": "Product Technical Advisor", "content": product_content})
        
        psychology_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        psychology_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        psychology_context += "\n\n=== Market Research Specialist ===\n\n" + market_content
        psychology_context += "\n\n=== Product Technical Advisor ===\n\n" + product_content
        psychology_context = strip_thinking_tokens(psychology_context)
        
        psychology_context_length = len(psychology_context)
        total_input_length += psychology_context_length
        
        psychology_result = self.run_agent(
            self.shark_psychology_expert, 
            "Shark Psychology Expert", 
            psychology_context
        )
        
        psychology_response = psychology_result["response"]
        psychology_content = psychology_result["content"]
        psychology_content = strip_thinking_tokens(psychology_content)
        
        psychology_response_length = len(psychology_content)
        total_output_length += psychology_response_length
        conversation_history.append({"role": "Shark Psychology Expert", "content": psychology_content})
        
        drafter_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        drafter_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        drafter_context += "\n\n=== Market Research Specialist ===\n\n" + market_content
        drafter_context += "\n\n=== Product Technical Advisor ===\n\n" + product_content
        drafter_context += "\n\n=== Shark Psychology Expert ===\n\n" + psychology_content
        drafter_context = strip_thinking_tokens(drafter_context)
        
        drafter_context_length = len(drafter_context)
        total_input_length += drafter_context_length
        
        drafter_result = self.run_agent(
            self.pitch_drafter, 
            "Pitch Drafter", 
            drafter_context
        )
        
        drafter_response = drafter_result["response"]
        drafter_content = drafter_result["content"]
        drafter_content = strip_thinking_tokens(drafter_content)
        
        drafter_response_length = len(drafter_content)
        total_output_length += drafter_response_length
        conversation_history.append({"role": "Pitch Drafter", "content": drafter_content})
        
        if hasattr(drafter_response, 'content') and isinstance(drafter_response.content, SharkTankPitch):
            draft_pitch = drafter_response.content
            self.logger.info("Successfully retrieved structured pitch from Pitch Drafter")
            draft_json = draft_pitch.model_dump()
        else:
            self.logger.error("Could not retrieve structured pitch from Pitch Drafter response. Using a placeholder.")
            draft_json = {
                "Pitch": "ERROR: Could not parse pitch from Drafter response.",
                "Initial_Offer": {
                    "Valuation": "Unknown",
                    "Equity_Offered": "Unknown",
                    "Funding_Amount": "Unknown",
                    "Key_Terms": "Unknown"
                }
            }
        
        critic_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        critic_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        critic_context += "\n\n=== Market Research Specialist ===\n\n" + market_content
        critic_context += "\n\n=== Product Technical Advisor ===\n\n" + product_content
        critic_context += "\n\n=== Shark Psychology Expert ===\n\n" + psychology_content
        critic_context += "\n\n=== Pitch Drafter ===\n\n" + drafter_content
        critic_context = strip_thinking_tokens(critic_context)
        
        critic_context_length = len(critic_context)
        total_input_length += critic_context_length
        
        critic_result = self.run_agent(
            self.pitch_critic, 
            "Pitch Critic", 
            critic_context
        )
        
        critic_response = critic_result["response"]
        critic_content = critic_result["content"]
        critic_content = strip_thinking_tokens(critic_content)
        
        critic_response_length = len(critic_content)
        total_output_length += critic_response_length
        conversation_history.append({"role": "Pitch Critic", "content": critic_content})
        
        finalizer_context = base_context + "\n\n=== MEETING ROOM CONVERSATION SO FAR ===\n\n"
        finalizer_context += "\n\n=== Financial Strategist ===\n\n" + financial_content
        finalizer_context += "\n\n=== Market Research Specialist ===\n\n" + market_content
        finalizer_context += "\n\n=== Product Technical Advisor ===\n\n" + product_content
        finalizer_context += "\n\n=== Shark Psychology Expert ===\n\n" + psychology_content
        finalizer_context += "\n\n=== Pitch Drafter ===\n\n" + drafter_content
        finalizer_context += "\n\n=== Pitch Critic ===\n\n" + critic_content
        finalizer_context = strip_thinking_tokens(finalizer_context)
        
        finalizer_context_length = len(finalizer_context)
        total_input_length += finalizer_context_length
        
        finalizer_result = self.run_agent(
            self.pitch_finalizer, 
            "Pitch Finalizer", 
            finalizer_context
        )
        
        finalizer_response = finalizer_result["response"]
        finalizer_content = finalizer_result["content"]
        finalizer_content = strip_thinking_tokens(finalizer_content)
        
        finalizer_response_length = len(finalizer_content)
        total_output_length += finalizer_response_length
        conversation_history.append({"role": "Pitch Finalizer", "content": finalizer_content})
        
        if hasattr(finalizer_response, 'content') and isinstance(finalizer_response.content, SharkTankPitch):
            final_pitch = finalizer_response.content
            self.logger.info("Successfully retrieved structured pitch from Pitch Finalizer")
            final_json = final_pitch.model_dump()
        else:
            self.logger.error("Could not retrieve structured pitch from Pitch Finalizer response. Using draft pitch.")
            final_json = draft_json
        
        execution_time = time.time() - start_time
        
        output_pitch_length = len(json.dumps(final_json, indent=2))
        total_output_length += output_pitch_length
        
        self.logger.info(f"Output pitch length: {output_pitch_length}")
        self.logger.info(f"Total input prompt length: {total_input_length}")
        self.logger.info(f"Total output response length: {total_output_length}")
        self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        with open(os.path.join(self.output_dir, "final_pitch.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "total_execution_time_seconds": execution_time,
                "total_input_prompt_length": total_input_length,
                "total_output_response_length": total_output_length,
                "agent_input_lengths": {
                    "financial_strategist": financial_context_length,
                    "market_research_specialist": market_context_length,
                    "product_technical_advisor": product_context_length,
                    "shark_psychology_expert": psychology_context_length,
                    "pitch_drafter": drafter_context_length,
                    "pitch_critic": critic_context_length,
                    "pitch_finalizer": finalizer_context_length
                },
                "agent_output_lengths": {
                    "financial_strategist": len(financial_content),
                    "market_research_specialist": len(market_content),
                    "product_technical_advisor": len(product_content),
                    "shark_psychology_expert": len(psychology_content),
                    "pitch_drafter": len(drafter_content),
                    "pitch_critic": len(critic_content),
                    "pitch_finalizer": len(finalizer_content)
                }
            }, f, indent=2)
        
        return final_json

def load_data(file_path: str) -> Any:
    """load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return {}

def setup_arg_parser():
    """set up and return the argument parser for command-line arguments"""
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches using a multi-agent system.')
    
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                        help='Model ID to use for all agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--product-key', type=str, default='facts_shark_tank_transcript_0_GarmaGuard.txt',
                        help='Product key/filename to process from the facts data (default: facts_shark_tank_transcript_0_GarmaGuard.txt)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug',
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    return parser

def main():
    """run the SharkTank pitch generator"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/sequential_pitch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_name = f"full_log_sequential_pitch_{timestamp}.log"
    log_file_path = os.path.join(output_dir, log_file_name)
    
    # clean up existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger(f"pitch_generator_{timestamp}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(log_file_path, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(log_file_path, 'a', 'utf-8', original_stderr)
    
    logger.info(f"Starting SharkTank pitch generator with model: {args.model}, product key: {args.product_key}, max_tokens: {args.max_tokens}, temperature: {args.temperature}")
    logger.info(f"Debug mode: {'enabled' if args.debug else 'disabled'}")
    
    facts_file_path = "./data/all_processed_facts.json"
    if not os.path.exists(facts_file_path):
        logger.error(f"Facts file not found at {facts_file_path}")
        return
    
    all_facts = load_data(facts_file_path)
    if not all_facts:
        logger.error("Failed to load facts data")
        return
    
    if args.product_key not in all_facts:
        available_keys = list(all_facts.keys())
        logger.error(f"Product key '{args.product_key}' not found in facts data.")
        logger.info(f"Available keys: {available_keys[:5]}{'...' if len(available_keys) > 5 else ''}")
        return
    
    episode_key = args.product_key
    episode_data = all_facts[episode_key]
    
    facts = str(episode_data.get('facts', "")) if isinstance(episode_data, dict) else str(episode_data)
    product_description = str(episode_data.get('product_description', "")) if isinstance(episode_data, dict) else str(episode_data)
    
    logger.info(f"Processing data for product file: {episode_key}")
    
    output_dir = f"./outputs/sequential_pitch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # save input data for later reference
    with open(os.path.join(output_dir, "input_data.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episode_key": episode_key,
            "facts": facts,
            "product_description": product_description,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature
        }, f, indent=2)
    
    team = SharkTankPitchTeam(
        output_dir=output_dir, 
        log_file_path=log_file_path, 
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    
    try:
        final_pitch = team.generate_pitch(facts, product_description, episode_key)
        logger.info(f"Final pitch generated and saved to {output_dir}/final_pitch.json")
    except Exception as e:
        logger.error(f"Error generating pitch: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return

if __name__ == "__main__":
    main() 