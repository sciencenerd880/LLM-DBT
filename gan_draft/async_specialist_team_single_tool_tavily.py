import os
import time
import json
import re
import datetime
import logging
import sys
import argparse
import asyncio
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
import io
from contextlib import redirect_stdout, redirect_stderr
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.tavily import TavilyTools

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
    if os.path.exists(llm_dbt_env_path):
        load_dotenv(llm_dbt_env_path)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
    if os.path.exists(llm_dbt_env_path):
        load_dotenv(llm_dbt_env_path)
        TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set GROQ_API_KEY in your environment or in the .env file.")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY not found in environment variables. Please set TAVILY_API_KEY in your environment or in the .env file.")

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
    
    think_pattern = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
    text = think_pattern.sub('', text).strip() 
    
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

class AsyncSpecialistTeamTavily:
    """coordinates a multi-agent team for SharkTank pitch generation using an asynchronous workflow with Tavily search tools"""
    
    def __init__(self, output_dir, log_file_path, model_id="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.7, debug_mode=True):
        """set up the team with output directory, logging, and model parameters"""
        self.logger = logging.getLogger(f"async_team_tavily_{os.path.basename(output_dir)}")
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
        
        if not TAVILY_API_KEY:
            self.logger.error("TAVILY_API_KEY not found. Please set TAVILY_API_KEY in your environment variables.")
            raise ValueError("TAVILY_API_KEY not found in environment variables. Please set TAVILY_API_KEY in your environment or in the .env file.")
        
        self.model_id = model_id
        self.debug_mode = debug_mode
        self.logger.info(f"Using model: {self.model_id}")
        self.logger.info(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'}")
        
        self.log_file_path = log_file_path
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== ASYNC SPECIALIST TEAM WITH TAVILY (Model: {self.model_id}) ===\n\n")
        
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.logger.info(f"Using max_tokens: {self.max_tokens}, temperature: {self.temperature}")
        
        # Initialize Tavily search tool
        self.tavily_tools = TavilyTools(api_key=TAVILY_API_KEY, search_depth="advanced")
        self.logger.info("Tavily search tools initialized")
        
        def create_groq_model():
            return Groq(
                id=self.model_id,
                api_key=GROQ_API_KEY,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        # Tool instructions for all specialist agents
        tool_usage_instructions = """
        IMPORTANT TOOL USAGE INSTRUCTIONS:
        You must use the Tavily Search tool EXACTLY ONCE during your analysis - no more, no less.
        
        When you use the search tool, follow this pattern:
        
        I'll use the Tavily Search tool to find information about [topic].
        
        DO NOT use any special formatting or syntax when referring to the tool.
        
        The search results will provide you with up-to-date, factual information that will enhance your analysis.
        Choose your search query carefully to get the most relevant information for your analysis.
        """
        
        # set up the team of specialist agents with original prompts but with tool instructions first
        self.financial_strategist = Agent(
            name="Financial Strategist",
            model=create_groq_model(),
            description="You are a Financial Strategist with expertise in startup valuation, investment analysis, and business model evaluation.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the financial aspects of a product for a SharkTank pitch.",
                "Always respond in English only.",
                "Use Tavily Search EXACTLY ONCE to research additional financial information such as:",
                "- Industry standard valuations for similar products",
                "- Recent funding trends in the product's sector", 
                "- Comparable companies and their financial metrics",
                "- Typical investment terms for similar businesses",
                "Choose your ONE search query carefully to get the most valuable financial information.",
                "Analyze the provided facts and product description to develop a financial strategy for a SharkTank pitch, including:",
                "1. A justified valuation for the company",
                "2. An appropriate investment amount to request",
                "3. A fair equity percentage to offer",
                "4. A breakdown of how the funds will be used",
                "5. A realistic ROI timeline",
                "6. Potential exit strategies",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not invent or assume financial data that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your recommendations."
            ],
            tools=[self.tavily_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.market_research_specialist = Agent(
            name="Market Research Specialist",
            model=create_groq_model(),
            description="You are a Market Research Specialist with deep knowledge of consumer trends, market analysis, and competitive landscapes.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the market aspects of a product for a SharkTank pitch.",
                "Always respond in English only.",
                "Use Tavily Search EXACTLY ONCE to research additional market information such as:",
                "- Current market size and growth projections",
                "- Emerging trends in the product's industry",
                "- Competitors and their market positions",
                "- Consumer behavior and preferences",
                "Choose your ONE search query carefully to get the most valuable market information.",
                "Analyze the provided facts and product description to develop market insights for a SharkTank pitch, including:",
                "1. The estimated size of the target market",
                "2. Description of target customer segments",
                "3. Analysis of competitors and their strengths/weaknesses",
                "4. Relevant market trends",
                "5. Potential growth opportunities",
                "6. Challenges in the market",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not make up market sizes or competitor information that contradicts what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            tools=[self.tavily_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.product_technical_advisor = Agent(
            name="Product Technical Advisor",
            model=create_groq_model(),
            description="You are a Product/Technical Advisor with expertise in product development, technical feasibility, and innovation assessment.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating the product/technical aspects for a SharkTank pitch.",
                "Always respond in English only.",
                "Use Tavily Search EXACTLY ONCE to research additional technical information such as:",
                "- Technical specifications for similar products",
                "- Manufacturing or production processes",
                "- Materials or components",
                "- Intellectual property considerations",
                "- Emerging technologies in the field",
                "Choose your ONE search query carefully to get the most valuable technical information.",
                "Analyze the provided facts and product description to develop product insights for a SharkTank pitch, including:",
                "1. Key product features to highlight",
                "2. Technical advantages over competitors",
                "3. How to effectively demonstrate the product",
                "4. Assessment of production/technical scalability",
                "5. Potential future product developments",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Do not invent capabilities or exaggerate performance in ways that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your assessments."
            ],
            tools=[self.tavily_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        self.shark_psychology_expert = Agent(
            name="Shark Psychology Expert",
            model=create_groq_model(),
            description="You are a Shark Psychology Expert who understands the motivations, preferences, and decision patterns of SharkTank investors.",
            instructions=[
                tool_usage_instructions,
                "You are a specialist tasked with evaluating investor psychology for a SharkTank pitch.",
                "Always respond in English only.",
                "Use Tavily Search EXACTLY ONCE to research additional information such as:",
                "- Recent investments made by Sharks",
                "- Investment preferences of individual Sharks",
                "- Successful negotiation tactics used on the show",
                "- Common reasons for Shark rejections",
                "Choose your ONE search query carefully to get the most valuable investor psychology information.",
                "Analyze the provided facts and product description to develop investor psychology insights for a SharkTank pitch, including:",
                "1. Points that will appeal to Sharks",
                "2. Potential objections and how to counter them",
                "3. Strategy for negotiating with Sharks",
                "4. Tips for effective presentation",
                "5. Sharks that might be the best fit and why",
                "Base your analysis primarily on the facts provided, but supplement with your single search result.",
                "Focus on general Shark psychology and preferences rather than making specific predictions that contradict what is explicitly stated.",
                "Be detailed, thorough, and well-organized in your response. Provide clear reasoning for your insights."
            ],
            tools=[self.tavily_tools],
            show_tool_calls=True,
            debug_mode=self.debug_mode
        )
        
        # Pitch Drafter - without search tool
        self.pitch_drafter = Agent(
            name="Pitch Drafter",
            model=create_groq_model(),
            description="You are a skilled pitch writer for entrepreneurs appearing on Shark Tank.",
            instructions=[
                "You are responsible for drafting a SharkTank pitch based on specialist analyses.",
                "Always respond in English only.",
                "Your task is to create a compelling pitch based on the specialist analyses provided. You will receive:",
                "1. The original product facts and description",
                "2. Financial analysis from the Financial Strategist",
                "3. Market research analysis from the Market Research Specialist",
                "4. Product/technical analysis from the Product Technical Advisor",
                "5. Shark psychology insights from the Shark Psychology Expert",
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
                "You are responsible for critiquing a SharkTank pitch draft.",
                "Always respond in English only.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. The draft pitch in JSON format",
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
                "You are responsible for finalizing a SharkTank pitch based on critic feedback.",
                "Always respond in English only.",
                "Your task is to refine and finalize the draft pitch based on the critic's feedback.",
                "You will be provided with:",
                "1. The original product facts and description",
                "2. The draft pitch in JSON format",
                "3. The pitch critic's feedback",
                "Create a polished, compelling final pitch that incorporates the strengths identified by the critic while addressing the areas for improvement. The final pitch should be concise, engaging, and strategically structured to maximize appeal to the Sharks."
            ],
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        self.logger.info("Async Specialist Team with Tavily search tools initialized") 

    def log_message(self, agent_name, message):
        """log a message from an agent to the consolidated log file"""
        clean_message = strip_ansi_codes(message)
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {agent_name} ===\n\n")
            f.write(clean_message)
    
    async def run_agent_async(self, agent, agent_name, context):
        """run an agent with given context and handle its response asynchronously"""
        self.logger.info(f"{agent_name} analyzing...")
        
        try:
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                # Use a separate thread for agent.run which can be blocking
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, lambda: agent.run(context))
                
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
                
                # track tool usage
                tool_calls = []
                tool_attempts = 0
                
                # check standard locations for tool calls
                if hasattr(response, 'tool_calls') and response.tool_calls:
                    tool_calls = response.tool_calls
                elif hasattr(response, 'metrics') and hasattr(response.metrics, 'tool_calls') and response.metrics.tool_calls:
                    tool_calls = response.metrics.tool_calls
                
                # count tool attempts from debug output
                if debug_output:
                    tool_attempts_in_debug = debug_output.count("Tool call Id:")
                    if tool_attempts_in_debug > 0:
                        tool_attempts = tool_attempts_in_debug
                        self.logger.info(f"{agent_name} attempted {tool_attempts} tool calls (from debug output)")
                
                if len(tool_calls) > 0:
                    self.logger.info(f"{agent_name} made {len(tool_calls)} successful tool calls")
                    
                    if len(tool_calls) > 1:
                        self.logger.warning(f"{agent_name} made {len(tool_calls)} tool calls - more than the required exactly one call")
                    
                    for i, tool_call in enumerate(tool_calls):
                        tool_name = getattr(tool_call, 'name', 'unknown')
                        tool_input = getattr(tool_call, 'input', {})
                        tool_output = getattr(tool_call, 'output', {})
                        
                        # log tool calls to main log
                        self.log_message(
                            f"{agent_name} Tool Call #{i+1}",
                            f"Tool: {tool_name}\nQuery: {tool_input}\nResults: {tool_output}"
                        )
                else:
                    # warn if no tool call made
                    self.logger.warning(f"{agent_name} did not make any tool calls - should have made exactly one call")
                
                # look for tool usage mentions in content
                mentions = 0
                if "tavily" in content_for_conversation.lower() or "search tool" in content_for_conversation.lower():
                    mentions = (
                        content_for_conversation.lower().count("tavily") + 
                        content_for_conversation.lower().count("search tool") +
                        content_for_conversation.lower().count("i'll use the") +
                        content_for_conversation.lower().count("searching for")
                    )
                    
                    if mentions > 0 and tool_attempts == 0:
                        self.logger.info(f"{agent_name} mentioned search tools {mentions} times but no tool calls were tracked")
                        tool_attempts = max(1, mentions // 2)  # estimate attempts from mentions
                
                # use max count between tracked calls, debug output and mentions
                actual_tool_calls = max(len(tool_calls) if tool_calls else 0, tool_attempts)
                
                return {
                    "response": response, 
                    "content": content_for_conversation,
                    "tool_calls": tool_calls if tool_calls else [],
                    "tool_attempts": actual_tool_calls
                }
            else:
                self.logger.warning(f"{agent_name} returned empty response")
                error_message = f"No response from {agent_name}"
                self.log_message(agent_name, error_message)
                return {"response": error_message, "content": error_message, "tool_calls": [], "tool_attempts": 0}
        except Exception as e:
            self.logger.error(f"Error getting {agent_name} response: {e}")
            error_message = f"No response from {agent_name} due to error: {str(e)}"
            self.log_message(agent_name, error_message)
            return {"response": error_message, "content": error_message, "tool_calls": [], "tool_attempts": 0}
    
    async def generate_pitch(self, facts, product_description, product_key):
        """generate a pitch for a SharkTank product using asynchronous specialists with Tavily search"""
        facts_str = str(facts) if not isinstance(facts, str) else facts
        product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
        
        self.logger.info(f"Generating pitch for: {product_key}")
        
        total_input_length = len(product_description_str) + len(facts_str)
        total_output_length = 0
        
        self.logger.info(f"Initial input prompt length: {total_input_length}")
        
        # Base context with NO TOOL INSTRUCTIONS (just facts and product description)
        base_context = f"""
        FACTS:
        {facts_str}
        
        PRODUCT DESCRIPTION:
        {product_description_str}
        """
        
        specialist_context = base_context + """
        INSTRUCTIONS:
        Analyze the provided facts and product description from your specialist perspective.
        Use the Tavily Search tool to find additional information that will enhance your analysis.
        """
        
        specialist_results = {}
        
        # track all debug outputs
        all_debug_outputs = ""
        
        start_time = time.time()
        
        # launch specialists concurrently
        self.logger.info("Starting all specialist agents concurrently with Tavily search enabled...")
        
        # create tasks for each specialist using specialist_context
        tasks = {
            "Financial Strategist": self.run_agent_async(
                self.financial_strategist, 
                "Financial Strategist", 
                specialist_context
            ),
            "Market Research Specialist": self.run_agent_async(
                self.market_research_specialist, 
                "Market Research Specialist", 
                specialist_context
            ),
            "Product Technical Advisor": self.run_agent_async(
                self.product_technical_advisor, 
                "Product Technical Advisor", 
                specialist_context
            ),
            "Shark Psychology Expert": self.run_agent_async(
                self.shark_psychology_expert, 
                "Shark Psychology Expert", 
                specialist_context
            )
        }
        
        # track progress
        specialist_progress = {name: "In progress" for name in tasks.keys()}
        specialist_contents = {}
        tool_calls_count = 0
        
        self.logger.info("Waiting for all specialists to complete concurrently...")
        
        # run tasks in parallel
        task_objs = list(tasks.values())
        task_names = list(tasks.keys())
        
        results = await asyncio.gather(*task_objs, return_exceptions=True)
        
        # process results
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                specialist_progress[name] = f"Error: {str(result)}"
                self.logger.error(f"{name}: Error - {str(result)}")
                continue
                
            specialist_results[name] = result
            
            if "content" in result:
                content = strip_thinking_tokens(result["content"])
                specialist_contents[name] = content
                specialist_progress[name] = "Completed"
                self.logger.info(f"{name}: Completed (Length: {len(content)})")
                total_output_length += len(content)
            else:
                specialist_progress[name] = "Error: No content"
                self.logger.warning(f"{name}: Error - No content")
            
            # count tool usage
            if "tool_attempts" in result:
                agent_tool_calls = result["tool_attempts"]
                tool_calls_count += agent_tool_calls
                self.logger.info(f"{name} made {agent_tool_calls} tool attempts")
            
            # collect debug info
            if "response" in result and hasattr(result["response"], "debug_output"):
                debug_output = result["response"].debug_output
                all_debug_outputs += debug_output
        
        self.logger.info("Specialist agents completion status:")
        for name, status in specialist_progress.items():
            self.logger.info(f"  {name}: {status}")
        
        self.logger.info(f"Total tool calls/attempts by specialists: {tool_calls_count}")
        
        # prep context for drafter using base_context without tool instructions
        drafter_context = base_context + "\n\n=== Suggestions from specialists ===\n\n"
        for name, content in specialist_contents.items():
            drafter_context += f"\n\n=== {name} ===\n\n{content}"
        
        drafter_context = strip_thinking_tokens(drafter_context)
        drafter_context_length = len(drafter_context)
        total_input_length += drafter_context_length
        
        self.logger.info("All specialists completed. Starting Pitch Drafter...")
        drafter_result = await self.run_agent_async(
            self.pitch_drafter, 
            "Pitch Drafter", 
            drafter_context
        )
        
        drafter_response = drafter_result["response"]
        drafter_content = drafter_result["content"]
        drafter_content = strip_thinking_tokens(drafter_content)
        
        drafter_response_length = len(drafter_content)
        total_output_length += drafter_response_length
        
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
        
        # Prepare context for Pitch Critic - only includes base context and draft pitch
        critic_context = base_context + f"\n\n=== Draft Pitch ===\n\n{drafter_content}"
        critic_context = strip_thinking_tokens(critic_context)
        
        critic_context_length = len(critic_context)
        total_input_length += critic_context_length
        
        self.logger.info("Starting Pitch Critic...")
        critic_result = await self.run_agent_async(
            self.pitch_critic, 
            "Pitch Critic", 
            critic_context
        )
        
        critic_response = critic_result["response"]
        critic_content = critic_result["content"]
        critic_content = strip_thinking_tokens(critic_content)
        
        critic_response_length = len(critic_content)
        total_output_length += critic_response_length
        
        # Prepare context for Pitch Finalizer - only includes base context, draft pitch, and critic feedback
        finalizer_context = base_context + f"\n\n=== Draft Pitch ===\n\n{drafter_content}"
        finalizer_context += f"\n\n=== Pitch Critic Feedback ===\n\n{critic_content}"
        finalizer_context = strip_thinking_tokens(finalizer_context)
        
        finalizer_context_length = len(finalizer_context)
        total_input_length += finalizer_context_length
        
        self.logger.info("Starting Pitch Finalizer...")
        finalizer_result = await self.run_agent_async(
            self.pitch_finalizer, 
            "Pitch Finalizer", 
            finalizer_context
        )
        
        finalizer_response = finalizer_result["response"]
        finalizer_content = finalizer_result["content"]
        finalizer_content = strip_thinking_tokens(finalizer_content)
        
        finalizer_response_length = len(finalizer_content)
        total_output_length += finalizer_response_length
        
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
        self.logger.info(f"Total tool calls/attempts: {tool_calls_count}")
        
        # Save final pitch and metrics to output files
        with open(os.path.join(self.output_dir, "final_pitch.json"), "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=2)
        
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({
                "total_execution_time_seconds": execution_time,
                "total_input_prompt_length": total_input_length,
                "total_output_response_length": total_output_length,
                "total_tool_calls": tool_calls_count,
                "specialist_status": specialist_progress,
                "agent_input_lengths": {
                    "specialists": len(specialist_context),
                    "pitch_drafter": drafter_context_length,
                    "pitch_critic": critic_context_length,
                    "pitch_finalizer": finalizer_context_length
                },
                "agent_output_lengths": {
                    **{name: len(content) for name, content in specialist_contents.items()},
                    "pitch_drafter": len(drafter_content),
                    "pitch_critic": len(critic_content),
                    "pitch_finalizer": len(finalizer_content)
                },
                "tool_usage": {
                    "enabled_for_specialists": True,
                    "tool_calls_count": tool_calls_count
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
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches using an asynchronous multi-agent system with Tavily search tools.')
    
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

async def main_async():
    """run the SharkTank pitch generator asynchronously with Tavily search tools"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/async_specialist_team_tavily_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_name = f"full_log_async_specialist_team_tavily_{timestamp}.log"
    log_file_path = os.path.join(output_dir, log_file_name)
    
    # clean up existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger(f"async_pitch_generator_tavily_{timestamp}")
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
    
    logger.info(f"Starting Async Specialist Team with Tavily search tools, model: {args.model}, product key: {args.product_key}, max_tokens: {args.max_tokens}, temperature: {args.temperature}")
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
    
    team = AsyncSpecialistTeamTavily(
        output_dir=output_dir, 
        log_file_path=log_file_path, 
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug
    )
    
    try:
        final_pitch = await team.generate_pitch(facts, product_description, episode_key)
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

def main():
    """entry point that calls the async main function"""
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main_async())

if __name__ == "__main__":
    main() 