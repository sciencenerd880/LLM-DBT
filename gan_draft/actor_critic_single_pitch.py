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

# try to load env vars from current directory first
load_dotenv()

# check for groq api key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
    if os.path.exists(llm_dbt_env_path):
        load_dotenv(llm_dbt_env_path)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# make sure we have the api key
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set GROQ_API_KEY in your environment or in the .env file.")

def strip_ansi_codes(text):
    """remove ANSI escape sequences to make logs more readable"""
    if not isinstance(text, str):
        return text
        
    # pattern to match ANSI escape codes
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def strip_thinking_tokens(text):
    """
    remove thinking tokens and return cleaned text plus whether tokens were found
    
    handles:
    - nested tokens
    - malformed tokens 
    - partial tokens
    - multiple token pairs
    """
    if not isinstance(text, str):
        return text, False
    
    # check if thinking tokens exist
    has_thinking = '<think>' in text and '</think>' in text
    
    if not has_thinking:
        return text, False
        
    try:
        # handle potential nesting by processing from inside out
        pattern = r'<think>(.*?)</think>'
        
        # keep replacing until no more replacements are made
        prev_text = None
        cleaned_text = text
        while prev_text != cleaned_text:
            prev_text = cleaned_text
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL)
        
        # handle any remaining unclosed thinking tags
        cleaned_text = re.sub(r'<think>.*?($|\n\n)', '', cleaned_text, flags=re.DOTALL)
        cleaned_text = re.sub(r'<think>', '', cleaned_text)
        cleaned_text = re.sub(r'</think>', '', cleaned_text)
        
        # remove any extra blank lines that might have been created
        cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
        
        # if we've removed everything, return original
        if cleaned_text.strip() == "":
            return text, True
        
        return cleaned_text, True
    except Exception as e:
        # if error occurs, return original but log the issue
        logging.warning(f"Error stripping thinking tokens: {str(e)}")
        return text, has_thinking

class TeeLogger:
    """duplicates output to both a file and the original stream"""
    def __init__(self, filename, mode='w', encoding='utf-8', stream=None):
        self.file = open(filename, mode, encoding=encoding)
        self.stream = stream if stream else sys.stdout
        self.encoding = encoding
        
    def write(self, data):
        # clean ANSI codes before writing to file
        clean_data = strip_ansi_codes(data)
        self.file.write(clean_data)
        self.file.flush()
        self.stream.write(data)  # keep original formatting for terminal
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

class ActorCriticPitchTeam:
    """
    implements an actor-critic approach for SharkTank pitch generation using two agents:
    a Pitch Drafter and a Pitch Critic in an iterative loop
    """
    
    def __init__(self, output_dir, log_file_path, model_id="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.7, debug_mode=True, max_iterations=5):
        """
        initialize the Actor-Critic Pitch Team
        
        args:
            output_dir: directory to save outputs
            log_file_path: path to log file
            model_id: model ID to use (default: llama-3.3-70b-versatile)
            max_tokens: max tokens for model responses (default: 4096)
            temperature: temperature for model responses (0.0-1.0, default: 0.7)
            debug_mode: enable debug mode for agents (default: True)
            max_iterations: max iterations between drafter and critic (default: 5)
        """
        # set up logging with unique name to avoid duplication
        self.logger = logging.getLogger(f"actor_critic_team_{os.path.basename(output_dir)}")
        self.logger.setLevel(logging.INFO)
        
        # clear existing handlers to prevent duplicate logging
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        self.logger.propagate = False  # prevent propagation to avoid duplicate logs
        
        # create handler for this specific logger
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(console_handler)
        
        # create output dir if needed
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # store config
        self.model_id = model_id
        self.debug_mode = debug_mode
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_iterations = max_iterations
        self.log_file_path = log_file_path
        
        self.logger.info(f"Using model: {self.model_id}")
        self.logger.info(f"Debug mode: {'enabled' if self.debug_mode else 'disabled'}")
        self.logger.info(f"Max iterations: {self.max_iterations}")
        
        # create single log file
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"=== ACTOR-CRITIC SHARK TANK PITCH GENERATION LOG (Model: {self.model_id}) ===\n\n")
        
        # configure groq model
        self.logger.info(f"Using max_tokens: {self.max_tokens}, temperature: {self.temperature}")
        
        def create_groq_model():
            return Groq(
                id=self.model_id,
                api_key=GROQ_API_KEY,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
        
        # create pitch drafter agent
        self.pitch_drafter = Agent(
            name="Pitch Drafter",
            model=create_groq_model(),
            description="You are a skilled pitch writer for entrepreneurs appearing on Shark Tank, working as part of a two-agent team.",
            instructions=[
                "You are the Pitch Drafter in a two-agent team working together to create a compelling Shark Tank pitch.",
                "Your partner is the Pitch Critic who will evaluate your drafts and provide feedback.",
                "Always respond in English only.",
                "Your task is to create a compelling pitch based on the provided facts and product description.",
                "The pitch should be structured to grab attention, clearly explain the product/service, highlight market potential, showcase competitive advantages, present financial data, make a specific investment ask, and close with a strong call to action.",
                "If you receive feedback from the Pitch Critic, analyze it carefully and improve your pitch accordingly.",
                "Focus on addressing the weaknesses and improvement areas highlighted by the critic while maintaining the identified strengths.",
                "You MUST output your response in the following JSON format, with no additional text before or after:",
                '''
                ###Example output format:
                {
                  "Pitch": "...",
                  "Initial_Offer": {
                    "Valuation": "$X million",
                    "Equity_Offered": "X%",
                    "Funding_Amount": "$X",
                    "Key_Terms": "Full distribution rights remain with the company"
                  }
                }
                '''
            ],
            add_history_to_messages=True,  # enable memory to access conversation history
            num_history_responses=5,  # number of prior exchanges to remember
            response_model=SharkTankPitch,
            debug_mode=self.debug_mode
        )
        
        # create pitch critic agent
        self.pitch_critic = Agent(
            name="Pitch Critic",
            model=create_groq_model(),
            description="You are a Pitch Critic who identifies strengths, weaknesses, and areas for improvement in SharkTank pitches, working as part of a two-agent team.",
            instructions=[
                "You are the Pitch Critic in a two-agent team working together to create a compelling Shark Tank pitch.",
                "Your partner is the Pitch Drafter who creates drafts for you to evaluate.",
                "Always respond in English only.",
                "Your task is to analyze the draft pitch and initial offer provided by the Pitch Drafter, then either provide feedback OR approve it as final.",
                "IMPORTANT: You must verify that the pitch accurately represents the facts provided. Any pitch containing inaccurate information, fabricated details, or claims not supported by the facts should receive a low score.",
                "Score the pitch from 0 to 10 using these strict guidelines:",
                "- 0-2: Poor pitch with major factual inaccuracies, missing key information, or unrealistic valuation",
                "- 3-4: Below average pitch with some factual inaccuracies or significant gaps in reasoning",
                "- 5-6: Average pitch with minor factual issues, needs significant improvement in structure or offer terms",
                "- 7-8: Good pitch with no factual errors, clear value proposition, and reasonable offer, but needs polishing",
                "- 9-10: Excellent to perfect pitch with perfect factual accuracy, compelling storytelling, and optimal offer structure",
                "Be extremely critical - most first drafts should score in the 3-6 range. A score of 9-10 should be rare and only for excellent pitches.",
                "Your response format depends on the score:",
                "1. If score < 9, return feedback JSON with: 'Score', 'Decision', 'Strengths', 'Weaknesses', 'Improvement_Areas', and 'Suggestions'",
                "2. If score >= 9, return the final pitch JSON with: 'Score', 'Decision', 'Final_Pitch', and 'Final_Initial_Offer'",
                "CRITICAL: When you mark a decision as 'Final' (with score >= 9), you MUST include:",
                "  - The 'Final_Pitch' field with the complete final pitch text",
                "  - The 'Final_Initial_Offer' field with all offer details",
                "CRITICAL: Your response MUST be a valid JSON object. Ensure all JSON properties are properly separated with commas, all strings are properly escaped, and the structure is valid.",
                "CRITICAL: Do NOT include any text, explanations, or commentary before or after the JSON object.",
                "CRITICAL: Ensure all property names and values in your JSON response are enclosed in double quotes.",
                "You MUST output ONLY JSON with no additional text before or after, in one of the following formats based on your scoring decision:",
                '''

                ###Example output format:

                For score < 9 (needs improvement):
                {
                  "Score": X,
                  "Decision": "Feedback",
                  "Strengths": [
                    "Clear explanation of the product functionality",
                    "Strong market size information",
                    "Compelling personal story"
                  ],
                  "Weaknesses": [
                    "Valuation seems too high for current traction",
                    "Equity offered is too low given the early stage",
                    "Not enough emphasis on competitive advantage"
                  ],
                  "Improvement_Areas": [
                    "Financial projections need more detail",
                    "Call to action could be more compelling",
                    "Need better explanation of customer acquisition strategy"
                  ],
                  "Suggestions": [
                    "Lower the valuation to $800K to appear more reasonable",
                    "Increase equity offer to 15% to make it more attractive",
                    "Add specific examples of customer success stories",
                    "Clarify the use of funds with specific allocation percentages"
                  ]
                }
                
                For score >= 9 (ready for presentation):
                {
                  "Score": X,
                  "Decision": "Final",
                  "Final_Pitch": "Hello, Sharks. I'm [name] from [company]...[FULL PITCH TEXT]",
                  "Final_Initial_Offer": {
                    "Valuation": "$X million",
                    "Equity_Offered": "X%",
                    "Funding_Amount": "$X",
                    "Key_Terms": "Full distribution rights remain with the company"
                  }
                }
                '''
            ],
            add_history_to_messages=True,  # enable memory to access conversation history
            num_history_responses=5,  # number of prior exchanges to remember
            debug_mode=self.debug_mode
        )
        
        self.logger.info("Actor-Critic Pitch Team initialized")
    
    def log_message(self, agent_name, message):
        """
        log a message to the consolidated log file
        
        args:
            agent_name: name of the agent sending the message
            message: message content
        """
        # clean message by removing ANSI escape sequences
        clean_message = strip_ansi_codes(message)
        
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n=== {agent_name} ===\n\n")
            f.write(clean_message)
    
    def run_agent(self, agent, agent_name, context):
        """
        run an agent with the given context and handle its response
        
        args:
            agent: agent to run
            agent_name: name of the agent
            context: context to provide to the agent
            
        returns:
            dict: response object, original content string, and filtered content string
        """
        self.logger.info(f"{agent_name} processing...")
        
        try:
            # capture stdout/stderr during agent run
            f = io.StringIO()
            with redirect_stdout(f), redirect_stderr(f):
                # use agno's built-in debugger
                response = agent.run(context)
                
            # get debug output
            debug_output = f.getvalue()
            
            # clean debug output
            debug_output = strip_ansi_codes(debug_output)
            
            # log debug output
            if debug_output:
                self.log_message(f"{agent_name} Debug Output", debug_output)
            
            if response:
                # extract and clean content
                if hasattr(response, 'content'):
                    if hasattr(response.content, 'model_dump'):
                        # pydantic model - convert to JSON string
                        content_original = json.dumps(response.content.model_dump(), indent=2)
                        content_filtered = content_original
                        has_thinking = False
                    else:
                        # string or other object - might have thinking tokens
                        content_original = strip_ansi_codes(str(response.content))
                        content_filtered, has_thinking = strip_thinking_tokens(content_original)
                else:
                    content_original = strip_ansi_codes(str(response))
                    content_filtered, has_thinking = strip_thinking_tokens(content_original)
                
                # log response length
                response_length = len(content_original)
                self.logger.info(f"{agent_name} response length: {response_length}")
                
                # log if thinking tokens found
                if has_thinking:
                    original_length = len(content_original)
                    filtered_length = len(content_filtered)
                    tokens_removed = original_length - filtered_length
                    self.logger.info(f"{agent_name} response includes thinking tokens ({tokens_removed} chars removed) which will be preserved in logs but filtered from context")
                    
                    # log more details in debug mode
                    if self.debug_mode:
                        thinking_token_pairs = content_original.count('<think>')
                        self.logger.debug(f"{agent_name} response contains {thinking_token_pairs} thinking token pairs")
                        
                        # check for mismatched tokens
                        if content_original.count('<think>') != content_original.count('</think>'):
                            self.logger.warning(f"{agent_name} response has mismatched thinking tokens: {content_original.count('<think>')} opening tags, {content_original.count('</think>')} closing tags")
                
                return {
                    "response": response, 
                    "content": content_original,  # original content with thinking tokens (for logs)
                    "filtered_content": content_filtered  # filtered content without thinking tokens (for context)
                }
            else:
                self.logger.warning(f"{agent_name} returned empty response")
                error_message = f"No response from {agent_name}"
                self.log_message(agent_name, error_message)
                return {"response": error_message, "content": error_message, "filtered_content": error_message}
        except Exception as e:
            self.logger.error(f"Error getting {agent_name} response: {e}")
            error_message = f"No response from {agent_name} due to error: {str(e)}"
            self.log_message(agent_name, error_message)
            return {"response": error_message, "content": error_message, "filtered_content": error_message}
    
    def generate_pitch(self, facts, product_description, product_key):
        """
        generate a pitch for a SharkTank product using the actor-critic approach
        
        args:
            facts: facts about the product
            product_description: description of the product
            product_key: key/filename for the product
            
        returns:
            dict: finalized pitch and metrics
        """
        # convert to strings if needed
        facts_str = str(facts) if not isinstance(facts, str) else facts
        product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
        
        # log using product key
        self.logger.info(f"Generating pitch for: {product_key}")
        
        # track input/output lengths
        total_input_length = len(product_description_str) + len(facts_str)
        total_output_length = 0
        
        # log input length
        self.logger.info(f"Initial input prompt length: {total_input_length}")
        
        # create context message
        base_context = f"""
        FACTS:
        {facts_str}
        
        PRODUCT DESCRIPTION:
        {product_description_str}
        """
        
        start_time = time.time()
        
        # track best pitch
        best_pitch = None
        best_score = -1
        all_iterations = []
        
        # iterate between drafter and critic
        for iteration in range(1, self.max_iterations + 1):
            self.logger.info(f"Starting iteration {iteration} of {self.max_iterations}")
            
            # prepare drafter context
            if iteration == 1:
                drafter_context = base_context
            else:
                drafter_context = getattr(self, 'next_drafter_context', "Please improve your pitch based on the feedback provided by the critic.")
            
            # run drafter
            drafter_context_length = len(drafter_context)
            total_input_length += drafter_context_length
            
            drafter_result = self.run_agent(
                self.pitch_drafter,
                f"Pitch Drafter (Iteration {iteration})",
                drafter_context
            )
            
            drafter_response = drafter_result["response"]
            drafter_content = drafter_result["content"]  # original content with thinking tokens
            
            # track response length
            drafter_response_length = len(drafter_content)
            total_output_length += drafter_response_length
            
            # extract draft pitch
            if hasattr(drafter_response, 'content') and isinstance(drafter_response.content, SharkTankPitch):
                draft_pitch = drafter_response.content
                self.logger.info("Successfully retrieved structured pitch from Pitch Drafter")
                draft_json = draft_pitch.model_dump()
            else:
                # try to parse as JSON if not a pydantic model
                try:
                    content = drafter_response.content if hasattr(drafter_response, 'content') else str(drafter_response)
                    
                    if isinstance(content, str):
                        # remove thinking tokens first
                        content, _ = strip_thinking_tokens(content)
                        
                        # check for JSON data
                        if '{' in content and '}' in content:
                            # extract JSON part
                            json_start = content.find('{')
                            json_end = content.rfind('}') + 1
                            json_str = content[json_start:json_end]
                            
                            # fix common JSON issues
                            json_str = re.sub(r'[\x00-\x1F\x7F]', '', json_str)  # remove control chars
                            
                            # try to parse as JSON
                            parsed_content = json.loads(json_str)
                            
                            # check structure
                            if 'Pitch' in parsed_content and 'Initial_Offer' in parsed_content:
                                draft_json = {
                                    "Pitch": parsed_content["Pitch"],
                                    "Initial_Offer": parsed_content["Initial_Offer"]
                                }
                                self.logger.info("Successfully extracted pitch from JSON in Drafter response")
                                draft_json = parsed_content
                                if not isinstance(draft_json["Initial_Offer"], dict):
                                    raise ValueError("Initial_Offer is not a dictionary")
                            else:
                                raise ValueError("JSON doesn't have required Pitch and Initial_Offer fields")
                        else:
                            raise ValueError("No JSON object found in response")
                    else:
                        raise ValueError("Content is not a string")
                except Exception as e:
                    self.logger.error(f"Could not retrieve structured pitch from Pitch Drafter response: {str(e)}. Using a placeholder.")
                    draft_json = {
                        "Pitch": "ERROR: Could not parse pitch from Drafter response.",
                        "Initial_Offer": {
                            "Valuation": "Unknown",
                            "Equity_Offered": "Unknown",
                            "Funding_Amount": "Unknown",
                            "Key_Terms": "Unknown"
                        }
                    }
            
            # prepare critic context
            if iteration == 1:
                critic_context = f"""
                {base_context}
                
                DRAFT PITCH TO EVALUATE:
                {draft_json["Pitch"]}
                
                INITIAL OFFER TO EVALUATE:
                Valuation: {draft_json["Initial_Offer"]["Valuation"]}
                Equity Offered: {draft_json["Initial_Offer"]["Equity_Offered"]}
                Funding Amount: {draft_json["Initial_Offer"]["Funding_Amount"]}
                Key Terms: {draft_json["Initial_Offer"]["Key_Terms"]}
                
                Please evaluate this specific draft pitch and initial offer provided by the Pitch Drafter.
                Score it from 0-10 using the strict guidelines in your instructions.
                Check carefully for factual accuracy against the FACTS and PRODUCT DESCRIPTION provided above.
                Remember to set the Decision field to 'Final' if score >= 9, or 'Feedback' if score < 9.
                
                IMPORTANT: If you mark the decision as 'Final' (score >= 9), you MUST include the "Final_Pitch" and "Final_Initial_Offer" fields in your JSON response.
                
                This is iteration {iteration} of maximum {self.max_iterations}.
                
                Respond ONLY with a JSON object and no other text.
                """
            else:
                critic_context = f"""
                {base_context}
                
                UPDATED DRAFT PITCH TO EVALUATE:
                {draft_json["Pitch"]}
                
                UPDATED INITIAL OFFER TO EVALUATE:
                Valuation: {draft_json["Initial_Offer"]["Valuation"]}
                Equity Offered: {draft_json["Initial_Offer"]["Equity_Offered"]}
                Funding Amount: {draft_json["Initial_Offer"]["Funding_Amount"]}
                Key Terms: {draft_json["Initial_Offer"]["Key_Terms"]}
                
                Please evaluate this specific updated draft pitch and initial offer from the Pitch Drafter.
                Score it from 0-10 using the strict guidelines in your instructions.
                Check carefully for factual accuracy against the FACTS and PRODUCT DESCRIPTION provided above.
                
                IMPORTANT: If you mark the decision as 'Final' (score >= 9), you MUST include the "Final_Pitch" and "Final_Initial_Offer" fields in your JSON response.
                
                This is iteration {iteration} of maximum {self.max_iterations}.
                
                If this is the final iteration (iteration {iteration} = max {self.max_iterations}), you may consider setting Decision to 'Final' even if score < 9, 
                since this will be the best version we can achieve within our iteration limit.
                
                Respond ONLY with a JSON object and no other text.
                """
            
            # run critic
            critic_context_length = len(critic_context)
            total_input_length += critic_context_length
            
            critic_result = self.run_agent(
                self.pitch_critic,
                f"Pitch Critic (Iteration {iteration})",
                critic_context
            )
            
            critic_response = critic_result["response"]
            critic_content = critic_result["content"]  # original content with thinking tokens
            critic_filtered_content = critic_result["filtered_content"]  # content without thinking tokens
            
            # track response length
            critic_response_length = len(critic_content)
            total_output_length += critic_response_length
            
            # extract critic feedback
            critic_json = None
            if hasattr(critic_response, 'content'):
                try:
                    # try to parse as JSON
                    content = critic_response.content
                    if isinstance(content, str):
                        # use filtered version without thinking tokens
                        content = critic_filtered_content
                        
                        # clean up content string
                        if '{' in content:
                            content = content[content.find('{'):]
                        if '}' in content:
                            content = content[:content.rfind('}')+1]
                        
                        # log cleaned content for debugging
                        self.logger.debug(f"Attempting to parse JSON from: {content[:100]}...")
                        
                        # fix common JSON format issues
                        content = re.sub(r'"\s*\n\s*"', '",\n"', content)  # fix missing commas
                        content = re.sub(r'[\x00-\x1F\x7F]', '', content)  # remove control chars
                        
                        try:
                            critic_json = json.loads(content)
                        except json.JSONDecodeError as json_err:
                            # try more lenient parsing
                            self.logger.warning(f"Standard JSON parsing failed: {json_err}. Attempting more lenient parsing.")
                            
                            # fix missing commas
                            fixed_content = re.sub(r'"\s*}\s*"', '"},\n"', content)
                            fixed_content = re.sub(r'"\s*]\s*"', '"],\n"', fixed_content)
                            
                            try:
                                critic_json = json.loads(fixed_content)
                            except json.JSONDecodeError:
                                # log full content when all parsing fails
                                self.logger.error(f"All JSON parsing attempts failed. Full content:\n{content}")
                                raise
                    elif hasattr(content, 'model_dump'):
                        critic_json = content.model_dump()
                    else:
                        critic_json = content
                except Exception as e:
                    self.logger.error(f"Could not parse critic response as JSON: {str(e)}")
                    if hasattr(critic_response, 'content') and isinstance(critic_response.content, str):
                        self.logger.error(f"Critic content: {critic_response.content[:500]}...")
            
            if critic_json and isinstance(critic_json, dict) and "Score" in critic_json and "Decision" in critic_json:
                self.logger.info(f"Critic Score: {critic_json['Score']}/10, Decision: {critic_json['Decision']}")
                feedback_json = critic_json
                
                # update best pitch if marked as final and has better score
                if feedback_json["Decision"] == "Final" and feedback_json["Score"] >= best_score:
                    if "Final_Pitch" in feedback_json and "Final_Initial_Offer" in feedback_json:
                        # create new pitch dict from critic's final decision
                        final_pitch_text = feedback_json["Final_Pitch"]
                        if isinstance(final_pitch_text, str):
                            final_pitch_text, _ = strip_thinking_tokens(final_pitch_text)
                            
                        # process final offer
                        final_offer = feedback_json["Final_Initial_Offer"]
                        if isinstance(final_offer, dict):
                            for key, value in final_offer.items():
                                if isinstance(value, str):
                                    final_offer[key], _ = strip_thinking_tokens(value)
                        
                        best_pitch = {
                            "Pitch": final_pitch_text,
                            "Initial_Offer": final_offer
                        }
                        best_score = feedback_json["Score"]
                        self.logger.info(f"Critic approved final pitch with score: {best_score}/10")
                    else:
                        # fall back to drafter's pitch
                        best_pitch = draft_json
                        best_score = feedback_json["Score"]
                        self.logger.info(f"Critic marked pitch as Final (score: {best_score}/10) but didn't include Final_Pitch fields; using drafter's pitch")
                        
                        # add debug info
                        if self.debug_mode:
                            self.logger.debug(f"Critic marked as Final but JSON keys present: {list(feedback_json.keys())}")
                # if not final but score is higher, update best pitch from drafter
                elif feedback_json["Score"] > best_score:
                    # remove thinking tokens from drafter's pitch
                    if isinstance(draft_json["Pitch"], str):
                        draft_json["Pitch"], _ = strip_thinking_tokens(draft_json["Pitch"])
                    
                    # process initial offer
                    if isinstance(draft_json["Initial_Offer"], dict):
                        for key, value in draft_json["Initial_Offer"].items():
                            if isinstance(value, str):
                                draft_json["Initial_Offer"][key], _ = strip_thinking_tokens(value)
                    
                    best_pitch = draft_json
                    best_score = feedback_json["Score"]
                    self.logger.info(f"New best pitch found with score: {best_score}/10")
            else:
                # try to extract partial info
                score = None
                decision = None
                
                if critic_json and isinstance(critic_json, dict):
                    score = critic_json.get("Score")
                    decision = critic_json.get("Decision")
                
                # log error for debugging
                if critic_json:
                    self.logger.error(f"Incomplete critic response. Found keys: {list(critic_json.keys()) if isinstance(critic_json, dict) else 'Not a dict'}")
                else:
                    self.logger.error("Invalid critic response format. Using a placeholder.")
                
                # use partial info
                feedback_json = {
                    "Score": score if score is not None else 0,
                    "Decision": decision if decision is not None else "Feedback",
                    "Strengths": ["Unknown"],
                    "Weaknesses": ["Could not parse critic feedback completely"],
                    "Improvement_Areas": ["Unknown"],
                    "Suggestions": ["Unknown"]
                }
                
                # update best pitch if valid score found
                if score and score > best_score:
                    best_pitch = draft_json
                    best_score = score
                    self.logger.info(f"Using partial information: New best pitch found with score: {best_score}/10")
            
            # store iteration results
            iteration_result = {
                "Iteration": iteration,
                "Draft": draft_json,
                "Feedback": feedback_json
            }
            all_iterations.append(iteration_result)
            
            # exit if marked as final
            if feedback_json["Decision"] == "Final":
                self.logger.info(f"Pitch critic has marked this as the final version with score: {feedback_json['Score']}/10")
                break
            
            # prepare feedback for next iteration
            if iteration < self.max_iterations:
                if feedback_json["Decision"] == "Feedback":
                    feedback_details = ""
                    
                    if "Strengths" in feedback_json:
                        feedback_details += "\nSTRENGTHS:\n"
                        for i, strength in enumerate(feedback_json["Strengths"], 1):
                            if isinstance(strength, str):
                                filtered_strength, _ = strip_thinking_tokens(strength)
                                feedback_details += f"{i}. {filtered_strength}\n"
                            else:
                                feedback_details += f"{i}. {strength}\n"
                            
                    if "Weaknesses" in feedback_json:
                        feedback_details += "\nWEAKNESSES:\n"
                        for i, weakness in enumerate(feedback_json["Weaknesses"], 1):
                            if isinstance(weakness, str):
                                filtered_weakness, _ = strip_thinking_tokens(weakness)
                                feedback_details += f"{i}. {filtered_weakness}\n"
                            else:
                                feedback_details += f"{i}. {weakness}\n"
                            
                    if "Improvement_Areas" in feedback_json:
                        feedback_details += "\nIMPROVEMENT AREAS:\n"
                        for i, area in enumerate(feedback_json["Improvement_Areas"], 1):
                            if isinstance(area, str):
                                filtered_area, _ = strip_thinking_tokens(area)
                                feedback_details += f"{i}. {filtered_area}\n"
                            else:
                                feedback_details += f"{i}. {area}\n"
                            
                    if "Suggestions" in feedback_json:
                        feedback_details += "\nSPECIFIC SUGGESTIONS:\n"
                        for i, suggestion in enumerate(feedback_json["Suggestions"], 1):
                            # filter suggestion if it's a string that might have thinking tokens
                            if isinstance(suggestion, str):
                                filtered_suggestion, _ = strip_thinking_tokens(suggestion)
                                feedback_details += f"{i}. {filtered_suggestion}\n"
                            else:
                                feedback_details += f"{i}. {suggestion}\n"
                            
                    # include detailed feedback in next drafter context
                    self.next_drafter_context = f"""
                    Please revise your pitch based on the following feedback from the Pitch Critic (Score: {feedback_json['Score']}/10):
                    {feedback_details}
                    
                    Remember to maintain the factual accuracy while addressing these points.
                    """
            
            # use best pitch if max iterations reached without good enough score
            if iteration == self.max_iterations:
                self.logger.info(f"Reached maximum iterations ({self.max_iterations}). Using best pitch with score: {best_score}/10")
        
        execution_time = time.time() - start_time
        
        # clean up any remaining thinking tokens in best pitch
        if best_pitch:
            if isinstance(best_pitch["Pitch"], str):
                best_pitch["Pitch"], _ = strip_thinking_tokens(best_pitch["Pitch"])
            
            if isinstance(best_pitch["Initial_Offer"], dict):
                for key, value in best_pitch["Initial_Offer"].items():
                    if isinstance(value, str):
                        best_pitch["Initial_Offer"][key], _ = strip_thinking_tokens(value)
        
        # log metrics
        output_pitch_length = len(json.dumps(best_pitch, indent=2))
        total_output_length += output_pitch_length
        
        self.logger.info(f"Output pitch length: {output_pitch_length}")
        self.logger.info(f"Total input prompt length: {total_input_length}")
        self.logger.info(f"Total output response length: {total_output_length}")
        self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
        
        # save outputs
        with open(os.path.join(self.output_dir, "final_pitch.json"), "w", encoding="utf-8") as f:
            json.dump(best_pitch, f, indent=2)
        
        with open(os.path.join(self.output_dir, "iterations.json"), "w", encoding="utf-8") as f:
            json.dump(all_iterations, f, indent=2)
        
        metrics = {
            "total_execution_time_seconds": execution_time,
            "total_input_prompt_length": total_input_length,
            "total_output_response_length": total_output_length,
            "best_score": best_score,
            "iterations_completed": len(all_iterations),
            "max_iterations": self.max_iterations
        }
        with open(os.path.join(self.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        return best_pitch

def load_data(file_path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        Any: The loaded data (could be dict, list, or other JSON-serializable type).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        return {}

def setup_arg_parser():
    """
    Set up and return the argument parser for command-line arguments.
    
    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches using an actor-critic approach.')
    
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                        help='Model ID to use for all agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--product-key', type=str, default='facts_shark_tank_transcript_0_GarmaGuard.txt',
                        help='Product key/filename to process from the facts data (default: facts_shark_tank_transcript_0_GarmaGuard.txt)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                        help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--max-iterations', type=int, default=5,
                        help='Maximum number of iterations between drafter and critic (default: 5)')
    
    parser.add_argument('--no-debug', action='store_false', dest='debug', default=True,
                        help='Disable debug mode for detailed agent logging (default: enabled)')
    
    return parser

def main():
    """
    Main function to run the SharkTank pitch generator with actor-critic approach.
    """
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"./outputs/actor_critic_pitch_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_name = f"full_log_actor_critic_pitch_{timestamp}.log"
    log_file_path = os.path.join(output_dir, log_file_name)
    
    # clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    logger = logging.getLogger(f"actor_critic_generator_{timestamp}")
    logger.setLevel(logging.INFO)
    
    if logger.handlers:
        logger.handlers.clear()
        
    logger.propagate = False
    
    # capture stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeLogger(log_file_path, 'a', 'utf-8', original_stdout)
    sys.stderr = TeeLogger(log_file_path, 'a', 'utf-8', original_stderr)
    
    # log startup info
    logger.info(f"Starting SharkTank pitch generator with actor-critic approach")
    logger.info(f"Model: {args.model}, product key: {args.product_key}")
    logger.info(f"Max tokens: {args.max_tokens}, temperature: {args.temperature}")
    logger.info(f"Max iterations: {args.max_iterations}")
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
    
    # save input data
    with open(os.path.join(output_dir, "input_data.json"), "w", encoding="utf-8") as f:
        json.dump({
            "episode_key": episode_key,
            "facts": facts,
            "product_description": product_description,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "max_iterations": args.max_iterations
        }, f, indent=2)
    
    team = ActorCriticPitchTeam(
        output_dir=output_dir, 
        log_file_path=log_file_path, 
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        debug_mode=args.debug,
        max_iterations=args.max_iterations
    )
    
    try:
        final_pitch = team.generate_pitch(facts, product_description, episode_key)
        logger.info(f"Final pitch generated and saved to {output_dir}/final_pitch.json")
    except Exception as e:
        logger.error(f"Error generating pitch: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # restore stdout/stderr
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return

if __name__ == "__main__":
    main() 