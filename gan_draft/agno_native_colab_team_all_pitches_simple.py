import os
import json
import argparse
import asyncio
import time
import csv
import logging
import io
from datetime import datetime
from textwrap import dedent
import pandas as pd
from dotenv import load_dotenv

# agno imports
from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team

# load environment variables
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

def create_collaborative_pitch_team(model_id="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.7):
    """
    Creates a collaborative team for SharkTank pitch generation.
    
    Args:
        model_id (str): The ID of the model to use
        max_tokens (int): Maximum number of tokens for model responses
        temperature (float): Temperature for model responses
        
    Returns:
        Team: The configured Agno collaborative team
    """
    # helper function to create groq model with same params
    def create_groq_model():
        return Groq(
            id=model_id,
            api_key=GROQ_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    financial_strategist = Agent(
        name="Financial Strategist",
        model=create_groq_model(),
        role="Analyze financial aspects of the business",
        add_name_to_instructions=True,
        instructions=dedent("""
            You are a Financial Strategist with expertise in startup valuation, investment analysis, and business model evaluation.
            Analyze the provided facts and product description to develop a financial strategy for a SharkTank pitch, including:
            1. A justified valuation for the company
            2. An appropriate investment amount to request
            3. A fair equity percentage to offer
            4. A breakdown of how the funds will be used
            5. A realistic ROI timeline
            6. Potential exit strategies
            Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.
        """),
        debug_mode=False
    )
    
    market_research_specialist = Agent(
        name="Market Research Specialist",
        model=create_groq_model(),
        role="Analyze market size, trends, and competition",
        add_name_to_instructions=True,
        instructions=dedent("""
            You are a Market Research Specialist with deep knowledge of consumer trends, market analysis, and competitive landscapes.
            Analyze the provided facts and product description to develop market insights for a SharkTank pitch, including:
            1. The estimated size of the target market
            2. Description of target customer segments
            3. Analysis of competitors and their strengths/weaknesses
            4. Relevant market trends
            5. Potential growth opportunities
            6. Challenges in the market
            Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.
        """),
        debug_mode=False
    )
    
    product_technical_advisor = Agent(
        name="Product Technical Advisor",
        model=create_groq_model(),
        role="Assess product features, advantages, and technical feasibility",
        add_name_to_instructions=True,
        instructions=dedent("""
            You are a Product/Technical Advisor with expertise in product development, technical feasibility, and innovation assessment.
            Analyze the provided facts and product description to develop product insights for a SharkTank pitch, including:
            1. Key product features to highlight
            2. Technical advantages over competitors
            3. How to effectively demonstrate the product
            4. Assessment of production/technical scalability
            5. Potential future product developments
            Base your analysis primarily on the facts provided, but you may use your industry knowledge for reasonable inferences.
        """),
        debug_mode=False
    )
    
    shark_psychology_expert = Agent(
        name="Shark Psychology Expert",
        model=create_groq_model(),
        role="Provide insights on Shark preferences and negotiation strategy",
        add_name_to_instructions=True,
        instructions=dedent("""
            You are a Shark Psychology Expert who understands the motivations, preferences, and decision patterns of SharkTank investors.
            Analyze the provided facts and product description to develop investor psychology insights for a SharkTank pitch, including:
            1. Points that will appeal to Sharks
            2. Potential objections and how to counter them
            3. Strategy for negotiating with Sharks
            4. Tips for effective presentation
            5. Sharks that might be the best fit and why
            Base your analysis primarily on the facts provided, but you may use your knowledge of Shark Tank investors for reasonable inferences.
        """),
        debug_mode=False
    )
    
    pitch_drafter = Agent(
        name="Pitch Drafter",
        model=create_groq_model(),
        role="Create the final pitch script",
        add_name_to_instructions=True,
        instructions=dedent("""
            You are a skilled pitch writer for entrepreneurs appearing on Shark Tank.
            
            IMPORTANT: Your final output MUST be in valid JSON format exactly as shown below:
            {
              "Pitch": "The complete pitch text goes here...", 
              "Initial_Offer": {
                "Valuation": "$X million", 
                "Equity_Offered": "X%", 
                "Funding_Amount": "$X", 
                "Key_Terms": "Any special terms..."
              }
            }
            
            Your task is to create a compelling pitch based on the specialist analyses provided and facts.
            The pitch should be structured to grab attention, clearly explain the product/service, highlight market potential,
            showcase competitive advantages, present financial data, make a specific investment ask, and close with a strong call to action.
            
            When writing the JSON:
            1. Use double quotes (") for all JSON keys and string values
            2. Escape any double quotes within text values with a backslash (\\")
            3. Avoid special characters or line breaks that would break JSON syntax
            4. Ensure all brackets and braces are properly closed
            
            REMINDER: The ENTIRE response must be in valid JSON format with 'Pitch' and 'Initial_Offer' fields.
        """),
        debug_mode=False
    )
    
    # set up the team with all our agents
    pitch_team = Team(
        name="SharkTank Pitch Team",
        mode="collaborate",
        model=create_groq_model(),
        members=[
            financial_strategist,
            market_research_specialist,
            product_technical_advisor, 
            shark_psychology_expert,
            pitch_drafter
        ],
        instructions=[
            "You are a discussion master and coordinator of a team creating a SharkTank pitch.",
            "IMPORTANT: You must ONLY stop the discussion when a genuine consensus has been reached among ALL team members.",
            "If there are any disagreements or incomplete perspectives, continue the discussion until true alignment is achieved.",
            "Each team member must have their specialized analysis fully incorporated in the final synthesis.",
            
            "ABSOLUTELY CRITICAL: Your FINAL response MUST ONLY contain valid JSON with NO TEXT BEFORE OR AFTER the JSON. No explanations, no markdown formatting, no additional comments - ONLY the JSON object.",
            "{",
            "  \"Pitch\": \"The complete pitch text goes here...\",", 
            "  \"Initial_Offer\": {",
            "    \"Valuation\": \"$X million\",", 
            "    \"Equity_Offered\": \"X%\",", 
            "    \"Funding_Amount\": \"$X\",", 
            "    \"Key_Terms\": \"Any special terms...\"",
            "  }",
            "}",
            
            "Review all team members' inputs carefully to synthesize a comprehensive pitch.",
            "Ensure the pitch is factually accurate based on the provided information.",
            "Structure the final pitch to include a compelling hook, product description, market opportunity, competitive advantages, financial details, and a clear investment ask.",
            
            "JSON FORMAT RULES:",
            "1. Use double quotes for all JSON keys and string values",
            "2. Escape any quotes within text using backslash (\\\")",
            "3. Avoid special characters or formatting that breaks JSON",
            "4. Make sure all brackets and braces are properly balanced",
            "5. DO NOT include any text, explanations or markdown outside the JSON object",
            
            "FINAL REMINDER: Your response must be ONLY the JSON object - no introductions, no explanations, no summaries - just the raw, valid JSON."
        ],
        success_criteria="The team has provided a comprehensive analysis with ALL team members in FULL agreement, and the coordinator has synthesized a compelling, factually-accurate pitch with a reasonable investment offer in valid JSON format with NO text outside the JSON object.",
        enable_agentic_context=True,
        markdown=True,
        show_members_responses=True,
        debug_mode=False
    )
    
    print(f"Created collaborative pitch team with model: {model_id}")
    return pitch_team

async def generate_pitch(team, facts, product_description, product_key, output_dir):
    """
    Generate a pitch for a SharkTank product and record metrics.
    
    Args:
        team: The Agno collaborative team
        facts: Facts about the product
        product_description: Description of the product
        product_key: Key identifier for the product
        output_dir: Directory to save results
        
    Returns:
        dict: Metrics about the run including time taken, input/output lengths, and final response
    """
    # convert inputs to strings if needed
    facts_str = str(facts) if not isinstance(facts, str) else facts
    product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
    
    # set up the initial context for the team
    context = f"""
    Please work together as a team to create a compelling SharkTank pitch based on the following information:
    
    FACTS:
    {facts_str}
    
    PRODUCT DESCRIPTION:
    {product_description_str}
    
    Each team member should analyze this information from their specialized perspective. The coordinator will then synthesize all inputs into a cohesive pitch.
    """
    
    print(f"Processing: {product_key}")
    
    # set up log files
    interactions_log_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_interactions.txt")
    
    # start tracking interactions
    with open(interactions_log_path, 'w', encoding='utf-8') as f:
        f.write(f"INITIAL PROMPT:\n{context}\n\n")
        f.write(f"INPUT LENGTH: {len(context)}\n\n")
        f.write("TEAM INTERACTIONS:\n\n")
    
    start_time = time.time()
    
    # helper class to track all the interactions
    class InteractionTracker:
        def __init__(self, log_path):
            self.log_path = log_path
            self.total_input_length = len(context)
            self.total_output_length = 0
        
        def log_interaction(self, role, is_input, content):
            with open(self.log_path, 'a', encoding='utf-8') as f:
                interaction_type = "INPUT" if is_input else "OUTPUT"
                f.write(f"[{role}] {interaction_type}:\n{content}\n\n")
            
            if is_input:
                self.total_input_length += len(content)
            else:
                self.total_output_length += len(content)
    
    tracker = InteractionTracker(interactions_log_path)
    
    # turn on verbose mode for all agents
    for agent in team.members:
        if hasattr(agent, 'verbose'):
            agent.verbose = True
    
    try:
        # run the team and get response
        response = await team.arun(
            message=context,
            show_members_responses=True,
            markdown=True,
            stream_intermediate_steps=True
        )
        
        # save raw response for debugging
        with open(os.path.join(output_dir, f"{product_key.replace('.txt', '')}_raw_response.txt"), 'w', encoding='utf-8') as f:
            f.write(str(response))
            
            if hasattr(response, '__dict__'):
                f.write("\n\nRESPONSE ATTRIBUTES:\n")
                for attr in dir(response):
                    if not attr.startswith('_'):
                        try:
                            value = getattr(response, attr)
                            f.write(f"{attr}: {value}\n")
                        except:
                            pass
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        # get the actual content
        content = ""
        if hasattr(response, 'content'):
            content = str(response.content)
        else:
            content = str(response)
        
        # log the final output
        tracker.log_interaction("FINAL", False, content)
        
        # save the response
        result_file_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}.txt")
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # try to get internal steps
        internal_steps = []
        if hasattr(team, '_steps'):
            internal_steps = team._steps
        elif hasattr(team, 'steps'):
            internal_steps = team.steps
        
        if internal_steps:
            with open(interactions_log_path, 'a', encoding='utf-8') as f:
                f.write("\nINTERNAL STEPS:\n\n")
                for i, step in enumerate(internal_steps):
                    f.write(f"Step {i}:\n{step}\n\n")
                    
                    if isinstance(step, dict):
                        if 'input' in step:
                            tracker.total_input_length += len(str(step['input']))
                        if 'output' in step:
                            tracker.total_output_length += len(str(step['output']))
        
        # estimate interaction lengths if tracking missed some
        if tracker.total_input_length <= len(context) * 1.1:
            for agent in team.members:
                agent_name = agent.name if hasattr(agent, 'name') else "Agent"
                tracker.log_interaction(
                    agent_name, 
                    True, 
                    f"[Estimated] Processing prompt for {agent_name}"
                )
                tracker.log_interaction(
                    agent_name, 
                    False, 
                    f"[Estimated] Response from {agent_name} (estimated equal to final output length)"
                )
                tracker.total_output_length += len(content)
        
        # save metrics summary
        metrics_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Initial Input Length: {len(context)}\n")
            f.write(f"Final Output Length: {len(content)}\n")
            f.write(f"Total Input Length: {tracker.total_input_length}\n")
            f.write(f"Total Output Length: {tracker.total_output_length}\n")
            f.write(f"Time Taken: {time_taken} seconds\n")
        
        metrics = {
            'Product_Key': product_key,
            'Total_time_taken': time_taken,
            'Total_Input_Prompt_String_Length': tracker.total_input_length,
            'Total_Output_Response_String_Length': tracker.total_output_length,
            'Final_Response': content
        }
        
        print(f"Completed {product_key} in {time_taken:.2f} seconds")
        return metrics
        
    except Exception as e:
        print(f"Error during processing: {e}")
        with open(interactions_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nERROR: {e}\n")
        raise

def load_data(file_path):
    """Load data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return {}

def create_output_directories():
    """Create output directories for this run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "eval_native_colab_team"
    run_dir = f"native_colab_team_{timestamp}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    output_dir = os.path.join(base_dir, run_dir)
    os.makedirs(output_dir)
    
    return output_dir, timestamp

async def main_async():
    """Async main function to run the SharkTank pitch generator on all products."""
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches for all products using Agno\'s native collaborative team approach.')
    
    parser.add_argument('--model', type=str, default='llama-3.3-70b-versatile',
                      help='Model ID to use for all agents (default: llama-3.3-70b-versatile)')
    
    parser.add_argument('--max-tokens', type=int, default=4096,
                      help='Maximum number of tokens for model responses (default: 4096)')
    
    parser.add_argument('--temperature', type=float, default=0.7,
                      help='Temperature for model responses (0.0-1.0, default: 0.7)')
    
    parser.add_argument('--limit', type=int, default=None,
                      help='Limit the number of products to process (default: process all)')
    
    args = parser.parse_args()
    
    output_dir, timestamp = create_output_directories()
    print(f"Results will be saved to: {output_dir}")
    
    facts_file_path = "./data/all_processed_facts.json"
    if not os.path.exists(facts_file_path):
        print(f"Facts file not found at {facts_file_path}")
        return
    
    all_facts = load_data(facts_file_path)
    if not all_facts:
        print("Failed to load facts data")
        return
    
    team = create_collaborative_pitch_team(
        model_id=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )
    
    product_keys = list(all_facts.keys())
    if args.limit:
        product_keys = product_keys[:args.limit]
    
    print(f"Processing {len(product_keys)} products...")
    
    all_metrics = []
    
    for product_key in product_keys:
        try:
            episode_data = all_facts[product_key]
            
            facts = str(episode_data.get('facts', "")) if isinstance(episode_data, dict) else str(episode_data)
            product_description = str(episode_data.get('product_description', "")) if isinstance(episode_data, dict) else str(episode_data)
            
            metrics = await generate_pitch(
                team=team,
                facts=facts,
                product_description=product_description,
                product_key=product_key,
                output_dir=output_dir
            )
            
            all_metrics.append(metrics)
            
        except Exception as e:
            import traceback
            error_message = f"Error processing {product_key}: {e}\n{traceback.format_exc()}"
            print(error_message)
            
            error_file_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_ERROR.txt")
            with open(error_file_path, "w", encoding="utf-8") as f:
                f.write(error_message)
    
    metrics_csv_filename = "native_colab_results_compile_llama-3.3-70b-versatile.csv"
    metrics_file_path = os.path.join(output_dir, metrics_csv_filename)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    metrics_df.to_csv(metrics_file_path, index=False)
    
    print(f"All runs completed. Results saved to {output_dir}")
    print(f"Metrics saved to {metrics_file_path}")

def main():
    """Main function that calls the async main function."""
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 