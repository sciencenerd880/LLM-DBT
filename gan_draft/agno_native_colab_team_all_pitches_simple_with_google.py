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

from agno.agent import Agent
from agno.models.groq import Groq
from agno.team.team import Team

from agno.tools.googlesearch import GoogleSearchTools

load_dotenv()

# get groq api key from env or LLM-DBT/.env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    llm_dbt_env_path = os.path.join(os.getcwd(), "LLM-DBT", ".env")
    if os.path.exists(llm_dbt_env_path):
        load_dotenv(llm_dbt_env_path)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set GROQ_API_KEY in your environment or in the .env file.")

def create_collaborative_pitch_team(model_id="llama-3.3-70b-versatile", max_tokens=4096, temperature=0.7):
    """
    Creates a collaborative team for SharkTank pitch generation with Google Search tools.
    
    Args:
        model_id (str): The ID of the model to use
        max_tokens (int): Maximum number of tokens for model responses
        temperature (float): Temperature for model responses
        
    Returns:
        Team: The configured Agno collaborative team
    """
    # helper function to create groq model instances
    def create_groq_model():
        return Groq(
            id=model_id,
            api_key=GROQ_API_KEY,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    google_search = GoogleSearchTools()
    
    # instructions for tool usage that will be added to all agents
    tool_usage_instructions = """
    IMPORTANT TOOL USAGE INSTRUCTIONS:
    When you need to use the Google Search tool, use the following exact format:
    
    I'll use the Google Search tool to find information about [topic].
    
    DO NOT use any special formatting or syntax when referring to tools.
    Wait for the coordinator to process your search request.
    """
    
    financial_strategist = Agent(
        name="Financial Strategist",
        model=create_groq_model(),
        role="Analyze financial aspects of the business",
        tools=[google_search],
        add_name_to_instructions=True,
        instructions=dedent(f"""
            You are a Financial Strategist with expertise in startup valuation, investment analysis, and business model evaluation.
            
            {tool_usage_instructions}
            
            Use Google Search to research additional financial information when needed, such as:
            - Industry standard valuations
            - Recent funding trends in the product's sector
            - Comparable companies and their financial metrics
            - Typical investment terms for similar businesses
            
            Analyze the provided facts and product description to develop a financial strategy for a SharkTank pitch, including:
            1. A justified valuation for the company
            2. An appropriate investment amount to request
            3. A fair equity percentage to offer
            4. A breakdown of how the funds will be used
            5. A realistic ROI timeline
            6. Potential exit strategies
            
            Base your analysis primarily on the facts provided, but supplement with research when necessary.
        """),
        debug_mode=False
    )
    
    market_research_specialist = Agent(
        name="Market Research Specialist",
        model=create_groq_model(),
        role="Analyze market size, trends, and competition",
        tools=[google_search],
        add_name_to_instructions=True,
        instructions=dedent(f"""
            You are a Market Research Specialist with deep knowledge of consumer trends, market analysis, and competitive landscapes.
            
            {tool_usage_instructions}
            
            Use Google Search to research additional market information when needed, such as:
            - Current market size and growth projections
            - Emerging trends in the product's industry
            - Competitors and their market positions
            - Consumer behavior and preferences
            
            Analyze the provided facts and product description to develop market insights for a SharkTank pitch, including:
            1. The estimated size of the target market
            2. Description of target customer segments
            3. Analysis of competitors and their strengths/weaknesses
            4. Relevant market trends
            5. Potential growth opportunities
            6. Challenges in the market
            
            Base your analysis primarily on the facts provided, but supplement with research when necessary.
        """),
        debug_mode=False
    )
    
    product_technical_advisor = Agent(
        name="Product Technical Advisor",
        model=create_groq_model(),
        role="Assess product features, advantages, and technical feasibility",
        tools=[google_search],
        add_name_to_instructions=True,
        instructions=dedent(f"""
            You are a Product/Technical Advisor with expertise in product development, technical feasibility, and innovation assessment.
            
            {tool_usage_instructions}
            
            Use Google Search to research additional technical information when needed, such as:
            - Technical specifications for similar products
            - Manufacturing or production processes
            - Materials or components
            - Intellectual property considerations
            - Emerging technologies in the field
            
            Analyze the provided facts and product description to develop product insights for a SharkTank pitch, including:
            1. Key product features to highlight
            2. Technical advantages over competitors
            3. How to effectively demonstrate the product
            4. Assessment of production/technical scalability
            5. Potential future product developments
            
            Base your analysis primarily on the facts provided, but supplement with research when necessary.
        """),
        debug_mode=False
    )
    
    shark_psychology_expert = Agent(
        name="Shark Psychology Expert",
        model=create_groq_model(),
        role="Provide insights on Shark preferences and negotiation strategy",
        tools=[google_search],
        add_name_to_instructions=True,
        instructions=dedent(f"""
            You are a Shark Psychology Expert who understands the motivations, preferences, and decision patterns of SharkTank investors.
            
            {tool_usage_instructions}
            
            Use Google Search to research additional information when needed, such as:
            - Recent investments made by Sharks
            - Investment preferences of individual Sharks
            - Successful negotiation tactics used on the show
            - Common reasons for Shark rejections
            
            Analyze the provided facts and product description to develop investor psychology insights for a SharkTank pitch, including:
            1. Points that will appeal to Sharks
            2. Potential objections and how to counter them
            3. Strategy for negotiating with Sharks
            4. Tips for effective presentation
            5. Sharks that might be the best fit and why
            
            Base your analysis primarily on the facts provided, but supplement with research when necessary.
        """),
        debug_mode=False
    )
    
    pitch_drafter = Agent(
        name="Pitch Drafter",
        model=create_groq_model(),
        role="Create the final pitch script",
        tools=[google_search],
        add_name_to_instructions=True,
        instructions=dedent(f"""
            You are a skilled pitch writer for entrepreneurs appearing on Shark Tank.
            
            {tool_usage_instructions}
            
            Use Google Search to research additional information when needed, such as:
            - Successful pitch structures and techniques
            - Compelling hooks and storytelling approaches
            - Effective ways to present data and numbers
            - Examples of successful pitches for similar products
            
            IMPORTANT: Your final output MUST be in valid JSON format exactly as shown below:
            {{
              "Pitch": "The complete pitch text goes here...", 
              "Initial_Offer": {{
                "Valuation": "$X million", 
                "Equity_Offered": "X%", 
                "Funding_Amount": "$X", 
                "Key_Terms": "Any special terms..."
              }}
            }}
            
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
            
            f"{tool_usage_instructions}",
            
            "TOOL COORDINATION INSTRUCTIONS:",
            "When team members want to search for information, help coordinate their searches by:",
            "1. Acknowledging their search intent",
            "2. Suggesting specific search queries if needed",
            "3. Asking them to share what they learned after the search",
            
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
            
            "Each team member has access to Google Search tools. Help them use these tools effectively.",
            "Review all team members' inputs carefully to synthesize a comprehensive pitch.",
            "Ensure the pitch is factually accurate based on the provided information and team member analyses.",
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
        show_tool_calls=True,
        debug_mode=False
    )
    
    print(f"Created collaborative pitch team with model: {model_id} and Google Search tools")
    return pitch_team

async def generate_pitch(team, facts, product_description, product_key, output_dir):
    # make a pitch for sharktank and track how it goes
    facts_str = str(facts) if not isinstance(facts, str) else facts
    product_description_str = str(product_description) if not isinstance(product_description, str) else product_description
    
    context = f"""
    Please work together as a team to create a compelling SharkTank pitch based on the following information:
    
    FACTS:
    {facts_str}
    
    PRODUCT DESCRIPTION:
    {product_description_str}
    
    IMPORTANT TOOL USAGE GUIDELINES:
    - When you need to search for information, use plain language to describe what you want to search for
    - DO NOT use special syntax or formatting for tool calls
    - The coordinator will help process search requests
    
    Each team member should analyze this information from their specialized perspective.
    Focus primarily on the facts provided, and only use search when absolutely necessary.
    The coordinator will synthesize all inputs into a cohesive pitch.
    """
    
    print(f"Processing: {product_key}")
    
    # set up files to track what happens
    interactions_log_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_interactions.txt")
    raw_response_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_raw_responses.txt")
    tools_usage_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_tools_usage.txt")
    
    # start tracking interactions
    with open(interactions_log_path, 'w', encoding='utf-8') as f:
        f.write(f"INITIAL PROMPT:\n{context}\n\n")
        f.write(f"INPUT LENGTH: {len(context)}\n\n")
        f.write("TEAM INTERACTIONS:\n\n")
    
    start_time = time.time()
    
    class InteractionTracker:
        def __init__(self, log_path, raw_path, tools_path):
            self.log_path = log_path
            self.raw_path = raw_path
            self.tools_path = tools_path
            self.total_input_length = len(context)  
            self.total_output_length = 0
            self.tool_calls_count = 0
            self.member_interactions = {}
            
            # track each team member
            for member in team.members:
                member_name = member.name if hasattr(member, 'name') else "Unknown"
                self.member_interactions[member_name] = {
                    'input_length': 0,
                    'output_length': 0,
                    'tool_calls': 0
                }
        
        def log_interaction(self, role, is_input, content):
            with open(self.log_path, 'a', encoding='utf-8') as f:
                interaction_type = "INPUT" if is_input else "OUTPUT"
                f.write(f"[{role}] {interaction_type}:\n{content}\n\n")
            
            content_length = len(str(content))
            if is_input:
                self.total_input_length += content_length
                if role in self.member_interactions:
                    self.member_interactions[role]['input_length'] += content_length
            else:
                self.total_output_length += content_length
                if role in self.member_interactions:
                    self.member_interactions[role]['output_length'] += content_length
        
        def log_raw_response(self, response):
            with open(self.raw_path, 'a', encoding='utf-8') as f:
                f.write(f"RAW RESPONSE OBJECT:\n{str(response)}\n\n")
                
                if hasattr(response, '__dict__'):
                    f.write("\nRESPONSE ATTRIBUTES:\n")
                    for attr in dir(response):
                        if not attr.startswith('_'):
                            try:
                                value = getattr(response, attr)
                                f.write(f"{attr}: {value}\n")
                                
                                attr_str = str(value)
                                attr_len = len(attr_str)
                                
                                # only count long content
                                if attr_len > 20:  
                                    self.total_output_length += attr_len
                            except:
                                pass
        
        def log_tool_usage(self, agent, tool_name, input_data, output_data):
            self.tool_calls_count += 1
            with open(self.tools_path, 'a', encoding='utf-8') as f:
                f.write(f"TOOL USAGE #{self.tool_calls_count}:\n")
                f.write(f"Agent: {agent}\n")
                f.write(f"Tool: {tool_name}\n")
                f.write(f"Input: {input_data}\n")
                f.write(f"Output: {output_data}\n\n")
            
            input_len = len(str(input_data))
            output_len = len(str(output_data))
            
            self.total_input_length += input_len
            self.total_output_length += output_len
            
            if agent in self.member_interactions:
                self.member_interactions[agent]['input_length'] += input_len
                self.member_interactions[agent]['output_length'] += output_len
                self.member_interactions[agent]['tool_calls'] += 1
        
        def extract_steps(self, steps):
            total_input_from_steps = 0
            total_output_from_steps = 0
            
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write("\nINTERNAL STEPS:\n\n")
                for i, step in enumerate(steps):
                    f.write(f"Step {i}:\n{step}\n\n")
                    
                    if isinstance(step, dict):
                        if 'input' in step:
                            input_str = str(step['input'])
                            f.write(f"Input length: {len(input_str)}\n")
                            total_input_from_steps += len(input_str)
                        
                        if 'output' in step:
                            output_str = str(step['output'])
                            f.write(f"Output length: {len(output_str)}\n")
                            total_output_from_steps += len(output_str)
                        
                        if 'tool_calls' in step or 'tools' in step:
                            tool_data = step.get('tool_calls', step.get('tools', []))
                            if tool_data:
                                f.write(f"Tool calls found in step {i}\n")
                                for tool_call in (tool_data if isinstance(tool_data, list) else [tool_data]):
                                    if isinstance(tool_call, dict):
                                        tool_inputs = str(tool_call.get('inputs', tool_call.get('arguments', '')))
                                        tool_outputs = str(tool_call.get('outputs', tool_call.get('result', '')))
                                        
                                        total_input_from_steps += len(tool_inputs)
                                        total_output_from_steps += len(tool_outputs)
                                        
                                        agent_name = tool_call.get('agent', 'Unknown')
                                        tool_name = tool_call.get('tool', tool_call.get('name', 'Unknown'))
                                        self.log_tool_usage(agent_name, tool_name, tool_inputs, tool_outputs)
            
            if total_input_from_steps > self.total_input_length:
                self.total_input_length = total_input_from_steps
            else:
                self.total_input_length += total_input_from_steps // 2
                
            if total_output_from_steps > self.total_output_length:
                self.total_output_length = total_output_from_steps
            else:
                self.total_output_length += total_output_from_steps // 2
            
            return total_input_from_steps, total_output_from_steps
        
        def summarize_metrics(self):
            with open(self.log_path, 'a', encoding='utf-8') as f:
                f.write("\nMETRICS SUMMARY:\n")
                f.write(f"Total Input Length: {self.total_input_length}\n")
                f.write(f"Total Output Length: {self.total_output_length}\n")
                f.write(f"Tool Calls Count: {self.tool_calls_count}\n\n")
                
                f.write("PER-MEMBER METRICS:\n")
                for member, metrics in self.member_interactions.items():
                    f.write(f"{member}:\n")
                    f.write(f"  Input Length: {metrics['input_length']}\n")
                    f.write(f"  Output Length: {metrics['output_length']}\n")
                    f.write(f"  Tool Calls: {metrics['tool_calls']}\n")
                f.write("\n")
    
    tracker = InteractionTracker(interactions_log_path, raw_response_path, tools_usage_path)
    
    # turn on verbose mode to get max logging
    for agent in team.members:
        if hasattr(agent, 'verbose'):
            agent.verbose = True

    try:
        # first try with google search enabled
        response = None
        try:
            # track tool usage
            for agent in team.members:
                if hasattr(agent, 'tools') and agent.tools:
                    for tool in agent.tools:
                        if hasattr(tool, 'name'):
                            tool_name = tool.name
                        elif hasattr(tool, '__class__'):
                            tool_name = tool.__class__.__name__
                        else:
                            tool_name = "Unknown"
                        
                        tracker.log_interaction(
                            agent.name if hasattr(agent, 'name') else "Unknown", 
                            True, 
                            f"Agent has tool: {tool_name}"
                        )
            
            response = await team.arun(
                message=context,
                show_members_responses=True,  
                markdown=True,
                stream=False,                  
                stream_intermediate_steps=False 
            )
            
            tracker.log_raw_response(response)
            
            internal_steps = []
            if hasattr(team, '_steps'):
                internal_steps = team._steps
            elif hasattr(team, 'steps'):
                internal_steps = team.steps
            elif hasattr(response, 'steps'):
                internal_steps = response.steps
                
            if internal_steps:
                input_from_steps, output_from_steps = tracker.extract_steps(internal_steps)
                tracker.log_interaction(
                    "SYSTEM", 
                    False, 
                    f"Extracted from steps: {input_from_steps} input chars, {output_from_steps} output chars"
                )
            
            tracker.log_interaction("SYSTEM", False, "Successfully completed with Google Search tools enabled")
            
        except Exception as e:
            error_message = f"Error with Google Search tools: {str(e)}"
            print(error_message)
            tracker.log_interaction("ERROR", False, error_message)
            
            # try again without tools if we get an error
            fallback_context = f"""
            Please work together as a team to create a compelling SharkTank pitch based on the following information:
            
            FACTS:
            {facts_str}
            
            PRODUCT DESCRIPTION:
            {product_description_str}
            
            IMPORTANT: Due to technical limitations, please DO NOT attempt to use any search tools or external resources.
            Focus ONLY on the information provided and your existing knowledge.
            
            Each team member should analyze this information from their specialized perspective.
            The coordinator will then synthesize all inputs into a cohesive pitch.
            
            The final output MUST be in valid JSON format with no additional text before or after.
            """
            
            tracker.log_interaction("SYSTEM", True, "Trying fallback approach without tool usage...")
            tracker.log_interaction("SYSTEM", True, fallback_context)
            
            response = await team.arun(
                message=fallback_context,
                show_members_responses=False,
                markdown=True,
                stream=False
            )
            
            tracker.log_raw_response(response)
            
            tracker.log_interaction("SYSTEM", False, "Successfully completed with fallback approach")
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        content = ""
        if hasattr(response, 'content'):
            content = str(response.content)
        else:
            content = str(response)
        
        if hasattr(response, 'member_responses'):
            member_responses = response.member_responses
            tracker.log_interaction("SYSTEM", False, f"Found member responses in response object (type: {type(member_responses).__name__})")
            
            if isinstance(member_responses, dict):
                for member_name, member_response in member_responses.items():
                    member_response_str = str(member_response)
                    tracker.log_interaction(member_name, False, member_response_str)
                    
                    if member_name in tracker.member_interactions:
                        tracker.member_interactions[member_name]['output_length'] += len(member_response_str)
            elif isinstance(member_responses, list):
                tracker.log_interaction("SYSTEM", False, f"Member responses is a list with {len(member_responses)} items")
                
                for i, member_response in enumerate(member_responses):
                    member_name = None
                    if hasattr(member_response, 'name'):
                        member_name = member_response.name
                    elif hasattr(member_response, 'agent_name'):
                        member_name = member_response.agent_name
                    elif isinstance(member_response, dict) and ('name' in member_response or 'agent_name' in member_response):
                        member_name = member_response.get('name', member_response.get('agent_name', f"Member_{i}"))
                    else:
                        member_name = f"Member_{i}"
                    
                    member_response_str = str(member_response)
                    tracker.log_interaction(member_name, False, member_response_str)
                    
                    matching_members = [m for m in tracker.member_interactions.keys() 
                                       if m.lower() == member_name.lower() or 
                                          member_name.lower().startswith(m.lower()) or 
                                          m.lower().startswith(member_name.lower())]
                    
                    if matching_members:
                        tracker.member_interactions[matching_members[0]]['output_length'] += len(member_response_str)
                    else:
                        tracker.total_output_length += len(member_response_str)
                        tracker.log_interaction("SYSTEM", False, f"No team member matched for response from {member_name}, added {len(member_response_str)} chars to total output")
            else:
                tracker.log_interaction("SYSTEM", False, f"Unexpected member_responses format: {type(member_responses).__name__}")
                tracker.log_interaction("SYSTEM", False, str(member_responses))
                tracker.total_output_length += len(str(member_responses))
        
        if hasattr(response, 'tool_calls') or hasattr(response, 'tools'):
            tool_data = getattr(response, 'tool_calls', getattr(response, 'tools', []))
            if tool_data:
                if isinstance(tool_data, list):
                    for tool_call in tool_data:
                        agent_name = getattr(tool_call, 'agent', 'Unknown')
                        tool_name = getattr(tool_call, 'tool', getattr(tool_call, 'name', 'Unknown'))
                        inputs = getattr(tool_call, 'inputs', getattr(tool_call, 'arguments', ''))
                        outputs = getattr(tool_call, 'outputs', getattr(tool_call, 'result', ''))
                        
                        tracker.log_tool_usage(agent_name, tool_name, inputs, outputs)
                elif isinstance(tool_data, dict):
                    for tool_name, tool_info in tool_data.items():
                        if isinstance(tool_info, dict):
                            inputs = tool_info.get('inputs', tool_info.get('arguments', ''))
                            outputs = tool_info.get('outputs', tool_info.get('result', ''))
                            agent_name = tool_info.get('agent', 'Unknown')
                            
                            tracker.log_tool_usage(agent_name, tool_name, inputs, outputs)
        
        # get just the json part
        json_content = ""
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1
        
        if start_idx >= 0 and end_idx > start_idx:
            json_content = content[start_idx:end_idx]
            try:
                json.loads(json_content)
                content = json_content
                tracker.log_interaction("SYSTEM", False, "Successfully extracted clean JSON content")
            except json.JSONDecodeError:
                tracker.log_interaction("SYSTEM", False, "Warning: Extracted content is not valid JSON. Keeping original response.")
        
        # add baseline interactions if we missed some
        for agent in team.members:
            agent_name = agent.name if hasattr(agent, 'name') else "Agent"
            
            if agent_name in tracker.member_interactions and tracker.member_interactions[agent_name]['input_length'] < len(context):
                tracker.member_interactions[agent_name]['input_length'] += len(context)
                
                if tracker.member_interactions[agent_name]['output_length'] < 100:
                    estimated_output = len(content) // len(team.members)
                    tracker.member_interactions[agent_name]['output_length'] += estimated_output
                    
                    tracker.log_interaction(
                        agent_name, 
                        False, 
                        f"[Estimated] Added baseline output estimate of {estimated_output} chars"
                    )
        
        tracker.log_interaction("FINAL", False, content)
        
        tracker.summarize_metrics()
        
        result_file_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}.txt")
        with open(result_file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        metrics_path = os.path.join(output_dir, f"{product_key.replace('.txt', '')}_metrics.txt")
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"Initial Input Length: {len(context)}\n")
            f.write(f"Final Output Length: {len(content)}\n")
            f.write(f"Total Input Length: {tracker.total_input_length}\n")
            f.write(f"Total Output Length: {tracker.total_output_length}\n")
            f.write(f"Tool Calls Count: {tracker.tool_calls_count}\n")
            f.write(f"Time Taken: {time_taken} seconds\n")
        
        metrics = {
            'Product_Key': product_key,
            'Total_time_taken': time_taken,
            'Total_Input_Prompt_String_Length': tracker.total_input_length,
            'Total_Output_Response_String_Length': tracker.total_output_length,
            'Tool_Calls_Count': tracker.tool_calls_count,
            'Final_Response': content
        }
        
        print(f"Completed {product_key} in {time_taken:.2f} seconds with {tracker.tool_calls_count} tool calls")
        return metrics
        
    except Exception as e:
        print(f"Error during processing: {e}")
        with open(interactions_log_path, 'a', encoding='utf-8') as f:
            f.write(f"\nERROR: {e}\n")
        raise

def load_data(file_path):
    # load data from a json file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return {}

def create_output_directories():
    # make directories for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = "eval_native_colab_team_with_google"
    run_dir = f"native_colab_team_with_google_{timestamp}"
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    output_dir = os.path.join(base_dir, run_dir)
    os.makedirs(output_dir)
    
    return output_dir, timestamp

async def main_async():
    # run the sharktank pitch generator on all products
    parser = argparse.ArgumentParser(description='Generate SharkTank pitches for all products using Agno\'s native collaborative team approach with Google Search tools.')
    
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
    
    metrics_csv_filename = "native_colab_with_google_results_compile_llama-3.3-70b-versatile.csv"
    metrics_file_path = os.path.join(output_dir, metrics_csv_filename)
    
    metrics_df = pd.DataFrame(all_metrics)
    
    if 'Tool_Calls_Count' not in metrics_df.columns and all_metrics:
        metrics_df['Tool_Calls_Count'] = 0
    
    metrics_df.to_csv(metrics_file_path, index=False)
    
    print(f"All runs completed. Results saved to: {output_dir}")
    print(f"Metrics saved to: {metrics_file_path}")
    
    if all_metrics:
        total_tool_calls = sum(m.get('Tool_Calls_Count', 0) for m in all_metrics)
        avg_input_length = sum(m['Total_Input_Prompt_String_Length'] for m in all_metrics) / len(all_metrics)
        avg_output_length = sum(m['Total_Output_Response_String_Length'] for m in all_metrics) / len(all_metrics)
        
        print(f"\nSUMMARY STATISTICS:")
        print(f"Total products processed: {len(all_metrics)}")
        print(f"Total tool calls across all products: {total_tool_calls}")
        print(f"Average input length: {avg_input_length:.2f} characters")
        print(f"Average output length: {avg_output_length:.2f} characters")
        
        if total_tool_calls > 0:
            avg_tool_calls = total_tool_calls / len(all_metrics)
            print(f"Average tool calls per product: {avg_tool_calls:.2f}")

def main():
    # call the async main function
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 
