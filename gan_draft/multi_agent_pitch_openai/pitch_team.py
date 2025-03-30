import os
import json
import asyncio
import logging
import uuid
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

from agents import (
    Agent, 
    Runner, 
    GuardrailFunctionOutput,
    RunContextWrapper,
    set_default_openai_key
)

# import modular components
from utils import load_api_key
from team_agents import create_pitch_team

# load api key when module imports
api_key = load_api_key()
print(f"API key loaded successfully: {api_key[:5]}...{api_key[-4:]}")
# make sure api key is set for both sdk and env var
set_default_openai_key(api_key)
os.environ["OPENAI_API_KEY"] = api_key

def parse_json_output(output: str) -> Dict[str, Any]:
    """
    parse json from agent output
    
    args:
        output: raw output string from agent
        
    returns:
        dict: parsed json data
    """
    # try getting json from code block first
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        # fallback to finding raw json
        json_str = output
    
    # attempt json parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # try cleaning and parsing again if first attempt fails
        try:
            # clean string to just json content
            cleaned_str = json_str[json_str.find('{'):json_str.rfind('}')+1]
            return json.loads(cleaned_str)
        except (json.JSONDecodeError, ValueError):
            # return raw text if all parsing fails
            logging.warning("All JSON parsing methods failed, returning original text in a structured format")
            return {"raw_text": output}

async def generate_pitch(facts: str, product_description: str, model: str = "gpt-4o") -> Dict[str, Any]:
    """
    generate pitch using multiple agents working together
    
    args:
        facts: product/company facts
        product_description: product details
        model: which openai model to use
        
    returns:
        dict: final pitch output
    """
    # unique id for tracking this pitch generation
    session_id = str(uuid.uuid4())
    logging.info(f"PROCESS START: Beginning pitch generation process with session ID: {session_id}")
    logging.info(f"USING MODEL: {model}")
    
    product_name = product_description if isinstance(product_description, str) else product_description.get('name', 'Unknown Product')
    logging.info(f"PRODUCT INFO: Generating pitch for '{product_name}'")
    
    # get the team ready
    logging.info(f"STAGE 1: Creating pitch team with model '{model}'...")
    team = create_pitch_team(model=model)
    logging.info(f"TEAM CREATED: Successfully initialized {len(team)} agents")
    
    # shared data between agents
    context = {
        "facts": facts,
        "product_description": product_description,
        "session_id": session_id,
        "attempt_counts": {}
    }
    
    # run specialists at same time
    logging.info(f"STAGE 2: Running specialist agents in parallel...")
    
    # kick off specialist tasks
    specialist_tasks = [
        Runner.run(team["financial_strategist"], 
                  f"Analyze the financial aspects of this product/company based on these facts:\n{facts}\n\nProduct description:\n{product_description}", 
                  context=context),
        Runner.run(team["market_research_specialist"], 
                  f"Analyze the market for this product/company based on these facts:\n{facts}\n\nProduct description:\n{product_description}", 
                  context=context),
        Runner.run(team["product_technical_advisor"], 
                  f"Analyze the product/technical aspects based on these facts:\n{facts}\n\nProduct description:\n{product_description}", 
                  context=context),
        Runner.run(team["shark_psychology_expert"], 
                  f"Analyze how to appeal to Shark Tank investors based on these facts:\n{facts}\n\nProduct description:\n{product_description}", 
                  context=context)
    ]
    
    logging.info(f"WAITING: Waiting for all specialist agents to complete their analysis...")
    specialist_results = await asyncio.gather(*specialist_tasks)
    logging.info(f"SPECIALISTS COMPLETE: All specialist agents have completed their analysis")
    
    # debug logging
    specialist_names = ["Financial Strategist", "Market Research Specialist", "Product/Technical Advisor", "Shark Psychology Expert"]
    specialist_outputs = []
    
    for i, result in enumerate(specialist_results):
        agent_name = specialist_names[i]
        logging.info(f"AGENT OUTPUT: [{agent_name}] provided the following response:")
        logging.info("-" * 80)
        logging.info(f"{result.final_output}")
        logging.info("-" * 80)
        
        try:
            parsed_output = parse_json_output(result.final_output)
            logging.info(f"PARSE SUCCESS: Successfully parsed {agent_name} output")
            specialist_outputs.append(parsed_output)
        except Exception as e:
            logging.error(f"PARSE ERROR: Failed to parse {agent_name} output: {str(e)}")
            specialist_outputs.append({"error": f"Failed to parse {agent_name} output", "raw_text": result.final_output})
        
        # check fact checking results
        if hasattr(result, 'output_guardrail_results') and result.output_guardrail_results:
            for guardrail_result in result.output_guardrail_results:
                if hasattr(guardrail_result, 'output_info') and guardrail_result.output_info:
                    if guardrail_result.output_info.get('yellow_flag', False):
                        attempt_count = guardrail_result.output_info.get('attempt_count', 0)
                        logging.warning(f"YELLOW FLAG: [{agent_name}] passed fact-checking after {attempt_count} attempts and may contain inaccuracies")
                        
                        corrections = guardrail_result.output_info.get('corrections', {})
                        if corrections:
                            logging.warning(f"CORRECTIONS SUGGESTED: {corrections}")
    
    # get first draft
    logging.info(f"STAGE 3: Running Pitch Drafter to create initial pitch...")
    drafter_input = f"""
    Create a compelling pitch for Shark Tank based on the following insights:
    
    FACTS:
    {facts}
    
    PRODUCT DESCRIPTION:
    {product_description}
    
    FINANCIAL STRATEGY:
    {json.dumps(specialist_outputs[0], indent=2)}
    
    MARKET RESEARCH:
    {json.dumps(specialist_outputs[1], indent=2)}
    
    PRODUCT INSIGHTS:
    {json.dumps(specialist_outputs[2], indent=2)}
    
    INVESTOR PSYCHOLOGY:
    {json.dumps(specialist_outputs[3], indent=2)}
    """
    
    logging.info(f"WAITING: Waiting for Pitch Drafter to complete...")
    draft_result = await Runner.run(team["pitch_drafter"], drafter_input, context=context)
    logging.info(f"DRAFTER COMPLETE: Pitch Drafter has completed the initial pitch")
    
    logging.info(f"AGENT OUTPUT: [Pitch Drafter] provided the following response:")
    logging.info("-" * 80)
    logging.info(f"{draft_result.final_output}")
    logging.info("-" * 80)
    
    try:
        draft_pitch = parse_json_output(draft_result.final_output)
        logging.info(f"PARSE SUCCESS: Successfully parsed Pitch Drafter output")
    except Exception as e:
        logging.error(f"PARSE ERROR: Failed to parse Pitch Drafter output: {str(e)}")
        draft_pitch = {"pitch": "Unable to parse draft pitch", "initial_offer": {"investment_amount": "$0", "equity_percentage": "0%"}}
    
    if hasattr(draft_result, 'output_guardrail_results') and draft_result.output_guardrail_results:
        for guardrail_result in draft_result.output_guardrail_results:
            if hasattr(guardrail_result, 'output_info') and guardrail_result.output_info:
                if guardrail_result.output_info.get('yellow_flag', False):
                    attempt_count = guardrail_result.output_info.get('attempt_count', 0)
                    logging.warning(f"YELLOW FLAG: [Pitch Drafter] passed fact-checking after {attempt_count} attempts and may contain inaccuracies")
                    
                    corrections = guardrail_result.output_info.get('corrections', {})
                    if corrections:
                        logging.warning(f"CORRECTIONS SUGGESTED: {corrections}")
    
    # get feedback on draft
    logging.info(f"STAGE 4: Running Pitch Critic to evaluate the draft pitch...")
    critic_input = f"""
    Critique the following pitch for Shark Tank:
    
    DRAFT PITCH:
    {json.dumps(draft_pitch, indent=2)}
    
    FACTS:
    {facts}
    
    PRODUCT DESCRIPTION:
    {product_description}
    
    FINANCIAL STRATEGY:
    {json.dumps(specialist_outputs[0], indent=2)}
    
    MARKET RESEARCH:
    {json.dumps(specialist_outputs[1], indent=2)}
    
    PRODUCT INSIGHTS:
    {json.dumps(specialist_outputs[2], indent=2)}
    
    INVESTOR PSYCHOLOGY:
    {json.dumps(specialist_outputs[3], indent=2)}
    """
    
    logging.info(f"WAITING: Waiting for Pitch Critic to complete evaluation...")
    critique_result = await Runner.run(team["pitch_critic"], critic_input, context=context)
    logging.info(f"CRITIC COMPLETE: Pitch Critic has completed the evaluation")
    
    logging.info(f"AGENT OUTPUT: [Pitch Critic] provided the following response:")
    logging.info("-" * 80)
    logging.info(f"{critique_result.final_output}")
    logging.info("-" * 80)
    
    try:
        critique = parse_json_output(critique_result.final_output)
        logging.info(f"PARSE SUCCESS: Successfully parsed Pitch Critic output")
    except Exception as e:
        logging.error(f"PARSE ERROR: Failed to parse Pitch Critic output: {str(e)}")
        critique = {"strengths": ["Unable to parse critique"], "weaknesses": ["Unknown"], "suggestions": ["Unknown"]}
    
    if hasattr(critique_result, 'output_guardrail_results') and critique_result.output_guardrail_results:
        for guardrail_result in critique_result.output_guardrail_results:
            if hasattr(guardrail_result, 'output_info') and guardrail_result.output_info:
                if guardrail_result.output_info.get('yellow_flag', False):
                    attempt_count = guardrail_result.output_info.get('attempt_count', 0)
                    logging.warning(f"YELLOW FLAG: [Pitch Critic] passed fact-checking after {attempt_count} attempts and may contain inaccuracies")
                    
                    corrections = guardrail_result.output_info.get('corrections', {})
                    if corrections:
                        logging.warning(f"CORRECTIONS SUGGESTED: {corrections}")
    
    # polish final version
    logging.info(f"STAGE 5: Running Pitch Finalizer to create the final pitch...")
    finalizer_input = f"""
    Create a final pitch for Shark Tank based on the draft and critique:
    
    DRAFT PITCH:
    {json.dumps(draft_pitch, indent=2)}
    
    CRITIQUE:
    {json.dumps(critique, indent=2)}
    
    FACTS:
    {facts}
    
    PRODUCT DESCRIPTION:
    {product_description}
    
    FINANCIAL STRATEGY:
    {json.dumps(specialist_outputs[0], indent=2)}
    
    MARKET RESEARCH:
    {json.dumps(specialist_outputs[1], indent=2)}
    
    PRODUCT INSIGHTS:
    {json.dumps(specialist_outputs[2], indent=2)}
    
    INVESTOR PSYCHOLOGY:
    {json.dumps(specialist_outputs[3], indent=2)}
    """
    
    logging.info(f"WAITING: Waiting for Pitch Finalizer to complete...")
    final_result = await Runner.run(team["pitch_finalizer"], finalizer_input, context=context)
    logging.info(f"FINALIZER COMPLETE: Pitch Finalizer has completed the final pitch")
    
    logging.info(f"AGENT OUTPUT: [Pitch Finalizer] provided the following response:")
    logging.info("-" * 80)
    logging.info(f"{final_result.final_output}")
    logging.info("-" * 80)
    
    try:
        final_pitch = parse_json_output(final_result.final_output)
        logging.info(f"PARSE SUCCESS: Successfully parsed Pitch Finalizer output")
    except Exception as e:
        logging.error(f"PARSE ERROR: Failed to parse Pitch Finalizer output: {str(e)}")
        final_pitch = {"pitch": "Unable to parse final pitch", "initial_offer": draft_pitch.get("initial_offer", {"investment_amount": "$0", "equity_percentage": "0%"})}
    
    if hasattr(final_result, 'output_guardrail_results') and final_result.output_guardrail_results:
        for guardrail_result in final_result.output_guardrail_results:
            if hasattr(guardrail_result, 'output_info') and guardrail_result.output_info:
                if guardrail_result.output_info.get('yellow_flag', False):
                    attempt_count = guardrail_result.output_info.get('attempt_count', 0)
                    logging.warning(f"YELLOW FLAG: [Pitch Finalizer] passed fact-checking after {attempt_count} attempts and may contain inaccuracies")
                    
                    corrections = guardrail_result.output_info.get('corrections', {})
                    if corrections:
                        logging.warning(f"CORRECTIONS SUGGESTED: {corrections}")
    
    logging.info(f"PROCESS COMPLETE: Successfully generated pitch for '{product_name}'")
    
    return {
        "financial_strategy": specialist_outputs[0],
        "market_research": specialist_outputs[1],
        "product_insights": specialist_outputs[2],
        "investor_psychology": specialist_outputs[3],
        "draft_pitch": draft_pitch,
        "critique": critique,
        "final_pitch": final_pitch
    }

def save_pitch_to_file(pitch_result: Dict[str, Any], output_file: str) -> None:
    """
    save pitch output to json file with timestamp
    
    args:
        pitch_result: pitch data to save
        output_file: where to save it
    """
    # add timestamp to avoid overwriting files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = output_file.rsplit('.', 1)
    if len(filename_parts) > 1:
        timestamped_output_file = f"{filename_parts[0]}_{timestamp}.{filename_parts[1]}"
    else:
        timestamped_output_file = f"{output_file}_{timestamp}"
    
    # create output dir if needed
    os.makedirs(os.path.dirname(timestamped_output_file), exist_ok=True)
    
    logging.info(f"SAVING RESULT: Saving generated pitch to {timestamped_output_file}")
    with open(timestamped_output_file, 'w', encoding='utf-8') as f:
        json.dump(pitch_result, f, indent=2, ensure_ascii=False)
    logging.info(f"SAVE COMPLETE: Generated pitch saved successfully to {timestamped_output_file}")
    
    # check if we got a final pitch
    if not pitch_result.get("final_pitch", {}).get("pitch"):
        logging.warning("WARNING: Final pitch not found in the result")
        print("\n=== PITCH GENERATION INCOMPLETE ===\n")
        print("The pitch generation process did not complete successfully.")
        print("Check the logs for more information.")
    else:
        print("\n=== PITCH GENERATION COMPLETE ===\n")
        print(f"The pitch has been generated and saved to {timestamped_output_file}")
        print("\nFINAL PITCH:")
        print("------------")
        print(pitch_result["final_pitch"]["pitch"])
        print("\nINITIAL OFFER:")
        print("-------------")
        print(json.dumps(pitch_result["final_pitch"]["initial_offer"], indent=2))

async def process_episode(episode_key: str, facts_data: Dict[str, Any], model: str = "gpt-4o") -> Dict[str, Any]:
    """
    process one sharktank episode to make a pitch
    
    args:
        episode_key: which episode to process
        facts_data: all episode data
        model: which openai model to use
    
    returns:
        dict: processing results with generated pitch
    """
    if episode_key not in facts_data:
        logging.error(f"ERROR: Episode {episode_key} not found in data")
        return {"error": f"Episode {episode_key} not found in data"}
    
    episode_data = facts_data[episode_key]
    facts = episode_data.get("facts", {})
    product_description = episode_data.get("product_description", {})
    
    product_name = product_description.get('name', 'Unknown')
    logging.info(f"EPISODE: Processing episode: {episode_key}")
    logging.info(f"PRODUCT: {product_name}")
    
    pitch_result = await generate_pitch(facts, product_description, model=model)
    
    logging.info(f"EPISODE COMPLETE: Successfully processed episode {episode_key} for product {product_name}")
    
    return {
        "episode_key": episode_key,
        "original_data": {
            "facts": facts,
            "product_description": product_description
        },
        "generated_pitch": pitch_result
    }

async def main():
    # get episode data
    data_path = Path("../data/all_processed_facts.json")
    logging.info(f"LOADING DATA: Loading episode data from {data_path}")
    with data_path.open("r", encoding="utf-8") as f:
        facts_data = json.load(f)
    logging.info(f"DATA LOADED: Successfully loaded {len(facts_data)} episodes")
    
    # test with first episode
    test_episode_key = list(facts_data.keys())[0]
    logging.info(f"TEST RUN: Processing test episode {test_episode_key}")
    result = await process_episode(test_episode_key, facts_data)
    
    # save with timestamp
    output_path = Path("./output")
    output_path.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"{test_episode_key}_pitch_{timestamp}.json"
    logging.info(f"SAVING RESULT: Saving generated pitch to {output_file}")
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    logging.info(f"SAVE COMPLETE: Generated pitch saved successfully")
    
    print("\n=== GENERATED PITCH ===\n")
    print(result["generated_pitch"]["final_pitch"]["pitch"])
    print("\n=== INITIAL OFFER ===\n")
    print(json.dumps(result["generated_pitch"]["final_pitch"]["initial_offer"], indent=2))
    
    logging.info(f"PROCESS COMPLETE: All tasks completed successfully")

if __name__ == "__main__":
    asyncio.run(main()) 