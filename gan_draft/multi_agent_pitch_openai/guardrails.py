import logging
import os
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel
import re

from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner
)

class FactCheckOutput(BaseModel):
    is_accurate: bool
    corrections: Optional[Dict[str, str]] = None
    yellow_flag: bool = False
    attempt_count: int = 0

# Global dictionary to track attempts across the entire process
GLOBAL_ATTEMPT_COUNTS = {}

async def fact_check_guardrail(
    ctx: RunContextWrapper, 
    agent: Agent, 
    output: Any,
    checker_agent: Agent,
    max_attempts: int = 3,
    facts: str = None,
    product_description: Any = None
) -> GuardrailFunctionOutput:
    """
    checks if agent's output is factually accurate and retries up to max_attempts if not
    
    args:
        ctx: run context
        agent: agent that produced the output 
        output: output to check
        checker_agent: agent that checks the output
        max_attempts: max total attempts allowed (default: 3)
        facts: facts to use for checking (passed in recursive calls)
        product_description: product description to use (passed in recursive calls)
        
    returns:
        GuardrailFunctionOutput: fact check result
    """
    # generate unique id for this agent
    agent_id = f"{agent.name}"
    
    # get current attempt count, starting at 1 for new agents
    if agent_id not in GLOBAL_ATTEMPT_COUNTS:
        GLOBAL_ATTEMPT_COUNTS[agent_id] = 1
    current_attempt = GLOBAL_ATTEMPT_COUNTS[agent_id]
    
    logging.info(f"ATTEMPT TRACKER: Agent [{agent_id}] is on attempt {current_attempt} of {max_attempts}")
    
    # immediately pass with yellow flag if max attempts reached
    if current_attempt > max_attempts:
        logging.warning(f"MAX ATTEMPTS EXCEEDED: Agent [{agent_id}] has already used {current_attempt-1} attempts, forcing pass with yellow flag")
        return GuardrailFunctionOutput(
            output_info={
                "is_accurate": True,
                "corrections": {},
                "yellow_flag": True,
                "attempt_count": current_attempt-1
            },
            tripwire_triggered=False
        )
    
    # get facts and product_description if not provided
    if facts is None or product_description is None:
        try:
            context_dict = getattr(ctx, 'context', {}) or {}
            
            facts = facts or context_dict.get("facts", None)
            product_description = product_description or context_dict.get("product_description", None)
            
            # try env vars as fallback
            if facts is None and "PITCH_FACTS" in os.environ:
                env_facts = os.environ.get("PITCH_FACTS")
                try:
                    facts = json.loads(env_facts)
                except json.JSONDecodeError:
                    facts = env_facts
                logging.debug(f"Retrieved facts from environment variable for {agent.name}")
                
            if product_description is None and "PITCH_PRODUCT_DESCRIPTION" in os.environ:
                env_product = os.environ.get("PITCH_PRODUCT_DESCRIPTION")
                try:
                    product_description = json.loads(env_product)
                except json.JSONDecodeError:
                    product_description = env_product
                logging.debug(f"Retrieved product_description from environment variable for {agent.name}")
            
            # use defaults if still none
            facts = facts or "No facts provided"
            product_description = product_description or "No product description provided"
            
            logging.debug(f"Extracted facts and product description for {agent.name}")
        except Exception as e:
            logging.warning(f"Error extracting facts from context: {e}. Using default values.")
            facts = facts or "No facts provided"
            product_description = product_description or "No product description provided"
    
    logging.info(f"FACT CHECK: Starting fact check for [{agent.name}] using [{checker_agent.name}]")
    logging.info(f"FACT CHECK: [ATTEMPT {current_attempt}/{max_attempts}] [{checker_agent.name}] checking [{agent.name}]'s output")
    
    fact_check_result_obj = await Runner.run(
        checker_agent,
        f"""
        Please fact check the following output from {agent.name}:
        
        {output}
        
        Verify that all statements are accurate based on the facts provided.
        
        FACTS:
        {facts}
        
        PRODUCT DESCRIPTION:
        {product_description}
        
        Your task is to ensure that the output does not contradict the facts or make claims that aren't supported by the facts.
        """
    )
    
    fact_check_result = fact_check_result_obj.final_output
    
    try:
        fact_check_output = parse_fact_check_output(fact_check_result)
    except Exception as e:
        logging.error(f"Error parsing fact check output: {e}")
        # default to passing if parsing fails
        fact_check_output = {
            "is_accurate": True,
            "corrections": {}
        }
    
    if fact_check_output.get("is_accurate", False):
        logging.info(f"FACT CHECK PASSED: [ATTEMPT {current_attempt}/{max_attempts}] [{checker_agent.name}] verified [{agent.name}]'s output as accurate")
        
        fact_check_output["attempt_count"] = current_attempt
        
        return GuardrailFunctionOutput(
            output_info=fact_check_output,
            tripwire_triggered=False
        )
    else:
        corrections = fact_check_output.get("corrections", {})
        
        logging.warning(f"FACT CHECK FAILED: [ATTEMPT {current_attempt}/{max_attempts}] [{checker_agent.name}] found inaccuracies in [{agent.name}]'s output")
        
        if not corrections or not isinstance(corrections, dict) or all(not v for v in corrections.values()):
            logging.warning("INVALID CORRECTIONS: Corrections are empty or invalid")
            corrections = {"The output contains inaccuracies that need to be addressed": "Please review the facts and ensure your response is accurate."}
        
        logging.warning(f"CORRECTIONS: [{checker_agent.name}] suggested these corrections for [{agent.name}]: {corrections}")
        
        if current_attempt >= max_attempts:
            logging.warning(f"MAX ATTEMPTS REACHED: Forcing pass with yellow flag for [{agent.name}] after {max_attempts} attempts")
            
            return GuardrailFunctionOutput(
                output_info={
                    "is_accurate": True,
                    "corrections": corrections,
                    "yellow_flag": True,
                    "attempt_count": current_attempt
                },
                tripwire_triggered=False
            )
        
        GLOBAL_ATTEMPT_COUNTS[agent_id] = current_attempt + 1
        next_attempt = current_attempt + 1
        
        logging.info(f"RETRY: [ATTEMPT {next_attempt}/{max_attempts}] Requesting [{agent.name}] to regenerate response")
        
        regenerate_prompt = f"""
        Your previous response contained some inaccuracies. This is attempt {next_attempt} of {max_attempts}.
        
        Please address the following issues and regenerate your response:
        
        {json.dumps(corrections, indent=2)}
        
        FACTS:
        {facts}
        
        PRODUCT DESCRIPTION:
        {product_description}
        
        If there are multiple issues, focus on the most important ones first.
        """
        
        regenerated_result = await Runner.run(agent, regenerate_prompt)
        regenerated_output = regenerated_result.final_output
        
        logging.info(f"REGENERATED RESPONSE: [{agent.name}] provided a new response (Attempt {next_attempt}/{max_attempts})")
        
        return await fact_check_guardrail(
            ctx, 
            agent, 
            regenerated_output, 
            checker_agent, 
            max_attempts=max_attempts,
            facts=facts,
            product_description=product_description
        )

def parse_fact_check_output(fact_check_result):
    """
    parses fact check result to get is_accurate flag and corrections
    
    args:
        fact_check_result: output from fact checker agent
        
    returns:
        dict with is_accurate and corrections fields
    """
    # try json first
    try:
        result_json = json.loads(fact_check_result)
        if isinstance(result_json, dict):
            if "is_accurate" in result_json:
                return {
                    "is_accurate": result_json.get("is_accurate", False),
                    "corrections": result_json.get("corrections", {})
                }
    except json.JSONDecodeError:
        pass
    
    # fallback to text parsing
    try:
        is_accurate = False
        if re.search(r'\b(accurate|correct|factual|valid|true|passes|pass)\b', fact_check_result.lower()):
            if not re.search(r'\b(not|isn\'t|aren\'t|doesn\'t|don\'t|no|false|fail|fails|inaccurate|incorrect)\s+\b(accurate|correct|factual|valid|true|passes|pass)\b', fact_check_result.lower()):
                is_accurate = True
        
        corrections = {}
        if not is_accurate:
            correction_sections = re.findall(r'(?:correction|inaccuracy|issue|problem|error)[s]?:?\s*(.*?)(?:\n\n|\Z)', fact_check_result, re.DOTALL | re.IGNORECASE)
            
            if correction_sections:
                for section in correction_sections:
                    pairs = re.findall(r'"([^"]+)"\s*:\s*"([^"]+)"', section)
                    if pairs:
                        for k, v in pairs:
                            corrections[k.strip()] = v.strip()
                    else:
                        if section.strip():
                            corrections["General correction"] = section.strip()
        
        return {
            "is_accurate": is_accurate,
            "corrections": corrections
        }
    except Exception as e:
        logging.error(f"Error parsing fact check output text: {e}")
        return {
            "is_accurate": False,
            "corrections": {"General error": "Could not parse fact check output"}
        } 