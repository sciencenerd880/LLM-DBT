from agents import Agent, output_guardrail
from guardrails import fact_check_guardrail

from prompts import (
    FINANCIAL_STRATEGIST_PROMPT,
    MARKET_RESEARCH_SPECIALIST_PROMPT,
    PRODUCT_TECHNICAL_ADVISOR_PROMPT,
    SHARK_PSYCHOLOGY_EXPERT_PROMPT,
    PITCH_DRAFTER_PROMPT,
    PITCH_CRITIC_PROMPT,
    PITCH_FINALIZER_PROMPT,
    PITCH_ORCHESTRATOR_PROMPT,
    FINANCIAL_FACT_CHECKER_PROMPT,
    MARKET_FACT_CHECKER_PROMPT,
    PRODUCT_FACT_CHECKER_PROMPT,
    PSYCHOLOGY_FACT_CHECKER_PROMPT,
    DRAFT_FACT_CHECKER_PROMPT,
    CRITIQUE_FACT_CHECKER_PROMPT,
    FINAL_FACT_CHECKER_PROMPT
)

from context import PitchTeamHooks

# Define the agents with their specific roles
def create_pitch_team(model="gpt-4o"):
    """
    Create a team of specialized agents for pitch development.
    
    Args:
        model (str): The OpenAI model to use for all agents. Defaults to "gpt-4o".
    
    Returns:
        dict: A dictionary of agents with their roles as keys.
    """
    # Define the guardrail functions for each specialist
    @output_guardrail
    async def financial_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, financial_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def market_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, market_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def product_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, product_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def psychology_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, investor_psychology_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def draft_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, draft_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def critique_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, critique_fact_checker, max_attempts=3)
    
    @output_guardrail
    async def final_fact_check_guardrail(ctx, agent, output):
        return await fact_check_guardrail(ctx, agent, output, final_fact_checker, max_attempts=3)
    
    # Create the fact checker agents
    financial_fact_checker = Agent(
        name="Financial Fact Checker",
        instructions=FINANCIAL_FACT_CHECKER_PROMPT,
        model=model
    )
    
    market_fact_checker = Agent(
        name="Market Research Fact Checker",
        instructions=MARKET_FACT_CHECKER_PROMPT,
        model=model
    )
    
    product_fact_checker = Agent(
        name="Product Fact Checker",
        instructions=PRODUCT_FACT_CHECKER_PROMPT,
        model=model
    )
    
    investor_psychology_fact_checker = Agent(
        name="Investor Psychology Fact Checker",
        instructions=PSYCHOLOGY_FACT_CHECKER_PROMPT,
        model=model
    )
    
    draft_fact_checker = Agent(
        name="Draft Fact Checker",
        instructions=DRAFT_FACT_CHECKER_PROMPT,
        model=model
    )
    
    critique_fact_checker = Agent(
        name="Critique Fact Checker",
        instructions=CRITIQUE_FACT_CHECKER_PROMPT,
        model=model
    )
    
    final_fact_checker = Agent(
        name="Final Fact Checker",
        instructions=FINAL_FACT_CHECKER_PROMPT,
        model=model
    )
    
    # Create the specialist agents with guardrails
    financial_strategist = Agent(
        name="Financial Strategist",
        instructions=FINANCIAL_STRATEGIST_PROMPT,
        model=model,
        output_guardrails=[financial_fact_check_guardrail]
    )
    
    market_research_specialist = Agent(
        name="Market Research Specialist",
        instructions=MARKET_RESEARCH_SPECIALIST_PROMPT,
        model=model,
        output_guardrails=[market_fact_check_guardrail]
    )
    
    product_technical_advisor = Agent(
        name="Product/Technical Advisor",
        instructions=PRODUCT_TECHNICAL_ADVISOR_PROMPT,
        model=model,
        output_guardrails=[product_fact_check_guardrail]
    )
    
    shark_psychology_expert = Agent(
        name="Shark Psychology Expert",
        instructions=SHARK_PSYCHOLOGY_EXPERT_PROMPT,
        model=model,
        output_guardrails=[psychology_fact_check_guardrail]
    )
    
    # Create the pitch development agents with guardrails
    pitch_drafter = Agent(
        name="Pitch Drafter",
        instructions=PITCH_DRAFTER_PROMPT,
        model=model,
        output_guardrails=[draft_fact_check_guardrail]
    )
    
    pitch_critic = Agent(
        name="Pitch Critic",
        instructions=PITCH_CRITIC_PROMPT,
        model=model,
        output_guardrails=[critique_fact_check_guardrail]
    )
    
    pitch_finalizer = Agent(
        name="Pitch Finalizer",
        instructions=PITCH_FINALIZER_PROMPT,
        model=model,
        output_guardrails=[final_fact_check_guardrail]
    )
    
    # Return the team as a dic
    return {
        "financial_strategist": financial_strategist,
        "market_research_specialist": market_research_specialist,
        "product_technical_advisor": product_technical_advisor,
        "shark_psychology_expert": shark_psychology_expert,
        "pitch_drafter": pitch_drafter,
        "pitch_critic": pitch_critic,
        "pitch_finalizer": pitch_finalizer,
        "financial_fact_checker": financial_fact_checker,
        "market_fact_checker": market_fact_checker,
        "product_fact_checker": product_fact_checker,
        "investor_psychology_fact_checker": investor_psychology_fact_checker,
        "draft_fact_checker": draft_fact_checker,
        "critique_fact_checker": critique_fact_checker,
        "final_fact_checker": final_fact_checker
    } 