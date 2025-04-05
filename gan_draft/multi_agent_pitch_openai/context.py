from typing import Dict, Any, Optional
from agents import AgentHooks, RunContextWrapper, Agent

from models import (
    FinancialStrategyOutput,
    MarketResearchOutput,
    ProductTechnicalOutput,
    SharkPsychologyOutput,
    PitchDraftOutput,
    PitchCritiqueOutput,
    FinalPitchOutput
)

# Context class to store shared data between agents
class PitchContext:
    def __init__(self, facts: Dict[str, Any], product_description: Dict[str, Any]):
        self.facts = facts
        self.product_description = product_description
        self.financial_strategy: Optional[FinancialStrategyOutput] = None
        self.market_research: Optional[MarketResearchOutput] = None
        self.product_technical: Optional[ProductTechnicalOutput] = None
        self.shark_psychology: Optional[SharkPsychologyOutput] = None
        self.pitch_draft: Optional[PitchDraftOutput] = None
        self.pitch_critique: Optional[PitchCritiqueOutput] = None
        self.final_pitch: Optional[FinalPitchOutput] = None
        self.specialist_count_completed = 0
        self.max_attempts = 3
        self.attempt_counts = {
            "financial_strategy": 0,
            "market_research": 0,
            "product_technical": 0,
            "shark_psychology": 0,
            "pitch_draft": 0,
            "pitch_critique": 0,
            "final_pitch": 0
        }

# Custom hooks for lifecycle events
class PitchTeamHooks(AgentHooks):
    async def on_agent_run_start(self, ctx: RunContextWrapper, agent: Agent) -> None:
        print(f"Starting agent: {agent.name}")
    
    async def on_agent_run_end(self, ctx: RunContextWrapper, agent: Agent) -> None:
        print(f"Completed agent: {agent.name}")
        
        # Track specialist completion
        if agent.name in ["Financial Strategist", "Market Research Specialist", 
                         "Product/Technical Advisor", "Shark Psychology Expert"]:
            ctx.context.specialist_count_completed += 1
            print(f"Specialists completed: {ctx.context.specialist_count_completed}/4") 