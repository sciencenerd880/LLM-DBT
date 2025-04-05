from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

# Define data models for structured outputs
class FactCheckOutput(BaseModel):
    is_accurate: bool = Field(..., description="Whether the information is factually accurate")
    corrections: Optional[Dict[str, str]] = Field(None, description="Corrections to any inaccurate information")
    yellow_flag: bool = Field(False, description="Whether the output was passed with a yellow flag")
    attempt_count: int = Field(0, description="Number of attempts made to generate an accurate response")
    
    model_config = {
        "json_schema_extra": {
            "required": ["is_accurate"]
        }
    }

class FinancialStrategyOutput(BaseModel):
    valuation_justification: str = Field(..., description="Reasoning behind the company valuation")
    investment_amount: str = Field(..., description="Amount of money requested")
    equity_percentage: str = Field(..., description="Percentage of company offered")
    use_of_funds_breakdown: Dict[str, float] = Field(..., description="Breakdown of how funds will be used")
    roi_timeline: str = Field(..., description="Expected timeline for return on investment")
    exit_strategy: Optional[str] = Field(None, description="Potential exit opportunities")
    
    model_config = {
        "json_schema_extra": {
            "required": ["valuation_justification", "investment_amount", "equity_percentage", 
                         "use_of_funds_breakdown", "roi_timeline"]
        }
    }

class CompetitorInfo(BaseModel):
    name: str = Field(..., description="Competitor name")
    strengths: str = Field(..., description="Their strengths")
    weaknesses: str = Field(..., description="Their weaknesses")
    
    model_config = {
        "json_schema_extra": {
            "required": ["name", "strengths", "weaknesses"]
        }
    }

class MarketResearchOutput(BaseModel):
    market_size: str = Field(..., description="Estimated size of the target market")
    target_customers: str = Field(..., description="Description of the target customer segments")
    competitors: List[CompetitorInfo] = Field(..., description="Analysis of key competitors")
    market_trends: List[str] = Field(..., description="Current trends in the market")
    growth_opportunities: List[str] = Field(..., description="Potential growth opportunities")
    market_challenges: List[str] = Field(..., description="Challenges in the market")
    
    model_config = {
        "json_schema_extra": {
            "required": ["market_size", "target_customers", "competitors", 
                         "market_trends", "growth_opportunities", "market_challenges"]
        }
    }

class ProductTechnicalOutput(BaseModel):
    key_features: List[str] = Field(..., description="Key features of the product")
    technical_advantages: List[str] = Field(..., description="Technical advantages over alternatives")
    demonstration_strategy: str = Field(..., description="How to effectively demonstrate the product")
    scalability: str = Field(..., description="Assessment of production/technical scalability")
    future_development: List[str] = Field(..., description="Planned future developments")
    
    model_config = {
        "json_schema_extra": {
            "required": ["key_features", "technical_advantages", "demonstration_strategy", 
                         "scalability", "future_development"]
        }
    }

class Objection(BaseModel):
    objection: str = Field(..., description="Potential objection from Sharks")
    counter: str = Field(..., description="Counter-argument to the objection")
    
    model_config = {
        "json_schema_extra": {
            "required": ["objection", "counter"]
        }
    }

class SharkFit(BaseModel):
    name: str = Field(..., description="Shark name")
    reason: str = Field(..., description="Why they're a good fit")
    
    model_config = {
        "json_schema_extra": {
            "required": ["name", "reason"]
        }
    }

class SharkPsychologyOutput(BaseModel):
    key_appeal_points: List[str] = Field(..., description="Points that will appeal to Sharks")
    potential_objections: List[Objection] = Field(..., description="Potential objections and counters")
    negotiation_strategy: str = Field(..., description="Strategy for negotiating with Sharks")
    presentation_tips: List[str] = Field(..., description="Tips for effective presentation")
    best_fit_sharks: List[SharkFit] = Field(..., description="Sharks that are the best fit")
    
    model_config = {
        "json_schema_extra": {
            "required": ["key_appeal_points", "potential_objections", "negotiation_strategy", 
                         "presentation_tips", "best_fit_sharks"]
        }
    }

class InitialOffer(BaseModel):
    investment_amount: str = Field(..., description="Amount of investment requested")
    equity_percentage: str = Field(..., description="Percentage of equity offered")
    valuation: str = Field(..., description="Implied valuation")
    
    model_config = {
        "json_schema_extra": {
            "required": ["investment_amount", "equity_percentage", "valuation"]
        }
    }

class PitchDraftOutput(BaseModel):
    draft_pitch: str = Field(..., description="The complete draft pitch script")
    draft_initial_offer: InitialOffer = Field(..., description="The initial offer to present")
    
    model_config = {
        "json_schema_extra": {
            "required": ["draft_pitch", "draft_initial_offer"]
        }
    }

class PitchCritiqueOutput(BaseModel):
    strengths: List[str] = Field(..., description="Strengths of the pitch")
    weaknesses: List[str] = Field(..., description="Weaknesses of the pitch")
    suggestions: List[str] = Field(..., description="Suggestions for improvement")
    overall_assessment: str = Field(..., description="Summary evaluation of the pitch")
    
    model_config = {
        "json_schema_extra": {
            "required": ["strengths", "weaknesses", "suggestions", "overall_assessment"]
        }
    }

class FinalPitchOutput(BaseModel):
    pitch: str = Field(..., description="The complete final pitch script")
    initial_offer: InitialOffer = Field(..., description="The initial offer to present")
    key_points_to_emphasize: List[str] = Field(..., description="Key points to emphasize")
    anticipated_questions: List[str] = Field(..., description="Questions to anticipate from Sharks")
    
    model_config = {
        "json_schema_extra": {
            "required": ["pitch", "initial_offer", "key_points_to_emphasize", "anticipated_questions"]
        }
    }

class PitchResult(BaseModel):
    final_pitch: FinalPitchOutput = Field(..., description="The final pitch")
    specialist_inputs: Dict[str, Union[FinancialStrategyOutput, MarketResearchOutput, 
                                      ProductTechnicalOutput, SharkPsychologyOutput]] = Field(
        ..., description="Inputs from specialist agents"
    )
    
    model_config = {
        "json_schema_extra": {
            "required": ["final_pitch", "specialist_inputs"]
        }
    } 