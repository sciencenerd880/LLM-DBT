from typing import Dict, Union
from agno.tools import Toolkit
import math
from pydantic import BaseModel, Field

def extract_metrics(response):
    metrics = response.metrics
    return {
        "latency": metrics.get("additional_metrics", [{}])[0].get("completion_time", None),
        "input_length": metrics.get("input_tokens", [0])[0],
        "output_length": metrics.get("output_tokens", [0])[0],
        "response": response.content#.model_dump()  
    }

# Define the structure for the investment offer
class InitialOffer(BaseModel):
    Valuation: str = Field(..., description="The company's estimated valuation in dollars (e.g., $10 million).")
    Equity_Offered: str = Field(..., description="The percentage of equity offered to investors (e.g., 10%).")
    Funding_Amount: str = Field(..., description="The amount of funding requested (e.g., $1 million).")
    Key_Terms: str = Field(..., description="Additional key terms about how the funding will be used and expected returns (optional).")

# Define the structure for the Shark Tank pitch response
class PitchResponse(BaseModel):
    Pitch: str = Field(..., description="The well-structured investment pitch highlighting the unique value proposition, market opportunity, and call to action.")
    Initial_Offer: InitialOffer = Field(..., description="Details of the investment proposal, including valuation, equity, and funding amount.")


class ValuationTools(Toolkit):
    def __init__(self):
        super().__init__(name="valuation_tools")
        self.register(self.revenue_multiple_valuation)
        self.register(self.profit_multiple_valuation)
        self.register(self.discounted_cash_flow_valuation)

    def revenue_multiple_valuation(self, revenue: float, multiple: Union[float, int] = 2) -> str:
        valuation = revenue * multiple
        return (
            f"Valuation Method: Revenue Multiple\n"
            f"Input: Revenue = ${revenue:,.2f}, Multiple = {multiple}\n"
            f"Valuation: ${valuation:,.2f}\n"
            f"Explanation: The business is valued at {multiple}x its revenue."
        )

    def profit_multiple_valuation(self, profit: float, multiple: Union[float, int] = 8) -> str:
        valuation = profit * multiple
        return (
            f"Valuation Method: Profit Multiple\n"
            f"Input: Profit = ${profit:,.2f}, Multiple = {multiple}\n"
            f"Valuation: ${valuation:,.2f}\n"
            f"Explanation: The business is valued at {multiple}x its profit."
        )

    def discounted_cash_flow_valuation(
        self,
        base_profit: float,
        growth_rate: float = 0.2,
        years: int = 5
    ) -> str:
        valuation = 0.0
        discount_rate = 0.4
        explanation = []
        for t in range(1, years + 1):
            projected = base_profit * ((1 + growth_rate) ** t)
            discounted = projected / ((1 + discount_rate) ** t)
            valuation += discounted
            explanation.append(f"Year {t}: Projected = ${projected:,.2f}, Discounted = ${discounted:,.2f}")
        breakdown = "\\n".join(explanation)
        return (
            f"Valuation Method: Discounted Cash Flow (DCF)\n"
            f"Input: Base Profit = ${base_profit:,.2f}, Growth Rate = {growth_rate*100:.0f}%, Years = {years}\n"
            f"Valuation: ${valuation:,.2f}\n"
            f"Breakdown:\n{breakdown}"
        )