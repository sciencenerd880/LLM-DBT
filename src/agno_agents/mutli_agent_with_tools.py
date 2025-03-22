from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime

from agno.agent import Agent
from agno.models.groq import Groq

from textwrap import dedent
from pydantic import BaseModel, Field

from utils import ValuationTools, extract_metrics
from agno.tools.calculator import CalculatorTools

from dotenv import load_dotenv
import os

# ======================================================== SETUP ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# === Pydantic Base Models ===
class InitialOffer(BaseModel):
    Valuation: str = Field(..., description="The company's estimated valuation in dollars (e.g., $10 million).")
    Equity_Offered: str = Field(..., description="The percentage of equity offered to investors (e.g., 10%).")
    Funding_Amount: str = Field(..., description="The amount of funding requested (e.g., $1 million).")
    Key_Terms: str = Field(..., description="Additional key terms about how the funding will be used and expected returns (optional).")

class PitchResponse(BaseModel):
    Pitch: str = Field(..., description="The well-structured investment pitch highlighting the unique value proposition, market opportunity, and call to action.")
    Initial_Offer: InitialOffer = Field(..., description="Details of the investment proposal, including valuation, equity, and funding amount.")

# === Agent Team in Collaborate Mode ===
def create_mas_coordinate_l0(model_id="deepseek-r1-distill-llama-70b"):
    # Create a pitch master
    pitch_master = Agent(
        name="Pitch Master",
        model=Groq(model_id),
        response_model=PitchResponse,
        markdown=False,
        description="You are a highly skilled investment consultant specializing in startup fundraising.",
        instructions=dedent("""
        ### Your Task:
        1. Write a **persuasive startup pitch** that effectively highlights:
           - The product’s unique value proposition
           - The market opportunity and competitive edge
           - The potential for growth and profitability
           - A call to action for investors

        2. Propose an **initial offer to investors** that includes:
           - A valuation justification
           - The amount of funding requested
           - An appropriate equity offer
           - Key investment terms (e.g., use of funds, investor value add)

        3. Return a well-structured response in valid JSON format.
         """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
    )

    financial_analyst = Agent(
        name="Financial Analyst",
        role="Analyze startup financials and compute valuation",
        model=Groq(model_id),
        tools=[CalculatorTools(add=True, subtract=True, multiply=True, divide=True),
                ValuationTools(),  # Custom tool from utils
                ],
        instructions=dedent("""
                You are a financial analyst for early-stage startups.

                Step 1: Extract relevant financial numbers from the prompt: revenue, profit, growth rate, etc.
                Step 2: Choose the most appropriate valuation method based on the available data:
                    - Use revenue_multiple_valuation if only revenue is known.
                    - Use profit_multiple_valuation if profit is available.
                    - Use discounted_cash_flow_valuation if profit and growth rate are available.
                Step 3: Use the tools to compute estimated valuation.
                Step 4: Evaluate if the initial funding offer (e.g., $100,000 for 10%) is fair compared to the estimated valuation.
                Step 5: Recommend whether to accept or adjust the offer.

                Always reason step-by-step, then call the tool with the right values.
            """),
        show_tool_calls=True,
        markdown=True,
        )

    # Create a collaborater team 
    startup_pitch_team = Agent(
        name="SharkTank Pitch Collaborator / CEO",
        model=Groq(model_id),
        team=[financial_analyst,pitch_master],
        instructions=[dedent("""\
        You are the CEO that leads the early start up! 

        Your role:
        1. Coordinate between the pitch master and financial analyst
        2. Combine their findings into a compelling pitch to ask for funding
        3. Stop when the team agrees on a final investor pitch and recommendation.
        4. Present a balanced view of both news and data
        5. Highlight key risks and opportunities.
        """)],
        markdown=True,
        # debug_mode=True,
    )
    return startup_pitch_team

# ======================================================== MAIN ========================================================
if __name__ == "__main__":
    with Path("src/agno_agents/data/outputs/facts_and_productdescriptions.json").open("r", encoding="utf-8") as f:
        facts_dict = json.loads(f.read())

    scenarios = list(facts_dict.keys())[:1]

    # Self defined
    provider = "groq"
    model_id = "deepseek-r1-distill-llama-70b"
    model_name = f"{provider}/{model_id}"
    startup_team = create_mas_coordinate_l0()

    for scenario in tqdm(scenarios, desc="Processing Pitches for Sharks", unit="pitch"):

        # Get the facts & product description of that scenario
        facts_json = json.dumps(facts_dict[scenario]["facts"], indent=2)
        product_des_json = json.dumps(facts_dict[scenario]["product_description"], indent=2)
        
        # Get time stamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Get the metrices after the start up team run the prompt
        prompt = f"Here is the startup's description and facts:\n{facts_json}\n{product_des_json}:\n"
        response = startup_team.run(prompt)
        metrices = extract_metrics(response)

        print(metrices)

        print("*"*100)



