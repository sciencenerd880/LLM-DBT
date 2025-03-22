from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Dict

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from dotenv import load_dotenv
import os

from pathlib import Path
import json
import pandas as pd

from datetime import datetime
from tqdm import tqdm  # for progress bar

from utils import ValuationTools
from financialAgent import create_financial_agent

# ======================================================== START: TO SET THE VARIABLES  ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class InitialOffer(BaseModel):
    Valuation: str = Field(..., description="The company's estimated valuation in dollars (e.g., $10 million).")
    Equity_Offered: str = Field(..., description="The percentage of equity offered to investors (e.g., 10%).")
    Funding_Amount: str = Field(..., description="The amount of funding requested (e.g., $1 million).")
    Key_Terms: str = Field(..., description="Additional key terms about how the funding will be used and expected returns (optional).")

class PitchResponse(BaseModel):
    Pitch: str = Field(..., description="The well-structured investment pitch highlighting the unique value proposition, market opportunity, and call to action.")
    Initial_Offer: InitialOffer = Field(..., description="Details of the investment proposal, including valuation, equity, and funding amount.")

MODEL_PROVIDERS = {
    "openai": OpenAIChat,
    "groq": Groq,
}

# ======================================================== AGENT CREATION ========================================================
def create_pitch_agent(provider: str = "groq", model_name: str = "deepseek-r1-distill-llama-70b", instructions_override=None):
    if provider not in MODEL_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Choose from: {list(MODEL_PROVIDERS.keys())}")

    model_class = MODEL_PROVIDERS[provider]
    model_instance = model_class(id=model_name)

    instructions_block = instructions_override or dedent("""
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
    """)

    return Agent(
        name="PitchMaster",
        model=model_instance,
        response_model=PitchResponse,
        markdown=False,
        description="You are a highly skilled investment consultant specializing in startup fundraising.",
        instructions=instructions_block,
        add_datetime_to_instructions=True,
        show_tool_calls=True,
    )

# ======================================================== METRIC EXTRACTOR ========================================================
def extract_metrics(response):
    metrics = response.metrics
    return {
        "latency": metrics.get("additional_metrics", [{}])[0].get("completion_time", None),
        "input_length": metrics.get("input_tokens", [0])[0],
        "output_length": metrics.get("output_tokens", [0])[0],
        "response": response.content.model_dump()
    }

# ======================================================== MAIN ========================================================
if __name__ == "__main__":
    facts_path = Path("src/agno_agents/data/outputs/facts_and_productdescriptions.json")
    with facts_path.open("r", encoding="utf-8") as f:
        facts_dict = json.loads(f.read())

    scenarios = list(facts_dict.keys())[:3]
    framework = "multi_level3_agents"
    layer = "N/A"
    provider = "groq"
    model_id = "deepseek-r1-distill-llama-70b"
    model_name = f"{provider}/{model_id}"

    financial_agent = create_financial_agent(provider, model_id)
    pitch_agent = create_pitch_agent(provider, model_id)
    valuation_tools = ValuationTools()

    results = []

    for scenario in tqdm(scenarios, desc="Processing Pitches", unit="pitch"):
        product_data = facts_dict[scenario]
        formatted_product_data = json.dumps(product_data, indent=2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        financial_response = financial_agent.run(json.dumps(product_data["facts"]))
        if not hasattr(financial_response.content, "model_dump"):
            print(f"⚠️ Skipping {scenario} due to invalid financial content.")
            continue

        financials = financial_response.content
        is_profitable = financials.is_profitable
        revenue = financials.revenue
        profit = financials.profit
        
        if is_profitable:
            val1 = valuation_tools.profit_multiple_valuation(profit)
            val2 = valuation_tools.discounted_cash_flow_valuation(profit)
            valuation = (val1 + val2) / 2
        else: #not profitable
            valuation = valuation_tools.revenue_multiple_valuation(revenue)

        funding_amount = 0.1 * valuation

        financial_summary = dedent(f"""
            The startup has the following financial metrics:
            - Revenue: ${revenue:,.0f}
            - Profit: ${profit:,.0f}
            - Is Profitable: {'Yes' if is_profitable else 'No'}

            Based on financial analysis, the estimated company valuation is ${valuation:,.0f}.
            The founders are asking for approximately ${funding_amount:,.0f} in funding.

            You should propose an appropriate equity percentage and key investment terms based on this context.
        """)

        prompt = f"Financial summary:\n{financial_summary}\n\nHere is the startup's description and facts:\n\n{formatted_product_data}"

        print(prompt)
        print("="*50)
        
    #     response = pitch_agent.run(prompt)
    #     metrics = extract_metrics(response)

    #     metrics.update({
    #         "scenario": scenario,
    #         "framework": framework,
    #         "layer": layer,
    #         "model_name": model_name,
    #         "model_identifier": f"{model_name}-{framework}_{layer}",
    #         "timestamp": timestamp,
    #         "prompt": prompt,
    #         "valuation": f"${valuation:,.0f}",
    #         "funding_amount": f"${funding_amount:,.0f}",
    #         "is_profitable": is_profitable
    #     })

    #     results.append(metrics)

    # df = pd.DataFrame(results)
    # column_order = [
    #     "scenario", "framework", "layer", "model_name", "model_identifier",
    #     "timestamp", "latency", "input_length", "output_length",
    #     "valuation", "funding_amount", "is_profitable", "response", "prompt"
    # ]
    # df = df[column_order]

    # output_excel_path = f"src/agno_agents/data/outputs/{framework}_{timestamp}.xlsx"
    # df.to_excel(output_excel_path, index=False)

    # print(f"\n>>>>>Results saved to {output_excel_path}\n")
