from textwrap import dedent
from pydantic import BaseModel, Field
from typing import Dict, Union

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from dotenv import load_dotenv
import os

from pathlib import Path
import json
import pandas as pd

from datetime import datetime
from tqdm import tqdm

# ======================================================== SETUP ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class FinancialMetrics(BaseModel):
    revenue: float = Field(..., description="Current or projected annual revenue (in dollars)")
    profit: float = Field(..., description="Current or projected net profit (in dollars)")
    is_profitable: bool = Field(..., description="Whether the startup is currently profitable")
    justification: str = Field(..., description="A short justification if the business is not profitable. Otherwise, set to 'N/A'.")
MODEL_PROVIDERS = {
    "openai": OpenAIChat,
    "groq": Groq,
}

# ======================================================== AGENT FACTORY ========================================================
def create_financial_agent(provider="groq", model_name="deepseek-r1-distill-llama-70b"):
    model_class = MODEL_PROVIDERS[provider]
    model_instance = model_class(id=model_name)

    return Agent(
        name="FinancialExtractor",
        model=model_instance,
        response_model=FinancialMetrics,
        markdown=False,
        description="You are a financial analyst extracting key startup metrics.",
        instructions=dedent("""
        You are a financial analyst extracting key startup metrics from raw business facts.

        Your job is to extract ONLY the following 4 financial fields:

        - "revenue": Total or projected annual revenue, in **USD as a number** (e.g., 1000000 for $1M).
        - "profit": Total or projected annual net profit, in **USD as a number**.
        - "is_profitable": Boolean — TRUE only if the company **explicitly mentions a profit or margin**.
        - "justification": Required if is_profitable is false. Briefly explain why the business is not profitable (e.g., "No profit mentioned and company is reinvesting in growth").

        ❗️ Do NOT guess or infer profit. Only use numbers that are clearly stated.
        ❗️ Do NOT assume profitability just because there is revenue.
        ❗️ If the profit or profitability is not mentioned, set:
            - \"profit\": 0
            - \"is_profitable\": false
            - Provide a justification

        ✅ Return valid JSON matching this schema:
        {
            "revenue": 1200000,
            "profit": 0,
            "is_profitable": false,
            "justification": "No profit mentioned and company is reinvesting in growth."
        }

        Output ONLY the JSON object — no extra explanation.
    """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        )

# ======================================================== FALLBACK PARSER ========================================================
def safe_parse_financial_output(raw: Union[str, Dict]) -> Union[FinancialMetrics, None]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None
    try:
        patched = {
            "revenue": raw.get("revenue") or 0.0,
            "profit": raw.get("profit") or 0.0,
            "is_profitable": raw.get("is_profitable") or raw.get("isprofitable") or False,
            "justification": raw.get("justification") or "N/A"

        }
        return FinancialMetrics(**patched)
    except Exception as e:
        print("❌ Failed to parse FinancialMetrics:", e)
        return None

# ======================================================== MAIN ========================================================
if __name__ == "__main__":
    print()
    facts_path = Path("src/agno_agents/data/outputs/facts_and_productdescriptions.json")
    with facts_path.open("r", encoding="utf-8") as f:
        facts_dict = json.loads(f.read())

    scenarios = list(facts_dict.keys())[:5]
    provider = "groq"
    model_id = "deepseek-r1-distill-llama-70b"
    model_name = f"{provider}/{model_id}"

    financial_agent = create_financial_agent(provider, model_id)
    results = []

    for scenario in tqdm(scenarios, desc="Processing Pitches for Financial", unit="pitch"):
        product_data = facts_dict[scenario]
        facts_json = json.dumps(product_data["facts"], indent=2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        response = financial_agent.run(facts_json)

        if isinstance(response.content, BaseModel):
            financial_numbers = response.content
        elif isinstance(response.content, dict) or isinstance(response.content, str):
            financial_numbers = safe_parse_financial_output(response.content)
        else:
            print("❌ Unknown response type")
            continue

        if financial_numbers:
            print(financial_numbers.model_dump())
            # print(financial_numbers.revenue, financial_numbers.profit, financial_numbers.is_profitable)
            results.append({
                "scenario": scenario,
                "timestamp": timestamp,
                "revenue": financial_numbers.revenue,
                "profit": financial_numbers.profit,
                "is_profitable": financial_numbers.is_profitable,
                "justification": financial_numbers.justification,
                "raw_prompt": facts_json
            })
        else:
            print("⚠️ Failed to extract financial metrics for:", scenario)
            continue
        print()

    df = pd.DataFrame(results)
    output_path = f"src/agno_agents/data/outputs/financial_outputs_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\n✅ Results saved to {output_path}\n")