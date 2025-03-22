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
from tqdm import tqdm  #for progress bar

# ======================================================== START: TO SET THE VARIABLES  ========================================================
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

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


# Define a dictionary of available providers and their corresponding model classes
MODEL_PROVIDERS = {
    "openai": OpenAIChat,
    "groq": Groq,
}
# ======================================================== END: TO SET THE VARIABLES  ========================================================

# Function to create the AI agent for generating Shark Tank pitches
def create_pitch_agent(provider: str = "groq", model_name: str = "deepseek-r1-distill-llama-70b"):
    if provider not in MODEL_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Choose from: {list(MODEL_PROVIDERS.keys())}")

    model_class = MODEL_PROVIDERS[provider]  # Get the model class based on provider
    model_instance = model_class(id=model_name)  # Create the model instance with the specified name

    pitch_agent = Agent(
        name="PitchMaster",
        model=model_instance,  
        response_model=PitchResponse,  # Ensures structured JSON output
        markdown=False,  # Set to False to ensure sentence-based output
        description=dedent("""\
            You are a highly skilled investment consultant specializing in startup fundraising.
            You are assisting a successful entrepreneur in crafting a compelling investment pitch for a new product.
            You will be given a product description along with some key facts.
        """),
        instructions=dedent("""\
            ### Your Task:
            1. Write a **persuasive startup pitch** that effectively highlights:
               - The product’s unique value proposition
               - The market opportunity and competitive edge
               - The potential for growth and profitability
               - A call to action for investors

            2. Propose an **initial offer to investors** that:
               - Raises as much equity as possible
               - Minimizes the stake given to investors
               - Includes key terms (e.g., valuation, percentage equity offered, funding amount)

            3. Return a well-structured response in valid JSON format.
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
    )
    
    return pitch_agent

# Function to extract the relevant metrics
def extract_metrics(response):
    metrics = response.metrics
    return {
        "latency": metrics.get("additional_metrics", [{}])[0].get("completion_time", None),
        "input_length": metrics.get("input_tokens", [0])[0],
        "output_length": metrics.get("output_tokens", [0])[0],
        # "total_time": metrics.get("additional_metrics", [{}])[0].get("total_time", None),
        "response": response.content.model_dump()  
    }


if __name__ == "__main__":
    print()
    # Input Path
    facts_path = Path("src/agno_agents/data/outputs/facts_and_productdescriptions.json")  
    
    # Open the facts_path and load into json
    with facts_path.open("r", encoding="utf-8") as f:
        facts_dict = json.loads(f.read())

    ##1 Get the product keys/scenario :5 for first 5, : for all
    scenarios = list(facts_dict.keys())[:2]

    ##2 Framework
    framework = "naive_level0agent"

    ##3 Layer - what is this???
    layer = "N/A"

    ##4 Model_name = LLM service provider name & model name
    provider = "groq"  # Change to "openai"
    model_id = "deepseek-r1-distill-llama-70b"  # Change model name as needed
    model_name = f"{provider}/{model_id}"

    # Initialize model agent
    pitch_agent = create_pitch_agent(provider, model_id)

    results = []

    for scenario in tqdm(scenarios, desc="Processing Pitches", unit="pitch"):
        product_data = facts_dict[scenario]  # Get product data
        formatted_product_data = json.dumps(product_data, indent=2)  # Convert to JSON

        ##5 Generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ## Construct prompt
        prompt = f"Turn this product's description and facts into a persuasive Shark Tank pitch:\n\n{formatted_product_data}"
        
        # Run the model and get response
        response = pitch_agent.run(prompt)

        # Extract response metrics
        metrics = extract_metrics(response)
        metrics["scenario"] = scenario  
        metrics["framework"] = framework
        metrics["layer"] = layer
        metrics["model_name"] = model_name
        metrics["model_identifier"] = f"{model_name}-{framework}_{layer}"
        metrics["timestamp"] = timestamp  
        metrics["prompt"] = prompt

        # Store results
        results.append(metrics)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    column_order = [
        "scenario", "framework", "layer", "model_name", "model_identifier",
        "timestamp", "latency", "input_length", "output_length", "response", "prompt"
    ]
    # Reorder DataFrame columns
    df = df[column_order]

    # Save to Excel
    output_excel_path = f"src/agno_agents/data/outputs/{framework}_{timestamp}.xlsx"
    df.to_excel(output_excel_path, index=False)

    print(f">>>>>Results saved to {output_excel_path}")
    print()

    # pitch_agent.print_response(
    #     f"Turn this product's description and facts into a persuasive Shark Tank pitch:\n\n{formatted_product_data}",
    #     stream=True,
    # )
