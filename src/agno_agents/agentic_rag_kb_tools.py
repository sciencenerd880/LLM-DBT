from textwrap import dedent
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

from utils import PitchResponse, extract_metrics

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking

from agno.tools.duckduckgo import DuckDuckGoTools

# from agno.embedder.huggingface import HuggingfaceCustomEmbedder
# ======================================================== START: TO SET THE VARIABLES  ========================================================
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv(key="OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")

# Define a dictionary of available providers and their corresponding model classes
MODEL_PROVIDERS = {
    "openai": OpenAIChat,
    "groq": Groq,
}

def load_knowledge_base(
    pdf_urls: list[str] = None,
    chunking_strategy_name: str = "fixed",
    db_path: str = "./chromadb_data",
    storage_file: str = "tmp/pdf_rag.db"
):
    # Step 1: Default PDFs
    if pdf_urls is None:
        pdf_urls = [
            "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_opportunities.pdf",
            "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_pitchdeck_sample.pdf",
        ]

    # Step 2: Chunking strategies
    chunking_options = {
        "fixed": FixedSizeChunking(),
        "agentic": AgenticChunking(),
        "semantic": SemanticChunking(),
    }

    chunking_strategy = chunking_options.get(chunking_strategy_name, FixedSizeChunking())

    # Step 3: Table & collection names
    table_name = f"HBS_{chunking_strategy_name}_agent"
    collection_name = f"HBS_{chunking_strategy_name}"

    # Step 4: Print debug info (optional)
    print("\n[Knowledge Loader] ================================")
    print(f">> Chunking Strategy: {chunking_strategy_name.upper()}")
    print(f">> Agent Storage File: {storage_file}")
    print(f">> Agent Table Name: {table_name}")
    print(f">> ChromaDB Collection Name: {collection_name}")
    print(f">> ChromaDB Storage Path: {db_path}")
    print(f">> PDFs Being Processed:")
    for url in pdf_urls:
        print(f"   - {url}")
    print("===================================================\n")

    # Step 5: Instantiate vector DB and knowledge base
    vector_db = ChromaDb(
        collection=collection_name,
        path=db_path,
        persistent_client=True,
        # embedder=HuggingfaceCustomEmbedder()
    )

    knowledge_base = PDFUrlKnowledgeBase(
        urls=pdf_urls,
        vector_db=vector_db,
        chunking_strategy=chunking_strategy
    )

    return knowledge_base

# ======================================================== END: TO SET THE VARIABLES  ========================================================
HBS_knowledge_base = load_knowledge_base()
# HBS_knowledge_base.load(recreate=False) #comment if used

# Function to create the AI agent for generating Shark Tank pitches
def create_pitch_agent(provider: str = "groq", model_name: str = "deepseek-r1-distill-llama-70b"):
    if provider not in MODEL_PROVIDERS:
        raise ValueError(f"Invalid provider '{provider}'. Choose from: {list(MODEL_PROVIDERS.keys())}")

    model_class = MODEL_PROVIDERS[provider]  # Get the model class based on provider
    model_instance = model_class(id=model_name,response_format={ "type": "json_object" })  # Create the model instance with the specified name

    pitch_agent = Agent(
        name="PitchMaster",
        model=model_instance,  
        # response_model=PitchResponse,  # Ensures structured JSON output
        markdown=False,  # Set to False to ensure sentence-based output
        description=dedent("""\
            You are a highly skilled investment consultant specializing in startup fundraising.
            You are assisting a successful entrepreneur in crafting a compelling investment pitch for a new product.
            You will be given a product description along with some key facts.
            You are also given a knowledge base for reference.
        """),
        instructions=dedent("""\
            ### Your Task:
            1. Search the knowledge base for relevant information to assist in your task. Do not make up things that you do not know.
            2. Write a **persuasive startup pitch** that effectively highlights:
               - The product’s unique value proposition
               - The market opportunity and competitive edge
               - The potential for growth and profitability
               - A call to action for investors

            3. Propose an **initial offer to investors** that:
               - Raises as much equity as possible
               - Minimizes the stake given to investors
               - Includes key terms (e.g., valuation, percentage equity offered, funding amount)

            4. Return a well-structured response in valid JSON format.
            ### Response Format
            Return your response strictly in valid JSON format with the following structure:
            {{
                "Pitch": "Your well-structured investment pitch here...",
                "Initial_Offer": {{
                    "Valuation": "Estimated company valuation (e.g., $10 million)",
                    "Equity_Offered": "Percentage of equity offered to investors (e.g., 10%)",
                    "Funding_Amount": "The amount of funding requested (e.g., $1 million)",
                    "Key_Terms": "Any additional key terms (optional)"
                }}
            }}
            """),
        add_datetime_to_instructions=True,
        show_tool_calls=False,
        knowledge= HBS_knowledge_base,
        search_knowledge=True,
        debug_mode=True,
        reasoning=True
    )
    
    return pitch_agent


if __name__ == "__main__":
    print()
    # Input Path
    facts_path = Path("src/agno_agents/data/outputs/facts_and_productdescriptions.json")  
    
    # Open the facts_path and load into json
    with facts_path.open("r", encoding="utf-8") as f:
        facts_dict = json.loads(f.read())

    ##1 Get the product keys/scenario :5 for first 5, : for all
    scenarios = list(facts_dict.keys())[:1]

    ##2 Framework
    framework = "level1_agent_rag_reasoning"

    ##3 Layer - what is this???
    layer = "N/A"

    ##4 Model_name = LLM service provider name & model name
    provider = "groq"  # Change to "openai"
    # model_id_list = ["deepseek-r1-distill-llama-70b","deepseek-r1-distill-qwen-32b", "gemma2-9b-it", "llama-3.3-70b-versatile","llama3-70b-8192","mistral-saba-24b","qwen-qwq-32b"]
    # model_id_list = ["deepseek-r1-distill-llama-70b","deepseek-r1-distill-qwen-32b", "gemma2-9b-it", "llama-3.3-70b-versatile","mistral-saba-24b","qwen-qwq-32b"]
    # model_id_list = ["deepseek-r1-distill-llama-70b"] #done
    # model_id_list = ["deepseek-r1-distill-qwen-32b"] #done
    # model_id_list = ["gemma2-9b-it"] # failed at 75%
    # model_id_list = ["llama-3.3-70b-versatile"] #cannot rag?
    # model_id_list = ["llama3-70b-8192"]
    # model_id_list = ["qwen-qwq-32b"] #done

    model_id_list = ["deepseek-r1-distill-llama-70b", "deepseek-r1-distill-qwen-32b", "qwen-qwq-32b"]
    results = []
    for i, model_id in enumerate(model_id_list):

        model_name = f"{provider}/{model_id}"

        # Initialize model agent
        pitch_agent = create_pitch_agent(provider, model_id)
        
        print()
        print(f"#{i+1}_{model_id}")
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
