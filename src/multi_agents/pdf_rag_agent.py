import typer
from rich.prompt import Prompt
from typing import Optional

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb

from agno.agent import Agent
from agno.models.groq import Groq

from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Use persistent storage for ChromaDB
collection_name="xxx"
vector_db = ChromaDb(collection=collection_name, 
                    path="./chromadb_data",  # Set storage location ./chromadb_data is at current working directory
                    persistent_client=True   # Enable persistence
                     )

knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/src/multi_agents/The-AI-Act.pdf"],
    vector_db=vector_db, 
)

def pdf_agent(user: str = "user", groq_model:str ="deepseek-r1-distill-llama-70b"):
    groq_agent = Agent(
        model=Groq(id=groq_model),
        show_tool_calls=True,
        debug_mode=True,
        knowledge=knowledge_base
    )
    print("Groq Agent initialized successfully.")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        groq_agent.print_response(message)

if __name__ == "__main__":
    ## Comment out after first run
    # knowledge_base.load(recreate=False)

    # Initialise and run pdf agent
    groq_model_name="llama3-70b-8192" # or "deepseek-r1-distill-llama-70b", "qwen-qwq-32b" or "llama3-70b-8192"
    typer.run(pdf_agent(groq_model=groq_model_name)
              )
    
    # what did the european council said in 2017     