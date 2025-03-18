import typer
from rich.prompt import Prompt
from typing import Optional

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.storage.agent.sqlite import SqliteAgentStorage

from agno.agent import Agent
from agno.models.groq import Groq

from dotenv import load_dotenv
import os

# ======================================================== START: TO SET THE VARIABLES  ========================================================
# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# define pdf rag agent storage file & storage table name
agent_storage_file: str = "tmp/pdf_rag.db"
table_name_agent: str = "HBS-pdf_rag_agent"

# Use persistent storage for ChromaDB
collection_name="HBS"
chroma_db_path="./chromadb_data"

# PDF urls 
pdf_urls=["https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_opportunities.pdf",
          "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_pitchdeck_sample.pdf"
          ]

# https://github.com/sciencenerd880/LLM-DBT/tree/main/data/pdfs

# ======================================================== END: TO SET THE VARIABLES  ==========================================================

vector_db = ChromaDb(collection=collection_name, 
                    path= chroma_db_path,
                    persistent_client=True   # Enable persistence
                     )

# Defines the knowledge base
knowledge_base = PDFUrlKnowledgeBase(
    urls=pdf_urls,
    vector_db=vector_db, 
    )   

def pdf_agent(user: str = "user", groq_model:str ="deepseek-r1-distill-llama-70b"):
    groq_agent = Agent(
        model=Groq(id=groq_model),
        storage=SqliteAgentStorage(table_name=table_name_agent, 
                                   db_file=agent_storage_file),
        show_tool_calls=True,
        debug_mode=True,
        knowledge=knowledge_base
    )
    print()
    print(f"Groq '({groq_model})' Agent initialized successfully.")
    print()

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message in ("exit", "bye"):
            break
        groq_agent.print_response(message)

if __name__ == "__main__":
    ## Comment out after first run
    # knowledge_base.load(recreate=False)

    # Initialise and run pdf agent
    groq_model_name="llama3-70b-8192" # or "deepseek-r1-distill-llama-70b", "qwen-qwq-32b" or "llama3-70b-8192" or "qwen-2.5-32b"
    typer.run(pdf_agent(groq_model=groq_model_name)
              )
    
    # what did the european council said in 2017     