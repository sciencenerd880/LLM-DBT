"""
Usage
to run:
streamlit run src/agno_agents/streamlit_app.py
"""

import streamlit as st
import os
from dotenv import load_dotenv
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.calculator import CalculatorTools
from agno.tools.duckduckgo import DuckDuckGoTools
from textwrap import dedent
import time

from pydantic import BaseModel

class PitchResponse(BaseModel):
    Pitch: str
    Initial_Offer: dict


# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv(key="OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# ============================== Streamlit UI ==============================
st.title("📄 MITB is here for you.")
st.markdown("### 🤖 xxxxxx")

st.sidebar.title("ℹ️ Instructions")
st.sidebar.markdown(
    """
    1. **Choose Chunking Strategy**: Pick how the PDFs were chunked.
    2. **Choose Model**: Select from supported Groq models.
    3. **Enter Prompt**: Provide facts and descriptions.
    4. **Click Ask**: The agent will generate your pitch.
    """
)

# ============================== Model and Chunking Selection ==============================
chunking_options = {
    "Fixed": FixedSizeChunking(),
    "Agentic": AgenticChunking(),
    "Semantic": SemanticChunking(),
}

model_id_list = [
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
    "llama3-70b-8192",
    "qwen-qwq-32b",
    "gemma2-9b-it"
]

selected_chunking = st.selectbox("📑 Choose Chunking Strategy", list(chunking_options.keys()))
selected_model = st.selectbox("🧠 Choose Groq Model", model_id_list)

# ============================== Tools ==============================
DDG = DuckDuckGoTools()
calculator_tool = CalculatorTools(
    add=True,
    subtract=True,
    multiply=True,
    divide=True,
    exponentiate=True,
    factorial=True,
    is_prime=True,
    square_root=True,
)

# ============================== Knowledge Base Loader ==============================
def load_knowledge_base(chunking_strategy_name: str = "fixed"):
    pdf_urls = [
        "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_opportunities.pdf",
        "https://raw.githubusercontent.com/sciencenerd880/LLM-DBT/main/data/pdfs/hbs_pitchdeck_sample.pdf",
    ]
    chunking_strategy = chunking_options.get(chunking_strategy_name, FixedSizeChunking())
    collection_name = f"HBS_{chunking_strategy_name}"

    vector_db = ChromaDb(
        collection=collection_name,
        path="./chromadb_data",
        persistent_client=True,
    )

    knowledge_base = PDFUrlKnowledgeBase(
        urls=pdf_urls,
        vector_db=vector_db,
        chunking_strategy=chunking_strategy,
    )
    knowledge_base.load(recreate=False)
    return knowledge_base

# ============================== Agent Creation ==============================
def create_pitch_agent(model_name: str):
    model_instance = Groq(id=model_name, response_format={"type": "json_object"})
    agent = Agent(
        name="PitchMaster",
        model=model_instance,
        description=dedent("""
            You are a highly skilled investment consultant specializing in startup fundraising.
            You are assisting a successful entrepreneur in crafting a compelling investment pitch for a new product.
        """),
        instructions=dedent("""
            ### Your Task:
            1. Search the knowledge base for relevant information to assist in your task. Do not make up things that you do not know.
            2. Using the provided duckduckgo tool, use the function duckduckgo_search and function duckduckgo_news to find out the latest market trends in 2025.
            3. Write a **persuasive startup pitch** that effectively highlights:
               - The product’s unique value proposition.
               - The market opportunity and competitive edge.
               - The potential for growth and profitability.
               - A call to action for investors.

            4. Propose an **initial offer to investors** that:
               - Raises as much equity as possible.
               - Minimizes the stake given to investors.
               - Includes key terms (e.g., valuation, percentage equity offered, funding amount).
               - To obtain the funding amount, you need to use the provided calculator tool to compute by using the 'Equity_Offered' and 'Valuation'.

            6. Return a well-structured response in valid JSON format. **WARNING**: Ensure you follow the ### Response Format.
            ### Response Format
            Return your response STRICTLY in valid JSON format with the following structure:
            {{
                "Pitch": "Your well-structured investment pitch here...",
                "Initial_Offer": {{
                    "Valuation": "Estimated company valuation (e.g., $10 million)",
                    "Equity_Offered": "Stated % Percentage of equity offered to investors (e.g., 10%)",
                    "Funding_Amount": "The amount of funding requested (e.g., $1 million)",
                    "Key_Terms": "Any additional key terms (optional)"
                }}
            }}
        """),
        add_datetime_to_instructions=True,
        show_tool_calls=True,
        tools=[calculator_tool, DDG],
        knowledge=load_knowledge_base(chunking_strategy_name=selected_chunking.lower()),
        search_knowledge=True,
        add_references=True,
        # enable_agentic_context=True,
        # response_model=PitchResponse
    )
    return agent

# ============================== Chat Interface ==============================
st.subheader("💬 Enter your product facts and description:")
user_input = st.text_area("Facts & Description")

if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some input.")
    else:
        st.markdown("### 🤖 AI Response:")
        pitch_agent = create_pitch_agent(selected_model)

        response_placeholder = st.empty()
        full_response = ""

        
        with st.spinner("🤖 Generating your pitch..."):
            response_obj = pitch_agent.run(user_input)

        try:
            import json
            parsed = json.loads(response_obj.content)

            st.markdown("### 🎯 Pitch")
            st.markdown(f"> {parsed['Pitch']}")

            st.markdown("---")
            st.markdown("### 💸 Investment Offer")

            col1, col2 = st.columns(2)
            col1.metric("🏷️ Valuation", parsed['Initial_Offer']['Valuation'])
            col2.metric("📊 Equity Offered", parsed['Initial_Offer']['Equity_Offered'])

            st.markdown(f"**💰 Funding Amount:** {parsed['Initial_Offer']['Funding_Amount']}")
            st.markdown(f"**📄 Key Terms:** {parsed['Initial_Offer']['Key_Terms']}")

        except Exception as e:
            st.error(f"❌ Couldn't format response: {e}")
            st.markdown(response_obj.content)