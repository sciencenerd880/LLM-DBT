import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv
import os

# ========================================================
# Load environment variables
# ========================================================
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

agent_storage: str = "tmp/agents.db"

# ========================================================
# Define the Web Search Agent
# ========================================================
web_agent = Agent(
    name="Web Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[DuckDuckGoTools()],
    instructions=["Always include sources"],
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True, company_news=True)],
    instructions=["Always use tables to display data"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

# ========================================================
# Streamlit UI
# ========================================================
st.title("📊 AI Agents Interface")

# Select the agent
agent_type = st.selectbox("Select an Agent", ["Web Agent", "Finance Agent"])

# Input field
user_input = st.text_input("Enter your query:")

# Run button
if st.button("Ask"):
    if user_input.strip() == "":
        st.warning("Please enter a query.")
    else:
        selected_agent = web_agent if agent_type == "Web Agent" else finance_agent
        response = selected_agent.run(user_input)
        st.markdown(f"### 🤖 {agent_type} Response:\n{response.content}")