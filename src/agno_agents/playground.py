"""
reference here
"""

from agno.agent import Agent  # type: ignore
from agno.models.openai import OpenAIChat  # type: ignore
from agno.models.xai import xAI
from agno.playground import Playground, serve_playground_app  # type: ignore
from agno.storage.agent.sqlite import SqliteAgentStorage
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from dotenv import load_dotenv

load_dotenv()

agent_storage: str = "db/playground.db"

# https://docs.agno.com/get-started/playground
# web_agent = Agent(
#     name="Web Agent",
#     model=OpenAIChat(id="gpt-4o"),
#     tools=[DuckDuckGoTools()],
#     instructions=["常に情報源を含めてください。"],
#     storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),
#     add_datetime_to_instructions=True,
#     add_history_to_messages=True,
#     num_history_responses=5,
#     markdown=True,
# )
# Create web search agent
web_agent = Agent(
    name="Web Search Agent",
    role="Search the web for accurate and up-to-date information",
    model=xAI(id="grok-beta"),
    tools=[DuckDuckGoTools()],
    instructions=[
        "Always include sources and citations",
        "Verify information from multiple sources when possible",
        "Present information in a clear, structured format",
    ],
    show_tool_calls=True,
    markdown=True,
    monitoring=True,  # Enable monitoring for better debugging

    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    storage=SqliteAgentStorage(table_name="web_agent", db_file=agent_storage),

)

finance_agent = Agent(
    name="Finance Agent",
    model=OpenAIChat(id="gpt-4o"),
    tools=[
        YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            company_info=True,
            company_news=True,
        )
    ],
    instructions=["データは常に表を使用して表示してください。"],
    storage=SqliteAgentStorage(table_name="finance_agent", db_file=agent_storage),
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
)

app = Playground(agents=[web_agent, finance_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)