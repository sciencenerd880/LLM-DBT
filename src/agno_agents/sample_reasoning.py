"""
only openai reasoning works well
for deepseek it doesnt work well due to output response issue
and taking too long
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.models.groq import Groq

from dotenv import load_dotenv
import os

from pydantic import BaseModel, Field
from typing import List

class ReasoningSteps(BaseModel):
    reasoning_steps: List[str] = Field(..., description="Ordered reasoning steps for solving the task")
    ascii_diagram: str = Field(..., description="ASCII diagram illustrating the final state or process")
    final_solution: str = Field(..., description="Summary of the final answer or outcome")

# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv(key="OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

task = (
    "Three missionaries and three cannibals need to cross a river. "
    "They have a boat that can carry up to two people at a time. "
    "If, at any time, the cannibals outnumber the missionaries on either side of the river, the cannibals will eat the missionaries. "
    "How can all six people get across the river safely? Provide a step-by-step solution and show the solutions as an ascii diagram"
)

# reasoning_agent = Agent(model=OpenAIChat(id="gpt-4o"), reasoning=True, markdown=True)
# reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)
# from agno.agent import Agent
# from agno.models.openai import OpenAIChat

# structured_reasoning_agent = Agent(
#     model=OpenAIChat(id="gpt-4o"),
#     reasoning_model=Groq(id="deepseek-r1-distill-llama-70b"),
#     response_model=ReasoningSteps,
#     markdown=False,
#     description="You are an expert in logic puzzles. Return a JSON object with your reasoning, solution, and ASCII diagram.",
# )

# structured_reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)

from agno.agent import Agent
from agno.models.openai import OpenAIChat

task = (
    "Three missionaries and three cannibals need to cross a river. "
    "They have a boat that can carry up to two people at a time. "
    "If, at any time, the cannibals outnumber the missionaries on either side of the river, the cannibals will eat the missionaries. "
    "How can all six people get across the river safely? Provide a step-by-step solution and show the solutions as an ascii diagram"
)
reasoning_agent = Agent(
    model=OpenAIChat(id="gpt-4o-2024-08-06"), reasoning=True, markdown=True
)
reasoning_agent.print_response(task, stream=True, show_full_reasoning=True)