import os, json, re
import random, time
import pandas as pd, numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

# Agno for llm agents
from agno.agent import Agent
from agno.models.groq import Groq
from agno.storage.agent.sqlite import SqliteAgentStorage

# Other LLM utils
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.wikipedia import WikipediaTools


SCENARIO = "./scenarios"
REFERENCE_MODELS = [
    'qwen-qwq-32b',
    'gemma2-9b-it',
    'deepseek-r1-distill-llama-70b',
    'llama3-70b-8192',
    # 'mixtral-8x7b-32768',
    'deepseek-r1-distill-qwen-32b', 
    'mistral-saba-24b'
]

# Other setups
agent_storage: str = "tmp/agents.db"

def load_facts(relative_file_path="./all_processed_facts.txt"):
    """Facts generated and saved as a .txt file."""
    with Path(relative_file_path).open("r", encoding="utf-8") as f:
        facts_store = json.loads(f.read())
    return facts_store

def dump_scenario(
    fact_dict,
    pitch_dict,
    shark_final_offer_dict,
    judge_score,
    judge_score_baseline,
    scenario_folder="",
    scenario_name=""
):
    """Dump all the scenario with the fact, pitch, shark offer, judge score, baseline"""

    scenario_dict = {
        'fact_dict': fact_dict,
        'pitch_dict': pitch_dict,
        'shark_final_offer_dict': shark_final_offer_dict,
        'judge_score': judge_score,
        'judge_score_baseline': judge_score_baseline
    }
    with open(f'{scenario_folder}/{scenario_name}.txt', "w", encoding="utf-8") as file:
        json.dump(scenario_dict, file, indent=4)    

def metrics_calculation(metric):
    # Input tokens
    input_length, output_length, latency = 0, 0, 0
    for m in metric:
        input_length += m['input_tokens'][0]
        output_length += m['output_tokens'][0]
        latency += m['additional_metrics'][0]['total_time']
    return input_length, output_length, latency

class OrchestratorResponse(BaseModel):
    goal: str = Field(..., description="The main objective to be achieved.")
    subtasks: List[dict] = Field(..., description="List of dynamically generated subtasks.")

class SysthesizerResponse(BaseModel):
    Pitch: str = Field(..., description="The well structured pitch.")
    Initial_Offer: dict = Field(..., description="initial offer details.")

class PitchOrchestrator:
    def __init__(self, orchestrator="llama-3.3-70b-versatile", reference=REFERENCE_MODELS):
        self.agents = {}
        self.logs = []
        self.orchestrator = orchestrator
        self.reference = reference
        # self.subtask_agent = self.create_subtask_agent()
        # self.synthesizer_agent = self.create_synthesizer_agent()

    def create_subtask_agent(self):
        """Instantiates the agent for orchestrating"""
        subtask_agent = Agent(
            name="Subroles and Subtask Generator",
            model=Groq(id=self.orchestrator),
            instructions="Given a `facts` dictionary, decompose a main goal into a list of effective subtasks for a mixture of experts to solve.",
            storage=SqliteAgentStorage(table_name="director_agent", db_file=agent_storage),
            add_datetime_to_instructions=True,
            add_history_to_messages=False,
        )
        return subtask_agent
    
    def create_synthesizer_agent(self):
        """Instantiates the synthesizer agent for combining inputs"""
        synthesizer = Agent(
            name="Pitch Synthesizer", 
            model=Groq(id=self.orchestrator),
            instructions="Combine inputs into a winning pitch.",
            storage=SqliteAgentStorage(table_name="synthesizer_agent", db_file=agent_storage),
            add_datetime_to_instructions=True,
            add_history_to_messages=False,
        )
        return synthesizer
    
    def tool_box():
        """Contains the tools that are available to the agent"""
        return [DuckDuckGoTools()]

    def generate_subtasks(self, goal, facts, have_tools=False):
        """Use an agent to break the main goal into subtasks dynamically."""
        subtask_agent = self.create_subtask_agent()
        prompt = f"Given facts: {facts}\nBreak down the following goal into 2-3 key subtasks:\n\nGoal: {goal}\n\nSubtasks:"
        if have_tools:
            prompt += f"You have the following tools at your disposal: [{','.join(self.tool_box())}]\nAssign them to your subtask strategically, and ensure they use these tools."
        prompt += """Format your response as valid JSON without the json markdown:
        {
            "goal": "...",
            "subtasks": [
                {
                    "name": "...",
                    "description": "...",
                    "assigned_tools": ["...", "..."]
                }
            ]
        }"""
        subtask_response = subtask_agent.run(prompt)

        # Log responses:
        self.logs.append(subtask_response.metrics)

        # Validate and parse response using Pydantic
        try:
            content = subtask_response.content.replace("`", "")
            content = content.replace("json", "").strip()
            structured_output = OrchestratorResponse.model_validate_json(content)
        except Exception as e:
            print("Parsing Error:", e)
            print("Response:")
            print(subtask_response.content)
            raise ValueError("Invalid response format from subtask agent.")

        # subtasks = [task.strip() for task in subtask_response.content.split("\n") if task.strip()]
        return structured_output.subtasks

    def create_agents(self, subtasks, facts):
        """Create agents dynamically based on the subtasks."""
        for i, subtask in enumerate(subtasks):
            agent_name = f"{i}"
            self.agents[agent_name] = Agent(
                name=agent_name, 
                model=Groq(id=random.choice(REFERENCE_MODELS), max_tokens=512), # limit agent output
                instructions=f"Given these facts: {facts}\n Do not hallucinate. Ensure strict adherence to facts. Keep it short. {subtask['name']}",
                storage=SqliteAgentStorage(table_name="agent_name", db_file=agent_storage),
                add_datetime_to_instructions=True,
                add_history_to_messages=False,
                # newly added here
                tools=self.tool_box(),
                show_tool_calls=True, #comment if not needed
                debug_mode=True, #comment if not needed
            )

    def run_agents(self, subtasks):
        """Run all agents on their respective tasks."""
        results = {}
        for name, agent in self.agents.items():
            agent_role = subtasks[int(name)]['name']
            subtask = subtasks[int(name)]['description']
            response = agent.run(f"Complete the following task to help achieve the goal: {subtask}")
            results[agent_role] = response.content
            self.logs.append(response.metrics)
        return results

    def synthesize_pitch(self, results):
        """Combine agent outputs into a final pitch."""
        synthesis_prompt = """
        Synthesize a compelling pitch from the following inputs without adding new information:
        Format your response as valid JSON without the json markdown:
        {
            "Pitch": "Your well-structured investment pitch here...",
            "Initial_Offer": {
                "Valuation": "Estimated company valuation (e.g., $10 million)",
                "Equity_Offered": "Percentage of equity offered to investors (e.g., 10%)",
                "Funding_Amount": "The amount of funding requested (e.g., $1 million)",
                "Key_Terms": "Any additional key terms (optional)"
            }
        }
        """
        for role, content in results.items():
            synthesis_prompt += f"- {role}: {content}\n"
        synthesizer_agent = self.create_synthesizer_agent()
        synthesizer_response = synthesizer_agent.run(synthesis_prompt)
        self.logs.append(synthesizer_response.metrics)

        try:
            content = synthesizer_response.content.replace("`", "")
            content = content.replace("json", "").strip()
            structured_output = SysthesizerResponse.model_validate_json(content)
        except Exception as e:
            print("Parsing Synthesizer Error:", e)
            print("Response:")
            print(synthesizer_response.content)
            raise ValueError("Invalid response format from subtask agent.")
        return content

    def orchestrate(self, goal, facts):
        """Full pipeline: generate subtasks, create agents, execute, and synthesize pitch."""
        time.sleep(1)
        subtasks = self.generate_subtasks(goal, facts, have_tools=True)
        time.sleep(1)
        self.create_agents(subtasks, facts)
        time.sleep(1)
        agent_outputs = self.run_agents(subtasks)
        time.sleep(1)
        return self.synthesize_pitch(agent_outputs)
