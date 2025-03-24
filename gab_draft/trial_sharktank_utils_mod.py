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

from textwrap import dedent

from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.document.chunking.fixed import FixedSizeChunking
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.semantic import SemanticChunking

from agno.tools.calculator import CalculatorTools

from dotenv import load_dotenv
import os


# Load environment variables from .env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = "" # Insert API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


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

EDIT_REFERENCE_MODELS = [
    'qwen-qwq-32b',
    # 'gemma2-9b-it',
    'deepseek-r1-distill-llama-70b',
    # 'llama3-70b-8192',
    # 'mixtral-8x7b-32768',
    'deepseek-r1-distill-qwen-32b', 
    # 'mistral-saba-24b'
]

# Other setups
agent_storage: str = "tmp/agents.db"

# ======================================================== START: TO SET THE KNOWLEDGE BASE  ========================================================
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

# ======================================================== END: TO SET THE KNOWLEDGE BASE  ========================================================
HBS_knowledge_base = load_knowledge_base()
HBS_knowledge_base.load(recreate=False) #comment if used

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
    """Data structure to standardize the orchestrator response"""
    goal: str = Field(..., description="The main objective to be achieved.")
    subtasks: List[dict] = Field(..., description="List of dynamically generated subtasks.")

class SysthesizerResponse(BaseModel):
    """Data structure to standardize the synthesizer response"""
    Pitch: str = Field(..., description="The well structured pitch.")
    Initial_Offer: dict = Field(..., description="initial offer details.")

class PitchOrchestrator:
    """Basic pitch orchestrator without RAG elements"""
    def __init__(self, orchestrator="llama-3.3-70b-versatile", reference=REFERENCE_MODELS):
        self.agents = {}
        self.logs = []
        self.orchestrator = orchestrator
        self.reference = reference
        self.toolbox_mapping = {tool.name:tool for tool in self.tool_box()}

    def create_subtask_agent(self):
        """Instantiates the agent for orchestrating"""
        subtask_agent = Agent(
            name="Subroles and Subtask Generator",
            response_model=OrchestratorResponse, #newly added to enforce the response to pydantic base model
            model=Groq(id=self.orchestrator),
            instructions="Given a `facts` dictionary, decompose a main goal into a list of effective subtasks for a mixture of experts to solve.",
            storage=SqliteAgentStorage(table_name="director_agent", db_file=agent_storage),
            add_datetime_to_instructions=True,
            add_history_to_messages=False,
        )
        return subtask_agent
    
    def create_synthesizer_agent(self):
        """Instantiates the synthesizer agent for combining inputs"""
        instructions_synthesizer = dedent("""\
        Combine inputs into a winning pitch.
        ### Your Task:
            Propose an **initial offer to investors** that:
            - Raises as much equity as possible.
            - Minimizes the stake given to investors.
            - Includes key terms (e.g., valuation, percentage equity offered, funding amount).
            - To obtain the funding amount, you need to use the provided calculator tool to compute by using the 'Equity_Offered' and 'Valuation'.
        """)
        synthesizer = Agent(
            name="Pitch Synthesizer", 
            model=Groq(id=self.orchestrator),
            # response_model=SysthesizerResponse, #newly added to enfore response compliant
            instructions=instructions_synthesizer,
            storage=SqliteAgentStorage(table_name="synthesizer_agent", db_file=agent_storage),
            ### /newly added
            knowledge= HBS_knowledge_base,
            search_knowledge=True, # not really required, agent will set as True
            # add_references=True, # enable RAG by adding references from AgentKnowledge to the user prompt.
            # enable_agentic_context=True, #If True, enable the team agent to update the team context and automatically send the team context to the members
            ### /newly added
            add_datetime_to_instructions=True,
            add_history_to_messages=False,
        )
        return synthesizer
    
    def tool_box(self):
        """Contains the tools that are available to the agent"""
        calculator_tool= CalculatorTools(
                add=True,
                subtract=True,
                multiply=True,
                divide=True,
                exponentiate=True,
                factorial=True,
                is_prime=True,
                square_root=True,
                 )

        return [DuckDuckGoTools(), calculator_tool]

    def generate_subtasks(self, goal, facts):
        """Use an agent to break the main goal into subtasks dynamically."""
        subtask_agent = self.create_subtask_agent()
        prompt = f"Given facts: {facts}\nBreak down the following goal into 2-3 key subtasks:\n\nGoal: {goal}\n\nSubtasks:"
        prompt += """Format your response as valid JSON without the markdown:
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
            # # Since enforced response then commented out workaround
            content = subtask_response.content
            content = content.split("</think>")[-1].strip()
            # #content = content.replace("json", "").strip()
            structured_output = OrchestratorResponse.model_validate_json(content)
            # print("sanity:", structured_output)
        except Exception as e:
            print("Parsing Error:", e)
            print("Response:")
            print(subtask_response.content)
            raise ValueError("Invalid response format from subtask agent.")

        # subtasks = [task.strip() for task in subtask_response.content.split("\n") if task.strip()]
        return structured_output
        # return subtask_response.content.subtasks

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
        synthesis_prompt = dedent("""\
        Using the agents output earlier, return a well-structured response in valid JSON format. **WARNING**: Ensure you follow the ### Response Format.
        ### Response Format
        Return your response STRICTLY in valid JSON format with the following structure.:
        {
            "Pitch": "Your well-structured investment pitch here...",
            "Initial_Offer": {
                "Valuation": "Estimated company valuation (e.g., <<insert the estimated valuation here>>)",
                "Equity_Offered": "Percentage of equity offered to investors (e.g., <<insert the estimated percentage here>>)",
                "Funding_Amount": "The amount of funding requested (e.g., <<insert the estimated funding amount here>>)",
                "Key_Terms": "Any additional key terms (optional)"
            }
        }
        """)
        # print(">>> Agent input:", results)
        for role, content in results.items():
            synthesis_prompt += f"- {role}: {content}\n"
        synthesizer_agent = self.create_synthesizer_agent()
        synthesizer_response = synthesizer_agent.run(synthesis_prompt)
        self.logs.append(synthesizer_response.metrics)
        # print(synthesizer_response.content)

        try:
            json_regex = re.compile(
                r'\{.*\}',
                re.DOTALL
            )

            # Extract JSON match
            match = json_regex.search(synthesizer_response.content)
            if match:
                json_str = match.group()
            else:
                print("No JSON found.")
            content = json_str.replace("`", "")
            content = content.replace("json", "").strip()
            structured_output = SysthesizerResponse.model_validate_json(content)
            # pass
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

class PitchEditor(PitchOrchestrator):
    """Pitch orchestrator with editing loop but without RAG elements"""
    def __init__(self, orchestrator="llama-3.3-70b-versatile", reference=REFERENCE_MODELS, editor='llama3-70b-8192', iterations=2):
        super().__init__(orchestrator=orchestrator, reference=reference)
        self.editor = editor
        self.iterations = iterations
    
    def create_editor_agent(self):
        instruction = """Given a pitch for a sharktank viewing. Give 2-3 points of feedback. Do not introduce new facts. 
        Answer shortly in the following format:
        - ...
        - ...
        """
        editor = Agent(
            name="Pitch Editor", 
            model=Groq(id=self.editor, max_tokens=1024),
            instructions=instruction,
            storage=SqliteAgentStorage(table_name="director_agent", db_file=agent_storage),
            add_datetime_to_instructions=True,
            add_history_to_messages=False,
        )
        return editor

    def edit_pitch(self, pitch):
        """Give feedback on how to improve the pitch"""
        self.editor_agent = self.create_editor_agent()
        editor_prompt = f"""here is the pitch: {pitch}. Give short pointers."""
        editor_response = self.editor_agent.run(editor_prompt)
        self.logs.append(editor_response.metrics)
        return_prompt = f"""Edit a pitch with the given facts. With this pitch: {pitch}. {editor_response.content}."""
        return return_prompt
    
    def orchestrate_with_edit(self, goal, facts, verbose=False, have_tools=True):
        """Full pipeline: generate subtasks, create agents, execute, and synthesize pitch."""
        try:
            for i in tqdm(
                range(self.iterations), 
                bar_format='[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}', 
                desc="Running Pitch Editing Iterations",
                colour='yellow'
            ):
                subtasks = self.generate_subtasks(goal, facts, have_tools=have_tools)
                if verbose:
                    print("\t>>> Subtasks:", subtasks)
                time.sleep(1)
                self.create_agents(subtasks, facts)
                time.sleep(1)
                agent_outputs = self.run_agents(subtasks)
                if verbose:
                    print("\t>>> Agent Outputs:", agent_outputs)
                time.sleep(1)
                pitch_initial_offer = self.synthesize_pitch(agent_outputs)
                if verbose:
                    print("\t>>> Pitch Draft:", pitch_initial_offer)
                time.sleep(1)
                if i != self.iterations - 1:
                    goal = self.edit_pitch(pitch_initial_offer)
                    if verbose:
                        print("\t>>> New goal:", goal)
                    time.sleep(1)
        except Exception as e:
            print(e, "for case:", facts['product_description'])
            return ""
        return pitch_initial_offer

class PitchEquipped(PitchEditor):
    def generate_subtasks(self, goal, facts, have_tools=False):
        """Use an agent to break the main goal into subtasks dynamically and assign tools."""
        subtask_agent = self.create_subtask_agent()
        tools_available = self.tool_box() if have_tools else []

        prompt = f"Given facts: {facts}\nBreak down the following goal into 2-3 key subtasks:\n\nGoal: {goal}\n\nSubtasks:"
        if tools_available:
            tool_names = [tool.name for tool in tools_available]  # Convert tool objects to string
            prompt += f"You have the following tools at your disposal: {tool_names}\nAssign them to your subtasks strategically."

        prompt += """ Format your response as valid JSON without the json markdown:
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
            pass
            # print(">>> sanity at orchestrator agent:", subtask_response)
        except Exception as e:
            print("Parsing Error:", e)
            print("Response:", subtask_response.content)
            raise ValueError("Invalid response format from subtask agent.")

        return subtask_response.content.subtasks

    def create_agents(self, subtasks, facts):
        """Create agents dynamically based on subtasks and assign tools accordingly."""
        for i, subtask in enumerate(subtasks):
            agent_name = f"{i}"
            assigned_tools = [tool for tool in self.tool_box() if tool.name in subtask.get("assigned_tools", [])]

            self.agents[agent_name] = Agent(
                name=agent_name, 
                model=Groq(id=random.choice(self.reference), max_tokens=512),  # limit agent output
                instructions=f"Given these facts: {facts}\nDo not hallucinate. Ensure strict adherence to facts. Keep it short.\n{subtask['name']}",
                storage=SqliteAgentStorage(table_name=agent_name, db_file=agent_storage),
                add_history_to_messages=False,
                tools=assigned_tools,  # Assign dynamically selected tools
                # show_tool_calls=True,
                # debug_mode=True,
            )